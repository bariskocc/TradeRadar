"""Market tarayıcı – multi-exchange, 4 aşamalı sinyal yönetimi.

1. Yeni CRT setup tespiti (4H) → status: pending_cisd
2. Bekleyen setup'larda CISD konfirmasyonu (15M) → status: active
3. Aktif sinyallerde sadece TP/SL takibi → status: expired (result: win/loss)
4. Expired (invalidated) sinyallerde breakeven kontrolü → fiyat entry'ye dönerse breakeven
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crt_engine import (
    CRTSetup,
    check_breakeven,
    check_cisd_confirmation,
    check_tp_sl_hit,
    compute_htf_bias,
    detect_crt_setup,
)
from app.exchange import (
    EXCHANGE_PER_MARKET,
    SYMBOLS_BY_MARKET,
    close_exchanges,
    create_exchanges,
    fetch_ohlcv,
    from_display_symbol,
    to_display_symbol,
)
from app.models import Signal, ScanLog
from app.telegram import send_signal_active, is_configured as tg_configured

log = logging.getLogger(__name__)
MIN_RR_RATIO = 2.0
MIN_QUALITY_SCORE = 5


async def _is_duplicate_setup(session: AsyncSession, setup: CRTSetup) -> bool:
    result = await session.execute(
        select(Signal).where(
            Signal.symbol == setup.symbol,
            Signal.direction == setup.direction,
            Signal.purge_time == setup.purge_time,
        )
    )
    return result.scalars().first() is not None


def _setup_to_db(setup: CRTSetup) -> Signal:
    return Signal(
        symbol=setup.symbol,
        direction=setup.direction,
        purge_type=setup.purge_type,
        bias=setup.bias,
        bias_score=setup.bias_score,
        key_level_high=setup.key_level_high,
        key_level_low=setup.key_level_low,
        crt_bar_time=setup.crt_bar_time,
        purge_time=setup.purge_time,
        invalidation_level=setup.invalidation_level,
        market_type=setup.market_type,
        timeframe=setup.timeframe,
        status="pending_cisd",
        created_at=datetime.now(timezone.utc),
    )


def _apply_cisd(signal: Signal, cisd) -> None:
    signal.cisd_confirmed = True
    signal.cisd_time = cisd.cisd_time
    signal.cisd_price = cisd.cisd_price
    signal.entry_price = cisd.entry_price
    signal.stop_loss = cisd.stop_loss
    signal.take_profit = cisd.take_profit
    signal.invalidation_level = cisd.invalidation_level
    signal.status = "active"


def _calc_planned_rr(entry: float | None, sl: float | None, tp: float | None) -> float | None:
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    reward = abs(tp - entry)
    return round(reward / risk, 2)


async def _persist_scan_log(
    session: AsyncSession,
    *,
    source: str,
    timeframe: str,
    status: str,
    started_at: datetime,
    result: dict,
    error_message: str | None = None,
) -> None:
    finished_at = datetime.now(timezone.utc)
    duration_seconds = round((finished_at - started_at).total_seconds(), 2)
    session.add(
        ScanLog(
            source=source,
            timeframe=timeframe,
            status=status,
            new_setups=len(result.get("new_setups", [])),
            activated=len(result.get("activated", [])),
            closed=len(result.get("closed", [])),
            breakeven=len(result.get("breakeven", [])),
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            error_message=error_message,
        )
    )
    await session.commit()


async def run_scan(
    session: AsyncSession,
    timeframe: str = "4h",
    market_types: list[str] | None = None,
    source: str = "manual",
) -> dict:
    if market_types is None:
        market_types = list(SYMBOLS_BY_MARKET.keys())

    exchanges = await create_exchanges()
    started_at = datetime.now(timezone.utc)
    result = {"new_setups": [], "activated": [], "closed": [], "breakeven": []}

    try:
        # ─── AŞAMA 1: Yeni CRT Setup Tespiti (4H) ───
        for market in market_types:
            symbols = SYMBOLS_BY_MARKET.get(market, [])
            exchange_id = EXCHANGE_PER_MARKET.get(market, "binance")
            exchange = exchanges.get(exchange_id)
            if not exchange:
                continue

            for symbol in symbols:
                try:
                    htf_bias = "NEUTRAL"
                    try:
                        df_daily = await fetch_ohlcv(exchange, symbol, "1d", limit=30)
                        htf_bias = compute_htf_bias(df_daily)
                    except Exception:
                        pass

                    df_4h = await fetch_ohlcv(exchange, symbol, "4h", limit=50)
                    display_sym = to_display_symbol(symbol)
                    setup = detect_crt_setup(df_4h, display_sym, market, htf_bias)
                    if setup is None:
                        continue
                    if int(setup.bias_score or 0) < MIN_QUALITY_SCORE:
                        log.info(
                            "SKIPPED (LOW QUALITY): %s %s score=%s (< %d)",
                            setup.symbol,
                            setup.direction,
                            setup.bias_score,
                            MIN_QUALITY_SCORE,
                        )
                        continue
                    if await _is_duplicate_setup(session, setup):
                        continue

                    db_signal = _setup_to_db(setup)
                    db_signal.htf_bias = htf_bias
                    session.add(db_signal)
                    result["new_setups"].append(setup)
                    log.info("NEW SETUP: %s %s %s HTF:%s (pending CISD) [%s]",
                             setup.symbol, setup.direction, setup.purge_type, htf_bias, exchange_id)

                except Exception as e:
                    log.warning("4H scan failed for %s on %s: %s", symbol, exchange_id, e)

        if result["new_setups"]:
            await session.commit()

        # ─── AŞAMA 2: CISD Konfirmasyonu (15M) ───
        pending = await session.execute(
            select(Signal).where(Signal.status == "pending_cisd")
        )
        pending_signals = pending.scalars().all()

        pending_changed = False
        for sig in pending_signals:
            try:
                ccxt_symbol, exchange_id = from_display_symbol(sig.symbol)
                exchange = exchanges.get(exchange_id)
                if not exchange:
                    continue

                since_ms = None
                if sig.crt_bar_time:
                    crt_ts = sig.crt_bar_time
                    if crt_ts.tzinfo is None:
                        crt_ts = crt_ts.replace(tzinfo=timezone.utc)
                    since_ms = int((crt_ts - timedelta(minutes=30)).timestamp() * 1000)

                df_15m = await fetch_ohlcv(
                    exchange,
                    ccxt_symbol,
                    "15m",
                    limit=200,
                    since_ms=since_ms,
                )

                purge_extreme = sig.key_level_low if sig.purge_type == "LOW" else sig.key_level_high
                last_4h_high = sig.key_level_high
                last_4h_low = sig.key_level_low

                fourh_since_ms = None
                purge_ts = None
                if sig.purge_time:
                    purge_ts = sig.purge_time
                    if purge_ts.tzinfo is None:
                        purge_ts = purge_ts.replace(tzinfo=timezone.utc)
                    fourh_since_ms = int((purge_ts - timedelta(hours=8)).timestamp() * 1000)

                df_4h_ctx = await fetch_ohlcv(
                    exchange,
                    ccxt_symbol,
                    "4h",
                    limit=10,
                    since_ms=fourh_since_ms,
                )
                if not df_4h_ctx.empty:
                    last_4h_high = float(df_4h_ctx.iloc[-1]["high"])
                    last_4h_low = float(df_4h_ctx.iloc[-1]["low"])

                    if purge_ts is not None:
                        closest_idx = min(
                            range(len(df_4h_ctx.index)),
                            key=lambda idx: abs(df_4h_ctx.index[idx] - purge_ts),
                        )
                        closest_ts = df_4h_ctx.index[closest_idx]
                        if abs(closest_ts - purge_ts) <= timedelta(hours=2):
                            purge_row = df_4h_ctx.iloc[closest_idx]
                            purge_extreme = float(
                                purge_row["low"] if sig.purge_type == "LOW" else purge_row["high"]
                            )

                temp_setup = CRTSetup(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    purge_type=sig.purge_type,
                    bias=sig.bias or "NEUTRAL",
                    bias_score=int(sig.bias_score or 0),
                    key_level_high=sig.key_level_high,
                    key_level_low=sig.key_level_low,
                    crt_bar_time=sig.crt_bar_time,
                    purge_time=sig.purge_time,
                    invalidation_level=sig.invalidation_level,
                    purge_extreme=purge_extreme,
                    last_4h_high=last_4h_high,
                    last_4h_low=last_4h_low,
                    market_type=sig.market_type,
                )

                cisd = check_cisd_confirmation(df_15m, temp_setup)
                if cisd:
                    planned_rr = _calc_planned_rr(cisd.entry_price, cisd.stop_loss, cisd.take_profit)
                    if planned_rr is None or planned_rr < MIN_RR_RATIO:
                        log.info(
                            "SKIPPED (LOW RR): %s %s → RR: %s (< %.2f)",
                            sig.symbol,
                            sig.direction,
                            planned_rr,
                            MIN_RR_RATIO,
                        )
                        continue

                    _apply_cisd(sig, cisd)
                    pending_changed = True
                    result["activated"].append(sig)
                    log.info(
                        "CISD CONFIRMED: %s %s → Entry: %s, SL: %s, TP: %s, RR: %.2f [%s]",
                        sig.symbol, sig.direction, cisd.entry_price,
                        cisd.stop_loss, cisd.take_profit, planned_rr or 0.0, exchange_id,
                    )

            except Exception as e:
                log.warning("15M CISD check failed for %s on %s: %s",
                            sig.symbol, exchange_id, e)

        if pending_changed:
            await session.commit()
            if result["activated"] and tg_configured():
                for sig in result["activated"]:
                    await send_signal_active(sig)

        # Pending CISD kayitlari UI/DB'de tutulmasin.
        pending_ids_r = await session.execute(
            select(Signal.id).where(Signal.status == "pending_cisd")
        )
        pending_ids = [row[0] for row in pending_ids_r.all()]
        if pending_ids:
            await session.execute(delete(Signal).where(Signal.id.in_(pending_ids)))
            await session.commit()
            log.info("PENDING CLEANUP: %d pending_cisd row deleted.", len(pending_ids))

        # ─── AŞAMA 3: Aktif Sinyal Takibi (yalnizca TP/SL) ───
        active = await session.execute(
            select(Signal).where(Signal.status == "active")
        )
        active_signals = active.scalars().all()

        for sig in active_signals:
            try:
                ccxt_symbol, exchange_id = from_display_symbol(sig.symbol)
                exchange = exchanges.get(exchange_id)
                if not exchange:
                    continue

                df_latest = await fetch_ohlcv(exchange, ccxt_symbol, "15m", limit=2)
                if df_latest.empty:
                    continue

                current_price = float(df_latest.iloc[-1]["close"])

                hit = check_tp_sl_hit(current_price, sig.direction, sig.take_profit, sig.stop_loss)
                if hit:
                    sig.status = "expired"
                    if hit == "hit_tp":
                        sig.result = "win"
                        risk = abs(sig.entry_price - sig.stop_loss)
                        reward = abs(sig.take_profit - sig.entry_price)
                        sig.rr_value = round(reward / risk, 2) if risk > 0 else 0
                    else:
                        sig.result = "loss"
                        sig.rr_value = -1.0

                    if sig.cisd_time:
                        delta = datetime.now(timezone.utc) - sig.cisd_time
                        sig.duration_hours = round(delta.total_seconds() / 3600, 1)

                    result["closed"].append({"symbol": sig.symbol, "status": "expired", "result": sig.result})
                    log.info("EXPIRED (%s): %s %s (R:R %.2f)", hit.upper(), sig.symbol, sig.direction, sig.rr_value)
                    continue

            except Exception as e:
                log.warning("Active signal check failed for %s: %s", sig.symbol, e)

        if result["closed"]:
            await session.commit()

        # ─── AŞAMA 4: Expired (invalidated) Sinyallerde Breakeven Kontrolü ───
        expired_q = await session.execute(
            select(Signal).where(
                Signal.status == "expired",
                Signal.result == "invalidated",
                Signal.entry_price.isnot(None),
            )
        )
        expired_signals = expired_q.scalars().all()

        for sig in expired_signals:
            try:
                ccxt_symbol, exchange_id = from_display_symbol(sig.symbol)
                exchange = exchanges.get(exchange_id)
                if not exchange:
                    continue

                df_latest = await fetch_ohlcv(exchange, ccxt_symbol, "15m", limit=2)
                if df_latest.empty:
                    continue

                current_price = float(df_latest.iloc[-1]["close"])

                if check_breakeven(current_price, sig.direction, sig.entry_price):
                    sig.status = "breakeven"
                    sig.result = "breakeven"
                    sig.rr_value = 0.0
                    if sig.cisd_time:
                        delta = datetime.now(timezone.utc) - sig.cisd_time
                        sig.duration_hours = round(delta.total_seconds() / 3600, 1)
                    result["breakeven"].append(sig.symbol)
                    log.info("BREAKEVEN: %s %s – price returned to entry",
                             sig.symbol, sig.direction)

            except Exception as e:
                log.warning("Breakeven check failed for %s: %s", sig.symbol, e)

        if result["breakeven"]:
            await session.commit()

        log.info(
            "Scan complete – %d setup(s), %d activated, %d closed, %d breakeven.",
            len(result["new_setups"]), len(result["activated"]),
            len(result["closed"]), len(result["breakeven"]),
        )
        try:
            await _persist_scan_log(
                session,
                source=source,
                timeframe=timeframe,
                status="success",
                started_at=started_at,
                result=result,
            )
        except Exception:
            log.exception("Failed to persist successful scan log")

    except Exception as exc:
        try:
            await session.rollback()
            await _persist_scan_log(
                session,
                source=source,
                timeframe=timeframe,
                status="failed",
                started_at=started_at,
                result=result,
                error_message=str(exc),
            )
        except Exception:
            log.exception("Failed to persist failed scan log")
        raise

    finally:
        await close_exchanges(exchanges)

    return result
