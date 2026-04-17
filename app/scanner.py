"""Market tarayıcı – multi-exchange, sinyal yönetimi.

1. Yeni CRT setup tespiti (4H)
2. CISD konfirmasyonu varsa DB'ye active yaz (pending kayit tutulmaz)
3. Legacy pending setup kontrolu/temizligi
4. Aktif sinyallerde sadece TP/SL takibi → status: expired (result: win/loss)
5. Expired (invalidated) sinyallerde breakeven kontrolü → fiyat entry'ye dönerse breakeven
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crt_engine import (
    CRTSetup,
    check_breakeven,
    check_cisd_confirmation,
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


def _detect_tp_sl_hit_from_ohlcv(
    df_15m,
    direction: str,
    take_profit: float | None,
    stop_loss: float | None,
) -> str | None:
    """TP/SL ihlalini mum high/low ile kontrol et.

    Not: Veri sağlayıcı yuvarlama farklılıklarında false hit üretmemek için
    seviyeye eşitlik (==) hit sayılmaz; seviye net olarak aşılmalıdır.
    """
    if take_profit is None or stop_loss is None or df_15m is None or df_15m.empty:
        return None

    for _, row in df_15m.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":
            if high > take_profit:
                return "hit_tp"
            if low < stop_loss:
                return "hit_sl"
        else:
            if low < take_profit:
                return "hit_tp"
            if high > stop_loss:
                return "hit_sl"

    return None


def _evaluate_active_signal_with_breakeven(
    df_15m,
    *,
    direction: str,
    take_profit: float | None,
    stop_loss: float | None,
    entry_price: float | None,
    invalidation_level: float | None,
    reached_50pct: bool,
) -> tuple[str | None, bool]:
    """Aktif sinyalde TP/SL ve %50 sonrasi breakeven stop akisini degerlendir.

    Returns:
        (event, arm_now)
        - event: "hit_tp" | "hit_sl" | "hit_be" | None
        - arm_now: Bu scan icinde %50 gecilip stop'un entry'ye tasinmasi gerekli mi?
    """
    if (
        take_profit is None
        or stop_loss is None
        or entry_price is None
        or invalidation_level is None
        or df_15m is None
        or df_15m.empty
    ):
        return None, False

    work = df_15m.sort_index()
    initial_armed = bool(reached_50pct)

    def _crosses_invalidation(high: float, low: float) -> bool:
        if direction == "LONG":
            return high > float(invalidation_level)
        return low < float(invalidation_level)

    def _hits_tp(high: float, low: float) -> bool:
        if direction == "LONG":
            return high > float(take_profit)
        return low < float(take_profit)

    def _hits_sl(high: float, low: float, sl_value: float) -> bool:
        if direction == "LONG":
            return low < sl_value
        return high > sl_value

    first_cross_idx = None
    for idx, (_, row) in enumerate(work.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        if _crosses_invalidation(high, low):
            first_cross_idx = idx
            break

    # Daha once %50 gorulduyse ve bu pencerede tekrar ilk gecis de gorunuyorsa,
    # breakeven stop'u o mumdan SONRA etkinlestirip gecmisi yanlis yorumlamayiz.
    armed = initial_armed and first_cross_idx is None
    arm_now = False

    for idx, (_, row) in enumerate(work.iterrows()):
        high = float(row["high"])
        low = float(row["low"])

        if _hits_tp(high, low):
            return "hit_tp", arm_now

        if armed:
            if _hits_sl(high, low, float(entry_price)):
                return "hit_be", arm_now
            continue

        if initial_armed:
            if first_cross_idx is not None and idx >= first_cross_idx:
                armed = True
            continue

        if _hits_sl(high, low, float(stop_loss)):
            return "hit_sl", arm_now

        if _crosses_invalidation(high, low):
            armed = True
            arm_now = True
            # Breakeven stop ayni mumda degil, bir sonraki mumdan itibaren gecerlidir.
            continue

    return None, arm_now


async def _is_setup_already_past_invalidation(
    exchange: object,
    symbol: str,
    setup: CRTSetup,
) -> bool:
    """Yeni setup'ta fiyat %50 invalidation seviyesini gecmis mi?"""
    try:
        df_latest = await fetch_ohlcv(exchange, symbol, "15m", limit=3)
        if df_latest is None or df_latest.empty:
            return False
        current_price = float(df_latest.iloc[-1]["close"])
        inv = float(setup.invalidation_level)
        if setup.direction == "LONG":
            return current_price > inv
        return current_price < inv
    except Exception:
        return False


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
        # ─── AŞAMA 1: Yeni CRT Setup + CISD Onayi (4H + 15M) ───
        stage1_activated: list[Signal] = []
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
                    if await _is_setup_already_past_invalidation(exchange, symbol, setup):
                        log.info(
                            "SKIPPED (PAST %%50): %s %s current already beyond invalidation.",
                            setup.symbol,
                            setup.direction,
                        )
                        continue

                    since_ms = None
                    if setup.crt_bar_time:
                        crt_ts = setup.crt_bar_time
                        if crt_ts.tzinfo is None:
                            crt_ts = crt_ts.replace(tzinfo=timezone.utc)
                        since_ms = int((crt_ts - timedelta(minutes=30)).timestamp() * 1000)

                    df_15m = await fetch_ohlcv(
                        exchange,
                        symbol,
                        "15m",
                        limit=200,
                        since_ms=since_ms,
                    )
                    if df_15m is None or df_15m.empty:
                        continue

                    temp_setup = CRTSetup(
                        symbol=setup.symbol,
                        direction=setup.direction,
                        purge_type=setup.purge_type,
                        bias=setup.bias or "NEUTRAL",
                        bias_score=int(setup.bias_score or 0),
                        key_level_high=setup.key_level_high,
                        key_level_low=setup.key_level_low,
                        crt_bar_time=setup.crt_bar_time,
                        purge_time=setup.purge_time,
                        invalidation_level=setup.invalidation_level,
                        purge_extreme=setup.purge_extreme,
                        last_4h_high=setup.last_4h_high,
                        last_4h_low=setup.last_4h_low,
                        market_type=setup.market_type,
                    )

                    cisd = check_cisd_confirmation(df_15m, temp_setup)
                    if cisd is None:
                        log.info(
                            "SKIPPED (NO CISD): %s %s setup not confirmed yet.",
                            setup.symbol,
                            setup.direction,
                        )
                        continue

                    planned_rr = _calc_planned_rr(cisd.entry_price, cisd.stop_loss, cisd.take_profit)
                    if planned_rr is None or planned_rr < MIN_RR_RATIO:
                        log.info(
                            "SKIPPED (LOW RR): %s %s → RR: %s (< %.2f)",
                            setup.symbol,
                            setup.direction,
                            planned_rr,
                            MIN_RR_RATIO,
                        )
                        continue

                    db_signal = _setup_to_db(setup)
                    _apply_cisd(db_signal, cisd)
                    db_signal.htf_bias = htf_bias
                    session.add(db_signal)
                    stage1_activated.append(db_signal)
                    result["new_setups"].append(setup)
                    result["activated"].append(db_signal)
                    log.info(
                        "NEW ACTIVE: %s %s %s HTF:%s Entry:%s SL:%s TP:%s RR:%.2f [%s]",
                        setup.symbol,
                        setup.direction,
                        setup.purge_type,
                        htf_bias,
                        cisd.entry_price,
                        cisd.stop_loss,
                        cisd.take_profit,
                        planned_rr or 0.0,
                        exchange_id,
                    )

                except Exception as e:
                    log.warning("4H scan failed for %s on %s: %s", symbol, exchange_id, e)

        if result["new_setups"]:
            await session.commit()
            if stage1_activated and tg_configured():
                for sig in stage1_activated:
                    await send_signal_active(sig)

        # ─── AŞAMA 2: Legacy Pending CISD Konfirmasyonu (15M) ───
        pending = await session.execute(
            select(Signal).where(Signal.status == "pending_cisd")
        )
        pending_signals = pending.scalars().all()

        pending_changed = False
        pending_activated: list[Signal] = []
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
                    pending_activated.append(sig)
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
            if pending_activated and tg_configured():
                for sig in pending_activated:
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

        # ─── AŞAMA 3: Aktif Sinyal Takibi (TP/SL + %50 sonra breakeven stop) ───
        active = await session.execute(
            select(Signal).where(Signal.status == "active")
        )
        active_signals = active.scalars().all()
        active_changed = False

        for sig in active_signals:
            try:
                ccxt_symbol, exchange_id = from_display_symbol(sig.symbol)
                exchange = exchanges.get(exchange_id)
                if not exchange:
                    continue

                since_ms = None
                cisd_ts = None
                if sig.cisd_time:
                    cisd_ts = sig.cisd_time
                    if cisd_ts.tzinfo is None:
                        cisd_ts = cisd_ts.replace(tzinfo=timezone.utc)
                    since_ms = int(cisd_ts.timestamp() * 1000)

                df_latest = await fetch_ohlcv(
                    exchange,
                    ccxt_symbol,
                    "15m",
                    limit=500,
                    since_ms=since_ms,
                )
                if df_latest.empty:
                    continue
                if cisd_ts is not None:
                    ts = pd.Timestamp(cisd_ts)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    # Entry, CISD onay mumu kapandiktan sonra kabul edilir.
                    # Bu nedenle ayni mumun intrabar hareketi TP/SL'e dahil edilmez.
                    df_latest = df_latest[df_latest.index > ts]
                    if df_latest.empty:
                        continue

                event, arm_now = _evaluate_active_signal_with_breakeven(
                    df_latest,
                    direction=sig.direction,
                    take_profit=sig.take_profit,
                    stop_loss=sig.stop_loss,
                    entry_price=sig.entry_price,
                    invalidation_level=sig.invalidation_level,
                    reached_50pct=bool(sig.reached_50pct),
                )

                if arm_now:
                    sig.reached_50pct = True
                    sig.stop_loss = sig.entry_price
                    active_changed = True
                    log.info(
                        "BREAKEVEN ARMED: %s %s reached %%50, SL moved to entry %s",
                        sig.symbol,
                        sig.direction,
                        sig.entry_price,
                    )

                if event:
                    sig.status = "expired"
                    if event == "hit_tp":
                        sig.result = "win"
                        risk = abs(sig.entry_price - sig.stop_loss)
                        reward = abs(sig.take_profit - sig.entry_price)
                        sig.rr_value = round(reward / risk, 2) if risk > 0 else 0
                        result["closed"].append({"symbol": sig.symbol, "status": "expired", "result": sig.result})
                    elif event == "hit_be":
                        sig.status = "breakeven"
                        sig.result = "breakeven"
                        sig.rr_value = 0.0
                        result["breakeven"].append(sig.symbol)
                    else:
                        sig.result = "loss"
                        sig.rr_value = -1.0
                        result["closed"].append({"symbol": sig.symbol, "status": "expired", "result": sig.result})

                    if sig.cisd_time:
                        delta = datetime.now(timezone.utc) - sig.cisd_time
                        sig.duration_hours = round(delta.total_seconds() / 3600, 1)

                    active_changed = True
                    log.info("CLOSED (%s): %s %s (R:R %.2f)", event.upper(), sig.symbol, sig.direction, sig.rr_value)
                    continue

            except Exception as e:
                log.warning("Active signal check failed for %s: %s", sig.symbol, e)

        if result["closed"] or result["breakeven"] or active_changed:
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
