"""Market tarayici - BingX WebSocket event-driven, limit-order giris modeli.

Sinyal yasam dongusu:
1. 4H CRT setup + 15M kapanista MSS (swing ustu/alti CLOSE) -> waiting_entry
2. CISD/MSS onay MUMUNDAN SONRAKI LTF mumu KAPANDIGINDA entry retest
   (mum SL'ye gitmediyse) -> active. Forming mumda fill YOK.
3. Entry dolmadan CRT mumunun %60'i gecilirse waiting silinir (invalidated)
4. Entry dolmadan TP'ye ulasilirsa waiting silinir (missed)
5. Entry dolmadan SL gecilirse waiting silinir (past_sl)
6. Retest aninda skor < 7 ise sonradan fill/active yok (firsat kacti)
7. Aktif: TP/SL + tek kademe koruma:
   - TP yolunun %75'i -> trail acilir (max(1R, 1.3x LTF range) MFE arkasi)

Olaylar market_data (BingX WS) tarafindan tetiklenir:
  - on_candle_closed -> setup / CISD + kapanis fill
  - on_price_update -> yalnizca aktif TP/SL/trail (fill yok)
run_scan() ise manuel tetikleme / REST mutabakati icin ayni mantigi calistirir.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BOOTSTRAP_LIMITS
from app.crt_engine import (
    CRTSetup,
    check_cisd_confirmation,
    check_signal_invalidation,
    check_smt_divergence,
    compute_daily_bias,
    compute_htf_bias,
    compute_weekly_bias,
    detect_crt_setup,
    detect_ltf_ifvg,
)
from app.database import async_session
from app.event_log import record_event
from app.exchange import (
    correlated_symbol,
    fetch_ohlcv,
    from_display_symbol,
    get_active_markets,
    get_d1h_markets,
    get_h1_markets,
    get_h1_symbols_flat,
    market_of,
    to_display_symbol,
)
from app.models import Signal
from app.telegram import is_configured as tg_configured
from app.telegram import send_signal_active, send_signal_result

log = logging.getLogger(__name__)

MIN_RR_RATIO = 2.0          # Varsayilan (metal/oil/index vb.)
CRYPTO_MIN_RR_RATIO = 2.0   # Kripto: 1:2 alti acilmaz
FX_MIN_RR_RATIO = 2.0       # FX: 1:2
MIN_QUALITY_SCORE = 7  # 7 alti sinyal acilmaz (SMT bonusu sonrasi skor)
SMT_QUALITY_BONUS = 2  # Korele parite ile 15M SMT divergence varsa +2
MAX_QUALITY_SCORE = 11  # baz tavan 9 + SMT 2
PREMIUM_QUALITY_SCORE = 11  # yalnizca 11 = Premium

# 1D bias (Swing structure + ICT): ayni yon veya NEUTRAL acilir.
# LONG=BULLISH/NEUTRAL, SHORT=BEARISH/NEUTRAL. Karsi yon -> bias_mismatch.
REQUIRE_HTF_BIAS_ALIGN = True

# Ayni yonde asiri yigin: YALNIZCA CRYPTO. Acik (waiting/active) + son N saatte
# olusan kripto sinyal limiti. FX/metal/oil/index bu limite dahil degil.
MAX_SAME_DIRECTION_OPEN = 3
MAX_SAME_DIRECTION_RECENT = 5
CLUSTER_WINDOW_HOURS = 4
CLUSTER_MARKET = "crypto"

# Entry'ye gore minimum stop mesafesi (%). Simdilik tum marketlerde kapali.
# Tekrar acmak icin ilgili satiri geri al (fx / index).
MIN_STOP_PCT_BY_MARKET: dict[str, float] = {
    # "fx": 0.20,
    # "index": 0.25,  # US100 / US500
}

# Fill: LTF mum acilisindan sonra en az bu kadar sn bekle (anlik fill azaltir).
MIN_FILL_BAR_AGE_SEC = 60
MIN_FILL_BAR_AGE_SEC_1D = 300
MIN_FILL_BAR_AGE_SEC_1H = 20

STRATEGY_4H = "4h"
STRATEGY_1D = "1d"
STRATEGY_1H = "1h"
STRATEGY_CFG = {
    STRATEGY_4H: {
        "htf": "4h",
        "ltf": "15m",
        "c2_hours": 4.0,
        "smt_window_hours": 4.0,
        "min_fill_sec": MIN_FILL_BAR_AGE_SEC,
    },
    STRATEGY_1D: {
        "htf": "1d",
        "ltf": "1h",
        "c2_hours": 24.0,
        "smt_window_hours": 24.0,
        "min_fill_sec": MIN_FILL_BAR_AGE_SEC_1D,
    },
    STRATEGY_1H: {
        "htf": "1h",
        "ltf": "5m",
        "c2_hours": 1.0,
        "smt_window_hours": 1.0,
        "min_fill_sec": MIN_FILL_BAR_AGE_SEC_1H,
    },
}

# Tek kademe koruma (tum marketler):
# TP yolunun TRAIL_ARM_TP_FRACTION'inda trail acilir; SL, MFE'nin
# max(1R, 1.3 * ort. LTF range) gerisine cekilir (entry'den aleyhte gitmez).
TRAIL_ARM_TP_FRACTION = 0.75
TRAIL_OFFSET_R = 1.0
TRAIL_RANGE_MULT = 1.3
TRAIL_RANGE_LOOKBACK = 20
MIN_STOP_RANGE_MULT = 1.0  # stop mesafesi < 1x ort. LTF range -> tight_stop

WAITING_STATUS = "waiting_entry"
PENDING_STATUS = "pending_cisd"
OPEN_STATUSES = (PENDING_STATUS, WAITING_STATUS, "active")


# ──────────────────── Radar (canli izleme durumu) ────────────────────
# Her sembol icin son degerlendirmenin ozeti bellekte tutulur; /radar
# sayfasi bunu okuyarak sistemin ne gordugunu gosterir (sinyal olmasa bile).
_RADAR: dict[tuple[str, str], dict] = {}
_RADAR_META: dict = {"last_update": None}


def _set_radar(
    display_symbol: str,
    market: str | None,
    state: str,
    *,
    direction: str | None = None,
    score: int | None = None,
    bias: str | None = None,
    weekly_bias: str | None = None,
    rr: float | None = None,
    entry: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    smt: str | None = None,
    pd: str | None = None,
    c2_closed: bool | None = None,
    ifvg: bool | None = None,
    strategy: str = STRATEGY_4H,
) -> None:
    now = datetime.now(timezone.utc)
    key = (strategy, display_symbol)
    prev = _RADAR.get(key) or {}
    if state in ("no_setup", "no_data") and c2_closed is None:
        stored_c2 = None
    elif c2_closed is not None:
        stored_c2 = bool(c2_closed)
    else:
        stored_c2 = prev.get("c2_closed")
    if state in ("no_setup", "no_data"):
        stored_ifvg = ifvg
    elif ifvg is not None:
        stored_ifvg = bool(ifvg)
    else:
        stored_ifvg = prev.get("ifvg")
    _RADAR[key] = {
        "symbol": display_symbol,
        "market": market,
        "state": state,
        "direction": direction,
        "score": score,
        "bias": bias,
        "weekly_bias": weekly_bias if weekly_bias is not None else prev.get("weekly_bias"),
        "rr": rr,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "smt": smt,
        "pd": pd,
        "c2_closed": stored_c2,
        "ifvg": stored_ifvg,
        "strategy": strategy,
        "updated_at": now,
    }
    _RADAR_META["last_update"] = now


def get_radar_snapshot(strategy: str = STRATEGY_4H) -> dict:
    symbols = [v for (st, _), v in _RADAR.items() if st == strategy]
    return {"last_update": _RADAR_META["last_update"], "symbols": symbols, "strategy": strategy}


def _min_rr_for_market(market_type: str | None) -> float:
    m = (market_type or "").lower()
    if m == "crypto":
        return CRYPTO_MIN_RR_RATIO
    if m == "fx":
        return FX_MIN_RR_RATIO
    return MIN_RR_RATIO


def _bias_aligned(direction: str | None, htf_bias: str | None) -> bool:
    """LONG: BULLISH veya NEUTRAL; SHORT: BEARISH veya NEUTRAL.

    htf_bias birlesik daily bias'tir (Swing structure + ICT).
    NEUTRAL sinyal acmaya izin verir; yalnizca karsi yon engellenir.
    """
    d = (direction or "").upper()
    b = (htf_bias or "").upper() or "NEUTRAL"
    if b == "NEUTRAL":
        return True
    if d == "LONG":
        return b == "BULLISH"
    if d == "SHORT":
        return b == "BEARISH"
    return False


def _min_stop_pct(market_type: str | None) -> float | None:
    return MIN_STOP_PCT_BY_MARKET.get((market_type or "").lower())


def _stop_distance_pct(entry: float, stop_loss: float) -> float | None:
    if not entry:
        return None
    return abs(float(entry) - float(stop_loss)) / abs(float(entry)) * 100.0


async def _count_direction_cluster(
    session: AsyncSession,
    direction: str,
    timeframe: str = STRATEGY_4H,
) -> tuple[int, int]:
    """Crypto'da ayni yon icin (acik_adet, son_penceredeki_adet) dondur."""
    open_q = await session.execute(
        select(func.count()).where(
            Signal.direction == direction,
            Signal.market_type == CLUSTER_MARKET,
            Signal.timeframe == timeframe,
            Signal.status.in_(list(OPEN_STATUSES)),
        )
    )
    open_n = int(open_q.scalar() or 0)

    since = datetime.now(timezone.utc) - timedelta(hours=CLUSTER_WINDOW_HOURS)
    recent_q = await session.execute(
        select(func.count()).where(
            Signal.direction == direction,
            Signal.market_type == CLUSTER_MARKET,
            Signal.timeframe == timeframe,
            Signal.created_at >= since,
        )
    )
    recent_n = int(recent_q.scalar() or 0)
    return open_n, recent_n


# ──────────────────── Acik sinyal sembol onbellegi ────────────────────
# Fiyat guncellemelerinde her sembol icin gereksiz DB sorgusu yapmamak icin,
# waiting_entry/active sinyali olan display sembolleri bellekte tutariz.
_OPEN_SYMBOLS: set[tuple[str, str]] = set()
_CREATE_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _create_lock(display_symbol: str, strategy: str) -> asyncio.Lock:
    key = (display_symbol, strategy)
    lock = _CREATE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CREATE_LOCKS[key] = lock
    return lock


async def refresh_open_symbols(session: AsyncSession) -> None:
    _OPEN_SYMBOLS.clear()
    rows = await session.execute(
        select(Signal.symbol, Signal.timeframe).where(
            Signal.status.in_(list(OPEN_STATUSES))
        )
    )
    for sym, tf in rows.all():
        _OPEN_SYMBOLS.add((sym, tf or STRATEGY_4H))


def has_open_symbol(display_symbol: str, timeframe: str | None = None) -> bool:
    if timeframe is not None:
        return (display_symbol, timeframe) in _OPEN_SYMBOLS
    return any(sym == display_symbol for sym, _tf in _OPEN_SYMBOLS)


async def _sync_open_symbol(
    session: AsyncSession,
    display_symbol: str,
    timeframe: str = STRATEGY_4H,
) -> None:
    r = await session.execute(
        select(func.count()).where(
            Signal.symbol == display_symbol,
            Signal.timeframe == timeframe,
            Signal.status.in_(list(OPEN_STATUSES)),
        )
    )
    key = (display_symbol, timeframe)
    if (r.scalar() or 0) > 0:
        _OPEN_SYMBOLS.add(key)
    else:
        _OPEN_SYMBOLS.discard(key)


# ──────────────────── DB yardimcilari ────────────────────


async def _is_duplicate_setup(session: AsyncSession, setup: CRTSetup) -> bool:
    result = await session.execute(
        select(Signal).where(
            Signal.symbol == setup.symbol,
            Signal.direction == setup.direction,
            Signal.purge_time == setup.purge_time,
            Signal.timeframe == (setup.timeframe or STRATEGY_4H),
        )
    )
    return result.scalars().first() is not None


async def _has_open_signal_for_symbol(
    session: AsyncSession,
    symbol: str,
    timeframe: str = STRATEGY_4H,
    *,
    exclude_id: int | None = None,
) -> bool:
    q = select(Signal.id).where(
        Signal.symbol == symbol,
        Signal.timeframe == timeframe,
        Signal.status.in_(list(OPEN_STATUSES)),
    )
    if exclude_id is not None:
        q = q.where(Signal.id != exclude_id)
    result = await session.execute(q)
    return result.first() is not None


async def _get_open_signal(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
) -> Signal | None:
    result = await session.execute(
        select(Signal).where(
            Signal.symbol == symbol,
            Signal.timeframe == timeframe,
            Signal.status.in_(list(OPEN_STATUSES)),
        )
    )
    return result.scalars().first()


async def _get_pending_signal(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
) -> Signal | None:
    result = await session.execute(
        select(Signal).where(
            Signal.symbol == symbol,
            Signal.timeframe == timeframe,
            Signal.status == PENDING_STATUS,
        )
    )
    return result.scalars().first()


async def _delete_pending(
    session: AsyncSession,
    sig: Signal | None,
    reason: str,
) -> None:
    if sig is None:
        return
    log.info("PENDING DELETE: %s %s (%s)", sig.symbol, sig.direction, reason)
    await session.delete(sig)
    await session.commit()
    _OPEN_SYMBOLS.discard((sig.symbol, sig.timeframe or STRATEGY_4H))


def _calc_planned_rr(entry: float | None, sl: float | None, tp: float | None) -> float | None:
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    reward = abs(tp - entry)
    return round(reward / risk, 2)


def _apply_signal_levels(
    sig: Signal,
    setup: CRTSetup,
    cisd,
    planned_rr: float,
    htf_bias: str,
    weekly_bias: str | None = None,
    *,
    status: str,
) -> None:
    sig.direction = setup.direction
    sig.purge_type = setup.purge_type
    sig.bias = setup.bias
    sig.bias_score = setup.bias_score
    sig.key_level_high = setup.key_level_high
    sig.key_level_low = setup.key_level_low
    sig.crt_bar_time = setup.crt_bar_time
    sig.purge_time = setup.purge_time
    sig.entry_price = cisd.entry_price
    sig.stop_loss = cisd.stop_loss
    sig.initial_stop_loss = cisd.stop_loss
    sig.take_profit = cisd.take_profit
    sig.invalidation_level = cisd.invalidation_level
    sig.cisd_confirmed = bool(cisd.confirmed)
    sig.cisd_time = cisd.cisd_time
    sig.cisd_price = cisd.cisd_price
    sig.mss_ref_time = getattr(cisd, "mss_ref_time", None)
    sig.planned_rr = planned_rr
    sig.rr_value = planned_rr
    sig.htf_bias = htf_bias
    sig.weekly_bias = weekly_bias
    sig.smt_pair = setup.smt_pair
    sig.pd_array = setup.pd_array
    sig.c2_closed = setup.c2_closed
    sig.market_type = setup.market_type
    sig.timeframe = setup.timeframe
    sig.entry_model = getattr(cisd, "entry_model", None) or "cisd"
    sig.ifvg_low = getattr(cisd, "ifvg_low", None)
    sig.ifvg_high = getattr(cisd, "ifvg_high", None)
    sig.status = status


def _build_waiting_signal(
    setup: CRTSetup,
    cisd,
    planned_rr: float,
    htf_bias: str,
    weekly_bias: str | None = None,
    *,
    status: str = WAITING_STATUS,
) -> Signal:
    sig = Signal(
        symbol=setup.symbol,
        created_at=datetime.now(timezone.utc),
    )
    _apply_signal_levels(
        sig, setup, cisd, planned_rr, htf_bias, weekly_bias, status=status,
    )
    return sig


# ──────────────────── Intrabar degerlendiriciler ────────────────────


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_post_cisd_candle(candle_ts: datetime | None, cisd_time: datetime | None) -> bool:
    """Fill/missed yalnizca MSS onay mumundan SONRAKI 15M mumlarinda gecerli.

    candle_ts ve cisd_time mum acilis zamanidir (BingX T / df index).
    Ayni mum (candle_ts <= cisd_time) wick'i retest SAYILMAZ.
    """
    c = _as_utc(candle_ts)
    cisd = _as_utc(cisd_time)
    if c is None or cisd is None:
        return False
    return c > cisd


def _waiting_event(
    direction: str,
    high: float,
    low: float,
    entry: float,
    take_profit: float,
    invalidation_level: float | None = None,
    *,
    close: float | None = None,
    bar_closed: bool = False,
) -> str | None:
    """Bekleyen giris icin: 'invalidated' | 'fill' | 'missed' | None.

    CRT %60: yalniz kapanmis LTF close (wick iptal yok); fill'den once.
    Fill ve missed iğnede. Ayni mumda entry+TP varsa fill (sonra aktif TP).
    """
    if (
        bar_closed
        and close is not None
        and invalidation_level is not None
        and check_signal_invalidation(
            float(close), direction, float(invalidation_level), entry,
        )
    ):
        return "invalidated"
    if direction == "LONG":
        if low <= entry:
            return "fill"
        if high >= take_profit:
            return "missed"
    else:
        if high >= entry:
            return "fill"
        if low <= take_profit:
            return "missed"
    return None


def _closed_ltf_bars(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Store forming son bari dus; yalniz kapanmis LTF mumlari."""
    d = df_15m.sort_index()
    if len(d) < 2:
        return d.iloc[0:0]
    return d.iloc[:-1]


def _slice_ltf_after(
    df_ltf: pd.DataFrame,
    after_time: datetime | None,
    *,
    closed_only: bool = False,
) -> pd.DataFrame:
    """after_time acilisindan SONRAKI LTF bari (onay mumu haric)."""
    if df_ltf is None or df_ltf.empty:
        return df_ltf.iloc[0:0] if df_ltf is not None else pd.DataFrame()
    work = _closed_ltf_bars(df_ltf) if closed_only else df_ltf.sort_index()
    if work.empty:
        return work
    start = _as_utc(after_time)
    if start is None:
        return work
    idx = work.index
    start_ts = pd.Timestamp(start)
    if idx.tz is None:
        start_ts = start_ts.tz_localize(None) if start_ts.tzinfo is not None else start_ts
    else:
        start_ts = (
            start_ts.tz_localize("UTC") if start_ts.tzinfo is None
            else start_ts.tz_convert(idx.tz)
        )
    return work[idx > start_ts]


def _price_past_crt_mid(
    df_15m: pd.DataFrame,
    direction: str,
    entry: float,
    invalidation_level: float,
    cisd_time: datetime | None,
    after_time: datetime | None = None,
) -> bool:
    """Kapanmis LTF close CRT %60 otesine gecti mi? Iğne sayilmaz."""
    closed = _slice_ltf_after(
        df_15m, _as_utc(cisd_time) or _as_utc(after_time), closed_only=True,
    )
    if closed.empty:
        return False
    for close in closed["close"]:
        if check_signal_invalidation(
            float(close), direction, invalidation_level, entry,
        ):
            return True
    return False


def _price_hit_sl_after(
    df_ltf: pd.DataFrame,
    direction: str,
    stop_loss: float | None,
    after_time: datetime | None,
) -> bool:
    """CISD sonrasi LTF (forming dahil) SL'ye iğne ile degdi mi?"""
    if stop_loss is None:
        return False
    work = _slice_ltf_after(df_ltf, after_time, closed_only=False)
    if work.empty:
        return False
    sl = float(stop_loss)
    for _, row in work.iterrows():
        if _hits_sl(direction, float(row["high"]), float(row["low"]), sl):
            return True
    return False


def _price_hit_tp_after(
    df_ltf: pd.DataFrame,
    direction: str,
    take_profit: float | None,
    after_time: datetime | None,
) -> bool:
    """Purge/CISD sonrasi LTF (forming dahil) TP'ye iğne ile degdi mi?"""
    if take_profit is None:
        return False
    work = _slice_ltf_after(df_ltf, after_time, closed_only=False)
    if work.empty:
        return False
    tp = float(take_profit)
    for _, row in work.iterrows():
        if _hits_tp(direction, float(row["high"]), float(row["low"]), tp):
            return True
    return False


def _asof_index(df: pd.DataFrame, asof: datetime) -> pd.Timestamp:
    t = pd.Timestamp(_as_utc(asof) or asof)
    if df.index.tz is None:
        return t.tz_localize(None) if t.tzinfo is not None else t
    if t.tzinfo is None:
        return t.tz_localize("UTC").tz_convert(df.index.tz)
    return t.tz_convert(df.index.tz)


def _cut_asof(df: pd.DataFrame | None, asof: datetime | None) -> pd.DataFrame | None:
    if df is None or df.empty or asof is None:
        return df
    return df[df.index <= _asof_index(df, asof)]


def _first_post_cisd_fill_ts(
    df_ltf: pd.DataFrame | None,
    direction: str,
    entry: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    cisd_time: datetime | None,
) -> datetime | None:
    """CISD sonrasi ilk gercek fill mumu (sweep/SL ayni mum degil)."""
    if df_ltf is None or df_ltf.empty or entry is None or cisd_time is None:
        return None
    work = _slice_ltf_after(df_ltf, cisd_time, closed_only=True)
    if work.empty:
        return None
    sl = float(stop_loss) if stop_loss is not None else None
    tp = float(take_profit) if take_profit is not None else None
    ent = float(entry)
    for ts, row in work.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if direction == "LONG":
            if low > ent:
                continue
        elif high < ent:
            continue
        if sl is not None and _hits_sl(direction, high, low, sl) and (
            tp is None or not _hits_tp(direction, high, low, tp)
        ):
            continue
        t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        return _as_utc(t)
    return None


def _quality_score_asof(
    df_htf: pd.DataFrame | None,
    df_ltf: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    symbol: str,
    market: str,
    strategy: str,
    asof: datetime,
    htf_bias: str,
) -> int | None:
    """Retest anindaki (asof) CRT kalite skoru. Setup yoksa None."""
    htf = _cut_asof(df_htf, asof)
    ltf = _cut_asof(df_ltf, asof)
    d1 = _cut_asof(df_1d, asof)
    if htf is None or htf.empty:
        return None
    bias = htf_bias
    if d1 is not None and not d1.empty:
        try:
            bias = compute_daily_bias(d1)
        except Exception:
            pass
    setup = detect_crt_setup(
        htf, symbol, market, bias, df_1d=d1, timeframe=strategy, df_ltf=ltf,
    )
    if setup is None:
        return None
    return int(setup.bias_score or 0)


def _entry_between_stops(direction: str, entry: float, sl: float, tp: float) -> bool:
    if direction == "LONG":
        return sl < entry < tp
    return tp < entry < sl


def _maybe_ifvg_entry(cisd, setup: CRTSetup, df_ltf: pd.DataFrame):
    """LTF IFVG varsa ve mid SL-TP arasindaysa onu kullan; yoksa CISD/MSS.

    CISD/MSS adayi check_cisd_confirmation icinde zaten daha iyi RR ile secilir.
    IFVG RR eşiği burada uygulanmaz (cagiran min_rr ile eler).
    """
    planned = _calc_planned_rr(cisd.entry_price, cisd.stop_loss, cisd.take_profit)
    zone = detect_ltf_ifvg(
        df_ltf, setup.direction, setup.purge_time,
        crt_low=setup.key_level_low,
        crt_high=setup.key_level_high,
        crt_bar_time=setup.crt_bar_time,
    )
    if zone is None:
        return cisd, planned
    cisd.ifvg_low = zone.low
    cisd.ifvg_high = zone.high
    if not _entry_between_stops(
        setup.direction, zone.mid, float(cisd.stop_loss), float(cisd.take_profit),
    ):
        return cisd, planned
    ifvg_rr = _calc_planned_rr(zone.mid, cisd.stop_loss, cisd.take_profit)
    cisd.entry_price = zone.mid
    cisd.entry_model = "ifvg"
    return cisd, ifvg_rr


def _preview_trade_levels(
    setup: CRTSetup,
    df_ltf: pd.DataFrame | None,
    c2_hours: float,
) -> dict:
    """Radar icin setup sonrasi RR / entry / IFVG. Seviye yoksa kismi dict."""
    if df_ltf is None or df_ltf.empty:
        return {}
    zone = detect_ltf_ifvg(
        df_ltf, setup.direction, setup.purge_time,
        crt_low=setup.key_level_low,
        crt_high=setup.key_level_high,
        crt_bar_time=setup.crt_bar_time,
    )
    ifvg = zone is not None
    cisd = check_cisd_confirmation(df_ltf, setup, c2_hours=c2_hours)
    if cisd is not None:
        cisd, planned = _maybe_ifvg_entry(cisd, setup, df_ltf)
        return {
            "rr": planned,
            "entry": cisd.entry_price,
            "sl": cisd.stop_loss,
            "tp": cisd.take_profit,
            "ifvg": ifvg or getattr(cisd, "entry_model", None) == "ifvg",
        }
    if zone is not None and setup.purge_extreme is not None:
        sl = float(setup.purge_extreme)
        tp = float(
            setup.key_level_high if setup.direction == "LONG" else setup.key_level_low
        )
        if _entry_between_stops(setup.direction, zone.mid, sl, tp):
            return {
                "rr": _calc_planned_rr(zone.mid, sl, tp),
                "entry": zone.mid,
                "sl": sl,
                "tp": tp,
                "ifvg": True,
            }
        return {"ifvg": True}
    return {"ifvg": False}


def _hits_tp(direction: str, high: float, low: float, take_profit: float) -> bool:
    return high >= take_profit if direction == "LONG" else low <= take_profit


def _hits_sl(direction: str, high: float, low: float, stop_loss: float) -> bool:
    return low <= stop_loss if direction == "LONG" else high >= stop_loss


def _hits_level(direction: str, high: float, low: float, level: float, *, favorable: bool) -> bool:
    """favorable=True: TP / partial yonu; False: SL yonu."""
    if favorable:
        return high >= level if direction == "LONG" else low <= level
    return low <= level if direction == "LONG" else high >= level


def _tp_path_level(
    direction: str, entry: float, take_profit: float, fraction: float
) -> float:
    """Entry→TP yolunun verilen fraksiyonundaki seviye."""
    if direction == "LONG":
        return entry + fraction * (take_profit - entry)
    return entry - fraction * (entry - take_profit)


def _trail_arm_level(direction: str, entry: float, take_profit: float) -> float:
    """Trail acilis asamasi: TP yolunun TRAIL_ARM_TP_FRACTION kadarindaki seviye."""
    return _tp_path_level(direction, entry, take_profit, TRAIL_ARM_TP_FRACTION)


def _update_mfe(direction: str, mfe: float | None, entry: float, high: float, low: float) -> float:
    if direction == "LONG":
        base = entry if mfe is None else mfe
        return max(base, high)
    base = entry if mfe is None else mfe
    return min(base, low)


def _avg_ltf_range(df: pd.DataFrame | None, lookback: int = TRAIL_RANGE_LOOKBACK) -> float | None:
    """Son N LTF mumunun ortalama high-low araligi (trail nefes payi)."""
    if df is None or df.empty:
        return None
    win = df.sort_index().tail(max(1, int(lookback)))
    if win.empty:
        return None
    avg = float((win["high"] - win["low"]).mean())
    return avg if avg > 0 else None


def _trail_offset(risk: float, avg_ltf_range: float | None) -> float:
    """Trail mesafesi: en az 1R, gürültülü paritede LTF range tabani."""
    offset = TRAIL_OFFSET_R * float(risk)
    if avg_ltf_range is not None and avg_ltf_range > 0:
        offset = max(offset, TRAIL_RANGE_MULT * float(avg_ltf_range))
    return offset


def _trail_stop(
    direction: str,
    mfe: float,
    entry: float,
    risk: float,
    avg_ltf_range: float | None = None,
) -> float:
    """MFE'nin dinamik trail gerisi; entry'den aleyhte gecmez."""
    offset = _trail_offset(risk, avg_ltf_range)
    if direction == "LONG":
        return max(entry, mfe - offset)
    return min(entry, mfe + offset)


def _risk_from_stops(entry: float, initial_sl: float) -> float:
    return abs(float(entry) - float(initial_sl))


def _rr_at_exit(direction: str, entry: float, exit_price: float, risk: float) -> float:
    if risk <= 0:
        return 0.0
    if direction == "LONG":
        return (float(exit_price) - float(entry)) / risk
    return (float(entry) - float(exit_price)) / risk


def _planned_rr(entry: float, initial_sl: float, take_profit: float) -> float:
    risk = _risk_from_stops(entry, initial_sl)
    if risk <= 0:
        return 0.0
    return abs(float(take_profit) - float(entry)) / risk


# ──────────────────── Veri yukleme ────────────────────


def _copy_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return df.copy()


async def _cpu(fn, /, *args, **kwargs):
    """Pandas/CRT hesaplarini event loop disinda calistir."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def _htf_biases(df_1d: pd.DataFrame | None) -> tuple[str, str, str]:
    """(structure, daily, weekly)."""
    if df_1d is None or df_1d.empty:
        return "NEUTRAL", "NEUTRAL", "NEUTRAL"
    structure = daily = weekly = "NEUTRAL"
    try:
        structure = compute_htf_bias(df_1d)
    except Exception:
        pass
    try:
        daily = compute_daily_bias(df_1d)
    except Exception:
        pass
    try:
        weekly = compute_weekly_bias(df_1d)
    except Exception:
        pass
    return structure, daily, weekly


async def _load_frames(
    bingx_symbol: str,
    store: object | None,
    client: object | None = None,
) -> dict[str, pd.DataFrame]:
    """4H / 1D / 15M / 1H / 5M dataframe'lerini store'dan (varsa) ya da REST'ten yukle."""
    frames: dict[str, pd.DataFrame] = {}

    if store is not None:
        frames["4h"] = _copy_df(store.get_df(bingx_symbol, "4h"))
        frames["15m"] = _copy_df(store.get_df(bingx_symbol, "15m"))
        frames["1d"] = _copy_df(store.get_df(bingx_symbol, "1d"))
        frames["1h"] = _copy_df(store.get_df(bingx_symbol, "1h"))
        frames["5m"] = _copy_df(store.get_df(bingx_symbol, "5m"))
    else:
        frames["4h"] = await fetch_ohlcv(bingx_symbol, "4h", limit=BOOTSTRAP_LIMITS["4h"], client=client)
        frames["15m"] = await fetch_ohlcv(bingx_symbol, "15m", limit=BOOTSTRAP_LIMITS["15m"], client=client)
        frames["1d"] = await fetch_ohlcv(bingx_symbol, "1d", limit=BOOTSTRAP_LIMITS["1d"], client=client)
        frames["1h"] = await fetch_ohlcv(bingx_symbol, "1h", limit=BOOTSTRAP_LIMITS["1h"], client=client)
        frames["5m"] = await fetch_ohlcv(bingx_symbol, "5m", limit=BOOTSTRAP_LIMITS["5m"], client=client)
    return frames


# ──────────────────── Setup tespiti (4H kapanisinda) ────────────────────


async def _load_corr_ltf(
    corr_symbol: str,
    store: object | None,
    client: object | None,
    ltf: str = "15m",
) -> pd.DataFrame | None:
    """SMT icin korele paritenin LTF verisini store'dan ya da REST'ten yukle."""
    df: pd.DataFrame | None = None
    if store is not None:
        try:
            df = _copy_df(store.get_df(corr_symbol, ltf))
        except Exception:
            df = None
    if (df is None or df.empty) and client is not None:
        df = await fetch_ohlcv(corr_symbol, ltf, limit=BOOTSTRAP_LIMITS.get(ltf, 200), client=client)
    return df


async def _apply_smt_bonus(
    setup: CRTSetup,
    df_ltf: pd.DataFrame | None,
    bingx_symbol: str,
    store: object | None,
    client: object | None,
    ltf: str = "15m",
    window_hours: float = 4.0,
) -> bool:
    """Korele parite ile LTF SMT divergence varsa setup skorunu +SMT_QUALITY_BONUS (max 11) yap.

    SMT bulunursa setup.smt_pair'e korele paritenin gosterim adi yazilir.
    """
    if df_ltf is None or df_ltf.empty:
        return False
    corr = correlated_symbol(bingx_symbol)
    if corr is None:
        return False
    corr_ltf = await _load_corr_ltf(corr, store, client, ltf=ltf)
    if corr_ltf is None or corr_ltf.empty:
        return False
    if not await _cpu(
        check_smt_divergence,
        df_ltf, corr_ltf, setup.direction, setup.purge_time,
        window_hours=window_hours,
    ):
        return False

    setup.smt_pair = to_display_symbol(corr)
    new_score = min(MAX_QUALITY_SCORE, int(setup.bias_score or 0) + SMT_QUALITY_BONUS)
    setup.bias_score = new_score
    if new_score >= 7:
        setup.bias = "BULLISH" if setup.direction == "LONG" else "BEARISH"
    elif new_score <= 3:
        setup.bias = "BEARISH" if setup.direction == "LONG" else "BULLISH"
    log.info("SMT+%d: %s %s (korele %s) -> skor %d",
             SMT_QUALITY_BONUS, setup.symbol, setup.direction, setup.smt_pair, new_score)
    return True


async def detect_and_create_waiting(
    session: AsyncSession,
    bingx_symbol: str,
    frames: dict[str, pd.DataFrame],
    store: object | None = None,
    client: object | None = None,
    strategy: str = STRATEGY_4H,
) -> Signal | None:
    """HTF CRT setup + LTF CISD onayi ara; varsa waiting_entry olustur.

    strategy='4h' → 4H CRT + 15M CISD, hard filter 1D bias.
    strategy='1d' → 1D CRT + 1H CISD, hard filter 1D bias (1W bilgi).
    strategy='1h' → 1H CRT + 5M CISD, hard filter 1D bias (XAU/EUR/US100/BTC).
    """
    cfg = STRATEGY_CFG.get(strategy, STRATEGY_CFG[STRATEGY_4H])
    if strategy == STRATEGY_1H and bingx_symbol not in get_h1_symbols_flat():
        return None
    htf_key = cfg["htf"]
    ltf_key = cfg["ltf"]
    df_htf = frames.get(htf_key)
    df_ltf = frames.get(ltf_key)
    df_1d = frames.get("1d")

    market = market_of(bingx_symbol)
    display_sym = to_display_symbol(bingx_symbol)
    async with _create_lock(display_sym, strategy):
        return await _detect_and_create_waiting_locked(
            session, bingx_symbol, frames, store, client, strategy,
            cfg, htf_key, ltf_key, df_htf, df_ltf, df_1d, market, display_sym,
        )


async def _detect_and_create_waiting_locked(
    session: AsyncSession,
    bingx_symbol: str,
    frames: dict[str, pd.DataFrame],
    store: object | None,
    client: object | None,
    strategy: str,
    cfg: dict,
    htf_key: str,
    ltf_key: str,
    df_htf,
    df_ltf,
    df_1d,
    market: str,
    display_sym: str,
) -> Signal | None:
    df_4h = df_htf
    df_15m = df_ltf

    def _radar(state: str, **kwargs):
        _set_radar(display_sym, market, state, strategy=strategy, **kwargs)

    if df_htf is None or df_htf.empty or len(df_htf) < 16:
        _radar("no_data")
        return None

    structure_bias, htf_bias, weekly_bias = await _cpu(_htf_biases, df_1d)

    filter_bias = htf_bias

    existing_pending = await _get_pending_signal(session, display_sym, strategy)

    setup = await _cpu(
        detect_crt_setup,
        df_htf, display_sym, market, htf_bias,
        df_1d=df_1d, timeframe=strategy, df_ltf=df_ltf,
    )
    if setup is None:
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "crt_gone")
        _radar("no_setup", bias=htf_bias, weekly_bias=weekly_bias)
        return None

    _radar_base = _radar
    preview = await _cpu(_preview_trade_levels, setup, df_ltf, cfg["c2_hours"])

    def _radar(state: str, **kwargs):
        kwargs.setdefault("c2_closed", bool(setup.c2_closed))
        for key, val in preview.items():
            kwargs.setdefault(key, val)
        _radar_base(state, **kwargs)

    # CRT/purge ayni renk hard filter DEGIL; skorda baz +2 yerine +1
    # (bkz. _calc_live_setup_bias). Eski hard filter (geri almak icin ac):
    # if not setup.color_opposite:
    #     _radar("same_color", direction=setup.direction,
    #                score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
    #                smt=setup.smt_pair, pd=setup.pd_array)
    #     log.info("SKIPPED (SAME COLOR): %s %s CRT/purge ayni renk.",
    #              setup.symbol, setup.direction)
    #     return None

    # Hard filter: uc stratejide de 1D bias. 1W hicbirinde hard degil.
    if REQUIRE_HTF_BIAS_ALIGN and not _bias_aligned(setup.direction, filter_bias):
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "bias_mismatch")
        _radar("bias_mismatch", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info(
            "SKIPPED (BIAS): %s %s filter=%s daily=%s weekly=%s structure=%s",
            setup.symbol, setup.direction, filter_bias, htf_bias, weekly_bias, structure_bias,
        )
        return None

    await _apply_smt_bonus(
        setup, df_ltf, bingx_symbol, store, client,
        ltf=ltf_key, window_hours=cfg["smt_window_hours"],
    )

    if int(setup.bias_score or 0) < MIN_QUALITY_SCORE:
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "low_quality")
        _radar("low_quality", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info("SKIPPED (LOW QUALITY): %s %s score=%s", setup.symbol, setup.direction, setup.bias_score)
        return None

    same_pending = (
        existing_pending is not None
        and existing_pending.purge_time is not None
        and setup.purge_time is not None
        and _as_utc(existing_pending.purge_time) == _as_utc(setup.purge_time)
    )
    if existing_pending is not None and not same_pending:
        await _delete_pending(session, existing_pending, "crt_replaced")
        existing_pending = None

    open_sig = await _get_open_signal(session, setup.symbol, strategy)
    if (
        open_sig is not None
        and open_sig.status == "active"
        and open_sig.entry_filled_time is not None
        and existing_pending is None
    ):
        q_fill = await _cpu(
            _quality_score_asof,
            df_htf, df_ltf, df_1d, setup.symbol, market, strategy,
            _as_utc(open_sig.entry_filled_time), filter_bias,
        )
        if q_fill is not None and q_fill < MIN_QUALITY_SCORE:
            await record_event(
                "cancelled",
                f"Retest aninda skor {q_fill} (< {MIN_QUALITY_SCORE}), sonradan fill gecersiz",
                symbol=setup.symbol, direction=setup.direction,
                market_type=setup.market_type, level="warning", session=session,
            )
            _radar(
                "missed_quality",
                direction=setup.direction,
                score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                entry=open_sig.entry_price, sl=open_sig.stop_loss, tp=open_sig.take_profit,
                smt=setup.smt_pair, pd=setup.pd_array,
            )
            log.info(
                "VOID FILL (LOW QUALITY AT ENTRY): %s %s fill=%s score_asof=%s",
                setup.symbol, setup.direction, open_sig.entry_filled_time, q_fill,
            )
            await session.delete(open_sig)
            await session.commit()
            _OPEN_SYMBOLS.discard((setup.symbol, strategy))
            open_sig = None

    exclude_id = existing_pending.id if existing_pending is not None else None
    if await _has_open_signal_for_symbol(
        session, setup.symbol, timeframe=strategy, exclude_id=exclude_id,
    ):
        _radar("has_open", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info("SKIPPED (OPEN EXISTS): %s already has waiting/active signal.", setup.symbol)
        return None

    corr = correlated_symbol(bingx_symbol)
    if corr is not None:
        corr_disp = to_display_symbol(corr)
        if await _has_open_signal_for_symbol(session, corr_disp, timeframe=strategy):
            _radar("corr_open", direction=setup.direction,
                       score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
            log.info("SKIPPED (CORR OPEN): %s korele parite %s zaten acik.", setup.symbol, corr_disp)
            return None

    if existing_pending is None and await _is_duplicate_setup(session, setup):
        _radar("duplicate", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        return None

    if (setup.market_type or "").lower() == CLUSTER_MARKET and existing_pending is None:
        open_n, recent_n = await _count_direction_cluster(session, setup.direction, timeframe=strategy)
        if open_n >= MAX_SAME_DIRECTION_OPEN or recent_n >= MAX_SAME_DIRECTION_RECENT:
            _radar("cluster_limit", direction=setup.direction,
                       score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                       smt=setup.smt_pair, pd=setup.pd_array)
            log.info(
                "SKIPPED (CLUSTER): %s %s open=%d/%d recent=%d/%d (%dh)",
                setup.symbol, setup.direction,
                open_n, MAX_SAME_DIRECTION_OPEN,
                recent_n, MAX_SAME_DIRECTION_RECENT,
                CLUSTER_WINDOW_HOURS,
            )
            return None

    if df_ltf is None or df_ltf.empty:
        _radar("no_data", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        return None

    cisd = await _cpu(check_cisd_confirmation, df_ltf, setup, c2_hours=cfg["c2_hours"])
    if cisd is None:
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "no_levels")
        _radar("no_cisd", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info("SKIPPED (NO LEVELS): %s %s entry/MSS seviyeleri henuz yok.", setup.symbol, setup.direction)
        return None

    min_rr = _min_rr_for_market(setup.market_type)
    cisd, planned_rr = await _cpu(_maybe_ifvg_entry, cisd, setup, df_ltf)
    if planned_rr is None or planned_rr < min_rr:
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "low_rr")
        _radar("low_rr", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
                   entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info("SKIPPED (LOW RR): %s %s RR=%s (< %.2f)", setup.symbol, setup.direction, planned_rr, min_rr)
        return None

    min_pct = _min_stop_pct(setup.market_type)
    stop_pct = _stop_distance_pct(cisd.entry_price, cisd.stop_loss)
    stop_dist = abs(float(cisd.entry_price) - float(cisd.stop_loss))
    avg_range = _avg_ltf_range(df_ltf)
    tight_range = (
        avg_range is not None
        and stop_dist < MIN_STOP_RANGE_MULT * avg_range
    )
    tight_pct = min_pct is not None and (stop_pct is None or stop_pct < min_pct)
    if tight_range or tight_pct:
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "tight_stop")
        _radar("tight_stop", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
                   entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                   smt=setup.smt_pair, pd=setup.pd_array)
        if tight_range:
            log.info(
                "SKIPPED (TIGHT STOP): %s %s dist=%.8f < %.2fx LTF range %.8f",
                setup.symbol, setup.direction, stop_dist, MIN_STOP_RANGE_MULT, avg_range,
            )
        else:
            log.info(
                "SKIPPED (TIGHT STOP): %s %s stop=%.3f%% < min %.2f%%",
                setup.symbol, setup.direction, stop_pct or 0.0, min_pct,
            )
        return None

    tp_after = cisd.cisd_time if cisd.confirmed else setup.purge_time
    if _price_hit_tp_after(df_ltf, setup.direction, cisd.take_profit, tp_after):
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "past_tp")
        _radar("missed", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
                   entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info(
            "SKIPPED (PAST TP): %s %s entry oncesi TP %s gecildi.",
            setup.symbol, setup.direction, cisd.take_profit,
        )
        return None

    if cisd.confirmed and _price_hit_sl_after(
        df_ltf, setup.direction, cisd.stop_loss, cisd.cisd_time,
    ):
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "past_sl")
        _radar("past_sl", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
                   entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info(
            "SKIPPED (PAST SL): %s %s CISD sonrasi SL %s gecildi, waiting yok.",
            setup.symbol, setup.direction, cisd.stop_loss,
        )
        return None

    if _price_past_crt_mid(
        df_ltf, setup.direction, cisd.entry_price,
        cisd.invalidation_level, cisd.cisd_time, after_time=setup.purge_time,
    ):
        if existing_pending is not None:
            await _delete_pending(session, existing_pending, "crt_50")
        _radar("invalidated", direction=setup.direction,
                   score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
                   entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                   smt=setup.smt_pair, pd=setup.pd_array)
        log.info(
            "SKIPPED (CRT 60%%): %s %s price past invalidation %s before entry.",
            setup.symbol, setup.direction, cisd.invalidation_level,
        )
        return None

    if cisd.confirmed:
        fill_ts = _first_post_cisd_fill_ts(
            df_ltf, setup.direction, cisd.entry_price,
            cisd.stop_loss, cisd.take_profit, cisd.cisd_time,
        )
        if fill_ts is not None:
            q_at_fill = await _cpu(
                _quality_score_asof,
                df_htf, df_ltf, df_1d, setup.symbol, market, strategy,
                fill_ts, filter_bias,
            )
            if q_at_fill is None or q_at_fill < MIN_QUALITY_SCORE:
                if existing_pending is not None:
                    await _delete_pending(session, existing_pending, "missed_quality")
                _radar(
                    "missed_quality",
                    direction=setup.direction,
                    score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias,
                    rr=planned_rr,
                    entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
                    smt=setup.smt_pair, pd=setup.pd_array,
                )
                log.info(
                    "SKIPPED (MISSED QUALITY): %s %s retest %s skor=%s (< %s), fill yok.",
                    setup.symbol, setup.direction, fill_ts, q_at_fill, MIN_QUALITY_SCORE,
                )
                return None

    status = WAITING_STATUS if cisd.confirmed else PENDING_STATUS
    if existing_pending is not None:
        _apply_signal_levels(
            existing_pending, setup, cisd, planned_rr, htf_bias,
            weekly_bias=weekly_bias, status=status,
        )
        signal = existing_pending
        if status == WAITING_STATUS:
            await record_event(
                "new_setup",
                f"MSS onay | {setup.purge_type} purge | Entry {cisd.entry_price} SL {cisd.stop_loss} TP {cisd.take_profit} (RR {planned_rr})",
                symbol=setup.symbol, direction=setup.direction,
                market_type=setup.market_type, level="info", session=session,
            )
            log.info(
                "PROMOTE WAITING: %s %s %s Entry:%s SL:%s TP:%s RR:%.2f model=%s",
                setup.symbol, setup.direction, setup.purge_type,
                cisd.entry_price, cisd.stop_loss, cisd.take_profit, planned_rr,
                getattr(cisd, "entry_model", "cisd"),
            )
        await session.commit()
    else:
        signal = _build_waiting_signal(
            setup, cisd, planned_rr, htf_bias, weekly_bias=weekly_bias, status=status,
        )
        session.add(signal)
        if status == WAITING_STATUS:
            await record_event(
                "new_setup",
                f"{setup.purge_type} purge | Entry {cisd.entry_price} SL {cisd.stop_loss} TP {cisd.take_profit} (RR {planned_rr})",
                symbol=setup.symbol, direction=setup.direction,
                market_type=setup.market_type, level="info", session=session,
            )
        await session.commit()
        log.info(
            "%s: %s %s %s Entry:%s SL:%s TP:%s RR:%.2f model=%s",
            "NEW WAITING" if status == WAITING_STATUS else "NEW PENDING",
            setup.symbol, setup.direction, setup.purge_type,
            cisd.entry_price, cisd.stop_loss, cisd.take_profit, planned_rr,
            getattr(cisd, "entry_model", "cisd"),
        )

    _OPEN_SYMBOLS.add((setup.symbol, strategy))
    _radar(
        "waiting" if status == WAITING_STATUS else "no_cisd",
        direction=setup.direction,
        score=setup.bias_score, bias=htf_bias, weekly_bias=weekly_bias, rr=planned_rr,
        entry=cisd.entry_price, sl=cisd.stop_loss, tp=cisd.take_profit,
        smt=setup.smt_pair, pd=setup.pd_array,
    )
    return signal


# ──────────────────── Fiyat guncellemesi yonetimi ────────────────────


async def manage_symbol_on_price(
    session: AsyncSession,
    display_symbol: str,
    high: float,
    low: float,
    close: float,
    ts: datetime | None = None,
    *,
    candle_ts: datetime | None = None,
    timeframe: str = STRATEGY_4H,
    avg_ltf_range: float | None = None,
    bar_closed: bool = False,
) -> dict:
    """Bir sembol+timeframe icin bekleyen giris dolumu + aktif TP/SL/breakeven.

    `candle_ts`: degerlendirilen LTF mumunun acilis zamani. Waiting fill/missed
    icin zorunlu; CISD onay mumundan sonraki mumlar haricinde fill yapilmaz.
    `bar_closed`: True ise (kapanmis LTF mumu) fill serbest; forming mumda fill yok.
    `ts`: olay zamani (fill/duration kaydi); yoksa utcnow.
    `timeframe`: '4h'→15m, '1d'→1h, '1h'→5m.
    """
    result = {"activated": [], "closed": [], "breakeven": [], "cancelled": []}
    now = _as_utc(ts) or datetime.now(timezone.utc)
    bar_ts = _as_utc(candle_ts)

    rows = await session.execute(
        select(Signal).where(
            Signal.symbol == display_symbol,
            Signal.timeframe == timeframe,
            Signal.status.in_(list(OPEN_STATUSES)),
        )
    )
    signals = rows.scalars().all()
    if not signals:
        _OPEN_SYMBOLS.discard((display_symbol, timeframe))
        return result

    changed = False
    activated: list[Signal] = []
    finished: list[Signal] = []  # TP/SL/BE kapanan sinyaller (sonuc reply'i icin)

    for sig in signals:
        try:
            if sig.status == PENDING_STATUS:
                if (
                    bar_closed
                    and sig.entry_price is not None
                    and sig.invalidation_level is not None
                    and check_signal_invalidation(
                        float(close), sig.direction,
                        float(sig.invalidation_level), float(sig.entry_price),
                    )
                ):
                    await record_event(
                        "cancelled",
                        f"MSS oncesi CRT %60 close ({sig.invalidation_level})",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="warning", session=session,
                    )
                    _set_radar(
                        sig.symbol, sig.market_type, "invalidated",
                        direction=sig.direction, score=sig.bias_score,
                        bias=sig.htf_bias, weekly_bias=sig.weekly_bias,
                        entry=sig.entry_price,
                        sl=sig.stop_loss, tp=sig.take_profit,
                        smt=sig.smt_pair, pd=sig.pd_array,
                        c2_closed=sig.c2_closed,
                        strategy=sig.timeframe or timeframe,
                    )
                    await session.delete(sig)
                    changed = True
                    result["cancelled"].append(sig.symbol)
                    log.info(
                        "PENDING INVALIDATED (CRT 60%% close): %s %s @%s.",
                        sig.symbol, sig.direction, sig.invalidation_level,
                    )
                    continue
                if (
                    sig.take_profit is not None
                    and _hits_tp(sig.direction, high, low, float(sig.take_profit))
                ):
                    await record_event(
                        "cancelled", "MSS oncesi TP'ye ulasildi, firsat kacti",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="warning", session=session,
                    )
                    _set_radar(
                        sig.symbol, sig.market_type, "missed",
                        direction=sig.direction, score=sig.bias_score,
                        bias=sig.htf_bias, weekly_bias=sig.weekly_bias,
                        entry=sig.entry_price,
                        sl=sig.stop_loss, tp=sig.take_profit,
                        smt=sig.smt_pair, pd=sig.pd_array,
                        c2_closed=sig.c2_closed,
                        strategy=sig.timeframe or timeframe,
                    )
                    await session.delete(sig)
                    changed = True
                    result["cancelled"].append(sig.symbol)
                    log.info("PENDING MISSED (TP): %s %s silindi.", sig.symbol, sig.direction)
                    continue
                continue

            if sig.status == WAITING_STATUS:
                if sig.entry_price is None or sig.take_profit is None:
                    continue
                # Onay mumunda (veya onceki mumlarda) wick ile fill/missed YASAK.
                if not _is_post_cisd_candle(bar_ts, sig.cisd_time):
                    continue
                sl_lvl = (
                    float(sig.initial_stop_loss)
                    if sig.initial_stop_loss is not None
                    else (float(sig.stop_loss) if sig.stop_loss is not None else None)
                )
                if sl_lvl is not None and _hits_sl(sig.direction, high, low, sl_lvl):
                    await record_event(
                        "cancelled", "Entry dolmadan SL gecildi",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="warning", session=session,
                    )
                    _set_radar(
                        sig.symbol, sig.market_type, "past_sl",
                        direction=sig.direction, score=sig.bias_score,
                        bias=sig.htf_bias, weekly_bias=sig.weekly_bias,
                        entry=sig.entry_price,
                        sl=sig.stop_loss, tp=sig.take_profit,
                        smt=sig.smt_pair, pd=sig.pd_array,
                        c2_closed=sig.c2_closed,
                        strategy=sig.timeframe or timeframe,
                    )
                    await session.delete(sig)
                    changed = True
                    result["cancelled"].append(sig.symbol)
                    log.info(
                        "PAST SL: %s %s waiting removed @%s.",
                        sig.symbol, sig.direction, sl_lvl,
                    )
                    continue
                ev = _waiting_event(
                    sig.direction, high, low,
                    float(sig.entry_price), float(sig.take_profit),
                    invalidation_level=(
                        float(sig.invalidation_level)
                        if sig.invalidation_level is not None else None
                    ),
                    close=close,
                    bar_closed=bar_closed,
                )
                if ev == "invalidated":
                    await record_event(
                        "cancelled",
                        f"Entry dolmadan CRT %60 ({sig.invalidation_level}) gecildi",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="warning", session=session,
                    )
                    _set_radar(
                        sig.symbol, sig.market_type, "invalidated",
                        direction=sig.direction, score=sig.bias_score,
                        bias=sig.htf_bias, weekly_bias=sig.weekly_bias,
                        entry=sig.entry_price,
                        sl=sig.stop_loss, tp=sig.take_profit,
                        smt=sig.smt_pair, pd=sig.pd_array,
                        c2_closed=sig.c2_closed,
                        strategy=sig.timeframe or timeframe,
                    )
                    await session.delete(sig)
                    changed = True
                    result["cancelled"].append(sig.symbol)
                    log.info(
                        "INVALIDATED (CRT 60%%): %s %s waiting entry removed @%s.",
                        sig.symbol, sig.direction, sig.invalidation_level,
                    )
                    continue
                if ev == "missed":
                    await record_event(
                        "cancelled", "Entry (CISD) dolmadan TP'ye ulasildi, firsat kacti",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="warning", session=session,
                    )
                    # Radar'i guncelle: waiting (sinyal acildi) durumunda takili
                    # kalmasin; firsat kacti olarak isaretle.
                    _set_radar(
                        sig.symbol, sig.market_type, "missed",
                        direction=sig.direction, score=sig.bias_score,
                        bias=sig.htf_bias, weekly_bias=sig.weekly_bias,
                        entry=sig.entry_price,
                        sl=sig.stop_loss, tp=sig.take_profit,
                        smt=sig.smt_pair, pd=sig.pd_array,
                        c2_closed=sig.c2_closed,
                        strategy=sig.timeframe or timeframe,
                    )
                    await session.delete(sig)
                    changed = True
                    result["cancelled"].append(sig.symbol)
                    log.info("MISSED (TP BEFORE FILL): %s %s waiting entry removed.", sig.symbol, sig.direction)
                    continue
                # Fill yalnizca kapanmis LTF mumunda.
                if ev == "fill" and not bar_closed:
                    continue
                if ev == "fill":
                    sig.status = "active"
                    sig.entry_filled_time = now
                    changed = True
                    activated.append(sig)
                    result["activated"].append(sig)
                    await record_event(
                        "filled", f"Entry {sig.entry_price} dolduruldu (kapanis retest)",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="success", session=session,
                    )
                    log.info("FILLED: %s %s entry %s (closed bar)", sig.symbol, sig.direction, sig.entry_price)
                    # continue YOK: kapanan mumda TP varsa win yazilabilir.
                else:
                    continue  # hala waiting, aktif degil

            # active (bu cagride yeni fill olan sinyaller de burada degerlendirilir)
            if None in (sig.take_profit, sig.stop_loss, sig.entry_price):
                continue

            entry = float(sig.entry_price)
            tp = float(sig.take_profit)
            if sig.initial_stop_loss is None:
                sig.initial_stop_loss = float(sig.stop_loss)
                changed = True
            initial_sl = float(sig.initial_stop_loss)
            risk = _risk_from_stops(entry, initial_sl)
            if risk <= 0:
                continue

            # MFE guncelle (trail icin)
            sig.mfe_price = _update_mfe(sig.direction, sig.mfe_price, entry, high, low)

            event: str | None = None

            def _classify_sl_hit() -> str:
                if sig.partial_hit or sig.trail_active:
                    rr_exit = _rr_at_exit(sig.direction, entry, float(sig.stop_loss), risk)
                    return "hit_be" if rr_exit <= 0.05 else "hit_trail"
                return "hit_sl"

            # Trail/BE SL, forming mumun ARM ONCESI iğnesiyle kapanmasin
            # (UNI: 16:15 bar low 7.05 iken fiyat 7.40 TP'ye gitti).
            stale_trail_wick = (
                (bool(sig.trail_active) or bool(sig.partial_hit))
                and not bar_closed
            )

            # Once mevcut SL/TP ile cikis (stop'u degistirmeden once)
            if _hits_tp(sig.direction, high, low, tp):
                event = "hit_tp"
            elif (
                not stale_trail_wick
                and _hits_sl(sig.direction, high, low, float(sig.stop_loss))
            ):
                event = _classify_sl_hit()
            else:
                # Bu mumda sikisan trail SL, ayni mumun karsi fitiliyle
                # hemen "trail exit" yazmasin; cikis eski SL ile degerlendirilir.
                sl_this_bar = float(sig.stop_loss)

                # Tek kademe: TP yolunun %75'i -> trail ac (gap'te SL entry tabani)
                if not sig.trail_active:
                    arm_lvl = _trail_arm_level(sig.direction, entry, tp)
                    if _hits_level(sig.direction, high, low, arm_lvl, favorable=True):
                        if not sig.partial_hit:
                            sig.partial_hit = True
                            sig.reached_50pct = True
                            sig.stop_loss = entry
                        sig.trail_active = True
                        changed = True
                        await record_event(
                            "trail_arm",
                            f"Trail acildi @{arm_lvl} (TP yolunun %{int(TRAIL_ARM_TP_FRACTION * 100)}s) | min {TRAIL_OFFSET_R}R / {TRAIL_RANGE_MULT}x LTF range",
                            symbol=sig.symbol, direction=sig.direction,
                            market_type=sig.market_type, level="info", session=session,
                        )
                        log.info(
                            "TRAIL ARM: %s %s level=%s offset=%.2fR",
                            sig.symbol, sig.direction, arm_lvl, TRAIL_OFFSET_R,
                        )

                # Trail: MFE arkasindan SL sikistir (entry'den geriye gitmez)
                if sig.trail_active and sig.mfe_price is not None:
                    new_sl = _trail_stop(
                        sig.direction, float(sig.mfe_price), entry, risk,
                        avg_ltf_range=avg_ltf_range,
                    )
                    cur_sl = float(sig.stop_loss)
                    tighter = (
                        new_sl > cur_sl if sig.direction == "LONG" else new_sl < cur_sl
                    )
                    if tighter:
                        sig.stop_loss = round(new_sl, 8)
                        changed = True
                        log.info(
                            "TRAIL: %s %s SL->%s (mfe=%s)",
                            sig.symbol, sig.direction, sig.stop_loss, sig.mfe_price,
                        )

                # Ayni mumda TP veya (trail oncesi) mevcut SL.
                # Yeni trail SL bir sonraki LTF mumunda gecerli olur.
                if _hits_tp(sig.direction, high, low, tp):
                    event = "hit_tp"
                elif (
                    not stale_trail_wick
                    and _hits_sl(sig.direction, high, low, sl_this_bar)
                ):
                    event = _classify_sl_hit()

            if event:
                if event == "hit_tp":
                    sig.status = "expired"
                    sig.result = "win"
                    sig.rr_value = round(_planned_rr(entry, initial_sl, tp), 2)
                    setattr(sig, "_exit_kind", "tp")
                    result["closed"].append({"symbol": sig.symbol, "result": "win"})
                    await record_event(
                        "closed_win", f"TP {sig.take_profit} vuruldu (+{sig.rr_value}R)",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="success", session=session,
                    )
                elif event == "hit_trail":
                    sig.status = "expired"
                    sig.result = "win"
                    sig.rr_value = round(
                        _rr_at_exit(sig.direction, entry, float(sig.stop_loss), risk), 2
                    )
                    setattr(sig, "_exit_kind", "trail")
                    result["closed"].append({"symbol": sig.symbol, "result": "win"})
                    await record_event(
                        "closed_win",
                        f"Trail SL {sig.stop_loss} (+{sig.rr_value}R, partial={bool(sig.partial_hit)})",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="success", session=session,
                    )
                elif event == "hit_be":
                    sig.status = "breakeven"
                    sig.result = "breakeven"
                    sig.rr_value = 0.0
                    setattr(sig, "_exit_kind", "be")
                    result["breakeven"].append(sig.symbol)
                    await record_event(
                        "breakeven", f"Breakeven / trail floor (entry {sig.entry_price})",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="info", session=session,
                    )
                else:  # hit_sl
                    sig.status = "expired"
                    sig.result = "loss"
                    sig.rr_value = -1.0
                    setattr(sig, "_exit_kind", "sl")
                    result["closed"].append({"symbol": sig.symbol, "result": "loss"})
                    await record_event(
                        "closed_loss", f"SL {sig.stop_loss} vuruldu (-1R)",
                        symbol=sig.symbol, direction=sig.direction,
                        market_type=sig.market_type, level="error", session=session,
                    )

                ref_time = sig.entry_filled_time or sig.cisd_time
                if ref_time:
                    if ref_time.tzinfo is None:
                        ref_time = ref_time.replace(tzinfo=timezone.utc)
                    sig.duration_hours = round((now - ref_time).total_seconds() / 3600, 1)
                changed = True
                finished.append(sig)
                log.info("CLOSED (%s): %s %s RR %.2f", event.upper(), sig.symbol, sig.direction, sig.rr_value or 0.0)

        except Exception as e:
            log.warning("manage failed for %s: %s", display_symbol, e)

    if changed:
        await session.commit()
        if tg_configured():
            tg_changed = False
            # Aktif olan sinyaller: mesaji gonder, message_id'yi sakla (reply icin).
            for sig in activated:
                try:
                    mid = await send_signal_active(sig)
                    if mid:
                        sig.tg_message_id = mid
                        tg_changed = True
                except Exception:
                    log.warning("telegram send failed for %s", sig.symbol)
            # Kapanan sinyaller: sonucu aktif sinyal mesajina reply olarak gonder.
            for sig in finished:
                try:
                    await send_signal_result(sig)
                except Exception:
                    log.warning("telegram result send failed for %s", sig.symbol)
            if tg_changed:
                await session.commit()

    await _sync_open_symbol(session, display_symbol, timeframe=timeframe)
    return result


# ──────────────────── Acik sinyal REST mutabakati ────────────────────


async def reconcile_open_signals(
    session: AsyncSession,
    store: object | None = None,
) -> dict:
    """Acik (waiting/active) sinyalleri REST/store ile TP/SL/BE mutabik hale getir.

    WS kopuklugunda (ozellikle metal/fx/endeks) fiyat guncellemesi kacinca
    sinyal active'de takili kalabiliyor. Bakim dongusu bu guvenlik agini calistirir.
    """
    await refresh_open_symbols(session)
    result = {"activated": [], "closed": [], "breakeven": [], "cancelled": []}
    if not _OPEN_SYMBOLS:
        return result

    import httpx

    from app.config import BINGX_REST_BASE

    rows = await session.execute(
        select(Signal).where(Signal.status.in_(list(OPEN_STATUSES)))
    )
    signals = rows.scalars().all()
    by_symbol: dict[str, list[Signal]] = {}
    for sig in signals:
        by_symbol.setdefault(sig.symbol, []).append(sig)

    client = httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0)
    try:
        for display_symbol, sigs in by_symbol.items():
            try:
                by_tf: dict[str, list[Signal]] = {}
                for s in sigs:
                    by_tf.setdefault(s.timeframe or STRATEGY_4H, []).append(s)
                for tf, tf_sigs in by_tf.items():
                    ltf = STRATEGY_CFG.get(tf, STRATEGY_CFG[STRATEGY_4H])["ltf"]
                    bingx_symbol, _ = from_display_symbol(display_symbol)
                    df = None
                    if store is not None:
                        try:
                            df = store.get_df(bingx_symbol, ltf)
                        except Exception:
                            df = None
                    rest_df = await fetch_ohlcv(bingx_symbol, ltf, limit=50, client=client)
                    if rest_df is not None and not rest_df.empty:
                        df = rest_df
                    if df is None or df.empty:
                        continue

                    starts: list[datetime] = []
                    for s in tf_sigs:
                        t = s.entry_filled_time or s.cisd_time or s.created_at
                        t = _as_utc(t)
                        if t is not None:
                            starts.append(t)
                    subset = df
                    if starts:
                        min_start = min(starts)
                        subset = df[df.index > min_start] if any(
                            s.status == WAITING_STATUS and s.entry_filled_time is None for s in tf_sigs
                        ) else df[df.index >= min_start]
                    if subset.empty:
                        subset = df.tail(1)

                    for idx, row in subset.iterrows():
                        bar_ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                        if getattr(bar_ts, "tzinfo", None) is None:
                            bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                        is_last = idx == subset.index[-1]
                        m = await manage_symbol_on_price(
                            session,
                            display_symbol,
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            ts=bar_ts,
                            candle_ts=bar_ts,
                            timeframe=tf,
                            avg_ltf_range=_avg_ltf_range(df),
                            bar_closed=not is_last,
                        )
                        result["activated"].extend(m["activated"])
                        result["closed"].extend(m["closed"])
                        result["breakeven"].extend(m["breakeven"])
                        result["cancelled"].extend(m.get("cancelled", []))
                        if not has_open_symbol(display_symbol, timeframe=tf):
                            break
            except Exception as e:
                log.warning("reconcile failed for %s: %s", display_symbol, e)
    finally:
        await client.aclose()

    if result["closed"] or result["breakeven"] or result["cancelled"] or result["activated"]:
        log.info(
            "Reconcile: %d activated, %d closed, %d breakeven, %d cancelled",
            len(result["activated"]), len(result["closed"]),
            len(result["breakeven"]), len(result["cancelled"]),
        )
    return result


# ──────────────────── WS olay isleyicileri ────────────────────


async def on_candle_closed(bingx_symbol: str, timeframe: str, store: object) -> None:
    """BingX WS: bir mum kapandiginda cagrilir.

    4H-15M: 4h/15m kapanisi → 4H CRT + 15M CISD.
    1D-1H: 1d/1h kapanisi → 1D CRT + 1H CISD.
    1H-5M: 1h/5m kapanisi → 1H CRT + 5M CISD (XAU/EUR/US100/BTC).
    Tum veriler store'dan okunur; REST'e gidilmez.
    """
    strategies: list[str] = []
    if timeframe in ("4h", "15m"):
        strategies.append(STRATEGY_4H)
    if timeframe in ("1d", "1h"):
        strategies.append(STRATEGY_1D)
    if timeframe in ("1h", "5m") and bingx_symbol in get_h1_symbols_flat():
        strategies.append(STRATEGY_1H)
    if not strategies:
        return
    try:
        frames = await _load_frames(bingx_symbol, store)
        display_symbol = to_display_symbol(bingx_symbol)
        ltf_strategy = {
            "15m": STRATEGY_4H,
            "1h": STRATEGY_1D,
            "5m": STRATEGY_1H,
        }.get(timeframe)
        async with async_session() as session:
            for strategy in strategies:
                await detect_and_create_waiting(
                    session, bingx_symbol, frames, store=store, strategy=strategy,
                )
            if ltf_strategy and has_open_symbol(display_symbol, timeframe=ltf_strategy):
                df_ltf = store.get_df(bingx_symbol, timeframe)
                if df_ltf is not None and len(df_ltf) >= 2:
                    closed = df_ltf.iloc[-2]
                    bar_ts = closed.name.to_pydatetime()
                    if getattr(bar_ts, "tzinfo", None) is None:
                        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                    await manage_symbol_on_price(
                        session, display_symbol,
                        float(closed["high"]), float(closed["low"]), float(closed["close"]),
                        ts=bar_ts, candle_ts=bar_ts,
                        timeframe=ltf_strategy,
                        avg_ltf_range=_avg_ltf_range(df_ltf),
                        bar_closed=True,
                    )
    except Exception:
        log.exception("on_candle_closed failed for %s %s", bingx_symbol, timeframe)


async def on_price_update(
    bingx_symbol: str,
    candle: dict,
    store: object,
    ltf: str = "15m",
) -> None:
    """BingX WS: forming mum guncellemesi (aktif TP/SL/trail; fill yok)."""
    display_symbol = to_display_symbol(bingx_symbol)
    if ltf == "5m":
        strategy = STRATEGY_1H
    elif ltf == "1h":
        strategy = STRATEGY_1D
    else:
        strategy = STRATEGY_4H
    if not has_open_symbol(display_symbol, timeframe=strategy):
        return
    try:
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        candle_ts = None
        if candle.get("ts_ms") is not None:
            candle_ts = datetime.fromtimestamp(int(candle["ts_ms"]) / 1000, tz=timezone.utc)
        avg_range = None
        if store is not None:
            try:
                avg_range = _avg_ltf_range(store.get_df(bingx_symbol, ltf))
            except Exception:
                avg_range = None
        async with async_session() as session:
            await manage_symbol_on_price(
                session, display_symbol, high, low, close,
                candle_ts=candle_ts,
                timeframe=strategy,
                avg_ltf_range=avg_range,
                bar_closed=False,
            )
    except Exception:
        log.exception("on_price_update failed for %s", bingx_symbol)


# ──────────────────── Manuel tarama / REST mutabakati ────────────────────


async def run_scan(
    session: AsyncSession,
    timeframe: str = "all",
    market_types: list[str] | None = None,
    source: str = "manual",
    store: object | None = None,
) -> dict:
    """Tum aktif semboller icin REST (veya store) ile tek seferlik tarama/mutabakat.

    WS gercek zamanli calisirken bu, manuel tetikleme ve guvenlik agi gorevindedir.
    """
    if timeframe in (None, "", "all", "both"):
        strategies = [STRATEGY_4H, STRATEGY_1D, STRATEGY_1H]
    elif timeframe in STRATEGY_CFG:
        strategies = [timeframe]
    else:
        strategies = [STRATEGY_4H]

    started_at = datetime.now(timezone.utc)
    result = {"new_setups": [], "activated": [], "closed": [], "breakeven": []}

    import httpx

    from app.config import BINGX_REST_BASE

    client = httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0) if store is None else None
    try:
        await refresh_open_symbols(session)

        for strategy in strategies:
            cfg = STRATEGY_CFG[strategy]
            if strategy == STRATEGY_4H:
                markets = get_active_markets()
            elif strategy == STRATEGY_1D:
                markets = get_d1h_markets()
            else:
                markets = get_h1_markets()
            scan_markets = list(markets.keys()) if market_types is None else [
                m for m in market_types if m in markets
            ]
            for market in scan_markets:
                for bingx_symbol in markets.get(market, []):
                    try:
                        frames = await _load_frames(bingx_symbol, store, client=client)

                        created = await detect_and_create_waiting(
                            session, bingx_symbol, frames, store=store, client=client,
                            strategy=strategy,
                        )
                        if created is not None:
                            result["new_setups"].append(created)

                        df_ltf = frames.get(cfg["ltf"])
                        display_symbol = to_display_symbol(bingx_symbol)
                        if df_ltf is not None and not df_ltf.empty and has_open_symbol(
                            display_symbol, timeframe=strategy,
                        ):
                            last = df_ltf.iloc[-1]
                            bar_ts = last.name.to_pydatetime()
                            if getattr(bar_ts, "tzinfo", None) is None:
                                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
                            m = await manage_symbol_on_price(
                                session,
                                display_symbol,
                                float(last["high"]),
                                float(last["low"]),
                                float(last["close"]),
                                candle_ts=bar_ts,
                                timeframe=strategy,
                                avg_ltf_range=_avg_ltf_range(df_ltf),
                            )
                            result["activated"].extend(m["activated"])
                            result["closed"].extend(m["closed"])
                            result["breakeven"].extend(m["breakeven"])
                    except Exception as e:
                        disp = to_display_symbol(bingx_symbol)
                        if (strategy, disp) not in _RADAR:
                            _set_radar(disp, market_of(bingx_symbol), "no_data", strategy=strategy)
                        log.warning("scan failed for %s %s: %s", strategy, bingx_symbol, e)
                    await asyncio.sleep(0)

        log.info(
            "Scan complete - %d waiting, %d activated, %d closed, %d breakeven.",
            len(result["new_setups"]), len(result["activated"]),
            len(result["closed"]), len(result["breakeven"]),
        )
        # Yalnizca manuel taramada tek bir ozet olay yazilir.
        # Periyodik REST mutabakati (scheduler) icin ozet log YAZILMAZ;
        # gercek durum degisiklikleri zaten olay bazli loglanir.
        if source == "manual":
            duration = round((datetime.now(timezone.utc) - started_at).total_seconds(), 2)
            await record_event(
                "scan",
                (f"Manuel tarama: {len(result['new_setups'])} yeni, "
                 f"{len(result['activated'])} dolan, {len(result['closed'])} kapanan, "
                 f"{len(result['breakeven'])} BE ({duration}s)"),
                level="info",
            )

    except Exception as exc:
        try:
            await session.rollback()
        except Exception:
            pass
        await record_event("error", f"Tarama hatasi: {exc}", level="error")
        raise
    finally:
        if client is not None:
            await client.aclose()

    return result
