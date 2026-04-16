"""CRT (Candle Range Theory) + 15M CISD konfirmasyon motoru.

Sinyal Yaşam Döngüsü:
1. 4H CRT pattern tespiti → purge HIGH veya LOW
2. 15M CISD (Change in State of Delivery) konfirmasyonu beklenir
   - LOW purge (LONG): 15M'de bullish market structure shift (higher high)
   - HIGH purge (SHORT): 15M'de bearish market structure shift (lower low)
3. CISD oluşunca → sinyal "active" olur, entry/TP/SL hesaplanır
4. İnvalidasyon: fiyat CRT mumunun %50 seviyesini geçerse → "expired"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

MIN_RANGE_ATR_RATIO = 0.4
MAX_RANGE_ATR_RATIO = 3.0
PURGE_THRESHOLD_PCT = 0.0005
REVERSAL_BODY_PCT = 0.25


@dataclass
class CRTSetup:
    """4H CRT pattern tespiti — henüz CISD konfirmasyonu yok."""
    symbol: str
    direction: str
    purge_type: str
    bias: str
    bias_score: int
    key_level_high: float
    key_level_low: float
    crt_bar_time: datetime
    purge_time: datetime
    invalidation_level: float     # CRT mumunun %50'si
    purge_extreme: float          # purge noktasındaki en uç fiyat
    last_4h_high: float           # son 4H mumunun high'ı (SL icin)
    last_4h_low: float            # son 4H mumunun low'u (SL icin)
    market_type: str = "crypto"
    timeframe: str = "4h"


@dataclass
class CISDConfirmation:
    """15M CISD konfirmasyonu + trade seviyeleri."""
    entry_price: float
    stop_loss: float
    take_profit: float
    invalidation_level: float
    cisd_time: datetime
    cisd_price: float


def compute_htf_bias(df_daily: pd.DataFrame) -> str:
    """Günlük veya haftalık veriyle HTF bias hesapla.

    - Son 20 mumun EMA trendi
    - Son 5 mumun kapanış yönü
    - Genel momentum
    """
    if len(df_daily) < 10:
        return "NEUTRAL"

    closes = df_daily["close"].values
    recent_5 = df_daily.iloc[-5:]
    bullish = (recent_5["close"] > recent_5["open"]).sum()
    bearish = (recent_5["close"] < recent_5["open"]).sum()

    ema_period = min(20, len(df_daily) - 1)
    ema = df_daily["close"].ewm(span=ema_period, adjust=False).mean()
    price_vs_ema = closes[-1] / ema.iloc[-1] - 1

    score = 0
    if bullish > bearish:
        score += (bullish - bearish)
    else:
        score -= (bearish - bullish)

    if price_vs_ema > 0.01:
        score += 2
    elif price_vs_ema < -0.01:
        score -= 2

    if closes[-1] > closes[-5]:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "BULLISH"
    elif score <= -2:
        return "BEARISH"
    return "NEUTRAL"


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=1).mean()


def _calc_bias(
    df: pd.DataFrame, idx: int, direction: str, htf_bias: str = "NEUTRAL"
) -> tuple[str, int]:
    """CRT sinyal kalite skoru hesapla (0-10).

    Kriterler:
    1. Baz puan (gecerli CRT pattern)       → 2 puan
    2. Reversal mum gucu (body/range)       → max 2 puan
    3. Wick rejection (fitil orani)         → max 1 puan
    4. Konfirmasyon mumu yonu               → max 1 puan
    5. HTF bias uyumu                       → max 2 puan
    6. Hacim (reversal mumunda yuksekse)    → max 2 puan
    """
    if idx + 2 >= len(df):
        return "NEUTRAL", 2

    crt_bar = df.iloc[idx]
    reversal_bar = df.iloc[idx + 1]
    confirm_bar = df.iloc[idx + 2]

    crt_range = crt_bar["high"] - crt_bar["low"]
    if crt_range <= 0:
        return "NEUTRAL", 2

    base_score = 2

    if direction == "SHORT":
        rev_body = reversal_bar["open"] - reversal_bar["close"]
    else:
        rev_body = reversal_bar["close"] - reversal_bar["open"]

    rev_ratio = max(rev_body, 0) / crt_range
    rev_score = 2 if rev_ratio > 0.4 else (1 if rev_ratio > 0.15 else 0)

    if direction == "SHORT":
        upper_wick = reversal_bar["high"] - max(reversal_bar["open"], reversal_bar["close"])
    else:
        upper_wick = min(reversal_bar["open"], reversal_bar["close"]) - reversal_bar["low"]
    wick_ratio = upper_wick / crt_range if crt_range > 0 else 0
    wick_score = 1 if wick_ratio > 0.15 else 0

    if direction == "SHORT":
        confirm_ok = confirm_bar["close"] < confirm_bar["open"]
    else:
        confirm_ok = confirm_bar["close"] > confirm_bar["open"]
    confirm_score = 1 if confirm_ok else 0

    htf_aligned = (
        (direction == "LONG" and htf_bias == "BULLISH")
        or (direction == "SHORT" and htf_bias == "BEARISH")
    )
    htf_contrary = (
        (direction == "LONG" and htf_bias == "BEARISH")
        or (direction == "SHORT" and htf_bias == "BULLISH")
    )
    htf_score = 2 if htf_aligned else (0 if htf_contrary else 1)

    lookback = min(idx, 5)
    recent = df.iloc[max(0, idx - lookback) : idx + 2]
    avg_vol = recent["volume"].mean() if len(recent) > 0 else 0
    rev_vol = reversal_bar["volume"]
    vol_ratio = rev_vol / avg_vol if avg_vol > 0 else 1.0
    vol_score = 2 if vol_ratio > 1.3 else (1 if vol_ratio > 0.9 else 0)

    raw = base_score + rev_score + wick_score + confirm_score + htf_score + vol_score
    score = max(0, min(10, raw))

    if score >= 7:
        bias = "BULLISH" if direction == "LONG" else "BEARISH"
    elif score <= 3:
        bias = "BEARISH" if direction == "LONG" else "BULLISH"
    else:
        bias = "NEUTRAL"

    return bias, score


def _calc_live_setup_bias(direction: str, htf_bias: str = "NEUTRAL") -> tuple[str, int]:
    """Canli (2-bar) setup icin daha hafif kalite skoru."""
    htf_aligned = (
        (direction == "LONG" and htf_bias == "BULLISH")
        or (direction == "SHORT" and htf_bias == "BEARISH")
    )
    htf_contrary = (
        (direction == "LONG" and htf_bias == "BEARISH")
        or (direction == "SHORT" and htf_bias == "BULLISH")
    )

    score = 6
    if htf_aligned:
        score += 2
    elif htf_contrary:
        score -= 2
    score = max(0, min(10, score))

    if score >= 7:
        bias = "BULLISH" if direction == "LONG" else "BEARISH"
    elif score <= 3:
        bias = "BEARISH" if direction == "LONG" else "BULLISH"
    else:
        bias = "NEUTRAL"
    return bias, score


def detect_crt_setup(
    df_4h: pd.DataFrame,
    symbol: str,
    market_type: str = "crypto",
    htf_bias: str = "NEUTRAL",
) -> Optional[CRTSetup]:
    """4H verisinde CRT pattern tespit et. Henüz CISD konfirmasyonu yok."""
    df_4h = df_4h.sort_index()

    if len(df_4h) < 16:
        return None

    atr = compute_atr(df_4h)
    i = len(df_4h) - 3
    crt_bar = df_4h.iloc[i]
    next_bar = df_4h.iloc[i + 1]
    confirm_bar = df_4h.iloc[i + 2]

    crt_range = crt_bar["high"] - crt_bar["low"]
    current_atr = atr.iloc[i]

    if current_atr == 0:
        return None

    ratio = crt_range / current_atr
    if ratio < MIN_RANGE_ATR_RATIO or ratio > MAX_RANGE_ATR_RATIO:
        return None

    mid_level = crt_bar["low"] + (crt_range * 0.5)

    # HIGH PURGE → SHORT setup
    purge_above = next_bar["high"] > crt_bar["high"] * (1 + PURGE_THRESHOLD_PCT)
    if purge_above:
        reversal_body = next_bar["open"] - next_bar["close"]
        if reversal_body > 0 and reversal_body > crt_range * REVERSAL_BODY_PCT:
            if confirm_bar["close"] < crt_bar["high"]:
                bias, score = _calc_bias(df_4h, i, "SHORT", htf_bias)
                return CRTSetup(
                    symbol=symbol,
                    direction="SHORT",
                    purge_type="HIGH",
                    bias=bias,
                    bias_score=score,
                    key_level_high=round(crt_bar["high"], 8),
                    key_level_low=round(crt_bar["low"], 8),
                    crt_bar_time=crt_bar.name.to_pydatetime(),
                    purge_time=next_bar.name.to_pydatetime(),
                    invalidation_level=round(mid_level, 8),
                    purge_extreme=round(next_bar["high"], 8),
                    last_4h_high=round(confirm_bar["high"], 8),
                    last_4h_low=round(confirm_bar["low"], 8),
                    market_type=market_type,
                )

    # LOW PURGE → LONG setup
    purge_below = next_bar["low"] < crt_bar["low"] * (1 - PURGE_THRESHOLD_PCT)
    if purge_below:
        reversal_body = next_bar["close"] - next_bar["open"]
        if reversal_body > 0 and reversal_body > crt_range * REVERSAL_BODY_PCT:
            if confirm_bar["close"] > crt_bar["low"]:
                bias, score = _calc_bias(df_4h, i, "LONG", htf_bias)
                return CRTSetup(
                    symbol=symbol,
                    direction="LONG",
                    purge_type="LOW",
                    bias=bias,
                    bias_score=score,
                    key_level_high=round(crt_bar["high"], 8),
                    key_level_low=round(crt_bar["low"], 8),
                    crt_bar_time=crt_bar.name.to_pydatetime(),
                    purge_time=next_bar.name.to_pydatetime(),
                    invalidation_level=round(mid_level, 8),
                    purge_extreme=round(next_bar["low"], 8),
                    last_4h_high=round(confirm_bar["high"], 8),
                    last_4h_low=round(confirm_bar["low"], 8),
                    market_type=market_type,
                )

    # FALLBACK (2-bar canli setup):
    # Son 3 mum icindeki iki olasi 2'li kombinasyonu kontrol et:
    #   - (len-2 -> len-1)
    #   - (len-3 -> len-2)
    # Boylece son mum acik olsa bile bir onceki 2'li (05->09 gibi) yakalanabilir.
    candidate_indices = [len(df_4h) - 2, len(df_4h) - 3]
    purge_threshold = 0.0 if market_type in {"fx", "index"} else PURGE_THRESHOLD_PCT
    for live_i in candidate_indices:
        if live_i < 0 or live_i + 1 >= len(df_4h):
            continue

        live_crt = df_4h.iloc[live_i]
        live_bar = df_4h.iloc[live_i + 1]
        live_range = live_crt["high"] - live_crt["low"]
        live_atr = atr.iloc[live_i]
        if live_atr == 0 or live_range <= 0:
            continue

        live_ratio = live_range / live_atr
        if live_ratio < MIN_RANGE_ATR_RATIO or live_ratio > MAX_RANGE_ATR_RATIO:
            continue

        live_mid = live_crt["low"] + (live_range * 0.5)

        # HIGH sweep + CRT high altina geri donus => SHORT setup
        live_purge_above = live_bar["high"] > live_crt["high"] * (1 + purge_threshold)
        if live_purge_above and live_bar["close"] <= live_crt["high"]:
            bias, score = _calc_live_setup_bias("SHORT", htf_bias)
            return CRTSetup(
                symbol=symbol,
                direction="SHORT",
                purge_type="HIGH",
                bias=bias,
                bias_score=score,
                key_level_high=round(live_crt["high"], 8),
                key_level_low=round(live_crt["low"], 8),
                crt_bar_time=live_crt.name.to_pydatetime(),
                purge_time=live_bar.name.to_pydatetime(),
                invalidation_level=round(live_mid, 8),
                purge_extreme=round(live_bar["high"], 8),
                last_4h_high=round(live_bar["high"], 8),
                last_4h_low=round(live_bar["low"], 8),
                market_type=market_type,
            )

        # LOW sweep + CRT low ustune geri donus => LONG setup
        live_purge_below = live_bar["low"] < live_crt["low"] * (1 - purge_threshold)
        if live_purge_below and live_bar["close"] >= live_crt["low"]:
            bias, score = _calc_live_setup_bias("LONG", htf_bias)
            return CRTSetup(
                symbol=symbol,
                direction="LONG",
                purge_type="LOW",
                bias=bias,
                bias_score=score,
                key_level_high=round(live_crt["high"], 8),
                key_level_low=round(live_crt["low"], 8),
                crt_bar_time=live_crt.name.to_pydatetime(),
                purge_time=live_bar.name.to_pydatetime(),
                invalidation_level=round(live_mid, 8),
                purge_extreme=round(live_bar["low"], 8),
                last_4h_high=round(live_bar["high"], 8),
                last_4h_low=round(live_bar["low"], 8),
                market_type=market_type,
            )

    return None


def check_cisd_confirmation(
    df_15m: pd.DataFrame,
    setup: CRTSetup,
) -> Optional[CISDConfirmation]:
    """15M verisinde CISD (Change in State of Delivery) konfirmasyonu ara.

    CISD seviyesi:
      - SHORT: Kirmizi mumdan onceki son yesil mumun LOW'u
      - LONG:  Yesil mumdan onceki son kirmizi mumun HIGH'i

    Not: Purge zamanindan itibaren (bir mum payla) kontrol edilir.
    """
    if len(df_15m) < 3:
        return None

    recent = df_15m.sort_index()
    if setup.crt_bar_time:
        crt_ts = pd.Timestamp(setup.crt_bar_time)
        if crt_ts.tzinfo is None:
            crt_ts = crt_ts.tz_localize("UTC")
        else:
            crt_ts = crt_ts.tz_convert("UTC")
        recent = recent[recent.index >= crt_ts]

    if len(recent) < 3:
        return None

    if setup.direction == "LONG":
        return _check_bullish_cisd(recent, setup)
    else:
        return _check_bearish_cisd(recent, setup)


def _check_bullish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
) -> Optional[CISDConfirmation]:
    """15M'de bullish CISD: son bearish (kirmizi) mumun HIGH'ini yukari kirmak.

    Yesil mumdan onceki son kirmizi mumun HIGH'i = CISD seviyesi.
    Fiyat bunun ustune kapanirsa → giris.
    """
    if len(df) < 3:
        return None

    work = df
    if setup.purge_time:
        purge_ts = pd.Timestamp(setup.purge_time)
        if purge_ts.tzinfo is None:
            purge_ts = purge_ts.tz_localize("UTC")
        else:
            purge_ts = purge_ts.tz_convert("UTC")
        work = df[df.index >= purge_ts]

    if len(work) < 3:
        return None

    ref_time: Optional[pd.Timestamp] = None
    cisd_level: Optional[float] = None

    for i in range(1, len(work)):
        cur = work.iloc[i]
        prev = work.iloc[i - 1]

        # Ilk gecerli CISD referansini sabitle:
        # son bullish mumdan sonra gelen bearish mumun high'i.
        if ref_time is None:
            is_bearish = cur["close"] < cur["open"]
            was_bullish = prev["close"] >= prev["open"]
            if is_bearish and was_bullish:
                ref_time = cur.name
                cisd_level = float(cur["high"])
            continue

        if cisd_level is not None and cur.name > ref_time and cur["close"] >= cisd_level:
            entry = round(cisd_level, 8)
            sl = round(setup.purge_extreme, 8)
            tp = round(setup.key_level_high, 8)

            return CISDConfirmation(
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                invalidation_level=setup.invalidation_level,
                cisd_time=ref_time.to_pydatetime(),
                cisd_price=round(cisd_level, 8),
            )

    return None


def _check_bearish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
) -> Optional[CISDConfirmation]:
    """15M'de bearish CISD: son bullish (yesil) mumun LOW'unu asagi kirmak.

    Kirmizi mumdan onceki son yesil mumun LOW'u = CISD seviyesi.
    Fiyat bunun altina kapanirsa → giris.
    """
    if len(df) < 3:
        return None

    work = df
    if setup.purge_time:
        purge_ts = pd.Timestamp(setup.purge_time)
        if purge_ts.tzinfo is None:
            purge_ts = purge_ts.tz_localize("UTC")
        else:
            purge_ts = purge_ts.tz_convert("UTC")
        work = df[df.index >= purge_ts]

    if len(work) < 3:
        return None

    ref_time: Optional[pd.Timestamp] = None
    cisd_level: Optional[float] = None

    for i in range(1, len(work)):
        cur = work.iloc[i]
        prev = work.iloc[i - 1]

        # Ilk gecerli CISD referansini sabitle:
        # son bearish mumdan sonra gelen bullish mumun low'u.
        if ref_time is None:
            is_bullish = cur["close"] > cur["open"]
            was_bearish = prev["close"] <= prev["open"]
            if is_bullish and was_bearish:
                ref_time = cur.name
                cisd_level = float(cur["low"])
            continue

        if cisd_level is not None and cur.name > ref_time and cur["close"] <= cisd_level:
            entry = round(cisd_level, 8)
            sl = round(setup.purge_extreme, 8)
            tp = round(setup.key_level_low, 8)

            return CISDConfirmation(
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                invalidation_level=setup.invalidation_level,
                cisd_time=ref_time.to_pydatetime(),
                cisd_price=round(cisd_level, 8),
            )

    return None


def check_signal_invalidation(
    current_price: float,
    direction: str,
    invalidation_level: float,
    entry_price: float,
) -> bool:
    """Fiyat CRT mumunun %50'sini gecti mi?

    LONG: entry alttan, fiyat %50'nin ustune cikarsa → expired
    SHORT: entry ustten, fiyat %50'nin altina duserse → expired
    Entry zaten %50'nin diger tarafindaysa (edge case) → False.
    """
    if direction == "LONG":
        if entry_price >= invalidation_level:
            return False
        return current_price > invalidation_level
    else:
        if entry_price <= invalidation_level:
            return False
        return current_price < invalidation_level


def check_breakeven(
    current_price: float,
    direction: str,
    entry_price: float,
) -> bool:
    """Expired sinyal breakeven mi? Fiyat %50'yi gecip entry seviyesine geri dondu mu?

    LONG expired: fiyat yukarı geçmişti, şimdi entry'ye geri indi mi?
    SHORT expired: fiyat aşağı geçmişti, şimdi entry'ye geri çıktı mı?
    """
    if direction == "LONG":
        return current_price <= entry_price
    else:
        return current_price >= entry_price


def check_tp_sl_hit(
    current_price: float,
    direction: str,
    take_profit: float,
    stop_loss: float,
) -> Optional[str]:
    """TP veya SL'ye ulaşılıp ulaşılmadığını kontrol et.

    Returns: 'hit_tp', 'hit_sl', veya None.
    """
    if direction == "LONG":
        if current_price >= take_profit:
            return "hit_tp"
        if current_price <= stop_loss:
            return "hit_sl"
    else:
        if current_price <= take_profit:
            return "hit_tp"
        if current_price >= stop_loss:
            return "hit_sl"
    return None
