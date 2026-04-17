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
MIN_CISD_TREND_CANDLES = 2
DOJI_BODY_RATIO_MAX = 0.10


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


def _candle_state(row: pd.Series) -> str:
    """Mum durumunu siniflandir: bullish / bearish / doji."""
    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    rng = h - l
    if rng <= 0:
        return "doji"

    body = abs(c - o)
    body_to_range = body / rng
    if body_to_range <= DOJI_BODY_RATIO_MAX:
        return "doji"
    if c > o:
        return "bullish"
    if c < o:
        return "bearish"
    return "doji"


def detect_crt_setup(
    df_4h: pd.DataFrame,
    symbol: str,
    market_type: str = "crypto",
    htf_bias: str = "NEUTRAL",
) -> Optional[CRTSetup]:
    """4H verisinde CRT pattern tespit et (son 3 mum icinde)."""
    df_4h = df_4h.sort_index()

    if len(df_4h) < 16:
        return None

    atr = compute_atr(df_4h)
    purge_threshold = 0.0 if market_type in {"fx", "index"} else PURGE_THRESHOLD_PCT

    # Son 3 mum icindeki ardışık 2'li kombinasyonlari kontrol et:
    #   1) len-2 -> len-1 (en guncel)
    #   2) len-3 -> len-2 (bir onceki)
    for live_i in (len(df_4h) - 2, len(df_4h) - 3):
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

    min_confirm_time: Optional[pd.Timestamp] = None
    if setup.purge_time:
        purge_ts = pd.Timestamp(setup.purge_time)
        if purge_ts.tzinfo is None:
            purge_ts = purge_ts.tz_localize("UTC")
        else:
            purge_ts = purge_ts.tz_convert("UTC")
        min_confirm_time = purge_ts

    if setup.direction == "LONG":
        return _check_bullish_cisd(recent, setup, min_confirm_time=min_confirm_time)
    else:
        return _check_bearish_cisd(recent, setup, min_confirm_time=min_confirm_time)


def _check_bullish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
    min_confirm_time: Optional[pd.Timestamp] = None,
) -> Optional[CISDConfirmation]:
    """15M bullish CISD:
    Son dusus (kirmizi) mum grubunun ilk mumunun OPEN seviyesinin
    ustunde kapanis aranir.

    Not: Tek mumluk dusus grubu CISD referansi olarak kabul edilmez.
    """
    if len(df) < 3:
        return None

    # CISD referansi, purge zamanindan once baslayan son mum grubunda da
    # olabilecegi icin crt penceresinden itibaren tum 15M dizi kullanilir.
    work = df

    if len(work) < 3:
        return None

    ref_time: Optional[pd.Timestamp] = None
    cisd_level: Optional[float] = None
    bearish_group_start: Optional[int] = None
    bearish_group_len = 0

    for i in range(len(work)):
        cur = work.iloc[i]
        cur_state = _candle_state(cur)
        cur_is_bearish = cur_state == "bearish"
        cur_is_bullish = cur_state == "bullish"
        cur_is_doji = cur_state == "doji"

        if cur_is_bearish:
            if bearish_group_start is None:
                bearish_group_start = i
                bearish_group_len = 1
            else:
                bearish_group_len += 1
            continue

        # Doji mumlar trend blogunu bozmaz; CISD akisini etkilemesin.
        if cur_is_doji:
            continue

        # Bearish grup bittiginde CISD referansini guncelle.
        # Tek mumluk gruplar atlanir; onceki gecerli referans korunur.
        if bearish_group_start is not None:
            if bearish_group_len >= MIN_CISD_TREND_CANDLES:
                first_bearish = work.iloc[bearish_group_start]
                ref_time = first_bearish.name
                cisd_level = float(first_bearish["open"])
            bearish_group_start = None
            bearish_group_len = 0

        if (
            ref_time is not None
            and cisd_level is not None
            and cur.name > ref_time
            and (min_confirm_time is None or cur.name > min_confirm_time)
            and cur_is_bullish
            and float(cur["close"]) > cisd_level
        ):
            entry = round(cisd_level, 8)
            sl = round(setup.purge_extreme, 8)
            tp = round(setup.key_level_high, 8)

            return CISDConfirmation(
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                invalidation_level=setup.invalidation_level,
                # Onay zamani, referans mum degil; kirilimi yapan mumdur.
                cisd_time=cur.name.to_pydatetime(),
                cisd_price=round(cisd_level, 8),
            )

    return None


def _check_bearish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
    min_confirm_time: Optional[pd.Timestamp] = None,
) -> Optional[CISDConfirmation]:
    """15M bearish CISD:
    Son yukselis (yesil) mum grubunun ilk mumunun OPEN seviyesinin
    altinda kapanis aranir.

    Not: Tek mumluk yukselis grubu CISD referansi olarak kabul edilmez.
    """
    if len(df) < 3:
        return None

    # CISD referansi, purge zamanindan once baslayan son mum grubunda da
    # olabilecegi icin crt penceresinden itibaren tum 15M dizi kullanilir.
    work = df

    if len(work) < 3:
        return None

    ref_time: Optional[pd.Timestamp] = None
    cisd_level: Optional[float] = None
    bullish_group_start: Optional[int] = None
    bullish_group_len = 0

    for i in range(len(work)):
        cur = work.iloc[i]
        cur_state = _candle_state(cur)
        cur_is_bullish = cur_state == "bullish"
        cur_is_bearish = cur_state == "bearish"
        cur_is_doji = cur_state == "doji"

        if cur_is_bullish:
            if bullish_group_start is None:
                bullish_group_start = i
                bullish_group_len = 1
            else:
                bullish_group_len += 1
            continue

        # Doji mumlar trend blogunu bozmaz; CISD akisini etkilemesin.
        if cur_is_doji:
            continue

        # Bullish grup bittiginde CISD referansini guncelle.
        # Tek mumluk gruplar atlanir; onceki gecerli referans korunur.
        if bullish_group_start is not None:
            if bullish_group_len >= MIN_CISD_TREND_CANDLES:
                first_bullish = work.iloc[bullish_group_start]
                ref_time = first_bullish.name
                cisd_level = float(first_bullish["open"])
            bullish_group_start = None
            bullish_group_len = 0

        if (
            ref_time is not None
            and cisd_level is not None
            and cur.name > ref_time
            and (min_confirm_time is None or cur.name > min_confirm_time)
            and cur_is_bearish
            and float(cur["close"]) < cisd_level
        ):
            entry = round(cisd_level, 8)
            sl = round(setup.purge_extreme, 8)
            tp = round(setup.key_level_low, 8)

            return CISDConfirmation(
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                invalidation_level=setup.invalidation_level,
                # Onay zamani, referans mum degil; kirilimi yapan mumdur.
                cisd_time=cur.name.to_pydatetime(),
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
