"""CRT (Candle Range Theory) + 15M CISD konfirmasyon motoru.

Sinyal Yaşam Döngüsü:
1. 4H CRT pattern tespiti → purge HIGH veya LOW
2. 15M CISD (Change in State of Delivery) konfirmasyonu beklenir
   - LOW purge (LONG): 15M'de bullish market structure shift (higher high)
   - HIGH purge (SHORT): 15M'de bearish market structure shift (lower low)
3. CISD oluşunca → sinyal "active" olur, entry/TP/SL hesaplanır
4. İnvalidasyon: fiyat CRT mumunun %60 seviyesini geçerse → "expired"
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
CRT_PURGE_SEARCH = 3  # C1'den sonra purge/C2 aramak icin bakilacak mum sayisi (1-3)
# C1 aday penceresi: son HTF bar (forming) haric geriye. 4H/1D: 4 aday.
# 1H: daha genis; 04:00 C1 + 07:00 C2 ogleden sonra da radarda kalsin.
CRT_C1_LOOKBACK = 5
CRT_C1_LOOKBACK_1H = 8
DOJI_BODY_RATIO_MAX = 0.10
# C2 rengi hard filtresi hangi HTF'lerde gecerli (25.09, kullanici karari): 1D'de
# kalkti, yanlis renk yalniz skorda (base 0) ceza olur. 4H/1H'te guclu yanlis renk
# C1'i elemeye devam eder.
C2_COLOR_HARD_FILTER_TFS = frozenset({"4h", "1h"})
# IFVG'nin mid'i C1 araliginda olmali mi (25.09 kapatildi, kullanici karari): purge
# fitilinin icindeki FVG'nin inversiyonu da gecerli IFVG. True eski davranis.
REQUIRE_IFVG_MID_IN_C1 = False
# Waiting/pending iptal: purge tarafından CRT range'in bu kadarı (LTF close).
CRT_INVALIDATION_FRAC = 0.60
# Swing pivot hassasiyeti:
# 1 => birer mum sol/sag karsilastirma (daha hizli MSS yakalar)
# 2+ => daha katı pivot tanimi
SWING_PIVOT_BARS = 1
# 1D yapisal bias icin biraz daha katı pivot (5-mum: 2 sol + pivot + 2 sag)
DAILY_SWING_PIVOT_BARS = 2
# Yapisal bias kalici bir durumdur; kirilim bu kadar KAPALI gunden eskiyse
# "bayat" sayilir ve gunluk bias karari taze okumaya (ICT) birakilir.
# 7: olculen kirilim yasi ortancasi 5 gun (26 sembol, 09.09.2026).
STRUCTURE_STALE_DAYS = 7
# MSS kapanis margin: max(seviye*pct, ort.15M range*mult).
# XAU (~4090): %0.06 ≈ 2.45$, range%30 ≈ 2$ → tipik 2-3$ bandi.
CISD_STRONG_CLOSE_PCT = 0.0006
CISD_STRONG_CLOSE_RANGE_MULT = 0.30
CISD_MARGIN_LOOKBACK = 20
# Purge rejection wick: fitil / CRT range esigi (kalite skoru)
PURGE_WICK_SCORE_PCT = 0.10
# C2'nin C1 araligina geri donusu: (C2 close) supurulen kenardan C1 range'inin
# yuzde kaci kadar iceri girdi. Supurup aralik dibinden kapatan bir C2, derin
# geri donen bir C2'den zayiftir; kod eskiden yalnizca "iceri kapandi mi"
# (yani %0 bile yeterli) bakiyordu. Esigin altinda skordan C2_RECLAIM_PENALTY
# dusulur -- HARD FILTRE DEGIL (ARB 09.09 %15 ile +3.16R yapti).
#
# 19.09: ceza 2 -> 4. 09.09'da "bu metrik kazananla kaybedeni ayirt etmiyor" diye not
# edilmisti; gercek setuplarla olculunce bu YANLISLANDI -- geri donus yuzdesi ile sonuc
# monoton iliskili cikti (olcum tablosu ve orneklem: IZLEME.md -> "C2 geri donus cezasi").
# Hard filtre yerine ceza buyutuldu -- SMT'li (+2) guclu setup hala absorbe edebilsin.
# Olcum ve karar kurali: IZLEME.md "5. C2 geri donus cezasi".
C2_RECLAIM_WEAK_PCT = 25.0
C2_RECLAIM_PENALTY = 4
# LTF IFVG asgari bosluk boyutu: gap >= bu oran * son 20 LTF mumunun ort. range'i.
# Amac: bir 5M mumunun kucuk bir kesridi kadar olan "bosluklari" (gurultu) eleyip
# yalnizca gercek displacement iceren FVG'leri kabul etmek. Ayarlanabilir.
MIN_IFVG_GAP_RANGE_FRAC = 0.15
_IFVG_RANGE_LOOKBACK = 20
# BPR (Balanced Price Range): invert olan ters yonlu FVG'ye ek olarak, inversiyonu yapan
# hamlenin biraktigi AYNI YONLU FVG de varsa ve ikisi KESISIYORSA ortak alan BPR'dir.
# Ikinci FVG'nin orta mumu inversiyon mumundan en fazla bu kadar sonra olabilir (LTF mumu).
_BPR_BARS_AFTER = 3
_PD_MAJOR_LABELS = frozenset({"PDH", "PDL", "PWH", "PWL"})
_PD_MONTHLY_LABELS = frozenset({"PMH", "PML"})
_PD_STRUCT_LABELS = frozenset({"FVG", "OB"})


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
    invalidation_level: float     # CRT mumunun %60'i (purge tarafindan)
    purge_extreme: float          # purge noktasındaki en uç fiyat
    last_4h_high: float           # son 4H mumunun high'ı (SL icin)
    last_4h_low: float            # son 4H mumunun low'u (SL icin)
    market_type: str = "crypto"
    timeframe: str = "4h"
    smt_pair: Optional[str] = None  # SMT divergence bulunan korele parite (gosterim adi)
    pd_array: Optional[str] = None  # Purge (C2) mumunun dokundugu PD array etiketleri (or. 'PDH,FVG')
    c2_closed: bool = False  # Purge (C2) mumu KAPANMIS mi? (forming/kapanmamis ise False)
    color_opposite: bool = True  # CRT/purge farkli renk; ayni renk skorda baz -1 (hard filter degil)
    target_consumed: bool = False  # C1 hedef tarafi sonradan supuruldu; sinyal yok, radar gosterir
    # Purge (SL) tarafi C2'den SONRA da alindi: fiyat purge_extreme'i gecti, yani bu setup'in
    # stopu fiilen yenmis. Hedef tarafinin aynasi (target_consumed) ve onun gibi aday secimini
    # etkiler (18.09): olu aday, canli aday varken slotu tutmaz. Tek adaysa yine donulur --
    # kapiyi scanner'in kendi past_sl kontrolu kapatir.
    stop_breached: bool = False
    c2_reclaim: Optional[float] = None  # C2 kapanisi C1 araliginin %kacina dondu
    # bias_score'un kalem kalem kirilimi (SCORE_PART_KEYS). Yalniz olcum icin: Setup
    # Journal'a yazilir, motor karari sadece bias_score'a bakar.
    score_parts: Optional[dict] = None
    # Kalemlerin arkasindaki surekli olcumler (SCORE_FEATURE_KEYS) -- yine yalniz olcum.
    score_features: Optional[dict] = None


@dataclass
class CISDConfirmation:
    """15M trade seviyeleri + (varsa) MSS kapanis onayi.

    cisd_time None ise seviyeler hesaplanmistir ama MSS kirilimi henuz yok.
    """
    entry_price: float
    stop_loss: float
    take_profit: float
    invalidation_level: float
    cisd_price: float  # MSS / market kirilim seviyesi
    cisd_time: Optional[datetime] = None  # MSS onay mumunun acilis zamani
    mss_ref_time: Optional[datetime] = None  # Kirilan swing mumunun acilis zamani
    entry_model: str = "cisd"  # cisd | mss | ifvg | bpr  (ifvg/bpr'yi scanner ezer)
    ifvg_low: Optional[float] = None
    ifvg_high: Optional[float] = None
    # BPR (iki FVG'nin kesisimi) varsa bolgesi; IFVG bolgesinin ICINDE kalir.
    bpr_low: Optional[float] = None
    bpr_high: Optional[float] = None
    # Kesisimi olusturan ayni yonlu FVG'nin KENDI sinirlari. YALNIZ OLCUM (22.09):
    # bacagin girise yakin kenari kesisiminkinin disinda kaldigi icin once dokunulur.
    bleg_low: Optional[float] = None
    bleg_high: Optional[float] = None
    # Iki adayin HAM seviyeleri. `_pick_wider_stop` yalnizca kazanani dondurdugu icin
    # kaybeden aday kayboluyordu; entry modeli karsilastirmasi (Setup Journal `entries`)
    # ayni setupta ikisini de izlemek zorunda. YALNIZ OLCUM -- motor karari bunlari kullanmaz.
    cisd_level: Optional[float] = None
    mss_level: Optional[float] = None

    @property
    def confirmed(self) -> bool:
        return self.cisd_time is not None


# NY 17:00 hizali seans kovalari (FX/metal/endeks/petrol) UTC'de onceki gunun
# 21:00/22:00'inda baslar. Gun/hafta/ay kovanin ISLEM GUNUNE gore belirlenmeli:
# 12 saat ileri kaydirmak bu kovalari islem gunlerine tasir, UTC gece yarisi
# mumlarini (kripto) ise ayni gunde birakir.
_SESSION_DAY_SHIFT = pd.Timedelta(hours=12)


def _trading_day_shift(index) -> pd.Timedelta:
    """Gunluk seri NY-hizali seans kovasi mi (UTC gece yarisi disinda baslayan bar)?"""
    idx = pd.DatetimeIndex(index)
    if len(idx) and bool(((idx.hour != 0) | (idx.minute != 0)).any()):
        return _SESSION_DAY_SHIFT
    return pd.Timedelta(0)


def _drop_forming_daily(df_daily: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Kapanmamis (olusan) gunluk mumu dus; bos/yetersiz ise oldugu gibi don.

    Karar zamanla verilir: son barin 24 saati `now` aninda bitmediyse olusan
    mumdur. Eskiden takvim tarihi karsilastiriliyordu; NY kovasi UTC'de onceki
    gun basladigi icin seans sembollerinde olusan mum gunun ~21 saati kapanmis
    sayiliyor, 1D bias yarim mumla hesaplaniyordu. UTC gece yarisi mumlarinda
    iki kural ayni sonucu verir.
    """
    if df_daily is None or df_daily.empty:
        return df_daily
    closed = df_daily.sort_index()
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    last_ts = pd.Timestamp(closed.index[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    if last_ts + pd.Timedelta(days=1) > now_utc:
        closed = closed.iloc[:-1]
    return closed


def compute_ict_bias(df_daily: pd.DataFrame, *, drop_forming_day: bool = True) -> str:
    """Tek-bar ICT/purge karakteri (momentum/EMA YOK).

    Son KAPANAN bar vs bir onceki:
      1. close > prev high                         -> BULLISH (temiz break)
      2. close < prev low                          -> BEARISH
      3. high > prev high & close icerde           -> BEARISH (failed high)
      4. low  < prev low  & close icerde           -> BULLISH (failed low)
      5. Outside / inside bar                      -> NEUTRAL

    1D trade filtresinde ASIL bias DEGILDIR; skor onayinda ve 1W bilgi
    sutununda kullanilir. Yetersiz veride NEUTRAL.
    """
    if df_daily is None or df_daily.empty:
        return "NEUTRAL"
    closed = _drop_forming_daily(df_daily) if drop_forming_day else df_daily.sort_index()
    if len(closed) < 2:
        return "NEUTRAL"

    d1 = closed.iloc[-1]
    d2 = closed.iloc[-2]
    d1_close = float(d1["close"])
    d1_high = float(d1["high"])
    d1_low = float(d1["low"])
    d2_high = float(d2["high"])
    d2_low = float(d2["low"])

    if d1_close > d2_high:
        return "BULLISH"
    if d1_close < d2_low:
        return "BEARISH"

    swept_high = d1_high > d2_high
    swept_low = d1_low < d2_low
    if swept_high and not swept_low:
        return "BEARISH"
    if swept_low and not swept_high:
        return "BULLISH"
    if swept_high and swept_low:
        return "NEUTRAL"
    return "NEUTRAL"


def compute_daily_bias(df_daily: pd.DataFrame) -> str:
    """1D trade bias — Swing structure + ICT, yapisal bayatlik gozetilerek.

    - Yapisal kirilim TAZE ise (<= STRUCTURE_STALE_DAYS): structure VE ict ayni
      yon olmali, aksi halde NEUTRAL.
    - Yapisal kirilim BAYAT ise (veya hic yoksa): structure yon belirtmez,
      karar taze okumaya (ict) birakilir.

    Neden: yapisal bias kalici bir durumdur ve aylarca aralikta kalan bir
    enstrumanda haftalar oncesinin yonunu tasir. Eski hali (kosulsuz AND)
    boyle bir bayat yonun taze okumayi vetolamasina izin veriyordu — XAUUSD
    09.09.2026: structure 25.08'den beri BULLISH (14 gun), ict BEARISH
    (08.09 kapanisi 07.09 low'unun altinda) -> daily NEUTRAL kaliyordu.

    Olculen etki (26 sembol, 09.09.2026): yonlu bias 18/26 -> 20/26.
    Yalnizca bayatlik esigi koyup AND'i korumak (yani ict'ye dusmemek) 15/26'ya
    DUSURUYORDU; structure'i bayat ama ict ile hemfikir olan ETH/US100/USDCAD
    gibi dogru okumalari da oldurdugu icin tercih edilmedi.

    UI, Telegram, hard filter ve skor 1D kalemi bu degeri kullanir.
    """
    if df_daily is None or df_daily.empty:
        return "NEUTRAL"
    try:
        structure, age = htf_bias_with_age(df_daily)
    except Exception:
        structure, age = "NEUTRAL", None
    try:
        ict = compute_ict_bias(df_daily)
    except Exception:
        ict = "NEUTRAL"
    if structure == "NEUTRAL" or age is None or age > STRUCTURE_STALE_DAYS:
        return ict
    return structure if structure == ict else "NEUTRAL"


def compute_htf_bias(df_daily: pd.DataFrame, *, drop_forming_day: bool = True) -> str:
    """1D yapisal bias — son kapanisla kirilan swing high/low yonu.

    Yasi da gerekiyorsa `htf_bias_with_age` kullan.
    """
    return htf_bias_with_age(df_daily, drop_forming_day=drop_forming_day)[0]


def htf_bias_with_age(
    df_daily: pd.DataFrame, *, drop_forming_day: bool = True,
) -> tuple[str, Optional[int]]:
    """(yapisal bias, son kirilimdan bu yana gecen KAPALI gun sayisi).

    Momentum/EMA YOK. Forming gun dusulur (drop_forming_day=True).
      - close > son teyitli swing high -> BULLISH (son kirilim)
      - close < son teyitli swing low  -> BEARISH
      - kirilim yoksa onceki bias korunur (inside bar NEUTRAL'e dusmez)
      - hic kirilim yoksa / veri yetmezse -> (NEUTRAL, None)

    Bias KALICI bir durumdur: ters kirilim olana kadar eski yonu tasir. Yas bu
    yuzden onemli — aylarca aralikta kalan bir enstrumanda yon bayatlar.
    `compute_daily_bias` bayat yapiyi vetodan dusurmek icin bu yasi kullanir.
    """
    if df_daily is None or df_daily.empty:
        return "NEUTRAL", None
    closed = _drop_forming_daily(df_daily) if drop_forming_day else df_daily.sort_index()
    span = DAILY_SWING_PIVOT_BARS
    # pivot teyidi icin sagda `span` mum gerekir
    if closed is None or len(closed) < span * 2 + 3:
        return "NEUTRAL", None

    n = len(closed)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(span, n - span):
        if _is_swing_high(closed, i, span):
            highs.append((i, float(closed.iloc[i]["high"])))
        if _is_swing_low(closed, i, span):
            lows.append((i, float(closed.iloc[i]["low"])))

    bias = "NEUTRAL"
    broke_at: Optional[int] = None
    for i in range(span * 2, n):
        close = float(closed.iloc[i]["close"])
        # Pivot j, j+span barindan itibaren teyitlidir
        conf_highs = [p for p in highs if p[0] + span <= i]
        conf_lows = [p for p in lows if p[0] + span <= i]

        broke_h = False
        broke_l = False
        sh_i = -1
        sl_i = -1
        if conf_highs:
            sh_i, sh = conf_highs[-1]
            broke_h = close > sh
        if conf_lows:
            sl_i, sl = conf_lows[-1]
            broke_l = close < sl

        if broke_h and broke_l:
            # Nadir: ayni barda iki seviye; daha yeni pivotun kirilimi kazanir
            bias = "BULLISH" if sh_i >= sl_i else "BEARISH"
            broke_at = i
        elif broke_h:
            bias = "BULLISH"
            broke_at = i
        elif broke_l:
            bias = "BEARISH"
            broke_at = i
        # else: bias degismez (inside / aralik ici kapanis)
    age = None if broke_at is None else (n - 1 - broke_at)
    return bias, age


def daily_to_weekly_ohlcv(df_1d: pd.DataFrame) -> pd.DataFrame:
    """1D OHLCV'yi ISO hafta (Pazartesi baslangic) bazinda haftalik mumlara cevir.

    Seans kovalari islem gunune kaydirilir (`_trading_day_shift`); aksi halde NY
    Pazartesi kovasi (Pazar 21:00 UTC) bir onceki haftaya dusuyordu.
    """
    if df_1d is None or df_1d.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    d = df_1d.sort_index().copy()
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        d.index = d.index.tz_convert("UTC")
    d.index = d.index + _trading_day_shift(d.index)
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in d.columns:
        agg["volume"] = "sum"
    weekly = d.resample("W-MON", label="left", closed="left").agg(agg).dropna(how="any")
    return weekly


def compute_weekly_bias(df_1d: pd.DataFrame, now: pd.Timestamp | None = None) -> str:
    """Haftalik bias — ICT/purge; kalite skorunda uyumda +1 (hard filter degil).

    1D -> haftalik OHLCV; forming hafta dusulur; son iki kapali haftaya
    `compute_ict_bias` uygulanir. Momentum/EMA yok.
    """
    weekly = daily_to_weekly_ohlcv(df_1d)
    if weekly.empty or len(weekly) < 2:
        return "NEUTRAL"

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    # Seans serisinde haftalar islem gunune kaydirildi; "su anki hafta" da oyle.
    cur_iso = (now + _trading_day_shift(df_1d.index)).isocalendar()
    cur_key = (int(cur_iso.year), int(cur_iso.week))

    def _week_key(ts) -> tuple[int, int]:
        iso = pd.Timestamp(ts).tz_convert("UTC").isocalendar()
        return (int(iso.year), int(iso.week))

    closed = weekly[[_week_key(ts) != cur_key for ts in weekly.index]]
    if len(closed) < 2:
        return "NEUTRAL"
    return compute_ict_bias(closed, drop_forming_day=False)


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


# Kalite skorunun kalem kalem kirilimi (olcum icin; motor karari yalnizca "score"u kullanir).
# Setup Journal'a JSON olarak yazilir -> hangi +1/+2 gercekten kazandiriyor sorusu olculebilsin.
SCORE_PART_KEYS = (
    "base", "htf", "weekly", "c2_closed", "pd_major", "pd_monthly", "pd_struct",
    "wick", "ifvg", "reclaim", "smt", "raw", "score",
)


# Kalemlerin arkasindaki SUREKLI olcumler + uygunluk bayraklari. Puan degil: skor tablosunun
# ESIKLERINI (wick %, IFVG bosluk, doji orani, ATR bandi, dar stop) sonradan birikmis veriyle
# sorgulamak icin saklanir. "0 puan" ile "bakilamadi" ayrimi da burada (*_checked / *_possible).
SCORE_FEATURE_KEYS = (
    "wick_frac", "c2_body_frac", "reclaim_pct", "ifvg_gap_frac", "c2_range_frac",
    "range_atr", "stop_range_mult", "stop_pct",
    "ifvg_checked", "bias_checked", "smt_possible",
)


def _empty_score_parts() -> dict:
    """Skor hesaplanamadigi erken donuslerde (gecersiz C1/C2) notr kirilim."""
    parts = {k: 0 for k in SCORE_PART_KEYS}
    parts["raw"] = parts["score"] = 1
    return parts


def _pd_array_score(pd_labels: Optional[list[str]]) -> int:
    """Kademeli PD skoru (ham; SMT'siz tavan 9'da kesilir).

    - Major HTF (PDH/PDL/PWH/PWL): +1
    - Aylik (PMH/PML): +1 (major ile stack)
    - Yapisal (FVG veya OB): +1 (ikisi birden yine +1)
    - EQH/EQL: 0 (yalnizca gosterim)
    """
    if not pd_labels:
        return 0
    labels = {str(x).upper() for x in pd_labels}
    score = 0
    if labels & _PD_MAJOR_LABELS:
        score += 1
    if labels & _PD_MONTHLY_LABELS:
        score += 1
    if labels & _PD_STRUCT_LABELS:
        score += 1
    return score


def c2_reclaim_pct(
    crt_row: pd.Series, purge_row: pd.Series, direction: str,
) -> Optional[float]:
    """C2 kapanisi, C1 araliginin supurulen kenardan yuzde kacina dondu.

    LONG (LOW purge): (C2.close - C1.low) / C1.range
    SHORT (HIGH purge): (C1.high - C2.close) / C1.range
    %100'u asabilir (C2 C1'in tamamen karsi tarafina kapatmissa).
    """
    hi = float(crt_row["high"])
    lo = float(crt_row["low"])
    rng = hi - lo
    if rng <= 0:
        return None
    close = float(purge_row["close"])
    inside = (close - lo) if direction == "LONG" else (hi - close)
    return inside / rng * 100.0


def _purge_wick_ratio(
    crt_range: float,
    purge_row: pd.Series,
    direction: str,
) -> float:
    """Purge rejection fitilinin CRT range'ine orani (skor bundan turetilir; olcum icin ayri)."""
    if crt_range <= 0:
        return 0.0
    o = float(purge_row["open"])
    c = float(purge_row["close"])
    h = float(purge_row["high"])
    l = float(purge_row["low"])
    if direction == "SHORT":
        wick = h - max(o, c)
    else:
        wick = min(o, c) - l
    return max(0.0, wick / crt_range)


def _purge_wick_score(
    crt_range: float,
    purge_row: pd.Series,
    direction: str,
) -> int:
    """Purge rejection wick: fitil / CRT range >= PURGE_WICK_SCORE_PCT -> +1."""
    return 1 if _purge_wick_ratio(crt_range, purge_row, direction) >= PURGE_WICK_SCORE_PCT else 0


def _calc_live_setup_bias(
    df_4h: pd.DataFrame,
    crt_idx: int,
    direction: str,
    htf_bias: str = "NEUTRAL",
    pd_labels: Optional[list[str]] = None,
    purge_idx: Optional[int] = None,
    df_1d: Optional[pd.DataFrame] = None,
    *,
    pd_hit: bool = False,  # legacy; pd_labels yoksa bool(pd_hit) -> tek yapisal puan yok
    df_ltf: Optional[pd.DataFrame] = None,
    timeframe: str = "4h",
    with_parts: bool = False,
) -> tuple[str, int] | tuple[str, int, dict, dict]:
    """Canli CRT setup icin zengin kalite skoru.

    Kriterler:
      - Baz: dogru C2 rengi +2 / doji veya ayni renk +1
      - 1D hiza VEYA reversal-at-PD             : +2
      - Weekly bias uyumu                       : +1
      - C2 gecerli (1D'de forming C2 de +1)     : +1
      - PDH/PDL/PWH/PWL                         : +1
      - PMH/PML (major ile stack)               : +1
      - HTF FVG veya OB                         : +1
      - Purge rejection wick                    : +1
      - LTF IFVG (CRT ici invert, acik bolge)   : +1

    Ham toplam 13'e cikabilir; SMT'siz tavan 9. SMT +2 ile max 11 (premium).

    `with_parts=True` ise kalem kalem kirilim (SCORE_PART_KEYS) VE kalemlerin arkasindaki
    SUREKLI olcumler (SCORE_FEATURE_KEYS: wick orani, C2 gövde orani, geri donus %, IFVG bosluk
    orani + uygunluk bayraklari) doner. Sureklileri saklamanin sebebi: bir kalemin esigini
    (ör. PURGE_WICK_SCORE_PCT) yeni veri beklemeden, birikmis olcumle sorgulayabilmek.
    Amac olcum: hangi +1/+2 gercekten kazandiriyor? Toplam skor Journal'da vardi ama
    kirilim hicbir yere yazilmiyordu, bu yuzden kalem bazli hicbir sey olculemiyordu.
    Kirilim MOTOR KARARINI DEGISTIRMEZ -- yalnizca ayni hesabin raporudur.
    """
    if purge_idx is None:
        purge_idx = crt_idx + 1
    if purge_idx >= len(df_4h):
        return ("NEUTRAL", 1, _empty_score_parts(), {}) if with_parts else ("NEUTRAL", 1)
    crt = df_4h.iloc[crt_idx]
    rev = df_4h.iloc[purge_idx]
    crt_range = float(crt["high"] - crt["low"])
    if crt_range <= 0:
        return ("NEUTRAL", 1, _empty_score_parts(), {}) if with_parts else ("NEUTRAL", 1)

    c2_state = _candle_state(rev)
    correct_c2 = (
        (direction == "SHORT" and c2_state == "bearish")
        or (direction == "LONG" and c2_state == "bullish")
    )
    doji_c2 = c2_state == "doji"
    # Dogru renk +2, doji +1, yanlis renk 0 (25.09: 1D'de yanlis renk artik setup
    # olabiliyor; 4H/1H'te yalniz Journal'in elenen aday skorunu etkiler).
    base_score = 2 if correct_c2 else (1 if doji_c2 else 0)

    labels = list(pd_labels) if pd_labels is not None else ([] if not pd_hit else ["FVG"])
    pd_score = _pd_array_score(labels)
    label_set = {str(x).upper() for x in labels}
    pd_major_score = 1 if label_set & _PD_MAJOR_LABELS else 0
    pd_monthly_score = 1 if label_set & _PD_MONTHLY_LABELS else 0
    pd_struct_score = 1 if label_set & _PD_STRUCT_LABELS else 0

    try:
        daily_bias = (
            compute_daily_bias(df_1d)
            if df_1d is not None and not df_1d.empty
            else (htf_bias or "NEUTRAL")
        )
    except Exception:
        daily_bias = htf_bias or "NEUTRAL"
    htf_aligned = (
        (direction == "LONG" and daily_bias == "BULLISH")
        or (direction == "SHORT" and daily_bias == "BEARISH")
    )
    htf_contrary = (
        (direction == "LONG" and daily_bias == "BEARISH")
        or (direction == "SHORT" and daily_bias == "BULLISH")
    )
    has_major_pd = bool(
        {str(x).upper() for x in labels} & (_PD_MAJOR_LABELS | _PD_MONTHLY_LABELS)
    )
    reversal_at_pd = (
        (not htf_aligned)
        and (not htf_contrary)
        and has_major_pd
        and (correct_c2 or doji_c2)
    )
    if htf_aligned or reversal_at_pd:
        htf_score = 2
    elif htf_contrary:
        htf_score = -2
    else:
        htf_score = 0

    try:
        weekly_bias = (
            compute_weekly_bias(df_1d)
            if df_1d is not None and not df_1d.empty
            else "NEUTRAL"
        )
    except Exception:
        weekly_bias = "NEUTRAL"
    weekly_aligned = (
        (direction == "LONG" and weekly_bias == "BULLISH")
        or (direction == "SHORT" and weekly_bias == "BEARISH")
    )
    weekly_score = 1 if weekly_aligned else 0

    # Yalniz kapali C2 +1 (tum TF). 25.09'a kadar 1D'de forming C2 de +1 aliyordu;
    # 1D C2 kapanisini beklemeyi biraktigi icin bedel skorda odenir (1H gibi).
    c2_closed_score = 1 if purge_idx < (len(df_4h) - 1) else 0

    wick_score = _purge_wick_score(crt_range, rev, direction)

    # Zayif geri donus cezasi: C2 supurup C1 araliginin dibinden kapattiysa
    # ters donus zayif sayilir. Ceza, tavan uygulanmadan ONCE raw'dan duser.
    reclaim = c2_reclaim_pct(crt, rev, direction)
    reclaim_score = (
        -C2_RECLAIM_PENALTY
        if reclaim is not None and reclaim < C2_RECLAIM_WEAK_PCT
        else 0
    )

    ifvg_score = 0
    ifvg_zone = None
    ifvg_checked = df_ltf is not None and not df_ltf.empty
    if ifvg_checked:
        purge_ts = df_4h.index[purge_idx]
        tf = (timeframe or "4h").lower()
        c2_h = 24.0 if tf == "1d" else (1.0 if tf == "1h" else 4.0)
        ifvg_zone = detect_ltf_ifvg(
            df_ltf, direction, purge_ts,
            crt_low=float(crt["low"]),
            crt_high=float(crt["high"]),
            crt_bar_time=df_4h.index[crt_idx],
            c2_hours=c2_h,
        )
        if ifvg_zone is not None:
            ifvg_score = 1

    raw = (
        base_score + htf_score + weekly_score
        + c2_closed_score + pd_score + wick_score + ifvg_score
        + reclaim_score
    )
    # SMT'siz tavan 9; +SMT 2 ile max 11 (premium).
    score = max(0, min(9, raw))

    if score >= 7:
        bias = "BULLISH" if direction == "LONG" else "BEARISH"
    elif score <= 3:
        bias = "BEARISH" if direction == "LONG" else "BULLISH"
    else:
        bias = "NEUTRAL"
    if not with_parts:
        return bias, score
    parts = {
        "base": base_score,
        "htf": htf_score,
        "weekly": weekly_score,
        "c2_closed": c2_closed_score,
        "pd_major": pd_major_score,
        "pd_monthly": pd_monthly_score,
        "pd_struct": pd_struct_score,
        "wick": wick_score,
        "ifvg": ifvg_score,
        "reclaim": reclaim_score,
        "smt": 0,          # scanner._apply_smt_bonus doldurur
        "raw": raw,        # tavan uygulanmadan onceki toplam
        "score": score,    # tavanli (motorun kullandigi)
    }
    rev_range = float(rev["high"] - rev["low"])
    features = {
        # Kalemlerin arkasindaki ham olcumler: esikleri sonradan veriyle sorgulamak icin.
        "wick_frac": round(_purge_wick_ratio(crt_range, rev, direction), 4),
        "c2_body_frac": round(abs(float(rev["close"]) - float(rev["open"])) / rev_range, 4)
        if rev_range > 0 else None,
        "reclaim_pct": round(reclaim, 2) if reclaim is not None else None,
        "ifvg_gap_frac": ifvg_zone.gap_frac if ifvg_zone is not None else None,
        "c2_range_frac": round(rev_range / crt_range, 4),
        # "0 puan" ile "bakilamadi"yi ayirmak icin uygunluk bayraklari (bkz. score_parts).
        "ifvg_checked": int(ifvg_checked),
        "bias_checked": int(df_1d is not None and not df_1d.empty),
    }
    return bias, score, parts, features


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


def _is_bear_body(row: pd.Series) -> bool:
    return _candle_state(row) == "bearish"


def _closes_up(row: pd.Series) -> bool:
    """CISD blogu icin: mum yukari kapatti mi (govde boyutundan bagimsiz)."""
    return float(row["close"]) > float(row["open"])


def _closes_down(row: pd.Series) -> bool:
    """CISD blogu icin: mum asagi kapatti mi (govde boyutundan bagimsiz)."""
    return float(row["close"]) < float(row["open"])


def _is_bull_body(row: pd.Series) -> bool:
    return _candle_state(row) == "bullish"


SMT_LOOKBACK_15M = 40  # purge_time yoksa fallback: son ~10 saatlik 15m penceresi
SMT_PIVOT_K = 1        # swing pivot icin sag/sol mum sayisi (1 => klasik 3-mum pivot)


def _swing_indices(vals, k: int, kind: str) -> list[int]:
    """Yerel tepe ('high') / dip ('low') pivot indekslerini (zaman sirali) dondur.

    Pivot: bir mumun high'i (veya low'u) her iki yandaki k mumdan KESIN daha uc.
    """
    idxs: list[int] = []
    n = len(vals)
    for i in range(k, n - k):
        left = vals[i - k:i]
        right = vals[i + 1:i + k + 1]
        if kind == "high":
            if vals[i] > left.max() and vals[i] > right.max():
                idxs.append(i)
        else:
            if vals[i] < left.min() and vals[i] < right.min():
                idxs.append(i)
    return idxs


def _pivot_divergence(a: pd.DataFrame, b: pd.DataFrame, kind: str) -> bool:
    """a'nin SON swing pivotu, onceki EN UC pivotuna gore a ile b ZIT gidiyorsa divergence.

    a ve b ayni indekse sahip olmali (ayni zaman izgarasi). Pivotlar a uzerinde
    bulunur; b ayni iki zaman noktasinda karsilastirilir.

    Referans (i1) olarak "onceki pivot" DEGIL, onceki pivotlar icindeki EN UC olan
    secilir: tepe icin en YUKSEK onceki tepe, dip icin en DUSUK onceki dip. Boylece
    supurulmeye calisilan ASIL likidite seviyesi baz alinir; minor ara pivotlar
    (dususte olusan kucuk lower-high'lar gibi) SMT'yi maskelemez.
    """
    col = "high" if kind == "high" else "low"
    piv = _swing_indices(a[col].values, SMT_PIVOT_K, kind)
    if len(piv) < 2:
        return False
    i2 = piv[-1]              # en son swing (yeni tepe/dip)
    earlier = piv[:-1]
    if kind == "high":
        i1 = max(earlier, key=lambda k: a["high"].iloc[k])  # onceki EN YUKSEK tepe
        a_dir = a["high"].iloc[i2] > a["high"].iloc[i1]  # a higher high?
        b_dir = b["high"].iloc[i2] > b["high"].iloc[i1]  # b higher high?
    else:
        i1 = min(earlier, key=lambda k: a["low"].iloc[k])   # onceki EN DUSUK dip
        a_dir = a["low"].iloc[i2] < a["low"].iloc[i1]    # a lower low?
        b_dir = b["low"].iloc[i2] < b["low"].iloc[i1]    # b lower low?
    return bool(a_dir != b_dir)


def check_smt_divergence(
    df_main: Optional[pd.DataFrame],
    df_corr: Optional[pd.DataFrame],
    direction: str,
    purge_time: Optional[datetime] = None,
    window_hours: float = 4.0,
) -> bool:
    """Korele parite ile 15M SMT (Smart Money Technique) divergence tespiti.

    Pozitif korele iki enstruman normalde birlikte tepe/dip yapar. Iki sembolun
    SON IKI swing (pivot) noktasi arasinda biri yon degistirirken digeri
    degistirmiyorsa divergence vardir (kurumsal manipulasyon izi):

      - SHORT (HIGH purge): biri son iki tepesinde HIGHER HIGH yaparken digeri
        LOWER HIGH yapiyorsa -> bearish SMT.
      - LONG (LOW purge):   biri son iki dibinde LOWER LOW yaparken digeri
        HIGHER LOW yapiyorsa -> bullish SMT.

    SIMETRIK: divergence her iki grafikte de gorunur bir olaydir. Pivotlar HEM
    ana sembol HEM korele parite uzerinde ayri ayri bulunur; herhangi birinin son
    iki pivotunda ayrisma varsa SMT kabul edilir. Boylece hangi sembol 'ana'
    olursa olsun sonuc ayni cikar (biri 10:00'da tepe pivotu yaparken digeri o an
    dususte olabilir; bu durumda tepe yapan sembolun pivotlari divergence'i yakalar).

    Karsilastirma "yari-yari max/min" yerine ESLESEN swing noktalarinda yapilir
    (yoksa ikinci swing'in daha dusuk/yuksek oldugu goz ardi edilirdi).

    Pencere purge 4H mumuna cipalanir: [purge_time-4h, purge_time+4h]. Alt sinir
    purge oncesi CRT 4H mumunu da kapsar ki supurulen referans tepe/dip pencereye
    girsin; ust sinir purge 4H mumunun sonuna kadar uzanir (divergence tipik olarak
    purge/manipulasyon mumu icinde olusur).
    """
    if df_main is None or df_corr is None or df_main.empty or df_corr.empty:
        return False

    m = df_main.sort_index()
    c = df_corr.sort_index()
    common = m.index.intersection(c.index)
    if len(common) < 5:
        return False
    m = m.loc[common]
    c = c.loc[common]

    if purge_time is not None:
        pt = pd.Timestamp(purge_time)
        pt = pt.tz_localize("UTC") if pt.tzinfo is None else pt.tz_convert("UTC")
        lo = pt - pd.Timedelta(hours=window_hours)
        hi = pt + pd.Timedelta(hours=window_hours)
        mask = (m.index >= lo) & (m.index <= hi)
        m = m[mask]
        c = c[mask]
    else:
        m = m.tail(SMT_LOOKBACK_15M)
        c = c.tail(SMT_LOOKBACK_15M)

    if len(m) < 5:
        return False

    kind = "high" if direction == "SHORT" else "low"
    # Simetri: pivotlari hem ana sembolde hem korele paritede ara; herhangi
    # birinin son iki pivotunda ayrisma varsa SMT var kabul et.
    return _pivot_divergence(m, c, kind) or _pivot_divergence(c, m, kind)


# ──────────────────── PD Array (Price Delivery) tespiti ────────────────────
PD_ARRAY_LOOKBACK_4H = 40   # FVG / OB icin geriye bakilacak 4H mum sayisi
EQ_TOLERANCE = 0.0006       # EQH/EQL "esit" toleransi (seviyeye orani, ~6bps)


def _as_of_ts(df: pd.DataFrame) -> pd.Timestamp:
    """Slice'in son bari = as-of (utcnow yok; replay ile canli ayni)."""
    ts = pd.Timestamp(df.sort_index().index[-1])
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _bar_date(ts) -> object:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    return t.date()


def _previous_day_levels(
    df_1d: Optional[pd.DataFrame], now: pd.Timestamp | None = None,
) -> tuple[Optional[float], Optional[float]]:
    """Onceki KAPANAN gunun (PDH/PDL) high/low'u = serinin son KAPANMIS 1D bari.

    Eskiden "dun" serinin SON SATIRINDAN turetiliyordu (`_as_of_ts` + takvim gunu
    karsilastirmasi) ve son satirin bugunun olusan bari oldugu varsayiliyordu. Altcoinlerin
    1D'si yalnizca bakim dongusunde (1800 sn) yenilendigi icin 00:00 UTC'den sonra 30 dk'ya
    varan bir pencerede son satir hala DUNDU; kod onu "bugun" sanip PDH/PDL'yi EVVELSI GUNE
    kaydiriyordu. 00:00 UTC ayni zamanda bir 4H kapanisi oldugundan gunun alti tam-evren
    taramasindan biri bu bayat pencereye denk geliyordu. Olculen etkiye gore hata cogunlukla
    HAK EDILEN puani yedirtiyordu (pencere ici `pd_major` orani pencere disinin cok altinda;
    oranlar IZLEME.md -> "Gece yarisi PDH/PDL kaymasi duzeltildi"); skor tam 7 ise
    `_score7_strict_ok` bos olmayan `pd_array` aradigi icin setup busbutun elenebiliyordu.

    Cozum: "dun"u takvimden tahmin etmek yerine olusan gunu `_drop_forming_daily` ile dusup
    kalan son bari almak. Store'da bugunun bari VARSA da YOKSA da sonuc ayni (degismezlik).
    Ayni primitif 1D bias'ta da kullaniliyor (11.09) ve seans kovalarinda dogru calisir --
    takvim gunu karsilastirmasi NY 17:00 kovasinda yaniltici olurdu.

    `now` yalnizca test/replay icin; verilmezse duvar saati (bias tarafiyla ayni kural).
    """
    if df_1d is None or df_1d.empty or len(df_1d) < 2:
        return None, None
    closed = _drop_forming_daily(df_1d.sort_index(), now)
    if closed is None or closed.empty:
        return None, None
    last = closed.iloc[-1]
    return float(last["high"]), float(last["low"])


def _pd_as_of(d: pd.DataFrame, now: pd.Timestamp | None) -> pd.Timestamp:
    """PD seviyeleri icin "simdi": verilmisse `now`, yoksa duvar saati; en az son bar.

    Son bara guvenmek bayat store'da (bkz. `_previous_day_levels`) hafta/ay sinirinda ayni
    bir-gun-kaymasini uretiyordu. Gecmis dilimlerde (replay/test) `now` verilir; verilmezse
    son bardan geri gitmemek icin ikisinin buyugu alinir.
    """
    last = _as_of_ts(d)
    cur = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if cur.tzinfo is None:
        cur = cur.tz_localize("UTC")
    return max(last, cur)


def _previous_week_levels(
    df_1d: Optional[pd.DataFrame], now: pd.Timestamp | None = None,
) -> tuple[Optional[float], Optional[float]]:
    """Onceki KAPANAN haftanin (PWH/PWL) high/low'u.

    1D verisini ISO hafta bazinda gruplar, icinde bulunulan haftayi disarida birakir.
    """
    if df_1d is None or df_1d.empty or len(df_1d) < 7:
        return None, None
    d = df_1d.sort_index()
    shift = _trading_day_shift(d.index)  # seans kovalari islem haftasina
    weeks: dict[tuple, tuple[float, float]] = {}
    order: list[tuple] = []
    for ts, row in d.iterrows():
        iso = (pd.Timestamp(ts) + shift).isocalendar()
        key = (int(iso[0]), int(iso[1]))  # (iso_year, iso_week)
        hi = float(row["high"]); lo = float(row["low"])
        if key not in weeks:
            weeks[key] = (hi, lo)
            order.append(key)
        else:
            ph, pl = weeks[key]
            weeks[key] = (max(ph, hi), min(pl, lo))
    as_iso = (_pd_as_of(d, now) + shift).isocalendar()
    cur_key = (int(as_iso[0]), int(as_iso[1]))
    completed = [k for k in order if k != cur_key]
    if not completed:
        return None, None
    last_wk = completed[-1]
    return weeks[last_wk]


def _previous_month_levels(
    df_1d: Optional[pd.DataFrame], now: pd.Timestamp | None = None,
) -> tuple[Optional[float], Optional[float]]:
    """Onceki KAPANAN takvim ayinin (PMH/PML) high/low'u."""
    if df_1d is None or df_1d.empty or len(df_1d) < 20:
        return None, None
    d = df_1d.sort_index()
    shift = _trading_day_shift(d.index)  # seans kovalari islem gunune
    months: dict[tuple[int, int], tuple[float, float]] = {}
    order: list[tuple[int, int]] = []
    for ts, row in d.iterrows():
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC")
        t = t + shift
        key = (int(t.year), int(t.month))
        hi = float(row["high"]); lo = float(row["low"])
        if key not in months:
            months[key] = (hi, lo)
            order.append(key)
        else:
            ph, pl = months[key]
            months[key] = (max(ph, hi), min(pl, lo))
    as_of = _pd_as_of(d, now) + shift
    cur_key = (int(as_of.year), int(as_of.month))
    completed = [k for k in order if k != cur_key]
    if not completed:
        return None, None
    last_m = completed[-1]
    return months[last_m]


def detect_pd_arrays(
    df_4h: pd.DataFrame,
    df_1d: Optional[pd.DataFrame],
    crt_idx: int,
    direction: str,
    purge_idx: Optional[int] = None,
    now: pd.Timestamp | None = None,
) -> list[str]:
    """Purge yapan C2 (4H) mumunun bir PD array'e dokunup dokunmadigini tespit et.

    "Dokunma": C2 mumunun [low, high] araligi ilgili seviye/bolge ile kesisiyorsa
    yeterli (kullanicinin istegi). Sadece purge YONUNDEKI (likidite tarafindaki)
    array'ler kontrol edilir:
      - SHORT (HIGH purge): PDH, PWH, bearish FVG, bearish OB, EQH
      - LONG  (LOW  purge): PDL, PWL, bullish FVG, bullish OB, EQL

    FVG/OB yalnizca purge'den ONCE olusmus (onceden var olan imbalans/blok)
    olarak aranir. Doner: dokunulan array etiketleri (or. ['PDH', 'FVG']).

    purge_idx verilmezse crt_idx+1 varsayilir (purge, C1'den 1-2 mum sonra olabilir).
    """
    labels: list[str] = []
    n = len(df_4h)
    if purge_idx is None:
        purge_idx = crt_idx + 1
    if purge_idx >= n:
        return labels

    c2 = df_4h.iloc[purge_idx]
    c2_low = float(c2["low"])
    c2_high = float(c2["high"])
    is_short = direction == "SHORT"

    def touches_level(level: float) -> bool:
        return c2_low <= level <= c2_high

    def touches_zone(z_lo: float, z_hi: float) -> bool:
        return not (c2_high < z_lo or c2_low > z_hi)

    # 1) PDH/PDL & PWH/PWL (1D turevli — bias verisiyle ayni kaynak)
    pdh, pdl = _previous_day_levels(df_1d, now)
    pwh, pwl = _previous_week_levels(df_1d, now)
    pmh, pml = _previous_month_levels(df_1d, now)
    if is_short:
        if pdh is not None and touches_level(pdh):
            labels.append("PDH")
        if pwh is not None and touches_level(pwh):
            labels.append("PWH")
        if pmh is not None and touches_level(pmh):
            labels.append("PMH")
    else:
        if pdl is not None and touches_level(pdl):
            labels.append("PDL")
        if pwl is not None and touches_level(pwl):
            labels.append("PWL")
        if pml is not None and touches_level(pml):
            labels.append("PML")

    lo = max(1, purge_idx - PD_ARRAY_LOOKBACK_4H)

    # 2) FVG (4H, purge oncesi olusmus 3-mum imbalansi)
    for i in range(lo, purge_idx - 1):
        a = df_4h.iloc[i - 1]
        b = df_4h.iloc[i + 1]
        if is_short:
            # bearish FVG (direnc): a.low > b.high => bosluk [b.high, a.low]
            if float(a["low"]) > float(b["high"]) and touches_zone(float(b["high"]), float(a["low"])):
                labels.append("FVG")
                break
        else:
            # bullish FVG (destek): a.high < b.low => bosluk [a.high, b.low]
            if float(a["high"]) < float(b["low"]) and touches_zone(float(a["high"]), float(b["low"])):
                labels.append("FVG")
                break

    # 3) Order Block (4H): son zit renkli mum + ardindan displacement
    for i in range(lo, purge_idx - 1):
        cur = df_4h.iloc[i]
        nxt = df_4h.iloc[i + 1]
        o = float(cur["open"]); c = float(cur["close"])
        h = float(cur["high"]); l = float(cur["low"])
        nc = float(nxt["close"])
        if is_short:
            # bearish OB (arz): yukari mum + ardindan asagi displacement (kapanis low altina)
            if c > o and nc < l and touches_zone(l, h):
                labels.append("OB")
                break
        else:
            # bullish OB (talep): asagi mum + ardindan yukari displacement (kapanis high ustune)
            if c < o and nc > h and touches_zone(l, h):
                labels.append("OB")
                break

    # 4) EQH/EQL (esit tepe/dip likiditesi)
    seg = df_4h.iloc[lo:purge_idx]
    if len(seg) >= 3:
        col = "high" if is_short else "low"
        vals = seg[col].values
        piv = _swing_indices(vals, 1, "high" if is_short else "low")
        levels = [float(vals[i]) for i in piv]
        found_eq = False
        for a_i in range(len(levels)):
            for b_i in range(a_i + 1, len(levels)):
                base = max(abs(levels[a_i]), 1e-9)
                if abs(levels[a_i] - levels[b_i]) / base <= EQ_TOLERANCE:
                    lvl = (levels[a_i] + levels[b_i]) / 2.0
                    if touches_level(lvl):
                        labels.append("EQH" if is_short else "EQL")
                        found_eq = True
                        break
            if found_eq:
                break

    return labels


@dataclass
class IFVGZone:
    """LTF inverted FVG: zone + %50 (mid) + inversion zamani."""
    low: float
    high: float
    mid: float
    inverted_time: datetime
    kind: str  # "bull" | "bear" | "bpr" (BPR'de low/high iki FVG'nin KESISIMIDIR)
    # Bosluk / son 20 LTF mumunun ort. range'i. Yalniz OLCUM (MIN_IFVG_GAP_RANGE_FRAC
    # esiginin dogru yerde olup olmadigini sonradan veriyle sorabilmek icin).
    gap_frac: Optional[float] = None
    # Yalniz kind="bpr" icin: kesisimi olusturan AYNI YONLU FVG'nin KENDI sinirlari
    # (kesisim degil). Kesisim bu FVG'nin icinde kaldigi icin bacagin girise yakin
    # kenari daima kesisiminkinden once dokunulur -- "BPR'ye degil bacaga limit
    # koysaydik" sorusu bu iki sayidan olculur. Motor KULLANMAZ (22.09).
    leg_low: Optional[float] = None
    leg_high: Optional[float] = None

    def entry_for(self, direction: str) -> float:
        """Girise EN YAKIN kenar: LONG'da ust, SHORT'ta alt.

        Retest'te fiyatin bolgeye ilk dokundugu nokta budur; mid'e gore fill
        olasiligi yuksek, karsiliginda entry SL'ye biraz uzak (RR bir miktar
        dusuk). mid alani hala zone bilgisi olarak duruyor.
        """
        return self.high if direction == "LONG" else self.low


def _drop_forming_bar(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    if len(df) < 2:
        return df
    return df.sort_index().iloc[:-1]


def _ifvg_mid_inside_crt(
    z_lo: float,
    z_hi: float,
    crt_low: float | None,
    crt_high: float | None,
) -> bool:
    """IFVG %50 (mid) CRT C1 araliginda mi?"""
    if crt_low is None or crt_high is None:
        return False
    lo, hi = float(crt_low), float(crt_high)
    if hi < lo:
        lo, hi = hi, lo
    mid = (float(z_lo) + float(z_hi)) / 2.0
    return lo <= mid <= hi


def detect_displacement_fvg(
    df_ltf: Optional[pd.DataFrame],
    direction: str,
    *,
    confirm_time,
    since_time=None,
    lookback: int = 12,
    bars_after: int = 1,
) -> Optional[IFVGZone]:
    """Karakter degisimini yapan hamlenin ARKASINDA biraktigi FVG (yalniz OLCUM).

    Kullanici onerisi (18.09), Setup Journal `entries` karsilastirmasinin 3. modeli:
    CISD/MSS kirildiktan sonra fiyat genelde geri cekilir; kirilimi yapan ivmeli bacak
    bir imbalans (FVG) birakir ve giris oradan alinabilir. Bu seviye tipik olarak CISD
    seviyesinden DAHA SIG'dir (LONG'da daha yukarida): dolum olasiligi yuksek, RR dusuk.
    Olcum tam bu takasi tartacak.

    LONG icin bogа FVG: high[i-1] < low[i+1] -> bosluk [high[i-1], low[i+1]].
    Geri cekilmede fiyatin ILK dokundugu kenar ust siniridir (low[i+1]) -> zone.high,
    yani `entry_for("LONG")` yine "girise en yakin kenar"i verir (IFVG ile ayni anlam).
    SHORT simetrik: low[i-1] > high[i+1] -> bosluk [high[i+1], low[i-1]].

    Pencere: `since_time` (verilirse purge ekstremi) ile onay mumu + `bars_after` arasi;
    birden fazla aday varsa kirilima EN YAKIN (en son) olan secilir. Gurultu filtresi
    UYGULANMAZ -- `gap_frac` kaydedilir, elemeyi rapor yapar (esik sorusu da olculsun).

    MOTOR KARARI BUNU KULLANMAZ; yalnizca Journal'a aday entry olarak yazilir.
    """
    if df_ltf is None or df_ltf.empty or confirm_time is None:
        return None
    work = _drop_forming_bar(df_ltf.sort_index())
    if work is None or len(work) < 3:
        return None
    work = work.copy()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    n = len(work)
    confirm_ts = _as_utc_ts(confirm_time)
    idx = [t for t in work.index if t <= confirm_ts]
    if not idx:
        return None
    confirm_i = len(idx) - 1
    lo_i = 1
    if since_time is not None:
        since_ts = _as_utc_ts(since_time)
        for k, t in enumerate(work.index):
            if t >= since_ts:
                lo_i = max(1, k)
                break
    lo_i = max(lo_i, confirm_i - lookback, 1)
    hi_i = min(n - 2, confirm_i + max(0, bars_after))
    if hi_i < lo_i:
        return None

    _avg_range = float((work["high"] - work["low"]).tail(_IFVG_RANGE_LOOKBACK).mean())
    bull = direction == "LONG"
    best: Optional[tuple[float, float, object]] = None
    for i in range(lo_i, hi_i + 1):
        a = work.iloc[i - 1]
        b = work.iloc[i + 1]
        if bull:
            z_lo, z_hi = float(a["high"]), float(b["low"])
        else:
            z_lo, z_hi = float(b["high"]), float(a["low"])
        if z_hi <= z_lo:
            continue                      # bosluk yok
        best = (z_lo, z_hi, work.index[i])  # en son aday kazanir (kirilima en yakin)
    if best is None:
        return None
    z_lo, z_hi, ts = best
    return IFVGZone(
        low=round(z_lo, 8),
        high=round(z_hi, 8),
        mid=round((z_lo + z_hi) / 2, 8),
        inverted_time=ts.to_pydatetime(),
        kind="bull" if bull else "bear",
        gap_frac=round((z_hi - z_lo) / _avg_range, 4) if _avg_range > 0 else None,
    )



# CISD kirilim FVG'si (22.09, kullanici istegi) -- Setup Journal `entries` karsilastirmasinin
# 5. modeli. Soru: "CISD onayindan sonra fiyat, onayi yapan hamlenin arkada biraktigi FVG'ye
# retest veriyor mu; giris oradan daha mi iyi?" IFVG'ye de MSS'e de BAKMAZ: yalnizca CISD
# onayi + o hamlenin biraktigi bosluk.
#
# `detect_displacement_fvg`ten farki PENCERE ve ZAMANLAMA:
#   - O fonksiyon purge -> onay arasini tarar ve seviyeler DONDUGU AN cagrilir; onay mumu o an
#     serinin son kapali mumu oldugu icin, orta mumu ONAY MUMU OLAN FVG'yi yapisal olarak hic
#     goremez (3 mumluk desen bir mum daha ister). Olculdu: 1519 setup'in yalniz 283'unde
#     dfvg seviyesi var ve 142'si CISD'den daha DERIN -- yani bulunan sey kirilim bacaginin
#     degil, purge sonrasi eski bir boslugun FVG'si.
#   - Bu fonksiyon onay mumu MERKEZLI calisir (orta mum: onay-1 .. onay+bars_after) ve setup
#     journal tarafindan mum mum, ARTIMLI cagrilir; FVG onaydan sonra tamamlansa da yakalanir.
#
# Girdi DataFrame degil (ts, high, low) uclusu: journal'in kapanan mumlardan elinde yalnizca
# bunlar var (`track_bar`) ve FVG icin fazlasi gerekmiyor. Boylece tespit tek yerde yasar --
# canli artimli yol ile dogrulama betigi ayni kodu cagirir.
def cisd_fvg_candidates(
    bars,
    direction: str,
    *,
    confirm_ts,
    bars_before: int = 1,
    bars_after: int = 2,
) -> list[tuple]:
    """Onay hamlesinin biraktigi FVG adaylari: [(lo, hi, olustugu_mum_ts), ...] olusum sirasiyla.

    LONG (boga FVG): low[i+1] > high[i-1] -> bosluk [high[i-1], low[i+1]].
    SHORT simetrik: high[i+1] < low[i-1] -> bosluk [high[i+1], low[i-1]].
    Bosluk `bars[i+1]` kapaninca BILINIR; donen ts o mumdur (izleme SONRAKI mumdan baslar --
    yakin kenar tanim geregi o mumun low'u/high'i oldugu icin "retest" sorusu kendiliginden
    "evet" cikardi).

    Adaylar artan i ile taranir; cagiran ILKINI (kirilim origin'ine en yakin, derin = RR yuksek)
    kullanir, sayilari ise esik sorusu icin kaydedilir. Gurultu filtresi UYGULANMAZ.
    """
    out: list[tuple] = []
    try:
        rows = [(t, float(h), float(l)) for t, h, l in bars]
    except Exception:
        return out
    if len(rows) < 3 or confirm_ts is None:
        return out
    k = None
    for i, (t, _, _) in enumerate(rows):
        if t is not None and t <= confirm_ts:
            k = i
    if k is None:
        return out
    long = direction == "LONG"
    lo_i = max(1, k - max(0, bars_before))
    hi_i = min(len(rows) - 2, k + max(0, bars_after))
    for i in range(lo_i, hi_i + 1):
        _, a_high, a_low = rows[i - 1]
        _, b_high, b_low = rows[i + 1]
        if long:
            z_lo, z_hi = a_high, b_low
        else:
            z_lo, z_hi = b_high, a_low
        if z_hi <= z_lo:
            continue                      # bosluk yok
        out.append((round(z_lo, 8), round(z_hi, 8), rows[i + 1][0]))
    return out


def detect_ltf_ifvg(
    df_ltf: Optional[pd.DataFrame],
    direction: str,
    purge_time,
    *,
    crt_low: float | None = None,
    crt_high: float | None = None,
    crt_bar_time=None,
    c2_hours: float | None = None,
) -> Optional[IFVGZone]:
    """CRT C1 icindeki, invert edilmis, henuz mitigate edilmemis LTF IFVG.

    LONG: bear FVG invert (close zone ustu) + bolge acik.
    SHORT: bull FVG invert (close zone alti) + bolge acik.
    FVG, CRT mumunun acilisindan itibaren aranir. Mid'in C1 high-low icinde
    olmasi sarti 25.09'da kalkti (REQUIRE_IFVG_MID_IN_C1): purge fitilindeki FVG
    de sayilir. Birden fazla aday varsa en son (guncel) acik bolge secilir.
    Inversion purge sonrasi. C2 HTF mumu icindeki LTF ekstrem mitigasyon
    sayilmaz (purge fitili IFVG'yi iptal etmez).

    Bosluk (gap = z_hi - z_lo) son 20 LTF mumunun ort. range'inin
    MIN_IFVG_GAP_RANGE_FRAC katindan kucukse aday elenir (gurultu filtresi).
    """
    if df_ltf is None or df_ltf.empty or purge_time is None:
        return None
    if crt_low is None or crt_high is None:
        return None
    work = _drop_forming_bar(df_ltf.sort_index())
    if work is None or len(work) < 3:
        return None
    if work.index.tz is None:
        work = work.copy()
        work.index = work.index.tz_localize("UTC")
    else:
        work = work.copy()
        work.index = work.index.tz_convert("UTC")
    purge_ts = _as_utc_ts(purge_time)
    crt_ts = _as_utc_ts(crt_bar_time) if crt_bar_time is not None else None
    want_bear = direction == "LONG"
    n = len(work)

    # Gurultu bosluklarini elemek icin asgari gap esigi (son N LTF mum range ort.).
    _avg_range = float(
        (work["high"] - work["low"]).tail(_IFVG_RANGE_LOOKBACK).mean()
    )
    min_gap = MIN_IFVG_GAP_RANGE_FRAC * _avg_range if _avg_range > 0 else 0.0
    start_i = 1
    if crt_ts is not None:
        # Sol mum CRT baslangicindan once olmasin (3'lu FVG).
        start_i = max(1, int(work.index.searchsorted(crt_ts, side="left")) + 1)
    picked: Optional[IFVGZone] = None
    for i in range(start_i, n - 1):
        a = work.iloc[i - 1]
        b = work.iloc[i + 1]
        if want_bear:
            if float(a["low"]) <= float(b["high"]):
                continue
            z_lo = float(b["high"])
            z_hi = float(a["low"])
            kind = "bear"
        else:
            if float(a["high"]) >= float(b["low"]):
                continue
            z_lo = float(a["high"])
            z_hi = float(b["low"])
            kind = "bull"
        if z_hi <= z_lo:
            continue
        if min_gap > 0 and (z_hi - z_lo) < min_gap:
            continue  # gurultu: gercek displacement yok
        if REQUIRE_IFVG_MID_IN_C1 and not _ifvg_mid_inside_crt(z_lo, z_hi, crt_low, crt_high):
            continue
        invert_from = max(i + 2, int(work.index.searchsorted(purge_ts, side="right")))
        inverted_k: Optional[int] = None
        for k in range(invert_from, n):
            c = float(work.iloc[k]["close"])
            if want_bear and c > z_hi:
                inverted_k = k
                break
            if (not want_bear) and c < z_lo:
                inverted_k = k
                break
        if inverted_k is None:
            continue
        c2_end = None
        if c2_hours is not None and c2_hours > 0 and purge_ts is not None:
            c2_end = purge_ts + pd.Timedelta(hours=float(c2_hours))
        mitigated = False
        for k in range(inverted_k + 1, n):
            bar_ts = work.index[k]
            if c2_end is not None and bar_ts < c2_end:
                continue
            if want_bear and float(work.iloc[k]["low"]) < z_lo:
                mitigated = True
                break
            if (not want_bear) and float(work.iloc[k]["high"]) > z_hi:
                mitigated = True
                break
        if mitigated:
            continue
        inv_ts = work.index[inverted_k]
        picked = IFVGZone(
            low=round(z_lo, 8),
            high=round(z_hi, 8),
            mid=round((z_lo + z_hi) / 2.0, 8),
            inverted_time=inv_ts.to_pydatetime(),
            kind=kind,
            gap_frac=round((z_hi - z_lo) / _avg_range, 4) if _avg_range > 0 else None,
        )
    return picked


def detect_ltf_bpr(
    df_ltf: Optional[pd.DataFrame],
    direction: str,
    zone: Optional[IFVGZone],
    *,
    bars_after: int = _BPR_BARS_AFTER,
) -> Optional[IFVGZone]:
    """BPR: invert olmus IFVG ile, inversiyonu yapan hamlenin biraktigi AYNI YONLU
    FVG'nin KESISIMI. Yoksa None.

    IFVG'de reversal kaniti tek sarta baglidir: fiyat ters yonlu FVG'nin tamamen
    otesinde kapanir. BPR'de buna ikinci bir kanit eklenir -- donus hamlesi kendi
    yonunde bir imbalans birakir ve bu imbalans eski FVG ile ust uste biner. Ortak
    alan hem "invert olmus arz/talep" hem "taze displacement boslugu" oldugu icin
    daha guclu bir konfirmasyon sayilir; entry de oradan alinir.

    LONG: invert olmus BEAR FVG + yeni BOGA FVG (high[i-1] < low[i+1]).
    SHORT: simetrik. Ikinci FVG'nin orta mumu inversiyon mumunun bir oncesinden
    `bars_after` sonrasina kadar aranir (tipik olarak inversiyon mumunun kendisi).

    Donen bolge KESISIMDIR: LONG'da [max(lo), min(hi)] -- yani IFVG'nin ust kenarindan
    daha asagida, daha derin bir entry (RR daha iyi, dolum olasiligi daha dusuk).
    `kind` "bpr" yazilir; `entry_for` aynen calisir (LONG'da ust, SHORT'ta alt kenar).

    Donen bolgede `leg_low`/`leg_high` ikinci FVG'nin KENDI sinirlaridir (kesisim degil).
    Yalniz olcum: kesisim bu araligin icinde kaldigi icin bacagin girise yakin kenari
    daima once dokunulur -- "kesisime degil bacaga limit koysaydik" sorusu oradan olculur.

    Ikinci FVG'ye ayni gurultu filtresi (MIN_IFVG_GAP_RANGE_FRAC) uygulanir; kesisimin
    kendisine asgari boyut sarti YOKTUR (kesisim tanim geregi daha kucuk). IFVG'nin
    mitigasyon kontrolu zaten yapilmistir, kesisim onun icinde kaldigi icin tekrarlanmaz.
    """
    if df_ltf is None or df_ltf.empty or zone is None:
        return None
    work = _drop_forming_bar(df_ltf.sort_index())
    if work is None or len(work) < 3:
        return None
    work = work.copy()
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")
    if zone.inverted_time is None:
        return None
    inv_ts = _as_utc_ts(zone.inverted_time)
    inverted_k = int(work.index.searchsorted(inv_ts, side="left"))
    n = len(work)
    if inverted_k >= n:
        return None
    z_lo, z_hi = float(zone.low), float(zone.high)
    if z_hi <= z_lo:
        return None
    _avg_range = float((work["high"] - work["low"]).tail(_IFVG_RANGE_LOOKBACK).mean())
    min_gap = MIN_IFVG_GAP_RANGE_FRAC * _avg_range if _avg_range > 0 else 0.0
    long = direction == "LONG"
    lo_i = max(1, inverted_k - 1)
    hi_i = min(n - 2, inverted_k + max(0, bars_after))
    picked: Optional[tuple[float, float, object, float, float]] = None
    for i in range(lo_i, hi_i + 1):
        a = work.iloc[i - 1]
        b = work.iloc[i + 1]
        if long:
            f_lo, f_hi = float(a["high"]), float(b["low"])
        else:
            f_lo, f_hi = float(b["high"]), float(a["low"])
        if f_hi <= f_lo:
            continue                                  # ayni yonlu FVG yok
        if min_gap > 0 and (f_hi - f_lo) < min_gap:
            continue                                  # gurultu: gercek displacement yok
        ov_lo, ov_hi = max(z_lo, f_lo), min(z_hi, f_hi)
        if ov_hi <= ov_lo:
            continue                                  # kesismiyor -> BPR degil, sadece IFVG
        # en son aday kazanir (IFVG ile ayni kural); bacagin kendi sinirlari da tasinir
        picked = (ov_lo, ov_hi, work.index[i], f_lo, f_hi)
    if picked is None:
        return None
    ov_lo, ov_hi, ts, leg_lo, leg_hi = picked
    return IFVGZone(
        low=round(ov_lo, 8),
        high=round(ov_hi, 8),
        mid=round((ov_lo + ov_hi) / 2.0, 8),
        inverted_time=ts.to_pydatetime(),
        kind="bpr",
        gap_frac=round((ov_hi - ov_lo) / _avg_range, 4) if _avg_range > 0 else None,
        leg_low=round(leg_lo, 8),
        leg_high=round(leg_hi, 8),
    )


def _build_crt_setup(
    df_4h: pd.DataFrame,
    live_i: int,
    purge_j: int,
    direction: str,
    symbol: str,
    market_type: str,
    htf_bias: str,
    *,
    df_1d: Optional[pd.DataFrame] = None,
    timeframe: str = "4h",
    df_ltf: Optional[pd.DataFrame] = None,
    range_atr: Optional[float] = None,   # yalniz olcum: C1 range / ATR (cagiran zaten hesapliyor)
) -> CRTSetup:
    """C1 (live_i) + C2 (purge_j) cifti icin CRTSetup: skor, PD array, hedef tarafi.

    `df_4h` sirali olmali. detect_crt_setup'in gecerli adaylari ve build_rejected_setup
    (elenen adayin seviyelerini hesaplamak icin) ayni kodu kullanir.
    """
    live_crt = df_4h.iloc[live_i]
    purge_row = df_4h.iloc[purge_j]
    live_range = float(live_crt["high"] - live_crt["low"])
    crt_high = float(live_crt["high"])
    crt_low = float(live_crt["low"])

    # C1'den SONRAKI mumlar (purge/C2 ve forming dahil) — hedef taraf kontrolu icin.
    after = df_4h.iloc[live_i + 1:]
    # Hedef taraf tuketilmis olsa da aday kalir (radar); sinyal acilmaz.
    target_consumed = (
        (direction == "SHORT" and float(after["low"].min()) < crt_low)
        or (direction == "LONG" and float(after["high"].max()) > crt_high)
    )

    # C2'den SONRAKI mumlar (forming dahil) — purge/SL tarafi da alinmis mi?
    # Esitlik dahil: scanner'in _hits_sl'i de degmeyi vurus sayar.
    after_c2 = df_4h.iloc[purge_j + 1:]
    purge_ext = float(purge_row["high"]) if direction == "SHORT" else float(purge_row["low"])
    stop_breached = not after_c2.empty and (
        (direction == "SHORT" and float(after_c2["high"].max()) >= purge_ext)
        or (direction == "LONG" and float(after_c2["low"].min()) <= purge_ext)
    )

    pd_labels = detect_pd_arrays(df_4h, df_1d, live_i, direction, purge_idx=purge_j)
    bias, score, score_parts, score_features = _calc_live_setup_bias(
        df_4h, live_i, direction, htf_bias,
        pd_labels=pd_labels, purge_idx=purge_j, df_1d=df_1d,
        df_ltf=df_ltf, timeframe=timeframe, with_parts=True,
    )
    if range_atr is not None:
        score_features["range_atr"] = round(float(range_atr), 4)
    purge_extreme = float(purge_row["high"]) if direction == "SHORT" else float(purge_row["low"])
    crt_bull = float(live_crt["close"]) > float(live_crt["open"])
    crt_bear = float(live_crt["close"]) < float(live_crt["open"])
    rev_bull = float(purge_row["close"]) > float(purge_row["open"])
    rev_bear = float(purge_row["close"]) < float(purge_row["open"])
    same_color = (crt_bull and rev_bull) or (crt_bear and rev_bear)
    if direction == "LONG":
        inv_level = crt_low + (live_range * CRT_INVALIDATION_FRAC)
    else:
        inv_level = crt_high - (live_range * CRT_INVALIDATION_FRAC)
    return CRTSetup(
        symbol=symbol,
        direction=direction,
        purge_type="HIGH" if direction == "SHORT" else "LOW",
        bias=bias,
        bias_score=score,
        key_level_high=round(crt_high, 8),
        key_level_low=round(crt_low, 8),
        crt_bar_time=live_crt.name.to_pydatetime(),
        purge_time=purge_row.name.to_pydatetime(),
        invalidation_level=round(inv_level, 8),
        purge_extreme=round(purge_extreme, 8),
        last_4h_high=round(float(purge_row["high"]), 8),
        last_4h_low=round(float(purge_row["low"]), 8),
        market_type=market_type,
        pd_array=",".join(pd_labels) if pd_labels else None,
        c2_closed=purge_j < (len(df_4h) - 1),
        color_opposite=not same_color,
        timeframe=timeframe,
        target_consumed=target_consumed,
        stop_breached=stop_breached,
        c2_reclaim=c2_reclaim_pct(live_crt, purge_row, direction),
        score_parts=score_parts,
        score_features=score_features,
    )


def build_rejected_setup(
    df_4h: pd.DataFrame,
    symbol: str,
    market_type: str,
    htf_bias: str,
    crt_bar_time,
    purge_time,
    direction: str,
    *,
    df_1d: Optional[pd.DataFrame] = None,
    timeframe: str = "4h",
    df_ltf: Optional[pd.DataFrame] = None,
) -> Optional[CRTSetup]:
    """detect_crt_setup(rejected=...) kaydindaki C1/C2 icin setup'i yeniden kur.

    Yalniz Setup Journal'in seviye/skor kaydi icindir; motor karari degildir.
    """
    try:
        df = df_4h.sort_index()
        live_i = int(df.index.get_loc(pd.Timestamp(crt_bar_time)))
        purge_j = int(df.index.get_loc(pd.Timestamp(purge_time)))
    except Exception:
        return None
    return _build_crt_setup(
        df, live_i, purge_j, direction, symbol, market_type, htf_bias,
        df_1d=df_1d, timeframe=timeframe, df_ltf=df_ltf,
    )


def detect_crt_setup(
    df_4h: pd.DataFrame,
    symbol: str,
    market_type: str = "crypto",
    htf_bias: str = "NEUTRAL",
    df_1d: Optional[pd.DataFrame] = None,
    timeframe: str = "4h",
    df_ltf: Optional[pd.DataFrame] = None,
    rejected: Optional[list] = None,
) -> Optional[CRTSetup]:
    """4H verisinde CRT pattern tespit et; en buyuk gecerli C1'i sec.

    Aday C1 mumlari: 4H/1D icin len-2 .. len-5, 1H icin len-2 .. len-8.
    Her aday icin purge/C2, C1'den SONRAKI CRT_PURGE_SEARCH (1-3) mum icinde
    aranir. Boylece purge hemen bir sonraki mum olmasa bile (ornegin araya
    kucuk bir "inside bar" girse de) daha BUYUK olan asil range mumu C1
    olarak yakalanabilir.

    Purge/C2: C1'in bir tarafini (SHORT'ta high, LONG'ta low) esik kadar asip ayni
    mumda C1 araligina GERI kapatan ilk mum. Bir mum tarafi asip geri kapatmadan
    (breakout) kaparsa o C1 gecersiz sayilir.

    Purge mum rengi (reversal, A kurali):
      - Sweep + iceri kapanis zorunlu.
      - C2 dogru renk (LONG yesil / SHORT kirmizi) VEYA doji (govde/range <= %10).
      - Guclu yanlis renk (doji degil) C1'i eler -- yalniz C2_COLOR_HARD_FILTER_TFS
        (4H/1H). 1D'de yanlis renk gecer, skorda base 0 alir (25.09).

    BAYATLIK KONTROLU: Purge mumundan ONCE araya giren bir mum, C1'in ilgili
    ekstremini (LONG'ta low, SHORT'ta high) HAM olarak (esik alti kucuk delme
    dahil) zaten delmisse, o taraftaki likidite onceden alinmis demektir; C1
    bayat/gecersiz sayilir ve daha guncel bir C1 tercih edilir.

    HEDEF TARAF GECERLILIGI: C1'in hedef (TP) tarafi, C1'den SONRAKI bir mum
    tarafindan zaten supurulmusse setup gecersizdir (hedef likidite tuketilmis):
      - SHORT (purge HIGH, TP = C1 low): C1'den sonra bir mumun low'u C1 low altina inmisse -> gecersiz.
      - LONG  (purge LOW,  TP = C1 high): C1'den sonra bir mumun high'i C1 high ustune cikmisse -> gecersiz.

    Birden fazla gecerli aday varsa once YASAYAN adaylar gelir -- hedefi tuketilmis
    (`target_consumed`) ya da purge/SL tarafi sonradan alinmis (`stop_breached`) aday, canli
    bir aday varken secilmez (18.09; oncesinde yalniz `target_consumed` boyle davraniyordu ve
    stopu yenmis eski bir C1 genis range'i sayesinde slotu saatlerce tutabiliyordu). Yasayanlar
    arasinda high-low mesafesi (range) EN BUYUK olan secilir (esitlikte hacimce buyuk, sonra
    en guncel). Hicbiri canli degilse yine en iyi olu aday donulur: kapiyi scanner kapatir,
    radar/Journal nedeni gosterir.
    `rejected` (liste verilirse): setup'a cevrilmeden elenen adaylar -- yalniz KAPANMIS C2 ve
    C1 ucu gercekten delinmisse -- {reason, direction, crt_bar_time, purge_time, setup} olarak
    eklenir. reason: c2_wrong_color / c2_breakout / c1_stale / range_atr / sweep_small /
    not_selected (gecerliydi, baska aday secildi; `setup` dolu). Motor karari DEGISMEZ; Setup
    Journal bu adaylarin sonrasini izler (14.09).
    """
    df_4h = df_4h.sort_index()

    if len(df_4h) < 16:
        return None

    n = len(df_4h)
    atr = compute_atr(df_4h)
    # Tum marketlerde ayni CRT purge eşiği kullanilir (crypto ile birebir).
    purge_threshold = PURGE_THRESHOLD_PCT

    # (range, volume, live_i, setup) — en sonda en buyuk range'li aday secilir.
    candidates: list[tuple[float, float, int, CRTSetup]] = []

    c1_back = CRT_C1_LOOKBACK_1H if (timeframe or "").lower() == "1h" else CRT_C1_LOOKBACK
    # 1D'de C2 rengi hard filtre degil (25.09); yanlis renk skorda base 0 alir.
    color_hard = (timeframe or "4h").lower() in C2_COLOR_HARD_FILTER_TFS

    def _reject(reason: str, direction: str, c1_i: int, c2_i: int) -> None:
        # Yalniz KAPANMIS C2: forming mumun rengi/kapanisi fiyatla degisir (gurultu).
        if rejected is None or c2_i >= n - 1:
            return
        rejected.append({
            "reason": reason,
            "direction": direction,
            "crt_bar_time": df_4h.index[c1_i].to_pydatetime(),
            "purge_time": df_4h.index[c2_i].to_pydatetime(),
            "setup": None,
        })

    def _scan_purge(c1_i: int, crt_high: float, crt_low: float):
        """Purge/C2'yi C1'den sonraki 1-3 mum icinde ara (ilk gecerli purge/breakout durdurur).

        Donus (direction, purge_j, red): gecerli purge'de direction ve purge_j dolu. C1 elendiyse
        red = (reason, direction, j): c2_breakout / c2_wrong_color / c1_stale / sweep_small.
        """
        # C1 ekstreminin purge'den ONCE (araya giren bir mumla) HAM olarak delinip
        # delinmedigini izle. Delinmisse o taraftaki likidite zaten alinmis demektir
        # ve C1 bayat/gecersiz sayilir (esik alti kucuk delmeler de dahil).
        pre_low_breach = False
        pre_high_breach = False
        raw_breach = None
        for j in range(c1_i + 1, min(c1_i + 1 + CRT_PURGE_SEARCH, n)):
            c2 = df_4h.iloc[j]
            c2_high = float(c2["high"])
            c2_low = float(c2["low"])
            c2_close = float(c2["close"])
            if c2_high > crt_high * (1 + purge_threshold):  # HIGH supuruldu
                # geri kapatti + (kirmizi veya doji) + high onceden delinmemis -> SHORT
                c2_state = _candle_state(c2)
                color_ok = c2_state in ("bearish", "doji") or not color_hard
                if (
                    c2_close <= crt_high
                    and color_ok
                    and not pre_high_breach
                ):
                    return "SHORT", j, None
                # asti / guclu yanlis renk / bayat -> C1 gecersiz
                if c2_close > crt_high:
                    reason = "c2_breakout"
                elif not color_ok:
                    reason = "c2_wrong_color"
                else:
                    reason = "c1_stale"
                return None, None, (reason, "SHORT", j)
            if c2_low < crt_low * (1 - purge_threshold):    # LOW supuruldu
                # geri kapatti + (yesil veya doji) + low onceden delinmemis -> LONG
                c2_state = _candle_state(c2)
                color_ok = c2_state in ("bullish", "doji") or not color_hard
                if (
                    c2_close >= crt_low
                    and color_ok
                    and not pre_low_breach
                ):
                    return "LONG", j, None
                if c2_close < crt_low:
                    reason = "c2_breakout"
                elif not color_ok:
                    reason = "c2_wrong_color"
                else:
                    reason = "c1_stale"
                return None, None, (reason, "LONG", j)
            # Bu mum esikli purge yapmadi; ama C1 ekstremini HAM olarak delmisse
            # ilgili taraf bayat sayilir (likidite onceden tuketilmis).
            if c2_low < crt_low:
                pre_low_breach = True
                raw_breach = raw_breach or ("LONG", j)
            if c2_high > crt_high:
                pre_high_breach = True
                raw_breach = raw_breach or ("SHORT", j)
        if raw_breach is not None:
            return None, None, ("sweep_small", raw_breach[0], raw_breach[1])
        return None, None, None

    for live_i in range(n - 2, n - c1_back - 1, -1):
        if live_i < 1 or live_i + 1 >= n:
            continue

        live_crt = df_4h.iloc[live_i]
        live_range = float(live_crt["high"] - live_crt["low"])
        live_atr = atr.iloc[live_i]
        if live_atr == 0 or live_range <= 0:
            continue

        crt_high = float(live_crt["high"])
        crt_low = float(live_crt["low"])
        crt_vol = float(live_crt["volume"])

        direction, purge_j, red = _scan_purge(live_i, crt_high, crt_low)

        live_ratio = live_range / live_atr
        if live_ratio < MIN_RANGE_ATR_RATIO or live_ratio > MAX_RANGE_ATR_RATIO:
            # Aralik ATR bandi disinda: yalniz gecerli bir purge olsaydi kaydet.
            if direction is not None:
                _reject("range_atr", direction, live_i, purge_j)
            continue

        if direction is None or purge_j is None:
            if red is not None:
                _reject(red[0], red[1], live_i, red[2])
            continue

        candidates.append((live_range, crt_vol, live_i, _build_crt_setup(
            df_4h, live_i, purge_j, direction, symbol, market_type, htf_bias,
            df_1d=df_1d, timeframe=timeframe, df_ltf=df_ltf, range_atr=live_ratio,
        )))

    if not candidates:
        return None

    # Once hedefi duran aday (eski kural, degismedi); esitse stopu yenmemis olan; sonra en buyuk
    # range, hacim, en guncel. Iki elemeyi AYRI kademede tutmak sart: tek kademeye katlamak,
    # hedefi tuketilmis genis adayin stopu yenmis dar adayin onune gecmesine yol aciyordu.
    best = max(
        candidates,
        key=lambda c: (0 if c[3].target_consumed else 1, 0 if c[3].stop_breached else 1,
                       c[0], c[1], c[2]),
    )
    if rejected is not None:
        for c in candidates:
            if c is not best and c[3].c2_closed:
                rejected.append({
                    "reason": "not_selected",
                    "direction": c[3].direction,
                    "crt_bar_time": c[3].crt_bar_time,
                    "purge_time": c[3].purge_time,
                    "setup": c[3],
                })
    return best[3]


def check_cisd_confirmation(
    df_15m: pd.DataFrame,
    setup: CRTSetup,
    c2_hours: float = 4.0,
) -> Optional[CISDConfirmation]:
    """15M seviyeleri + (varsa) MSS kapanis onayi.

    Seviyeler MSS kirilimindan bagimsiz hesaplanir; cisd_time None olabilir
    (henuz onay yok). Scanner RR/tight_stop gibi filtreleri onaydan once uygular;
    no_cisd yalnizca filtreler gectikten sonra kullanilir.

    Iki aday (tek seviye, birlikte DEGIL): son swing VE dusen/yukselen blogun
    ILK mumu (CISD). Hangisi daha genis stop (dusuk RR) veriyorsa o baz alinir —
    entry, kirilim ve mss_ref_time ayni adaya aittir.

    SL = purge ucu, TP = CRT karsi seviye.
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
        return _check_bullish_cisd(
            recent, setup, min_confirm_time=min_confirm_time, c2_hours=c2_hours,
        )
    else:
        return _check_bearish_cisd(
            recent, setup, min_confirm_time=min_confirm_time, c2_hours=c2_hours,
        )


def _is_swing_high(df: pd.DataFrame, idx: int, span: int = SWING_PIVOT_BARS) -> bool:
    if idx - span < 0 or idx + span >= len(df):
        return False
    level = float(df.iloc[idx]["high"])
    left_max = float(df.iloc[idx - span:idx]["high"].max())
    right_max = float(df.iloc[idx + 1:idx + 1 + span]["high"].max())
    return level > left_max and level > right_max


def _is_swing_low(df: pd.DataFrame, idx: int, span: int = SWING_PIVOT_BARS) -> bool:
    if idx - span < 0 or idx + span >= len(df):
        return False
    level = float(df.iloc[idx]["low"])
    left_min = float(df.iloc[idx - span:idx]["low"].min())
    right_min = float(df.iloc[idx + 1:idx + 1 + span]["low"].min())
    return level < left_min and level < right_min


def _bar_time(df: pd.DataFrame, idx: int) -> datetime:
    return df.index[idx].to_pydatetime()


def _planned_rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(float(entry) - float(sl))
    if risk <= 0:
        return -1.0
    return abs(float(tp) - float(entry)) / risk


def _pick_wider_stop(
    swing_level: float,
    swing_time: datetime,
    cisd_level: float,
    cisd_time: datetime,
    sl: float,
    tp: float,
    direction: str,
) -> Optional[tuple[float, datetime, str]]:
    """Swing vs CISD blogu: gecerli ve daha iyi RR (daha dar stop) olan adayi don.

    Yuksek RR = entry SL'ye yakin. Iki aday da gecerliyse |entry-SL| daha
    kucuk olani sec (RR hala scanner MIN_RR ile elenir).
    LONG: entry SL ile TP arasinda, SL < entry < TP.
    SHORT: TP < entry < SL.
    Esit mesafede swing tercih edilir.

    Donen ucuncu deger KAZANAN modeldir ("mss" | "cisd") - UI'daki Entry Model
    sutunu icin. Eskiden bu bilgi kayboluyordu ve `CISDConfirmation.entry_model`
    dataclass varsayilani olarak her halukarda "cisd" kaliyordu; yani MSS ile
    girilen islemler de "cisd" yaziliyordu.
    """
    def _ok(level: float) -> bool:
        if direction == "LONG":
            return sl < level < tp
        return tp < level < sl

    swing_ok = _ok(swing_level)
    cisd_ok = _ok(cisd_level)
    if not swing_ok and not cisd_ok:
        return None
    if swing_ok and not cisd_ok:
        return float(swing_level), swing_time, "mss"
    if cisd_ok and not swing_ok:
        return float(cisd_level), cisd_time, "cisd"
    if abs(float(cisd_level) - float(sl)) < abs(float(swing_level) - float(sl)):
        return float(cisd_level), cisd_time, "cisd"
    return float(swing_level), swing_time, "mss"


def _price_eq(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12


def _is_equal_plateau_high(
    df: pd.DataFrame, idx: int, span: int = SWING_PIVOT_BARS,
) -> bool:
    """idx esit-high run'unun sag kenari ve dis komsular kati daha alçak."""
    if idx <= 0 or idx + 1 >= len(df):
        return False
    level = float(df.iloc[idx]["high"])
    if _price_eq(float(df.iloc[idx + 1]["high"]), level):
        return False
    left = idx
    while left - 1 >= 0 and _price_eq(float(df.iloc[left - 1]["high"]), level):
        left -= 1
    # En az 3 mumluk plato (2+ esit komsu). 2 mumluk durak fractal'e birakilir.
    if idx - left + 1 < 3:
        return False
    if left - span < 0 or idx + span >= len(df):
        return False
    left_max = float(df.iloc[left - span:left]["high"].max())
    right_max = float(df.iloc[idx + 1:idx + 1 + span]["high"].max())
    return level > left_max and level > right_max


def _is_equal_plateau_low(
    df: pd.DataFrame, idx: int, span: int = SWING_PIVOT_BARS,
) -> bool:
    """idx esit-low run'unun sag kenari ve dis komsular kati daha yuksek."""
    if idx <= 0 or idx + 1 >= len(df):
        return False
    level = float(df.iloc[idx]["low"])
    if _price_eq(float(df.iloc[idx + 1]["low"]), level):
        return False
    left = idx
    while left - 1 >= 0 and _price_eq(float(df.iloc[left - 1]["low"]), level):
        left -= 1
    if idx - left + 1 < 3:
        return False
    if left - span < 0 or idx + span >= len(df):
        return False
    left_min = float(df.iloc[left - span:left]["low"].min())
    right_min = float(df.iloc[idx + 1:idx + 1 + span]["low"].min())
    return level < left_min and level < right_min


def _is_mss_swing_high(df: pd.DataFrame, idx: int) -> bool:
    """MSS swing: klasik 1-bar fractal VEYA esit-tepe platosu."""
    return _is_swing_high(df, idx) or _is_equal_plateau_high(df, idx)


def _is_mss_swing_low(df: pd.DataFrame, idx: int) -> bool:
    """MSS swing: klasik 1-bar fractal VEYA esit-dip platosu."""
    return _is_swing_low(df, idx) or _is_equal_plateau_low(df, idx)


def _last_swing_high_before(
    df: pd.DataFrame, idx: int,
) -> Optional[tuple[float, datetime]]:
    """idx'ten (dip) ONCEKI son swing high: (high, mum_zamani).

    MSS yapisi 1: dip'e giden son dusus baginin fractal / esit-tepe platosu.
    Pivot yoksa fallback [0, idx) max high. CISD blogu (yapi 2) ayri hesaplanir.
    """
    for j in range(idx - 1, -1, -1):
        if _is_mss_swing_high(df, j):
            return float(df.iloc[j]["high"]), _bar_time(df, j)
    if idx > 0:
        j = int(df.iloc[:idx]["high"].values.argmax())
        return float(df.iloc[j]["high"]), _bar_time(df, j)
    return None


def _last_swing_low_before(
    df: pd.DataFrame, idx: int,
) -> Optional[tuple[float, datetime]]:
    """idx'ten (tepe) ONCEKI son swing low: (low, mum_zamani).

    MSS yapisi 1: tepeye giden son yukselis baginin fractal / esit-dip platosu.
    Pivot yoksa fallback [0, idx) min low. CISD blogu (yapi 2) ayri hesaplanir.
    """
    for j in range(idx - 1, -1, -1):
        if _is_mss_swing_low(df, j):
            return float(df.iloc[j]["low"]), _bar_time(df, j)
    if idx > 0:
        j = int(df.iloc[:idx]["low"].values.argmin())
        return float(df.iloc[j]["low"]), _bar_time(df, j)
    return None


def _strong_break_margin(df: pd.DataFrame, idx: int, level: float) -> float:
    start = max(0, idx - CISD_MARGIN_LOOKBACK + 1)
    window = df.iloc[start:idx + 1]
    avg_range = 0.0
    if not window.empty:
        avg_range = float((window["high"] - window["low"]).mean())
    pct_margin = abs(level) * CISD_STRONG_CLOSE_PCT
    range_margin = avg_range * CISD_STRONG_CLOSE_RANGE_MULT
    return max(pct_margin, range_margin, 1e-8)


def _as_utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _c2_extreme_iloc(
    work: pd.DataFrame,
    purge_time,
    direction: str,
    c2_hours: float = 4.0,
) -> Optional[int]:
    """4H purge (C2) mum araligindaki 15M ekstreminin work iloc'u.

    SHORT: C2 icindeki max high mumu; LONG: C2 icindeki min low mumu.
    Boylece C2 sonrasi yeni HH/LL MSS referansini kaydiramaz.
    """
    if purge_time is None or work is None or work.empty:
        return None
    start = _as_utc_ts(purge_time)
    end = start + pd.Timedelta(hours=c2_hours)
    seg = work[(work.index >= start) & (work.index < end)]
    if seg.empty:
        return None
    if direction == "LONG":
        j = int(seg["low"].values.argmin())
    else:
        j = int(seg["high"].values.argmax())
    loc = work.index.get_loc(seg.index[j])
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, (list, tuple)) or getattr(loc, "size", 1) != 1:
        return int(loc[0])
    return int(loc)


def _check_bullish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
    min_confirm_time: Optional[pd.Timestamp] = None,
    c2_hours: float = 4.0,
) -> Optional[CISDConfirmation]:
    """15M bullish onay: swing vs bearish blog (CISD), tek aday (genis stop).

    - Dip = 4H C2 purge araligindaki 15M min low (C2 sonrasi yeni LL sayilmaz).
    - CISD: C2 ekstreminden ONCEKI dusus blogunun ILK acilisi (yukari kapanan mum bitirir).
    - Swing: o dipten ONCEKI son swing high (esit tepe platosu dahil).
    - Entry + kirilim + ref = daha genis stop veren aday.
    """
    work = df.sort_index()
    if len(work) < 3:
        return None

    # Dip: C2 (4H purge) icindeki 15M ekstrem; yoksa eski fallback (purge sonrasi min).
    low_iloc = _c2_extreme_iloc(work, setup.purge_time, "LONG", c2_hours=c2_hours)
    if low_iloc is None:
        if min_confirm_time is not None:
            cand = work[work.index > min_confirm_time]
        else:
            cand = work
        if len(cand) < 1:
            return None
        low_pos_in_cand = int(cand["low"].values.argmin())
        low_iloc = (len(work) - len(cand)) + low_pos_in_cand

    # Dusus blogunu bul. Likiditeyi supuren mum KENDISI temiz bir dusus govdesi
    # ise (genis wick'li doji degil) CISD delivery'sinin basladigi yer odur;
    # bloga dahil edilir (acilisi = CISD seviyesi). Aksi halde (doji / wick
    # rejection) ondan ONCEKI dusus blogu alinir.
    # Blogun basini ters yonde KAPANAN ilk mum bitirir -- govdesi %10'un altinda
    # (doji) olsa bile (25.09, US100 1D: 24.09 10:00 TSI yesil %9.7 govdeli mum
    # blogu bolmedigi icin CISD 30450'ye kaymisti, dogrusu 11:00 acilisi 30264).
    # Ayni renkli doji blogu BOLMEZ (GBPCHF 07.09: 06:00/07:00 TSI yesil dojiler
    # yukselis blogunun icindeydi, ilk mum 04:00).
    run_end = low_iloc if _is_bear_body(work.iloc[low_iloc]) else low_iloc - 1
    while run_end >= 0 and not _is_bear_body(work.iloc[run_end]):
        run_end -= 1
    if run_end < 0:
        return None
    run_start = run_end
    while run_start - 1 >= 0 and not _closes_up(work.iloc[run_start - 1]):
        run_start -= 1

    # Serinin ILK (en yuksek acilisli) mumunun acilisi = CISD seviyesi.
    cisd_level = float(work.iloc[run_start]["open"])

    # Swing adayi: C2 dip ekstreminden ONCEKI son swing high (plato dahil).
    swing = _last_swing_high_before(work, low_iloc)
    if swing is None:
        return None
    mss_level, mss_ref_time = swing

    sl = round(setup.purge_extreme, 8)
    tp = round(setup.key_level_high, 8)
    cisd_ref_time = _bar_time(work, run_start)

    picked = _pick_wider_stop(
        mss_level, mss_ref_time, cisd_level, cisd_ref_time, sl, tp, "LONG",
    )
    if picked is None:
        return None
    break_level, mss_ref_time, entry_model = picked
    entry = round(break_level, 8)

    confirm_time: Optional[datetime] = None
    # Dipten sonra break_level USTUNDE kapatan ilk mum -> onay.
    # Strong-close margin gecici olarak kapali (FX'te onayi geciktirip CRT%60 ile catisiyordu).
    for i in range(low_iloc, len(work)):
        cur = work.iloc[i]
        if min_confirm_time is not None and cur.name <= min_confirm_time:
            continue
        # margin = _strong_break_margin(work, i, break_level)
        # if float(cur["close"]) > break_level + margin:
        if float(cur["close"]) > break_level:
            confirm_time = cur.name.to_pydatetime()
            break

    return CISDConfirmation(
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        invalidation_level=setup.invalidation_level,
        cisd_price=round(break_level, 8),
        cisd_time=confirm_time,
        mss_ref_time=mss_ref_time,
        entry_model=entry_model,
        cisd_level=round(float(cisd_level), 8),
        mss_level=round(float(mss_level), 8),
    )


def _check_bearish_cisd(
    df: pd.DataFrame,
    setup: CRTSetup,
    min_confirm_time: Optional[pd.Timestamp] = None,
    c2_hours: float = 4.0,
) -> Optional[CISDConfirmation]:
    """15M bearish onay: swing vs bullish blog (CISD), tek aday (genis stop).

    - Tepe = 4H C2 purge araligindaki 15M max high (C2 sonrasi yeni HH sayilmaz).
    - CISD: C2 ekstreminden ONCEKI yukselis blogunun ILK acilisi (asagi kapanan mum bitirir).
    - Swing: o tepeden ONCEKI son swing low (esit dip platosu dahil).
    - Entry + kirilim + ref = daha genis stop veren aday.
    """
    work = df.sort_index()
    if len(work) < 3:
        return None

    # Tepe: C2 (4H purge) icindeki 15M ekstrem; yoksa eski fallback.
    high_iloc = _c2_extreme_iloc(work, setup.purge_time, "SHORT", c2_hours=c2_hours)
    if high_iloc is None:
        if min_confirm_time is not None:
            cand = work[work.index > min_confirm_time]
        else:
            cand = work
        if len(cand) < 1:
            return None
        high_pos_in_cand = int(cand["high"].values.argmax())
        high_iloc = (len(work) - len(cand)) + high_pos_in_cand

    # Yukselis blogunu bul. Likiditeyi supuren mum KENDISI temiz bir yukselis
    # govdesi ise (genis wick'li doji degil) CISD delivery'sinin basladigi yer
    # odur; bloga dahil edilir (acilisi = CISD seviyesi). Aksi halde (doji /
    # wick rejection) ondan ONCEKI yukselis blogu alinir.
    # Blogun basini ters yonde KAPANAN ilk mum bitirir (doji olsa bile, 25.09);
    # ayni renkli doji blogu BOLMEZ (GBPCHF 07.09). Bkz. _check_bullish_cisd.
    run_end = high_iloc if _is_bull_body(work.iloc[high_iloc]) else high_iloc - 1
    while run_end >= 0 and not _is_bull_body(work.iloc[run_end]):
        run_end -= 1
    if run_end < 0:
        return None
    run_start = run_end
    while run_start - 1 >= 0 and not _closes_down(work.iloc[run_start - 1]):
        run_start -= 1

    # Serinin ILK (en dusuk acilisli) mumunun acilisi = CISD seviyesi.
    cisd_level = float(work.iloc[run_start]["open"])

    # Swing adayi: C2 tepe ekstreminden ONCEKI son swing low (plato dahil).
    swing = _last_swing_low_before(work, high_iloc)
    if swing is None:
        return None
    mss_level, mss_ref_time = swing

    sl = round(setup.purge_extreme, 8)
    tp = round(setup.key_level_low, 8)
    cisd_ref_time = _bar_time(work, run_start)

    picked = _pick_wider_stop(
        mss_level, mss_ref_time, cisd_level, cisd_ref_time, sl, tp, "SHORT",
    )
    if picked is None:
        return None
    break_level, mss_ref_time, entry_model = picked
    entry = round(break_level, 8)

    confirm_time: Optional[datetime] = None
    # Tepeden sonra break_level ALTINDA kapatan ilk mum -> onay.
    # Strong-close margin gecici olarak kapali (FX'te onayi geciktirip CRT%60 ile catisiyordu).
    for i in range(high_iloc, len(work)):
        cur = work.iloc[i]
        if min_confirm_time is not None and cur.name <= min_confirm_time:
            continue
        # margin = _strong_break_margin(work, i, break_level)
        # if float(cur["close"]) < break_level - margin:
        if float(cur["close"]) < break_level:
            confirm_time = cur.name.to_pydatetime()
            break

    return CISDConfirmation(
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        invalidation_level=setup.invalidation_level,
        cisd_price=round(break_level, 8),
        cisd_time=confirm_time,
        mss_ref_time=mss_ref_time,
        entry_model=entry_model,
        cisd_level=round(float(cisd_level), 8),
        mss_level=round(float(mss_level), 8),
    )


def check_signal_invalidation(
    current_price: float,
    direction: str,
    invalidation_level: float,
    entry_price: float,
) -> bool:
    """Fiyat CRT mumunun %60'ini (purge tarafindan) gecti mi?

    LONG: entry alttan, fiyat %60'in ustune cikarsa → expired
    SHORT: entry ustten, fiyat %60'in altina duserse → expired
    Entry zaten %60'in diger tarafindaysa (edge case) → False.
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
    """Expired sinyal breakeven mi? Fiyat %60'i gecip entry seviyesine geri dondu mu?

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
