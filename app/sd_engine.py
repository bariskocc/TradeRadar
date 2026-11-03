"""Supply / Demand bolge motoru (03.11, kullanici istegi) -- saf hesap, OLCUM MODU.

CRT'den bagimsiz dorduncu strateji (`sd`): 4H'te bolge, 1D bias ile yon, 15m'de dolum/sonuc izleme
(Setup Journal). Sinyal ACMAZ, Telegram'a gitmez; amac once "hangi bolge tipi kazaniyor" sorusunu
Journal'la olcmek, canli sinyale veriyle gecmek.

Bolge tanimi (uc parca):
  leg-in  -> base'e gelen hamle (LEG_IN_BARS mumda >= LEG_IN_MIN_ATR; yoksa yon "?")
  base    -> 1..BASE_MAX_BARS kucuk mum (araligi <= BASE_MAX_RANGE_ATR x ATR)
  leg-out -> base'den sert cikis: ilk mumun govdesi >= LEG_BODY_MIN_ATR x ATR, base disinda kapanis,
             LEG_OUT_BARS icinde proximal'den >= LEG_OUT_MIN_ATR x ATR yol
Demand (cikis yukari): distal = base'in en dusuk low'u, proximal = base govdelerinin en yukseği.
Supply (cikis asagi): distal = base'in en yuksek high'i, proximal = base govdelerinin en dusugu.
Tip: leg-in harfi + B + leg-out harfi -> RBR / DBD (devam), DBR / RBD (donus), ?BR / ?BD (leg-in zayif).

Tazelik: olusumdan sonra proximal'e dokunan her ayri mum kosusu bir "dokunus"; 0 taze, 1 test edilmis,
>= 2 ya da distal'in otesine fitil (SL_BUFFER_ATR) -> olu (islem aranmaz, hedef de olmaz).

Plan: giris proximal, SL distal + SL_BUFFER_ATR x ATR, TP karsi taraftaki ilk canli bolgenin proximal'i,
yoksa fiyatin otesindeki ilk fraktal swing (TARGET_SWING_K). RR < MIN_RR -> `sd_low_rr`.

Her sey durumsuz: `detect_zones` kapanmis 4H serisinden bolgeleri, dokunuslari ve kirilmayi her cagrida
yeniden hesaplar; bolgenin kimligi base'in ilk mumunun acilis zamanidir (Journal anahtari).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.crt_engine import compute_atr

STRATEGY_SD = "sd"
HTF_DELTA = timedelta(hours=4)

ATR_PERIOD = 14
BASE_MAX_BARS = 3
BASE_MAX_RANGE_ATR = 0.6
LEG_OUT_MIN_ATR = 2.0
LEG_OUT_BARS = 3
LEG_BODY_MIN_ATR = 1.0
LEG_IN_BARS = 3
LEG_IN_MIN_ATR = 1.0
SL_BUFFER_ATR = 0.1
MAX_TOUCHES = 1              # bu kadar dokunusa kadar canli; fazlasi olu
LOOKBACK_BARS = 200          # tespit penceresi (kapanmis 4H mumu)
MAX_AGE_BARS = 120           # daha eski base'ler Journal'a yazilmaz (ilk calismada eski bolge seli olmasin)
TARGET_SWING_K = 2           # hedef swing: 5'li fraktal
MIN_RR = 2.0
SWING_LOOKBACK = 30          # bos / sweep icin base'den onceki swing aramasi (mum)
BOS_EXTRA_BARS = 2           # bos: formed mumundan sonra bu kadar mum daha bakilir

# Skor kalemleri (olcum; esik yok -- hangisinin kazandirdigi Journal'dan okunur).
SCORE_KEYS = ("strength", "base", "fvg", "fresh", "htf")
STRENGTH_STRONG_ATR = 3.0    # leg-out >= 3 ATR -> +2 (2..3 ATR +1)
MAX_SCORE = 9


@dataclass
class SDZone:
    side: str                 # demand | supply
    direction: str            # LONG | SHORT
    zone_type: str            # RBR / DBR / DBD / RBD / ?BR / ?BD
    base_start: pd.Timestamp
    base_end: pd.Timestamp
    legout_time: pd.Timestamp
    formed_time: pd.Timestamp  # leg-out hedefine ulasan mum (dokunuslar bundan sonra sayilir)
    proximal: float
    distal: float
    atr: float
    legout_atr: float
    legin_atr: float
    base_bars: int
    base_max_atr: float
    fvg: bool
    formed_iloc: int
    # 03.11 (kullanici sorusu 1-2, olcum -- puan/kapi yok): leg-out base'den onceki son swing'i kapanisla kirdi mi
    # (SMC "yapiyi kiran hamlenin kaynagi"), base/leg-in onceki swing'i supurdu mu (likidite temizligi).
    bos: bool = False
    sweep: bool = False
    touches: int = 0
    last_touch: Optional[pd.Timestamp] = None
    broken_at: Optional[pd.Timestamp] = None
    in_zone: bool = False

    @property
    def dead(self) -> bool:
        return self.broken_at is not None or self.touches > MAX_TOUCHES

    @property
    def fresh(self) -> bool:
        return self.touches == 0

    @property
    def height_atr(self) -> float:
        return abs(self.proximal - self.distal) / self.atr if self.atr else 0.0


@dataclass
class SDPlan:
    entry: float
    sl: float
    tp: Optional[float]
    rr: Optional[float]
    target_kind: Optional[str]   # zone | swing | None


def closed_bars(df: pd.DataFrame, now=None, delta: timedelta = HTF_DELTA) -> pd.DataFrame:
    """Oluşan son mumu dusur: acilis + mum suresi > now ise mum kapanmamistir."""
    if df is None or df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    last = df.index[-1]
    if getattr(last, "tzinfo", None) is None:
        now = now.replace(tzinfo=None) if getattr(now, "tzinfo", None) else now
    elif getattr(now, "tzinfo", None) is None:
        now = now.replace(tzinfo=timezone.utc)
    if last + delta > now:
        return df.iloc[:-1]
    return df


def _legin_letter(legin_atr: float) -> str:
    if legin_atr >= LEG_IN_MIN_ATR:
        return "R"
    if legin_atr <= -LEG_IN_MIN_ATR:
        return "D"
    return "?"


def detect_zones(df: pd.DataFrame, lookback: int = LOOKBACK_BARS) -> list[SDZone]:
    """Kapanmis HTF serisinden bolgeler (zaman sirali). Dokunus/kirilma da burada hesaplanir."""
    if df is None or len(df) < ATR_PERIOD + LEG_IN_BARS + BASE_MAX_BARS + 2:
        return []
    df = df.iloc[-lookback:] if len(df) > lookback else df
    idx = df.index
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr = compute_atr(df, ATR_PERIOD).to_numpy(dtype=float)
    n = len(df)
    zones: list[SDZone] = []
    j = LEG_IN_BARS + 1
    while j < n:
        a = atr[j - 1]
        if not np.isfinite(a) or a <= 0:
            j += 1
            continue
        body = c[j] - o[j]
        if abs(body) < LEG_BODY_MIN_ATR * a:
            j += 1
            continue
        up = body > 0
        # base: j-1'den geriye kucuk mumlar
        bs = j - 1
        while bs > 0 and (j - bs) < BASE_MAX_BARS and (h[bs - 1] - l[bs - 1]) <= BASE_MAX_RANGE_ATR * a:
            bs -= 1
        if (h[j - 1] - l[j - 1]) > BASE_MAX_RANGE_ATR * a:
            j += 1
            continue
        base_hi = h[bs:j].max()
        base_lo = l[bs:j].min()
        if up and c[j] <= base_hi:
            j += 1
            continue
        if not up and c[j] >= base_lo:
            j += 1
            continue
        bodies_hi = np.maximum(o[bs:j], c[bs:j]).max()
        bodies_lo = np.minimum(o[bs:j], c[bs:j]).min()
        if up:
            proximal, distal = bodies_hi, base_lo
        else:
            proximal, distal = bodies_lo, base_hi
        # leg-out: proximal'den LEG_OUT_BARS icinde en az LEG_OUT_MIN_ATR
        end = min(n, j + LEG_OUT_BARS)
        formed = None
        legout = 0.0
        for k in range(j, end):
            ext = h[k] if up else l[k]
            legout = max(legout, (ext - proximal) if up else (proximal - ext))
            if legout >= LEG_OUT_MIN_ATR * a:
                formed = k
                break
        if formed is None:
            j += 1
            continue
        # leg-in: base'den once LEG_IN_BARS mumluk hamle (ATR cinsinden, isaretli)
        li0 = bs - 1 - LEG_IN_BARS
        legin = (c[bs - 1] - c[li0]) / a if li0 >= 0 else 0.0
        letter_out = "R" if up else "D"
        zone_type = f"{_legin_letter(legin)}B{letter_out}"
        # FVG: leg-out icinde 3'lu bosluk (base son mumu / leg-out mumu / sonraki)
        fvg = False
        for k in range(j, min(n - 1, j + LEG_OUT_BARS)):
            if up and l[k + 1] > h[k - 1]:
                fvg = True
                break
            if not up and h[k + 1] < l[k - 1]:
                fvg = True
                break
        # bos / sweep: base'den onceki son swing (5'li fraktal, SWING_LOOKBACK icinde)
        sw_hi = _last_swing_before(h, bs, "high")
        sw_lo = _last_swing_before(l, bs, "low")
        bos_end = min(n, formed + BOS_EXTRA_BARS + 1)
        if up:
            sweep = sw_lo is not None and base_lo < sw_lo
            bos = sw_hi is not None and bool((c[j:bos_end] > sw_hi).any())
        else:
            sweep = sw_hi is not None and base_hi > sw_hi
            bos = sw_lo is not None and bool((c[j:bos_end] < sw_lo).any())
        z = SDZone(
            side="demand" if up else "supply",
            direction="LONG" if up else "SHORT",
            zone_type=zone_type,
            base_start=idx[bs], base_end=idx[j - 1], legout_time=idx[j], formed_time=idx[formed],
            proximal=float(proximal), distal=float(distal), atr=float(a),
            legout_atr=round(float(legout / a), 3), legin_atr=round(float(legin), 3),
            base_bars=int(j - bs), base_max_atr=round(float((h[bs:j] - l[bs:j]).max() / a), 3),
            fvg=bool(fvg), formed_iloc=int(formed), bos=bool(bos), sweep=bool(sweep),
        )
        _walk_after(z, h, l, idx)
        zones.append(z)
        j = formed + 1
    return zones


def _walk_after(z: SDZone, h: np.ndarray, l: np.ndarray, idx) -> None:
    """Olusumdan sonraki mumlar: dokunus kosulari ve distal kirilmasi."""
    buf = SL_BUFFER_ATR * z.atr
    in_run = False
    for i in range(z.formed_iloc + 1, len(h)):
        if z.side == "demand":
            touching = l[i] <= z.proximal
            beyond = l[i] < z.distal - buf
        else:
            touching = h[i] >= z.proximal
            beyond = h[i] > z.distal + buf
        if touching and not in_run:
            z.touches += 1
            z.last_touch = idx[i]
        in_run = touching
        if beyond:
            z.broken_at = idx[i]
            break
    z.in_zone = in_run and z.broken_at is None


def _last_swing_before(vals: np.ndarray, end: int, kind: str, k: int = TARGET_SWING_K,
                       lookback: int = SWING_LOOKBACK) -> Optional[float]:
    """`end`'den (haric) onceki son fraktal tepe/dip; sag kanadi da `end`'den once kapanmis olmali."""
    for i in range(end - 1 - k, max(k, end - lookback) - 1, -1):
        left, right = vals[i - k:i], vals[i + 1:i + k + 1]
        if kind == "high" and vals[i] > left.max() and vals[i] > right.max():
            return float(vals[i])
        if kind == "low" and vals[i] < left.min() and vals[i] < right.min():
            return float(vals[i])
    return None


def _swing_levels(df: pd.DataFrame, kind: str, k: int = TARGET_SWING_K) -> list[float]:
    vals = (df["high"] if kind == "high" else df["low"]).to_numpy(dtype=float)
    out: list[float] = []
    n = len(vals)
    for i in range(k, n - k):
        left, right = vals[i - k:i], vals[i + 1:i + k + 1]
        if kind == "high" and vals[i] > left.max() and vals[i] > right.max():
            out.append(float(vals[i]))
        elif kind == "low" and vals[i] < left.min() and vals[i] < right.min():
            out.append(float(vals[i]))
    return out


def plan_zone(z: SDZone, zones: list[SDZone], df: pd.DataFrame, last_close: float | None = None) -> SDPlan:
    """Giris / SL / TP: TP karsi taraftaki ilk canli bolge, yoksa fiyatin otesindeki ilk swing."""
    buf = SL_BUFFER_ATR * z.atr
    entry = z.proximal
    long = z.direction == "LONG"
    sl = z.distal - buf if long else z.distal + buf
    if last_close is None:
        last_close = float(df["close"].iloc[-1])
    ref = max(entry, last_close) if long else min(entry, last_close)
    tp, kind = None, None
    cands = [o.proximal for o in zones
             if o.side != z.side and o.broken_at is None and ((o.proximal > ref) if long else (o.proximal < ref))]
    if cands:
        tp, kind = (min(cands) if long else max(cands)), "zone"
    else:
        swings = [s for s in _swing_levels(df, "high" if long else "low") if ((s > ref) if long else (s < ref))]
        if swings:
            tp, kind = (min(swings) if long else max(swings)), "swing"
    rr = None
    risk = abs(entry - sl)
    if tp is not None and risk > 0:
        rr = round(abs(tp - entry) / risk, 2)
    return SDPlan(entry=float(entry), sl=float(sl), tp=tp, rr=rr, target_kind=kind)


def score_zone(z: SDZone, bias: str | None) -> dict:
    """Kalem kalem skor (olcum). Tavan MAX_SCORE; esik yok."""
    parts = {
        "strength": 2 if z.legout_atr >= STRENGTH_STRONG_ATR else 1,
        "base": {1: 2, 2: 1}.get(z.base_bars, 0),
        "fvg": 1 if z.fvg else 0,
        "fresh": 2 if z.fresh else 0,
        "htf": 0,
    }
    b = (bias or "NEUTRAL").upper()
    if b == ("BULLISH" if z.direction == "LONG" else "BEARISH"):
        parts["htf"] = 2
    elif b == ("BEARISH" if z.direction == "LONG" else "BULLISH"):
        parts["htf"] = -2
    raw = sum(parts.values())
    parts["raw"] = raw
    parts["score"] = max(0, min(MAX_SCORE, raw))
    return parts


def zone_features(z: SDZone, last_close: float, n_bars_since: int, plan: SDPlan | None = None) -> dict:
    dist = ((last_close - z.proximal) if z.direction == "LONG" else (z.proximal - last_close)) / z.atr if z.atr else None
    out = {
        "zone_type": z.zone_type,
        "legout_atr": z.legout_atr,
        "legin_atr": z.legin_atr,
        "base_bars": z.base_bars,
        "base_max_atr": z.base_max_atr,
        "zone_atr": round(z.height_atr, 3),
        "fvg": 1 if z.fvg else 0,
        "bos": 1 if z.bos else 0,
        "sweep": 1 if z.sweep else 0,
        "touches": z.touches,
        "in_zone": 1 if z.in_zone else 0,
        "dist_atr": round(float(dist), 3) if dist is not None else None,
        "age_bars": int(n_bars_since),
    }
    if plan is not None:
        out["target_kind"] = plan.target_kind
    return out


def bias_aligned(direction: str, bias: str | None) -> bool:
    b = (bias or "NEUTRAL").upper()
    return b != ("BEARISH" if direction == "LONG" else "BULLISH")
