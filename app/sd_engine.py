"""Supply / Demand bolge motoru (03.11, kullanici istegi) -- saf hesap, OLCUM MODU.

CRT'den bagimsiz dorduncu strateji (`sd`): 4H'te bolge, 1D bias ile yon, 15m'de dolum/sonuc izleme
(Setup Journal). Sinyal ACMAZ, Telegram'a gitmez; amac once "hangi bolge tipi kazaniyor" sorusunu
Journal'la olcmek, canli sinyale veriyle gecmek.

Bolge tanimi (uc parca):
  leg-in  -> base'e gelen hamle (LEG_IN_BARS mumda >= LEG_IN_MIN_ATR; yoksa yon "?")
  base    -> 1..BASE_MAX_BARS kucuk mum (araligi <= BASE_MAX_RANGE_ATR x ATR; 03.11 kullanici: 3 mum / 0.6 ATR XAUUSD 1H
             02.11 15:00 TSI demand'ini 0.03 ATR'yle kaciriyordu -> 4 mum / 0.8 ATR)
  leg-out -> base'den sert cikis: ilk mumun govdesi >= LEG_BODY_MIN_ATR x ATR, base disinda kapanis,
             LEG_OUT_BARS icinde proximal'den >= LEG_OUT_MIN_ATR x ATR yol
Demand (cikis yukari): distal = base'in en dusuk low'u, proximal = base'in en yuksek high'i (fitil-fitil, 09.11).
Supply (cikis asagi): distal = base'in en yuksek high'i, proximal = base'in en dusuk low'u.
09.11'e kadar proximal base govdelerinin ucuydu (govde-fitil); o kenar `body_prox`, leg-out ve premium / discount hala ondan.
Tip: leg-in harfi + B + leg-out harfi -> RBR / DBD (devam), DBR / RBD (donus), ?BR / ?BD (leg-in zayif).

Tazelik: olusumdan sonra proximal'e dokunan her ayri mum kosusu bir "dokunus" (kosu ancak fiyat proximal'den
TOUCH_EXIT_ATR x ATR uzaklasinca biter -- 03.11: XAUUSD'de ayni tepkinin iki dalisi 2 dokunus sayilip bolgeyi olduruyordu);
0 taze, 1 test edilmis, >= 2 ya da distal'in otesine fitil (SL_BUFFER_ATR) -> olu (islem aranmaz, hedef de olmaz).

Plan: giris proximal, SL distal + SL_BUFFER_ATR x ATR ama risk en az MIN_RISK_ATR x ATR (03.11 kullanici: base dar olunca
8 puanlik stop + her hedef 4R tavaninda; 1.0 ATR bile dar bulundu -> 1.5; 06.11 -> 1.75, EURJPY 15m),
TP fiyatin otesindeki en yakin engel (karsi canli bolgenin proximal'i ya da fraktal swing, TARGET_SWING_K); engel RR < MIN_RR
veriyorsa TP = MIN_RR noktasi (06.11; `target_kind` "rr2"; RR < FLOOR_RR_MIN ara engeller atlanir). Eski plan (1.5 ATR + ilk RR >= 2 engel) `features.sl_old/tp_old`.

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
BASE_MAX_BARS = 4
# Base mumu = KARARSIZ mum: govde <= BASE_MAX_BODY_ATR x ATR (03.11 kullanici vakasi ETH 15m 03.11 15:00 TSI: govde 0.48 ATR ama
# fitilli aralik 1.32 ATR, eski 'aralik <= 0.8' kurali bolgeyi kaciriyordu); aralik tavani yalniz asiri fitilli mumu elemek icin.
BASE_MAX_BODY_ATR = 0.5
BASE_MAX_RANGE_ATR = 1.5
LEG_OUT_MIN_ATR = 2.0
LEG_OUT_BARS = 3
LEG_BODY_MIN_ATR = 1.0
# 05.11 (kullanici, KAS 4H 02.11 15:00 TSI doji base'i: cikis mumu govdesi 0.91 ATR, bolge goruldu ama sayilmadi): govdesi
# NEAR_LEG_BODY_ATR..LEG_BODY_MIN_ATR arasi cikislar ayri "yakin kacan bolge" (Journal `sd_near`) -- Signals'a girmez, olcum.
NEAR_LEG_BODY_ATR = 0.8
# 05.11 (kullanici, KAS: stop 0.03273, 4 tick ustundeki 02.11 15:00 tepesi 0.03279; fiyat 0.03277'ye dokunup dondu): stopun
# otesinde STOP_LIQ_ATR icinde alinmamis tepe/dip varsa (base oncesi SWING_LOOKBACK mum, fraktal sarti yok) stop onun otesine --
# yalniz OLCUM (Journal golge stop "liq", plan.sl degismez).
STOP_LIQ_ATR = 0.5
LEG_IN_BARS = 3
LEG_IN_MIN_ATR = 1.0
SL_BUFFER_ATR = 0.1
# 06.11 (kullanici, EURJPY 15m SHORT 09:45 TSI: 1.5 ATR stop tepeyi 0.55 pip gecip stop oldu, sonra TP): 1.5 -> 1.75.
# Eski plan (OLD_MIN_RISK_ATR + eski hedef kurali) Journal'da golge kol "old" olarak olculur.
MIN_RISK_ATR = 1.75          # stop mesafesi en az bu kadar ATR
OLD_MIN_RISK_ATR = 1.5
TOUCH_EXIT_ATR = 0.5         # dokunus kosusu: fiyat proximal'den bu kadar uzaklasinca biter
# 09.11 (kullanici): bolge siniri fitil-fitil -- proximal = base'in karsi fitili (demand tepe, supply dip); giris, dokunus / tazelik,
# plan ve karsi bolge hedefi bu kenardan. Tespit (leg-out yolu, premium / discount) govde kenarindan kaldi, bolge kumesi ayni.
# False = eski govde-fitil (07.11'e kadar). Eski giris Journal varyanti `zone_body`.
ZONE_WICK_BOUNDARY = True
# 09.11 (kullanici, EURUSD 1H 06.11 13:00 UTC+3 supply, fitil-fitil 1.39 ATR): bolge boyu >= MID_ENTRY_MIN_ATR ise giris bolgenin
# orta noktasi (`SDPlan.entry_kind` "mid"); STOP AYNI YERDE (taban kenardan hesaplanir) -- risk kuculur, RR / TP girisle yeniden.
# Dokunus / tazelik kenardan. None = kapali (her bolge kenardan).
MID_ENTRY_MIN_ATR: float | None = 1.25
MAX_TOUCHES = 1              # bu kadar dokunusa kadar canli; fazlasi olu
LOOKBACK_BARS = 200          # tespit penceresi (kapanmis 4H mumu)
MAX_AGE_BARS = 60            # daha eski base'ler Journal'a yazilmaz (03.11 kullanici: 120 FX'te ~4 hafta, 12.10'lu bolgeler listedeydi -> 60)
TARGET_SWING_K = 2           # hedef swing: 5'li fraktal
# 07.11 (kullanici, XAUUSD 4H TP 4174.38 = 05.10 tepesi, 06.10'da alinmis; EURJPY 15m'de de): hedef yalniz alinmamis swing
TARGET_UNTAKEN_ONLY = True
MIN_RR = 2.0
# 06.11 (kullanici, EURJPY 15m: "TP'si 1.7 falan olanlari 2'ye sabitle"): ilk anlamli engel RR FLOOR_RR_MIN..MIN_RR arasindaysa
# TP = MIN_RR noktasi (`target_kind` "rr2"). Bunun altindaki ara engeller (04.11 NZDUSD RR 1.1, 03.11 XAUUSD 3924 RR 0.7) eskisi
# gibi atlanir -- yoksa 2R her kucuk swing'in otesine yazilirdi.
FLOOR_RR_MIN = 1.5
# 03.11 (kullanici: "40R sacma, 3-4R'den fazlasi olmasin"): yapisal hedef (en yakin karsi bolge YA DA swing) MAX_RR'yi asarsa
# TP = giris + MAX_RR x risk (`target_kind` "cap"); yapisal RR `features.rr_struct`'ta olculmeye devam eder.
# 07.11 (kullanici: "TP'yi max 3RR yapalim, 3 olunca islem TP olarak kapansin"): 4 -> 3, butun S/D TF'leri. Yapisal hedefe gercekten
# gidildi mi: Journal golge kol "struct" (setup_journal._shadow_sd_struct; 4H/1H onay kolunda sdc.sh["struct"]).
MAX_RR = 3.0
# Girise uzaklik kapisi (`sd_far`): 03.11 kullanici -- 4R, 1.5 ATR stopla 6 ATR uzagi kabul ediyordu; fiyat oraya gunlerce donmuyor.
MAX_DIST_R = 3.0
# Hedef fiyatin hemen dibindeki swing olmasin (03.11 canli: BTC 1H SHORT hedefi son kapanisin %0.04 altindaki swing low'du, ilk
# 15m mumunda 'TP before entry' yazildi): aday engel son kapanistan en az bu kadar ATR otede olmali.
TARGET_MIN_BEYOND_ATR = 0.5
SWING_LOOKBACK = 30          # bos / sweep icin base'den onceki swing aramasi (mum)
# 05.11 (kullanici: 1H kaliteli bolge kriterleri -- sweep, displacement + MSS, fresh, premium/discount). Hepsi OLCUM, puan/kapi yok.
PD_RANGE_BARS = 72          # premium/discount araligi: base'e kadarki son 72 HTF mumu (demand alt yari, supply ust yari)
EQ_TOL_ATR = 0.2            # esit dip/tepe toleransi (CRT key level ile ayni)
BOS_EXTRA_BARS = 2           # bos: formed mumundan sonra bu kadar mum daha bakilir
# 05.11 (kullanici, INJ 4H 02.11 demand): asil likidite = genis fraktal dip/tepe (her yanda MAJOR_SWING_K mum). INJ'de base 5'li
# fraktal 7.001'i (ara dip) degil, haftanin dibi 6.868'i supurmustu. Yalniz ara swing alindiysa sweep sayilmaz, `sweep_minor` olcumu.
MAJOR_SWING_K = 5

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
    sweep_kind: Optional[str] = None   # major / eq / pdl (pdh) -- base hangi likiditeyi aldi (ilk bulunan; 05.11)
    sweep_minor: bool = False          # yalniz ara swing (5'li fraktal) alindi -- sweep sayilmaz, olcum (05.11)
    legout_body_atr: float = 0.0       # leg-out ilk mumunun govdesi / ATR (05.11, yakin kacan bolge olcumu)
    mss: bool = False                  # MSS: donus bolgesinin (DBR/RBD) kirilimi; devam bolgesinde ayni kirilim BOS (05.11)
    pd_pos: Optional[float] = None     # proximal'in PD_RANGE_BARS araligindaki yeri: 0 = dip, 1 = tepe (05.11)
    # 07.11 (kullanici: "fiyat onceki onemli swing low ile swing high arasinda..."): ayni yer, aralik = base'ten onceki son genis
    # fraktal tepe ve dip (her yanda MAJOR_SWING_K mum, PD_RANGE_BARS icinde). < 0 / > 1 = aralik disi. Yalniz OLCUM.
    pd_pos_sw: Optional[float] = None
    # 07.11 (kullanici, XAUUSD 1D 27.08 doji base): 'fitil-fitil' kenar = base'in karsi fitili (demand tepe, supply dip);
    # 09.11'den beri ZONE_WICK_BOUNDARY ile proximal'in kendisi. `body_prox` = govde kenari (eski proximal, tespit olcumleri).
    wick_prox: Optional[float] = None
    body_prox: Optional[float] = None
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
    target_kind: Optional[str]   # zone | swing | cap | rr2 | None
    tp_struct: Optional[float] = None   # tavan uygulanmadan onceki yapisal hedef
    sl_zone: Optional[float] = None     # taban uygulanmadan onceki bolge stopu (distal + tampon)
    tp_zone: Optional[float] = None     # fiyatin otesindeki ilk karsi canli bolge (olcum: 'hedef karsi bolge olsa?')
    rr_struct: Optional[float] = None
    dist_r: Optional[float] = None      # son kapanisin girise uzakligi, R cinsinden (pozitif = giris henuz dolmadi)
    sl_liq: Optional[float] = None      # stopu iten alinmamis tepe/dip (05.11; None = itilmedi)
    sl_old: Optional[float] = None      # 06.11 oncesi plan (golge "old"): 1.5 ATR taban
    tp_old: Optional[float] = None      # ... ve ilk RR >= 2 engel (MAX_RR tavanli); None = eski kuralda hedef yok
    rr_old: Optional[float] = None
    obst_rr: Optional[float] = None     # 06.11 olcum: fiyatin otesindeki EN YAKIN engelin RR'si (atlansa da); < 1 -> izleme maddesi
    obst_kind: Optional[str] = None
    entry_kind: str = "edge"            # 09.11: edge = proximal kenar, mid = buyuk bolgede orta nokta (MID_ENTRY_MIN_ATR)


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


def detect_zones(df: pd.DataFrame, lookback: int = LOOKBACK_BARS, leg_body_min: float = LEG_BODY_MIN_ATR) -> list[SDZone]:
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
        if abs(body) < leg_body_min * a:
            j += 1
            continue
        up = body > 0
        # base: j-1'den geriye kucuk mumlar
        def _is_base(i: int) -> bool:
            return abs(c[i] - o[i]) <= BASE_MAX_BODY_ATR * a and (h[i] - l[i]) <= BASE_MAX_RANGE_ATR * a
        bs = j - 1
        while bs > 0 and (j - bs) < BASE_MAX_BARS and _is_base(bs - 1):
            bs -= 1
        if not _is_base(j - 1):
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
        # leg-out: govde kenarindan (proximal) LEG_OUT_BARS icinde en az LEG_OUT_MIN_ATR -- 09.11 fitil siniri tespiti degistirmez
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
        # bos / sweep: base'den onceki son swing (SWING_LOOKBACK icinde). Swing'in kendisi base'den once, sag kanadi base
        # mumlarina tasabilir (05.11, INJ 4H: 02.11 08:00 UTC tepesi 7.307'nin ikinci sag mumu base'in ilk mumuydu, kod 7.55'e
        # kadar geri gidip BOS'u kaciriyordu; base mumlari leg-out'tan once kapanmis).
        bos_end = min(n, formed + BOS_EXTRA_BARS + 1)
        if up:
            sw_hi = _last_swing_before(h, bs, "high", wing_end=j - 1)
            bos = sw_hi is not None and bool((c[j:bos_end] > sw_hi).any())
        else:
            sw_lo = _last_swing_before(l, bs, "low", wing_end=j - 1)
            bos = sw_lo is not None and bool((c[j:bos_end] < sw_lo).any())
        # sweep: asil swing (genis fraktal) alindi mi; yalniz ara swing alindiysa `sweep_minor`
        side_vals, side_kind = (l, "low") if up else (h, "high")
        mj = _last_swing_before(side_vals, bs, side_kind, k=MAJOR_SWING_K, wing_end=j - 1)
        mn = _last_swing_before(side_vals, bs, side_kind, wing_end=j - 1)

        def _took(v):
            return v is not None and ((base_lo < v) if up else (base_hi > v))
        sweep = _took(mj)
        sweep_minor = not sweep and _took(mn)
        # Genis likidite (05.11): asil swing'e ek olarak esit dipler / tepeler ve onceki gunun ucu (UTC gunu).
        sweep_kind = "major" if sweep else None
        if not sweep:
            eq = _equal_level_before(l if up else h, bs, "low" if up else "high", EQ_TOL_ATR * a)
            if eq is not None and ((base_lo < eq) if up else (base_hi > eq)):
                sweep, sweep_kind = True, "eq"
        if not sweep:
            pdx = _prev_day_extreme(idx, l if up else h, bs, "low" if up else "high")
            if pdx is not None and ((base_lo < pdx) if up else (base_hi > pdx)):
                sweep, sweep_kind = True, ("pdl" if up else "pdh")
        if sweep:
            sweep_minor = False
        # Kirilim donus bolgesinde MSS, devam bolgesinde BOS (05.11, kullanici; SMC kullanimi). Sweep ayri etiket, MSS'e girmez.
        reversal = zone_type[0] != "?" and zone_type[0] != zone_type[2]
        mss = bool(reversal and bos)
        w0 = max(0, bs - PD_RANGE_BARS + 1)
        rng_hi, rng_lo = float(h[w0:bs + 1].max()), float(l[w0:bs + 1].min())
        pd_pos = round((proximal - rng_lo) / (rng_hi - rng_lo), 3) if rng_hi > rng_lo else None
        sw_hi = _last_swing_before(h, bs, "high", k=MAJOR_SWING_K, lookback=PD_RANGE_BARS, wing_end=j - 1)
        sw_lo = _last_swing_before(l, bs, "low", k=MAJOR_SWING_K, lookback=PD_RANGE_BARS, wing_end=j - 1)
        pd_pos_sw = (round((proximal - sw_lo) / (sw_hi - sw_lo), 3)
                     if sw_hi is not None and sw_lo is not None and sw_hi > sw_lo else None)
        z = SDZone(
            side="demand" if up else "supply",
            direction="LONG" if up else "SHORT",
            zone_type=zone_type,
            base_start=idx[bs], base_end=idx[j - 1], legout_time=idx[j], formed_time=idx[formed],
            proximal=float((base_hi if up else base_lo) if ZONE_WICK_BOUNDARY else proximal), distal=float(distal), atr=float(a),
            legout_atr=round(float(legout / a), 3), legin_atr=round(float(legin), 3),
            base_bars=int(j - bs), base_max_atr=round(float((h[bs:j] - l[bs:j]).max() / a), 3),
            fvg=bool(fvg), formed_iloc=int(formed), bos=bool(bos), sweep=bool(sweep),
            sweep_kind=sweep_kind, sweep_minor=bool(sweep_minor), mss=mss, pd_pos=pd_pos, pd_pos_sw=pd_pos_sw,
            legout_body_atr=round(float(abs(body) / a), 3),
            wick_prox=float(base_hi if up else base_lo), body_prox=float(proximal),
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
            in_run = True
        elif in_run:
            left = (l[i] > z.proximal + TOUCH_EXIT_ATR * z.atr) if z.side == "demand" \
                else (h[i] < z.proximal - TOUCH_EXIT_ATR * z.atr)
            if left:
                in_run = False
        if beyond:
            z.broken_at = idx[i]
            break
    z.in_zone = in_run and z.broken_at is None


def _last_swing_before(vals: np.ndarray, end: int, kind: str, k: int = TARGET_SWING_K,
                       lookback: int = SWING_LOOKBACK, wing_end: Optional[int] = None) -> Optional[float]:
    """`end`'den (haric) onceki son fraktal tepe/dip. Sag kanat en fazla `wing_end` mumuna (dahil) uzanir; verilmezse
    `end`'den once kapanmis olmali (eski davranis)."""
    last = (end - 1) if wing_end is None else min(int(wing_end), len(vals) - 1)
    for i in range(min(end - 1, last - k), max(k, end - lookback) - 1, -1):
        left, right = vals[i - k:i], vals[i + 1:i + k + 1]
        if kind == "high" and vals[i] > left.max() and vals[i] > right.max():
            return float(vals[i])
        if kind == "low" and vals[i] < left.min() and vals[i] < right.min():
            return float(vals[i])
    return None


def _equal_level_before(vals: np.ndarray, end: int, kind: str, tol: float, k: int = TARGET_SWING_K,
                        lookback: int = SWING_LOOKBACK) -> Optional[float]:
    """`end`'den onceki pencerede birbirine `tol` kadar yakin iki fraktal dip/tepe (esit dipler = SSL havuzu). Havuzun ucu."""
    piv = []
    for i in range(max(k, end - lookback), end - k):
        left, right = vals[i - k:i], vals[i + 1:i + k + 1]
        if kind == "low" and vals[i] < left.min() and vals[i] < right.min():
            piv.append(float(vals[i]))
        elif kind == "high" and vals[i] > left.max() and vals[i] > right.max():
            piv.append(float(vals[i]))
    best = None
    for a_i in range(len(piv)):
        for b_i in range(a_i + 1, len(piv)):
            if abs(piv[a_i] - piv[b_i]) <= tol:
                lvl = min(piv[a_i], piv[b_i]) if kind == "low" else max(piv[a_i], piv[b_i])
                best = lvl if best is None else (min(best, lvl) if kind == "high" else max(best, lvl))
    return best


def _prev_day_extreme(idx, vals: np.ndarray, end: int, kind: str) -> Optional[float]:
    """Base'in gununden bir onceki UTC gunun dibi / tepesi (seans hizasi yok -- olcum icin yeterli)."""
    try:
        day = pd.Timestamp(idx[end]).normalize()
        days = pd.DatetimeIndex(idx[:end]).normalize()
        prev = days[days < day]
        if len(prev) == 0:
            return None
        sel = vals[:end][days == prev.max()]
        return float(sel.min() if kind == "low" else sel.max()) if len(sel) else None
    except Exception:
        return None


def _swing_levels(df: pd.DataFrame, kind: str, k: int = TARGET_SWING_K, untaken: bool = TARGET_UNTAKEN_ONLY) -> list[float]:
    """Fraktal tepe / dipler. `untaken` (07.11): sonradan fiyatin gecmedigi (alinmamis) swing'ler -- alinmis swing likidite degil."""
    vals = (df["high"] if kind == "high" else df["low"]).to_numpy(dtype=float)
    out: list[float] = []
    n = len(vals)
    for i in range(k, n - k):
        left, right = vals[i - k:i], vals[i + 1:i + k + 1]
        if kind == "high" and vals[i] > left.max() and vals[i] > right.max():
            if untaken and (vals[i + 1:] > vals[i]).any():
                continue
            out.append(float(vals[i]))
        elif kind == "low" and vals[i] < left.min() and vals[i] < right.min():
            if untaken and (vals[i + 1:] < vals[i]).any():
                continue
            out.append(float(vals[i]))
    return out


def _liq_beyond_stop(z: SDZone, df: pd.DataFrame, sl: float) -> Optional[float]:
    """Stopun otesinde STOP_LIQ_ATR icindeki en uzak alinmamis tepe (supply) / dip (demand): base'den onceki SWING_LOOKBACK mum
    (her mum, fraktal sarti yok -- KAS'ta seviye bir doji base'in tepesiydi), sonrasinda bugune kadar asilmamis."""
    try:
        bs = int(df.index.get_loc(z.base_start))
    except Exception:
        return None
    long = z.direction == "LONG"
    vals = (df["low"] if long else df["high"]).to_numpy(dtype=float)
    band = STOP_LIQ_ATR * z.atr
    best = None
    for i in range(max(0, bs - SWING_LOOKBACK), bs):
        v = vals[i]
        inside = (sl - band <= v < sl) if long else (sl < v <= sl + band)
        if not inside:
            continue
        later = vals[i + 1:]
        if len(later) and ((later.min() < v) if long else (later.max() > v)):
            continue                                  # sonradan alinmis -> likidite degil
        if best is None or ((v < best) if long else (v > best)):
            best = float(v)
    return best


def plan_zone(z: SDZone, zones: list[SDZone], df: pd.DataFrame, last_close: float | None = None) -> SDPlan:
    """Giris / SL / TP: TP = en yakin engel (karsi canli bolgenin proximal'i ya da fiyatin otesindeki ilk swing), MAX_RR tavani."""
    buf = SL_BUFFER_ATR * z.atr
    mid = MID_ENTRY_MIN_ATR is not None and z.height_atr >= MID_ENTRY_MIN_ATR
    entry = (z.proximal + z.distal) / 2 if mid else z.proximal
    long = z.direction == "LONG"
    sl_zone = z.distal - buf if long else z.distal + buf

    def _sl(mult: float) -> float:                # taban kenardan (09.11: orta giriste stop yerinde kalir)
        floor = mult * z.atr
        return min(sl_zone, z.proximal - floor) if long else max(sl_zone, z.proximal + floor)

    sl = _sl(MIN_RISK_ATR)
    # Olcum (05.11): stopu otesindeki alinmamis tepe/dibin otesine itmek ana stopa UYGULANMAZ -- onbellekteki 4H/1H serilerinde
    # canli bolgelerin %58'inde (3 mumluk pencerede bile %29) stop degisiyordu; Journal golgesi "liq" ile olculur.
    sl_liq = _liq_beyond_stop(z, df, sl)
    if last_close is None:
        last_close = float(df["close"].iloc[-1])
    ref = max(entry, last_close) if long else min(entry, last_close)
    ref = ref + TARGET_MIN_BEYOND_ATR * z.atr if long else ref - TARGET_MIN_BEYOND_ATR * z.atr
    tp, kind = None, None
    cands = [(o.proximal, "zone") for o in zones
             if o.side != z.side and o.broken_at is None and ((o.proximal > ref) if long else (o.proximal < ref))]
    cands += [(s, "swing") for s in _swing_levels(df, "high" if long else "low") if ((s > ref) if long else (s < ref))]
    risk = abs(entry - sl)
    zone_lvls = [lvl for lvl, k in cands if k == "zone"]
    tp_zone = (min(zone_lvls) if long else max(zone_lvls)) if zone_lvls else None
    ordered = sorted(cands, reverse=not long)

    def _target(risk_: float, floor_rr: bool):
        """(tp, kind, tp_struct, rr_struct, rr). Eski kural (04.11, NZDUSD 1H RR 1.1): sirayla ilk RR >= MIN_RR engel; hicbiri
        vermiyorsa en yakini (satir `sd_low_rr` kalir). floor_rr (06.11): RR < FLOOR_RR_MIN ara engelleri atladiktan sonraki ilk
        engel RR < MIN_RR veriyorsa TP = MIN_RR noktasi ("rr2"; yapisal = o engel) -- o nokta fiyatin otesinde degilse eski kural."""
        if not ordered or risk_ <= 0:
            return None, None, None, None, None
        tp_, kind_ = ordered[0]
        if floor_rr:
            first = next(((lvl, k) for lvl, k in ordered if abs(lvl - entry) / risk_ >= FLOOR_RR_MIN), None)
            p2 = entry + MIN_RR * risk_ if long else entry - MIN_RR * risk_
            if first is not None and abs(first[0] - entry) / risk_ < MIN_RR and ((p2 > ref) if long else (p2 < ref)):
                return p2, "rr2", first[0], round(abs(first[0] - entry) / risk_, 2), MIN_RR
        if abs(tp_ - entry) / risk_ < MIN_RR:
            for lvl, k in ordered:
                if abs(lvl - entry) / risk_ >= MIN_RR:
                    tp_, kind_ = lvl, k
                    break
        rs = round(abs(tp_ - entry) / risk_, 2)
        ts_ = tp_
        if rs > MAX_RR:
            tp_ = entry + MAX_RR * risk_ if long else entry - MAX_RR * risk_
            kind_ = "cap"
        return tp_, kind_, ts_, rs, round(abs(tp_ - entry) / risk_, 2)

    tp, kind, tp_struct, rr_struct, rr = _target(risk, True)
    obst_rr = round(abs(ordered[0][0] - entry) / risk, 2) if ordered and risk > 0 else None
    dist_r = round(((last_close - entry) if long else (entry - last_close)) / risk, 2) if risk > 0 else None
    sl_old = _sl(OLD_MIN_RISK_ATR)
    tp_old, _, _, _, rr_old = _target(abs(entry - sl_old), False)
    return SDPlan(entry=float(entry), sl=float(sl), tp=tp, rr=rr, target_kind=kind,
                  tp_struct=tp_struct, rr_struct=rr_struct, dist_r=dist_r, sl_zone=float(sl_zone),
                  tp_zone=float(tp_zone) if tp_zone is not None else None, sl_liq=sl_liq,
                  sl_old=float(sl_old), tp_old=float(tp_old) if tp_old is not None else None, rr_old=rr_old,
                  obst_rr=obst_rr, obst_kind=ordered[0][1] if ordered else None, entry_kind="mid" if mid else "edge")


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
        "sweep_kind": z.sweep_kind,
        "sweep_minor": 1 if z.sweep_minor else 0,
        "legout_body_atr": z.legout_body_atr,
        "mss": 1 if z.mss else 0,
        "pd_pos": z.pd_pos,
        # demand alt yarida (discount), supply ust yarida (premium)
        "pd_ok": None if z.pd_pos is None else (1 if ((z.pd_pos < 0.5) if z.side == "demand" else (z.pd_pos > 0.5)) else 0),
        # 07.11: ayni kural swing araligiyla (olcum; /sd-izleme "S/D premium / discount: 72 mum mu, swing araligi mi?")
        "pd_pos_sw": z.pd_pos_sw,
        "pd_ok_sw": None if z.pd_pos_sw is None else (1 if ((z.pd_pos_sw < 0.5) if z.side == "demand" else (z.pd_pos_sw > 0.5)) else 0),
        "touches": z.touches,
        "distal": float(z.distal),           # 07.11: LTF likidite bandi (sd_confirm, distal ile stop arasi)
        "in_zone": 1 if z.in_zone else 0,
        "dist_atr": round(float(dist), 3) if dist is not None else None,
        "age_bars": int(n_bars_since),
    }
    if plan is not None:
        out["target_kind"] = plan.target_kind
        out["entry_kind"] = plan.entry_kind
        out["rr_struct"] = plan.rr_struct
        out["dist_r"] = plan.dist_r
        out["risk_atr"] = round(abs(plan.entry - plan.sl) / z.atr, 2) if z.atr else None
        out["sl_zone"] = plan.sl_zone
        out["tp_zone"] = plan.tp_zone
        out["sl_liq"] = plan.sl_liq
        out["sl_old"] = plan.sl_old
        out["tp_old"] = plan.tp_old
        out["rr_old"] = plan.rr_old
        out["obst_rr"] = plan.obst_rr
        out["obst_kind"] = plan.obst_kind
        if plan.sl_liq is not None and z.atr:
            b = SL_BUFFER_ATR * z.atr
            out["sl_liq_stop"] = plan.sl_liq - b if z.direction == "LONG" else plan.sl_liq + b
    return out


def bias_aligned(direction: str, bias: str | None) -> bool:
    b = (bias or "NEUTRAL").upper()
    return b != ("BEARISH" if direction == "LONG" else "BULLISH")
