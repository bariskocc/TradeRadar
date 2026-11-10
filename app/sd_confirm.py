"""S/D "LTF onay + retest" giris varyanti (03.11; 05.11'de yeniden yazildi -- kullanici) -- OLCUM, motor karari degil.

Soru: bolgeye dokununca dogrudan limit (bugunku kural) mi, yoksa LTF'de CHOCH bekleyip bir LTF bolgesinin retestinde mi girmeli?
Ayni Journal satirinda izlenir; durum `entries["sdc"]` (varyant donguleri `_STATE_KEYS` ile atlar). LTF = Journal'in izleme serisi
(4H bolge -> 15m, 1H bolge -> 5m; `setup_journal.LTF_OF`). Journal mumlari open tasimaz: open ~ onceki kapanis.

Akis (LONG; SHORT simetrik):
  0. Dokunustan ONCE de son PRE_BARS mum tamponda tutulur (05.11: bolgeye gelen dususun yapisi gerekli -- V donusunde dip
     dokunus mumunun kendisiydi, oncesinde mum olmadigi icin swing bulunamiyor, CHOCH hic tetiklenmiyordu; TP olan islemlerde
     "Waiting CHOCH" yaziyordu).
  1. Dokunus (satirin `entry_touched_at`) -> `wait_choch`. Dokunustan beri en dusuk dip = `lo`.
  2. CHOCH seviyesi = `lo`'dan ONCEKI son swing tepe (her yanda CHOCH_SWING_K mum -- 05.11: 3'lu fraktal yukselis icindeki minik
     duraklamayi CHOCH sayiyordu, KAS 4H; tampon dokunus oncesini de kapsar).
     LTF sweep + kisa swing (05.11, kullanici: XRP 4H -- dusus selale gibi indi, 3'lu ara tepe yok, CHOCH dususun basina
     kaydi, giris en tepeden RR 0.56): dip, bolgenin base'inden beri olusmus ve dokunusa kadar alinmamis LTF swing dip(ler)ini
     aldiysa (`ltf_sweep`, her yanda CHOCH_SWING_K mum; tohum `features.ltf_liq`, sd_scanner) VE 3'lu swing'in MSS girisi
     RR < SHORT_SWING_RR veriyorsa CHOCH seviyesi dipten onceki son 1'li tepe olur (`st["kk"]` = 1).
     Bir LTF kapanisi bu seviyenin ustunde -> CHOCH. Dip satirin SL'ini once gecerse: `failed` (zone failed before CHOCH).
     10.11 (kullanici, EURCHF 1H 19953): SWEEP_CHOCH akisinda CHOCH seviyesi TEYITLI swing olmak zorunda (CHOCH_PIVOT: pencerede her
     yanda CHOCH_SWING_K mumla dogrulanmis son dip / tepe); swing yoksa CHOCH aranmaz. 07.11-10.11 arasi pencerenin en dusuk dibi
     seviyeydi -- tek mumluk pencerede dokunus mumunun dibi 'kiriliyor', islem stop oluyordu (replay: 7 CHOCH'un 2'si islem olmazdi).
  3. CHOCH aninda giris adaylari (hepsi `lo` ile CHOCH kapanisi arasinda), her biri ayri limit emri:
       fvg  -- dip -> CHOCH hamlesinin biraktigi yukselis FVG'si, ust kenar (fiyata en yakini)
       ifvg -- dususun biraktigi dusus FVG'si, CHOCH kapanisi ust kenarinin ustunde (invert), ust kenar (en yakini)
       cisd -- `lo`'ya inen son dusus blogunun ilk mumunun acilisi
       mss  -- kirilan swing tepe
     Ortak SL = satirin bolge plani stopu (07.11, kullanici -- XAUUSD 4H: dip stopu 0.54 ATR, RR 6.16, stop oldu; bolge stopuyla
     Cuma +0.4R). Eski dip stopu (lo - SL_BUFFER_FRAC x (mss - lo)) `sl_dip`'te, golge kol "dip". TP = yapisal hedef (`tps`: satir
     tavanliysa features.tp_struct, degilse satirin TP'si), her adayda kendi riskinin sd_engine.MAX_RR katiyla tavanli (`vars[k].tp`,
     07.11); tavanli islemde yapisal hedefe gidildi mi golge "struct". FVG / IFVG boslugu >= MIN_GAP_FRAC x son 20 LTF mumunun
     ortalama boyu (05.11 -- CRT'deki MIN_IFVG_GAP_RANGE_FRAC; INJ / KAS'ta tek tick'lik bosluk aday oluyordu).
  4. Her aday: girise donmeden TP -> kacti; dolum -> TP / SL. Sayfadaki hikaye "ilk dolan" adaydan (ayni mumda sira fvg, ifvg,
     cisd, mss): hangi bolgeden retest geldi, sonra ne oldu. Hicbir aday dolmadan TP -> `missed` (no retest, price left).
     Ufukta: `no_touch` / `no_choch` / `open`.
  Olcum (06.11, kullanici -- motor karari degil): `disp` = CHOCH hamlesindeki (dipten sonra, CHOCH mumu dahil) en buyuk islem yonlu
  govde / son 20 LTF mumunun ort. boyu (CRT'deki cisd_disp gibi; >= DISP_STRONG "displacement"); `cc` = CHOCH kapanisi;
  `rt` / `rt_t` = CHOCH'tan sonra TP ya da SL'ye kadar fiyatin en derin geri cekilmesi (TP mumu haric -- mum ici sira bilinmez);
  derinlik `retest_depth` = 0 (CHOCH kapanisi) .. 1 (dip). "Displacement + yalniz FVG retest" modeli vars["fvg"] + disp'ten okunur.
  Min RR (06.11, kullanici -- CADJPY 4H RR 1.44, USDCHF 1H RR 1.66 ile girmisti): islem yalniz RR >= MIN_RR adaylardan; dusuk RR'li
  adaylar `vars`'ta olculmeye devam eder. Hicbir aday RR >= MIN_RR degilse `rr_wait` (islem yok, adaylar izleniyor) -> `low_rr`.
  Golge kollar (06.11, kullanici -- 6 islemin 6'si stop: CHOCH stopu mu dar, BE kurtarir mi?): ana islemin girisi ve TP'si ayni,
  `sh` = zs (SL = satirin bolge stopu; daha genis degilse invalid) / be1r (+1R gorulunce SL girise) / be50 (TP yolunun %50'si gorulunce
  SL girise). Dolum mumunda BE kurulmaz, TP gorulurse ambiguous (CRT'nin on_fill_bar kurali). Ana islem bitince de izlenir.
Not: sd15m'de bu kol yok (kullanici 04.11: "CHOCH + retest 1H ve 4H icin").
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

SDC_KEY = "sdc"
STRATEGIES = ("sd", "sd1h")
PRE_BARS = 40                    # dokunus oncesi tutulan LTF mumu
MAX_BUFFER = 400
SL_BUFFER_FRAC = 0.10            # SL = dip - %10 x (MSS seviyesi - dip)
CHOCH_SWING_K = 3                # CHOCH swing'i: her yanda bu kadar mum (05.11; once 1)
SHORT_SWING_K = 1                # LTF sweep varken 3'lu swing RR'yi oldururse dusulen kisa swing (05.11)
SHORT_SWING_RR = 2.0             # 3'lu swing'in MSS girisi bu RR'nin altindaysa "swing cok yukarida"
DISP_STRONG = 1.0                # CHOCH displacement esigi (ort. LTF mum boyu kati)
MIN_RR = 2.0                     # islem acilan aday en az bu RR (06.11; bolge planiyla ayni)
BE_ARM_R = 1.0                   # golge be1r: +1R gorulunce SL girise
BE_ARM_TP_FRAC = 0.5             # golge be50: TP yolunun %50'si (CRT be_arm_tp_fraction)
# 07.11 (kullanici): ana stop satirin bolge plani stopu (`stop` = "zone"); False -> 06.11'e kadarki dip stopu
ZONE_STOP_MAIN = True
SHADOWS = ("dip", "zs", "be1r", "be50", "struct")
SHADOW_LABEL = {"dip": "dip stop", "zs": "zone stop", "be1r": "BE at +1R", "be50": "BE at 50% to TP", "struct": "structural target"}
LIQ_SEED_CAP = 40                # features.ltf_liq: en fazla bu kadar swing (en yenileri)
MIN_GAP_FRAC = 0.15              # FVG / IFVG en kucuk bosluk: son 20 LTF mumunun ort. boyu x bu (05.11)
GAP_AVG_BARS = 20
VARIANTS = ("fvg", "ifvg", "cisd", "mss")
VARIANT_LABEL = {"fvg": "FVG", "ifvg": "IFVG", "cisd": "CISD", "mss": "MSS level"}
FINAL = ("failed", "missed", "win", "loss", "ambiguous", "no_touch", "no_choch", "no_sweep", "open", "week_close", "low_rr", "invalid")
LABELS = {
    "pre": "—", "wait_choch": "Waiting CHOCH", "wait_retest": "CHOCH — waiting retest", "filled": "Retest — in trade",
    "win": "Worked", "loss": "Zone failed after CHOCH", "ambiguous": "Same bar (?)", "missed": "No retest — price left",
    "failed": "Zone failed before CHOCH", "no_touch": "No retest in horizon", "no_choch": "No CHOCH in horizon",
    "invalid": "Zone invalidated — no trade",
    "open": "Open at horizon", "week_close": "Closed at week end", "no_sweep": "No LTF sweep — no trade",
    "rr_wait": "CHOCH — RR < 2, no trade", "low_rr": "CHOCH — RR < 2, no trade",
}


def _naive(ts):
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _dt(v):
    """datetime ya da ISO metin -> naive UTC datetime."""
    return _naive(datetime.fromisoformat(v) if isinstance(v, str) else v)


def _iso(ts) -> str | None:
    ts = _naive(ts)
    return ts.isoformat() if ts is not None else None


def _state(rec: dict) -> dict:
    ents = rec.get("entries")
    if ents is None:
        ents = rec["entries"] = {}
    st = ents.get(SDC_KEY)
    if st is None or "vars" not in st:          # 05.11 oncesi durum bicimi -> bastan
        st = ents[SDC_KEY] = {"o": "pre", "b": [], "lo": None, "lo_i": None, "lvl": None, "ct": None, "vars": {},
                              "via": None, "e": None, "sl": None, "rr": None, "t": None, "at": None, "until": None}
    return st


def _swing_before(bars: list, end: int, long: bool, k: int = CHOCH_SWING_K) -> float | None:
    """`end`'den (haric) onceki son fraktal tepe (LONG) / dip (SHORT): her yanda `k` mum; sag kanat en fazla `end` mumuna uzanir."""
    last = min(end, len(bars) - 1)
    for i in range(last - k, k - 1, -1):
        side = bars[i - k:i] + bars[i + 1:i + k + 1]
        if long and all(bars[i][1] > x[1] for x in side):
            return bars[i][1]
        if not long and all(bars[i][2] < x[2] for x in side):
            return bars[i][2]
    return None


def _fractals(bars: list, long: bool, k: int = CHOCH_SWING_K) -> list:
    """Tampondaki LTF likiditesi: LONG'da swing dipler (her yanda k mum daha yuksek dip), SHORT'ta swing tepeler -> [(i, fiyat)]."""
    out = []
    for i in range(k, len(bars) - k):
        side = bars[i - k:i] + bars[i + 1:i + k + 1]
        if long and all(bars[i][2] < x[2] for x in side):
            out.append((i, bars[i][2]))
        if not long and all(bars[i][1] > x[1] for x in side):
            out.append((i, bars[i][1]))
    return out


def ltf_liq(df, start, until, long: bool, k: int = CHOCH_SWING_K) -> list:
    """sd_scanner tohumu (features.ltf_liq): `start` (bolgenin base'i) ile `until` arasinda kapanmis LTF mumlarindan swing dip (LONG) /
    tepe (SHORT) -> [[iso, fiyat, ilk_alan_mum_iso | None]]. Tampon dokunustan yalniz PRE_BARS once basladigi icin base icindeki
    swing'ler buradan gelir (XRP: 1.2440 / 1.2444 / 1.2468 base'te, dokunustan ~1 gun once)."""
    if df is None or len(df) == 0:
        return []
    import pandas as pd
    idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) is not None else df.index
    m = (idx >= pd.Timestamp(_naive(start))) & (idx < pd.Timestamp(_naive(until)))
    if not m.any():
        return []
    sub_idx, hi, lw = idx[m], df["high"].to_numpy()[m], df["low"].to_numpy()[m]
    bars = [[t.isoformat(), float(h), float(l), 0.0] for t, h, l in zip(sub_idx.to_pydatetime(), hi, lw)]
    out = []
    for i, p in _fractals(bars, long, k):
        tk = next((b[0] for b in bars[i + 1:] if ((b[2] < p) if long else (b[1] > p))), None)
        out.append([bars[i][0], round(p, 10), tk])
    return out[-LIQ_SEED_CAP:]


def _ltf_sweep(rec: dict, st: dict, long: bool) -> list:
    """Dokunustan sonra alinan LTF swing'leri (fiyatlar): base'ten beri olusmus, dokunusa kadar alinmamis. Tohum + tampon."""
    bars = st["b"]
    if not bars or rec.get("entry_touched_at") is None:
        return []
    touch, b0 = _iso(rec["entry_touched_at"]), bars[0][0]
    bk = bars[min(CHOCH_SWING_K, len(bars) - 1)][0]   # tamponun ilk k mumunda sol kanat yok -> oradaki swing tohumdan
    swings = {}
    for item in (rec.get("features") or {}).get("ltf_liq") or []:
        ts, p, tk = item[0], float(item[1]), item[2]
        if ts < bk:
            swings[ts] = (p, tk if (tk is not None and tk < b0) else "scan")
    for i, p in _fractals(bars, long):
        swings[bars[i][0]] = (float(p), "scan")
    out = []
    for ts, (p, tk) in sorted(swings.items()):
        if tk != "scan":                          # tamponun basindan once alinmis -> dokunustan once
            continue
        first = next((b[0] for b in bars if b[0] > ts and ((b[2] < p) if long else (b[1] > p))), None)
        if first is not None and first >= touch:
            out.append(p)
    return out


# --- 07.11 LTF sweep + CHOCH (kullanici, XAUUSD 4H 18174) ----------------------------------------------------------------
# Likidite (LONG; SHORT simetrik): bolgenin ALTINDA, distal ile satirin stopu arasindaki alinmamis LTF dipleri -- her yanda LIQ_K
# mum (HTF MAJOR_SWING_K gibi) ya da esit dipler (her yanda EQ_K mum, fark <= EQ_TOL_FRAC x ort. LTF mum boyu, ikincisi birinciyi
# almamis). Base mumlarinin icindekiler sayilmaz; base'ten LIQ_PRE_HTF HTF mumu oncesinden aranir (tohum `features.ltf_liq2`,
# sd_scanner), dokunustan sonraki tepkide olusan dipler de sayilir. Sweep = bir LTF mumu boyle bir dibin altina iner. CHOCH seviyesi
# = son alinan likidite dibi (dokunustan onceyse dokunus) ile sweep dibi arasindaki en yuksek tepe; kapanis bunun ustunde -> CHOCH.
# Sweep olmadan CHOCH aranmaz (ufukta `no_sweep`). Eski akis (`SWEEP_CHOCH = False`): ilk 3'lu swing kirilimi + kisa swing yedegi.
SWEEP_CHOCH = True
# 10.11 (kullanici, EURCHF 1H #19953: 11:05 TSI "CHOCH"u tek mumluk pencerenin dibiydi -- swing yoktu, kucuk geri cekilmeyle fiyat yukari
# devam etti): CHOCH seviyesi = pencerede (son alinan likidite / dokunus .. sweep mumu oncesi) her yanda CHOCH_SWING_K mumla TEYITLI son
# swing (LONG tepe / SHORT dip); teyitli swing yoksa CHOCH aranmaz, beklenir. False = 07.11 davranisi (pencerenin en yuksek tepesi / en
# dusuk dibi, swing sarti yok).
CHOCH_PIVOT = True
# 10.11 (kullanici, USDJPY 1H 19726: 1H 03:00 TSI mumu bolgenin %50'sinin ustunde, 5m 03:50 distal'in ustunde kapandi; kol yine 04:10'da
# CHOCH alip stop oldu): bolge GECERSIZ olunca CHOCH aranmaz -> `invalid` (islem yok). Gecersiz = (1) bir LTF mumu distal'in OTESINDE
# KAPANDI (fitil sweep serbest, govdeyle gecis kirilim) ya da (2) HTF mumu kutunun INVALID_CLOSE_FRAC'inin (%50) otesinde kapandi
# (`features.inv_close_at`, sd_engine._walk_after; HTF kapanisindan sonraki ilk LTF mumunda). Yalniz CHOCH oncesi (pre / wait_choch);
# CHOCH'tan sonra islem kendi stopuyla surer. Replay (7 CHOCH'lu satir, 7'si SL): 6'si islem olmazdi, EURCHF'i CHOCH_PIVOT eler.
# Bolge limiti kolu etkilenmez (limit ilk dokunusta doluyor; iptal dolumdan once 131 satirda 2 kez ateslerdi, ikisi SL).
INVALID_CLOSE = True
# 07.11 (kullanici, CADJPY 4H 28.10 base dibi): bolgenin kendi dibi (distal = base'in en dusuk fitili) de likidite (k = LIQ_K)
BASE_LOW_LIQ = True
# 07.11 (kullanici, CADJPY 4H: 15m CHOCH yavas, kucuk mumlarla geldi): LTF displacement = CHOCH'u yapan (seviyenin otesinde kapanan)
# mumun govdesi >= CHOCH_DISP_MIN x son GAP_AVG_BARS mumun ort. boyu (`sdc.bdisp`). Yalniz etiket; eski `disp` (hamledeki en buyuk
# govde) olcum olarak kaliyor -- 1.0 esiginde yavas kirilimlari da displacement sayiyordu.
CHOCH_DISP_MIN = 1.5
LIQ_K = 5
EQ_K = 2
EQ_TOL_FRAC = 0.2
LIQ_PRE_HTF = 8
LIQ2_CAP = 60
HTF_OF = {"sd": timedelta(hours=4), "sd1h": timedelta(hours=1)}


def _is_piv(vals: list, i: int, long: bool, k: int) -> bool:
    if i - k < 0 or i + k >= len(vals):
        return False
    x = vals[i]
    side = vals[i - k:i] + vals[i + 1:i + k + 1]
    return all(x < y for y in side) if long else all(x > y for y in side)


def ltf_liq2(df, base_start, base_end, until, long: bool, stop: float, distal: float, htf: timedelta) -> list:
    """sd_scanner tohumu (07.11, `features.ltf_liq2`): base'ten LIQ_PRE_HTF HTF mumu once ile `until` arasindaki kapanmis LTF
    mumlarindan bolgenin altindaki (distal ile stop arasi) dipler -> [[iso, fiyat, ilk_alan_mum_iso | None, k]] (k = LIQ_K ya da EQ_K).
    Base mumlarinin (base_start .. base_end) icindekiler yok."""
    if df is None or len(df) == 0:
        return []
    import pandas as pd
    idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) is not None else df.index
    t0 = pd.Timestamp(_naive(base_start)) - LIQ_PRE_HTF * htf
    m = (idx >= t0) & (idx < pd.Timestamp(_naive(until)))
    if not m.any():
        return []
    sub = idx[m]
    vals = [float(x) for x in (df["low"] if long else df["high"]).to_numpy()[m]]
    b0, b1 = pd.Timestamp(_naive(base_start)), pd.Timestamp(_naive(base_end))
    out = []
    if BASE_LOW_LIQ:                              # base'in kendi dibi / tepesi = distal
        ib = [i for i in range(len(vals)) if b0 <= sub[i] < b1]
        if ib:
            i = min(ib, key=lambda j: vals[j]) if long else max(ib, key=lambda j: vals[j])
            p = vals[i]
            tk = next((sub[j] for j in range(i + 1, len(vals)) if ((vals[j] < p) if long else (vals[j] > p))), None)
            out.append([sub[i].isoformat(), round(p, 10), tk.isoformat() if tk is not None else None, LIQ_K])
    for i in range(EQ_K, len(vals) - EQ_K):
        p = vals[i]
        if not _is_piv(vals, i, long, EQ_K) or (b0 <= sub[i] < b1):
            continue
        if not ((stop < p < distal) if long else (distal < p < stop)):
            continue
        k = LIQ_K if _is_piv(vals, i, long, LIQ_K) else EQ_K
        tk = next((sub[j] for j in range(i + 1, len(vals)) if ((vals[j] < p) if long else (vals[j] > p))), None)
        out.append([sub[i].isoformat(), round(p, 10), tk.isoformat() if tk is not None else None, k])
    out.sort(key=lambda x: x[0])
    return out[-LIQ2_CAP:]


def _liq_pool(rec: dict, st: dict, long: bool, bars: list) -> list:
    """Bolgenin altindaki likidite dipleri: [(iso, fiyat, k)] -- tohum (tamponun ilk LIQ_K mumundan oncekiler) + tampon."""
    f = rec.get("features") or {}
    distal, stop = f.get("distal"), float(rec["sl"])
    if distal is None or not bars:
        return []
    distal = float(distal)
    htf = HTF_OF.get(rec.get("strategy"), timedelta(hours=4))
    b_lo = _dt(rec["purge_time"]) if rec.get("purge_time") is not None else None
    b_hi = b_lo + int(f.get("base_bars") or 1) * htf if b_lo is not None else None
    bk = bars[min(LIQ_K, len(bars) - 1)][0]
    b0 = bars[0][0]
    out = {}
    for it in f.get("ltf_liq2") or []:
        ts_, p, tk, k = it[0], float(it[1]), it[2], int(it[3])
        in_base = b_lo is not None and b_lo <= _dt(ts_) < b_hi        # base dibi (BASE_LOW_LIQ): tampona dussa da tohumdan
        if (ts_ < bk or in_base) and not (tk is not None and tk < b0):
            out[ts_] = (p, k)
    vals = [b[2] if long else b[1] for b in bars]
    for i in range(EQ_K, len(bars) - EQ_K):
        if bars[i][0] < bk or not _is_piv(vals, i, long, EQ_K):
            continue
        t_i = _dt(bars[i][0])
        if b_lo is not None and b_lo <= t_i < b_hi:
            continue
        p = vals[i]
        if not ((stop < p < distal) if long else (distal < p < stop)):
            continue
        out[bars[i][0]] = (p, LIQ_K if _is_piv(vals, i, long, LIQ_K) else EQ_K)
    return sorted((t, p, k) for t, (p, k) in out.items())


def _sweep_choch_level(rec: dict, st: dict, long: bool, bars: list, ts, ext: float) -> float | None:
    """Son mum likidite aldi mi (st["sw_lv"] birikir), sweep dibi (`slo`) ve CHOCH seviyesi. Sweep yoksa None."""
    st.setdefault("sw_lv", [])
    st.setdefault("slo", None)
    st.setdefault("slo_i", None)
    st.setdefault("slo_t", None)
    ci = len(bars) - 1
    vals = [b[2] if long else b[1] for b in bars]
    pool = _liq_pool(rec, st, long, bars)
    times = [b[0] for b in bars]
    rng = [b[1] - b[2] for b in bars[max(0, ci - GAP_AVG_BARS + 1):ci + 1]]
    tol = EQ_TOL_FRAC * (sum(rng) / len(rng)) if rng else 0.0
    beyond = (lambda v, p: v < p) if long else (lambda v, p: v > p)

    def taken_at(t_piv: str, p: float) -> str | None:     # tamponda dibi ilk alan mum (pivotun kendisinden sonra)
        return next((times[j] for j in range(len(bars)) if times[j] > t_piv and beyond(vals[j], p)), None)

    def confirmed(t_piv: str, k: int) -> bool:            # sag kanadi bu mumdan once tamamlanmis
        if t_piv < times[0]:
            return True
        i = times.index(t_piv) if t_piv in times else None
        return i is not None and i + k <= ci - 1

    levels = []                                           # (seviye, likiditenin en gec olustugu an)
    for t_piv, p, k in pool:
        if k >= LIQ_K and confirmed(t_piv, LIQ_K):
            levels.append((p, t_piv, "major"))
    for a in range(len(pool)):                           # esit dipler: ikincisi birinciyi almamis
        ta, pa, _ = pool[a]
        tka = taken_at(ta, pa)
        for b in range(a + 1, len(pool)):
            tb, pb, _ = pool[b]
            if abs(pa - pb) > tol or not confirmed(tb, EQ_K) or (tka is not None and tka <= tb):
                continue
            levels.append((min(pa, pb) if long else max(pa, pb), tb, "eq"))
    done = {round(x[1], 10) for x in st["sw_lv"]}
    new = []
    for p, t_piv, kind in levels:
        if round(p, 10) in done:
            continue
        tk = taken_at(t_piv, p)
        if tk == times[ci]:                               # bu mum aldi (oncesinde alinmamisti)
            new.append([t_piv, round(p, 10), kind])
            done.add(round(p, 10))
    if new:
        st["sw_lv"].extend(new)
        st["slo"], st["slo_i"], st["slo_t"] = float(ext), ci, times[ci]
    elif st["slo"] is not None and beyond(ext, st["slo"]):
        st["slo"], st["slo_i"], st["slo_t"] = float(ext), ci, times[ci]
    st["sw"], st["swp"] = len(st["sw_lv"]), [x[1] for x in st["sw_lv"][-8:]]
    if not st["sw_lv"] or st["slo_i"] is None:
        return None
    t_liq = max(x[0] for x in st["sw_lv"])
    t_touch = _iso(_dt(rec["entry_touched_at"])) if rec.get("entry_touched_at") is not None else None
    start = max(t_liq, t_touch) if t_touch else t_liq
    hi = [b[1] if long else b[2] for b in bars]
    if CHOCH_PIVOT:
        # 10.11: sweep mumundan onceki, sag kanadi bu mumdan once tamamlanmis (j + k <= ci - 1) son teyitli swing; yoksa CHOCH yok
        piv = [j for j in range(st["slo_i"]) if times[j] >= start and j + CHOCH_SWING_K <= ci - 1
               and _is_piv(hi, j, not long, CHOCH_SWING_K)]
        return float(hi[piv[-1]]) if piv else None
    win = [hi[j] for j in range(st["slo_i"] + 1) if times[j] >= start]
    if not win:
        return None
    return float(max(win) if long else min(win))


def _open_choch(rec: dict, st: dict, bars: list, ci: int, long: bool, lvl: float, close: float, sl_row: float, ts) -> None:
    """CHOCH aninda: giris adaylari, stop (bolge plani; dip stopu golgede), tavanli TP'ler."""
    cands = _candidates(bars, st["lo_i"], ci, long, lvl, close, st["lo"])
    ref = abs(lvl - st["lo"])
    dip_sl = st["lo"] - SL_BUFFER_FRAC * ref if long else st["lo"] + SL_BUFFER_FRAC * ref
    sl = sl_row if ZONE_STOP_MAIN else dip_sl
    st["sl_dip"], st["stop"] = float(dip_sl), ("zone" if ZONE_STOP_MAIN else "dip")
    st["sl"], st["ct"], st["o"] = float(sl), _iso(ts), "wait_retest"
    st["cc"], st["disp"] = float(close), _disp(bars, st["lo_i"], ci, long)
    st["bdisp"] = _break_disp(bars, ci, long)
    # 07.11 (kullanici): TP her adayda kendi riskinin MAX_RR katiyla tavanli; yapisal hedef (`tps`) ayri, golge "struct"
    st["tps"] = float(_struct_tp(rec, long))
    st["vars"] = {}
    for k, e in cands.items():
        d = abs(e - sl)
        tp_v = _capped_tp(st["tps"], e, d, long) if d > 0 else st["tps"]
        st["vars"][k] = {"e": e, "tp": float(tp_v), "rr": round(abs(tp_v - e) / d, 4) if d > 0 else None,
                         "rs": round(abs(st["tps"] - e) / d, 4) if d > 0 else None,
                         "o": "pending", "t": None, "at": None}
    if st["vars"] and not _eligible(st):
        st["o"] = "rr_wait"
    st["b"] = []                                  # tampon artik gerekmez (JSON kucuk kalsin)


def _break_disp(bars: list, ci: int, long: bool) -> float | None:
    """CHOCH mumunun islem yonlu govdesi (open ~ onceki kapanis) / ondan onceki GAP_AVG_BARS mumun ort. boyu (07.11)."""
    if ci < 1:
        return None
    rng = [b[1] - b[2] for b in bars[max(0, ci - GAP_AVG_BARS):ci]]
    avg = sum(rng) / len(rng) if rng else 0.0
    body = (bars[ci][3] - bars[ci - 1][3]) if long else (bars[ci - 1][3] - bars[ci][3])
    return round(body / avg, 3) if avg > 0 else None


def is_disp(st: dict | None) -> bool:
    """LTF displacement etiketi (07.11 tanimi: CHOCH mumu)."""
    v = (st or {}).get("bdisp")
    return v is not None and float(v) >= CHOCH_DISP_MIN


def _disp(bars: list, lo_i: int, ci: int, long: bool) -> float | None:
    """CHOCH hamlesinin en guclu mumu: islem yonlu govde (open ~ onceki kapanis) / son GAP_AVG_BARS mumun ort. boyu."""
    rng = [b[1] - b[2] for b in bars[max(0, ci - GAP_AVG_BARS + 1):ci + 1]]
    avg = sum(rng) / len(rng) if rng else 0.0
    if avg <= 0:
        return None
    bodies = [(bars[i][3] - bars[i - 1][3]) * (1 if long else -1) for i in range(max(lo_i + 1, 1), ci + 1)]
    return round(max(bodies) / avg, 3) if bodies else None


def retest_depth(st: dict | None, direction: str | None = None) -> float | None:
    """CHOCH sonrasi en derin geri cekilme: 0 = CHOCH kapanisi, 1 = dip (lo), > 1 dibin otesi."""
    if not st or st.get("rt") is None or st.get("cc") is None or st.get("lo") is None:
        return None
    span = abs(float(st["cc"]) - float(st["lo"]))
    if span <= 0:
        return None
    long = (direction == "LONG") if direction else float(st["lo"]) < float(st["cc"])
    return round(((float(st["cc"]) - float(st["rt"])) if long else (float(st["rt"]) - float(st["cc"]))) / span, 3)


def _mss_rr(lvl: float, lo: float, tp: float) -> float | None:
    risk = abs(lvl - lo) * (1 + SL_BUFFER_FRAC)
    return abs(tp - lvl) / risk if risk > 0 else None


def _candidates(bars: list, lo_i: int, ci: int, long: bool, lvl: float, close: float, lo: float) -> dict:
    """CHOCH aninda giris adaylari (fiyat: lo ile CHOCH kapanisi arasinda)."""
    out: dict[str, float] = {"mss": float(lvl)}

    def opn(i: int) -> float:                     # open ~ onceki kapanis
        return bars[i - 1][3] if i > 0 else bars[i][3]

    def against(i: int) -> bool:                  # LONG: dususle kapanan mum
        return bars[i][3] < opn(i) if long else bars[i][3] > opn(i)

    j = lo_i
    while j > 0 and against(j - 1):
        j -= 1
    if against(lo_i) or j < lo_i:
        out["cisd"] = float(opn(j))
    rng = [b[1] - b[2] for b in bars[max(0, ci - GAP_AVG_BARS + 1):ci + 1]]
    min_gap = MIN_GAP_FRAC * (sum(rng) / len(rng)) if rng else 0.0
    gaps = []                                     # dip -> CHOCH hamlesinin FVG'si
    for i in range(max(lo_i, 1), ci):
        if long and bars[i + 1][2] - bars[i - 1][1] >= max(min_gap, 1e-12):
            gaps.append(bars[i + 1][2])
        if not long and bars[i - 1][2] - bars[i + 1][1] >= max(min_gap, 1e-12):
            gaps.append(bars[i + 1][1])
    if gaps:
        out["fvg"] = float(max(gaps) if long else min(gaps))
    inv = []                                      # dususun FVG'si, CHOCH kapanisiyla invert
    for i in range(1, lo_i):
        if long and bars[i - 1][2] - bars[i + 1][1] >= max(min_gap, 1e-12) and close > bars[i - 1][2]:
            inv.append(bars[i - 1][2])
        if not long and bars[i + 1][2] - bars[i - 1][1] >= max(min_gap, 1e-12) and close < bars[i - 1][1]:
            inv.append(bars[i - 1][1])
    if inv:
        out["ifvg"] = float(max(inv) if long else min(inv))
    return {k: v for k, v in out.items() if ((lo < v < close) if long else (close < v < lo))}


def step(rec: dict, bar_ts, high: float, low: float, close: float, horizon: timedelta) -> bool:
    """Kapanan LTF mumunu uygula (dokunus oncesi dahil). Degisiklik olduysa True."""
    if rec.get("strategy") not in STRATEGIES or rec.get("levels_at") is None or rec.get("entry") is None:
        return False
    st = _state(rec)
    if st["o"] in FINAL and not _sh_live(st):
        return False
    ts = _naive(bar_ts)
    if ts is None or (st.get("until") is not None and ts <= _naive(st["until"])):
        return False
    st["until"] = ts
    long = rec.get("direction") == "LONG"
    if INVALID_CLOSE and st["o"] in ("pre", "wait_choch") and _zone_invalid(rec, st, long, ts, close):
        st["o"], st["at"], st["b"] = "invalid", ts, []   # 10.11: bolge gecersiz, CHOCH aranmaz (islem yok)
        return True
    if st["o"] in FINAL:                          # ana islem bitti, golge kollar suruyor
        _sh_step(rec, st, long, ts, high, low, horizon)
        return True
    bars = st["b"]
    touched = rec.get("entry_touched_at") is not None and ts >= _naive(rec["entry_touched_at"])
    if st["o"] == "pre":
        bars.append([_iso(ts), float(high), float(low), float(close)])
        if not touched:
            del bars[:-PRE_BARS]
            return True
        st["o"] = "wait_choch"
    sl_row, tp = float(rec["sl"]), float(rec["tp"])
    if st["o"] == "wait_choch":
        if bars[-1][0] != _iso(ts):
            bars.append([_iso(ts), float(high), float(low), float(close)])
        if len(bars) > MAX_BUFFER:
            drop = len(bars) - MAX_BUFFER
            del bars[:drop]
            if st["lo_i"] is not None:
                st["lo_i"] = max(0, st["lo_i"] - drop)
        if (low <= sl_row) if long else (high >= sl_row):
            st["o"], st["at"] = "failed", ts
            return True
        ext = low if long else high
        if st["lo"] is None or ((ext < st["lo"]) if long else (ext > st["lo"])):
            st["lo"], st["lo_i"] = float(ext), len(bars) - 1
        ci = len(bars) - 1
        if SWEEP_CHOCH:
            # 07.11 (kullanici, XAUUSD 4H): once asil LTF likiditesi alinmali, CHOCH = sweep bacaginin basladigi tepe
            lvl = _sweep_choch_level(rec, st, long, bars, ts, low if long else high)
            st["lvl"], st["kk"] = lvl, None
            if lvl is not None and st["slo_i"] is not None and st["slo_i"] < ci and ((close > lvl) if long else (close < lvl)):
                st["lo"], st["lo_i"] = float(st["slo"]), int(st["slo_i"])
                _open_choch(rec, st, bars, ci, long, lvl, close, sl_row, ts)
            elif ts >= rec["levels_at"] + horizon:
                st["o"], st["at"] = ("no_choch" if st.get("sw_lv") else "no_sweep"), ts
            return True
        swept = _ltf_sweep(rec, st, long)
        st["sw"], st["swp"] = len(swept), [round(p, 10) for p in swept[-8:]]
        lvl, st["kk"] = _swing_before(bars, st["lo_i"], long), CHOCH_SWING_K
        if swept:
            rr3 = _mss_rr(lvl, st["lo"], tp) if lvl is not None else None
            if rr3 is None or rr3 < SHORT_SWING_RR:
                short = _swing_before(bars, st["lo_i"], long, k=SHORT_SWING_K)
                if short is not None and (lvl is None or ((short < lvl) if long else (short > lvl))):
                    lvl, st["kk"] = short, SHORT_SWING_K
        st["lvl"] = lvl
        if lvl is not None and st["lo_i"] < ci and ((close > lvl) if long else (close < lvl)):
            _open_choch(rec, st, bars, ci, long, lvl, close, sl_row, ts)
        elif ts >= rec["levels_at"] + horizon:
            st["o"], st["at"] = "no_choch", ts
        return True
    # CHOCH sonrasi: her aday ayri limit emri
    sl = float(st["sl"])
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    if not hit_tp:                                # en derin retest (TP mumu haric)
        ext = low if long else high
        if st.get("rt") is None or ((ext < st["rt"]) if long else (ext > st["rt"])):
            st["rt"], st["rt_t"] = float(ext), _iso(ts)
    for v in st["vars"].values():
        tpv = v.get("tp", tp)                     # 07.11: adayin kendi (tavanli) TP'si; eski satirlarda satirin TP'si
        hit_tp = high >= tpv if long else low <= tpv
        if v["o"] == "pending":
            hit_e = low <= v["e"] if long else high >= v["e"]
            if hit_tp and not hit_e:
                v["o"], v["at"] = "missed", _iso(ts)
            elif hit_e:
                v["t"] = _iso(ts)
                v["o"] = "ambiguous" if hit_tp else ("loss" if hit_sl else "filled")
                if v["o"] != "filled":
                    v["at"] = _iso(ts)
        elif v["o"] == "filled":
            if hit_tp and hit_sl:
                v["o"], v["at"] = "ambiguous", _iso(ts)
            elif hit_tp:
                v["o"], v["at"] = "win", _iso(ts)
            elif hit_sl:
                v["o"], v["at"] = "loss", _iso(ts)
    elig = _eligible(st)                          # 06.11: islem yalniz RR >= MIN_RR adaydan
    filled = [v["t"] for v in elig.values() if v.get("t")]
    if filled:
        t0 = min(filled)
        via = next(k for k in VARIANTS if elig.get(k, {}).get("t") == t0)
        v = elig[via]
        # ust duzey t/at/until datetime (Journal `_ENTRY_TS` donusumu); adaylarin (vars) zamanlari ISO metin
        st["via"], st["e"], st["rr"], st["t"] = via, v["e"], v["rr"], datetime.fromisoformat(v["t"])
        st["tp"] = float(v.get("tp", tp))
        st["o"] = v["o"] if v["o"] in ("win", "loss", "ambiguous") else "filled"
        if st["o"] != "filled":
            st["at"] = ts
    elif st["vars"] and not elig:                 # RR >= 2 aday yok: islem yok, adaylar olculmeye devam
        st["o"] = "rr_wait"
        if all(v["o"] not in ("pending", "filled") for v in st["vars"].values()):
            st["o"], st["at"] = "low_rr", ts
    elif (elig and all(v["o"] == "missed" for v in elig.values())) or (not st["vars"] and hit_tp):
        st["o"], st["at"] = "missed", ts
    if st["o"] in ("wait_retest", "filled", "rr_wait") and ts >= rec["levels_at"] + horizon:
        st["o"], st["at"] = {"wait_retest": "no_touch", "filled": "open", "rr_wait": "low_rr"}[st["o"]], ts
    if st.get("e") is not None:
        _sh_step(rec, st, long, ts, high, low, horizon)
    return True


def _zone_invalid(rec: dict, st: dict, long: bool, ts, close: float) -> bool:
    """10.11: LTF kapanisi distal'in otesinde (`inv` = ltf) ya da HTF mumu kutunun %50'sinin otesinde kapandi ve o kapanis gecti (`inv` = htf)."""
    f = rec.get("features") or {}
    d = f.get("distal")
    if d is not None and ((float(close) < float(d)) if long else (float(close) > float(d))):
        st["inv"] = "ltf"
        return True
    inv = f.get("inv_close_at")
    if inv is not None:
        t_inv = _naive(_dt(inv))
        if t_inv is not None and ts >= t_inv + HTF_OF.get(rec.get("strategy"), timedelta(hours=4)):
            st["inv"] = "htf"
            return True
    return False


def _eligible(st: dict) -> dict:
    return {k: v for k, v in (st.get("vars") or {}).items() if (v.get("rr") or 0) >= MIN_RR}


# --- Golge kollar (06.11, kullanici): ayni giris + TP; bolge stopu / BE ------------------------------------------------------
def _sh_live(st: dict) -> bool:
    return any(v.get("o") == "filled" for v in (st.get("sh") or {}).values())


def _capped_tp(tps: float, e: float, risk: float, long: bool) -> float:
    """Yapisal hedef `tps`, girisin MAX_RR x risk otesindeyse tavan noktasi (sd_engine.MAX_RR)."""
    from app.sd_engine import MAX_RR
    cap = e + MAX_RR * risk if long else e - MAX_RR * risk
    return min(tps, cap) if long else max(tps, cap)


def _struct_tp(rec: dict, long: bool) -> float:
    """Onay kolunun yapisal hedefi: satir TP'si tavanliysa (`features.target_kind` cap) `tp_struct`, degilse satirin TP'si."""
    tp = float(rec["tp"])
    f = rec.get("features") or {}
    ts_ = f.get("tp_struct")
    if f.get("target_kind") == "cap" and ts_ is not None and ((float(ts_) > tp) if long else (float(ts_) < tp)):
        return float(ts_)
    return tp


def _sh_init(rec: dict, st: dict, long: bool) -> None:
    e, sl, tp, zsl = float(st["e"]), float(st["sl"]), float(st.get("tp", rec["tp"])), float(rec["sl"])
    risk = abs(e - sl)
    sh = {}
    dsl = st.get("sl_dip")
    if st.get("stop") == "zone" and dsl is not None and abs(e - float(dsl)) > 0 and ((float(dsl) < e) if long else (float(dsl) > e)):
        sh["dip"] = {"sl": float(dsl), "sl0": float(dsl), "rr": round(abs(tp - e) / abs(e - float(dsl)), 4), "o": "filled", "at": None}
    if ((zsl < sl) if long else (zsl > sl)) and abs(e - zsl) > 0:
        sh["zs"] = {"sl": zsl, "sl0": zsl, "rr": round(abs(tp - e) / abs(e - zsl), 4), "o": "filled", "at": None}
    else:
        sh["zs"] = {"o": "invalid", "why": "not_wider"}
    for k, arm in (("be1r", e + BE_ARM_R * risk if long else e - BE_ARM_R * risk), ("be50", e + BE_ARM_TP_FRAC * (tp - e))):
        sh[k] = {"sl": sl, "sl0": sl, "arm": arm, "be": None, "rr": st.get("rr"), "o": "filled", "at": None}
    tps = st.get("tps")
    if tps is not None and risk > 0 and ((float(tps) > tp) if long else (float(tps) < tp)):
        # 07.11: tavanli islemde yapisal hedef (swing / karsi bolge) gercekten goruldu mu
        sh["struct"] = {"sl": sl, "sl0": sl, "tp": float(tps), "rr": round(abs(float(tps) - e) / risk, 4), "o": "filled", "at": None}
    st["sh"] = sh


def _sh_step(rec: dict, st: dict, long: bool, ts, high: float, low: float, horizon: timedelta) -> None:
    """Golge kollarin mumu. Dolum mumunda TP gorulurse ambiguous, BE kurulmaz; BE kurulan mumdan sonra SL = giris."""
    if "sh" not in st:
        if st.get("e") is None or st.get("t") is None:
            return
        _sh_init(rec, st, long)
        if ts > _dt(st["t"]):                     # 06.11 oncesi dolmus islem: golge dolumdan sonra basladi, analiz disi
            for v in st["sh"].values():
                v["late"] = True
    fill_bar = ts <= _dt(st["t"])
    e, tp = float(st["e"]), float(st.get("tp", rec["tp"]))
    for v in st["sh"].values():
        if v.get("o") != "filled":
            continue
        tpv = v.get("tp", tp)                     # "struct" kolunun kendi TP'si
        hit_tp = high >= tpv if long else low <= tpv
        hit_sl = low <= v["sl"] if long else high >= v["sl"]
        if hit_sl:
            v["o"] = "ambiguous" if hit_tp else ("be" if v.get("be") else "loss")
        elif hit_tp:
            v["o"] = "ambiguous" if fill_bar else "win"
        if v["o"] != "filled":
            v["at"] = _iso(ts)
            continue
        if "arm" in v and v.get("be") is None and not fill_bar and ((high >= v["arm"]) if long else (low <= v["arm"])):
            v["be"], v["sl"] = _iso(ts), e
        if ts >= rec["levels_at"] + horizon:
            v["o"], v["at"] = "open", _iso(ts)


def shadow_r(v: dict | None, e, direction: str | None) -> float | None:
    """Golge kolun R'si (kendi ilk stopunun riskiyle): win = rr, loss = -1, be / ambiguous / open = 0, hafta kapanisi fiyattan."""
    if not v:
        return None
    o = v.get("o")
    if o == "win":
        return float(v.get("rr") or 0)
    if o == "loss":
        return -1.0
    if o in ("be", "ambiguous", "open"):
        return 0.0
    if o == "week_close":
        return week_close_r(e, v.get("sl0"), v.get("px"), direction)
    return None


def describe(st: dict | None, ltf: str) -> dict:
    """Sayfa icin: kisa etiket + hikaye satiri + ipucu (her adayin seviyesi ve sonucu)."""
    if not st or st.get("o") in (None, "pre") or "vars" not in st:
        return {"label": None, "path": None, "tip": None}
    o = st["o"]
    via = st.get("via")
    via_label = f"{ltf} {VARIANT_LABEL.get(via, via)}" if via else None

    def hhmm(v):                                  # 07.11: UTC+3 + tarih ("06.11 05:30"; kullanici saatleri UTC+3 okuyor)
        if not v:
            return "?"
        try:
            return (_dt(v) + timedelta(hours=3)).strftime("%d.%m %H:%M")
        except Exception:
            return str(v)[5:16]

    parts = []
    if st.get("sw"):
        parts.append(f"LTF sweep ×{st['sw']}")
    if st.get("ct"):
        parts.append(f"CHOCH {hhmm(st['ct'])}" + (" (short swing)" if st.get("kk") == SHORT_SWING_K else ""))
    if via:
        parts.append(f"retest {via_label} {hhmm(st.get('t'))}")
    label = LABELS.get(o, o)
    if o == "wait_choch" and "sw_lv" in st and not st.get("sw_lv"):
        label = "Waiting LTF sweep"               # 07.11: once likidite alinmali
    if o == "win" and via_label:
        label = f"Worked — retest {via_label}"
    tip = " · ".join(f"{VARIANT_LABEL[k]} {v['e']:.6g} → {v['o']}" for k, v in (st.get("vars") or {}).items())
    extra = []
    if st.get("sw_lv"):
        extra.append("swept " + ", ".join(f"{x[1]:.6g}" + (" (EQ)" if x[2] == "eq" else "") for x in st["sw_lv"][-6:]))
    if st.get("lvl") is not None and st.get("o") in ("wait_choch", "no_choch"):
        extra.append(f"CHOCH level {st['lvl']:.6g}")
    if st.get("bdisp") is not None:
        extra.append(f"CHOCH candle {st['bdisp']:.2f}× avg" + (" (displacement)" if is_disp(st) else " (no displacement)"))
    elif st.get("disp") is not None:
        extra.append(f"displacement {st['disp']:.2f}×")
    d = retest_depth(st)
    if d is not None:
        extra.append(f"deepest retest {st['rt']:.6g} ({d:.0%} back to the low)")
    if st.get("inv"):
        extra.append("invalidated by " + ("LTF close beyond the zone" if st["inv"] == "ltf" else "HTF close beyond 50% of the zone"))
    for k, v in (st.get("sh") or {}).items():
        if v.get("o") not in (None, "invalid"):
            extra.append(f"{SHADOW_LABEL.get(k, k)} → {v['o']}")
    tip = " · ".join(x for x in [tip] + extra if x)
    return {"label": label, "path": " · ".join(parts) or None, "tip": tip or None}


# --- Karsi-bolge hedefi (05.11, kullanici): "TP bir sonraki S/D bolgesi olsa daha garanti olabilir" --------------------------
TPZ_KEY = "tpz"


def tpz_step(rec: dict, bar_ts, high: float, low: float, horizon: timedelta) -> bool:
    """Ayni dolum (satirin girisi) ve ayni SL ile hedef = seviye andaki en yakin karsi canli bolge (`features.tp_zone`).
    Durum entries["tpz"]: o = pending (dolum var, hedef bekliyor) | win | loss | open (ufuk) | none (karsi bolge yoktu)."""
    if rec.get("strategy") not in ("sd", "sd1h", "sd15m") or rec.get("levels_at") is None or rec.get("entry_touched_at") is None:
        return False
    ents = rec.get("entries")
    if ents is None:
        ents = rec["entries"] = {}
    st = ents.get(TPZ_KEY)
    if st is None:
        f = rec.get("features_at_levels") or rec.get("features") or {}
        tpz = f.get("tp_zone")
        e, sl = rec.get("entry"), rec.get("sl")
        if tpz is None or e is None or sl is None or abs(float(e) - float(sl)) <= 0:
            st = ents[TPZ_KEY] = {"o": "none", "tp": None, "rr": None, "at": None, "until": None}
            return True
        st = ents[TPZ_KEY] = {"o": "pending", "tp": float(tpz), "rr": round(abs(float(tpz) - float(e)) / abs(float(e) - float(sl)), 4),
                              "at": None, "until": None}
    if st["o"] != "pending":
        return False
    ts = _naive(bar_ts)
    if ts is None or ts < _naive(rec["entry_touched_at"]):
        return False
    if st.get("until") is not None and ts <= _naive(st["until"]):
        return False
    st["until"] = ts
    long = rec.get("direction") == "LONG"
    sl = float(rec["sl"])
    hit_tp = high >= st["tp"] if long else low <= st["tp"]
    hit_sl = low <= sl if long else high >= sl
    if hit_tp and hit_sl:
        st["o"], st["at"] = "ambiguous", ts
    elif hit_tp:
        st["o"], st["at"] = "win", ts
    elif hit_sl:
        st["o"], st["at"] = "loss", ts
    elif ts >= rec["levels_at"] + horizon:
        st["o"], st["at"] = "open", ts
    return True


def final_r(st: dict | None) -> float | None:
    if not st:
        return None
    if st.get("o") == "win":
        return float(st.get("rr") or 0)
    if st.get("o") == "loss":
        return -1.0
    if st.get("o") in ("ambiguous", "missed", "failed", "no_touch", "no_choch", "no_sweep", "open", "low_rr", "rr_wait", "invalid"):
        return 0.0
    if st.get("o") == "week_close":
        return week_close_r(st.get("e"), st.get("sl"), st.get("px"), st.get("dir"))
    return None


def week_close_r(e, sl, px, direction: str | None = None) -> float | None:
    """Hafta kapanisi R'si: (kapanis - giris) / risk, yon SL'nin tarafindan (direction verilmezse)."""
    if e is None or sl is None or px is None or abs(float(e) - float(sl)) <= 0:
        return None
    long = (direction == "LONG") if direction else float(sl) < float(e)
    risk = abs(float(e) - float(sl))
    return round(((float(px) - float(e)) if long else (float(e) - float(px))) / risk, 2)
