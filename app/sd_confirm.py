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
  3. CHOCH aninda giris adaylari (hepsi `lo` ile CHOCH kapanisi arasinda), her biri ayri limit emri:
       fvg  -- dip -> CHOCH hamlesinin biraktigi yukselis FVG'si, ust kenar (fiyata en yakini)
       ifvg -- dususun biraktigi dusus FVG'si, CHOCH kapanisi ust kenarinin ustunde (invert), ust kenar (en yakini)
       cisd -- `lo`'ya inen son dusus blogunun ilk mumunun acilisi
       mss  -- kirilan swing tepe
     Ortak SL = lo - SL_BUFFER_FRAC x (mss - lo); TP satirin TP'si. FVG / IFVG boslugu >= MIN_GAP_FRAC x son 20 LTF mumunun
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
SHADOWS = ("zs", "be1r", "be50")
SHADOW_LABEL = {"zs": "zone stop", "be1r": "BE at +1R", "be50": "BE at 50% to TP"}
LIQ_SEED_CAP = 40                # features.ltf_liq: en fazla bu kadar swing (en yenileri)
MIN_GAP_FRAC = 0.15              # FVG / IFVG en kucuk bosluk: son 20 LTF mumunun ort. boyu x bu (05.11)
GAP_AVG_BARS = 20
VARIANTS = ("fvg", "ifvg", "cisd", "mss")
VARIANT_LABEL = {"fvg": "FVG", "ifvg": "IFVG", "cisd": "CISD", "mss": "MSS level"}
FINAL = ("failed", "missed", "win", "loss", "ambiguous", "no_touch", "no_choch", "open", "week_close", "low_rr")
LABELS = {
    "pre": "—", "wait_choch": "Waiting CHOCH", "wait_retest": "CHOCH — waiting retest", "filled": "Retest — in trade",
    "win": "Worked", "loss": "Zone failed after CHOCH", "ambiguous": "Same bar (?)", "missed": "No retest — price left",
    "failed": "Zone failed before CHOCH", "no_touch": "No retest in horizon", "no_choch": "No CHOCH in horizon",
    "open": "Open at horizon", "week_close": "Closed at week end",
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
        ci = len(bars) - 1
        if lvl is not None and st["lo_i"] < ci and ((close > lvl) if long else (close < lvl)):
            cands = _candidates(bars, st["lo_i"], ci, long, lvl, close, st["lo"])
            ref = abs(lvl - st["lo"])
            sl = st["lo"] - SL_BUFFER_FRAC * ref if long else st["lo"] + SL_BUFFER_FRAC * ref
            st["sl"], st["ct"], st["o"] = float(sl), _iso(ts), "wait_retest"
            st["cc"], st["disp"] = float(close), _disp(bars, st["lo_i"], ci, long)
            st["vars"] = {k: {"e": e, "rr": round(abs(tp - e) / abs(e - sl), 4) if abs(e - sl) > 0 else None,
                              "o": "pending", "t": None, "at": None} for k, e in cands.items()}
            if st["vars"] and not _eligible(st):
                st["o"] = "rr_wait"
            st["b"] = []                          # tampon artik gerekmez (JSON kucuk kalsin)
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


def _eligible(st: dict) -> dict:
    return {k: v for k, v in (st.get("vars") or {}).items() if (v.get("rr") or 0) >= MIN_RR}


# --- Golge kollar (06.11, kullanici): ayni giris + TP; bolge stopu / BE ------------------------------------------------------
def _sh_live(st: dict) -> bool:
    return any(v.get("o") == "filled" for v in (st.get("sh") or {}).values())


def _sh_init(rec: dict, st: dict, long: bool) -> None:
    e, sl, tp, zsl = float(st["e"]), float(st["sl"]), float(rec["tp"]), float(rec["sl"])
    risk = abs(e - sl)
    sh = {}
    if ((zsl < sl) if long else (zsl > sl)) and abs(e - zsl) > 0:
        sh["zs"] = {"sl": zsl, "sl0": zsl, "rr": round(abs(tp - e) / abs(e - zsl), 4), "o": "filled", "at": None}
    else:
        sh["zs"] = {"o": "invalid", "why": "not_wider"}
    for k, arm in (("be1r", e + BE_ARM_R * risk if long else e - BE_ARM_R * risk), ("be50", e + BE_ARM_TP_FRAC * (tp - e))):
        sh[k] = {"sl": sl, "sl0": sl, "arm": arm, "be": None, "rr": st.get("rr"), "o": "filled", "at": None}
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
    e, tp = float(st["e"]), float(rec["tp"])
    hit_tp = high >= tp if long else low <= tp
    for v in st["sh"].values():
        if v.get("o") != "filled":
            continue
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

    def hhmm(v):
        if not v:
            return "?"
        return v.strftime("%H:%M") if hasattr(v, "strftime") else str(v)[11:16]

    parts = []
    if st.get("sw"):
        parts.append(f"LTF sweep ×{st['sw']}")
    if st.get("ct"):
        parts.append(f"CHOCH {hhmm(st['ct'])}" + (" (short swing)" if st.get("kk") == SHORT_SWING_K else ""))
    if via:
        parts.append(f"retest {via_label} {hhmm(st.get('t'))}")
    label = LABELS.get(o, o)
    if o == "win" and via_label:
        label = f"Worked — retest {via_label}"
    tip = " · ".join(f"{VARIANT_LABEL[k]} {v['e']:.6g} → {v['o']}" for k, v in (st.get("vars") or {}).items())
    extra = []
    if st.get("disp") is not None:
        extra.append(f"displacement {st['disp']:.2f}×")
    d = retest_depth(st)
    if d is not None:
        extra.append(f"deepest retest {st['rt']:.6g} ({d:.0%} back to the low)")
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
    if st.get("o") in ("ambiguous", "missed", "failed", "no_touch", "no_choch", "open", "low_rr", "rr_wait"):
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
