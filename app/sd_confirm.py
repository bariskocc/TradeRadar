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
MIN_GAP_FRAC = 0.15              # FVG / IFVG en kucuk bosluk: son 20 LTF mumunun ort. boyu x bu (05.11)
GAP_AVG_BARS = 20
VARIANTS = ("fvg", "ifvg", "cisd", "mss")
VARIANT_LABEL = {"fvg": "FVG", "ifvg": "IFVG", "cisd": "CISD", "mss": "MSS level"}
FINAL = ("failed", "missed", "win", "loss", "ambiguous", "no_touch", "no_choch", "open", "week_close")
LABELS = {
    "pre": "—", "wait_choch": "Waiting CHOCH", "wait_retest": "CHOCH — waiting retest", "filled": "Retest — in trade",
    "win": "Worked", "loss": "Zone failed after CHOCH", "ambiguous": "Same bar (?)", "missed": "No retest — price left",
    "failed": "Zone failed before CHOCH", "no_touch": "No retest in horizon", "no_choch": "No CHOCH in horizon",
    "open": "Open at horizon", "week_close": "Closed at week end",
}


def _naive(ts):
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


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
    if st["o"] in FINAL:
        return False
    ts = _naive(bar_ts)
    if ts is None or (st.get("until") is not None and ts <= _naive(st["until"])):
        return False
    st["until"] = ts
    long = rec.get("direction") == "LONG"
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
        st["lvl"] = _swing_before(bars, st["lo_i"], long)
        lvl, ci = st["lvl"], len(bars) - 1
        if lvl is not None and st["lo_i"] < ci and ((close > lvl) if long else (close < lvl)):
            cands = _candidates(bars, st["lo_i"], ci, long, lvl, close, st["lo"])
            ref = abs(lvl - st["lo"])
            sl = st["lo"] - SL_BUFFER_FRAC * ref if long else st["lo"] + SL_BUFFER_FRAC * ref
            st["sl"], st["ct"], st["o"] = float(sl), _iso(ts), "wait_retest"
            st["vars"] = {k: {"e": e, "rr": round(abs(tp - e) / abs(e - sl), 4) if abs(e - sl) > 0 else None,
                              "o": "pending", "t": None, "at": None} for k, e in cands.items()}
            st["b"] = []                          # tampon artik gerekmez (JSON kucuk kalsin)
        elif ts >= rec["levels_at"] + horizon:
            st["o"], st["at"] = "no_choch", ts
        return True
    # CHOCH sonrasi: her aday ayri limit emri
    sl = float(st["sl"])
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
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
    filled = [v["t"] for v in st["vars"].values() if v.get("t")]
    if filled:
        t0 = min(filled)
        via = next(k for k in VARIANTS if st["vars"].get(k, {}).get("t") == t0)
        v = st["vars"][via]
        # ust duzey t/at/until datetime (Journal `_ENTRY_TS` donusumu); adaylarin (vars) zamanlari ISO metin
        st["via"], st["e"], st["rr"], st["t"] = via, v["e"], v["rr"], datetime.fromisoformat(v["t"])
        st["o"] = v["o"] if v["o"] in ("win", "loss", "ambiguous") else "filled"
        if st["o"] != "filled":
            st["at"] = ts
    elif (st["vars"] and all(v["o"] == "missed" for v in st["vars"].values())) or (not st["vars"] and hit_tp):
        st["o"], st["at"] = "missed", ts
    if st["o"] in ("wait_retest", "filled") and ts >= rec["levels_at"] + horizon:
        st["o"], st["at"] = ("no_touch" if st["o"] == "wait_retest" else "open"), ts
    return True


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
    if st.get("ct"):
        parts.append(f"CHOCH {hhmm(st['ct'])}")
    if via:
        parts.append(f"retest {via_label} {hhmm(st.get('t'))}")
    label = LABELS.get(o, o)
    if o == "win" and via_label:
        label = f"Worked — retest {via_label}"
    tip = " · ".join(f"{VARIANT_LABEL[k]} {v['e']:.6g} → {v['o']}" for k, v in (st.get("vars") or {}).items())
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
    if st.get("o") in ("ambiguous", "missed", "failed", "no_touch", "no_choch", "open"):
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
