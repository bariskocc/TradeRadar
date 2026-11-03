"""S/D "LTF onay + retest" giris varyanti (03.11, kullanici) -- OLCUM, motor karari degil.

Soru: bolgeye dokununca dogrudan limit (bugunku kural) mi, yoksa 15m'de CHOCH/MSB bekleyip kirilan seviyenin
retestinde mi girmeli? Ayni Journal satirinda ikisi paralel izlenir; bu modul ikinci kolu yurutur. Durum
`setup_journal` kaydinin `entries["sdc"]` sozlugunde tutulur (`o` = asama; varyant donguleri `_STATE_KEYS` ile atlar).

Akis (LONG icin; SHORT simetrik):
  1. Dokunus: satirin `entry_touched_at`'i (fiyat proximal'e degdi) -> tampon baslar, `o = "wait_choch"`.
  2. Bolgeye gelen dususun son dibi (`lo`) ve o dipten ONCEKI son pivot tepe (k=1; yoksa dipten onceki en yuksek high) =
     CHOCH seviyesi (`lvl`). Dip satirin SL'ini (distal + taban) gecerse bolge basarisiz: `o = "failed"`.
  3. Bir 15m kapanisi `lvl` ustunde -> onay: giris = `lvl` (kirilan yapinin retesti), SL = dip - %10 risk, TP satirin
     TP'si; `o = "wait_retest"`.
  4. Fiyat girise donmeden TP'ye giderse `o = "missed"` (islem yok); girise gelirse `filled`, sonra `win` / `loss`
     (ayni mumda ikisi: `ambiguous`). Ufuk satirin ufku: dolmadan biterse `no_touch`, onay gelmeden biterse `no_choch`.
Not: 1H bolgeleri 03.11'den beri 5m ile izlenir (config.SD_5M_ENABLED, 1D evreni); 5m yoksa 15m.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

SDC_KEY = "sdc"
STRATEGIES = ("sd", "sd1h")
MAX_BUFFER = 400                 # 15m mum (~4 gun); daha eskisi CHOCH icin gerekmez
SL_BUFFER_FRAC = 0.10            # SL = dip - %10 x (seviye - dip)
FINAL = ("failed", "missed", "win", "loss", "ambiguous", "no_touch", "no_choch", "open")
LABELS = {
    "wait_choch": "Waiting CHOCH", "wait_retest": "CHOCH — waiting retest", "filled": "Filled (confirm)",
    "win": "TP", "loss": "SL", "ambiguous": "Same bar (?)", "missed": "Missed (no retest)",
    "failed": "Zone failed before CHOCH", "no_touch": "No retest in horizon", "no_choch": "No CHOCH in horizon",
    "open": "Open at horizon",
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


def _init(rec: dict) -> dict:
    return {"o": "wait_choch", "t0": _iso(rec.get("entry_touched_at")), "b": [], "lo": None, "lo_i": None,
            "lvl": None, "e": None, "sl": None, "rr": None, "ct": None, "t": None, "at": None, "until": None}


def _choch_level(bars: list, lo_i: int, long: bool) -> float | None:
    """Dipten (lo_i) onceki son pivot tepe (k=1); yoksa dipten onceki en uc nokta. Dip ilk mumdaysa None."""
    if lo_i <= 0:
        return None
    hi = [b[1] for b in bars]
    lo = [b[2] for b in bars]
    vals = hi if long else lo
    for i in range(lo_i - 1, 0, -1):
        if long and vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
            return vals[i]
        if not long and vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            return vals[i]
    seg = vals[:lo_i]
    return max(seg) if long else min(seg)


def step(rec: dict, bar_ts, high: float, low: float, close: float, horizon: timedelta) -> bool:
    """Kapanan 15m mumunu uygula. Degisiklik olduysa True."""
    if rec.get("strategy") not in STRATEGIES or rec.get("levels_at") is None or rec.get("entry_touched_at") is None:
        return False
    ents = rec.get("entries")
    if ents is None:
        ents = rec["entries"] = {}
    st = ents.get(SDC_KEY)
    if st is None:
        st = ents[SDC_KEY] = _init(rec)
    if st["o"] in FINAL:
        return False
    ts = _naive(bar_ts)
    if ts is None or ts < _naive(rec["entry_touched_at"]):
        return False
    if st.get("until") is not None and ts <= _naive(st["until"]):
        return False
    st["until"] = ts
    long = rec.get("direction") == "LONG"
    e_row, sl_row, tp = float(rec["entry"]), float(rec["sl"]), float(rec["tp"])
    hit_sl_row = low <= sl_row if long else high >= sl_row
    if st["o"] == "wait_choch":
        bars = st["b"]
        bars.append([_iso(ts), float(high), float(low), float(close)])
        if len(bars) > MAX_BUFFER:
            drop = len(bars) - MAX_BUFFER
            del bars[:drop]
            if st["lo_i"] is not None:
                st["lo_i"] = max(0, st["lo_i"] - drop)
        if hit_sl_row:
            st["o"], st["at"] = "failed", ts
            return True
        ext = low if long else high
        if st["lo"] is None or ((ext < st["lo"]) if long else (ext > st["lo"])):
            st["lo"], st["lo_i"] = float(ext), len(bars) - 1
        st["lvl"] = _choch_level(bars, st["lo_i"], long)
        lvl = st["lvl"]
        if lvl is not None and st["lo_i"] < len(bars) - 1 and ((close > lvl) if long else (close < lvl)):
            risk = abs(lvl - st["lo"])
            if risk <= 0:
                return True
            st["e"] = float(lvl)
            st["sl"] = float(st["lo"] - SL_BUFFER_FRAC * risk) if long else float(st["lo"] + SL_BUFFER_FRAC * risk)
            st["rr"] = round(abs(tp - st["e"]) / abs(st["e"] - st["sl"]), 4)
            st["ct"], st["o"] = _iso(ts), "wait_retest"
            st["b"] = []                         # tampon artik gerekmez (JSON kucuk kalsin)
        elif ts >= rec["levels_at"] + horizon:
            st["o"], st["at"] = "no_choch", ts
        return True
    e, sl = float(st["e"]), float(st["sl"])
    hit_e = low <= e if long else high >= e
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    if st["o"] == "wait_retest":
        if hit_tp and not hit_e:
            st["o"], st["at"] = "missed", ts
        elif hit_e:
            st["t"] = ts
            if hit_tp:
                st["o"], st["at"] = "ambiguous", ts
            elif hit_sl:
                st["o"], st["at"] = "loss", ts
            else:
                st["o"] = "filled"
        elif ts >= rec["levels_at"] + horizon:
            st["o"], st["at"] = "no_touch", ts
        return True
    if st["o"] == "filled":
        if hit_tp and hit_sl:
            st["o"], st["at"] = "ambiguous", ts
        elif hit_tp:
            st["o"], st["at"] = "win", ts
        elif hit_sl:
            st["o"], st["at"] = "loss", ts
        elif ts >= rec["levels_at"] + horizon:
            st["o"], st["at"] = "open", ts
        return True
    return False


def final_r(st: dict | None) -> float | None:
    if not st:
        return None
    if st.get("o") == "win":
        return float(st.get("rr") or 0)
    if st.get("o") == "loss":
        return -1.0
    if st.get("o") in ("ambiguous", "missed", "failed", "no_touch", "no_choch", "open"):
        return 0.0
    return None
