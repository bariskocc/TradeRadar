"""Setup Journal: her setup'in motor kapilarindaki yolu ve elendikten sonra fiyatin ne yaptigi.

Neden: eleme/silme nedenleri yalnizca log satiriydi ve hangi stratejiye ait oldugu yazmiyordu;
"sinyal neden gelmedi" sorusu her seferinde logdan elle eslestiriliyordu. Radar bellekte ve
restart'ta siliniyor.

Kayit anahtari: strateji + sembol + yon + purge (C2) zamani -> setup basina TEK satir.

Akis:
- `scanner._set_radar` setup iceren her durumda `note(...)` cagirir: bellekteki kayit guncellenir.
  DB'ye her degerlendirmede yazilmaz; `flush(session)` degisenleri tek commit'te yazar
  (on_candle_closed / on_price_update / run_scan / reconcile / hafta kapanisi sonunda).
- Silinen pending/waiting: `note_deleted(sig, reason)`.
- Sonuc izleme: sinyale donusmemis ve seviyeli kayit icin kapanan her LTF mumunda `track_bar(...)`:
  fiyat once entry'ye mi, TP'ye mi, SL'ye mi degdi? Seviyeler ILK goruldugu haliyle dondurulur
  (setup sonradan yeniden seviyelenirse izleme kaymasin). Ufuk strateji bazli (HORIZON).
- Restart: `ensure_loaded(session, store)` son KEEP gunun kayitlarini yukler ve izlenen kayitlari
  store'daki kapanmis LTF mumlariyla telafi eder.

Bu modulun hicbir hatasi islem akisini bozmamali: disariya acilan her fonksiyon hatayi yutar ve loglar.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models import SetupJournal

log = logging.getLogger(__name__)

# Motor kapilari, geriden ileriye. Kayit ulastigi EN ILERI asamayi tutar.
STAGES: list[tuple[str, str]] = [
    # detect_crt_setup'in setup'a CEVIRMEDEN eledigi CRT adaylari (14.09) -- en geri kapilar.
    # Yalniz kapanmis C2 ve gercekten delinmis C1 ucu; bkz. crt_engine.detect_crt_setup(rejected=).
    ("sweep_small", "CRT: sweep below threshold"),
    ("range_atr", "CRT: C1 range outside ATR band"),
    ("c1_stale", "CRT: C1 extreme taken earlier"),
    ("c2_breakout", "CRT: C2 closed outside C1"),
    ("c2_wrong_color", "CRT: C2 wrong color"),
    ("not_selected", "CRT: another candidate chosen"),
    ("bias_mismatch", "1D bias opposite"),
    ("target_taken", "Target already taken"),
    ("low_quality", "Score < 7"),
    ("no_cisd", "No entry levels / MSS"),
    ("low_rr", "RR < 2"),
    ("score7", "Score-7 gate"),
    ("tight_stop", "Stop too tight"),
    ("past_tp", "TP before entry"),
    ("past_sl", "SL before fill"),
    ("invalidated", "CRT 60% crossed"),
    ("missed_quality", "Score < 7 at retest"),
    ("stale", "Retest too old"),
    ("cluster_limit", "Cluster limit"),
    ("corr_open", "Correlated pair open"),
    ("has_open", "Open signal exists"),
    ("duplicate", "Already saved"),
    ("c2_open", "Waiting for C2 close"),
    ("waiting", "Signal opened"),
    ("week_close", "Cancelled at week close"),
]
STAGE_LABELS = dict(STAGES)
PRE_SETUP_STAGES = frozenset(("sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "not_selected"))
_RANK = {code: i for i, (code, _) in enumerate(STAGES)}
SIGNAL_STAGE = "waiting"

OUTCOME_LABELS = {
    "pending": "Tracking (no entry yet)",
    "filled": "Tracking (entry touched)",
    "tp_before_entry": "TP before entry",
    "win": "Entry then TP",
    "loss": "Entry then SL",
    "no_touch": "No entry in horizon",
    "open": "Entry, no TP/SL in horizon",
    "ambiguous": "Same bar (order unknown)",
    "signal": "Became a signal",
}
TRACKING = ("pending", "filled")
HORIZON = {"4h": timedelta(hours=48), "1d": timedelta(hours=120), "1h": timedelta(hours=12)}
LTF_OF = {"4h": "15m", "1d": "1h", "1h": "5m"}
KEEP = timedelta(days=10)
# Yalnizca last_seen ilerlediyse bu kadar bekleyip yaz (her 15 dk'da her satiri yazmamak icin).
_TOUCH_EVERY = timedelta(minutes=30)

# Golge izleme (14.09): "stop daha yakin olsaydi?" sorusu icin geriye donuk replay yerine canli
# veri (once 1H, ayni gun 4H/1D). Seviyeleri olan her setup, SL = entry + k x (SL - entry) ile
# ayrica izlenir; k=1 bugunku kural (sinyale donusen setup'ta ana izleme durdugu icin temel cizgi
# burada tutulur). Duz TP/SL: 4H/1D'deki kismi kar + BE yok. Degerlendirme ve karar kurali:
# IZLEME.md "Dar stop — golge izleme".
SHADOW_SL = {"4h": (1.0, 0.75, 0.5), "1d": (1.0, 0.75, 0.5), "1h": (1.0, 0.75, 0.5)}
# JSON'a yazilirken/okunurken datetime'a cevrilecek alanlar. Golgede "e" dolum ani; entry
# varyantlarinda "e" ENTRY FIYATI olduğu icin dolum ani ayri alanda ("t") tutulur.
_SHADOW_TS = ("e", "at", "until")
_ENTRY_TS = ("t", "at", "until")

# C1 varyanti (17.09): sabit kesir degil, setup'in KENDI C1 ucu (LONG'da C1 low, SHORT'ta C1 high).
# Soru: "SL purge ucu yerine C1 ucu olsa?" TP zaten karsi C1 ucu oldugu icin bu, islemi saf range
# trade'ine cevirir; k setup basina degisir (olculen medyan ~0.64). Her zaman izlenmez:
#   wrong_side  -> entry C1'in disinda kalmis (CISD/MSS seviyesi purge bolgesinde), stop gecersiz
#   not_tighter -> C1 ucu purge ucundan uzak; daraltma degil, genisletme olurdu
# Bu iki durumda varyant "invalid" olarak yazilir (izlenmez) -- kuralin ne siklikta uygulanamadigi
# da olcumun parcasi. Asiri dar C1 stopu (dar stop kapisinin elecegi) izlenir ama `mult` ile
# isaretlenir: k x stop_range_mult < 1.0 ise motor bu setup'i zaten "tight_stop" diye elerdi.
# Degerlendirme ve karar kurali: IZLEME.md "C1 ucu stop — golge izleme".
SHADOW_C1 = "c1"

# Entry modeli karsilastirmasi (18.09). AYNI setupta butun aday entry'ler ayri izlenir:
# SL (purge ucu) ve TP (karsi C1 ucu) SABIT, yalniz entry degisir. Boylece "hangi model daha
# iyi" sorusu secilim etkisinden arinir -- bugunku olcum farkli setuplari kiyasliyordu
# (IFVG yalniz CISD adayi kotuyken seciliyor, cunku bolgenin EN KOTU RR'li noktasiyla yarisiyor).
# Varyantlar scanner tarafindan hesaplanip `note(entries=...)` ile dondurulur:
#   cisd / mss                -> iki yapisal aday (motor yalnizca kazanani tutuyordu)
#   ifvg_near / mid / far     -> IFVG bolgesinin ust-orta-alt noktasi (LONG'da near = ust)
#   bpr_near / mid / far      -> BPR (iki FVG kesisimi) bolgesinin ayni uc noktasi (21.09)
#   dfvg_near / mid / far     -> kirilim sonrasi birakilan FVG (yeni model)
#   chosen                    -> motorun fiilen sectigi entry (temel cizgi)
# Entry SL-TP arasinda degilse varyant "invalid" yazilir (izlenmez) -- ne siklikta
# uygulanamadigi da olcumun parcasi. Karar kurali: IZLEME.md "Entry modeli karsilastirmasi".
ENTRY_VARIANTS = ("chosen", "cisd", "mss",
                  "ifvg_near", "ifvg_mid", "ifvg_far",
                  "bpr_near", "bpr_mid", "bpr_far",
                  "dfvg_near", "dfvg_mid", "dfvg_far")

_CACHE: dict[tuple, dict] = {}
_DIRTY: set[tuple] = set()
_STATE = {"loaded": False}


def _shadow_init(rec: dict) -> dict | None:
    fractions = SHADOW_SL.get(rec.get("strategy"))
    e, sl, tp = rec.get("entry"), rec.get("sl"), rec.get("tp")
    if not fractions or None in (e, sl, tp) or e == sl:
        return None
    risk = abs(e - sl)
    out = {
        f"{k:g}": {"k": k, "sl": e + k * (sl - e), "rr": round(abs(tp - e) / (k * risk), 4),
                   "o": "pending", "e": None, "at": None, "until": None}
        for k in fractions
    }
    c1 = _shadow_c1(rec, e, sl, tp, risk)
    if c1 is not None:
        out[SHADOW_C1] = c1
    return out


def _shadow_c1(rec: dict, e: float, sl: float, tp: float, risk: float) -> dict | None:
    """C1 ucu stop varyanti: sabit kesir yerine setup'in kendi C1 seviyesi (yukaridaki nota bkz.)."""
    c1 = rec.get("c1")
    if c1 is None or not risk:
        return None
    c1 = float(c1)
    base = {"k": None, "sl": c1, "rr": None, "o": "invalid", "e": None, "at": None, "until": None}
    long = rec.get("direction") == "LONG"
    if (c1 >= e) if long else (c1 <= e):
        base["why"] = "wrong_side"     # entry C1'in disinda: C1 stop entry'nin yanlis tarafinda
        return base
    if (c1 <= sl) if long else (c1 >= sl):
        base["why"] = "not_tighter"    # C1 ucu purge ucundan uzak: daraltmiyor
        return base
    k = abs(e - c1) / risk
    base.update(k=round(k, 4), rr=round(abs(tp - e) / abs(e - c1), 4), o="pending")
    mult = ((rec.get("features_at_levels") or rec.get("features") or {}) or {}).get("stop_range_mult")
    if mult:
        # Dar stop kapisinin bu varyanttaki karsiligi: motor 1.0'in altini "tight_stop" diye eler.
        base["mult"] = round(k * float(mult), 4)
    return base


def _entries_init(rec: dict) -> dict | None:
    """Aday entry'leri dondur: her biri kendi RR'siyle, SL/TP ortak."""
    cands = dict(rec.get("entry_cands") or {})
    ref = cands.pop("_ref", None)          # seviye dondugu andaki fiyat (son LTF kapanisi)
    sl, tp = rec.get("sl"), rec.get("tp")
    if not cands or sl is None or tp is None:
        return None
    long = rec.get("direction") == "LONG"
    out: dict = {}
    for name in ENTRY_VARIANTS:
        e = cands.get(name)
        if e is None:
            continue
        e = float(e)
        v = {"e": e, "rr": None, "o": "invalid", "t": None, "at": None, "until": None}
        if not ((sl < e < tp) if long else (tp < e < sl)):
            v["why"] = "out_of_range"      # entry SL-TP disinda: limit emri anlamsiz
        else:
            v["rr"] = round(abs(tp - e) / abs(e - sl), 4)
            v["o"] = "pending"
            # Fiyat seviyeyi ZATEN gecmisse limit emri aninda dolardi -- "fiyat geri geldi mi"
            # sorusunun cevabi degildir ve sig varyantlari haksiz yere avantajli gosterir.
            # Izlemeye devam edilir (islem gerceklesirdi) ama isaretlenir; rapor ayirir.
            if ref is not None and ((e >= float(ref)) if long else (e <= float(ref))):
                v["imm"] = True
        out[name] = v
    return out or None


def _retrace_init(rec: dict) -> dict | None:
    """Geri cekilme olcegi: 0.0 = seviye dondugu andaki fiyat (ref), 1.0 = SL.

    Aday entry'ler de bu olcege tasinir (`d_of`) -- "IFVG ust kenari genelde d=0.35'te" gibi
    okunabilsin ve sonradan akla gelen seviyeler de ayni egriden cevap alsin.
    """
    cands = rec.get("entry_cands") or {}
    ref, sl = cands.get("_ref"), rec.get("sl")
    if ref is None or sl is None:
        return None
    ref, sl = float(ref), float(sl)
    span = (ref - sl) if rec.get("direction") == "LONG" else (sl - ref)
    if span <= 0:
        return None                      # fiyat zaten SL'nin otesinde: olcek tanimsiz
    d_of = {}
    for name in ENTRY_VARIANTS:
        e = cands.get(name)
        if e is None:
            continue
        d_of[name] = round(
            ((ref - float(e)) if rec.get("direction") == "LONG" else (float(e) - ref)) / span, 4)
    return {"ref": round(ref, 8), "span": round(span, 8), "d_max": 0.0, "d_tp": 0.0,
            "d_tp_bar": None, "tp_first": None, "d_of": d_of, "done": False, "until": None}


def _retrace_active(rec: dict, bar_ts=None) -> bool:
    r = rec.get("retrace")
    if not r or r.get("done") or rec.get("levels_at") is None:
        return False
    if bar_ts is not None:
        if bar_ts < rec["levels_at"]:
            return False
        if r.get("until") is not None and bar_ts <= r["until"]:
            return False
    return True


def _apply_retrace(rec: dict, bar_ts, high: float, low: float) -> None:
    """Mum basina derinlik guncellemesi; TP/SL ile sonuclanir."""
    r = rec["retrace"]
    long = rec["direction"] == "LONG"
    sl, tp, ref, span = rec["sl"], rec["tp"], r["ref"], r["span"]
    r["until"] = bar_ts
    ext = low if long else high                       # aleyhte ucta ne kadar geri gelindi
    d_bar = ((ref - ext) if long else (ext - ref)) / span
    d_bar = max(0.0, d_bar)
    if d_bar > r["d_max"]:
        r["d_max"] = round(d_bar, 4)
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    if hit_tp and hit_sl:
        # Ayni mumda ikisi birden: sira bilinmiyor, sonuc yazilmaz (rapor disarida birakir).
        r["done"], r["amb"] = True, True
        return
    if hit_tp:
        # TP mumunun dibi d_tp'ye KATILMAZ: mum ici sira bilinmedigi icin o derinlikteki bir
        # emrin TP'den once dolup dolmadigi belirsiz. Duyarlilik icin ayri alanda saklanir.
        r["d_tp_bar"] = round(d_bar, 4)
        r["tp_first"], r["done"] = True, True
        return
    if d_bar > r["d_tp"]:
        r["d_tp"] = round(d_bar, 4)
    if hit_sl:
        r["tp_first"], r["done"] = False, True
        return
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    if bar_ts >= rec["levels_at"] + horizon:
        r["done"] = True                              # tp_first None: sonuclanmadi


# ──────────────────── Dolum oncesi kosu (18.09) ────────────────────
# Soru: emir dolmadan ONCE fiyat TP yolunun ne kadarini yurumustu, ve bu oran
# sonucla ilgili mi? Hipotez (ENA 4H #71, 17.09): hareket emir girilmeden once
# olursa geri donen fiyat entry'yi doldurur ama kalan yol tukendigi icin SL'e
# gider. `target_taken` kapisi bunu YAKALAMAZ -- o kapi tam TP'nin tuketilmesine
# bakar; %50-%100 arasinda duran hareket kapidan gecer.
#
# Olcek: 0.0 = entry, 1.0 = TP. Kaydedilen `f_max` dolum mumundan ONCEKI mumlarin
# en iyisi; dolum mumunun kendi kar yonu AYRI alanda (`f_bar`), cunku mum ici sira
# bilinmiyor -- o hareket dolumdan once mi sonra mi belirsiz (scanner'in
# `on_fill_bar` disiplininin aynisi).
#
# ⚠️ Journal seviyeleri ILK gorulen halleriyle dondurur; motor entry'yi sonradan
# tasidiysa `f_max` o ilk seviyelere gore olculur. Istatistik icin tanim tutarli,
# ama tek bir gercek islemin sayisini birebir vermez.

def _prefill_init(rec: dict) -> dict | None:
    e, tp = rec.get("entry"), rec.get("tp")
    if e is None or tp is None:
        return None
    e, tp = float(e), float(tp)
    path = (tp - e) if rec.get("direction") == "LONG" else (e - tp)
    if path <= 0:
        return None
    return {"path": round(path, 8), "f_max": 0.0, "f_bar": None, "bars": 0,
            "hit": None, "done": False, "until": None}


def _prefill_active(rec: dict, bar_ts=None) -> bool:
    """Sinyale donusen setupta da surer: outcome 'signal' olunca durmamali."""
    p = rec.get("prefill")
    if not p or p.get("done") or rec.get("levels_at") is None:
        return False
    if bar_ts is not None:
        if bar_ts < rec["levels_at"]:
            return False
        if p.get("until") is not None and bar_ts <= p["until"]:
            return False
    return True


def _apply_prefill(rec: dict, bar_ts, high: float, low: float) -> None:
    p = rec["prefill"]
    long = rec["direction"] == "LONG"
    e, path = float(rec["entry"]), p["path"]
    p["until"] = bar_ts
    ext = high if long else low                      # kar yonundeki uc
    f_bar = max(0.0, ((ext - e) if long else (e - ext)) / path)
    hit_e = low <= e if long else high >= e
    if hit_e:
        # Dolum mumu f_max'a KATILMAZ (mum ici sira bilinmiyor); ayri alanda saklanir.
        p["f_bar"] = round(f_bar, 4)
        p["hit"] = bar_ts
        p["done"] = True
        return
    if f_bar > p["f_max"]:
        p["f_max"] = round(f_bar, 4)
    p["bars"] += 1
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    if bar_ts >= rec["levels_at"] + horizon:
        p["done"] = True                             # hit None: emir hic dolmadi


def _prefill_dump(p: dict | None) -> str | None:
    if not p:
        return None
    out = dict(p)
    for k in ("until", "hit"):
        if out.get(k) is not None:
            out[k] = out[k].isoformat()
    return json.dumps(out, separators=(",", ":"))


def _prefill_load(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        d = json.loads(raw)
        for k in ("until", "hit"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
        return d
    except Exception:
        return None


def _entries_active(rec: dict, bar_ts=None) -> bool:
    ents = rec.get("entries")
    if not ents or rec.get("levels_at") is None:
        return False
    if bar_ts is not None and bar_ts < rec["levels_at"]:
        return False
    return any(
        v["o"] in TRACKING and (bar_ts is None or v.get("until") is None or bar_ts > v["until"])
        for v in ents.values()
    )


def _apply_entries(rec: dict, bar_ts, high: float, low: float) -> None:
    """_apply_bar'in ayni kurallari, her ENTRY varyanti icin ayri (SL/TP ortak).

    Sorunun ta kendisi: "bu seviyeye limit emri koysaydik dolar miydi, sonra ne olurdu?"
    """
    long = rec["direction"] == "LONG"
    sl, tp = rec["sl"], rec["tp"]
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    for v in rec["entries"].values():
        if v["o"] not in TRACKING or (v.get("until") is not None and bar_ts <= v["until"]):
            continue
        e = v["e"]
        hit_e = low <= e if long else high >= e
        v["until"] = bar_ts
        if v["o"] == "pending":
            if hit_tp and not hit_e:
                v["o"], v["at"] = "tp_before_entry", bar_ts
            elif hit_e:
                v["t"] = bar_ts                      # emir doldu
                if hit_tp:
                    v["o"], v["at"] = "ambiguous", bar_ts
                elif hit_sl:
                    v["o"], v["at"] = "loss", bar_ts
                else:
                    v["o"] = "filled"
        elif v["o"] == "filled":
            if hit_tp and hit_sl:
                v["o"], v["at"] = "ambiguous", bar_ts
            elif hit_tp:
                v["o"], v["at"] = "win", bar_ts
            elif hit_sl:
                v["o"], v["at"] = "loss", bar_ts
        if v["o"] in TRACKING and bar_ts >= rec["levels_at"] + horizon:
            v["o"], v["at"] = ("no_touch" if v["o"] == "pending" else "open"), bar_ts


def _variants_dump(data: dict | None, ts_fields: tuple) -> str | None:
    if not data:
        return None
    out = {
        name: {f: (v[f].isoformat() if f in ts_fields and v.get(f) is not None else v.get(f)) for f in v}
        for name, v in data.items()
    }
    return json.dumps(out, separators=(",", ":"))


def _variants_load(raw: str | None, ts_fields: tuple) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        for v in data.values():
            for f in ts_fields:
                if v.get(f):
                    v[f] = datetime.fromisoformat(v[f])
        return data
    except Exception:
        return None


def _shadow_dump(shadow: dict | None) -> str | None:
    return _variants_dump(shadow, _SHADOW_TS)


def _retrace_dump(r: dict | None) -> str | None:
    if not r:
        return None
    out = dict(r)
    if out.get("until") is not None:
        out["until"] = out["until"].isoformat()
    return json.dumps(out, separators=(",", ":"))


def _retrace_load(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        d = json.loads(raw)
        if d.get("until"):
            d["until"] = datetime.fromisoformat(d["until"])
        return d
    except Exception:
        return None


def _entries_dump(entries: dict | None) -> str | None:
    return _variants_dump(entries, _ENTRY_TS)


def _entries_load(raw: str | None) -> dict | None:
    return _variants_load(raw, _ENTRY_TS)


def _shadow_load(raw: str | None) -> dict | None:
    return _variants_load(raw, _SHADOW_TS)


def _parts_dump(parts: dict | None) -> str | None:
    """Skor kirilimini JSON'a cevir (kalemleri int tut, dosya kucuk kalsin)."""
    if not parts:
        return None
    try:
        return json.dumps({k: int(v) for k, v in parts.items()}, separators=(",", ":"))
    except Exception:
        return None


def _json_dump(data: dict | None) -> str | None:
    """Olcum sozlugunu JSON'a cevir (float oranlar oldugu gibi kalir)."""
    if not data:
        return None
    try:
        return json.dumps(data, separators=(",", ":"))
    except Exception:
        return None


def _parts_load(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _shadow_active(rec: dict, bar_ts=None) -> bool:
    shadow = rec.get("shadow")
    if not shadow or rec.get("levels_at") is None:
        return False
    if bar_ts is not None and bar_ts < rec["levels_at"]:
        return False
    return any(
        v["o"] in TRACKING and (bar_ts is None or v.get("until") is None or bar_ts > v["until"])
        for v in shadow.values()
    )


def _apply_shadow(rec: dict, bar_ts, high: float, low: float) -> None:
    """_apply_bar'in ayni kurallari, her SL varyanti icin ayri."""
    long = rec["direction"] == "LONG"
    e, tp = rec["entry"], rec["tp"]
    hit_e = low <= e if long else high >= e
    hit_tp = high >= tp if long else low <= tp
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    for v in rec["shadow"].values():
        if v["o"] not in TRACKING or (v.get("until") is not None and bar_ts <= v["until"]):
            continue
        hit_sl = low <= v["sl"] if long else high >= v["sl"]
        v["until"] = bar_ts
        if v["o"] == "pending":
            if hit_tp and not hit_e:
                v["o"], v["at"] = "tp_before_entry", bar_ts
            elif hit_e:
                v["e"] = bar_ts
                if hit_tp:
                    v["o"], v["at"] = "ambiguous", bar_ts
                elif hit_sl:
                    v["o"], v["at"] = "loss", bar_ts
                else:
                    v["o"] = "filled"
        elif v["o"] == "filled":
            if hit_tp and hit_sl:
                v["o"], v["at"] = "ambiguous", bar_ts
            elif hit_tp:
                v["o"], v["at"] = "win", bar_ts
            elif hit_sl:
                v["o"], v["at"] = "loss", bar_ts
        if v["o"] in TRACKING and bar_ts >= rec["levels_at"] + horizon:
            v["o"], v["at"] = ("no_touch" if v["o"] == "pending" else "open"), bar_ts

_COLUMNS = (
    "market_type", "crt_bar_time", "first_seen", "last_seen", "last_stage", "best_stage", "best_stage_at",
    "detail", "score", "htf_bias", "weekly_bias", "c2_closed", "model", "entry", "sl", "tp", "rr", "levels_at",
    "deleted_reason", "deleted_at", "outcome", "outcome_at", "entry_touched_at", "tracked_until",
)


def _naive_utc(ts):
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _now(now=None):
    return _naive_utc(now) if now is not None else datetime.now(timezone.utc).replace(tzinfo=None)


def _key(strategy, symbol, direction, purge_time) -> tuple:
    return (strategy or "4h", symbol, direction, _naive_utc(purge_time))


def stage_rank(stage: str | None) -> int:
    return _RANK.get(stage or "", -1)


def _detail(score, rr, bias, weekly, c2_closed) -> str | None:
    parts = []
    if score is not None:
        parts.append(f"score {int(score)}")
    if rr is not None:
        parts.append(f"RR {float(rr):.2f}")
    if bias:
        parts.append(f"1D {bias}")
    if weekly:
        parts.append(f"1W {weekly}")
    if c2_closed is False:
        parts.append("C2 open")
    return " · ".join(parts) or None


def note(
    strategy: str,
    symbol: str,
    market: str | None,
    stage: str,
    *,
    direction: str | None = None,
    purge_time=None,
    crt_bar_time=None,
    score=None,
    bias: str | None = None,
    weekly_bias: str | None = None,
    rr=None,
    entry=None,
    sl=None,
    tp=None,
    c2_closed: bool | None = None,
    model: str | None = None,
    parts: dict | None = None,
    features: dict | None = None,
    c1=None,
    entries: dict | None = None,
    now=None,
) -> None:
    """Bir degerlendirmenin sonucunu kayda isle (yalnizca bellek).

    `c1`: setup'in yakin C1 ucu (LONG'da key_level_low, SHORT'ta key_level_high) -- yalniz
    golge izlemenin C1 varyanti icin; motor karari kullanmaz, DB'de ayri kolonu yok
    (shadow JSON'undaki varyantin `sl`'i olarak durur).
    """
    try:
        if direction is None or purge_time is None or stage not in _RANK:
            return
        now = _now(now)
        key = _key(strategy, symbol, direction, purge_time)
        rec = _CACHE.get(key)
        changed = False
        if rec is None:
            rec = {"id": None, "strategy": key[0], "symbol": symbol, "direction": direction,
                   "purge_time": key[3], "first_seen": now, "best_stage": stage, "best_stage_at": now,
                   "outcome": None, "_flushed_seen": None}
            _CACHE[key] = rec
            changed = True
        rec["last_seen"] = now
        if (
            stage not in PRE_SETUP_STAGES
            and rec.get("best_stage") in PRE_SETUP_STAGES
            and rec.get("levels_at") is not None
        ):
            # Ayni anahtar (yon + C2) once elenen aday olarak -- belki farkli C1 ile -- kaydedildi;
            # artik motorun setup'i: izleme onun seviyeleriyle yeniden baslar.
            for k in ("entry", "sl", "tp", "rr", "levels_at", "outcome", "outcome_at",
                      "entry_touched_at", "tracked_until", "shadow", "entries", "entry_cands",
                      "retrace", "prefill", "parts_at_levels", "features_at_levels"):
                rec[k] = None
            changed = True
        if market:
            rec["market_type"] = market
        if crt_bar_time is not None:
            rec["crt_bar_time"] = _naive_utc(crt_bar_time)
        if rec.get("last_stage") != stage:
            rec["last_stage"] = stage
            changed = True
        if stage_rank(stage) > stage_rank(rec.get("best_stage")):
            rec["best_stage"], rec["best_stage_at"] = stage, now
            changed = True
        for k, v in (("score", score), ("htf_bias", bias), ("weekly_bias", weekly_bias),
                     ("c2_closed", c2_closed), ("model", model)):
            if v is not None and rec.get(k) != v:
                rec[k] = v
                changed = True
        # Skor kirilimi (olcum): en guncel hali tutulur -- C2 kapanisi / SMT skoru degistirebilir.
        if parts and rec.get("score_parts") != parts:
            rec["score_parts"] = dict(parts)
            changed = True
        if features:
            # Olcumler birikimli: motor tarafi (wick/IFVG/ATR) ile seviye tarafi (dar stop)
            # ayri cagrilardan gelebiliyor; gelen deger eskisini ezer, eksikler korunur.
            merged = dict(rec.get("features") or {})
            merged.update({k: v for k, v in features.items() if v is not None})
            if merged != rec.get("features"):
                rec["features"] = merged
                changed = True
        if entry is not None and sl is not None and tp is not None and rec.get("levels_at") is None:
            # Ilk gorulen seviyeler dondurulur: sonuc izleme bunlarla yapilir.
            rec.update(entry=float(entry), sl=float(sl), tp=float(tp),
                       rr=float(rr) if rr is not None else None, levels_at=now,
                       c1=float(c1) if c1 is not None else None,
                       entry_cands=dict(entries) if entries else None)
            rec["shadow"] = _shadow_init(rec)
            rec["entries"] = _entries_init(rec)
            rec["retrace"] = _retrace_init(rec)
            rec["prefill"] = _prefill_init(rec)
            # Skor ve olcumler de AYNI AN dondurulur: sonuc bu seviyelerden izlendigi icin,
            # sonradan degisen skor (C2 kapanisi, SMT) ile sonucu eslestirmek yaniltici olur.
            rec["parts_at_levels"] = dict(rec.get("score_parts") or {}) or None
            rec["features_at_levels"] = dict(rec.get("features") or {}) or None
            changed = True
        detail = _detail(score, rr, bias, weekly_bias, c2_closed)
        # Skor/RR tasimayan bir degerlendirme (or. seviyesi bir kez hesaplanan elenen aday)
        # onceki skorlu detayi ezmesin.
        if detail and rec.get("detail") != detail and (score is not None or rr is not None or not rec.get("detail")):
            rec["detail"] = detail
            changed = True
        if stage == SIGNAL_STAGE and rec.get("outcome") != "signal":
            rec["outcome"] = "signal"
            changed = True
        elif rec.get("levels_at") is not None and rec.get("outcome") is None:
            rec["outcome"] = "pending"
            changed = True
        seen = rec.get("_flushed_seen")
        if changed or seen is None or now - seen >= _TOUCH_EVERY:
            _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.note failed for %s %s", strategy, symbol)


def note_deleted(sig, reason: str, now=None) -> None:
    """Silinen pending/waiting sinyali kayda isle."""
    try:
        if sig is None or getattr(sig, "purge_time", None) is None:
            return
        now = _now(now)
        strategy = getattr(sig, "timeframe", None) or "4h"
        key = _key(strategy, sig.symbol, sig.direction, sig.purge_time)
        if key not in _CACHE:
            stage = SIGNAL_STAGE if sig.status in ("waiting_entry", "active") else "c2_open"
            note(strategy, sig.symbol, sig.market_type, stage, direction=sig.direction, purge_time=sig.purge_time,
                 crt_bar_time=sig.crt_bar_time, score=sig.bias_score, bias=sig.htf_bias,
                 weekly_bias=sig.weekly_bias, rr=sig.planned_rr, entry=sig.entry_price, sl=sig.stop_loss,
                 tp=sig.take_profit, c2_closed=sig.c2_closed, model=sig.entry_model,
                 c1=(sig.key_level_low if sig.direction == "LONG" else sig.key_level_high), now=now)
        rec = _CACHE.get(key)
        if rec is None:
            return
        rec["deleted_reason"], rec["deleted_at"] = reason, now
        _resume_tracking(rec, now)
        _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.note_deleted failed for %s", getattr(sig, "symbol", "?"))


def _resume_tracking(rec: dict, now) -> None:
    """Silinen `waiting` setup'in SONRASINI olcmeye devam et (19.09, kullanici sorusu).

    `waiting`e ulasan setupta `note` outcome'u "signal" yapip izlemeyi durduruyordu; setup
    sonradan silinince (`missed_quality`, `low_rr`, `stale`, `crt_gone`...) izleme geri
    acilmiyor ve o kapinin maliyeti olculemiyordu -- diger butun kapilarda tutulan defter
    (entry'ye deldi mi -> TP mi SL mi) yalniz burada eksikti. Ilk olcum: 6 vakanin 4'u RR >= 2.

    Silme ani limit emrinin kalktigi andir. Aktif sinyal silinmez (bkz. scanner._note_deleted
    cagri noktalari; hafta kapanisinda `sig.status != "active"` suzgeci var), yani emir hic
    dolmamistir -> izleme "pending"den devam eder ve dolum bilgisi sifirlanir.
    Ufuk DEGISMEZ (`levels_at` + HORIZON) ki sonuc diger kapilarla ayni cetvelde okunsun.
    Ayri bir outcome degerine gerek yok: bu satirlar `best_stage='waiting'` + dolu
    `deleted_reason` ile ayirt edilir (sinyale donusup yasayanlarda `deleted_reason` NULL).
    """
    if rec.get("outcome") != "signal" or rec.get("levels_at") is None or now is None:
        return
    rec["outcome"], rec["outcome_at"], rec["entry_touched_at"] = "pending", None, None
    # Silme anindan ONCE acilmis mum sayilmaz: mum ici sira bilinmedigi icin emrin hala canli
    # oldugu bolumu yeni olcume katmak yanlis olur (scanner'daki `on_fill_bar` disiplini).
    rec["tracked_until"] = now


def has_levels(strategy: str, symbol: str, direction: str, purge_time) -> bool:
    """Bu anahtarin (bellekte) dondurulmus seviyeleri var mi? Elenen aday icin tekrar hesaplamamak."""
    try:
        rec = _CACHE.get(_key(strategy, symbol, direction, purge_time))
        return bool(rec and rec.get("levels_at") is not None)
    except Exception:
        return False


def _final(rec: dict, outcome: str, ts) -> None:
    rec["outcome"], rec["outcome_at"] = outcome, ts


def _apply_bar(rec: dict, bar_ts, high: float, low: float) -> None:
    long = rec["direction"] == "LONG"
    e, sl, tp = rec["entry"], rec["sl"], rec["tp"]
    hit_e = low <= e if long else high >= e
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    rec["tracked_until"] = bar_ts
    if rec["outcome"] == "pending":
        if hit_tp and not hit_e:
            _final(rec, "tp_before_entry", bar_ts)
        elif hit_e:
            rec["entry_touched_at"] = bar_ts
            if hit_tp:
                # Ayni mumda entry ve TP: TP once mi (dolmazdi) sonra mi (kazanc) bilinmez.
                _final(rec, "ambiguous", bar_ts)
            elif hit_sl:
                # SL entry'nin otesinde; fiyat entry tarafindan geldigi icin once entry dolar, sonra SL.
                _final(rec, "loss", bar_ts)
            else:
                rec["outcome"] = "filled"
    elif rec["outcome"] == "filled":
        if hit_tp and hit_sl:
            _final(rec, "ambiguous", bar_ts)
        elif hit_tp:
            _final(rec, "win", bar_ts)
        elif hit_sl:
            _final(rec, "loss", bar_ts)
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    if rec["outcome"] in TRACKING and bar_ts >= rec["levels_at"] + horizon:
        _final(rec, "no_touch" if rec["outcome"] == "pending" else "open", bar_ts)


def _is_tracking(rec: dict, bar_ts) -> bool:
    if rec.get("outcome") not in TRACKING or rec.get("levels_at") is None:
        return False
    if bar_ts < rec["levels_at"]:
        return False
    return rec.get("tracked_until") is None or bar_ts > rec["tracked_until"]


def track_bar(strategy: str, symbol: str, bar_ts, high: float, low: float) -> None:
    """Kapanan LTF mumunu, o sembol+stratejinin izlenen kayitlarina uygula."""
    try:
        bar_ts = _naive_utc(bar_ts)
        for key, rec in _CACHE.items():
            if key[0] != strategy or key[1] != symbol:
                continue
            touched = False
            if _is_tracking(rec, bar_ts):
                _apply_bar(rec, bar_ts, float(high), float(low))
                touched = True
            if _shadow_active(rec, bar_ts):
                _apply_shadow(rec, bar_ts, float(high), float(low))
                touched = True
            if _entries_active(rec, bar_ts):
                _apply_entries(rec, bar_ts, float(high), float(low))
                touched = True
            if _retrace_active(rec, bar_ts):
                _apply_retrace(rec, bar_ts, float(high), float(low))
                touched = True
            if _prefill_active(rec, bar_ts):
                _apply_prefill(rec, bar_ts, float(high), float(low))
                touched = True
            if touched:
                _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.track_bar failed for %s %s", strategy, symbol)


def _merge_row(rec: dict, row: SetupJournal) -> None:
    """DB satiri ile (restart oncesi) bellekteki yeni kaydi birlestir."""
    rec["id"] = row.id
    if row.first_seen and (rec.get("first_seen") is None or row.first_seen < rec["first_seen"]):
        rec["first_seen"] = row.first_seen
    if stage_rank(row.best_stage) > stage_rank(rec.get("best_stage")):
        rec["best_stage"], rec["best_stage_at"] = row.best_stage, row.best_stage_at
    if row.levels_at is not None and (rec.get("levels_at") is None or row.levels_at <= rec["levels_at"]):
        for k in ("entry", "sl", "tp", "rr", "levels_at", "outcome", "outcome_at", "entry_touched_at",
                  "tracked_until"):
            rec[k] = getattr(row, k)
        rec["shadow"] = _shadow_load(row.shadow) or _shadow_init(rec)
        rec["entries"] = _entries_load(row.entries) or rec.get("entries")
        rec["retrace"] = _retrace_load(row.retrace) or rec.get("retrace")
        rec["prefill"] = _prefill_load(row.prefill) or rec.get("prefill")
    for k in ("deleted_reason", "deleted_at", "crt_bar_time", "market_type"):
        if rec.get(k) is None and getattr(row, k) is not None:
            rec[k] = getattr(row, k)
    for k, raw in (("score_parts", row.score_parts), ("features", row.features),
                   ("parts_at_levels", row.parts_at_levels),
                   ("features_at_levels", row.features_at_levels)):
        if rec.get(k) is None:
            rec[k] = _parts_load(raw)
    if rec.get("outcome") != "signal" and row.outcome == "signal":
        rec["outcome"] = "signal"
    # Silinmis satirda izleme geri acilir (`_resume_tracking`). Yukaridaki iki dal da DB'deki
    # eski "signal" damgasini geri yazabiliyor (seviye dali outcome'u oldugu gibi kopyalar),
    # bu da restart'ta izlemeyi sessizce ikinci kez durdururdu; silme bilgisi kopyalandiktan
    # SONRA yeniden uygulanir. Cozulmus satira dokunmaz (yalniz outcome == "signal" iken calisir).
    if rec.get("deleted_reason") is not None:
        _resume_tracking(rec, rec.get("deleted_at"))


async def ensure_loaded(session, store=None, symbol_resolver=None) -> None:
    """Ilk cagrida son KEEP gunun kayitlarini yukle, izlenenleri store mumlariyla telafi et."""
    if _STATE["loaded"]:
        return
    _STATE["loaded"] = True
    try:
        since = _now() - KEEP
        rows = (await session.execute(select(SetupJournal).where(SetupJournal.last_seen >= since))).scalars().all()
        for row in rows:
            key = _key(row.strategy, row.symbol, row.direction, row.purge_time)
            rec = _CACHE.get(key)
            if rec is None:
                rec = {c: getattr(row, c) for c in _COLUMNS}
                rec.update(id=row.id, strategy=row.strategy, symbol=row.symbol, direction=row.direction,
                           purge_time=row.purge_time, _flushed_seen=row.last_seen)
                rec["shadow"] = _shadow_load(row.shadow)
                rec["entries"] = _entries_load(row.entries)
                rec["retrace"] = _retrace_load(row.retrace)
                rec["prefill"] = _prefill_load(row.prefill)
                rec["score_parts"] = _parts_load(row.score_parts)
                rec["features"] = _parts_load(row.features)
                rec["parts_at_levels"] = _parts_load(row.parts_at_levels)
                rec["features_at_levels"] = _parts_load(row.features_at_levels)
                _CACHE[key] = rec
            else:
                _merge_row(rec, row)
                _DIRTY.add(key)
        caught = 0
        if store is not None and symbol_resolver is not None:
            for key, rec in _CACHE.items():
                if rec.get("levels_at") is None or (
                    rec.get("outcome") not in TRACKING
                    and not _shadow_active(rec) and not _entries_active(rec)
                    and not _retrace_active(rec) and not _prefill_active(rec)
                ):
                    continue
                try:
                    bingx = symbol_resolver(rec["symbol"])
                    df = store.get_df(bingx, LTF_OF.get(rec["strategy"], "15m"))
                except Exception:
                    continue
                if df is None or len(df) < 2:
                    continue
                for ts, bar in df.iloc[:-1].iterrows():      # son satir forming
                    bar_ts = _naive_utc(ts)
                    hit = False
                    if _is_tracking(rec, bar_ts):
                        _apply_bar(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if _shadow_active(rec, bar_ts):
                        _apply_shadow(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if _entries_active(rec, bar_ts):
                        _apply_entries(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if _retrace_active(rec, bar_ts):
                        _apply_retrace(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if _prefill_active(rec, bar_ts):
                        _apply_prefill(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if hit:
                        _DIRTY.add(key)
                        caught += 1
        log.info("Setup journal yuklendi: %d kayit, %d telafi mumu.", len(rows), caught)
    except Exception:
        log.exception("setup_journal.ensure_loaded failed")


async def flush(session) -> None:
    """Degisen kayitlari tek commit'te yaz."""
    if not _DIRTY:
        return
    keys = list(_DIRTY)
    _DIRTY.clear()
    try:
        for key in keys:
            rec = _CACHE.get(key)
            if rec is None:
                continue
            row = await session.get(SetupJournal, rec["id"]) if rec.get("id") else None
            if row is None:
                row = (await session.execute(select(SetupJournal).where(
                    SetupJournal.strategy == key[0], SetupJournal.symbol == key[1],
                    SetupJournal.direction == key[2], SetupJournal.purge_time == key[3],
                ))).scalars().first()
                if row is not None:
                    _merge_row(rec, row)
            if row is None:
                row = SetupJournal(strategy=key[0], symbol=key[1], direction=key[2], purge_time=key[3])
                session.add(row)
            for c in _COLUMNS:
                setattr(row, c, rec.get(c))
            row.shadow = _shadow_dump(rec.get("shadow"))
            row.entries = _entries_dump(rec.get("entries"))
            row.retrace = _retrace_dump(rec.get("retrace"))
            row.prefill = _prefill_dump(rec.get("prefill"))
            row.score_parts = _parts_dump(rec.get("score_parts"))
            row.features = _json_dump(rec.get("features"))
            row.parts_at_levels = _parts_dump(rec.get("parts_at_levels"))
            row.features_at_levels = _json_dump(rec.get("features_at_levels"))
            rec["_flushed_seen"] = rec.get("last_seen")
        await session.commit()
        for key in keys:
            rec = _CACHE.get(key)
            if rec is not None and rec.get("id") is None:
                row = (await session.execute(select(SetupJournal.id).where(
                    SetupJournal.strategy == key[0], SetupJournal.symbol == key[1],
                    SetupJournal.direction == key[2], SetupJournal.purge_time == key[3],
                ))).scalar()
                rec["id"] = row
        _prune()
    except Exception:
        _DIRTY.update(keys)
        try:
            await session.rollback()
        except Exception:
            pass
        log.exception("setup_journal.flush failed (%d kayit tekrar denenecek)", len(keys))


def _prune() -> None:
    cutoff = _now() - KEEP
    for key in [k for k, r in _CACHE.items()
                if r.get("last_seen") and r["last_seen"] < cutoff and r.get("outcome") not in TRACKING
                and not _shadow_active(r) and not _prefill_active(r) and k not in _DIRTY]:
        _CACHE.pop(key, None)
