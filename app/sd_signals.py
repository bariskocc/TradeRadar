"""S/D Signals sayfasi (03.11, kullanici: "aktif sinyallerin sonucunu takip edebilecegim sayfa").

Olcum modunda gercek sinyal yok; "sinyal" = Setup Journal'da butun kapilari gecen S/D bolgesi (`best_stage == "sd_ready"`,
strateji `sd` / `sd1h`). Sayfa Journal satirini Open Signals diliyle gosterir: Waiting (giris dolmadi) / Filled (dolum var,
sonuc yok) / Closed (TP, SL, dolmadan TP, dokunus yok, ufuk doldu). Anlik fiyat ve canli R WS store'undan (REST yok).
Sonuc: dolumdan sonra TP mi SL mi (03.11, S/D dolum ister); duz TP/SL, kismi kar/BE yok.

05.11 (kullanici): 4H ve 1H'te ANA GIRIS = LTF onay + retest (`sd_confirm`, Journal entries["sdc"]). Bolgeye dokunus emir degil
uyaridir; islem CHOCH'tan sonra ilk dolan adaydan (FVG / IFVG / CISD / MSS) acilir, SL = geri cekilmenin ucu + tampon. Ilk Journal
okumasi: 4H limit 5 TP / 20 SL (-7.8R), onay 5 TP / 3 SL (+11.6R); kayiplarin 27/31'i CHOCH'tan once stop. Bolge limiti (eski ana
giris) ayni satirda olcum olarak kalir ("Zone limit" sutunu + ozet ikinci satiri). 15m'de onay kolu yok -> ana giris limit.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SetupJournal
from app import sd_confirm

log = logging.getLogger(__name__)

TSI = timezone(timedelta(hours=3))
PAGE_SIZE = 50
STRATEGIES = ("sd", "sd1h", "sd15m")
# Onay kolunun LTF'si (Journal izleme serisi): 4H bolge 15m, 1H/15m bolge 5m (SD_5M_ENABLED).
from app.config import SD_5M_ENABLED  # noqa: E402
LTF_LABEL = {"sd": "15m", "sd1h": "5m" if SD_5M_ENABLED else "15m", "sd15m": "5m" if SD_5M_ENABLED else "15m"}
TF_LABELS = {"sd": "4H", "sd1h": "1H", "sd15m": "15m"}   # 03.11 (kullanici): sutunda yalniz TF
# Bolge tipi acik adiyla (kullanici: "DBD'nin ne oldugunu anlamiyorum"): leg-in · Base · leg-out, devam / donus.
ZONE_NAMES = {
    "RBR": "Rally · Base · Rally (continuation)", "DBR": "Drop · Base · Rally (reversal)",
    "DBD": "Drop · Base · Drop (continuation)", "RBD": "Rally · Base · Drop (reversal)",
    "?BR": "Base · Rally (weak leg-in)", "?BD": "Base · Drop (weak leg-in)",
}
TF_FILTERS = [("all", "All"), ("sd", "S/D 4H"), ("sd1h", "S/D 1H"), ("sd15m", "S/D 15m")]
MARKET_FILTERS = [("all", "All"), ("crypto", "Crypto"), ("fx", "FX")]
DIRECTION_FILTERS = [("all", "All"), ("long", "Long"), ("short", "Short")]
STATUS_FILTERS = [("all", "All"), ("waiting", "Waiting"), ("filled", "Filled"), ("closed", "Closed")]
# 05.11 (kullanici): ana giris LTF onay + retest olan stratejiler (sd_confirm kolu olanlar)
MAIN_CONFIRM = tuple(sd_confirm.STRATEGIES)
# 05.11 (kullanici): sonuc ve LTF onay sonucu filtreleri -- filtre -> eslesen outcome kumesi
RESULT_FILTERS = [("all", "All"), ("tp", "TP"), ("sl", "SL"), ("no_trade", "No trade (zone failed / no retest)"),
                  ("week_close", "Week close"), ("expired", "Expired"), ("other", "Same bar / missed")]
_RESULT_SET = {"tp": {"win"}, "sl": {"loss"}, "no_trade": {"failed", "missed"}, "week_close": {"week_close"},
               "expired": {"no_touch", "open", "no_choch"}, "other": {"ambiguous", "tp_before_entry", "unmeasured"}}
CONFIRM_FILTERS = [("all", "All"), ("worked", "Worked"), ("failed", "Zone failed"), ("no_retest", "No retest"),
                   ("waiting", "Waiting / in trade"), ("none", "No touch yet")]
_CONFIRM_SET = {"worked": {"win"}, "failed": {"loss", "failed"}, "no_retest": {"missed", "no_touch", "no_choch", "open"},
                "waiting": {"wait_choch", "wait_retest", "filled"}, "none": {None}}

# outcome -> (durum sekmesi, etiket, renk, "sonuc" etiketi, sonuc rengi)
_OUTCOME = {
    "pending":         ("waiting", "Waiting entry",   "blue",   "—",               "gray"),
    "filled":          ("filled",  "Filled",          "yellow", "—",               "gray"),
    "win":             ("closed",  "Closed",          "gray",   "TP",              "green"),
    # 03.11 (kullanici): dolum olmadan TP sonuc degil -- motor artik bu satiri uretmez (bekleyis surer); eski satirlar "Missed", R yok
    "tp_before_entry": ("closed",  "Missed",          "gray",   "TP before entry", "yellow"),
    "loss":            ("closed",  "Closed",          "gray",   "SL",              "red"),
    "ambiguous":       ("closed",  "Closed",          "gray",   "Same bar (?)",    "yellow"),
    "no_touch":        ("closed",  "Expired",         "gray",   "No entry",        "gray"),
    "open":            ("closed",  "Expired",         "gray",   "Open at horizon", "gray"),
    # 05.11 (kullanici): kripto disi dolmus islem Cuma 17:00 NY'de haftanin son fiyatiyla kapanir
    "week_close":      ("closed",  "Closed",          "gray",   "Week close",      "gray"),
    # 05.11: LTF onay ana giris (4H / 1H) -- bolgeye dokunus uyari, islem CHOCH + retest'te
    "wait_choch":      ("waiting", "Waiting CHOCH",   "blue",   "—",               "gray"),
    "wait_retest":     ("waiting", "Waiting retest",  "blue",   "—",               "gray"),
    "failed":          ("closed",  "No trade",        "gray",   "Zone failed",     "gray"),
    "missed":          ("closed",  "No trade",        "gray",   "No retest",       "yellow"),
    "no_choch":        ("closed",  "Expired",         "gray",   "No CHOCH",        "gray"),
    "unmeasured":      ("closed",  "Closed",          "gray",   "Not measured",    "gray"),
}
_LIMIT_SHORT = {"pending": "Waiting", "filled": "Filled", "win": "TP", "loss": "SL", "tp_before_entry": "TP before entry",
                "ambiguous": "Same bar", "no_touch": "No entry", "open": "Open", "week_close": "Week close"}


def week_start_tsi(now: datetime | None = None) -> str:
    n = (now or datetime.now(timezone.utc)).astimezone(TSI)
    return (n - timedelta(days=n.weekday())).strftime("%Y-%m-%d")


def _tsi_day(value: str, end: bool = False) -> datetime | None:
    try:
        d = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=TSI)
    except (TypeError, ValueError):
        return None
    if end:
        d += timedelta(days=1)
    return d.astimezone(timezone.utc).replace(tzinfo=None)


def _fmt_tsi(v) -> str:
    if v is None:
        return "—"
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.astimezone(TSI).strftime("%d.%m %H:%M")


def _live_price(symbol: str) -> float | None:
    """WS store'undan son 15m kapanisi (forming dahil); REST yok. Test betigi bunu yamalar."""
    try:
        from app import market_data
        from app.exchange import from_display_symbol
        bingx, _ = from_display_symbol(symbol)
        store = getattr(market_data, "store", None)
        if store is None:
            return None
        for ltf in ("15m", "1h"):
            df = store.get_df(bingx, ltf)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])
    except Exception:
        return None
    return None


def _fmt_p(v) -> str:
    """Fiyat: >= 10 ise 2 ondalik (kullanici: "virgulden sonra 2 basamak yeter"), kucuk fiyatlarda (FX/altcoin) 5'e kadar."""
    if v is None:
        return "—"
    v = float(v)
    s = f"{v:.2f}" if abs(v) >= 10 else f"{v:.5f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _feats(row) -> dict:
    try:
        return json.loads(row.features_at_levels or row.features or "{}") or {}
    except Exception:
        return {}


def _ts(v):
    """sdc zamanlari: datetime ya da ISO metin (adaylar) -> naive UTC datetime."""
    if v is None or isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def _limit_final_r(r: SetupJournal, ents: dict) -> float | None:
    """Bolge limiti (proximal) kolunun duz R'si -- 05.11'e kadar ana giris, simdi 4H / 1H'te olcum."""
    oc = r.outcome or "pending"
    if oc == "win":
        return r.rr
    if oc == "loss":
        return -1.0
    if oc in ("ambiguous", "no_touch", "open"):
        return 0.0
    if oc == "week_close":
        return sd_confirm.week_close_r(r.entry, r.sl, (ents.get("_wc") or {}).get("px"), r.direction)
    return None


def _confirm_trade(r: SetupJournal, sdc: dict) -> dict:
    """LTF onay kolundan ana islem: durum anahtari + seviyeler. Bolge dokunulmadan / CHOCH beklerken seviyeler bolgeninkidir."""
    long = r.direction == "LONG"
    o = sdc.get("o")
    row_oc = r.outcome or "pending"
    zone = {"entry": r.entry, "sl": r.sl, "rr": r.rr, "kind": "zone"}
    if o in (None, "pre"):
        oc = row_oc if row_oc in ("pending", "no_touch") else "unmeasured"   # dolmus / kapanmis ama onay izi yok (05.11 oncesi)
        return {"oc": oc, **zone}
    if o in ("wait_choch", "failed", "no_choch"):
        return {"oc": o, **zone}
    sl = sdc.get("sl")
    if sdc.get("e") is not None:
        return {"oc": o, "entry": float(sdc["e"]), "sl": sl, "rr": sdc.get("rr"), "kind": "trade"}
    pend = [v for v in (sdc.get("vars") or {}).values() if v.get("o") == "pending"] or list((sdc.get("vars") or {}).values())
    if pend:                                     # CHOCH sonrasi: fiyata en yakin aday ilk dolacak olan
        v = max(pend, key=lambda x: x["e"]) if long else min(pend, key=lambda x: x["e"])
        return {"oc": o, "entry": float(v["e"]), "sl": sl, "rr": v.get("rr"), "kind": "trade"}
    return {"oc": o, **zone}


def _row_view(r: SetupJournal, price_fn) -> dict:
    f = _feats(r)
    long = r.direction == "LONG"
    try:
        ents = (json.loads(r.entries) if r.entries else {}) or {}
    except Exception:
        ents = {}
    sdc = ents.get(sd_confirm.SDC_KEY) or {}
    confirm_main = r.strategy in MAIN_CONFIRM
    tp = r.tp
    if confirm_main:
        ct = _confirm_trade(r, sdc)
        oc, entry, sl, rr, lv_kind = ct["oc"], ct["entry"], ct["sl"], ct["rr"], ct["kind"]
        filled_ts = _ts(sdc.get("t")) if lv_kind == "trade" else None
        closed_ts = _ts(sdc.get("at"))
    else:
        oc, entry, sl, rr, lv_kind = r.outcome or "pending", r.entry, r.sl, r.rr, "trade"
        filled_ts, closed_ts = r.entry_touched_at, r.outcome_at
    risk = abs(entry - sl) if entry is not None and sl is not None else 0.0
    tab, status_label, status_color, result_label, result_color = _OUTCOME.get(oc, _OUTCOME["pending"])
    if confirm_main and oc == "pending":
        status_label = "Waiting zone"            # bolgeye dokunus = uyari, emir degil
    price = price_fn(r.symbol) if tab in ("waiting", "filled") else None
    live_r = dist_r = None
    if price is not None and entry is not None and risk > 0:
        move = (price - entry) if long else (entry - price)
        if oc == "filled":
            live_r = round(move / risk, 2)
        elif tab == "waiting":
            dist_r = round(move / risk, 2)           # pozitif = fiyat girise henuz gelmedi (LONG'da ustunde, SHORT'ta altinda)
    if confirm_main and sdc.get("o") not in (None, "pre"):
        final_r = sd_confirm.final_r(sdc)
        if oc in ("wait_choch", "wait_retest", "filled"):
            final_r = None
    else:
        final_r = _limit_final_r(r, ents) if tab == "closed" else None
        if oc == "unmeasured":
            final_r = None
    touches = f.get("touches")
    zt = r.model or f.get("zone_type") or "?"
    c_o = sdc.get("o")
    # Bolge limiti olcum kolu (4H / 1H); 15m'de ana giris zaten limit
    lim = None
    if confirm_main:
        l_oc = r.outcome or "pending"
        lim = {"label": _LIMIT_SHORT.get(l_oc, l_oc), "o": l_oc, "r": _limit_final_r(r, ents)}
    return {
        "symbol": r.symbol, "market": r.market_type, "tf": r.strategy, "tf_label": TF_LABELS.get(r.strategy, r.strategy),
        "direction": r.direction, "zone_type": zt, "bias": r.htf_bias,
        "zone_label": ("Demand" if long else "Supply") + " · " + ZONE_NAMES.get(zt, zt),
        "entry_s": _fmt_p(entry), "sl_s": _fmt_p(sl), "tp_s": _fmt_p(tp),
        "confirm_o": c_o if c_o != "pre" else None, "confirm_rr": sdc.get("rr"), "confirm_main": confirm_main,
        **{f"confirm_{k}": v for k, v in sd_confirm.describe(sdc, LTF_LABEL.get(r.strategy, "15m")).items()},
        "levels_kind": lv_kind, "limit": lim, "zone_entry_s": _fmt_p(r.entry),
        "touched_at": _fmt_tsi(r.entry_touched_at) if r.entry_touched_at else None,
        "setup_at": _fmt_tsi(r.purge_time), "retest_at": _fmt_tsi(filled_ts) if filled_ts else None,
        "_sort": (-(int(r.score) if r.score is not None else -1), -(r.best_stage_at.timestamp() if r.best_stage_at else 0)),
        "fresh": None if touches is None else int(touches) == 0,
        "bos": bool(f.get("bos")), "sweep": bool(f.get("sweep")), "fvg": bool(f.get("fvg")),
        "mss": bool(f.get("mss")), "sweep_kind": f.get("sweep_kind"), "pd_ok": f.get("pd_ok"), "pd_pos": f.get("pd_pos"),
        "score": int(r.score) if r.score is not None else None,
        "entry": entry, "sl": sl, "tp": tp, "rr": rr, "rr_struct": f.get("rr_struct"), "target_kind": f.get("target_kind"),
        "price": price, "live_r": live_r, "dist_r": dist_r, "final_r": final_r,
        "tab": tab, "status_label": status_label, "status_color": status_color,
        "result_label": result_label, "result_color": result_color, "outcome": oc,
        "ready_at": _fmt_tsi(r.best_stage_at), "formed": _fmt_tsi(r.purge_time),
        "filled_at": _fmt_tsi(filled_ts) if filled_ts else None,
        "closed_at": _fmt_tsi(closed_ts) if closed_ts and tab == "closed" else None,
        "hours": round((closed_ts - filled_ts).total_seconds() / 3600, 1)
        if closed_ts and filled_ts and tab == "closed" and oc in ("win", "loss", "ambiguous", "week_close") else None,
    }


async def build_sd_signals(
    db: AsyncSession, *, tf: str = "all", status: str = "all", market: str = "all", direction: str = "all",
    date_from: str = "", date_to: str = "", page: int = 1, price_fn=None, result: str = "all", confirm: str = "all",
) -> dict:
    tf = tf if tf in STRATEGIES else "all"
    status = status if status in dict(STATUS_FILTERS) else "all"
    market = market if market in ("crypto", "fx") else "all"
    direction = direction if direction in ("long", "short") else "all"
    result = result if result in _RESULT_SET else "all"
    confirm = confirm if confirm in _CONFIRM_SET else "all"
    price_fn = price_fn or _live_price
    t = SetupJournal
    where = [t.strategy.in_(STRATEGIES), t.best_stage == "sd_ready", t.entry.is_not(None)]
    if tf != "all":
        where.append(t.strategy == tf)
    if market == "crypto":
        where.append(t.market_type == "crypto")
    elif market == "fx":
        where.append(t.market_type != "crypto")
    if direction != "all":
        where.append(t.direction == direction.upper())
    d0, d1 = _tsi_day(date_from), _tsi_day(date_to, end=True)
    if d0 is not None:
        where.append(t.best_stage_at >= d0)
    if d1 is not None:
        where.append(t.best_stage_at < d1)
    error = False
    try:
        rows = (await db.execute(select(t).where(*where).order_by(t.best_stage_at.desc()))).scalars().all()
    except Exception:
        log.exception("sd_signals query failed")
        rows, error = [], True
    views = []
    cache: dict[str, float | None] = {}

    def _price(sym: str) -> float | None:
        if sym not in cache:
            cache[sym] = price_fn(sym)
        return cache[sym]

    for r in rows:
        views.append(_row_view(r, _price))
    views.sort(key=lambda v: v["_sort"])      # 03.11 (kullanici): skor azalan, esitte en yeni (CRT sayfalari gibi)
    # Sonuc / LTF onay filtresi sekme sayaclarina ve ozete de uyar (Signals sayfasi sozlesmesi)
    if result != "all":
        views = [v for v in views if v["outcome"] in _RESULT_SET[result]]
    if confirm != "all":
        views = [v for v in views if v["confirm_o"] in _CONFIRM_SET[confirm]]
    counts = {k: 0 for k, _ in STATUS_FILTERS}
    summary = {}
    for v in views:
        counts["all"] += 1
        counts[v["tab"]] += 1
        s = summary.setdefault(v["tf"], {"label": v["tf_label"], "n": 0, "tp": 0, "sl": 0, "r": 0.0, "open": 0, "no_trade": 0,
                                         "confirm_main": v["confirm_main"], "l_tp": 0, "l_sl": 0, "l_r": 0.0})
        s["n"] += 1
        if v["outcome"] == "win":
            s["tp"] += 1
            s["r"] += float(v["final_r"] or 0)
        elif v["outcome"] == "loss":
            s["sl"] += 1
            s["r"] -= 1.0
        elif v["outcome"] == "week_close":
            s["r"] += float(v["final_r"] or 0)
        elif v["outcome"] in ("failed", "missed"):
            s["no_trade"] += 1
        elif v["tab"] != "closed":
            s["open"] += 1
        lim = v["limit"]
        if lim:                                   # bolge limiti olcum kolu
            if lim["o"] == "win":
                s["l_tp"] += 1
            elif lim["o"] == "loss":
                s["l_sl"] += 1
            if lim["o"] in ("win", "loss", "week_close"):
                s["l_r"] += float(lim["r"] or 0)
    for s in summary.values():
        n = s["tp"] + s["sl"]
        s["wr"] = round(100 * s["tp"] / n) if n else None
        s["r"] = round(s["r"], 2)
        s["l_r"] = round(s["l_r"], 2)
    if status != "all":
        views = [v for v in views if v["tab"] == status]
    pages = max(1, (len(views) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), pages)
    return {
        "rows": views[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], "page": page, "pages": pages,
        "counts": counts, "summary": [summary[k] for k in STRATEGIES if k in summary], "error": error,
        "tf": tf, "status": status, "market": market, "direction": direction, "date_from": date_from, "date_to": date_to,
        "result": result, "confirm": confirm,
        "tf_filters": TF_FILTERS, "market_filters": MARKET_FILTERS, "direction_filters": DIRECTION_FILTERS,
        "status_filters": STATUS_FILTERS, "result_filters": RESULT_FILTERS, "confirm_filters": CONFIRM_FILTERS,
    }
