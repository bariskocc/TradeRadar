"""S/D Signals sayfasi (03.11, kullanici: "aktif sinyallerin sonucunu takip edebilecegim sayfa").

Olcum modunda gercek sinyal yok; "sinyal" = Setup Journal'da butun kapilari gecen S/D bolgesi (`best_stage == "sd_ready"`,
strateji `sd` / `sd1h`). Sayfa Journal satirini Open Signals diliyle gosterir: Waiting (giris dolmadi) / Filled (dolum var,
sonuc yok) / Closed (TP, SL, dolmadan TP, dokunus yok, ufuk doldu). Anlik fiyat ve canli R WS store'undan (REST yok).
Sonuc kurali CRT'dekiyle ayni: TP once mi SL once mi -- `win` + `tp_before_entry` = TP, `loss` = SL; duz TP/SL, kismi kar/BE yok.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SetupJournal

log = logging.getLogger(__name__)

TSI = timezone(timedelta(hours=3))
PAGE_SIZE = 50
STRATEGIES = ("sd", "sd1h")
TF_LABELS = {"sd": "S/D 4H", "sd1h": "S/D 1H"}
TF_FILTERS = [("all", "All"), ("sd", "S/D 4H"), ("sd1h", "S/D 1H")]
MARKET_FILTERS = [("all", "All"), ("crypto", "Crypto"), ("fx", "FX")]
DIRECTION_FILTERS = [("all", "All"), ("long", "Long"), ("short", "Short")]
STATUS_FILTERS = [("all", "All"), ("waiting", "Waiting"), ("filled", "Filled"), ("closed", "Closed")]

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
}


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


def _feats(row) -> dict:
    try:
        return json.loads(row.features_at_levels or row.features or "{}") or {}
    except Exception:
        return {}


def _row_view(r: SetupJournal, price: float | None) -> dict:
    f = _feats(r)
    long = r.direction == "LONG"
    entry, sl, tp, rr = r.entry, r.sl, r.tp, r.rr
    risk = abs(entry - sl) if entry is not None and sl is not None else 0.0
    oc = r.outcome or "pending"
    tab, status_label, status_color, result_label, result_color = _OUTCOME.get(oc, _OUTCOME["pending"])
    live_r = dist_r = None
    if price is not None and entry is not None and risk > 0:
        move = (price - entry) if long else (entry - price)
        if oc == "filled":
            live_r = round(move / risk, 2)
        elif oc == "pending":
            dist_r = round(move / risk, 2)           # pozitif = fiyat girise henuz gelmedi (LONG'da ustunde, SHORT'ta altinda)
    final_r = None
    if oc == "win":
        final_r = rr
    elif oc == "loss":
        final_r = -1.0
    elif oc in ("ambiguous", "no_touch", "open"):
        final_r = 0.0
    touches = f.get("touches")
    return {
        "symbol": r.symbol, "market": r.market_type, "tf": r.strategy, "tf_label": TF_LABELS.get(r.strategy, r.strategy),
        "direction": r.direction, "zone_type": r.model or f.get("zone_type") or "—",
        "fresh": None if touches is None else int(touches) == 0,
        "bos": bool(f.get("bos")), "sweep": bool(f.get("sweep")), "fvg": bool(f.get("fvg")),
        "score": int(r.score) if r.score is not None else None,
        "entry": entry, "sl": sl, "tp": tp, "rr": rr, "rr_struct": f.get("rr_struct"), "target_kind": f.get("target_kind"),
        "price": price, "live_r": live_r, "dist_r": dist_r, "final_r": final_r,
        "tab": tab, "status_label": status_label, "status_color": status_color,
        "result_label": result_label, "result_color": result_color, "outcome": oc,
        "ready_at": _fmt_tsi(r.best_stage_at), "formed": _fmt_tsi(r.purge_time),
        "filled_at": _fmt_tsi(r.entry_touched_at) if r.entry_touched_at else None,
        "closed_at": _fmt_tsi(r.outcome_at) if r.outcome_at and tab == "closed" else None,
        "hours": round((r.outcome_at - (r.entry_touched_at or r.levels_at)).total_seconds() / 3600, 1)
        if r.outcome_at and (r.entry_touched_at or r.levels_at) and tab == "closed" else None,
    }


async def build_sd_signals(
    db: AsyncSession, *, tf: str = "all", status: str = "all", market: str = "all", direction: str = "all",
    date_from: str = "", date_to: str = "", page: int = 1, price_fn=None,
) -> dict:
    tf = tf if tf in STRATEGIES else "all"
    status = status if status in dict(STATUS_FILTERS) else "all"
    market = market if market in ("crypto", "fx") else "all"
    direction = direction if direction in ("long", "short") else "all"
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
    for r in rows:
        if r.symbol not in cache:
            cache[r.symbol] = price_fn(r.symbol) if (r.outcome or "pending") in ("pending", "filled") else None
        views.append(_row_view(r, cache[r.symbol]))
    counts = {k: 0 for k, _ in STATUS_FILTERS}
    summary = {}
    for v in views:
        counts["all"] += 1
        counts[v["tab"]] += 1
        s = summary.setdefault(v["tf"], {"label": v["tf_label"], "n": 0, "tp": 0, "sl": 0, "r": 0.0, "open": 0})
        s["n"] += 1
        if v["outcome"] == "win":
            s["tp"] += 1
            s["r"] += float(v["rr"] or 0)
        elif v["outcome"] == "loss":
            s["sl"] += 1
            s["r"] -= 1.0
        elif v["tab"] != "closed":
            s["open"] += 1
    for s in summary.values():
        n = s["tp"] + s["sl"]
        s["wr"] = round(100 * s["tp"] / n) if n else None
        s["r"] = round(s["r"], 2)
    if status != "all":
        views = [v for v in views if v["tab"] == status]
    pages = max(1, (len(views) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), pages)
    return {
        "rows": views[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], "page": page, "pages": pages,
        "counts": counts, "summary": [summary[k] for k in STRATEGIES if k in summary], "error": error,
        "tf": tf, "status": status, "market": market, "direction": direction, "date_from": date_from, "date_to": date_to,
        "tf_filters": TF_FILTERS, "market_filters": MARKET_FILTERS, "direction_filters": DIRECTION_FILTERS,
        "status_filters": STATUS_FILTERS,
    }
