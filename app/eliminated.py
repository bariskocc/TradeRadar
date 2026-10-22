"""Elenen setuplar sayfasi (22.10, kullanici istegi) -- `/eliminated`.

Soru: motorun gordugu ama AKTIF SINYALE DONUSMEYEN butun setuplar (CRT saymadigi adaylar dahil) sonra ne yapti --
hangi filtre kac setup eledi, elenenler TP'ye mi SL'ye mi gitti, filtre koruyor mu kazanc mi kesiyor. Veri
Setup Journal (`setup_journal`); Journal sayfasi ayni veriyi gosteriyor ama okunmuyordu (kullanici: "journal'dan
bu verileri okumak cok zor").

Kapsam: `outcome != 'signal'` (sinyale donusen satirin izlemesi durur; waiting'e ulasip dolmadan silinen setup
izlemeye geri doner -> burada "Signal deleted before fill"). Sonuc DUZ TP/SL: Journal seviyeleri ilk goruldugu
haliyle donar, kismi kar + BE yok. Motor kararina dokunmaz; yalniz okur.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SetupJournal, Signal
from app.setup_journal import PRE_SETUP_STAGES, STAGES

PAGE_SIZE = 50
DAYS_CHOICES = (7, 30, 90, 0)          # 0 = tum zamanlar
MIN_RESOLVED = 10                      # hukum icin en az cozulmus (TP+SL) setup
VERDICT_BAND = 0.15                    # R/setup bandi -- IZLEME karar kurallariyla ayni olcek

# Sayfada gosterilen filtre adi: Journal etiketleri gelistirici dilinde, burada "neden elendi" diye okunur.
GATE_LABELS = {
    "sweep_small": "Sweep too small",
    "range_atr": "C1 range outside ATR band",
    "c1_stale": "C1 extreme already taken",
    "c2_breakout": "C2 closed outside C1",
    "c2_wrong_color": "C2 wrong color",
    "c1_weak": "C1 body too small",
    "not_selected": "Another CRT candidate chosen",
    "bias_mismatch": "1D bias opposite",
    "target_taken": "Target already taken",
    "low_quality": "Score below threshold",
    "no_cisd": "No CISD / MSS",
    "low_rr": "RR below 2",
    "score7": "Score-7 gate",
    "tight_stop": "Stop too tight",
    "past_tp": "Price passed TP",
    "past_sl": "Price passed SL",
    "invalidated": "CRT 60% crossed",
    "missed_quality": "Score < 7 at retest",
    "stale": "Retest too old",
    "week_gap": "CISD before week close",
    "cluster_limit": "Crypto cluster limit",
    "corr_open": "Correlated pair open",
    "has_open": "Open signal on symbol",
    "duplicate": "Already saved",
    "deleted_before": "Deleted once — not reopened",
    "c2_open": "C2 never closed valid",
    "waiting": "Signal deleted before fill",
    "week_close": "Cancelled at week close",
}
# Silinme nedeni (`deleted_reason`) -- kapi kodu degilse buradan; yoksa GATE_LABELS.
DELETE_LABELS = {"crt_gone": "CRT no longer valid", "crt_replaced": "replaced by another CRT",
                 "score7_gate": "Score-7 gate", "manual": "manual"}
# Motorun kapilari hangi asamada: CRT tespiti (PRE_SETUP) / setup kapilari / sinyal sonrasi.
GROUP_PRE = "CRT detection"
GROUP_GATE = "Setup gates"
GROUP_POST = "After signal"
_POST_STAGES = {"waiting", "week_close"}

RESULT_META = {   # outcome -> (etiket, renk sinifi)
    "win": ("TP", "bg-accent-green/15 text-accent-green"),
    "loss": ("SL", "bg-accent-red/15 text-accent-red"),
    "tp_before_entry": ("TP, no entry", "bg-accent-yellow/15 text-accent-yellow"),
    "ambiguous": ("Same bar", "bg-dark-600 text-gray-300"),
    "no_touch": ("No entry", "bg-dark-600 text-gray-400"),
    "open": ("No TP/SL in time", "bg-dark-600 text-gray-400"),
    "pending": ("Tracking", "bg-accent-blue/15 text-accent-blue"),
    "filled": ("Tracking (filled)", "bg-accent-blue/15 text-accent-blue"),
    None: ("No levels", "bg-dark-600 text-gray-500"),
}
RESULT_FILTERS = {  # ?result= -> outcome kumesi
    "tp": ("win",),
    "sl": ("loss",),
    "tbe": ("tp_before_entry",),
    "tracking": ("pending", "filled"),
    "other": ("ambiguous", "no_touch", "open"),
}

VERDICT_META = {
    "protect": ("Protecting", "text-accent-green", "eliminated setups lose — the gate avoided losses"),
    "cut": ("Cutting winners", "text-accent-red", "eliminated setups win — the gate blocks profit"),
    "neutral": ("No clear effect", "text-gray-400", "eliminated setups are about breakeven"),
    "few": ("Too few", "text-gray-500", f"fewer than {MIN_RESOLVED} setups reached TP or SL"),
}


def _gate_group(code: str) -> str:
    if code in PRE_SETUP_STAGES:
        return GROUP_PRE
    return GROUP_POST if code in _POST_STAGES else GROUP_GATE


def verdict(tp: int, sl: int, r: float) -> str:
    """Filtrenin hukmu: elenenler cozuldugunde R/setup. Negatif = filtre kaybi onlemis."""
    n = tp + sl
    if n < MIN_RESOLVED:
        return "few"
    per = r / n
    if per <= -VERDICT_BAND:
        return "protect"
    if per >= VERDICT_BAND:
        return "cut"
    return "neutral"


def _js(v) -> dict:
    if not v:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return {}


def _pct(x) -> Optional[str]:
    return None if x is None else f"{float(x) * 100:.0f}%"


def why_text(row: SetupJournal) -> str:
    """Elemenin nedeni, filtreye ozgu olcumle (ör. C2 rengi elemesinde C2 govde orani)."""
    f = _js(row.features_at_levels) or _js(row.features)
    code = row.best_stage
    parts: list[str] = []
    if code == "c2_wrong_color":
        b = _pct(f.get("c2_body_frac"))
        parts.append(f"C2 closed {'red' if row.direction == 'LONG' else 'green'}" + (f", body {b} of range" if b else ""))
    elif code == "c1_weak":
        b = _pct(f.get("c1_body_frac"))
        parts.append(f"C1 body {b} of range (min 30%)" if b else "C1 body small")
    elif code == "c2_breakout":
        parts.append("C2 closed beyond the C1 extreme it swept")
    elif code == "range_atr" and f.get("range_atr") is not None:
        parts.append(f"C1 range {float(f['range_atr']):.2f}× ATR")
    elif code == "sweep_small" and f.get("sweep_pct") is not None:
        parts.append(f"sweep {float(f['sweep_pct']):.3f}%")
    elif code == "bias_mismatch":
        parts.append(f"1D bias {row.htf_bias or '?'} vs {row.direction}")
    elif code == "low_rr" and row.rr is not None:
        parts.append(f"RR {row.rr:.2f} < 2")
    elif code in ("low_quality", "score7", "missed_quality") and row.score is not None:
        parts.append(f"score {row.score:.0f}")
    elif code == "tight_stop" and f.get("stop_range_mult") is not None:
        parts.append(f"stop {float(f['stop_range_mult']):.2f}× LTF range")
    elif row.deleted_reason:
        parts.append(f"deleted: {DELETE_LABELS.get(row.deleted_reason) or GATE_LABELS.get(row.deleted_reason, row.deleted_reason)}")
    rec = f.get("reclaim_pct")
    if rec is not None and code not in ("c2_breakout",) and float(rec) < 25:
        parts.append(f"weak C2 reclaim {float(rec):.0f}%")
    return " · ".join(parts)


def row_r(row: SetupJournal) -> Optional[float]:
    if row.outcome == "win" and row.rr is not None:
        return round(float(row.rr), 2)
    if row.outcome == "loss":
        return -1.0
    return None


def row_hours(row: SetupJournal) -> Optional[float]:
    """Sonuca kac saatte gidildi: TP/SL'de girisin dolumundan, girissiz TP'de seviyelerin donmasindan."""
    if row.outcome_at is None:
        return None
    start = row.entry_touched_at if row.outcome in ("win", "loss") else row.levels_at
    if start is None:
        return None
    return round((row.outcome_at - start).total_seconds() / 3600, 1)


def _base_filters(tf: str, market: str, days: int, symbol: str) -> list:
    t = SetupJournal
    flt = [or_(t.outcome.is_(None), t.outcome != "signal")]
    if tf != "all":
        flt.append(t.strategy == tf)
    if market == "crypto":
        flt.append(t.market_type == "crypto")
    elif market == "fx":
        flt.append(t.market_type != "crypto")
    if days:
        flt.append(t.purge_time >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days))
    if symbol:
        flt.append(t.symbol.ilike(f"%{symbol}%"))
    return flt


async def _signal_reference(db: AsyncSession, tf: str, market: str, days: int) -> dict:
    """Ayni donemde gercekten acilan sinyaller (kapanis zamanina gore): W/L/BE + gercek R -- kiyas icin."""
    flt = [Signal.closed_at.is_not(None), Signal.result.is_not(None)]
    if tf != "all":
        flt.append(Signal.timeframe == tf)
    if market == "crypto":
        flt.append(Signal.market_type == "crypto")
    elif market == "fx":
        flt.append(Signal.market_type != "crypto")
    if days:
        flt.append(Signal.closed_at >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days))
    rows = (await db.execute(select(Signal.result, Signal.exit_reason, Signal.rr_value).where(*flt))).all()
    w = l = be = 0
    r = 0.0
    for res, ex, rv in rows:
        if ex == "be":
            be += 1          # kismi karli BE DB'de result='win' -- main._trade_outcome kurali
        elif res == "win":
            w += 1
        elif res == "loss":
            l += 1
        else:
            be += 1
        r += float(rv or 0.0)
    n = w + l + be
    return {"n": n, "w": w, "l": l, "be": be, "r": round(r, 2),
            "wr": round(w / (w + l) * 100, 1) if w + l else None}


async def build_eliminated(db: AsyncSession, *, tf: str = "all", market: str = "all", days: int = 30,
                           gate: str = "all", result: str = "all", symbol: str = "", page: int = 1) -> dict:
    tf = tf if tf in ("all", "4h", "1d", "1w", "1h") else "all"
    market = market if market in ("all", "crypto", "fx") else "all"
    days = days if days in DAYS_CHOICES else 30
    gate = gate if gate in GATE_LABELS else "all"
    result = result if result in RESULT_FILTERS else "all"
    symbol = (symbol or "").strip()
    t = SetupJournal
    base = _base_filters(tf, market, days, symbol)

    # 1) Filtre tablosu: SQL'de gruplanir (tum zamanlar ~11k satir).
    win_r = func.sum(case((t.outcome == "win", t.rr), else_=0.0))

    def _n(cond):
        return func.sum(case((cond, 1), else_=0))

    grouped = (await db.execute(select(
        t.best_stage, func.count(t.id), _n(t.outcome == "win"), _n(t.outcome == "loss"),
        _n(t.outcome == "tp_before_entry"), _n(t.outcome.in_(("pending", "filled"))),
        _n(or_(t.outcome.is_(None), t.outcome.in_(("no_touch", "open", "ambiguous")))), win_r,
    ).where(*base).group_by(t.best_stage))).all()
    per = {g[0]: g for g in grouped}
    gates = []
    for code, _label in STAGES:
        g = per.get(code)
        if not g:
            continue
        _, n, tp, sl, tbe, trk, other, wr_sum = g
        tp, sl = int(tp or 0), int(sl or 0)
        r = float(wr_sum or 0.0) - sl
        v = verdict(tp, sl, r)
        gates.append({
            "code": code, "label": GATE_LABELS.get(code, code), "group": _gate_group(code),
            "n": int(n), "tp": tp, "sl": sl, "tbe": int(tbe or 0), "tracking": int(trk or 0), "other": int(other or 0),
            "wr": round(tp / (tp + sl) * 100, 1) if tp + sl else None,
            "r": round(r, 1), "r_per": round(r / (tp + sl), 2) if tp + sl else None,
            "verdict": v, "verdict_label": VERDICT_META[v][0], "verdict_class": VERDICT_META[v][1],
        })
    groups = []
    for name in (GROUP_PRE, GROUP_GATE, GROUP_POST):
        rows_ = [g for g in gates if g["group"] == name]
        if rows_:
            groups.append({"name": name, "gates": rows_})

    tp = sum(g["tp"] for g in gates)
    sl = sum(g["sl"] for g in gates)
    r = sum(g["r"] for g in gates)
    totals = {
        "n": sum(g["n"] for g in gates), "tp": tp, "sl": sl,
        "tbe": sum(g["tbe"] for g in gates), "tracking": sum(g["tracking"] for g in gates),
        "other": sum(g["other"] for g in gates),
        "wr": round(tp / (tp + sl) * 100, 1) if tp + sl else None,
        "r": round(r, 1), "r_per": round(r / (tp + sl), 2) if tp + sl else None,
    }
    # CRT sayilan / sayilmayan ayri satir: "en bastaki" filtreler toplamin cogunu olusturuyor.
    split = []
    for name, cond in (("Counted as CRT", lambda g: g["group"] != GROUP_PRE),
                       ("Rejected at CRT detection", lambda g: g["group"] == GROUP_PRE)):
        rows_ = [g for g in gates if cond(g)]
        s_tp, s_sl = sum(g["tp"] for g in rows_), sum(g["sl"] for g in rows_)
        s_r = sum(g["r"] for g in rows_)
        split.append({"name": name, "n": sum(g["n"] for g in rows_), "tp": s_tp, "sl": s_sl,
                      "wr": round(s_tp / (s_tp + s_sl) * 100, 1) if s_tp + s_sl else None,
                      "r_per": round(s_r / (s_tp + s_sl), 2) if s_tp + s_sl else None})

    # 2) Setup listesi
    flt = list(base)
    if gate != "all":
        flt.append(t.best_stage == gate)
    if result != "all":
        flt.append(t.outcome.in_(RESULT_FILTERS[result]))
    total = await db.scalar(select(func.count(t.id)).where(*flt)) or 0
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), total_pages)
    recs = (await db.execute(select(t).where(*flt).order_by(t.purge_time.desc(), t.id.desc())
                             .offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE))).scalars().all()
    rows = []
    for x in recs:
        label, cls = RESULT_META.get(x.outcome, RESULT_META[None])
        rows.append({
            "id": x.id, "symbol": x.symbol, "market_type": x.market_type, "tf": (x.strategy or "").upper(),
            "direction": x.direction, "c1": x.crt_bar_time, "c2": x.purge_time,
            "gate": GATE_LABELS.get(x.best_stage, x.best_stage), "gate_code": x.best_stage,
            "pre": x.best_stage in PRE_SETUP_STAGES, "why": why_text(x),
            "score": x.score, "rr": x.rr, "bias": x.htf_bias, "model": x.model,
            "entry": x.entry, "sl": x.sl, "tp": x.tp,
            "result": label, "result_class": cls, "r": row_r(x), "hours": row_hours(x),
            "outcome_at": x.outcome_at,
        })

    return {
        "tf": tf, "market": market, "days": days, "gate": gate, "result": result, "symbol": symbol,
        "rows": rows, "page": page, "total": total, "total_pages": total_pages,
        "groups": groups, "totals": totals, "split": split,
        "reference": await _signal_reference(db, tf, market, days),
        "gate_labels": GATE_LABELS, "days_choices": DAYS_CHOICES,
        "verdict_meta": VERDICT_META, "min_resolved": MIN_RESOLVED, "band": VERDICT_BAND,
    }
