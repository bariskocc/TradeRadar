import logging
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request, Form, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, case, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BASE_DIR
from app.logging_config import setup_logging
from app.database import init_db, get_db
from app.models import Signal, EventLog
from app.auth import verify_credentials, create_access_token, get_current_user
from app.scanner import run_scan, SMT_QUALITY_BONUS, MAX_QUALITY_SCORE
from app.scheduler import get_scheduler_status
from app.market_data import market_data

# Uvicorn app'i import eder etmez logu kur; boyle her giris noktasinda (run.py,
# dogrudan `uvicorn app.main:app`) karar loglari gorunur.
setup_logging()

log = logging.getLogger(__name__)

TSI_OFFSET = timedelta(hours=3)


def _fmt_price(value):
    """Bilimsel notasyon olmadan fiyat formatla (4.03e-06 → 0,00000403)."""
    if value is None:
        return "-"
    d = Decimal(str(value))
    sign, digits, exponent = d.as_tuple()
    if exponent < -8:
        exponent = -8
    plain = format(d, 'f')
    if '.' in plain:
        plain = plain.rstrip('0').rstrip('.')
    return plain.replace('.', ',')


def _fmt_date_tsi(value):
    """UTC datetime → TSİ (UTC+3) formatla."""
    if value is None:
        return "-"
    tsi = value + TSI_OFFSET
    return tsi.strftime('%d.%m %H:%M')


def _fmt_ui_symbol(symbol: str | None, market_type: str | None = None) -> str:
    """UI'da sembol gorunumunu sadeleştir."""
    if not symbol:
        return "-"
    sym = str(symbol)
    market = (market_type or "").lower()
    if sym.endswith("USDT.P") and (market == "crypto" or market == ""):
        return sym[:-6]
    return sym


def _calc_rr_ratio(entry, sl, tp):
    """Potansiyel R:R oranı hesapla."""
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return None
    return round(reward / risk, 2)


def _signal_planned_rr(signal) -> float | None:
    """Kayitli planned_rr; yoksa entry + orijinal SL + TP'den hesapla."""
    if getattr(signal, "planned_rr", None) is not None:
        return float(signal.planned_rr)
    sl = getattr(signal, "initial_stop_loss", None) or signal.stop_loss
    return _calc_rr_ratio(signal.entry_price, sl, signal.take_profit)


def _signal_realized_rr(signal) -> float | None:
    """Kapanmis islemde gerceklesen R; acik/waiting ise None."""
    if signal.result is None:
        return None
    if signal.rr_value is None:
        return None
    return float(signal.rr_value)

scan_state: dict = {"running": False, "last_run": None, "last_result": None}


def _parse_tf(raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if v in ("1d", "1h"):
        return v
    return "4h"


def _tf_label(tf: str) -> str:
    if tf == "1d":
        return "1D-1H"
    if tf == "1h":
        return "1H-5M"
    return "4H-15M"


def _tf_filter(tf: str):
    if tf == "1d":
        return Signal.timeframe == "1d"
    if tf == "1h":
        return Signal.timeframe == "1h"
    return or_(Signal.timeframe == "4h", Signal.timeframe.is_(None))

# Radar state code -> display (label / color / sort rank)
RADAR_STATE_META = {
    "waiting":        {"label": "Signal opened (waiting)",   "color": "green",  "rank": 0},
    "c2_open":        {"label": "Waiting for C2 close",       "color": "blue",   "rank": 1},
    "no_cisd":        {"label": "Waiting for MSS",            "color": "blue",   "rank": 1},
    "low_rr":         {"label": "Low RR",                     "color": "yellow", "rank": 2},
    "tight_stop":     {"label": "Stop too tight (LTF range)", "color": "yellow", "rank": 3},
    "missed":         {"label": "Missed (TP before fill)",    "color": "yellow", "rank": 4},
    "missed_quality": {"label": "Missed (score<7 at entry)",  "color": "yellow", "rank": 4},
    "invalidated":    {"label": "CRT 60% crossed",            "color": "yellow", "rank": 4},
    "past_sl":        {"label": "SL before fill",             "color": "yellow", "rank": 4},
    "stale":          {"label": "Waiting expired (stale)",    "color": "yellow", "rank": 4},
    "same_bar_sl":    {"label": "SL before fill",             "color": "yellow", "rank": 4},
    # "same_color":   {"label": "CRT/purge same color",      "color": "red",    "rank": 4},
    "bias_mismatch":  {"label": "1D bias opposite",           "color": "orange", "rank": 5},
    "cluster_limit":  {"label": "Same-direction cluster limit", "color": "orange", "rank": 6},
    "has_open":       {"label": "Open signal exists",         "color": "purple", "rank": 7},
    "corr_open":      {"label": "Correlated pair open",       "color": "purple", "rank": 8},
    "duplicate":      {"label": "Setup already saved",        "color": "purple", "rank": 9},
    "low_quality":    {"label": "Low quality (score<7)",      "color": "gray",   "rank": 10},
    "no_setup":       {"label": "No setup",                   "color": "dim",    "rank": 11},
    "no_data":        {"label": "Insufficient data",          "color": "dim",    "rank": 12},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await market_data.start()
    yield
    await market_data.stop()


app = FastAPI(title="TradeRadar", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")
templates.env.filters["fmt_price"] = _fmt_price
templates.env.filters["fmt_date_tsi"] = _fmt_date_tsi
templates.env.globals["calc_rr_ratio"] = _calc_rr_ratio
templates.env.globals["signal_planned_rr"] = _signal_planned_rr
templates.env.globals["signal_realized_rr"] = _signal_realized_rr
templates.env.globals["fmt_ui_symbol"] = _fmt_ui_symbol


# ──────────────────── Auth Routes ────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not verify_credentials(username, password):
        return templates.TemplateResponse(request=request, name="login.html", context={"error": "Invalid username or password"})

    token = create_access_token(data={"sub": username})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=60 * 60 * 24,
        samesite="lax",
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response


# ──────────────────── Dashboard (Stats) ────────────────────


def _build_dashboard_stats(signals: list[Signal]) -> dict:
    closed = [s for s in signals if s.result is not None]
    active = [s for s in signals if s.status == "active"]
    wins = [s for s in closed if s.result == "win"]
    losses = [s for s in closed if s.result == "loss"]
    breakevens = [s for s in closed if s.result == "breakeven"]

    total_signals = len(signals)
    total_closed = len(closed)
    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(breakevens)
    win_rate = round((win_count / total_closed * 100), 1) if total_closed > 0 else 0

    rr_values = [s.rr_value for s in closed if s.rr_value is not None]
    total_rr = round(sum(rr_values), 2) if rr_values else 0
    avg_rr = round(total_rr / len(rr_values), 2) if rr_values else 0

    durations = [s.duration_hours for s in closed if s.duration_hours is not None]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

    pair_rr: dict[str, float] = {}
    pair_wins: dict[str, list[int]] = {}
    pair_trades: Counter = Counter()
    for s in closed:
        pair_rr[s.symbol] = pair_rr.get(s.symbol, 0) + (s.rr_value or 0)
        pair_trades[s.symbol] += 1
        if s.symbol not in pair_wins:
            pair_wins[s.symbol] = [0, 0]
        if s.result == "win":
            pair_wins[s.symbol][0] += 1
        pair_wins[s.symbol][1] += 1

    profitable_pairs = {sym: total for sym, total in pair_rr.items() if total > 0}
    if profitable_pairs:
        most_profitable = max(profitable_pairs, key=profitable_pairs.get)
        most_profitable_rr = round(profitable_pairs[most_profitable], 2)
    else:
        most_profitable = "-"
        most_profitable_rr = 0
    most_traded = pair_trades.most_common(1)[0] if pair_trades else ("-", 0)

    best_wr_pair = "-"
    best_wr_pct = 0
    for sym, (w, t) in pair_wins.items():
        if t >= 3:
            wr = w / t * 100
            if wr > best_wr_pct:
                best_wr_pct = round(wr, 1)
                best_wr_pair = sym

    streak = 0
    max_streak = 0
    for s in sorted(closed, key=lambda x: x.created_at):
        if s.result == "win":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    week_rr: dict[str, float] = {}
    for s in closed:
        if s.created_at and s.rr_value is not None:
            week_key = s.created_at.strftime("%Y-W%W")
            week_rr[week_key] = week_rr.get(week_key, 0) + s.rr_value
    best_week = max(week_rr, key=week_rr.get) if week_rr else "-"
    best_week_rr = round(week_rr.get(best_week, 0), 2) if week_rr else 0

    top_edge_sym = "-"
    top_edge_rr = 0
    for sym, total_r in pair_rr.items():
        count = pair_trades[sym]
        if count >= 3:
            avg = total_r / count
            if avg > top_edge_rr:
                top_edge_rr = round(avg, 2)
                top_edge_sym = sym

    return {
        "total_signals": total_signals,
        "active_signals": len(active),
        "win_count": win_count,
        "loss_count": loss_count,
        "be_count": be_count,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "total_rr": total_rr,
        "avg_duration": avg_duration,
        "most_profitable": most_profitable,
        "most_profitable_rr": most_profitable_rr,
        "most_traded": most_traded[0],
        "most_traded_count": most_traded[1],
        "best_wr_pair": best_wr_pair,
        "best_wr_pct": best_wr_pct,
        "max_streak": max_streak,
        "best_week": best_week,
        "best_week_rr": best_week_rr,
        "top_edge_sym": top_edge_sym,
        "top_edge_rr": top_edge_rr,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tf: str = Query(default="4h"),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    tf = _parse_tf(tf)
    result = await db.execute(
        select(Signal).where(
            Signal.status.notin_(["waiting_entry", "pending_cisd"]),
            _tf_filter(tf),
        )
    )
    all_signals = result.scalars().all()

    stats = _build_dashboard_stats(all_signals)
    crypto_signals = [s for s in all_signals if s.market_type == "crypto"]
    global_signals = [s for s in all_signals if s.market_type in ("fx", "index", "metal", "oil")]

    stats_crypto = _build_dashboard_stats(crypto_signals)
    stats_global = _build_dashboard_stats(global_signals)

    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "user": user,
        "stats": stats,
        "stats_crypto": stats_crypto,
        "stats_global": stats_global,
        "global_markets_label": "Global Markets",
        "tf": tf,
        "tf_label": _tf_label(tf),
    })


# ──────────────────── Signals ────────────────────

ITEMS_PER_PAGE = 20
LOGS_PER_PAGE = 50

SIGNAL_SEGMENT_MARKETS = {
    "crypto": ["crypto"],
    "global": ["fx", "index", "metal", "oil"],
}
SIGNAL_SEGMENT_LABELS = {
    "crypto": "Crypto Signals",
    "global": "FX Signals",
}


async def _render_signals_page(
    request: Request,
    db: AsyncSession,
    *,
    segment: str | None = None,
    signals_base_path: str = "/signals",
    tab: str = "all",
    symbol: str = "",
    direction: str = "",
    market_type: list[str] | None = None,
    status: str = "",
    result_filter: str = "",
    date_from: str = "",
    date_to: str = "",
    page: int = 1,
    tf: str = "4h",
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    tf = _parse_tf(tf)
    arrival_date_col = Signal.created_at
    base_query = select(Signal).where(_tf_filter(tf))

    raw_market_types = request.query_params.getlist("market_type") or (market_type or [])
    normalized_market_values: list[str] = []
    for value in raw_market_types:
        if not value:
            continue
        # Bazi istemcilerde coklu secim tek parametrede "crypto,fx" gelebilir.
        normalized_market_values.extend([v for v in value.split(",") if v])

    selected_markets = [m.strip().lower() for m in normalized_market_values if m and m.strip()]
    allowed_markets = {"crypto", "fx", "index", "metal", "oil"}
    selected_markets = [m for m in selected_markets if m in allowed_markets]
    if not selected_markets and segment:
        selected_markets = list(SIGNAL_SEGMENT_MARKETS[segment])

    if selected_markets:
        base_query = base_query.where(Signal.market_type.in_(selected_markets))

    query = base_query

    if tab == "active":
        query = query.where(Signal.status == "active")
    elif tab == "waiting":
        query = query.where(Signal.status.in_(["waiting_entry", "pending_cisd"]))
    elif tab == "closed":
        query = query.where(Signal.status.in_(["expired", "breakeven"]))

    if symbol:
        query = query.where(Signal.symbol.ilike(f"%{symbol}%"))
    if direction:
        query = query.where(Signal.direction == direction.upper())
    if status:
        query = query.where(Signal.status == status)
    if result_filter:
        query = query.where(Signal.result == result_filter)
    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            query = query.where(arrival_date_col >= dt_from)
        except ValueError:
            pass
    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            query = query.where(arrival_date_col <= dt_to)
        except ValueError:
            pass

    count_query = select(func.count()).select_from(query.subquery())
    count_result = await db.execute(count_query)
    total = count_result.scalar()
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    status_priority = case(
        (Signal.status == "active", 0),
        (Signal.status == "waiting_entry", 1),
        (Signal.status == "pending_cisd", 2),
        (Signal.status == "breakeven", 3),
        (Signal.status == "expired", 4),
        else_=5,
    )
    query = query.order_by(status_priority.asc(), desc(arrival_date_col))
    query = query.offset((page - 1) * ITEMS_PER_PAGE).limit(ITEMS_PER_PAGE)
    result = await db.execute(query)
    signals = result.scalars().all()

    market_filter = [Signal.market_type.in_(selected_markets), _tf_filter(tf)] if selected_markets else [_tf_filter(tf)]

    active_count_r = await db.execute(
        select(func.count()).where(*market_filter, Signal.status == "active")
    )
    active_count = active_count_r.scalar()
    waiting_count_r = await db.execute(
        select(func.count()).where(
            *market_filter, Signal.status.in_(["waiting_entry", "pending_cisd"]),
        )
    )
    waiting_count = waiting_count_r.scalar()
    closed_count_r = await db.execute(
        select(func.count()).where(*market_filter, Signal.status.in_(["expired", "breakeven"]))
    )
    closed_count = closed_count_r.scalar()
    total_count = active_count + waiting_count + closed_count

    tf_active_counts: dict[str, int] = {}
    for tf_key in ("4h", "1d", "1h"):
        tf_filters = [_tf_filter(tf_key), Signal.status == "active"]
        if selected_markets:
            tf_filters.append(Signal.market_type.in_(selected_markets))
        cnt = await db.execute(select(func.count()).where(*tf_filters))
        tf_active_counts[tf_key] = int(cnt.scalar() or 0)

    page_title = SIGNAL_SEGMENT_LABELS.get(segment, "All Signals") if segment else "All Signals"
    active_page = f"signals_{segment}" if segment else "signals"

    return templates.TemplateResponse(request=request, name="signals.html", context={
        "user": user,
        "signals": signals,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "tab": tab,
        "active_count": active_count,
        "waiting_count": waiting_count,
        "closed_count": closed_count,
        "total_count": total_count,
        "segment": segment,
        "page_title": page_title,
        "active_page": active_page,
        "signals_base_path": signals_base_path,
        "tf": tf,
        "tf_label": _tf_label(tf),
        "tf_active_counts": tf_active_counts,
        "filters": {
            "symbol": symbol,
            "direction": direction,
            "market_types": selected_markets,
            "status": status,
            "result": result_filter,
            "date_from": date_from,
            "date_to": date_to,
        },
    })


@app.get("/signals/crypto", response_class=HTMLResponse)
async def signals_crypto_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    symbol: str = Query(default=""),
    direction: str = Query(default=""),
    market_type: list[str] = Query(default=[]),
    status: str = Query(default=""),
    result_filter: str = Query(default="", alias="result"),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    tf: str = Query(default="4h"),
):
    return await _render_signals_page(
        request,
        db,
        segment="crypto",
        signals_base_path="/signals/crypto",
        tab=tab,
        symbol=symbol,
        direction=direction,
        market_type=market_type,
        status=status,
        result_filter=result_filter,
        date_from=date_from,
        date_to=date_to,
        page=page,
        tf=tf,
    )


@app.get("/signals/global", response_class=HTMLResponse)
async def signals_global_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    symbol: str = Query(default=""),
    direction: str = Query(default=""),
    market_type: list[str] = Query(default=[]),
    status: str = Query(default=""),
    result_filter: str = Query(default="", alias="result"),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    tf: str = Query(default="4h"),
):
    return await _render_signals_page(
        request,
        db,
        segment="global",
        signals_base_path="/signals/global",
        tab=tab,
        symbol=symbol,
        direction=direction,
        market_type=market_type,
        status=status,
        result_filter=result_filter,
        date_from=date_from,
        date_to=date_to,
        page=page,
        tf=tf,
    )


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    symbol: str = Query(default=""),
    direction: str = Query(default=""),
    market_type: list[str] = Query(default=[]),
    status: str = Query(default=""),
    result_filter: str = Query(default="", alias="result"),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    tf: str = Query(default="4h"),
):
    return await _render_signals_page(
        request,
        db,
        tab=tab,
        symbol=symbol,
        direction=direction,
        market_type=market_type,
        status=status,
        result_filter=result_filter,
        date_from=date_from,
        date_to=date_to,
        page=page,
        tf=tf,
    )


# ──────────────────── Analytics Page ────────────────────

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tf: str = Query(default="4h"),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    tf = _parse_tf(tf)
    result = await db.execute(
        select(Signal).where(_tf_filter(tf)).order_by(Signal.created_at)
    )
    all_signals = result.scalars().all()

    closed = [s for s in all_signals if s.result is not None and s.rr_value is not None]

    equity_data = []
    cumulative = 0.0
    for s in closed:
        cumulative += s.rr_value
        equity_data.append({
            "date": s.created_at.strftime("%Y-%m-%d") if s.created_at else "",
            "rr": round(cumulative, 2),
        })

    results_count = {"win": 0, "loss": 0, "breakeven": 0}
    for s in closed:
        results_count[s.result] = results_count.get(s.result, 0) + 1

    direction_count = {"LONG": 0, "SHORT": 0}
    for s in closed:
        direction_count[s.direction] = direction_count.get(s.direction, 0) + 1

    market_count: dict[str, int] = {}
    for s in closed:
        market_count[s.market_type] = market_count.get(s.market_type, 0) + 1

    weekly_rr: dict[str, float] = {}
    for s in closed:
        if s.created_at:
            week_key = s.created_at.strftime("%Y-W%W")
            weekly_rr[week_key] = round(weekly_rr.get(week_key, 0) + s.rr_value, 2)

    weekly_labels = list(weekly_rr.keys())[-12:]
    weekly_values = [weekly_rr[k] for k in weekly_labels]

    symbol_perf: dict[str, dict] = {}
    for s in closed:
        if s.symbol not in symbol_perf:
            symbol_perf[s.symbol] = {"total_rr": 0, "count": 0, "wins": 0}
        symbol_perf[s.symbol]["total_rr"] += s.rr_value
        symbol_perf[s.symbol]["count"] += 1
        if s.result == "win":
            symbol_perf[s.symbol]["wins"] += 1

    top_symbols = sorted(symbol_perf.items(), key=lambda x: x[1]["total_rr"], reverse=True)[:10]
    top_sym_labels = [s[0] for s in top_symbols]
    top_sym_values = [round(s[1]["total_rr"], 2) for s in top_symbols]

    return templates.TemplateResponse(request=request, name="analytics.html", context={
        "user": user,
        "equity_data": equity_data,
        "results_count": results_count,
        "direction_count": direction_count,
        "market_count": market_count,
        "weekly_labels": weekly_labels,
        "weekly_values": weekly_values,
        "top_sym_labels": top_sym_labels,
        "top_sym_values": top_sym_values,
        "total_closed": len(closed),
        "tf": tf,
        "tf_label": _tf_label(tf),
    })


# ──────────────────── Scan Logs Page ────────────────────

@app.get("/logs", response_class=HTMLResponse)
async def logs_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    count_result = await db.execute(select(func.count()).select_from(EventLog))
    total = count_result.scalar() or 0
    total_pages = max(1, (total + LOGS_PER_PAGE - 1) // LOGS_PER_PAGE)

    logs_result = await db.execute(
        select(EventLog)
        .order_by(desc(EventLog.created_at))
        .offset((page - 1) * LOGS_PER_PAGE)
        .limit(LOGS_PER_PAGE)
    )
    logs = logs_result.scalars().all()

    return templates.TemplateResponse(request=request, name="logs.html", context={
        "user": user,
        "logs": logs,
        "page": page,
        "total_pages": total_pages,
        "total": total,
    })


# ──────────────────── Radar (Canli Izleme) ────────────────────

@app.get("/radar", response_class=HTMLResponse)
async def radar_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    tf = _parse_tf(request.query_params.get("tf"))
    return templates.TemplateResponse(request=request, name="radar.html", context={
        "user": user,
        "active_page": "radar",
        "tf": tf,
        "tf_label": _tf_label(tf),
    })


@app.get("/api/radar")
async def api_radar(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    from collections import Counter as _Counter

    from app.scanner import get_radar_snapshot

    tf = _parse_tf(request.query_params.get("tf"))
    snap = get_radar_snapshot(strategy=tf)
    ws = market_data.status()

    symbols = []
    for e in snap["symbols"]:
        meta = dict(RADAR_STATE_META.get(e["state"], {"label": e["state"], "color": "gray", "rank": 9}))
        if e["state"] == "bias_mismatch":
            meta["label"] = "1D bias opposite (structure/ICT)"
        symbols.append({
            "symbol": _fmt_ui_symbol(e["symbol"], e.get("market")),
            "market": (e.get("market") or "").upper(),
            "state": e["state"],
            "label": meta["label"],
            "color": meta["color"],
            "rank": meta["rank"],
            "direction": e.get("direction"),
            "score": e.get("score"),
            "bias": e.get("bias"),
            "weekly_bias": e.get("weekly_bias"),
            "rr": e.get("rr"),
            "entry": _fmt_price(e["entry"]) if e.get("entry") is not None else None,
            "sl": _fmt_price(e["sl"]) if e.get("sl") is not None else None,
            "tp": _fmt_price(e["tp"]) if e.get("tp") is not None else None,
            "smt": _fmt_ui_symbol(e.get("smt"), "crypto") if e.get("smt") else None,
            "pd": e.get("pd"),
            "c2_closed": e.get("c2_closed"),
            "ifvg": e.get("ifvg"),
            # "same_color": bool(e.get("same_color")),  # eski bilgi alani; UI'da gosterilmiyor
            "updated_at": _fmt_date_tsi(e["updated_at"]) if e.get("updated_at") else None,
        })
    symbols.sort(key=lambda x: (x["rank"], x["symbol"]))

    cnt = _Counter(s["state"] for s in symbols)
    summary = {
        "total": len(symbols),
        "waiting": cnt.get("waiting", 0),
        "potential": (
            cnt.get("no_cisd", 0) + cnt.get("c2_open", 0) + cnt.get("low_rr", 0) + cnt.get("missed", 0)
            + cnt.get("invalidated", 0)  # + cnt.get("same_color", 0)  # eski hard filter
            + cnt.get("tight_stop", 0) + cnt.get("bias_mismatch", 0)
            + cnt.get("cluster_limit", 0)
            + cnt.get("stale", 0) + cnt.get("same_bar_sl", 0)
            + cnt.get("past_sl", 0) + cnt.get("missed_quality", 0)
        ),
        "setups": sum(cnt.get(k, 0) for k in (
            "waiting", "c2_open", "no_cisd", "low_rr", "missed", "invalidated",
            # "same_color",  # eski hard filter
            "tight_stop", "bias_mismatch", "cluster_limit", "low_quality",
            "has_open", "duplicate", "corr_open", "stale", "same_bar_sl",
            "past_sl", "missed_quality",
        )),
        "idle": cnt.get("no_setup", 0) + cnt.get("no_data", 0),
    }

    return JSONResponse(content={
        "ws": {
            "connected": bool(ws.get("connected")),
            "running": bool(ws.get("running")),
            "last_message_at": ws.get("last_message_at"),
            "symbol_count": ws.get("symbol_count"),
            "subscription_count": ws.get("subscription_count"),
        },
        "last_update": _fmt_date_tsi(snap["last_update"]) if snap["last_update"] else None,
        "summary": summary,
        "symbols": symbols,
    })


# ──────────────────── Scanner Page ────────────────────

@app.get("/scanner", response_class=HTMLResponse)
async def scanner_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    from app.exchange import get_active_markets, is_weekend, to_display_symbol
    from app.telegram import is_configured as tg_configured

    active_markets = get_active_markets()
    markets_info = []
    total_count = 0
    for market, syms in active_markets.items():
        display_syms = [_fmt_ui_symbol(to_display_symbol(s), market) for s in syms]
        markets_info.append({
            "name": market.upper(),
            "exchange": "BingX",
            "symbols": display_syms,
        })
        total_count += len(syms)

    scan_mode = "Weekend (Crypto Only)" if is_weekend() else "Weekday (All Markets)"

    sched_status = get_scheduler_status()

    return templates.TemplateResponse(request=request, name="scanner.html", context={
        "user": user,
        "markets": markets_info,
        "symbol_count": total_count,
        "telegram_configured": tg_configured(),
        "scheduler": sched_status,
        "scan_mode": scan_mode,
    })


# ──────────────────── Scanner API ────────────────────

@app.post("/api/scan")
async def trigger_scan(request: Request, db: AsyncSession = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    if scan_state["running"]:
        return JSONResponse(status_code=409, content={"error": "Scan already running"})

    scan_state["running"] = True
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        raw_tf = (body or {}).get("tf") or request.query_params.get("tf")
        scan_tf = "all" if raw_tf in (None, "", "all", "both") else _parse_tf(raw_tf)
        scan_result = await run_scan(db, timeframe=scan_tf, source="manual", store=market_data.store)
        scan_state["last_run"] = datetime.now(timezone.utc).isoformat()
        scan_state["last_result"] = (
            f"{len(scan_result['new_setups'])} setup, "
            f"{len(scan_result['activated'])} active, "
            f"{len(scan_result['closed'])} closed, "
            f"{len(scan_result['breakeven'])} breakeven"
        )
        return JSONResponse(content={
            "success": True,
            "new_setups": len(scan_result["new_setups"]),
            "activated": len(scan_result["activated"]),
            "closed": len(scan_result["closed"]),
            "breakeven": len(scan_result["breakeven"]),
            "setups": [
                {
                    "symbol": _fmt_ui_symbol(s.symbol, getattr(s, "market_type", None)),
                    "direction": s.direction,
                    "purge_type": s.purge_type,
                }
                for s in scan_result["new_setups"]
            ],
            "active_signals": [
                {
                    "symbol": _fmt_ui_symbol(s.symbol, s.market_type),
                    "direction": s.direction,
                    "entry": s.entry_price, "sl": s.stop_loss, "tp": s.take_profit,
                }
                for s in scan_result["activated"]
            ],
            "breakeven_symbols": [_fmt_ui_symbol(sym, "crypto") for sym in scan_result["breakeven"]],
        })
    except Exception as e:
        log.exception("Scan failed")
        scan_state["last_result"] = f"Error: {e}"
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        scan_state["running"] = False


@app.get("/api/scan-status")
async def get_scan_status(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return JSONResponse(content=scan_state)


@app.post("/api/recalc-scores")
async def recalc_scores(request: Request, db: AsyncSession = Depends(get_db)):
    """Mevcut sinyallerin kalite skorlarını yeniden hesapla."""
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    import httpx
    import pandas as pd

    from app.config import BINGX_REST_BASE, BOOTSTRAP_LIMITS
    from app.crt_engine import _calc_live_setup_bias, check_smt_divergence, detect_pd_arrays
    from app.exchange import (
        correlated_symbol,
        fetch_ohlcv,
        from_display_symbol,
        to_display_symbol,
    )

    signals_q = await db.execute(select(Signal))
    all_sigs = signals_q.scalars().all()
    if not all_sigs:
        return JSONResponse(content={"updated": 0})

    updated = 0
    async with httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0) as client:
        for sig in all_sigs:
            try:
                bingx_symbol, _market = from_display_symbol(sig.symbol)

                from app.scanner import STRATEGY_CFG, STRATEGY_4H
                cfg = STRATEGY_CFG.get(sig.timeframe or STRATEGY_4H, STRATEGY_CFG[STRATEGY_4H])
                htf = cfg["htf"]
                ltf = cfg["ltf"]
                smt_hours = float(cfg["smt_window_hours"])
                df_4h = await fetch_ohlcv(bingx_symbol, htf, limit=60, client=client)
                if df_4h is None or len(df_4h) < 16:
                    continue
                df_4h = df_4h.sort_index()

                htf_bias = sig.htf_bias or "NEUTRAL"

                # Skoru olusturma anindaki mantikla ayni tut: setup'in CRT mumunu
                # crt_bar_time uzerinden bul ve canli skorlayiciyi uygula.
                if sig.crt_bar_time is None:
                    continue
                crt_ts = pd.Timestamp(sig.crt_bar_time)
                crt_ts = crt_ts.tz_localize("UTC") if crt_ts.tzinfo is None else crt_ts.tz_convert("UTC")
                if crt_ts not in df_4h.index:
                    continue
                crt_idx = df_4h.index.get_loc(crt_ts)
                if crt_idx + 1 >= len(df_4h):
                    continue

                # Purge indeksi (C1'den 1-2 mum sonra olabilir) purge_time'dan bulunur.
                purge_idx = None
                if sig.purge_time is not None:
                    pt = pd.Timestamp(sig.purge_time)
                    pt = pt.tz_localize("UTC") if pt.tzinfo is None else pt.tz_convert("UTC")
                    if pt in df_4h.index:
                        purge_idx = df_4h.index.get_loc(pt)

                # PD array tespiti (purge C2) -> kademeli skor (major + FVG|OB)
                df_1d = await fetch_ohlcv(bingx_symbol, "1d", limit=BOOTSTRAP_LIMITS.get("1d", 60), client=client)
                if df_1d is not None:
                    df_1d = df_1d.sort_index()
                pd_labels = detect_pd_arrays(df_4h, df_1d, crt_idx, sig.direction, purge_idx=purge_idx)

                df_ltf = await fetch_ohlcv(
                    bingx_symbol, ltf, limit=BOOTSTRAP_LIMITS.get(ltf, 200), client=client,
                )
                bias, score = _calc_live_setup_bias(
                    df_4h, crt_idx, sig.direction, htf_bias,
                    pd_labels=pd_labels, purge_idx=purge_idx, df_1d=df_1d,
                    df_ltf=df_ltf, timeframe=sig.timeframe or STRATEGY_4H,
                )

                # SMT bonusu (detection ile ayni mantik): korele parite 15M
                # divergence varsa +SMT_QUALITY_BONUS (max 10). purge_time'a gore sabitlenir.
                corr = correlated_symbol(bingx_symbol)
                smt_pair = None
                if corr is not None and sig.purge_time is not None:
                    df_15m = df_ltf
                    corr_15m = await fetch_ohlcv(corr, ltf, limit=BOOTSTRAP_LIMITS.get(ltf, 200), client=client)
                    if check_smt_divergence(
                        df_15m, corr_15m, sig.direction, sig.purge_time, window_hours=smt_hours,
                    ):
                        smt_pair = to_display_symbol(corr)
                        score = min(MAX_QUALITY_SCORE, int(score) + SMT_QUALITY_BONUS)
                        if score >= 7:
                            bias = "BULLISH" if sig.direction == "LONG" else "BEARISH"
                        elif score <= 3:
                            bias = "BEARISH" if sig.direction == "LONG" else "BULLISH"

                sig.bias = bias
                sig.bias_score = score
                sig.smt_pair = smt_pair
                sig.pd_array = ",".join(pd_labels) if pd_labels else None
                if purge_idx is not None:
                    sig.c2_closed = bool(purge_idx < (len(df_4h) - 1))
                updated += 1
            except Exception:
                continue

        await db.commit()

    return JSONResponse(content={"updated": updated})


# ──────────────────── Telegram API ────────────────────

@app.post("/api/telegram-test")
async def telegram_test(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    from app.telegram import is_configured, send_test_message  # noqa: E402

    if not is_configured():
        return JSONResponse(content={
            "success": False,
            "error": "Telegram token or chat ID is not configured. Check your .env file.",
        })

    ok = await send_test_message()
    return JSONResponse(content={
        "success": ok,
        "error": None if ok else "Message could not be sent. Check your token and chat ID.",
    })
