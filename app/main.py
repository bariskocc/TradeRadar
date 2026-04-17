import logging
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from fastapi import FastAPI, Request, Form, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BASE_DIR
from app.database import init_db, get_db
from app.models import Signal, ScanLog
from app.auth import verify_credentials, create_access_token, get_current_user
from app.scanner import run_scan
from app.scheduler import start_scheduler, stop_scheduler, get_scheduler_status

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
    return tsi.strftime('%d.%m.%Y %H:%M')


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
    if not entry or not sl or not tp:
        return None
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return None
    return round(reward / risk, 1)

scan_state: dict = {"running": False, "last_run": None, "last_result": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="TradeRadar", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")
templates.env.filters["fmt_price"] = _fmt_price
templates.env.filters["fmt_date_tsi"] = _fmt_date_tsi
templates.env.globals["calc_rr_ratio"] = _calc_rr_ratio
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

    total = len(closed)
    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(breakevens)
    win_rate = round((win_count / total * 100), 1) if total > 0 else 0

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
        "total_signals": total,
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
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    result = await db.execute(select(Signal))
    all_signals = result.scalars().all()

    stats = _build_dashboard_stats(all_signals)
    crypto_signals = [s for s in all_signals if s.market_type == "crypto"]
    global_signals = [s for s in all_signals if s.market_type in ("fx", "index", "metal")]

    stats_crypto = _build_dashboard_stats(crypto_signals)
    stats_global = _build_dashboard_stats(global_signals)

    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "user": user,
        "stats": stats,
        "stats_crypto": stats_crypto,
        "stats_global": stats_global,
        "global_markets_label": "Global Markets",
    })


# ──────────────────── All Signals ────────────────────

ITEMS_PER_PAGE = 20
LOGS_PER_PAGE = 50


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    symbol: str = Query(default=""),
    direction: str = Query(default=""),
    market_type: str = Query(default=""),
    status: str = Query(default=""),
    result_filter: str = Query(default="", alias="result"),
    date_from: str = Query(default=""),
    date_to: str = Query(default=""),
    page: int = Query(default=1, ge=1),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    arrival_date_col = Signal.created_at
    query = select(Signal).where(Signal.status != "pending_cisd")

    if tab == "active":
        query = query.where(Signal.status == "active")
    elif tab == "closed":
        query = query.where(Signal.status.in_(["expired", "breakeven"]))

    if symbol:
        query = query.where(Signal.symbol.ilike(f"%{symbol}%"))
    if direction:
        query = query.where(Signal.direction == direction.upper())
    if market_type:
        query = query.where(Signal.market_type == market_type)
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
        (Signal.status == "pending_cisd", 1),
        (Signal.status == "breakeven", 2),
        (Signal.status == "expired", 3),
        else_=4,
    )
    query = query.order_by(status_priority.asc(), desc(arrival_date_col))
    query = query.offset((page - 1) * ITEMS_PER_PAGE).limit(ITEMS_PER_PAGE)
    result = await db.execute(query)
    signals = result.scalars().all()

    active_count_r = await db.execute(select(func.count()).where(Signal.status == "active"))
    active_count = active_count_r.scalar()
    total_count_r = await db.execute(
        select(func.count()).where(Signal.status != "pending_cisd")
    )
    total_count = total_count_r.scalar()
    closed_count = total_count - active_count

    return templates.TemplateResponse(request=request, name="signals.html", context={
        "user": user,
        "signals": signals,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "tab": tab,
        "active_count": active_count,
        "closed_count": closed_count,
        "total_count": total_count,
        "filters": {
            "symbol": symbol,
            "direction": direction,
            "market_type": market_type,
            "status": status,
            "result": result_filter,
            "date_from": date_from,
            "date_to": date_to,
        },
    })


# ──────────────────── Analytics Page ────────────────────

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    result = await db.execute(select(Signal).order_by(Signal.created_at))
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

    count_result = await db.execute(select(func.count()).select_from(ScanLog))
    total = count_result.scalar() or 0
    total_pages = max(1, (total + LOGS_PER_PAGE - 1) // LOGS_PER_PAGE)

    logs_result = await db.execute(
        select(ScanLog)
        .order_by(desc(ScanLog.started_at))
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


# ──────────────────── Scanner Page ────────────────────

@app.get("/scanner", response_class=HTMLResponse)
async def scanner_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    from app.exchange import SYMBOLS_BY_MARKET, EXCHANGE_PER_MARKET, to_display_symbol
    from app.telegram import is_configured as tg_configured

    markets_info = []
    total_count = 0
    for market, syms in SYMBOLS_BY_MARKET.items():
        exchange_id = EXCHANGE_PER_MARKET.get(market, "?")
        display_syms = [_fmt_ui_symbol(to_display_symbol(s), market) for s in syms]
        markets_info.append({
            "name": market.upper(),
            "exchange": exchange_id.capitalize(),
            "symbols": display_syms,
        })
        total_count += len(syms)

    sched_status = get_scheduler_status()

    return templates.TemplateResponse(request=request, name="scanner.html", context={
        "user": user,
        "markets": markets_info,
        "symbol_count": total_count,
        "telegram_configured": tg_configured(),
        "scheduler": sched_status,
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
        scan_result = await run_scan(db, timeframe="4h", source="manual")
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

    from app.crt_engine import _calc_bias, compute_htf_bias
    from app.exchange import create_exchanges, close_exchanges, fetch_ohlcv, from_display_symbol

    signals_q = await db.execute(select(Signal))
    all_sigs = signals_q.scalars().all()
    if not all_sigs:
        return JSONResponse(content={"updated": 0})

    exchanges = await create_exchanges()
    updated = 0
    try:
        for sig in all_sigs:
            try:
                ccxt_symbol, exchange_id = from_display_symbol(sig.symbol)
                exchange = exchanges.get(exchange_id)
                if not exchange:
                    continue

                df_4h = await fetch_ohlcv(exchange, ccxt_symbol, "4h", limit=50)
                if len(df_4h) < 16:
                    continue

                htf_bias = sig.htf_bias or "NEUTRAL"

                idx = len(df_4h) - 3
                bias, score = _calc_bias(df_4h, idx, sig.direction, htf_bias)
                sig.bias = bias
                sig.bias_score = score
                updated += 1
            except Exception:
                continue

        await db.commit()
    finally:
        await close_exchanges(exchanges)

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
