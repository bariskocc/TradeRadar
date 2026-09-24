import asyncio
import json
import logging
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from urllib.parse import urlencode

from fastapi import FastAPI, Request, Form, Depends, Query, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, case, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import BASE_DIR
from app.log_report import build_report
from app.logging_config import recent_issues, setup_logging
from app.database import init_db, get_db
from app.models import Signal, EventLog, SetupJournal, PaperTrade
from app.watchlist import build_watchlist
from app import paper_trades as paper
from app import tv_import as tv
from app.setup_journal import PRE_SETUP_STAGES
from app.session import NY as FX_NY, SESSION_HOUR as FX_SESSION_HOUR
from app.telegram import is_configured as tg_is_configured
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
    "stale":          {"label": "Retest too old (stale)",     "color": "yellow", "rank": 4},
    "week_gap":       {"label": "CISD before week close",     "color": "yellow", "rank": 4},
    "week_close":     {"label": "Cancelled (week close)",      "color": "yellow", "rank": 4},
    "same_bar_sl":    {"label": "SL before fill",             "color": "yellow", "rank": 4},
    # "same_color":   {"label": "CRT/purge same color",      "color": "red",    "rank": 4},
    "bias_mismatch":  {"label": "1D bias opposite",           "color": "orange", "rank": 5},
    "cluster_limit":  {"label": "Same-direction cluster limit", "color": "orange", "rank": 6},
    "has_open":       {"label": "Open signal exists",         "color": "purple", "rank": 7},
    "corr_open":      {"label": "Correlated pair open",       "color": "purple", "rank": 8},
    "duplicate":      {"label": "Setup already saved",        "color": "purple", "rank": 9},
    "low_quality":    {"label": "Low quality (score<7)",      "color": "gray",   "rank": 10},
    # detect_crt_setup'in setup'a cevirmeden eledigi CRT (setup yoksa en guncel adayin nedeni).
    "c2_wrong_color": {"label": "CRT rejected: C2 wrong color",    "color": "dim", "rank": 11},
    "c2_breakout":    {"label": "CRT rejected: C2 closed outside", "color": "dim", "rank": 11},
    "c1_stale":       {"label": "CRT rejected: C1 extreme taken",  "color": "dim", "rank": 11},
    "range_atr":      {"label": "CRT rejected: C1 range vs ATR",   "color": "dim", "rank": 11},
    "sweep_small":    {"label": "CRT rejected: sweep too small",   "color": "dim", "rank": 11},
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
templates.env.globals["fmt_money"] = paper.fmt_money


def _static_version(name: str = "css/app.css") -> str:
    """Statik dosyanin mtime damgasi -- sablonlarda `?v=` olarak kullanilir.

    Tailwind'i yeniden derlemek yetmiyordu: StaticFiles `cache-control` yazmadigi icin tarayici
    sezgisel onbellekle eski app.css'i gunlerce kullanabiliyor ve YENI siniflar uygulanmiyor
    (20.09: takvimde `grid-cols-7` gelmedigi icin hucreler tam genislikte alt alta dizildi --
    `display:grid` var, `grid-template-columns` yok). Damga degisince tarayici yeniden indirir.
    """
    try:
        return str(int((BASE_DIR / "app" / "static" / name).stat().st_mtime))
    except OSError:
        return "0"


templates.env.globals["static_v"] = _static_version


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


# Tek ekran: saglik seridi (canli) + donem ozeti (haftalik/aylik) + strateji kartlari
# (tum zamanlar, Crypto/FX) + acik islemler / son kapananlar / potansiyel 1D / dikkat (canli).
# Canli parcalar /dashboard/live ile 30 sn'de bir yenilenir; istatistikler sayfa acilisinda.

_TSI = timezone(TSI_OFFSET)

# Kismi kar (%50'de yari + BE) vs sadece BE karsilastirmasi: bu kadar kismi karli islem birikince
# Dashboard "Dikkat" panosu olcumu hatirlatir (scripts/partial_vs_be.py). Karar kurali ve sinirlar:
# IZLEME.md -> "Kismi kar vs BE-only". Olcum yapilinca esigi yukselt ya da hatirlatmayi kaldir.
PARTIAL_REVIEW_MIN_TRADES = 25
# LONG/SHORT ayrismasi: her YONDE bu kadar kapali islem birikince Dikkat panosu olcumu hatirlatir
# (scripts/direction_stat.py). Pencere 08.09.2026'da basliyor -- ilk kapali islem, izlemenin basi.
# Karar kurali ve sinirlar: IZLEME.md -> "LONG/SHORT ayrismasi". Karar verilince esigi yukselt
# ya da hatirlatmayi kaldir, yoksa surekli bagirir.
DIRECTION_REVIEW_MIN_TRADES = 30
DIRECTION_REVIEW_SINCE = datetime(2026, 9, 8)
_DASH_TFS = ("4h", "1d", "1h")
# "FX" = kripto disi tum seans sembolleri (fx/metal/endeks/petrol).
_DASH_MARKETS = (
    ("crypto", "Crypto", ("crypto",)),
    ("fx", "FX", ("fx", "index", "metal", "oil")),
)
_TR_MONTHS = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
              "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
_LOW_SAMPLE_PERIOD = 10      # donem ozetinde "az ornek" rozeti
_LOW_SAMPLE_ALL = 20         # strateji kartinda "az ornek" rozeti
_WS_STALE_SEC = 120          # bu kadar suredir WS mesaji yoksa akis durmus sayilir
_EXIT_LABELS = {"tp": "TP", "sl": "SL", "be": "BE", "trail": "Trail", "week_close": "Hafta kapanışı"}


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _signal_closed_at(s) -> datetime | None:
    """Kapanis ani (UTC aware); closed_at bos eski kayitta dolum (yoksa CISD) + sure."""
    if s.result is None:
        return None
    if getattr(s, "closed_at", None) is not None:
        return _utc(s.closed_at)
    ref = _utc(s.entry_filled_time or s.cisd_time)
    if ref is None or s.duration_hours is None:
        return None
    return ref + timedelta(hours=float(s.duration_hours))


def _fmt_span(td: timedelta) -> str:
    mins = max(0, int(td.total_seconds() // 60))
    days, rem = divmod(mins, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}g {hours}s"
    if hours:
        return f"{hours}s {mins:02d}dk"
    return f"{mins}dk"


def _fmt_ago(seconds: float | None) -> str:
    if seconds is None:
        return "yok"
    if seconds < 60:
        return f"{int(seconds)} sn önce"
    return f"{_fmt_span(timedelta(seconds=seconds))} önce"


def _period_range(kind: str, offset: int, now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """Donem [start, end) UTC + etiket. Hafta: Pazartesi 00:00 TSI; ay: takvim ayi (TSI)."""
    now_tsi = (now or datetime.now(timezone.utc)).astimezone(_TSI)
    if kind == "month":
        idx = now_tsi.year * 12 + (now_tsi.month - 1) + offset
        year, month0 = divmod(idx, 12)
        end_year, end_month0 = divmod(idx + 1, 12)
        start = datetime(year, month0 + 1, 1, tzinfo=_TSI)
        end = datetime(end_year, end_month0 + 1, 1, tzinfo=_TSI)
        label = f"{_TR_MONTHS[month0]} {year}"
    else:
        monday = (now_tsi - timedelta(days=now_tsi.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        start = monday + timedelta(weeks=offset)
        end = start + timedelta(weeks=1)
        last = end - timedelta(days=1)
        if start.month == last.month:
            label = f"{start.day}–{last.day} {_TR_MONTHS[start.month - 1]} {last.year}"
        else:
            label = (f"{start.day} {_TR_MONTHS[start.month - 1]} – "
                     f"{last.day} {_TR_MONTHS[last.month - 1]} {last.year}")
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc), label


def _period_summary(closed: list[tuple[datetime, Signal]], start: datetime, end: datetime) -> dict:
    """Donemde KAPANAN islemlerin ozeti (acik islemler dahil degil)."""
    rows = sorted((c for c in closed if start <= c[0] < end), key=lambda c: c[0])
    sigs = [s for _, s in rows]
    rs = [float(s.rr_value or 0.0) for s in sigs]
    n = len(sigs)
    wins = sum(1 for s in sigs if s.result == "win")
    losses = sum(1 for s in sigs if s.result == "loss")
    total = round(sum(rs), 2)

    def _pick(s):
        return {
            "symbol": _fmt_ui_symbol(s.symbol, s.market_type),
            "tf_label": _tf_label(s.timeframe or "4h"),
            "r": round(float(s.rr_value or 0.0), 2),
        }

    def _group(pred):
        vals = [float(s.rr_value or 0.0) for s in sigs if pred(s)]
        return {"r": round(sum(vals), 2), "n": len(vals)}

    by_r = lambda s: float(s.rr_value or 0.0)  # noqa: E731
    return {
        "n": n,
        "total_r": total,
        "wins": wins,
        "losses": losses,
        "be": n - wins - losses,
        "win_rate": round(wins / n * 100) if n else None,
        "avg_r": round(total / n, 2) if n else None,
        "best": _pick(max(sigs, key=by_r)) if n else None,
        "worst": _pick(min(sigs, key=by_r)) if n > 1 else None,
        "by_tf": [
            {"label": _tf_label(tf), **_group(lambda s, tf=tf: (s.timeframe or "4h") == tf)}
            for tf in _DASH_TFS
        ],
        "by_market": [
            {"label": label, **_group(lambda s, mk=mk: s.market_type in mk)}
            for _, label, mk in _DASH_MARKETS
        ],
        "low_sample": n < _LOW_SAMPLE_PERIOD,
    }


async def _period_activity(db: AsyncSession, start: datetime, end: datetime) -> dict:
    """Motor aktivitesi: donemde ilk gorulen setup'lar ve sinyale donusenler (Setup Journal), dolumlar."""
    s0, e0 = start.replace(tzinfo=None), end.replace(tzinfo=None)
    # Motorun CRT saymadigi adaylar (PRE_SETUP_STAGES) "setup gorundu" sayisina girmez.
    in_range = (
        SetupJournal.first_seen >= s0,
        SetupJournal.first_seen < e0,
        SetupJournal.best_stage.notin_(tuple(PRE_SETUP_STAGES)),
    )
    setups = (await db.execute(select(func.count()).select_from(SetupJournal).where(*in_range))).scalar() or 0
    signals = (await db.execute(
        select(func.count()).select_from(SetupJournal)
        .where(*in_range, SetupJournal.best_stage.in_(("waiting", "week_close")))
    )).scalar() or 0
    filled = (await db.execute(
        select(func.count()).select_from(Signal)
        .where(Signal.entry_filled_time >= s0, Signal.entry_filled_time < e0)
    )).scalar() or 0
    since = _utc((await db.execute(select(func.min(SetupJournal.first_seen)))).scalar())
    return {
        "setups": setups,
        "signals": signals,
        "filled": filled,
        "journal_since": since.astimezone(_TSI).strftime("%d.%m.%Y") if since else None,
        # Journal donem basindan sonra basladiysa setup sayilari eksik.
        "partial": since is None or since > start,
    }


def _strategy_cards(all_signals, closed, open_tf_counts, now: datetime) -> list[dict]:
    """Strateji basina tum zamanlar istatistigi: toplam + Crypto/FX + son 7 gun."""
    cards = []
    for tf in _DASH_TFS:
        tf_sigs = [s for s in all_signals if (s.timeframe or "4h") == tf]
        total = _build_dashboard_stats(tf_sigs)
        markets = {
            key: _build_dashboard_stats([s for s in tf_sigs if s.market_type in mk])
            for key, _, mk in _DASH_MARKETS
        }
        week = [
            float(s.rr_value or 0.0) for ca, s in closed
            if (s.timeframe or "4h") == tf and ca >= now - timedelta(days=7)
        ]
        n_closed = total["win_count"] + total["loss_count"] + total["be_count"]
        cards.append({
            "tf": tf,
            "label": _tf_label(tf),
            "total": total,
            "crypto": markets["crypto"],
            "fx": markets["fx"],
            "closed": n_closed,
            "last7_r": round(sum(week), 2) if week else None,
            "last7_n": len(week),
            "open_count": int(open_tf_counts.get(tf, 0)),
            "low_sample": n_closed < _LOW_SAMPLE_ALL,
        })
    return cards


def _fx_week_state(now: datetime) -> dict:
    """Seans sembolleri haftasi: Cuma 17:00 NY kapanis, Pazar 17:00 NY acilis (app.session).

    Kapanis her sembolde ayni an. Acilis **spot FX'in** acilisidir: CME grubu
    (XAU/XAG/US100/US500/petrol) 18:00 NY'de, yani bir saat sonra acilir - serit
    FX evrenini gosterdigi icin tek geri sayim birakildi (bkz. session.session_hour).
    """
    ny = now.astimezone(FX_NY)
    wd, hour = ny.weekday(), ny.hour
    closed = (wd == 4 and hour >= FX_SESSION_HOUR) or wd == 5 or (wd == 6 and hour < FX_SESSION_HOUR)
    days = (6 - wd) % 7 if closed else (4 - wd) % 7
    target = (ny + timedelta(days=days)).replace(hour=FX_SESSION_HOUR, minute=0, second=0, microsecond=0)
    # Ayni ZoneInfo'lu iki aware datetime farki DST'yi yok sayar; UTC'de cikar.
    left = target.astimezone(timezone.utc) - now
    if closed:
        return {"open": False, "soon": False, "label": f"FX kapalı · açılışa {_fmt_span(left)}"}
    return {"open": True, "soon": left < timedelta(hours=6), "label": f"FX hafta kapanışına {_fmt_span(left)}"}


async def _dashboard_live_context(db: AsyncSession) -> dict:
    """Saglik seridi + acik islemler + son kapananlar + potansiyel 1D + dikkat listesi."""
    now = datetime.now(timezone.utc)
    st = market_data.status()
    issues = recent_issues(24.0)
    running, connected = bool(st.get("running")), bool(st.get("connected"))
    last_msg = st.get("last_message_dt")
    msg_age = (now - last_msg).total_seconds() if last_msg else None
    stale = running and connected and (msg_age is None or msg_age > _WS_STALE_SEC)
    booting = running and st.get("bootstrap_done_at") is None
    if not running:
        status, status_label = "down", "Bot çalışmıyor"
    elif booting:
        status, status_label = "warn", "Veri yükleniyor"
    elif not connected:
        status, status_label = "warn", "WS bağlı değil"
    elif stale:
        status, status_label = "warn", "Veri akışı durdu"
    elif issues["errors"]:
        status, status_label = "warn", "Hata kaydı var"
    else:
        status, status_label = "ok", "Sağlıklı"

    totals = (await db.execute(
        select(func.count(), func.sum(Signal.rr_value)).where(Signal.result.isnot(None))
    )).one()
    started = st.get("started_at")
    health = {
        "status": status,
        "status_label": status_label,
        "uptime": _fmt_span(now - started) if started else "—",
        "connected": connected,
        "stale": stale,
        "last_msg_label": _fmt_ago(msg_age),
        "symbols": st.get("symbol_count", 0),
        "subs": st.get("subscription_count", 0),
        "ws_disconnects": st.get("ws_disconnects", 0),
        "warnings": issues["warnings"],
        "errors": issues["errors"],
        "telegram": tg_is_configured(),
        "fx": _fx_week_state(now),
        "totals_n": int(totals[0] or 0),
        "totals_r": round(float(totals[1] or 0.0), 2),
    }

    open_ctx = await _open_signals_context(db, "all", "all")
    open_rows = open_ctx["rows"][:8]

    res = await db.execute(
        select(Signal).where(Signal.result.isnot(None)).order_by(Signal.closed_at.desc()).limit(6)
    )
    recent_closed = []
    for s in res.scalars().all():
        ca = _signal_closed_at(s)
        recent_closed.append({
            "sig": s,
            "tf_label": _tf_label(s.timeframe or "4h"),
            "r": float(s.rr_value) if s.rr_value is not None else None,
            "how": _EXIT_LABELS.get(s.exit_reason or "", {"win": "Kazanç", "loss": "Kayıp"}.get(s.result, "BE")),
            "closed_label": ca.astimezone(_TSI).strftime("%d.%m %H:%M") if ca else "-",
        })

    # Potansiyel 1D: CISD onayli, sinyale donusmus ama dolmamis 1D setup'lar (Telegram bildirimiyle ayni kume).
    potentials = []
    for r in open_ctx["rows"]:
        s = r["sig"]
        if r["tf"] != "1d" or s.status == "active" or not s.cisd_confirmed:
            continue
        left = None
        if r["c2_open"] and r["c2_close_at"] is not None:
            left = _fmt_span(r["c2_close_at"].replace(tzinfo=timezone.utc) - now)
        potentials.append({"sig": s, "c2_open": r["c2_open"], "c2_left": left or "-", "to_entry_r": r["to_entry_r"]})

    attention: list[dict] = []

    def _add(level: str, ts: datetime | None, text: str) -> None:
        ts = ts or now
        attention.append({
            "level": level, "ts": ts,
            "ts_label": ts.astimezone(_TSI).strftime("%d.%m %H:%M"),
            "text": text,
        })

    if not running:
        _add("error", None, "Bot çalışmıyor — sunucu başlatılmamış ya da durmuş.")
    elif not connected:
        _add("error", None, "BingX WebSocket bağlı değil — fiyat akışı yok, TP/SL takibi gecikir.")
    elif stale:
        _add("warn", last_msg, f"Son WebSocket verisi {_fmt_ago(msg_age)} geldi; akış durmuş olabilir.")
    last_drop = st.get("last_disconnect_at")
    if last_drop and now - last_drop < timedelta(hours=24):
        _add("warn", last_drop, f"WebSocket koptu ({st.get('last_disconnect_reason') or 'neden yok'}).")
    backfills = await db.execute(
        select(EventLog)
        .where(
            EventLog.event_type == "filled",
            EventLog.message.like("%gecmis retest%"),
            EventLog.created_at >= (now - timedelta(hours=48)).replace(tzinfo=None),
        )
        .order_by(EventLog.created_at.desc())
        .limit(5)
    )
    for e in backfills.scalars().all():
        _add("info", _utc(e.created_at),
             f"Geçmiş dolum (BACKFILL): {_fmt_ui_symbol(e.symbol, e.market_type)} {e.direction or ''} — "
             "dolum ve sonrası gerçek mi kontrol et.")
    for item in issues["items"][-6:]:
        _add("error" if item["level"] == "ERROR" else "warn", item["ts"], f"{item['logger']}: {item['msg']}")
    # Kismi kar vs sadece BE olcumu: esik dolunca KENDILIGINDEN hatirlat (kimse takvim tutmasin).
    # Olcum yapilip IZLEME.md'ye yazildiginda bu esik yukseltilir ya da blok kaldirilir.
    partial_done = await db.scalar(
        select(func.count(Signal.id)).where(
            Signal.status == "expired",
            Signal.rr_value.is_not(None),
            Signal.partial_size.is_not(None),
            Signal.partial_rr.is_not(None),
        )
    )
    if (partial_done or 0) >= PARTIAL_REVIEW_MIN_TRADES:
        _add("info", None,
             f"Kısmi kâr vs sadece BE: {partial_done} işlem birikti — "
             "`python scripts/partial_vs_be.py` ile ölç, sonucu IZLEME.md'ye yaz.")
    # LONG/SHORT ayrismasi olcumu: iki yon de esigi doldurunca hatirlat. Betik esigin altinda
    # zaten "KARAR YOK" basar; buradaki sayac yalnizca "artik bakilabilir" demek.
    dir_rows = await db.execute(
        select(Signal.direction, func.count(Signal.id))
        .where(
            Signal.closed_at.is_not(None),
            Signal.closed_at >= DIRECTION_REVIEW_SINCE,
            Signal.rr_value.is_not(None),
        )
        .group_by(Signal.direction)
    )
    dir_counts = {d: n for d, n in dir_rows.all()}
    n_long, n_short = dir_counts.get("LONG", 0), dir_counts.get("SHORT", 0)
    if min(n_long, n_short) >= DIRECTION_REVIEW_MIN_TRADES:
        _add("info", None,
             f"LONG/SHORT ayrışması: {n_long} LONG / {n_short} SHORT işlem birikti — "
             "`python scripts/direction_stat.py` ile ölç, sonucu IZLEME.md'ye yaz.")
    attention.sort(key=lambda a: a["ts"], reverse=True)

    return {
        "health": health,
        "open_rows": open_rows,
        "open_more": max(0, len(open_ctx["rows"]) - len(open_rows)),
        "open_counts": open_ctx["counts"],
        "open_r": open_ctx["open_r"],
        "open_tf_counts": open_ctx["counts"]["tf"],
        "recent_closed": recent_closed,
        "potentials": potentials,
        "attention": attention[:8],
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    period: str = Query(default="week"),
    offset: int = Query(default=0),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    kind = "month" if period == "month" else "week"
    offset = max(-520, min(0, offset))
    now = datetime.now(timezone.utc)
    start, end, label = _period_range(kind, offset, now)
    prev_start, prev_end, _ = _period_range(kind, offset - 1, now)

    result = await db.execute(
        select(Signal).where(Signal.status.notin_(["waiting_entry", "pending_cisd"]))
    )
    all_signals = result.scalars().all()
    closed = [
        (ca, s) for s in all_signals
        if s.rr_value is not None and (ca := _signal_closed_at(s)) is not None
    ]

    summary = _period_summary(closed, start, end)
    prev = _period_summary(closed, prev_start, prev_end)
    summary["prev_total_r"] = prev["total_r"] if prev["n"] else None
    summary["delta_r"] = round(summary["total_r"] - prev["total_r"], 2) if prev["n"] else None

    # "Tum zamanlar" = DB'deki ilk sinyal kaydindan beri; baslikta yazsin ki aralik belirsiz kalmasin.
    first_created = _utc((await db.execute(select(func.min(Signal.created_at)))).scalar())

    live = await _dashboard_live_context(db)
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "user": user,
        "stats_since": first_created.astimezone(_TSI).strftime("%d.%m.%Y") if first_created else None,
        "period": {"kind": kind, "offset": offset, "label": label},
        "summary": summary,
        "activity": await _period_activity(db, start, end),
        "strategies": _strategy_cards(all_signals, closed, live["open_tf_counts"], now),
        **live,
    })


@app.get("/dashboard/live", response_class=HTMLResponse)
async def dashboard_live(request: Request, db: AsyncSession = Depends(get_db)):
    """Dashboard'un 30 sn'lik yenilemesi: saglik seridi + alt pano (HTML parcasi)."""
    user = get_current_user(request)
    if not user:
        return HTMLResponse(status_code=401, content="")
    ctx = await _dashboard_live_context(db)
    return templates.TemplateResponse(request=request, name="dashboard_live.html", context=ctx)


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
    tf: str = "all",
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Tek sayfada uc strateji (TF sutunu); tf yalniz istege bagli filtre (14.09).
    tf = (tf or "all").strip().lower()
    tf = tf if tf in ("4h", "1d", "1h") else "all"
    arrival_date_col = Signal.created_at

    # Arrival date varsayilani: bu haftanin Pazartesi'si (TSI). Parametre HIC yoksa uygulanir;
    # kullanici alani bosaltip gonderirse (date_from=) tarih filtresi yok.
    if "date_from" not in request.query_params:
        date_from = _period_range("week", 0)[0].astimezone(_TSI).strftime("%Y-%m-%d")

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

    def _tsi_day_start(value: str) -> datetime | None:
        # Tarih alanlari TSI gunu (tablodaki ARRIVAL da TSI gosteriliyor).
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=_TSI).astimezone(timezone.utc)
        except ValueError:
            return None

    # Sekme DISINDAKI tum filtreler: liste ve sekme sayaclari ayni kosullarla sayilir.
    conds = []
    if tf != "all":
        conds.append(_tf_filter(tf))
    if selected_markets:
        conds.append(Signal.market_type.in_(selected_markets))
    if symbol:
        conds.append(Signal.symbol.ilike(f"%{symbol}%"))
    if direction:
        conds.append(Signal.direction == direction.upper())
    if status:
        conds.append(Signal.status == status)
    if result_filter:
        conds.append(Signal.result == result_filter)
    dt_from = _tsi_day_start(date_from) if date_from else None
    if dt_from is not None:
        conds.append(arrival_date_col >= dt_from)
    dt_to = _tsi_day_start(date_to) if date_to else None
    if dt_to is not None:
        conds.append(arrival_date_col < dt_to + timedelta(days=1))

    tab_statuses = {
        "active": ["active"],
        "waiting": ["waiting_entry", "pending_cisd"],
        "closed": ["expired", "breakeven"],
    }
    query = select(Signal).where(*conds)
    if tab in tab_statuses:
        query = query.where(Signal.status.in_(tab_statuses[tab]))

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

    tab_counts = {}
    for key, statuses in tab_statuses.items():
        cnt = await db.execute(select(func.count()).where(*conds, Signal.status.in_(statuses)))
        tab_counts[key] = int(cnt.scalar() or 0)
    active_count = tab_counts["active"]
    waiting_count = tab_counts["waiting"]
    closed_count = tab_counts["closed"]
    total_count = active_count + waiting_count + closed_count

    # Sekme / sayfa linkleri filtreleri tasir (tarih bos da olsa: "tum tarihler" secimi korunur).
    qs_items: list[tuple[str, str]] = []
    if tf != "all":
        qs_items.append(("tf", tf))
    for key, value in (("symbol", symbol), ("direction", direction), ("status", status), ("result", result_filter)):
        if value:
            qs_items.append((key, value))
    if request.query_params.getlist("market_type"):
        qs_items.extend(("market_type", m) for m in selected_markets)
    qs_items.extend((("date_from", date_from or ""), ("date_to", date_to or "")))
    filter_qs = urlencode(qs_items)

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
        "tf_labels": _OPEN_TF_LABELS,
        "filter_qs": filter_qs,
        "week_start": _period_range("week", 0)[0].astimezone(_TSI).strftime("%Y-%m-%d"),
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
    tf: str = Query(default="all"),
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
    tf: str = Query(default="all"),
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
    tf: str = Query(default="all"),
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


# ──────────────────── Open Signals ────────────────────
# Uc stratejinin aktif + bekleyen sinyalleri tek panoda. Signals sayfalari
# gecmisi (TF bazli, sayfali) gosterir; burasi "su an ne acik" sorusu icin.

_OPEN_STATUSES = ("active", "waiting_entry", "pending_cisd")
_OPEN_TF_LABELS = {"4h": "4H-15M", "1d": "1D-1H", "1h": "1H-5M"}
_OPEN_LTF = {"4h": "15m", "1d": "1h", "1h": "5m"}
_OPEN_C2_HOURS = {"4h": 4, "1d": 24, "1h": 1}


def _last_ltf_price(signal) -> float | None:
    """Stratejinin LTF'indeki son fiyat (forming mum dahil). WebSocket store'undan; REST yok."""
    from app.exchange import from_display_symbol

    try:
        bingx_symbol, _ = from_display_symbol(signal.symbol)
    except Exception:
        return None
    store = getattr(market_data, "store", None)
    if store is None:
        return None
    tf = signal.timeframe or "4h"
    for ltf in dict.fromkeys((_OPEN_LTF.get(tf, "15m"), "15m", "1h", "5m")):
        try:
            df = store.get_df(bingx_symbol, ltf)
        except Exception:
            continue
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
    return None


def _fmt_age(ts) -> str:
    if ts is None:
        return "-"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    mins = max(0, int((datetime.now(timezone.utc) - ts).total_seconds() // 60))
    if mins < 60:
        return f"{mins}m"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {mins:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


async def _open_signals_context(db: AsyncSession, tab: str, tf: str) -> dict:
    tab = tab if tab in ("all", "active", "waiting") else "all"
    tf = tf if tf in ("all", "4h", "1d", "1h") else "all"
    result = await db.execute(select(Signal).where(Signal.status.in_(_OPEN_STATUSES)))
    open_signals = result.scalars().all()

    def _tf_of(s) -> str:
        return s.timeframe or "4h"

    def _tab_ok(s) -> bool:
        return tab == "all" or (s.status == "active") == (tab == "active")

    in_tf = [s for s in open_signals if tf == "all" or _tf_of(s) == tf]
    tf_counts = Counter(_tf_of(s) for s in open_signals if _tab_ok(s))
    tf_counts["all"] = sum(1 for s in open_signals if _tab_ok(s))
    counts = {
        "all": len(in_tf),
        "active": sum(1 for s in in_tf if s.status == "active"),
        "waiting_entry": sum(1 for s in in_tf if s.status == "waiting_entry"),
        "pending_cisd": sum(1 for s in in_tf if s.status == "pending_cisd"),
        "tf": tf_counts,
    }
    counts["waiting"] = counts["waiting_entry"] + counts["pending_cisd"]

    rows = []
    for s in in_tf:
        if not _tab_ok(s):
            continue
        entry = float(s.entry_price) if s.entry_price is not None else None
        sl0 = s.initial_stop_loss if s.initial_stop_loss is not None else s.stop_loss
        risk = abs(entry - float(sl0)) if entry is not None and sl0 is not None else 0.0
        price = _last_ltf_price(s)
        live_r = to_entry_r = None
        if price is not None and entry is not None and risk > 0:
            move = (price - entry) if s.direction == "LONG" else (entry - price)
            if s.status == "active":
                rest_r = move / risk
                # Kismi kar alinmissa motorla ayni agirlik: kesir x kismi R + kalan x anlik R.
                size = float(s.partial_size or 0.0)
                live_r = size * float(s.partial_rr or 0.0) + (1.0 - size) * rest_r if size > 0 else rest_r
            else:
                to_entry_r = abs(move) / risk
        banked_r = (
            round(float(s.partial_size) * float(s.partial_rr), 2)
            if s.partial_size and s.partial_rr is not None else None
        )
        # MFE: islem lehine gorulen EN IYI nokta. Live R ile ayni cetvelde olsun diye kismi kar
        # agirligi burada da uygulanir (yani "o an kasada ne olurdu"). mfe_price yalniz aktifken
        # guncelleniyor; bekleyen sinyalde anlamsiz.
        mfe_r = None
        if s.status == "active" and s.mfe_price is not None and entry is not None and risk > 0:
            mfe_move = (float(s.mfe_price) - entry) if s.direction == "LONG" else (entry - float(s.mfe_price))
            mfe_rest = mfe_move / risk
            size = float(s.partial_size or 0.0)
            mfe_r = size * float(s.partial_rr or 0.0) + (1.0 - size) * mfe_rest if size > 0 else mfe_rest
        sl_be = (
            s.status == "active" and entry is not None and s.stop_loss is not None
            and abs(float(s.stop_loss) - entry) <= abs(entry) * 1e-9
        )
        since = s.entry_filled_time if (s.status == "active" and s.entry_filled_time) else s.created_at
        # C2 durumu anlik: kayittaki c2_closed olusturma anindan kalir, aktif islemde bayatlar.
        purge = s.purge_time
        if purge is not None and purge.tzinfo is not None:
            purge = purge.astimezone(timezone.utc).replace(tzinfo=None)
        c2_close_at = purge + timedelta(hours=_OPEN_C2_HOURS.get(_tf_of(s), 4)) if purge is not None else None
        c2_open = c2_close_at is not None and c2_close_at > datetime.now(timezone.utc).replace(tzinfo=None)
        rows.append({
            "sig": s,
            "tf": _tf_of(s),
            "tf_label": _OPEN_TF_LABELS.get(_tf_of(s), _tf_of(s).upper()),
            "price": price,
            "live_r": live_r,
            "to_entry_r": to_entry_r,
            "banked_r": banked_r,
            "mfe_r": mfe_r,
            "sl_be": sl_be,
            "c2_close_at": c2_close_at,
            "c2_open": c2_open,
            "since": since,
            "age": _fmt_age(since),
        })

    # Once aktifler, sonra waiting entry, en son waiting MSS; her grupta en yeni ustte.
    rank = {"active": 0, "waiting_entry": 1, "pending_cisd": 2}
    rows.sort(key=lambda r: r["since"] or datetime.min, reverse=True)
    rows.sort(key=lambda r: rank.get(r["sig"].status, 9))

    live = [r["live_r"] for r in rows if r["sig"].status == "active" and r["live_r"] is not None]
    return {
        "tab": tab,
        "tf": tf,
        "counts": counts,
        "rows": rows,
        "open_r": round(sum(live), 2) if live else None,
        "banked_r": round(sum(r["banked_r"] for r in rows if r["banked_r"] is not None), 2),
    }


@app.get("/open-signals", response_class=HTMLResponse)
async def open_signals_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    tf: str = Query(default="all"),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    ctx = await _open_signals_context(db, tab, tf)
    return templates.TemplateResponse(request=request, name="open_signals.html", context={
        "user": user,
        "active_page": "open_signals",
        **ctx,
    })


@app.get("/open-signals/table", response_class=HTMLResponse)
async def open_signals_table(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tab: str = Query(default="all"),
    tf: str = Query(default="all"),
):
    """Sayfanin 30 sn'lik yenilemesi: yalnizca pano parcasi (HTML)."""
    user = get_current_user(request)
    if not user:
        return HTMLResponse(status_code=401, content="")
    ctx = await _open_signals_context(db, tab, tf)
    return templates.TemplateResponse(request=request, name="open_signals_table.html", context=ctx)


# ──────────────────── Setup Journal ────────────────────
# Motorun gordugu her setup (strateji+sembol+yon+C2): nereye kadar gitti, neden durdu,
# sonra fiyat ne yapti. Yazan: app/setup_journal.py.

# Ozet (ustteki tablo) asil is; satir listesi katlanmis durur ve sayfalanir. Eskiden pencere
# icindeki TUM satirlar (gunde ~400) belege yukleniyor, ozet Python'da sayiliyordu; artik ozet
# SQL'de gruplaniyor, satirlar yalniz acildiginda sayfa sayfa cekiliyor. Veri kaybi yok --
# degisen sadece gosterim.
_JOURNAL_PAGE_SIZE = 50


@app.get("/izleme", response_class=HTMLResponse)
async def watchlist_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Izleme konulari: statu + tetige kalan + sonuc. Veri kaynagi app/watchlist.py,
    detay metinleri IZLEME.md'den okunur (bkz. modul basligi)."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    # Karara baglanmis maddeler varsayilan olarak gizli (sayfa karisiyordu); ?done=1 geri getirir.
    data = await build_watchlist(db, show_done=request.query_params.get("done") == "1")
    return templates.TemplateResponse(request=request, name="watchlist.html", context={
        "user": user,
        "active_page": "watchlist",
        **data,
    })


@app.get("/setup-journal", response_class=HTMLResponse)
async def setup_journal_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    tf: str = Query(default="all"),
    stage: str = Query(default="all"),
    symbol: str = Query(default=""),
    days: int = Query(default=7),
    page: int = Query(default=1),
    rows_open: int = Query(default=0),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    from app import setup_journal as journal
    from app.models import SetupJournal

    tf = tf if tf in ("all", "4h", "1d", "1h") else "all"
    stage = stage if stage in journal.STAGE_LABELS else "all"
    days = days if days in (1, 3, 7, 30, 90) else 7
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    filters = [SetupJournal.last_seen >= since]
    if tf != "all":
        filters.append(SetupJournal.strategy == tf)
    if symbol:
        filters.append(SetupJournal.symbol.ilike(f"%{symbol}%"))

    # Ozet asama filtresinden ONCE ve SQL'de: hangi kapi kac kazanci engelledi / kac kaybi onledi.
    grouped = (await db.execute(
        select(
            SetupJournal.best_stage,
            SetupJournal.outcome,
            func.count(SetupJournal.id),
            func.sum(case((SetupJournal.outcome == "win", SetupJournal.rr), else_=0.0)),
        )
        .where(*filters)
        .group_by(SetupJournal.best_stage, SetupJournal.outcome)
    )).all()
    per_stage: dict[str, dict] = {}
    for best_stage, outcome, n, win_rr in grouped:
        acc = per_stage.setdefault(best_stage, {"oc": Counter(), "n": 0, "r_if": 0.0})
        acc["oc"][outcome or "-"] += n
        acc["n"] += n
        if outcome == "win":
            acc["r_if"] += float(win_rr or 0.0)
        elif outcome == "loss":
            acc["r_if"] -= n
    summary = [
        {"code": code, "label": label, "n": acc["n"], "oc": acc["oc"], "r_if": round(acc["r_if"], 2)}
        for code, label in journal.STAGES
        if (acc := per_stage.get(code))
    ]
    totals = {
        "setups": sum(a["n"] for a in per_stage.values()),
        "win": sum(a["oc"].get("win", 0) for a in per_stage.values()),
        "loss": sum(a["oc"].get("loss", 0) for a in per_stage.values()),
        "tp_before_entry": sum(a["oc"].get("tp_before_entry", 0) for a in per_stage.values()),
        "no_touch": sum(a["oc"].get("no_touch", 0) for a in per_stage.values()),
        "tracking": sum(a["oc"].get("pending", 0) + a["oc"].get("filled", 0) for a in per_stage.values()),
        "signal": sum(a["oc"].get("signal", 0) for a in per_stage.values()),
        "r_if": round(sum(a["r_if"] for a in per_stage.values()), 2),
    }

    # Satir listesi: katlanmis durur; yalniz acikken (veya kapi linkiyle gelindiginde) cekilir.
    if stage != "all":
        filters.append(SetupJournal.best_stage == stage)
    total = (await db.execute(select(func.count(SetupJournal.id)).where(*filters))).scalar() or 0
    total_pages = max(1, (total + _JOURNAL_PAGE_SIZE - 1) // _JOURNAL_PAGE_SIZE)
    page = min(max(1, page), total_pages)
    show_rows = bool(rows_open) or stage != "all"
    rows = []
    if show_rows:
        rows = (await db.execute(
            select(SetupJournal).where(*filters)
            .order_by(desc(SetupJournal.last_seen))
            .offset((page - 1) * _JOURNAL_PAGE_SIZE).limit(_JOURNAL_PAGE_SIZE)
        )).scalars().all()
    filter_qs = urlencode({"tf": tf, "stage": stage, "symbol": symbol, "days": days, "rows_open": 1})
    return templates.TemplateResponse(request=request, name="setup_journal.html", context={
        "user": user,
        "active_page": "setup_journal",
        "rows": rows,
        "show_rows": show_rows,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "filter_qs": filter_qs,
        "summary": summary,
        "totals": totals,
        "tf": tf,
        "stage": stage,
        "symbol": symbol,
        "days": days,
        "stages": journal.STAGES,
        "stage_labels": journal.STAGE_LABELS,
        "outcome_labels": journal.OUTCOME_LABELS,
    })


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
    # Equity ve haftalik R islemin KAPANDIGI ana gore; acilis haftasina gore gruplayinca
    # haftayi asan islem yanlis haftaya yaziliyordu.
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    closed.sort(key=lambda s: _signal_closed_at(s) or _utc(s.created_at) or _epoch)

    equity_data = []
    cumulative = 0.0
    for s in closed:
        cumulative += s.rr_value
        ca = _signal_closed_at(s) or _utc(s.created_at)
        equity_data.append({
            "date": ca.astimezone(_TSI).strftime("%Y-%m-%d") if ca else "",
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
        ca = _signal_closed_at(s) or _utc(s.created_at)
        if ca:
            week_key = ca.astimezone(_TSI).strftime("%Y-W%W")
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

LOG_REPORT_WINDOWS = [(24.0, "24 saat"), (72.0, "3 gün"), (168.0, "7 gün"), (0.0, "Tümü")]


@app.get("/logs", response_class=HTMLResponse)
async def logs_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1),
    view: str = Query(default="events"),
    hours: float = Query(default=24.0, ge=0),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    if view == "report":
        # Motor logu (dosya): yasam dongusu, cikislar, bias, WS sagligi. Kapi elemeleri /setup-journal.
        report = await asyncio.to_thread(build_report, None, hours or None)
        return templates.TemplateResponse(request=request, name="log_report.html", context={
            "user": user,
            "report": report,
            "hours": hours,
            "windows": LOG_REPORT_WINDOWS,
            "view": "report",
        })

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
        "view": "events",
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
            "bpr": e.get("bpr"),
            "model": e.get("model"),
            # "same_color": bool(e.get("same_color")),  # eski bilgi alani; UI'da gosterilmiyor
            # Setup'i tanimlayan mumlarin acilis zamani (TSI, "16.09 15:00"): purge = C2,
            # crt = C1. Radar'da "bu setup hangi muma ait" sorusu ancak boyle cevaplanıyordu.
            "purge_time": _fmt_date_tsi(e["purge_time"]) if e.get("purge_time") else None,
            "crt_bar_time": _fmt_date_tsi(e["crt_bar_time"]) if e.get("crt_bar_time") else None,
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
            + cnt.get("stale", 0) + cnt.get("same_bar_sl", 0) + cnt.get("week_gap", 0)
            + cnt.get("past_sl", 0) + cnt.get("missed_quality", 0)
        ),
        "setups": sum(cnt.get(k, 0) for k in (
            "waiting", "c2_open", "no_cisd", "low_rr", "missed", "invalidated",
            # "same_color",  # eski hard filter
            "tight_stop", "bias_mismatch", "cluster_limit", "low_quality",
            "has_open", "duplicate", "corr_open", "stale", "same_bar_sl", "week_gap",
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
            "subscription_breakdown": ws.get("subscription_breakdown"),
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


# ──────────────────── Paper Trades (deneme islem gunlugu) ────────────────────
# Elle tutulan islem kaydi: motorun karnesi Signal'da, burasi INSAN kararini olcer.
# Mantik app/paper_trades.py'de; burada yalniz route + sablon baglami.

_PAPER_VIEWS = ("overview", "trades", "calendar", "import")
_PAPER_TABS = ("all", "open", "closed")
_PAPER_MARKET_KEYS = {key for key, _ in paper.MARKET_OPTIONS}


def _paper_num_str(value) -> str:
    """Form alani icin sade sayi metni (bilimsel notasyon ve virgul olmadan)."""
    if value is None:
        return ""
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _paper_filters(request: Request) -> dict:
    q = request.query_params
    f = {key: (q.get(key) or "").strip()
         for key in ("symbol", "direction", "strategy", "source", "result",
                     "date_from", "date_to", "date_basis")}
    markets: list[str] = []
    for value in q.getlist("market_type"):
        markets.extend([v for v in value.split(",") if v])
    f["markets"] = [mk for mk in markets if mk in _PAPER_MARKET_KEYS]
    # Varsayilan giris tarihi: bu haftanin Pazartesi'si (TSI) -- Signals sayfasiyla ayni sozlesme.
    # Parametre HIC yoksa uygulanir; kullanici alani bosaltip gonderirse tarih filtresi kalkar.
    if "date_from" not in q:
        f["date_from"] = _period_range("week", 0)[0].astimezone(_TSI).strftime("%Y-%m-%d")
    return f


def _paper_qs(f: dict, view: str, tab: str, kind: str, offset: int) -> tuple[str, str]:
    """(qs, filter_qs). qs = her sey (form gonderimi / satir islemleri geri donerken);
    filter_qs = YALNIZ filtreler -- sekme ve donem linkleri view/tab/period'i kendileri yazar
    (tekrarlanan parametrede Starlette ilkini aldigi icin ikisini birlestirmek kirilgan)."""
    items: list[tuple[str, str]] = [
        (key, f[key]) for key in ("symbol", "direction", "strategy", "source", "result") if f.get(key)
    ]
    items.extend(("market_type", mk) for mk in f["markets"])
    items.extend((("date_from", f.get("date_from") or ""), ("date_to", f.get("date_to") or "")))
    if f.get("date_basis") == "closed":
        items.append(("date_basis", "closed"))
    head = [("view", view), ("tab", tab), ("period", kind), ("offset", str(offset))]
    return urlencode(head + items), urlencode(items)


def _paper_form_ctx(request: Request, trade, form_data: dict | None) -> dict:
    """Form'un dolu degerleri: POST hatasi > duzenlenen kayit > URL onyuklemesi > bos."""
    if form_data is not None:
        ctx = {key: (form_data.get(key) or "") for key in (
            "symbol", "direction", "strategy", "source", "entry_price", "stop_loss", "take_profit",
            "entered_at", "confidence", "gate", "signal_id", "chart_url", "note",
            "partial_price", "partial_fraction", "qty", "currency")}
        ctx["id"] = form_data.get("id") or (trade.id if trade else "")
        return ctx
    if trade is not None:
        return {
            "id": trade.id,
            "symbol": trade.symbol or "",
            "direction": trade.direction or "LONG",
            "strategy": trade.strategy or "other",
            "source": trade.source or "manual",
            "entry_price": _paper_num_str(trade.entry_price),
            "stop_loss": _paper_num_str(trade.stop_loss),
            "take_profit": _paper_num_str(trade.take_profit),
            "entered_at": paper.to_input_dt(trade.entered_at),
            "confidence": str(trade.confidence) if trade.confidence else "",
            "gate": trade.gate or "",
            "signal_id": str(trade.signal_id) if trade.signal_id else "",
            "chart_url": trade.chart_url or "",
            "note": trade.note or "",
            "partial_price": _paper_num_str(trade.partial_price),
            "partial_fraction": _paper_num_str(trade.partial_fraction),
            "qty": _paper_num_str(trade.qty),
            "currency": trade.currency or paper.DEFAULT_CURRENCY,
        }
    # URL onyuklemesi: Open Signals / Signals sayfasindaki "Günlüğe ekle" butonu boyle gelir.
    q = request.query_params
    return {
        "id": "",
        "symbol": q.get("symbol") or "",
        "direction": q.get("direction") or "LONG",
        "strategy": q.get("strategy") or "other",
        "source": q.get("source") or ("bot" if q.get("signal_id") else "manual"),
        "entry_price": q.get("entry") or "",
        "stop_loss": q.get("sl") or "",
        "take_profit": q.get("tp") or "",
        "entered_at": "",
        "confidence": "",
        "gate": q.get("gate") or "",
        "signal_id": q.get("signal_id") or "",
        "chart_url": "",
        "note": "",
        "partial_price": "",
        "partial_fraction": "",
        "qty": "",
        "currency": paper.DEFAULT_CURRENCY,
    }


async def _paper_context(
    request: Request, db: AsyncSession, *, errors: list[str] | None = None,
    form_data: dict | None = None, edit_id: int | None = None, full: bool = True,
) -> dict:
    """Sayfa baglami. `full=False` yalnizca tablo parcasi icin: 30 sn'de bir yenilenen
    istek panolari (donem ozeti / ben vs motor / disiplin / sinyal onerileri) hesaplamaz."""
    q = request.query_params
    view = (q.get("view") or "overview").lower()
    view = view if view in _PAPER_VIEWS else "overview"
    tab = (q.get("tab") or "all").lower()
    tab = tab if tab in _PAPER_TABS else "all"
    # Takvim her zaman aylik; donem secicisi o sekmede gizlenir.
    kind = "month" if (q.get("period") == "month" or view == "calendar") else "week"
    try:
        offset = max(-520, min(0, int(q.get("offset") or 0)))
    except ValueError:
        offset = 0
    try:
        page = max(1, int(q.get("page") or 1))
    except ValueError:
        page = 1

    f = _paper_filters(request)
    qs, filter_qs = _paper_qs(f, view, tab, kind, offset)

    now = datetime.now(timezone.utc)
    start, end, label = _period_range(kind, offset, now)
    prev_start, prev_end, _ = _period_range(kind, offset - 1, now)
    naive = lambda dt: dt.replace(tzinfo=None)  # noqa: E731  (DB'nin her yeri naive UTC)

    listing = await paper.list_page(db, f, tab, page)
    # Her sekme yalniz kendi panosunu hesaplar (tablo parcasi hicbirini).
    panels: dict = {"summary": {}, "vs": {}, "disc": {}, "suggestions": [], "calendar": None}
    if full:
        panels["suggestions"] = await paper.signal_suggestions(db)
    if full and view == "calendar":
        panels["calendar"] = paper.calendar_month(
            await paper.closed_trades(db, naive(start), naive(end)),
            start.astimezone(_TSI).date(),
            (end - timedelta(days=1)).astimezone(_TSI).date(),
            await paper.open_trades_between(db, naive(start), naive(end)))
    elif full and view == "overview":
        summary = paper.period_summary(await paper.closed_trades(db, naive(start), naive(end)))
        prev = paper.period_summary(await paper.closed_trades(db, naive(prev_start), naive(prev_end)))
        summary["delta_r"] = round(summary["total_r"] - prev["total_r"], 2) if prev["n"] else None
        all_closed = await paper.closed_trades(db)
        panels["summary"] = summary
        panels["vs"] = await paper.vs_engine(db, all_closed)
        panels["disc"] = paper.discipline(all_closed)

    # Duzenlenen / kapatilan kayit
    trade = None
    edit_raw = edit_id if edit_id is not None else q.get("edit")
    if edit_raw:
        try:
            trade = await db.get(PaperTrade, int(edit_raw))
        except (ValueError, TypeError):
            trade = None
    close_target = None
    if q.get("close"):
        try:
            target = await db.get(PaperTrade, int(q.get("close")))
        except (ValueError, TypeError):
            target = None
        if target is not None and target.status == "open":
            close_target = paper.row_view(target)
            close_target["price"] = _paper_num_str(close_target["price"])

    # "Tabloyu bu doneme getir": ozet donem seciciye, tablo kendi tarih filtresine bagli.
    table_link = "/paper-trades?" + urlencode([
        ("view", "trades"), ("tab", tab), ("period", kind), ("offset", str(offset)),
        ("date_basis", "closed"),
        ("date_from", start.astimezone(_TSI).strftime("%Y-%m-%d")),
        ("date_to", (end - timedelta(days=1)).astimezone(_TSI).strftime("%Y-%m-%d")),
    ])

    return {
        "user": get_current_user(request),
        "active_page": "paper_trades",
        "view": view,
        "errors": errors or [],
        "form": _paper_form_ctx(request, trade, form_data),
        "form_open": bool(q.get("new") or edit_raw or errors or form_data),
        "close_target": close_target,
        "symbols": paper.symbol_choices(),
        "sources": paper.SOURCES,
        "strategies": paper.STRATEGIES,
        "exit_reasons": paper.EXIT_REASONS,
        "mistake_tags": paper.MISTAKE_TAGS,
        "markets": paper.MARKET_OPTIONS,
        "period": {"kind": kind, "offset": offset, "label": label, "table_link": table_link},
        **panels,
        "listing": listing,
        "rows": listing["rows"],
        "total": listing["total"],
        "page": listing["page"],
        "total_pages": listing["total_pages"],
        "tab": tab,
        "filters": f,
        "qs": qs,
        "filter_qs": filter_qs,
        "week_start": _period_range("week", 0)[0].astimezone(_TSI).strftime("%Y-%m-%d"),
        "now_input": paper.to_input_dt(paper.now_utc()),
        "imported": q.get("imported"),
        "import_result": None,
    }


async def _paper_form_data(request: Request) -> dict:
    raw = await request.form()
    data = {key: raw.get(key) for key in raw.keys()}
    data["mistakes"] = raw.getlist("mistakes")
    return data


def _paper_redirect(data: dict) -> RedirectResponse:
    qs = (data.get("qs") or "").strip()
    return RedirectResponse(url=f"/paper-trades?{qs}" if qs else "/paper-trades", status_code=303)


@app.get("/paper-trades", response_class=HTMLResponse)
async def paper_trades_page(request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    ctx = await _paper_context(request, db)
    return templates.TemplateResponse(request=request, name="paper_trades.html", context=ctx)


@app.get("/paper-trades/table", response_class=HTMLResponse)
async def paper_trades_table(request: Request, db: AsyncSession = Depends(get_db)):
    """Tablonun 30 sn'lik yenilemesi (acik islemlerde canli fiyat + canli R)."""
    if not get_current_user(request):
        return HTMLResponse(status_code=401, content="")
    ctx = await _paper_context(request, db, full=False)
    return templates.TemplateResponse(request=request, name="paper_trades_table.html", context=ctx)


@app.post("/paper-trades/new")
async def paper_trade_new(request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    data = await _paper_form_data(request)
    trade = PaperTrade()
    errors = paper.apply_plan(trade, data)
    # Hizli kayit: cikis da girildiyse islem dogrudan kapali dogar (gecmis islemleri toplu girmek icin).
    if not errors and (data.get("exit_price") or "").strip():
        errors = paper.apply_close(trade, data)
    if errors:
        ctx = await _paper_context(request, db, errors=errors, form_data=data)
        return templates.TemplateResponse(request=request, name="paper_trades.html", context=ctx)
    db.add(trade)
    await db.commit()
    return _paper_redirect(data)


@app.post("/paper-trades/{trade_id}/edit")
async def paper_trade_edit(trade_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    trade = await db.get(PaperTrade, trade_id)
    if trade is None:
        return RedirectResponse(url="/paper-trades", status_code=303)
    data = await _paper_form_data(request)
    errors = paper.apply_plan(trade, data) + paper.apply_partial(trade, data)
    if errors:
        data["id"] = trade_id
        ctx = await _paper_context(request, db, errors=errors, form_data=data, edit_id=trade_id)
        return templates.TemplateResponse(request=request, name="paper_trades.html", context=ctx)
    # Seviyeler degistiyse kapanmis islemin R'si de yeniden hesaplanir (elle girilmez).
    if trade.status == "closed" and trade.exit_price is not None:
        trade.rr_value = round(paper.rr_at(trade, trade.exit_price) or 0.0, 4)
        trade.result = paper.result_of(trade.rr_value)
    await db.commit()
    return _paper_redirect(data)


@app.post("/paper-trades/{trade_id}/close")
async def paper_trade_close(trade_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    trade = await db.get(PaperTrade, trade_id)
    if trade is None:
        return RedirectResponse(url="/paper-trades", status_code=303)
    data = await _paper_form_data(request)
    errors = paper.apply_close(trade, data)
    if errors:
        ctx = await _paper_context(request, db, errors=errors)
        return templates.TemplateResponse(request=request, name="paper_trades.html", context=ctx)
    await db.commit()
    return _paper_redirect(data)


@app.post("/paper-trades/{trade_id}/reopen")
async def paper_trade_reopen(trade_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    trade = await db.get(PaperTrade, trade_id)
    if trade is not None:
        paper.reopen(trade)
        await db.commit()
    return _paper_redirect(await _paper_form_data(request))


@app.post("/paper-trades/{trade_id}/delete")
async def paper_trade_delete(trade_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    trade = await db.get(PaperTrade, trade_id)
    if trade is not None:
        await db.delete(trade)
        await db.commit()
    return _paper_redirect(await _paper_form_data(request))


# ──────────────────── TradingView içe aktarma ────────────────────
# Iki CSV (islem gecmisi + emirler) okunur, birlestirilir, ONIZLEME gosterilir; yazma ayri
# bir onaydan gecer. Dogrudan yazmiyoruz cunku: SL emir dosyasi yoksa eksik kalir (elle
# girilir), sembol eslesmeyebilir ve ayni dosya iki kez yuklenebilir.


def _decode_upload(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _import_payload(rows: list[dict]) -> str:
    """Onizleme satirlarini gizli alanda tasiyacak JSON (datetime -> ISO)."""
    def _clean(row: dict) -> dict:
        out = {}
        for key, value in row.items():
            out[key] = value.isoformat() if isinstance(value, datetime) else value
        return out
    return json.dumps([_clean(r) for r in rows], ensure_ascii=False)


def _payload_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


@app.post("/paper-trades/import/preview", response_class=HTMLResponse)
async def paper_import_preview(
    request: Request,
    db: AsyncSession = Depends(get_db),
    trades_file: UploadFile = File(...),
    orders_file: UploadFile | None = File(None),
    link_signals: str = Form(default=""),
):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)

    errors: list[str] = []
    rows: list[dict] = []
    warnings: list[str] = []
    orders_n = 0
    try:
        trades, warnings = tv.parse_trades(_decode_upload(await trades_file.read()))
        orders = []
        if orders_file is not None and orders_file.filename:
            orders = tv.parse_orders(_decode_upload(await orders_file.read()))
            orders_n = len(orders)
        rows = tv.merge(trades, orders)
    except ValueError as exc:
        errors.append(str(exc))
    except Exception as exc:                                  # noqa: BLE001
        log.exception("paper import: dosya okunamadi")
        errors.append(f"Dosya okunamadı: {exc}")

    if rows:
        # Mukerrer koruma: ayni emir no daha once aktarildiysa satir kilitli gelir.
        seen = set((await db.execute(
            select(PaperTrade.ext_id).where(PaperTrade.ext_id.in_([r["ext_id"] for r in rows]))
        )).scalars().all())
        for row in rows:
            row["exists"] = row["ext_id"] in seen
            row["sl_side_ok"] = tv.sl_side_ok(row)
            row["signal_id"] = None
            if link_signals and not row["exists"]:
                row["signal_id"] = await paper.match_signal(
                    db, row["symbol"], row["direction"], row["entered_at"])
        if not orders_n:
            warnings.append("Emir dosyası verilmedi — SL/TP gelmedi, R hesaplanamaz. "
                            "Aşağıdaki SL alanlarını elle doldurabilirsin.")

    ctx = await _paper_context(request, db)
    ctx.update({
        "view": "import",
        "import_result": {
            "rows": rows,
            "warnings": warnings,
            "orders_n": orders_n,
            "new_n": sum(1 for r in rows if not r["exists"]),
            "exists_n": sum(1 for r in rows if r["exists"]),
            "no_sl_n": sum(1 for r in rows if not r["exists"] and r["sl"] is None),
            "linked_n": sum(1 for r in rows if r.get("signal_id")),
            "payload": _import_payload(rows),
            "link_signals": bool(link_signals),
        },
        "errors": errors,
    })
    return templates.TemplateResponse(request=request, name="paper_trades.html", context=ctx)


@app.post("/paper-trades/import/commit")
async def paper_import_commit(request: Request, db: AsyncSession = Depends(get_db)):
    if not get_current_user(request):
        return RedirectResponse(url="/login", status_code=303)
    data = await _paper_form_data(request)
    try:
        rows = json.loads(data.get("payload") or "[]")
    except json.JSONDecodeError:
        rows = []
    picked = {v for v in (await request.form()).getlist("pick") if v}
    source = (data.get("source") or "manual").lower()
    source = source if source in paper.SOURCE_LABELS else "manual"

    seen = set((await db.execute(
        select(PaperTrade.ext_id).where(PaperTrade.ext_id.in_([r.get("ext_id") for r in rows]))
    )).scalars().all()) if rows else set()

    created = 0
    for row in rows:
        ext_id = row.get("ext_id")
        if ext_id in seen or (picked and ext_id not in picked):
            continue
        sl = paper.parse_num(data.get(f"sl_{ext_id}")) or row.get("sl")
        entry = row.get("entry")
        trade = PaperTrade(
            symbol=row.get("symbol"), market_type=row.get("market_type"),
            direction=row.get("direction"), strategy="other",
            source="bot" if row.get("signal_id") else source,
            signal_id=row.get("signal_id"),
            entry_price=entry, stop_loss=sl, take_profit=row.get("tp"),
            planned_rr=paper.calc_planned_rr(entry, sl, row.get("tp")),
            entered_at=_payload_dt(row.get("entered_at")) or paper.now_utc(),
            partial_price=row.get("partial_price"), partial_fraction=row.get("partial_fraction"),
            exit_price=row.get("exit"), closed_at=_payload_dt(row.get("closed_at")),
            exit_reason="manual" if row.get("exit") is not None else None,
            status=row.get("status") or "open",
            currency=row.get("currency") or paper.DEFAULT_CURRENCY,
            qty=row.get("qty"), notional=row.get("notional"), fees=row.get("fees"),
            pnl_amount=row.get("pnl"), return_pct=row.get("return_pct"),
            leverage=row.get("leverage"), margin=row.get("margin"),
            ext_source="tv", ext_id=ext_id,
            note=f"TradingView içe aktarma · trade #{row.get('trade_no')}",
        )
        if trade.status == "closed":
            rr = paper.rr_at(trade, trade.exit_price)
            trade.rr_value = round(rr, 4) if rr is not None else None
            trade.result = paper.result_of(trade.rr_value, paper.pnl_of(trade))
            if trade.entered_at and trade.closed_at:
                trade.duration_hours = round(
                    max(0.0, (trade.closed_at - trade.entered_at).total_seconds() / 3600.0), 2)
        db.add(trade)
        created += 1
    if created:
        await db.commit()
    return RedirectResponse(
        url=f"/paper-trades?view=trades&tab=all&date_from=&imported={created}", status_code=303)
