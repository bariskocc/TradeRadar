"""Deneme (paper) islem gunlugu -- ELLE tutulan islem kaydinin mantigi.

Motorun karnesi `Signal` tablosunda; burasi INSAN kararini olcer (bkz. models.PaperTrade).
Sayfa: /paper-trades. Bu modul saf mantik + sorgu; route'lar app/main.py'de.

Cetvel motorla AYNI tutulur, yoksa "ben vs motor" kiyasi anlamsiz olur:
  - R'nin paydasi daima ilk SL (`stop_loss`), BE'ye cekilen stop paydayi degistirmez
  - kismi cikis varsa R agirlikli: `f x kismi R + (1 - f) x kalan R`
  - `result` R'nin isaretinden yazilir (win / loss / breakeven)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.exchange import from_display_symbol, get_all_symbols_flat
from app.market_data import market_data
from app.models import PaperTrade, Signal

log = logging.getLogger(__name__)

TSI = timezone(timedelta(hours=3))

# Karar kaynagi: islemi neden aldim?
SOURCES = (
    ("bot", "Bot sinyali"),
    ("gated", "Elenen setup"),
    ("manual", "Kendi setup'ım"),
)
SOURCE_LABELS = dict(SOURCES)

STRATEGIES = (("4h", "4H-15M"), ("1d", "1D-1H"), ("1h", "1H-5M"), ("other", "Diğer"))
STRATEGY_LABELS = dict(STRATEGIES)

EXIT_REASONS = (
    ("tp", "TP"), ("sl", "SL"), ("be", "BE"),
    ("trail", "Trail"), ("manual", "Elle kapattım"), ("week_close", "Hafta kapanışı"),
)
EXIT_LABELS = dict(EXIT_REASONS)

RESULT_LABELS = {"win": "Win", "loss": "Loss", "breakeven": "Breakeven"}

# Kapanista sorulan hata etiketleri. Panoda her etiketin KAC R'ye mal oldugu gosterilir --
# "hangi hatayi yapiyorum" degil, "hangi hata pahali" sorusu ise yarayan.
MISTAKE_TAGS = (
    ("early_entry", "Erken giriş"),
    ("late_entry", "Geç giriş"),
    ("chased", "Fiyatı kovaladım"),
    ("moved_sl", "SL'yi ittim"),
    ("early_exit", "Erken kapattım"),
    ("late_exit", "Geç kapattım"),
    ("no_setup", "Plan dışı setup"),
    ("oversize", "Fazla risk"),
    ("revenge", "İntikam işlemi"),
)
MISTAKE_LABELS = dict(MISTAKE_TAGS)

# Takvim: hafta Pazartesi baslar (donem ozetiyle ayni kural).
WEEKDAY_LABELS = ("Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz")

MARKET_OPTIONS = (
    ("crypto", "Crypto"), ("fx", "FX"), ("index", "Index"),
    ("metal", "Metal"), ("oil", "Oil"), ("other", "Diğer"),
)

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "TRY": "₺", "GBP": "£"}
DEFAULT_CURRENCY = "USD"

_R_EPS = 0.005          # bu bandin icinde kalan R breakeven sayilir (motorla ayni)
_MONEY_EPS = 0.005      # kurus altinda kalan K/Z breakeven
LOW_SAMPLE = 10         # bu sayinin altinda oranlar "az ornek"
PER_PAGE = 25
_LTF = {"4h": "15m", "1d": "1h", "1h": "5m"}

_KNOWN_SYMBOLS = tuple(sorted(get_all_symbols_flat()))
_KNOWN_SET = set(_KNOWN_SYMBOLS)


# ──────────────────── Sembol / sayi / zaman ────────────────────

def symbol_choices() -> tuple[str, ...]:
    """Form'daki datalist icin bilinen gosterim sembolleri."""
    return _KNOWN_SYMBOLS


def normalize_symbol(raw: str | None) -> str:
    """Kullanicinin yazdigini bilinen gosterim adina cevir ("FET" -> "FETUSDT.P").

    Eslesmezse yazilan hali korunur: botun izlemedigi bir sembolde de islem
    kaydedilebilmeli (canli fiyat olmaz, geri kalan her sey calisir).
    """
    s = (raw or "").strip().upper()
    if not s:
        return ""
    for cand in (s, f"{s}USDT.P", f"{s}.P", f"{s}USDT"):
        if cand in _KNOWN_SET:
            return cand
    return s


def market_for(symbol: str) -> str:
    if symbol not in _KNOWN_SET:
        return "other"
    try:
        return from_display_symbol(symbol)[1]
    except Exception:
        return "other"


def parse_num(raw) -> float | None:
    """Ondalikta hem nokta hem virgul kabul et (UI fiyatlari virgullu gosteriyor)."""
    if raw is None:
        return None
    s = str(raw).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")    # 1.694,50
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_dt(raw) -> datetime | None:
    """Form'daki datetime-local (TSI) -> naive UTC. DB'nin her yeri naive UTC."""
    if not raw:
        return None
    s = str(raw).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return dt.replace(tzinfo=TSI).astimezone(timezone.utc).replace(tzinfo=None)
    return None


def to_input_dt(dt: datetime | None) -> str:
    """naive UTC -> datetime-local degeri (TSI)."""
    if dt is None:
        return ""
    return dt.replace(tzinfo=timezone.utc).astimezone(TSI).strftime("%Y-%m-%dT%H:%M")


def fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.replace(tzinfo=timezone.utc).astimezone(TSI).strftime("%d.%m %H:%M")


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ──────────────────── R hesabi ────────────────────

def _move(direction: str, entry: float, price: float) -> float:
    return (price - entry) if (direction or "LONG") == "LONG" else (entry - price)


def risk_of(t) -> float | None:
    if t.entry_price is None or t.stop_loss is None:
        return None
    risk = abs(float(t.entry_price) - float(t.stop_loss))
    return risk if risk > 0 else None


def calc_planned_rr(entry, sl, tp) -> float | None:
    if entry is None or sl is None or tp is None:
        return None
    risk = abs(float(entry) - float(sl))
    if risk <= 0:
        return None
    return round(abs(float(tp) - float(entry)) / risk, 2)


def rr_at(t, price: float | None) -> float | None:
    """Fiyat `price` iken islemin R'si. Kismi cikis varsa AGIRLIKLI (motorla ayni)."""
    risk = risk_of(t)
    if risk is None or price is None:
        return None
    rest = _move(t.direction, float(t.entry_price), float(price)) / risk
    frac = float(t.partial_fraction or 0.0)
    if frac > 0 and t.partial_price is not None:
        part = _move(t.direction, float(t.entry_price), float(t.partial_price)) / risk
        return frac * part + (1.0 - frac) * rest
    return rest


def computed_pnl(t, price: float | None) -> float | None:
    """Miktardan net K/Z: `qty x fiyat farki` (kismi cikis dahil) - komisyon.

    Yalnizca `qty` girilmisse hesaplanir. Ice aktarilan islemlerde TradingView'in kendi
    rakami (`pnl_amount`) yazildigi icin bu hesap devreye girmez -- borsanin rakami esastir.
    """
    qty = float(t.qty or 0.0)
    if qty <= 0 or price is None or t.entry_price is None:
        return None
    entry = float(t.entry_price)
    frac = float(t.partial_fraction or 0.0)
    if frac > 0 and t.partial_price is not None:
        gross = (qty * frac * _move(t.direction, entry, float(t.partial_price))
                 + qty * (1.0 - frac) * _move(t.direction, entry, float(price)))
    else:
        gross = qty * _move(t.direction, entry, float(price))
    return round(gross - float(t.fees or 0.0), 2)


def pnl_of(t) -> float | None:
    """Kapanmis islemin net K/Z'si: kayitli tutar varsa o, yoksa miktardan hesap."""
    if t.pnl_amount is not None:
        return float(t.pnl_amount)
    return computed_pnl(t, t.exit_price)


def result_of(rr: float | None, pnl: float | None = None) -> str | None:
    """Sonuc R'nin isaretinden; R yoksa (SL bilinmiyor) paranin isaretinden."""
    if rr is not None:
        if rr > _R_EPS:
            return "win"
        if rr < -_R_EPS:
            return "loss"
        return "breakeven"
    if pnl is not None:
        if pnl > _MONEY_EPS:
            return "win"
        if pnl < -_MONEY_EPS:
            return "loss"
        return "breakeven"
    return None


def fmt_money(value, currency: str | None = None) -> str:
    """-615.93 USD -> "-$615.93". Sablonlarda global olarak kullanilir."""
    if value is None:
        return "—"
    sym = CURRENCY_SYMBOLS.get((currency or DEFAULT_CURRENCY).upper())
    sign = "-" if value < 0 else ""
    body = f"{abs(float(value)):,.2f}"
    return f"{sign}{sym}{body}" if sym else f"{sign}{body} {(currency or '').upper()}".strip()


def last_price(symbol: str, strategy: str | None) -> float | None:
    """WebSocket store'undan son fiyat (REST yok). Bilinmeyen sembolde None."""
    if symbol not in _KNOWN_SET:
        return None
    try:
        bingx_symbol, _ = from_display_symbol(symbol)
    except Exception:
        return None
    store = getattr(market_data, "store", None)
    if store is None:
        return None
    for ltf in dict.fromkeys((_LTF.get(strategy or "4h", "15m"), "15m", "1h", "5m")):
        try:
            df = store.get_df(bingx_symbol, ltf)
        except Exception:
            continue
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
    return None


# ──────────────────── Kayit olusturma / guncelleme ────────────────────

def _mistakes_from(values) -> str | None:
    keys = [v for v in (values or []) if v in MISTAKE_LABELS]
    return ",".join(keys) if keys else None


def mistakes_of(t) -> list[str]:
    return [k for k in (t.mistakes or "").split(",") if k in MISTAKE_LABELS]


def apply_plan(t: PaperTrade, form: dict) -> list[str]:
    """Plan alanlarini forma gore yaz; donen liste hatalardir (bos = gecerli)."""
    errors: list[str] = []
    t.symbol = normalize_symbol(form.get("symbol"))
    if not t.symbol:
        errors.append("Sembol boş olamaz.")
    t.market_type = market_for(t.symbol)
    t.direction = "SHORT" if (form.get("direction") or "").upper() == "SHORT" else "LONG"
    strategy = (form.get("strategy") or "other").lower()
    t.strategy = strategy if strategy in STRATEGY_LABELS else "other"
    source = (form.get("source") or "manual").lower()
    t.source = source if source in SOURCE_LABELS else "manual"

    entry = parse_num(form.get("entry_price"))
    sl = parse_num(form.get("stop_loss"))
    tp = parse_num(form.get("take_profit"))
    if entry is None:
        errors.append("Giriş fiyatı gerekli.")
    # SL zorunlu DEGIL: TradingView'den gelen ya da stop emirsiz acilmis islemde bos kalabilir.
    # O kayitta R hesaplanmaz (sonuc paradan yazilir), karne R tarafinda onu saymaz.
    if entry is not None and sl is not None:
        if entry == sl:
            errors.append("SL giriş fiyatına eşit olamaz.")
        elif t.direction == "LONG" and sl > entry:
            errors.append("LONG işlemde SL girişin altında olmalı.")
        elif t.direction == "SHORT" and sl < entry:
            errors.append("SHORT işlemde SL girişin üstünde olmalı.")
    if entry is not None and tp is not None:
        if t.direction == "LONG" and tp < entry:
            errors.append("LONG işlemde TP girişin üstünde olmalı.")
        elif t.direction == "SHORT" and tp > entry:
            errors.append("SHORT işlemde TP girişin altında olmalı.")
    t.entry_price, t.stop_loss, t.take_profit = entry, sl, tp
    t.planned_rr = calc_planned_rr(entry, sl, tp)
    t.entered_at = parse_dt(form.get("entered_at")) or t.entered_at or now_utc()

    conf = parse_num(form.get("confidence"))
    t.confidence = int(conf) if conf is not None and 1 <= conf <= 5 else None
    t.signal_id = int(parse_num(form.get("signal_id")) or 0) or None
    t.journal_id = int(parse_num(form.get("journal_id")) or 0) or None
    t.gate = (form.get("gate") or "").strip() or None
    t.chart_url = (form.get("chart_url") or "").strip() or None
    # Para: miktar girilirse K/Z fiyat farkindan hesaplanabilir; kur kaydin kendi alaninda durur.
    t.qty = parse_num(form.get("qty"))
    t.currency = ((form.get("currency") or t.currency or DEFAULT_CURRENCY).strip().upper() or
                  DEFAULT_CURRENCY)
    if t.qty and entry is not None:
        t.notional = round(t.qty * entry, 2)
    note = (form.get("note") or "").strip()
    t.note = note or None
    t.updated_at = now_utc()
    return errors


def apply_partial(t: PaperTrade, form: dict) -> list[str]:
    """Kismi cikis (opsiyonel). Fiyat girilmisse kesir de gerekir."""
    price = parse_num(form.get("partial_price"))
    frac = parse_num(form.get("partial_fraction"))
    if price is None:
        t.partial_price = t.partial_fraction = t.partial_at = None
        return []
    if frac is None:
        frac = 0.5
    if frac > 1:                    # "50" yazilmis olabilir
        frac = frac / 100.0
    if not 0 < frac < 1:
        return ["Kısmi çıkış oranı 0 ile 1 arasında olmalı (ör. 0,5)."]
    t.partial_price, t.partial_fraction = price, frac
    t.partial_at = parse_dt(form.get("partial_at")) or t.partial_at
    return []


def apply_close(t: PaperTrade, form: dict) -> list[str]:
    """Kapanis + ogrenme alanlari. R ve result fiyatlardan hesaplanir, elle girilmez."""
    errors = apply_partial(t, form)
    exit_price = parse_num(form.get("exit_price"))
    if exit_price is None:
        errors.append("Çıkış fiyatı gerekli.")
    if errors:
        return errors
    reason = (form.get("exit_reason") or "manual").lower()
    closed_at = parse_dt(form.get("closed_at")) or now_utc()
    t.exit_price = exit_price
    t.exit_reason = reason if reason in EXIT_LABELS else "manual"
    t.closed_at = closed_at
    rr = rr_at(t, exit_price)
    t.rr_value = round(rr, 4) if rr is not None else None
    fees = parse_num(form.get("fees"))
    if fees is not None or "fees" in form:
        t.fees = fees
    explicit = parse_num(form.get("pnl_amount"))
    t.pnl_amount = explicit if explicit is not None else computed_pnl(t, exit_price)
    if t.pnl_amount is not None and t.notional:
        t.return_pct = round(t.pnl_amount / float(t.notional) * 100, 2)
    t.result = result_of(t.rr_value, t.pnl_amount)
    if t.entered_at:
        t.duration_hours = round(max(0.0, (closed_at - t.entered_at).total_seconds() / 3600.0), 2)
    t.status = "closed"
    fp = (form.get("followed_plan") or "").lower()
    t.followed_plan = True if fp == "yes" else (False if fp == "no" else None)
    t.mistakes = _mistakes_from(form.get("mistakes"))
    note = (form.get("note") or "").strip()
    if note:
        t.note = note
    t.updated_at = now_utc()
    return []


def reopen(t: PaperTrade) -> None:
    """Kapanisi geri al (yanlis kapatildiysa)."""
    t.exit_price = t.exit_reason = t.closed_at = None
    t.rr_value = t.result = t.duration_hours = None
    t.pnl_amount = t.return_pct = None
    t.status = "open"
    t.updated_at = now_utc()


# ──────────────────── Gorunum ────────────────────

def ui_symbol(symbol: str | None, market_type: str | None = None) -> str:
    """Kripto perp adini kisalt (FETUSDT.P -> FET) -- main._fmt_ui_symbol ile ayni gorunum."""
    if not symbol:
        return "-"
    sym = str(symbol)
    if sym.endswith("USDT.P") and (market_type or "crypto") == "crypto":
        return sym[:-6]
    return sym


def row_view(t: PaperTrade) -> dict:
    price = last_price(t.symbol, t.strategy) if t.status == "open" else None
    live_r = rr_at(t, price) if price is not None else None
    banked = None
    if t.partial_fraction and t.partial_price is not None:
        risk = risk_of(t)
        if risk:
            moved = _move(t.direction, float(t.entry_price), float(t.partial_price)) / risk
            banked = round(float(t.partial_fraction) * moved, 2)
    pnl = pnl_of(t) if t.status == "closed" else None
    return {
        "t": t,
        "symbol": ui_symbol(t.symbol, t.market_type),
        "pnl": pnl,
        "live_pnl": computed_pnl(t, price) if t.status == "open" else None,
        "currency": t.currency or DEFAULT_CURRENCY,
        "strategy_label": STRATEGY_LABELS.get(t.strategy or "other", "Diğer"),
        "source_label": SOURCE_LABELS.get(t.source, t.source),
        "exit_label": EXIT_LABELS.get(t.exit_reason or "", t.exit_reason or ""),
        "result_label": RESULT_LABELS.get(t.result or "", ""),
        "price": price,
        "live_r": round(live_r, 2) if live_r is not None else None,
        "rr": round(float(t.rr_value), 2) if t.rr_value is not None else None,
        "banked_r": banked,
        "mistakes": [MISTAKE_LABELS[k] for k in mistakes_of(t)],
        "entered_label": fmt_dt(t.entered_at),
        "closed_label": fmt_dt(t.closed_at) if t.closed_at else "-",
        "unknown_symbol": t.symbol not in _KNOWN_SET,
    }


# ──────────────────── Sorgular ────────────────────

def _tsi_day(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=TSI).astimezone(
            timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


def build_conditions(f: dict) -> list:
    """Sekme DISINDAKI filtreler; sekme sayaclari da ayni kosullarla sayilir."""
    conds = []
    if f.get("symbol"):
        conds.append(PaperTrade.symbol.ilike(f"%{f['symbol']}%"))
    if f.get("direction"):
        conds.append(PaperTrade.direction == f["direction"].upper())
    if f.get("strategy"):
        conds.append(PaperTrade.strategy == f["strategy"])
    if f.get("markets"):
        conds.append(PaperTrade.market_type.in_(f["markets"]))
    if f.get("source"):
        conds.append(PaperTrade.source == f["source"])
    if f.get("result"):
        conds.append(PaperTrade.result == f["result"])
    # Tarih hangi alana uygulanacak: giris (varsayilan) ya da kapanis. Takvimden bir gune
    # tiklandiginda "kapanis" kullanilir -- takvim de donem ozeti de islemi KAPANDIGI gune sayar.
    date_col = PaperTrade.closed_at if f.get("date_basis") == "closed" else PaperTrade.entered_at
    if f.get("date_from"):
        dt = _tsi_day(f["date_from"])
        if dt is not None:
            conds.append(date_col >= dt)
    if f.get("date_to"):
        dt = _tsi_day(f["date_to"])
        if dt is not None:
            conds.append(date_col < dt + timedelta(days=1))
    return conds


_TAB_STATUS = {"open": ["open"], "closed": ["closed"]}


async def list_page(db: AsyncSession, f: dict, tab: str, page: int) -> dict:
    conds = build_conditions(f)
    query = select(PaperTrade).where(*conds)
    if tab in _TAB_STATUS:
        query = query.where(PaperTrade.status.in_(_TAB_STATUS[tab]))
    total = int((await db.execute(
        select(func.count()).select_from(query.subquery()))).scalar() or 0)
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = max(1, min(page, total_pages))
    # Acik islemler her zaman ustte; sonra en yeni giris.
    rows = (await db.execute(
        query.order_by(
            (PaperTrade.status == "closed").asc(),
            PaperTrade.entered_at.desc(),
            PaperTrade.id.desc(),
        ).offset((page - 1) * PER_PAGE).limit(PER_PAGE)
    )).scalars().all()
    counts = {}
    for key, statuses in _TAB_STATUS.items():
        counts[key] = int((await db.execute(
            select(func.count()).select_from(PaperTrade)
            .where(*conds, PaperTrade.status.in_(statuses)))).scalar() or 0)
    counts["all"] = counts["open"] + counts["closed"]
    return {
        "rows": [row_view(t) for t in rows],
        "total": total, "page": page, "total_pages": total_pages, "counts": counts,
    }


def _group(rows, pred) -> dict:
    """Bir kumenin karnesi. R'si olmayan kayit (SL bilinmiyor) R toplamina GIRMEZ -- 0R
    saymak karneyi sessizce sulandirirdi; para tarafinda ise sayilir."""
    picked = [t for t in rows if pred(t)]
    rs = [float(t.rr_value) for t in picked if t.rr_value is not None]
    money = [p for p in (pnl_of(t) for t in picked) if p is not None]
    wins = sum(1 for t in picked if t.result == "win")
    n = len(picked)
    return {
        "n": n,
        "r": round(sum(rs), 2),
        "r_n": len(rs),
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "pnl": round(sum(money), 2) if money else None,
        "win_rate": round(wins / n * 100) if n else None,
        "low": n < LOW_SAMPLE,
    }


async def closed_trades(db: AsyncSession, start=None, end=None) -> list[PaperTrade]:
    q = select(PaperTrade).where(PaperTrade.status == "closed", PaperTrade.rr_value.is_not(None))
    if start is not None:
        q = q.where(PaperTrade.closed_at >= start)
    if end is not None:
        q = q.where(PaperTrade.closed_at < end)
    return list((await db.execute(q)).scalars().all())


def period_summary(rows) -> dict:
    """Donemde KAPANAN islemlerin ozeti -- Dashboard'daki blogun ayni sozlesmesi."""
    n = len(rows)
    rs = [float(t.rr_value) for t in rows if t.rr_value is not None]
    money = [p for p in (pnl_of(t) for t in rows) if p is not None]
    wins = sum(1 for t in rows if t.result == "win")
    losses = sum(1 for t in rows if t.result == "loss")
    total = round(sum(rs), 2)
    total_pnl = round(sum(money), 2) if money else None
    # En iyi / en kotu: donemde para varsa parayla, yoksa R ile siralanir.
    rank = ((lambda t: pnl_of(t) or 0.0) if money else (lambda t: float(t.rr_value or 0.0)))

    def _pick(t):
        return {
            "symbol": ui_symbol(t.symbol, t.market_type),
            "tf_label": STRATEGY_LABELS.get(t.strategy or "other", "Diğer"),
            "r": round(float(t.rr_value), 2) if t.rr_value is not None else None,
            "pnl": pnl_of(t),
        }

    return {
        "n": n,
        "total_r": total,
        "r_n": len(rs),
        "total_pnl": total_pnl,
        "currency": next((t.currency for t in rows if t.currency), DEFAULT_CURRENCY),
        "wins": wins,
        "losses": losses,
        "be": n - wins - losses,
        "win_rate": round(wins / n * 100) if n else None,
        "avg_r": round(total / n, 2) if n else None,
        "best": _pick(max(rows, key=rank)) if n else None,
        "worst": _pick(min(rows, key=rank)) if n > 1 else None,
        "by_source": [
            {"label": label, **_group(rows, lambda t, k=key: t.source == k)}
            for key, label in SOURCES
        ],
        "by_tf": [
            {"label": label, **_group(rows, lambda t, k=key: (t.strategy or "other") == k)}
            for key, label in STRATEGIES
        ],
        "low_sample": n < LOW_SAMPLE,
    }



def calendar_month(rows, first_day: date, last_day: date, open_rows=()) -> dict:
    """Ay takvimi: gun hucreleri + hafta toplamlari + ay ozeti.

    K/Z gunu = islemin **KAPANDIGI** gun, TSI takvimine gore (donem ozetiyle ayni kural:
    sonuc kapanista kesinlesir). Acik islemler tutara girmez ama **girildikleri gunde**
    sayilarak gosterilir (`open_n`) -- yoksa "islem ekledim, takvim bos" sasirtmasi oluyor.

    Renk yogunlugu ayin en buyuk |degeri|ne gore olceklenir; Tailwind dinamik sinif
    uretemedigi icin hucrede inline alpha.
    """
    by_day: dict[date, dict] = {}
    for t in rows:
        if t.closed_at is None:
            continue
        day = t.closed_at.replace(tzinfo=timezone.utc).astimezone(TSI).date()
        cell = by_day.setdefault(day, {"r": 0.0, "n": 0, "wins": 0, "losses": 0,
                                       "pnl": None, "cur": t.currency or DEFAULT_CURRENCY})
        if t.rr_value is not None:
            cell["r"] += float(t.rr_value)
        money = pnl_of(t)
        if money is not None:
            cell["pnl"] = round((cell["pnl"] or 0.0) + money, 2)
        cell["n"] += 1
        if t.result == "win":
            cell["wins"] += 1
        elif t.result == "loss":
            cell["losses"] += 1
    for cell in by_day.values():
        cell["r"] = round(cell["r"], 2)
    # Renk ve siralama olcutu: para varsa para, yoksa R (ikisi de ayni isarete sahip olmak
    # zorunda degil -- komisyon kucuk bir kazanci eksiye cevirebilir).
    has_money = any(c["pnl"] is not None for c in by_day.values())
    _val = (lambda c: c["pnl"] if c["pnl"] is not None else 0.0) if has_money else (lambda c: c["r"])
    max_abs = max((abs(_val(c)) for c in by_day.values()), default=0.0)

    # Acik islemler: tutar yok, yalnizca "o gun su kadar islem girdim" isareti.
    open_by_day: dict[date, int] = {}
    for t in open_rows or ():
        if t.entered_at is None:
            continue
        day = t.entered_at.replace(tzinfo=timezone.utc).astimezone(TSI).date()
        open_by_day[day] = open_by_day.get(day, 0) + 1

    weeks: list[dict] = []
    day = first_day - timedelta(days=first_day.weekday())        # ilk Pazartesi
    grid_end = last_day + timedelta(days=6 - last_day.weekday())  # son Pazar
    while day <= grid_end:
        cells = []
        for _ in range(7):
            data = by_day.get(day)
            r = data["r"] if data else None
            value = _val(data) if data else None
            tone = "flat"
            if value is not None and abs(value) > (_MONEY_EPS if has_money else _R_EPS):
                tone = "win" if value > 0 else "loss"
            cells.append({
                "day": day.day,
                "iso": day.isoformat(),
                "in_month": first_day <= day <= last_day,
                "n": data["n"] if data else 0,
                "r": r,
                "pnl": data["pnl"] if data else None,
                "currency": data["cur"] if data else DEFAULT_CURRENCY,
                "wins": data["wins"] if data else 0,
                "losses": data["losses"] if data else 0,
                "open_n": open_by_day.get(day, 0),
                "tone": tone,
                # 0.10-0.45 bandi: en kucuk gun bile secilebilsin, en buyugu ekrani yakmasin.
                "alpha": round(0.10 + 0.35 * (abs(value) / max_abs), 2) if value is not None and max_abs else 0.0,
            })
            day += timedelta(days=1)
        wm = [c["pnl"] for c in cells if c["in_month"] and c["pnl"] is not None]
        weeks.append({
            "cells": cells,
            "open_n": sum(c["open_n"] for c in cells if c["in_month"]),
            "r": round(sum(c["r"] or 0.0 for c in cells if c["in_month"]), 2),
            "pnl": round(sum(wm), 2) if wm else None,
            "n": sum(c["n"] for c in cells if c["in_month"]),
        })

    days = [(d, c) for d, c in by_day.items() if first_day <= d <= last_day]
    total_r = round(sum(c["r"] for _, c in days), 2)
    month_money = [c["pnl"] for _, c in days if c["pnl"] is not None]
    total_pnl = round(sum(month_money), 2) if month_money else None
    best = max(days, key=lambda x: _val(x[1])) if days else None
    worst = min(days, key=lambda x: _val(x[1])) if days else None
    return {
        "weeks": weeks,
        "weekdays": WEEKDAY_LABELS,
        "trading_days": len(days),
        "n": sum(c["n"] for _, c in days),
        "open_n": sum(v for d, v in open_by_day.items() if first_day <= d <= last_day),
        "total_r": total_r,
        "total_pnl": total_pnl,
        "currency": next((c["cur"] for _, c in days), DEFAULT_CURRENCY),
        "has_money": has_money,
        "win_days": sum(1 for _, c in days if _val(c) > (_MONEY_EPS if has_money else _R_EPS)),
        "loss_days": sum(1 for _, c in days if _val(c) < -(_MONEY_EPS if has_money else _R_EPS)),
        "avg_day_r": round(total_r / len(days), 2) if days else None,
        "avg_day_pnl": round(total_pnl / len(days), 2) if days and total_pnl is not None else None,
        "best": {"day": best[0].day, "r": best[1]["r"], "pnl": best[1]["pnl"],
                 "iso": best[0].isoformat()} if best else None,
        "worst": {"day": worst[0].day, "r": worst[1]["r"], "pnl": worst[1]["pnl"],
                  "iso": worst[0].isoformat()} if worst else None,
    }

async def vs_engine(db: AsyncSession, rows) -> dict:
    """Ben vs motor: kaynak bazinda karne + ayni setupta motorun R'si + kapi dokumu."""
    by_source = [
        {"key": key, "label": label, **_group(rows, lambda t, k=key: t.source == k)}
        for key, label in SOURCES
    ]

    pairs: list[dict] = []
    linked = [t for t in rows if t.signal_id]
    if linked:
        sigs = (await db.execute(
            select(Signal).where(Signal.id.in_([t.signal_id for t in linked])))).scalars().all()
        by_id = {s.id: s for s in sigs}
        for t in linked:
            s = by_id.get(t.signal_id)
            if s is None or s.rr_value is None or s.result is None:
                continue
            mine, engine = round(float(t.rr_value or 0.0), 2), round(float(s.rr_value), 2)
            pairs.append({
                "symbol": ui_symbol(t.symbol, t.market_type),
                "direction": t.direction,
                "mine": mine, "engine": engine, "diff": round(mine - engine, 2),
                "signal_id": s.id,
            })
    pairs.sort(key=lambda p: p["diff"])
    diff_total = round(sum(p["diff"] for p in pairs), 2) if pairs else None

    gates: dict[str, dict] = {}
    for t in rows:
        if t.source != "gated":
            continue
        key = t.gate or "bilinmiyor"
        g = gates.setdefault(key, {"gate": key, "n": 0, "r": 0.0, "wins": 0})
        g["n"] += 1
        g["r"] += float(t.rr_value or 0.0)
        g["wins"] += 1 if t.result == "win" else 0
    gate_rows = sorted(gates.values(), key=lambda g: g["r"])
    for g in gate_rows:
        g["r"] = round(g["r"], 2)
        g["win_rate"] = round(g["wins"] / g["n"] * 100) if g["n"] else None

    return {
        "by_source": by_source,
        "pairs": pairs[:10],
        "pairs_n": len(pairs),
        "diff_total": diff_total,
        "gates": gate_rows,
    }


def discipline(rows) -> dict:
    """Plana uyma orani + her hata etiketinin R maliyeti."""
    asked = [t for t in rows if t.followed_plan is not None]
    followed = _group(asked, lambda t: t.followed_plan is True)
    broken = _group(asked, lambda t: t.followed_plan is False)
    tags = []
    for key, label in MISTAKE_TAGS:
        group = _group(rows, lambda t, k=key: k in mistakes_of(t))
        if group["n"]:
            tags.append({"key": key, "label": label, **group})
    tags.sort(key=lambda x: x["r"])
    clean = _group(rows, lambda t: not mistakes_of(t))
    return {
        "asked_n": len(asked),
        "followed": followed,
        "broken": broken,
        "tags": tags,
        "clean": clean,
        "no_data": not asked and not tags,
    }


async def signal_suggestions(db: AsyncSession, limit: int = 40) -> list[dict]:
    """Form'daki "bot sinyalinden doldur" listesi: acik sinyaller."""
    try:
        sigs = (await db.execute(
            select(Signal).where(Signal.status.in_(("active", "waiting_entry", "pending_cisd")))
            .order_by(Signal.created_at.desc()).limit(limit))).scalars().all()
    except Exception:
        log.exception("paper: sinyal onerileri okunamadi")
        return []
    out = []
    for s in sigs:
        sl = s.initial_stop_loss if s.initial_stop_loss is not None else s.stop_loss
        out.append({
            "id": s.id,
            "symbol": s.symbol,
            "label": f"{ui_symbol(s.symbol, s.market_type)} {s.direction} "
                     f"({STRATEGY_LABELS.get(s.timeframe or '4h', s.timeframe or '4h')})",
            "direction": s.direction,
            "strategy": s.timeframe or "4h",
            "entry": s.entry_price,
            "sl": sl,
            "tp": s.take_profit,
        })
    return out


async def match_signal(db: AsyncSession, symbol: str, direction: str,
                       when: datetime | None, hours: int = 48) -> int | None:
    """Ice aktarilan isleme karsilik gelebilecek bot sinyali: ayni sembol + yon, +-48 saat.

    Yalnizca ONERI -- onizlemede gosterilir, kullanici istemezse baglanmaz. Amaci
    "ben vs motor" panosunun kendiliginden dolmasi.
    """
    if when is None or not symbol:
        return None
    try:
        rows = (await db.execute(
            select(Signal).where(
                Signal.symbol == symbol,
                Signal.direction == direction,
                Signal.created_at >= when - timedelta(hours=hours),
                Signal.created_at <= when + timedelta(hours=hours),
            )
        )).scalars().all()
    except Exception:
        log.exception("paper: sinyal eslesmesi aranamadi")
        return None
    if not rows:
        return None

    def _gap(sig) -> float:
        ref = sig.entry_filled_time or sig.created_at
        return abs((ref - when).total_seconds()) if ref else float("inf")

    return min(rows, key=_gap).id


async def open_trades_between(db: AsyncSession, start: datetime, end: datetime):
    """Bu aralikta ACILMIS ve hala acik islemler (takvimde isaret olarak gosterilir)."""
    return list((await db.execute(
        select(PaperTrade).where(
            PaperTrade.status == "open",
            PaperTrade.entered_at >= start,
            PaperTrade.entered_at < end,
        )
    )).scalars().all())
