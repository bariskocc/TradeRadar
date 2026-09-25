"""Deneme (paper) islem gunlugu -- ELLE tutulan islem kaydinin mantigi.

Motorun karnesi `Signal` tablosunda; burasi INSAN kararini olcer (bkz. models.PaperTrade).
Sayfa: /paper-trades. Bu modul saf mantik + sorgu; route'lar app/main.py'de.

Sayfa yalniz PARA ve win/loss gosterir (25.09, kullanici karari). SL/TP/R, TF, kaynak, cikis
nedeni ve disiplin alanlari kalkti; kolonlar DB'de duruyor (eski kayitlar), yeni kayitta bos.
`result` formda secildiyse o (Durum kutusu), yoksa paranin isaretinden, o da yoksa fiyat
hareketinin yonunden (`apply_close`, `result_of`). Giris/cikis fiyati zorunlu degil; giris zamani
her kayitta, cikis zamani kapali kayitta zorunlu.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.exchange import from_display_symbol, get_all_symbols_flat
from app.market_data import market_data
from app.models import PaperTrade

log = logging.getLogger(__name__)

TSI = timezone(timedelta(hours=3))

RESULT_LABELS = {"win": "Win", "loss": "Loss", "breakeven": "Breakeven"}

# Yeni / duzenle formundaki Durum kutusu (25.09): tablodaki DURUM sutunuyla ayni dort deger.
STATUS_OPTIONS = (("open", "Açık"), ("win", "Win"), ("loss", "Loss"), ("breakeven", "Breakeven"))
STATUS_LABELS = dict(STATUS_OPTIONS)

# Takvim: hafta Pazartesi baslar (donem ozetiyle ayni kural).
WEEKDAY_LABELS = ("Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz")

MARKET_OPTIONS = (
    ("crypto", "Crypto"), ("fx", "FX"), ("index", "Index"),
    ("metal", "Metal"), ("oil", "Oil"), ("other", "Diğer"),
)

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "TRY": "₺", "GBP": "£"}
DEFAULT_CURRENCY = "USD"

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


# ──────────────────── K/Z ────────────────────

def _move(direction: str, entry: float, price: float) -> float:
    return (price - entry) if (direction or "LONG") == "LONG" else (entry - price)


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


def result_of(pnl: float | None, move: float | None = None) -> str | None:
    """Sonuc paranin isaretinden (komisyon kucuk bir fiyat kazancini eksiye cevirdiginde "Win" +
    kirmizi tutar celiskisi olmasin). Para yoksa (miktar girilmemis elle kayit) fiyat hareketinin
    yonune duser (`move` = islem lehine fiyat farki)."""
    value = pnl if pnl is not None else move
    if value is None:
        return None
    if value > _MONEY_EPS:
        return "win"
    if value < -_MONEY_EPS:
        return "loss"
    return "breakeven"


def result_for(t) -> str | None:
    """Kapanmis islemin sonucu: para, yoksa giris -> cikis yonu."""
    move = None
    if t.entry_price is not None and t.exit_price is not None:
        move = _move(t.direction, float(t.entry_price), float(t.exit_price))
    return result_of(pnl_of(t), move)


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

def apply_plan(t: PaperTrade, form: dict) -> list[str]:
    """Acilis alanlarini forma gore yaz; donen liste hatalardir (bos = gecerli)."""
    errors: list[str] = []
    t.symbol = normalize_symbol(form.get("symbol"))
    if not t.symbol:
        errors.append("Sembol boş olamaz.")
    t.market_type = market_for(t.symbol)
    t.direction = "SHORT" if (form.get("direction") or "").upper() == "SHORT" else "LONG"
    t.strategy = t.strategy or "other"
    t.source = t.source or "manual"

    # Giris fiyati zorunlu DEGIL, giris zamani ZORUNLU (25.09, kullanici karari): kimi zaman yalniz
    # sonuc ve K/Z kaydediliyor ama tarih her zaman bilinmeli -- takvim, donem ozeti ve tarih
    # filtresi ona bakiyor; bos zamani "simdi" saymak gecmis islemi bugune yaziyordu.
    entry = parse_num(form.get("entry_price"))
    t.entry_price = entry
    entered_at = parse_dt(form.get("entered_at"))
    if entered_at is None:
        errors.append("Giriş zamanı gerekli.")
    t.entered_at = entered_at or t.entered_at
    # Miktar formda yok (25.09, yerine K/Z geldi); ice aktarilan kaydin miktari korunur.
    t.currency = ((form.get("currency") or t.currency or DEFAULT_CURRENCY).strip().upper() or
                  DEFAULT_CURRENCY)
    if t.qty and entry is not None:
        t.notional = round(t.qty * entry, 2)
    t.updated_at = now_utc()
    return errors


def apply_status(t: PaperTrade, form: dict) -> list[str]:
    """Formdaki Durum kutusu: Acik ise islem acik kalir (kapaliysa geri acilir), Win/Loss/BE ise
    o sonucla kapanir. Acik secilip K/Z ya da cikis fiyati girildiyse hata -- sessizce yok
    saymak girilen tutari kaybederdi."""
    status = (form.get("status") or "open").lower()
    if status not in STATUS_LABELS:
        status = "open"
    if status == "open":
        if parse_num(form.get("pnl_amount")) is not None or parse_num(form.get("exit_price")) is not None:
            return ["K/Z ya da çıkış fiyatı girdin ama durum Açık — işlem kapandıysa Win/Loss/Breakeven seç."]
        if t.status == "closed":
            reopen(t)
        return []
    return apply_close(t, {**form, "result": status})


def apply_close(t: PaperTrade, form: dict) -> list[str]:
    """Kapanis. Cikis fiyati zorunlu degil, cikis zamani ZORUNLU. Sonuc: formda secildiyse o, yoksa
    K/Z'nin isaretinden, o da yoksa giris -> cikis yonunden; hicbiri yoksa hata."""
    exit_price = parse_num(form.get("exit_price"))
    closed_at = parse_dt(form.get("closed_at"))
    if closed_at is None:
        return ["Çıkış zamanı gerekli."]
    if t.entered_at is not None and closed_at < t.entered_at:
        return ["Çıkış zamanı girişten önce olamaz."]
    t.exit_price = exit_price
    t.exit_reason = "manual"
    t.closed_at = closed_at
    fees = parse_num(form.get("fees"))
    if fees is not None or "fees" in form:
        t.fees = fees
    explicit = parse_num(form.get("pnl_amount"))
    t.pnl_amount = explicit if explicit is not None else computed_pnl(t, exit_price)
    # Ice aktarilan kayitta getiri % borsanin rakami; duzenleme onu ezmesin.
    if t.pnl_amount is not None and t.notional and not t.ext_source:
        t.return_pct = round(t.pnl_amount / float(t.notional) * 100, 2)
    chosen = (form.get("result") or "").lower()
    t.result = chosen if chosen in RESULT_LABELS else result_for(t)
    if t.result is None:
        return ["Sonuç belirlenemedi — K/Z, çıkış fiyatı ya da durum (Win/Loss/Breakeven) gir."]
    t.duration_hours = (round(max(0.0, (closed_at - t.entered_at).total_seconds() / 3600.0), 2)
                        if t.entered_at else None)
    t.status = "closed"
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
    pnl = pnl_of(t) if t.status == "closed" else None
    return {
        "t": t,
        "symbol": ui_symbol(t.symbol, t.market_type),
        "pnl": pnl,
        "live_pnl": computed_pnl(t, price) if t.status == "open" else None,
        "currency": t.currency or DEFAULT_CURRENCY,
        "result_label": RESULT_LABELS.get(t.result or "", ""),
        "price": price,
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
    if f.get("markets"):
        conds.append(PaperTrade.market_type.in_(f["markets"]))
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


async def closed_trades(db: AsyncSession, start=None, end=None) -> list[PaperTrade]:
    # R sarti YOK (25.09): SL'siz islemin R'si bos ama parasi var; eskiden ozetten ve takvimden
    # sessizce dusuyordu.
    q = select(PaperTrade).where(PaperTrade.status == "closed")
    if start is not None:
        q = q.where(PaperTrade.closed_at >= start)
    if end is not None:
        q = q.where(PaperTrade.closed_at < end)
    return list((await db.execute(q)).scalars().all())


def period_summary(rows) -> dict:
    """Donemde KAPANAN islemlerin ozeti -- Dashboard'daki blogun ayni sozlesmesi."""
    n = len(rows)
    money = [p for p in (pnl_of(t) for t in rows) if p is not None]
    wins = sum(1 for t in rows if t.result == "win")
    losses = sum(1 for t in rows if t.result == "loss")
    total_pnl = round(sum(money), 2) if money else None
    rank = lambda t: pnl_of(t) or 0.0  # noqa: E731

    def _pick(t):
        return {
            "symbol": ui_symbol(t.symbol, t.market_type),
            "pnl": pnl_of(t),
        }

    return {
        "n": n,
        "total_pnl": total_pnl,
        "currency": next((t.currency for t in rows if t.currency), DEFAULT_CURRENCY),
        "wins": wins,
        "losses": losses,
        "be": n - wins - losses,
        "win_rate": round(wins / n * 100) if n else None,
        "avg_pnl": round(total_pnl / len(money), 2) if money else None,
        "best": _pick(max(rows, key=rank)) if n else None,
        "worst": _pick(min(rows, key=rank)) if n > 1 else None,
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
        cell = by_day.setdefault(day, {"n": 0, "wins": 0, "losses": 0, "be": 0,
                                       "pnl": None, "cur": t.currency or DEFAULT_CURRENCY})
        money = pnl_of(t)
        if money is not None:
            cell["pnl"] = round((cell["pnl"] or 0.0) + money, 2)
        cell["n"] += 1
        if t.result == "win":
            cell["wins"] += 1
        elif t.result == "loss":
            cell["losses"] += 1
        elif t.result == "breakeven":
            cell["be"] += 1
    # Renk ve siralama olcutu: para (25.09'dan beri R gosterilmiyor).
    _val = lambda c: c["pnl"] if c["pnl"] is not None else 0.0  # noqa: E731
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
            value = _val(data) if data else None
            tone = "flat"
            if value is not None and abs(value) > _MONEY_EPS:
                tone = "win" if value > 0 else "loss"
            cells.append({
                "day": day.day,
                "iso": day.isoformat(),
                "in_month": first_day <= day <= last_day,
                "n": data["n"] if data else 0,
                "pnl": data["pnl"] if data else None,
                "currency": data["cur"] if data else DEFAULT_CURRENCY,
                "wins": data["wins"] if data else 0,
                "losses": data["losses"] if data else 0,
                "be": data["be"] if data else 0,
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
            "pnl": round(sum(wm), 2) if wm else None,
            "n": sum(c["n"] for c in cells if c["in_month"]),
        })

    days = [(d, c) for d, c in by_day.items() if first_day <= d <= last_day]
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
        "total_pnl": total_pnl,
        "currency": next((c["cur"] for _, c in days), DEFAULT_CURRENCY),
        "win_days": sum(1 for _, c in days if _val(c) > _MONEY_EPS),
        "loss_days": sum(1 for _, c in days if _val(c) < -_MONEY_EPS),
        "avg_day_pnl": round(total_pnl / len(days), 2) if days and total_pnl is not None else None,
        "best": {"day": best[0].day, "pnl": best[1]["pnl"],
                 "iso": best[0].isoformat()} if best else None,
        "worst": {"day": worst[0].day, "pnl": worst[1]["pnl"],
                  "iso": worst[0].isoformat()} if worst else None,
    }


async def open_trades_between(db: AsyncSession, start: datetime, end: datetime):
    """Bu aralikta ACILMIS ve hala acik islemler (takvimde isaret olarak gosterilir)."""
    return list((await db.execute(
        select(PaperTrade).where(
            PaperTrade.status == "open",
            PaperTrade.entered_at >= start,
            PaperTrade.entered_at < end,
        )
    )).scalars().all())
