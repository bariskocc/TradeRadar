"""TradingView paper trading CSV'lerini okur ve tek islem listesinde birlestirir.

Iki dosya:

1. **İşlem geçmişi** (`paper-trading-islem-gecmisi-*.csv`) — TradingView round trip'leri
   ZATEN eslemis veriyor (`Trade number` girisi ile cikisi gruplar), yani FIFO kurmaya gerek
   yok. Verdigi: sembol, yon, giris/cikis fiyati + zamani, miktar, net K/Z, komisyon, getiri %.
2. **Emirler** (`paper-trading-emirler-gerceklesti-*.csv`) — bracket seviyeleri BURADA:
   `Kâr Al` = TP, `Zarar durdur` = SL, ayrica kaldirac ve teminat. Islem gecmisi dosyasi stop
   tasimadigi icin R'yi ancak bu dosya verir; emir dosyasi yoksa SL onizlemede elle girilir.

Eslestirme: once **emir no**, tutmazsa sembol + yon + gerceklesme fiyati + **miktar** + zaman
yakinligi. Ikinci yol gerekli, cunku iki dosya farkli anlarda export edilince emir numaralari
ortusmeyebiliyor; miktar sarti da bu yolda yanlis eslesmeyi engellemek icin var (ayni sembolde
ayni fiyattan acilmis BASKA bir islemin stopu yazilmasin).

Zaman: TradingView grafigin saat dilimiyle yaziyor; **TSİ kabul edilir** ve naive UTC'ye cevrilir.
Onizlemede kullaniciya boyle oldugu soylenir ki yanlissa yakalasin.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone

from app.paper_trades import (
    DEFAULT_CURRENCY, TSI, calc_planned_rr, market_for, normalize_symbol, parse_num, ui_symbol,
)

log = logging.getLogger(__name__)

# Turkce ay kisaltmalari (normalize edilmis) + Ingilizce karsiliklari.
_MONTHS = {
    "oca": 1, "sub": 2, "mar": 3, "nis": 4, "may": 5, "haz": 6,
    "tem": 7, "agu": 8, "eyl": 9, "eki": 10, "kas": 11, "ara": 12,
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6,
    "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
    "jan": 1, "feb": 2, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_TR_MAP = str.maketrans("ıİşŞğĞüÜöÖçÇâÂî", "iisSgGuUoOcCaAi")

# Fallback eslesmede tolerans
_PRICE_TOL = 0.002      # %0.2
_QTY_TOL = 0.01         # %1
_TIME_TOL = timedelta(days=2)


def _norm(text: str | None) -> str:
    return " ".join((text or "").translate(_TR_MAP).lower().split())


def _read_csv(text: str) -> list[dict]:
    text = text.lstrip("﻿")
    # TradingView virgul kullaniyor; noktali virgulle export eden yerel ayarlar da var.
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    return [row for row in csv.DictReader(io.StringIO(text), delimiter=delim) if any(row.values())]


def _find(headers, *candidates: str, used: set | None = None) -> str | None:
    """Basligi normalize ederek bul: once tam esitlik, sonra basi, sonra icerme.

    `used` ile bir baslik iki alana birden atanmaz -- "Durdur fiyatı" (stop emri fiyati) ile
    "Zarar durdur" (SL) ayni kelimeyi tasidigi icin bu sart onemli.
    """
    used = used if used is not None else set()
    pairs = [(h, _norm(h)) for h in headers if h and h not in used]
    for test in (lambda n, c: n == c, lambda n, c: n.startswith(c), lambda n, c: c in n):
        for cand in candidates:
            for head, norm in pairs:
                if test(norm, cand):
                    used.add(head)
                    return head
    return None


def _num(value) -> float | None:
    """Sayiyi oku; "1,41 USD" gibi birim tasiyan alanlari da temizler."""
    if value is None:
        return None
    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch in ",.-")
    return parse_num(cleaned)


def _currency_of(header: str | None) -> str:
    """"Net PnL USD" -> "USD"."""
    parts = _norm(header).split()
    if parts and len(parts[-1]) == 3 and parts[-1].isalpha():
        return parts[-1].upper()
    return DEFAULT_CURRENCY


def parse_dt(value: str | None) -> datetime | None:
    """"20 Eyl 2026 10:01" ve "2026-09-20 10:07:32" -> naive UTC (girdi TSİ kabul edilir)."""
    raw = (value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=TSI).astimezone(
                timezone.utc).replace(tzinfo=None)
        except ValueError:
            pass
    parts = _norm(raw).replace(",", " ").split()
    if len(parts) >= 3:
        try:
            day = int(parts[0])
            month = _MONTHS.get(parts[1][:3]) or _MONTHS.get(parts[1])
            year = int(parts[2])
            hour = minute = 0
            if len(parts) >= 4 and ":" in parts[3]:
                bits = parts[3].split(":")
                hour, minute = int(bits[0]), int(bits[1])
            if month:
                return datetime(year, month, day, hour, minute, tzinfo=TSI).astimezone(
                    timezone.utc).replace(tzinfo=None)
        except (ValueError, IndexError):
            return None
    return None


# Broker sonekleri ve takma adlari: "ALCHEMY:USTEC.R" -> "US100".
# Yalnizca sembol ham haliyle TANINMADIGINDA denenir; kripto perp soneki ".P" bu yuzden guvende.
_BROKER_SUFFIXES = (".R", ".C", ".RAW", ".PRO")
_SYMBOL_ALIASES = {
    "USTEC": "US100", "NAS100": "US100", "NDX": "US100",
    "SPX500": "US500", "SPX": "US500", "US30": "US30",
    "USOIL": "OILWTI", "WTI": "OILWTI", "UKOIL": "OILBRENT", "BRENT": "OILBRENT",
    "GOLD": "XAUUSD", "SILVER": "XAGUSD",
}


def display_symbol(raw: str | None) -> str:
    """"BINGX:FETUSDT.P" -> "FETUSDT.P", "ALCHEMY:USTEC.R" -> "US100"."""
    text = (raw or "").strip()
    if ":" in text:
        text = text.split(":")[-1]
    known = normalize_symbol(text)
    if market_for(known) != "other":
        return known                                   # zaten taniniyor, dokunma
    bare = text.upper()
    for suffix in _BROKER_SUFFIXES:
        if bare.endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    alias = normalize_symbol(_SYMBOL_ALIASES.get(bare, bare))
    return alias if market_for(alias) != "other" else known


# ──────────────────── İşlem geçmişi ────────────────────

def parse_trades(text: str) -> tuple[list[dict], list[str]]:
    """Round trip listesi + uyarilar. Bir `Trade number` = bir islem."""
    rows = _read_csv(text)
    if not rows:
        raise ValueError("İşlem geçmişi dosyası boş.")
    heads = list(rows[0].keys())
    used: set = set()
    c = {
        "symbol": _find(heads, "sembol", "symbol", used=used),
        "trade": _find(heads, "trade number", "trade no", "islem no", used=used),
        "type": _find(heads, "tip", "type", used=used),
        "dt": _find(heads, "tarih ve saat", "tarih", "date/time", "date", used=used),
        "order": _find(heads, "emir no", "order id", "order no", used=used),
        "price": _find(heads, "fiyat", "price", used=used),
        "qty": _find(heads, "boyut (miktar)", "miktar", "qty", "quantity", used=used),
        "value": _find(heads, "boyut (deger)", "deger", "value", used=used),
        "pnl": _find(heads, "net pnl", "net p&l", "net k/z", "net kar", used=used),
        "ret": _find(heads, "getiri", "return", used=used),
        "fee": _find(heads, "komisyon", "commission", "fee", used=used),
    }
    missing = [key for key in ("symbol", "trade", "type", "dt", "price") if not c[key]]
    if missing:
        raise ValueError(
            "İşlem geçmişi dosyasında beklenen sütunlar yok (" + ", ".join(missing) + "). "
            "Bulunan sütunlar: " + ", ".join(heads))
    currency = _currency_of(c["pnl"])

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row[c["symbol"]], row[c["trade"]]), []).append(row)

    trades, warnings = [], []
    for (raw_symbol, trade_no), items in groups.items():
        parsed = []
        for row in items:
            kind = _norm(row.get(c["type"]))
            parsed.append({
                "entry": ("entry" in kind or "giris" in kind),
                "short": "short" in kind or "sat" in kind,
                "dt": parse_dt(row.get(c["dt"])),
                "price": _num(row.get(c["price"])),
                "qty": _num(row.get(c["qty"])) or 0.0,
                "order": (row.get(c["order"]) or "").strip(),
                "pnl": _num(row.get(c["pnl"])) if c["pnl"] else None,
                "fee": _num(row.get(c["fee"])) if c["fee"] else None,
                "ret": _num(row.get(c["ret"])) if c["ret"] else None,
                "value": _num(row.get(c["value"])) if c["value"] else None,
            })
        parsed.sort(key=lambda x: (x["dt"] or datetime.min, not x["entry"]))
        entries = [x for x in parsed if x["entry"]]
        exits = [x for x in parsed if not x["entry"]]
        if not entries or entries[0]["price"] is None:
            warnings.append(f"{raw_symbol} #{trade_no}: giriş satırı okunamadı, atlandı.")
            continue

        entry_qty = sum(x["qty"] for x in entries) or (entries[0]["qty"] or 0.0)
        entry_price = _weighted(entries) if len(entries) > 1 else entries[0]["price"]
        if len(entries) > 1:
            warnings.append(f"{raw_symbol} #{trade_no}: {len(entries)} girişli pozisyon, "
                            "giriş fiyatı ağırlıklı ortalama alındı.")

        partial_price = partial_fraction = None
        exit_price = closed_at = None
        if exits:
            final = exits[-1]
            exit_price, closed_at = final["price"], final["dt"]
            if len(exits) > 1:
                early = exits[:-1]
                partial_price = _weighted(early)
                early_qty = sum(x["qty"] for x in early)
                if entry_qty > 0 and 0 < early_qty < entry_qty:
                    partial_fraction = round(early_qty / entry_qty, 4)

        last = exits[-1] if exits else entries[-1]
        display = display_symbol(raw_symbol)
        trades.append({
            "ext_id": entries[0]["order"] or f"{raw_symbol}#{trade_no}",
            "exit_order": (exits[-1]["order"] if exits else ""),
            "trade_no": trade_no,
            "raw_symbol": raw_symbol,
            "symbol": display,
            "known": market_for(display) != "other",
            "market_type": market_for(display),
            "direction": "SHORT" if entries[0]["short"] else "LONG",
            "entry": entry_price,
            "exit": exit_price,
            "entered_at": entries[0]["dt"],
            "closed_at": closed_at,
            "qty": entry_qty or None,
            "notional": entries[0]["value"],
            "pnl": last["pnl"],
            "fees": last["fee"],
            "return_pct": last["ret"],
            "currency": currency,
            "partial_price": partial_price,
            "partial_fraction": partial_fraction,
            "sl": None, "tp": None, "leverage": None, "margin": None,
            "match": "yok",
            "status": "closed" if exit_price is not None else "open",
        })
    trades.sort(key=lambda t: t["entered_at"] or datetime.min)
    return trades, warnings


def _weighted(items: list[dict]) -> float | None:
    total = sum(x["qty"] for x in items)
    prices = [x for x in items if x["price"] is not None]
    if not prices:
        return None
    if total <= 0:
        return round(sum(x["price"] for x in prices) / len(prices), 10)
    return round(sum(x["price"] * x["qty"] for x in prices) / total, 10)


# ──────────────────── Emirler ────────────────────

def parse_orders(text: str) -> list[dict]:
    """Emir satirlari. SL/TP iki bicimde gelebilir:

    * **Sutun**: eski "emirler gerceklesti" exportunda giris emrinin satirinda `Kâr Al` / `Zarar durdur`.
    * **Ayri satir**: "emir gecmisi (tumu)" exportunda her bracket kendi satiri -- `Tür` = `Stop` ise
      seviye `Durdur fiyatı`ndadir (SL), `Limit` ise `Limit fiyatı`ndadir (TP). Bunlari `merge`
      icindeki `_attach_brackets` islemlere baglar.
    """
    rows = _read_csv(text)
    if not rows:
        return []
    heads = list(rows[0].keys())
    used: set = set()
    c = {
        "symbol": _find(heads, "sembol", "symbol", used=used),
        "side": _find(heads, "pozisyon", "side", "position", used=used),
        "tp": _find(heads, "kar al", "take profit", used=used),
        "sl": _find(heads, "zarar durdur", "stop loss", used=used),
        "fill": _find(heads, "gerceklesme fiyati", "gerceklesme", "fill price", "filled", used=used),
        "limit": _find(heads, "limit fiyati", "limit price", used=used),
        "qty": _find(heads, "miktar", "qty", "quantity", used=used),
        "dt": _find(heads, "emir zamani", "order time", "placing time", "time", used=used),
        "order": _find(heads, "emir no", "order id", "order no", used=used),
        "lev": _find(heads, "kaldirac", "leverage", used=used),
        "margin": _find(heads, "teminat", "margin", used=used),
        "kind": _find(heads, "tur", "type", used=used),
        "status": _find(heads, "durum", "status", used=used),
        "trigger": _find(heads, "durdur fiyati", "stop price", used=used),
        "closed": _find(heads, "kapanis saati", "closing time", "close time", used=used),
    }
    out = []
    for row in rows:
        side = _norm(row.get(c["side"])) if c["side"] else ""
        fill = _num(row.get(c["fill"])) if c["fill"] else None
        kind_raw = _norm(row.get(c["kind"])) if c["kind"] else ""
        kind = ("stop" if kind_raw.startswith(_STOP_KINDS)
                else "limit" if kind_raw.startswith(_LIMIT_KINDS) else "market")
        level = None
        if kind == "stop" and c["trigger"]:
            level = _num(row.get(c["trigger"]))
        elif kind == "limit" and c["limit"]:
            level = _num(row.get(c["limit"]))
        out.append({
            "kind": kind,
            "level": level,
            "status": _norm(row.get(c["status"])) if c["status"] else "",
            "t_close": parse_dt(row.get(c["closed"])) if c["closed"] else None,
            "symbol": display_symbol(row.get(c["symbol"])) if c["symbol"] else "",
            "side": side,
            "is_buy": side.startswith("al") or side.startswith("buy") or side.startswith("long"),
            "price": fill if fill is not None else (_num(row.get(c["limit"])) if c["limit"] else None),
            "qty": _num(row.get(c["qty"])) if c["qty"] else None,
            "tp": _num(row.get(c["tp"])) if c["tp"] else None,
            "sl": _num(row.get(c["sl"])) if c["sl"] else None,
            "dt": parse_dt(row.get(c["dt"])) if c["dt"] else None,
            "order": (row.get(c["order"]) or "").strip() if c["order"] else "",
            "leverage": _num(row.get(c["lev"])) if c["lev"] else None,
            "margin": _num(row.get(c["margin"])) if c["margin"] else None,
        })
    return out


def _close(a, b, tol: float) -> bool:
    if a is None or b is None:
        return False
    scale = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / scale <= tol


def _fallback_order(trade: dict, orders: list[dict]) -> dict | None:
    """Emir no tutmadiginda: ayni sembol + ayni yonde giris + fiyat + MIKTAR + zaman yakinligi."""
    want_buy = trade["direction"] == "LONG"
    best, best_gap = None, None
    for order in orders:
        if order["symbol"] != trade["symbol"] or order["is_buy"] != want_buy:
            continue
        if order["sl"] is None and order["tp"] is None:
            continue                                   # seviyesi olmayan emrin faydasi yok
        if not _close(order["price"], trade["entry"], _PRICE_TOL):
            continue
        if not _close(order["qty"], trade["qty"], _QTY_TOL):
            continue
        if order["dt"] is None or trade["entered_at"] is None:
            continue
        gap = abs(order["dt"] - trade["entered_at"])
        if gap > _TIME_TOL:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = order, gap
    return best


# Bracket penceresinin ucundaki tolerans. Giris/cikis emrinin SANIYELI zamanini
# cozebildigimizde 5 sn yeter; cozemezsek islem gecmisinin dakikaya yuvarli zamanina
# duseriz ve pencereyi 2 dk acmak zorunda kaliriz (bkz. _attach_brackets).
# `Tür` sutunundaki bracket adlari. TradingView ayni dosyada iki adlandirma kullaniyor:
# emirle birlikte konan koruma "Stop"/"Limit", pozisyona SONRADAN eklenen ise
# "Zarar Durdur"/"Kâr Al" yaziyor (22.09 #21'de fark edildi: satir "market" sanilip elenmis,
# islem SL'siz gelmisti). Seviye yeri ikisinde de ayni: stop -> `Durdur fiyatı`, limit -> `Limit fiyatı`.
_STOP_KINDS = ("stop", "zarar durdur", "stop loss")
_LIMIT_KINDS = ("limit", "kar al", "take profit")

_BRACKET_GRACE = timedelta(seconds=5)
_BRACKET_LOOSE_GRACE = timedelta(minutes=2)


def _attach_brackets(trades: list[dict], orders: list[dict]) -> None:
    """Ayri satir olarak gelen Stop/Limit emirlerini islemlere baglar.

    Kural: koruma emri **giris emrinden sonra veya ayni anda** girilmis ve **cikista**
    (dolarak ya da iptal edilerek) olmus olmali. Kullanici SL'i market emriyle birlikte de
    koyuyor, girdikten 10 sn sonra da -- ikisi de bu kurala uyar.

    Zaman penceresi islemin kendi zamanindan degil **emir dosyasindaki saniyeli** zamandan
    kurulur: islem gecmisi dakikaya yuvarli yaziyor ve o yuvarlamayla, girisin hemen oncesinde
    reddedilmis bir emirden kalan bayat bracket islemin ustune yapisiyor (21.09 #6 vakasi).

    Seviye birden cok kademeliyse (stop tasindi / pozisyon iki hedefe bolundu) **ilk girilen**
    yazilir -- R'nin paydasi daima ilk SL. Kademelerin tamami `sl_steps`/`tp_steps`'te durur,
    onizleme uyari rozetini oradan basar.
    """
    pool = [o for o in orders
            if o.get("kind") in ("stop", "limit") and o.get("level") is not None
            and not o.get("status", "").startswith(("reddedildi", "rejected"))]
    if not pool:
        return
    by_no = {o["order"]: o for o in orders if o.get("order")}
    for trade in trades:
        if trade.get("sl") is not None and trade.get("tp") is not None:
            continue
        entry_order = by_no.get(trade.get("ext_id") or "")
        exit_order = by_no.get(trade.get("exit_order") or "")
        exact = entry_order is not None and entry_order.get("dt") is not None
        start = entry_order["dt"] if exact else trade.get("entered_at")
        if start is None:
            continue
        end = exit_order.get("t_close") if exit_order is not None else None
        if end is None:
            end = trade.get("closed_at")
            exact = False
        grace = _BRACKET_GRACE if exact else _BRACKET_LOOSE_GRACE
        want_buy = trade["direction"] == "SHORT"       # koruma emri pozisyonun tersi yonde
        qty = trade.get("qty") or 0.0
        picked = []
        for order in pool:
            if order["symbol"] != trade["symbol"] or order["is_buy"] != want_buy:
                continue
            # Bolunmus bracket (7 birim -> 4 + 3) kucuk gelir; buyugu baska pozisyonundur.
            if qty and order.get("qty") and order["qty"] > qty * (1 + _QTY_TOL):
                continue
            if order.get("dt") is None or order["dt"] < start:
                continue
            if end is not None and (order.get("t_close") is None or order["t_close"] > end + grace):
                continue
            picked.append(order)
        if not picked:
            continue
        picked.sort(key=lambda o: o["dt"])
        steps = {"stop": [], "limit": []}
        for order in picked:
            bucket = steps[order["kind"]]
            if order["level"] not in bucket:
                bucket.append(order["level"])           # kronolojik, tekrarsiz
        if trade.get("sl") is None and steps["stop"]:
            trade["sl"] = steps["stop"][0]
        if trade.get("tp") is None and steps["limit"]:
            trade["tp"] = steps["limit"][0]
        trade["sl_steps"] = steps["stop"]
        trade["tp_steps"] = steps["limit"]
        trade["match"] = "bracket" if exact else "bracket (yakın)"


def merge(trades: list[dict], orders: list[dict]) -> list[dict]:
    """Islemlere emir dosyasindan SL/TP (+ kaldirac, teminat) ekler ve plan R'yi hesaplar."""
    by_no = {o["order"]: o for o in orders if o.get("order")}
    for trade in trades:
        trade.setdefault("sl_steps", [])
        trade.setdefault("tp_steps", [])
        order = by_no.get(trade["ext_id"])
        match = "emir no" if order is not None else None
        if order is None:
            order = _fallback_order(trade, orders)
            match = "yakın eşleşme" if order is not None else "yok"
        if order is not None:
            trade["sl"] = order.get("sl")
            trade["tp"] = order.get("tp")
            trade["leverage"] = order.get("leverage")
            trade["margin"] = order.get("margin")
        trade["match"] = match
    # Sutun bicimli dosyada seviyeler zaten geldi; bracket yolu yalnizca o bicim yokken calisir
    # ki eski export'un davranisi birebir korunsun.
    if not any(o.get("sl") is not None or o.get("tp") is not None for o in orders):
        _attach_brackets(trades, orders)
    for trade in trades:
        _recalc(trade)
    return trades


def _recalc(trade: dict) -> None:
    """Onizlemede gosterilen plan R ve gerceklesen R (SL yoksa ikisi de None)."""
    trade["planned_rr"] = calc_planned_rr(trade["entry"], trade["sl"], trade["tp"])
    rr = None
    if trade["sl"] is not None and trade["exit"] is not None and trade["entry"] is not None:
        risk = abs(float(trade["entry"]) - float(trade["sl"]))
        if risk > 0:
            move = ((float(trade["exit"]) - float(trade["entry"]))
                    if trade["direction"] == "LONG" else
                    (float(trade["entry"]) - float(trade["exit"])))
            frac = trade.get("partial_fraction") or 0.0
            if frac and trade.get("partial_price") is not None:
                part = ((float(trade["partial_price"]) - float(trade["entry"]))
                        if trade["direction"] == "LONG" else
                        (float(trade["entry"]) - float(trade["partial_price"])))
                rr = round(frac * (part / risk) + (1 - frac) * (move / risk), 2)
            else:
                rr = round(move / risk, 2)
    trade["rr"] = rr
    trade["label"] = ui_symbol(trade["symbol"], trade["market_type"])


def sl_side_ok(trade: dict) -> bool:
    """SL dogru tarafta mi (LONG'da girisin altinda)? Yanlis eslesmeyi onizlemede yakalar."""
    if trade.get("sl") is None or trade.get("entry") is None:
        return True
    return (float(trade["sl"]) < float(trade["entry"])) == (trade["direction"] == "LONG")
