"""TradingView paper trading **islem gecmisi** CSV'sini okur (`paper-trading-islem-gecmisi-*.csv`).

TradingView round trip'leri ZATEN esliyor (`Trade number` girisi ile cikisi gruplar), yani FIFO
kurmaya gerek yok. Verdigi: sembol, yon, giris/cikis fiyati + zamani, miktar, net K/Z, komisyon,
getiri %. Sayfa yalniz para + win/loss gosterdigi icin (25.09) bu tek dosya yeterli; SL/TP tasiyan
emir dosyasi artik okunmuyor.

Zaman: TradingView grafigin saat dilimiyle yaziyor; **TSİ kabul edilir** ve naive UTC'ye cevrilir.
Onizlemede kullaniciya boyle oldugu soylenir ki yanlissa yakalasin.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone

from app.paper_trades import (
    DEFAULT_CURRENCY, TSI, market_for, normalize_symbol, parse_num, ui_symbol,
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

    `used` ile bir baslik iki alana birden atanmaz ("Net PnL" ile "PnL" gibi).
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
        market = market_for(display)
        trades.append({
            "ext_id": entries[0]["order"] or f"{raw_symbol}#{trade_no}",
            "trade_no": trade_no,
            "raw_symbol": raw_symbol,
            "symbol": display,
            "label": ui_symbol(display, market),
            "known": market != "other",
            "market_type": market,
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
