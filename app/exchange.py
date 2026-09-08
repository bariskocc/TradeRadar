"""Piyasa veri katmani - BingX OHLCV cekimi (REST bootstrap + WS icin sembol haritasi).

Tek kaynak: BingX USDT-M Perpetual Swap.
  - Kripto:  BTC-USDT, ETH-USDT, ...
  - Metal:   NCCOGOLD2USD (XAUUSD), NCCOXAG2USD (XAGUSD)
  - Petrol:  NCCO1OILWTI2USD (OILWTI), NCCO1OILBRENT2USD (OILBRENT)
  - Endeks:  NCSINASDAQ1002USD (US100), NCSISP5002USD (US500)
  - Forex:   NCFXEUR2USD (EURUSD), ...

Forex/endeks/metal/petrol enstrumanlari piyasa saatlerine tabidir; seans disinda
veri gelmeyebilir. Hafta sonu yalnizca kripto aktiftir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

from app.config import BINGX_REST_BASE

log = logging.getLogger(__name__)

# ──────────────────── Sembol listeleri (BingX) ────────────────────

# Kripto listesi 7 gun (haftaici + haftasonu) aktiftir.
_CRYPTO: list[str] = [
    # ── Ust sira (piyasa degeri + hacim liderleri) ──
    "BTC-USDT", "ETH-USDT",
    "BNB-USDT", "SOL-USDT", "XRP-USDT", "TRX-USDT", "DOGE-USDT",
    "HYPE-USDT", "ADA-USDT",
    # ── Layer-1 / Layer-2 ──
    "AVAX-USDT", "DOT-USDT", "LTC-USDT", "BCH-USDT", "ATOM-USDT",
    "XLM-USDT", "APT-USDT", "ARB-USDT", "OP-USDT", "SUI-USDT",
    "NEAR-USDT", "SEI-USDT", "TIA-USDT", "ETC-USDT",
    "KAS-USDT", "HBAR-USDT", "ICP-USDT",
    # ── DeFi / RWA / altyapi ──
    "LINK-USDT", "UNI-USDT", "AAVE-USDT", "ENA-USDT", "ONDO-USDT",
    "PENDLE-USDT", "LDO-USDT", "CRV-USDT", "INJ-USDT",
    "RUNE-USDT", "FIL-USDT", "POL-USDT", "JUP-USDT",
    "DYDX-USDT",
    # ── AI ──
    "TAO-USDT", "FET-USDT", "RENDER-USDT", "WLD-USDT",
    # ── Gizlilik ──
    "ZEC-USDT", "XMR-USDT",
    # ── Meme ──
    "1000PEPE-USDT",
]

# 1D-1H CRT: kripto yalnizca BTC + ETH (alt yok).
_D1H_CRYPTO: list[str] = ["BTC-USDT", "ETH-USDT"]

SYMBOLS_BY_MARKET: dict[str, list[str]] = {
    "crypto": _CRYPTO,
    "metal": [
        "NCCOGOLD2USD-USDT",       # Gold  (XAUUSD)
        "NCCOXAG2USD-USDT",        # Silver (XAGUSD)
    ],
    "oil": [
        "NCCO1OILWTI2USD-USDT",    # WTI Crude (OILWTI)
        "NCCO1OILBRENT2USD-USDT",  # Brent Crude (OILBRENT)
    ],
    "index": [
        "NCSINASDAQ1002USD-USDT",  # Nasdaq 100 (US100)
        "NCSISP5002USD-USDT",      # S&P 500 (US500)
    ],
    "fx": [
        "NCFXEUR2USD-USDT",        # EURUSD
        "NCFXGBP2USD-USDT",        # GBPUSD
        "NCFXUSD2JPY-USDT",        # USDJPY
        "NCFXAUD2USD-USDT",        # AUDUSD
        "NCFXUSD2CAD-USDT",        # USDCAD
        "NCFXUSD2CHF-USDT",        # USDCHF
        "NCFXNZD2USD-USDT",        # NZDUSD
        "NCFXEUR2JPY-USDT",        # EURJPY
        "NCFXGBP2JPY-USDT",        # GBPJPY
        "NCFXAUD2JPY-USDT",        # AUDJPY
        "NCFXCAD2JPY-USDT",        # CADJPY
        "NCFXNZD2JPY-USDT",        # NZDJPY
        "NCFXEUR2GBP-USDT",        # EURGBP
        "NCFXEUR2CHF-USDT",        # EURCHF
        "NCFXEUR2CAD-USDT",        # EURCAD
        "NCFXGBP2AUD-USDT",        # GBPAUD
        "NCFXGBP2CHF-USDT",        # GBPCHF
        "NCFXNZD2CAD-USDT",        # NZDCAD
    ],
}
_CRYPTO_SYMBOL_SET = set(_CRYPTO)

# BingX ham sembol → temiz gosterim adi (DB/UI)
# NOT: 1000x carpanli meme kontratlari (1000PEPE/1000SHIB/1000BONK) BingX
# vadelide bu adlarla islem gorur; ozel eslemesi YOKTUR ki gosterimde de
# gercek vadeli ticker korunsun (or. "1000PEPEUSDT.P"). Spot adiyla
# ("PEPEUSDT") gosterilirse vadelide olmayan bir sembolle karistirilir.
DISPLAY_NAMES: dict[str, str] = {
    # Metal
    "NCCOGOLD2USD-USDT": "XAUUSD",
    "NCCOXAG2USD-USDT": "XAGUSD",
    # Petrol
    "NCCO1OILWTI2USD-USDT": "OILWTI",
    "NCCO1OILBRENT2USD-USDT": "OILBRENT",
    # Endeks
    "NCSINASDAQ1002USD-USDT": "US100",
    "NCSISP5002USD-USDT": "US500",
    # Forex
    "NCFXEUR2USD-USDT": "EURUSD",
    "NCFXGBP2USD-USDT": "GBPUSD",
    "NCFXUSD2JPY-USDT": "USDJPY",
    "NCFXAUD2USD-USDT": "AUDUSD",
    "NCFXUSD2CAD-USDT": "USDCAD",
    "NCFXUSD2CHF-USDT": "USDCHF",
    "NCFXNZD2USD-USDT": "NZDUSD",
    "NCFXEUR2JPY-USDT": "EURJPY",
    "NCFXGBP2JPY-USDT": "GBPJPY",
    "NCFXAUD2JPY-USDT": "AUDJPY",
    "NCFXCAD2JPY-USDT": "CADJPY",
    "NCFXNZD2JPY-USDT": "NZDJPY",
    "NCFXEUR2GBP-USDT": "EURGBP",
    "NCFXEUR2CHF-USDT": "EURCHF",
    "NCFXEUR2CAD-USDT": "EURCAD",
    "NCFXGBP2AUD-USDT": "GBPAUD",
    "NCFXGBP2CHF-USDT": "GBPCHF",
    "NCFXNZD2CAD-USDT": "NZDCAD",
}

# Her sembolun ait oldugu market
MARKET_BY_SYMBOL: dict[str, str] = {}
for _market, _symbols in SYMBOLS_BY_MARKET.items():
    for _sym in _symbols:
        MARKET_BY_SYMBOL[_sym] = _market


# ──────────────────── SMT korelasyon haritasi ────────────────────
# SMT (Smart Money Technique) divergence icin POZITIF korele parite ciftleri.
# Ikisi de ayni yone hareket etmesi beklenir; biri yeni tepe/dip yaparken
# digeri yapamiyorsa divergence vardir. Asagidaki ciftler cift yonlu
# (a->b ve b->a). Kripto: BTC <-> ETH ozel cift; DIGER TUM ALTLAR -> BTC
# (bkz. correlated_symbol). Alt->BTC eslemesi BTC'nin ETH referansini
# ezmemek icin tek yonlu tutulur.
_SMT_BTC = "BTC-USDT"
_SMT_PAIRS: list[tuple[str, str]] = [
    (_SMT_BTC, "ETH-USDT"),                                # kripto majors
    ("NCSINASDAQ1002USD-USDT", "NCSISP5002USD-USDT"),      # US100 <-> US500
    ("NCCO1OILWTI2USD-USDT", "NCCO1OILBRENT2USD-USDT"),    # OILWTI <-> OILBRENT
    ("NCCOGOLD2USD-USDT", "NCCOXAG2USD-USDT"),             # XAUUSD <-> XAGUSD
    ("NCFXEUR2USD-USDT", "NCFXGBP2USD-USDT"),              # EURUSD <-> GBPUSD
    ("NCFXAUD2USD-USDT", "NCFXNZD2USD-USDT"),              # AUDUSD <-> NZDUSD
    ("NCFXEUR2JPY-USDT", "NCFXGBP2JPY-USDT"),              # EURJPY <-> GBPJPY
    ("NCFXAUD2JPY-USDT", "NCFXNZD2JPY-USDT"),              # AUDJPY <-> NZDJPY (risk-on JPY)
    ("NCFXUSD2JPY-USDT", "NCFXUSD2CHF-USDT"),              # USDJPY <-> USDCHF (USD gucu)
    ("NCFXEUR2CHF-USDT", "NCFXGBP2CHF-USDT"),              # EURCHF <-> GBPCHF (CHF quote)
]

SMT_CORRELATION: dict[str, str] = {}
for _a, _b in _SMT_PAIRS:
    SMT_CORRELATION[_a] = _b
    SMT_CORRELATION[_b] = _a


def correlated_symbol(bingx_symbol: str) -> str | None:
    """SMT icin korele parite.

    - Acik ciftler (BTC<->ETH, FX/metal/endeks/petrol): SMT_CORRELATION
    - Diger tum kripto altlar: BTC (tek yon; BTC tarafinda ETH kalir)
    - Eslesme yoksa None
    """
    if bingx_symbol in SMT_CORRELATION:
        return SMT_CORRELATION[bingx_symbol]
    if bingx_symbol in _CRYPTO_SYMBOL_SET and bingx_symbol != _SMT_BTC:
        return _SMT_BTC
    return None


def to_display_symbol(raw_symbol: str) -> str:
    """BingX ham sembolunu DB/UI adina cevir."""
    if raw_symbol in DISPLAY_NAMES:
        display = DISPLAY_NAMES[raw_symbol]
    else:
        # Kripto: "BTC-USDT" → "BTCUSDT"
        display = raw_symbol.replace("-", "")
    # Kripto perpetual pariteleri UI/DB'de ".P" ile gosterilir.
    if raw_symbol in _CRYPTO_SYMBOL_SET and not display.endswith(".P"):
        display = f"{display}.P"
    return display


# DB'deki temiz isimden → (BingX sembolu, market) geri donus
_REVERSE_DISPLAY: dict[str, tuple[str, str]] = {}
for _market, _symbols in SYMBOLS_BY_MARKET.items():
    for _sym in _symbols:
        _REVERSE_DISPLAY[to_display_symbol(_sym)] = (_sym, _market)


def from_display_symbol(db_symbol: str) -> tuple[str, str]:
    """DB sembolunden (bingx_symbol, market) dondur."""
    if db_symbol in _REVERSE_DISPLAY:
        return _REVERSE_DISPLAY[db_symbol]
    # Bilinmeyen kripto: "BTCUSDT.P" → "BTC-USDT"
    normalized = db_symbol.upper().strip()
    if normalized.endswith(".P"):
        normalized = normalized[:-2]
    if normalized.endswith("USDT"):
        base = normalized[:-4]
        return f"{base}-USDT", "crypto"
    return normalized, "crypto"


def market_of(bingx_symbol: str) -> str:
    return MARKET_BY_SYMBOL.get(bingx_symbol, "crypto")


# ──────────────────── Gun bazli aktif market secimi ────────────────────


def is_weekend() -> bool:
    return datetime.now(timezone.utc).weekday() >= 5


def get_active_markets() -> dict[str, list[str]]:
    """4H-15M evreni: kripto (7 gun) + XAUUSD (haftaici).

    Diger metal/fx/oil/index 4H'te kapali; 1D-1H `get_d1h_markets` ile acik.
    """
    active: dict[str, list[str]] = {"crypto": _CRYPTO}
    if not is_weekend():
        active["metal"] = ["NCCOGOLD2USD-USDT"]  # XAUUSD
        # active["metal"] = SYMBOLS_BY_MARKET["metal"]  # XAGUSD
        # active["oil"] = SYMBOLS_BY_MARKET["oil"]
        # active["index"] = SYMBOLS_BY_MARKET["index"]
        # active["fx"] = SYMBOLS_BY_MARKET["fx"]
    return active


def get_active_symbols_flat() -> list[str]:
    """Gun bazli aktif tum BingX sembollerini tek listede dondur."""
    result: list[str] = []
    for syms in get_active_markets().values():
        result.extend(syms)
    return result


def get_d1h_markets() -> dict[str, list[str]]:
    """1D-1H CRT evreni: kripto yalnizca BTC+ETH; global marketler 4H ile ayni.

    Hafta sonu yalnizca BTC ve ETH (FX/metal/endeks/petrol kapali).
    """
    active: dict[str, list[str]] = {"crypto": list(_D1H_CRYPTO)}
    if not is_weekend():
        active["metal"] = SYMBOLS_BY_MARKET["metal"]
        active["oil"] = SYMBOLS_BY_MARKET["oil"]
        active["index"] = SYMBOLS_BY_MARKET["index"]
        active["fx"] = SYMBOLS_BY_MARKET["fx"]
    return active


def get_d1h_symbols_flat() -> list[str]:
    """1D-1H CRT icin gun bazli BingX sembolleri."""
    result: list[str] = []
    for syms in get_d1h_markets().values():
        result.extend(syms)
    return result


_D1H_ALL: set[str] = set(_D1H_CRYPTO)
for _m in ("metal", "oil", "index", "fx"):
    _D1H_ALL.update(SYMBOLS_BY_MARKET[_m])


def is_d1h_symbol(bingx_symbol: str) -> bool:
    return bingx_symbol in _D1H_ALL


# 1H-5M CRT: XAU, EURUSD, Nasdaq (US100), BTC.
# SMT esleri CRT degil; 5M abone edilir (XAG, GBPUSD, US500, ETH).
_H1_CRT: list[str] = [
    "NCCOGOLD2USD-USDT",       # XAUUSD
    "NCFXEUR2USD-USDT",        # EURUSD
    "NCSINASDAQ1002USD-USDT",  # US100
    "BTC-USDT",
]
_H1_CRT_SET = set(_H1_CRT)
_H1_SMT_EXTRA: list[str] = [
    "NCCOXAG2USD-USDT",        # XAU SMT
    "NCFXGBP2USD-USDT",        # EUR SMT
    "NCSISP5002USD-USDT",      # US100 SMT
    "ETH-USDT",                # BTC SMT
]


def _h1_open_now(bingx_symbol: str) -> bool:
    """FX/metal/endeks hafta sonu kapali; kripto 7 gun."""
    if not is_weekend():
        return True
    return MARKET_BY_SYMBOL.get(bingx_symbol, "crypto") == "crypto"


def get_h1_symbols_flat() -> list[str]:
    """1H-5M CRT evreni. Hafta sonu yalnizca BTC."""
    return [s for s in _H1_CRT if _h1_open_now(s)]


def get_h1_5m_symbols() -> list[str]:
    """5M WS/bootstrap: CRT + SMT esleri. Hafta sonu BTC+ETH."""
    out: list[str] = []
    seen: set[str] = set()
    for s in _H1_CRT + _H1_SMT_EXTRA:
        if not _h1_open_now(s) or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def is_h1_symbol(bingx_symbol: str) -> bool:
    return bingx_symbol in _H1_CRT_SET


def get_h1_markets() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for s in get_h1_symbols_flat():
        result.setdefault(market_of(s), []).append(s)
    return result


# ──────────────────── Timeframe eslemesi ────────────────────

# Uygulama timeframe'i → BingX interval (native destekli)
_TIMEFRAME_MAP: dict[str, str] = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def to_bingx_interval(timeframe: str) -> str:
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_MAP[timeframe]


# ──────────────────── OHLCV (REST bootstrap) ────────────────────

_KLINES_PATH = "/openApi/swap/v3/quote/klines"


def _klines_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    records = []
    for r in rows:
        records.append({
            "timestamp": pd.to_datetime(int(r["time"]), unit="ms", utc=True),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 0.0)),
        })
    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    # Ayni timestamp tekrar gelirse sonuncuyu koru.
    df = df[~df.index.duplicated(keep="last")]
    return df


async def fetch_ohlcv(
    symbol: str,
    timeframe: str = "4h",
    limit: int = 50,
    since_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """BingX REST ile OHLCV cek (bootstrap / reconciliation icin).

    `symbol` BingX ham sembolu (or. "BTC-USDT", "NCFXEUR2USD-USDT").
    """
    interval = to_bingx_interval(timeframe)
    params: dict[str, object] = {
        "symbol": symbol,
        "interval": interval,
        "limit": max(1, min(int(limit), 1000)),
    }
    if since_ms is not None:
        params["startTime"] = int(since_ms)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0)
    try:
        resp = await client.get(_KLINES_PATH, params=params)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        log.warning("BingX kline fetch failed for %s %s: %s", symbol, timeframe, e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    finally:
        if owns_client:
            await client.aclose()

    if payload.get("code") not in (0, None):
        log.warning("BingX kline error %s for %s: %s",
                    payload.get("code"), symbol, payload.get("msg"))
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    rows = payload.get("data") or []
    df = _klines_to_df(rows)
    if limit > 0 and not df.empty:
        df = df.tail(limit)
    return df


def get_all_symbols_flat() -> list[str]:
    """UI'da gosterim icin tum display sembollerini dondur."""
    result = []
    for symbols in SYMBOLS_BY_MARKET.values():
        for sym in symbols:
            result.append(to_display_symbol(sym))
    return result
