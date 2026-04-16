"""ccxt wrapper – multi-exchange OHLCV verisi çekme.

Desteklenen exchange'ler:
  - Binance: Crypto spot (BTC/USDT, ETH/USDT, ...)
  - Bybit:   Crypto/Metal/Index/FX linear perpetual (XAU/USDT:USDT, ...)
"""

from __future__ import annotations

import asyncio
import logging

import ccxt.async_support as ccxt
import pandas as pd

log = logging.getLogger(__name__)

# ──────────────────── Market → Exchange mapping ────────────────────

EXCHANGE_PER_MARKET: dict[str, str] = {
    "crypto": "binance",
    "metal":  "bybit",
    "index":  "bybit",
    "fx":     "bybit",
}

SYMBOLS_BY_MARKET: dict[str, list[str]] = {
    "crypto": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
        "SHIB/USDT", "TRX/USDT", "NEAR/USDT", "SUI/USDT", "APT/USDT",
        "OP/USDT", "ARB/USDT", "INJ/USDT", "FTM/USDT", "PEPE/USDT",
        "WIF/USDT", "FET/USDT", "ONDO/USDT", "RENDER/USDT", "TIA/USDT",
        "SEI/USDT", "AAVE/USDT", "UNI/USDT", "LTC/USDT", "BCH/USDT",
    ],
    "metal": [
        "XAU/USDT:USDT",       # Gold
        "XAG/USDT:USDT",       # Silver
    ],
    "index": [
        "BTC/USD:BTC",         # BTC Inverse (index benchmark)
    ],
    "fx": [],
}

# ccxt symbol → temiz gösterim adı
DISPLAY_NAMES: dict[str, str] = {
    "XAU/USDT:USDT": "XAUUSD",
    "XAG/USDT:USDT": "XAGUSD",
    "BTC/USD:BTC": "BTCUSD_INV",
}

# DB'deki temiz isimden → ccxt sembolüne geri dönüş
_REVERSE_DISPLAY: dict[str, tuple[str, str]] = {}
for _market, _symbols in SYMBOLS_BY_MARKET.items():
    _exc = EXCHANGE_PER_MARKET[_market]
    for _sym in _symbols:
        _db_name = DISPLAY_NAMES.get(_sym, _sym.replace("/", "").replace(":USDT", ""))
        _REVERSE_DISPLAY[_db_name] = (_sym, _exc)


def to_display_symbol(ccxt_symbol: str) -> str:
    """ccxt sembolünü DB/UI gösterim adına çevir. BTC/USDT → BTCUSDT, XAU/USDT:USDT → XAUUSD"""
    if ccxt_symbol in DISPLAY_NAMES:
        return DISPLAY_NAMES[ccxt_symbol]
    return ccxt_symbol.replace("/", "").replace(":USDT", "")


def from_display_symbol(db_symbol: str) -> tuple[str, str]:
    """DB sembolünden (ccxt_symbol, exchange_id) döndür."""
    if db_symbol in _REVERSE_DISPLAY:
        return _REVERSE_DISPLAY[db_symbol]
    return db_symbol.replace("USDT", "/USDT"), "binance"


# ──────────────────── Exchange factory ────────────────────

_EXCHANGE_CONFIGS: dict[str, dict] = {
    "binance": {
        "enableRateLimit": True,
    },
    "bybit": {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    },
}


async def create_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    exchange_cls = getattr(ccxt, exchange_id)
    config = _EXCHANGE_CONFIGS.get(exchange_id, {"enableRateLimit": True})
    exchange: ccxt.Exchange = exchange_cls(config)
    return exchange


async def create_exchanges() -> dict[str, ccxt.Exchange]:
    """Her exchange_id için bir instance oluştur."""
    needed = set(EXCHANGE_PER_MARKET.values())
    exchanges: dict[str, ccxt.Exchange] = {}
    for eid in needed:
        exchanges[eid] = await create_exchange(eid)
    return exchanges


async def close_exchanges(exchanges: dict[str, ccxt.Exchange]) -> None:
    for exc in exchanges.values():
        try:
            await exc.close()
        except Exception:
            pass


async def close_exchange(exchange: ccxt.Exchange) -> None:
    await exchange.close()


# ──────────────────── OHLCV data ────────────────────

async def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "4h",
    limit: int = 50,
    since_ms: int | None = None,
) -> pd.DataFrame:
    raw = await exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def get_all_symbols_flat() -> list[str]:
    """UI'da gösterim için tüm display sembollerini döndür."""
    result = []
    for market, symbols in SYMBOLS_BY_MARKET.items():
        for sym in symbols:
            result.append(to_display_symbol(sym))
    return result
