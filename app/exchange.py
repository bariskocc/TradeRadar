"""Piyasa veri katmani - ccxt + yfinance OHLCV cekimi.

Desteklenen kaynaklar:
  - Binance (ccxt): Crypto spot
  - Bybit (ccxt):   Metal
  - Yahoo Finance:  Index + FX
"""

from __future__ import annotations

import asyncio
import logging

import ccxt.async_support as ccxt
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ──────────────────── Market → Exchange mapping ────────────────────

EXCHANGE_PER_MARKET: dict[str, str] = {
    "crypto": "binance",
    "metal":  "bybit",
    "index":  "yfinance",
    "fx":     "yfinance",
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
        "^NDX",                # Nasdaq 100 (US100)
    ],
    # Populer ve gorece korelasyonu dusuk 10 FX paritesi
    # (EURUSD varsa GBPUSD alinmiyor gibi):
    "fx": [
        "EURUSD=X",
        "USDJPY=X",
        "AUDUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "NZDUSD=X",
        "EURJPY=X",
        "GBPJPY=X",
        "AUDJPY=X",
        "CADJPY=X",
    ],
}

# ccxt symbol → temiz gösterim adı
DISPLAY_NAMES: dict[str, str] = {
    "XAU/USDT:USDT": "XAUUSD",
    "XAG/USDT:USDT": "XAGUSD",
    "^NDX": "US100",
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY",
    "AUDUSD=X": "AUDUSD",
    "USDCAD=X": "USDCAD",
    "USDCHF=X": "USDCHF",
    "NZDUSD=X": "NZDUSD",
    "EURJPY=X": "EURJPY",
    "GBPJPY=X": "GBPJPY",
    "AUDJPY=X": "AUDJPY",
    "CADJPY=X": "CADJPY",
}

# DB'deki temiz isimden → ccxt sembolüne geri dönüş
_REVERSE_DISPLAY: dict[str, tuple[str, str]] = {}
for _market, _symbols in SYMBOLS_BY_MARKET.items():
    _exc = EXCHANGE_PER_MARKET[_market]
    for _sym in _symbols:
        _db_name = DISPLAY_NAMES.get(_sym, _sym.replace("/", "").replace(":USDT", ""))
        _REVERSE_DISPLAY[_db_name] = (_sym, _exc)


def to_display_symbol(raw_symbol: str) -> str:
    """Kaynak sembolunu DB/UI adina cevir."""
    if raw_symbol in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw_symbol]
    return raw_symbol.replace("/", "").replace(":USDT", "").replace("=X", "")


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


async def create_exchanges() -> dict[str, object]:
    """Her kaynak için bir instance oluştur."""
    needed = set(EXCHANGE_PER_MARKET.values())
    exchanges: dict[str, object] = {}
    for eid in needed:
        if eid == "yfinance":
            exchanges[eid] = {"provider": "yfinance"}
            continue
        exchanges[eid] = await create_exchange(eid)
    return exchanges


async def close_exchanges(exchanges: dict[str, object]) -> None:
    for exc in exchanges.values():
        if isinstance(exc, dict):
            continue
        try:
            await exc.close()
        except Exception:
            pass


async def close_exchange(exchange: ccxt.Exchange) -> None:
    await exchange.close()


# ──────────────────── OHLCV data ────────────────────

async def fetch_ohlcv(
    exchange: object,
    symbol: str,
    timeframe: str = "4h",
    limit: int = 50,
    since_ms: int | None = None,
) -> pd.DataFrame:
    if isinstance(exchange, dict) and exchange.get("provider") == "yfinance":
        return await _fetch_yfinance_ohlcv(symbol, timeframe, limit, since_ms)

    raw = await exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance bazi surumlerde multi-index donebilir
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0

    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx
    return df[keep].dropna(subset=["open", "high", "low", "close"])


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h

    # FX/Index tarafinda 4H mumlarin 01/05/09/... UTC eksenine hizalanmasi
    out = df_1h.resample("4h", origin="start_day", offset="1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"])


async def _fetch_yfinance_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
    since_ms: int | None,
) -> pd.DataFrame:
    if timeframe == "4h":
        interval = "60m"
        period = "60d"
    elif timeframe == "15m":
        interval = "15m"
        period = "60d"
    elif timeframe == "1d":
        interval = "1d"
        period = "1y"
    else:
        raise ValueError(f"Unsupported yfinance timeframe: {timeframe}")

    def _download() -> pd.DataFrame:
        return yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

    raw = await asyncio.to_thread(_download)
    df = _normalize_yf_df(raw)
    if timeframe == "4h":
        df = _resample_4h(df)

    if since_ms is not None and not df.empty:
        since_ts = pd.to_datetime(since_ms, unit="ms", utc=True)
        df = df[df.index >= since_ts]

    if limit > 0 and not df.empty:
        df = df.tail(limit)
    return df


def get_all_symbols_flat() -> list[str]:
    """UI'da gösterim için tüm display sembollerini döndür."""
    result = []
    for market, symbols in SYMBOLS_BY_MARKET.items():
        for sym in symbols:
            result.append(to_display_symbol(sym))
    return result
