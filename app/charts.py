"""Charts sayfasinin mum verisi (23.10) — motordan bagimsiz, yalniz gosterim.

Canli kisim botun WS store'undan gelir (REST yok). Uzun gecmis icin her (sembol, TF) basina
BingX'e EN FAZLA BIR istek atilir; sonuc bellekte ve diskte (`chart_cache/`) tutulur:

- Restart sonrasi disk + store arasinda bosluk yoksa hic istek atilmaz (bot birkac gunden kisa
  kapali kaldiysa store'un 15m/4H penceresi boslugu kapatir).
- Istekler sirali ve en az `_MIN_REQ_GAP_SEC` arayla gider; bootstrap bitmeden hic gitmez
  (botun kendi yuklemesiyle yarisip hiz sinirina takilmasin).
- Hata alan (sembol, TF) bu surecte bir daha istenmez: BingX 15 dk'da 10'dan fazla HATALI
  istekte IP'yi kilitliyor (109429) ve bu canli botu da durdurur.
- Seans sembollerinde (FX/metal/endeks/petrol) 1H/4H/1D/1W REST'ten CEKILMEZ — BingX'in
  mumlari UTC'ye hizali, motorun NY-hizali serisi kullanilir. Yalniz 15m cekilebilir (kova
  sinirlari ayni) ve spot FX'te olu seansta o da istenmez (BingX duraklamasi = hatali istek).

Store'u degistirmez: motorun serileri okunur, kopyalanir.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app import session as fx_session
from app.config import BASE_DIR
from app.exchange import fetch_ohlcv, from_display_symbol, is_session_symbol, market_of

log = logging.getLogger(__name__)

CHART_TFS = ("15m", "1h", "4h", "1d", "1w")
_COLS = ["open", "high", "low", "close"]
_TF_RULE = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
_TF_SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 7 * 86400}
_REST_LIMIT = 1000                 # BingX tek istek tavani
_MIN_REQ_GAP_SEC = 1.0
_PERSIST_EVERY_SEC = 600           # canli uzayan seri diske en fazla 10 dk'da bir
TSI_OFFSET_SEC = 3 * 3600          # Lightweight Charts saati UTC basar; TSİ gorunsun diye kaydir

CACHE_DIR = BASE_DIR / "chart_cache"

_HIST: dict[tuple[str, str], pd.DataFrame] = {}     # REST/disk gecmisi
_DISK_LOADED: set[tuple[str, str]] = set()
_FETCHED: set[tuple[str, str]] = set()              # bu surecte istek atildi (basarili)
_FAILED: set[tuple[str, str]] = set()               # bu surecte bir daha istenmez
_INFLIGHT: dict[tuple[str, str], asyncio.Task] = {}
_LAST_PERSIST: dict[tuple[str, str], float] = {}
_REQ_LOCK = asyncio.Lock()
_last_req_at = 0.0


# ──────────────────── yardimcilar ────────────────────

def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def _clean(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    out = df[_COLS].astype(float).copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _merge(*frames: pd.DataFrame) -> pd.DataFrame:
    """Sonraki kaynak ayni mumda oncekini ezer (daha canli kaynak sona)."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return _empty()
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Ince seriden kaba TF (UTC hizali, kripto). Ilk kova eksikse atilir."""
    if df.empty:
        return _empty()
    rule = _TF_RULE[tf]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if not out.empty and df.index[0] > out.index[0]:
        out = out.iloc[1:]
    return out


def _weekly(df_1d: pd.DataFrame) -> pd.DataFrame:
    """1D -> 1W; motorla ayni gruplama (index = haftanin ilk 1D bari)."""
    from app.crt_engine import daily_to_weekly_bars

    if df_1d.empty:
        return _empty()
    return _clean(daily_to_weekly_bars(df_1d.assign(volume=0.0)))


def _cache_path(raw: str, tf: str) -> Path:
    return CACHE_DIR / f"{raw.replace('/', '_')}_{tf}.pkl"


def _load_disk(key: tuple[str, str]) -> None:
    if key in _DISK_LOADED:
        return
    _DISK_LOADED.add(key)
    p = _cache_path(*key)
    if not p.exists():
        return
    try:
        _HIST[key] = _clean(pd.read_pickle(p))
    except Exception as e:
        log.warning("CHART cache okunamadi %s: %s", p.name, e)


def _save_disk(key: tuple[str, str], df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        df.tail(_REST_LIMIT * 3).to_pickle(_cache_path(*key))
        _LAST_PERSIST[key] = time.monotonic()
    except Exception as e:
        log.warning("CHART cache yazilamadi %s %s: %s", key[0], key[1], e)


# ──────────────────── canli seri (store) ────────────────────

def _store(store, raw: str, tf: str) -> pd.DataFrame:
    try:
        return _clean(store.get_df(raw, tf))
    except Exception:
        return _empty()


def _live(store, raw: str, tf: str) -> pd.DataFrame:
    """Store'dan canli seri. Kriptoda dinlenmeyen TF'nin olusan mumu inceden uretilir."""
    if is_session_symbol(raw):
        if tf in ("1d", "1w"):
            # 1W evrenindeki seans sembollerinde "1d_deep" ~1 yil (bootstrap'ta bir kez); "1d" canli, o kazanir
            d1 = _merge(_store(store, raw, "1d_deep"), _store(store, raw, "1d"))
            return d1 if tf == "1d" else _weekly(d1)
        return _store(store, raw, tf)
    m15 = _store(store, raw, "15m")
    h4 = _extend(_store(store, raw, "4h"), _resample(m15, "4h"))
    if tf == "15m":
        return m15
    if tf == "1h":
        return _extend(_store(store, raw, "1h"), _resample(m15, "1h"))
    if tf == "4h":
        return h4
    d1 = _extend(_merge(_store(store, raw, "1d_deep"), _store(store, raw, "1d")), _resample(h4, "1d"))
    return d1 if tf == "1d" else _weekly(d1)


def _extend(base: pd.DataFrame, finer: pd.DataFrame) -> pd.DataFrame:
    """Dogrudan seriyi inceden uretilen mumlarla YALNIZ sondan uzat (olusan mum + yenileri).

    Ince seride bosluk olursa (WS kacagi) dogrudan serinin dogru mumlarini ezmesin.
    """
    if base.empty:
        return finer
    if finer.empty:
        return base
    return _merge(base, finer[finer.index >= base.index[-1]])


# ──────────────────── REST gecmisi ────────────────────

def _rest_allowed(raw: str, tf: str) -> bool:
    if tf == "1w":
        return False                       # 1D'den uretilir
    if not is_session_symbol(raw):
        return True
    if tf != "15m":
        return False                       # motorun NY-hizali serisi kullanilir
    if market_of(raw) == "fx" and fx_session.is_dead_session(datetime.now(timezone.utc), raw):
        return False                       # BingX FX'i durdurur: istek hata doner
    return True


async def _fetch(key: tuple[str, str]) -> None:
    global _last_req_at
    raw, tf = key
    async with _REQ_LOCK:
        wait = _MIN_REQ_GAP_SEC - (time.monotonic() - _last_req_at)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            df = await fetch_ohlcv(raw, tf, limit=_REST_LIMIT)
        except Exception as e:
            log.warning("CHART gecmis istegi basarisiz %s %s: %s", raw, tf, e)
            df = _empty()
            df.attrs["bingx_error"] = "exc"
        finally:
            _last_req_at = time.monotonic()
    if df.empty:
        _FAILED.add(key)
        log.info("CHART gecmis bos/hata %s %s (%s) -- bu surecte tekrar istenmez",
                 raw, tf, df.attrs.get("bingx_error"))
        return
    if is_session_symbol(raw):
        df = fx_session.drop_dead_session(df, raw)
    hist = _merge(_HIST.get(key, _empty()), _clean(df))
    _HIST[key] = hist
    _FETCHED.add(key)
    _save_disk(key, hist)
    log.info("CHART gecmis %s %s: %d mum", raw, tf, len(df))


def _needs_fetch(key: tuple[str, str], live: pd.DataFrame) -> bool:
    if key in _FETCHED or key in _FAILED or not _rest_allowed(*key):
        return False
    hist = _HIST.get(key)
    if hist is None or hist.empty:
        return True
    if live.empty:
        return True                        # canli kaynak yok: bu surecte bir kez tazele
    # Disk ile store arasinda bosluk var mi? (restart once uzun kapali kaldiysa)
    return hist.index[-1] < live.index[0] - pd.Timedelta(seconds=_TF_SEC[key[1]])


# ──────────────────── dis arayuz ────────────────────

def bar_close_utc(raw: str, tf: str, last_ts: pd.Timestamp) -> int | None:
    """Son mumun kapanis ani (UTC epoch sn) — fiyat etiketinin altindaki geri sayim icin.

    Seans sembollerinde hafta Cuma 17:00 NY'de kapanir: son gunluk/haftalik kova (CME'de gun
    18:00 NY'de basladigi icin +24s Cuma 18:00'i gosterirdi) o ana kirpilir.
    """
    try:
        close = last_ts + pd.Timedelta(seconds=_TF_SEC[tf])
        if is_session_symbol(raw):
            prev = fx_session.last_week_close(last_ts)
            if prev is not None:
                nxt = (prev.tz_convert(fx_session.NY).tz_localize(None) + pd.Timedelta(days=7)
                       ).tz_localize(fx_session.NY).tz_convert("UTC")
                close = min(close, nxt)
        return int(close.timestamp())
    except Exception:
        return None


def price_precision(price: float | None) -> int:
    if not price or price <= 0:
        return 2
    return int(max(2, min(10, 5 - math.floor(math.log10(price)))))


async def get_candles(store, display_symbol: str, tf: str, bootstrap_done: bool,
                      since: int | None = None) -> dict:
    """Grafik verisi: gecmis (disk/REST) + canli (store). `since` = TSİ kaydirmali sn."""
    raw, _market = from_display_symbol(display_symbol)
    base_tf = "1d" if tf == "1w" else tf
    key = (raw, base_tf)
    _load_disk(key)

    live = _live(store, raw, tf)
    live_base = live if tf != "1w" else _live(store, raw, "1d")

    history = "store"                      # REST'e gerek yok / izin yok
    if _rest_allowed(raw, base_tf):
        if key in _FAILED:
            history = "failed"
        elif _needs_fetch(key, live_base):
            if not bootstrap_done:
                history = "pending"
            else:
                task = _INFLIGHT.get(key)
                if task is None:
                    task = asyncio.create_task(_fetch(key))
                    _INFLIGHT[key] = task
                    task.add_done_callback(lambda _t, k=key: _INFLIGHT.pop(k, None))
                try:
                    await asyncio.shield(task)
                except Exception as e:
                    log.warning("CHART gecmis gorevi hata %s %s: %s", raw, base_tf, e)
                history = "failed" if key in _FAILED else "ready"
        else:
            history = "ready"

    hist = _HIST.get(key, _empty())
    if tf == "1w":
        df = _weekly(_merge(hist, live_base))
    else:
        df = _merge(hist, live)
        # Canli seri diskteki gecmisi uzatiyor: bir sonraki restart'ta bosluk kalmasin.
        if key in _HIST and not live.empty and df.index[-1] > hist.index[-1]:
            if time.monotonic() - _LAST_PERSIST.get(key, 0.0) > _PERSIST_EVERY_SEC:
                _HIST[key] = df
                _save_disk(key, df)

    candles = [
        {"time": int(ts.timestamp()) + TSI_OFFSET_SEC,
         "open": r[0], "high": r[1], "low": r[2], "close": r[3]}
        for ts, r in zip(df.index, df[_COLS].itertuples(index=False, name=None))
    ]
    if since is not None:
        candles = [c for c in candles if c["time"] >= since]
    last = float(df["close"].iloc[-1]) if not df.empty else None
    bar_close = bar_close_utc(raw, tf, df.index[-1]) if not df.empty else None
    return {
        "symbol": display_symbol,
        "tf": tf,
        "history": history,
        "precision": price_precision(last),
        "bar_sec": _TF_SEC[tf],
        "bar_close": bar_close,            # UTC epoch sn; gecmisse mum kapali (piyasa kapali)
        "candles": candles,
    }


def ts_tsi(dt: datetime | None) -> int | None:
    """DB zamani (naive = UTC) -> grafigin TSİ kaydirmali saniyesi."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) + TSI_OFFSET_SEC
