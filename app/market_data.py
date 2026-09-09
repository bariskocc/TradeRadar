"""BingX WebSocket veri katmani.

- Baslangicta REST ile gecmis mumlari doldurur (bootstrap).
- Kalici WebSocket ile 15m/4h (tum aktif), 1h (1D-1H evreni) ve 5m
  (1H-5M: XAU/EUR/US100/BTC + SMT esleri) kline akisini dinler. Tek baglanti.
- Mum kapanisi (T degisimi) ve forming mum guncellemelerini scanner olaylarina baglar.
- 1D forming bar, 1H mumlarindan birlestirilir (ayri kline_1d aboneligi yok).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import websockets

from app.config import (
    BINGX_REST_BASE,
    BINGX_WS_URL,
    BOOTSTRAP_LIMITS,
    MAINTENANCE_INTERVAL_SEC,
    SESSION_1H_BARS,
)
from app.database import async_session
from app.event_log import record_event
from app.exchange import (
    fetch_ohlcv,
    fetch_ohlcv_deep,
    get_active_symbols_flat,
    get_d1h_symbols_flat,
    get_h1_5m_symbols,
    is_session_symbol,
    to_display_symbol,
)
from app import scanner
from app import session as fx_session

log = logging.getLogger(__name__)

# Frame basina saklanacak azami mum. Seans sembollerinin 4H/1D serileri 1H'ten
# sentezlendigi icin 1H derin tutulur (60 islem gunu ~ 2016 takvim saati).
_MAX_ROWS_DEFAULT = 500
_MAX_ROWS = {"1h": 2400}
_WS_CORE_TFS = ["15m", "4h"]
_WS_D1H_TFS = ["1h"]
_WS_H1_TFS = ["5m"]
_TF_SUFFIX = {
    "5m": "kline_5m",
    "15m": "kline_15m",
    "4h": "kline_4h",
    "1h": "kline_1h",
}
_SUFFIX_TF = {v: k for k, v in _TF_SUFFIX.items()}


class MarketDataStore:
    """Symbol x timeframe rolling OHLCV deposu (bellek ici)."""

    def __init__(self) -> None:
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    @staticmethod
    def _cap(timeframe: str) -> int:
        return _MAX_ROWS.get(timeframe, _MAX_ROWS_DEFAULT)

    def set_df(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self._frames[(symbol, timeframe)] = df.tail(self._cap(timeframe)).copy()

    def get_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        df = self._frames.get((symbol, timeframe))
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return df

    def upsert_candle(self, symbol: str, timeframe: str, ts_ms: int, candle: dict) -> None:
        key = (symbol, timeframe)
        df = self._frames.get(key)
        idx = pd.to_datetime(int(ts_ms), unit="ms", utc=True)
        row = {
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle.get("volume", 0.0)),
        }
        if df is None or df.empty:
            df = pd.DataFrame([row], index=[idx])
        else:
            df.loc[idx] = row
            cap = self._cap(timeframe)
            if len(df) > cap:
                df = df.tail(cap)
        df.sort_index(inplace=True)
        self._frames[key] = df


class BingXMarketData:
    def __init__(self) -> None:
        self.store = MarketDataStore()
        self._symbols: list[str] = []
        self._d1h_symbols: list[str] = []
        self._h1_5m_symbols: list[str] = []
        self._last_ts: dict[tuple[str, str], int] = {}
        self._pending_price: dict[tuple[str, str], dict] = {}
        self._closed_queue: asyncio.Queue = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._connected = False
        self._last_message_at: datetime | None = None
        self._last_1d_day = None
        # Cuma kapanisi (kripto-disi pozisyonlari duzlestirme): haftada bir kez.
        self._week_close_pending = False
        self._week_close_marker: str | None = None

    # ──────────────── Bootstrap ────────────────

    async def bootstrap(self) -> None:
        self._symbols = get_active_symbols_flat()
        self._d1h_symbols = get_d1h_symbols_flat()
        self._h1_5m_symbols = get_h1_5m_symbols()
        d1h_set = set(self._d1h_symbols)
        h1_5m_set = set(self._h1_5m_symbols)
        log.info(
            "Bootstrap: %d symbol (4H) + %d symbol (1D-1H 1h) + %d symbol (1H-5M 5m)...",
            len(self._symbols), len(self._d1h_symbols), len(self._h1_5m_symbols),
        )
        async with httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0) as client:
            for sym in self._symbols:
                if is_session_symbol(sym):
                    # FX/metal/endeks/petrol: 4H ve 1D BingX'ten cekilmez,
                    # 1H'ten NY-hizali sentezlenir (bkz. app/session.py).
                    await self._bootstrap_session_symbol(sym, client)
                    continue
                for tf in _WS_CORE_TFS:
                    df = await fetch_ohlcv(sym, tf, limit=BOOTSTRAP_LIMITS[tf], client=client)
                    if not df.empty:
                        self.store.set_df(sym, tf, df)
                        last_ms = int(df.index[-1].value // 1_000_000)
                        self._last_ts[(sym, tf)] = last_ms
                # 1D (HTF bias + 1D CRT): WS ile akmiyor; store'da tutulur, gunluk yenilenir.
                try:
                    df1d = await fetch_ohlcv(sym, "1d", limit=BOOTSTRAP_LIMITS["1d"], client=client)
                    if not df1d.empty:
                        self.store.set_df(sym, "1d", df1d)
                except Exception as e:
                    log.warning("1D bootstrap basarisiz %s: %s", sym, e)
                if sym in d1h_set:
                    try:
                        df1h = await fetch_ohlcv(sym, "1h", limit=BOOTSTRAP_LIMITS["1h"], client=client)
                        if not df1h.empty:
                            self.store.set_df(sym, "1h", df1h)
                            last_ms = int(df1h.index[-1].value // 1_000_000)
                            self._last_ts[(sym, "1h")] = last_ms
                            self._rebuild_forming_1d(sym)
                    except Exception as e:
                        log.warning("1H bootstrap basarisiz %s: %s", sym, e)
            # 4H evreninde olmayan 1D-1H pariteleri (FX/XAG/oil/index)
            fourh_set = set(self._symbols)
            for sym in self._d1h_symbols:
                if sym not in fourh_set:
                    if is_session_symbol(sym):
                        await self._bootstrap_session_symbol(sym, client, with_ltf=False)
                        continue
                    try:
                        df1d = await fetch_ohlcv(sym, "1d", limit=BOOTSTRAP_LIMITS["1d"], client=client)
                        if not df1d.empty:
                            self.store.set_df(sym, "1d", df1d)
                    except Exception as e:
                        log.warning("1D bootstrap basarisiz %s: %s", sym, e)
                    try:
                        df1h = await fetch_ohlcv(sym, "1h", limit=BOOTSTRAP_LIMITS["1h"], client=client)
                        if not df1h.empty:
                            self.store.set_df(sym, "1h", df1h)
                            last_ms = int(df1h.index[-1].value // 1_000_000)
                            self._last_ts[(sym, "1h")] = last_ms
                            self._rebuild_forming_1d(sym)
                    except Exception as e:
                        log.warning("1H bootstrap basarisiz %s: %s", sym, e)
            for sym in self._h1_5m_symbols:
                if sym in h1_5m_set:
                    try:
                        df5m = await fetch_ohlcv(sym, "5m", limit=BOOTSTRAP_LIMITS["5m"], client=client)
                        if not df5m.empty:
                            last_ms = int(df5m.index[-1].value // 1_000_000)
                            if is_session_symbol(sym):
                                df5m = fx_session.drop_dead_session(df5m)
                            self.store.set_df(sym, "5m", df5m)
                            self._last_ts[(sym, "5m")] = last_ms
                    except Exception as e:
                        log.warning("5M bootstrap basarisiz %s: %s", sym, e)
        self._last_1d_day = datetime.now(timezone.utc).date()
        log.info("Bootstrap tamamlandi.")

    async def _bootstrap_session_symbol(self, sym: str, client, with_ltf: bool = True) -> None:
        """FX/metal/endeks/petrol bootstrap'i.

        4H ve 1D BingX'ten CEKILMEZ: BingX'in mumlari UTC gece yarisina hizali,
        gercek FX mumu ise 17:00 NY'ye. Ikisi de derin 1H serisinden NY-hizali
        sentezlenir ve olu seans barlari elenir.
        """
        if with_ltf:
            try:
                df15 = await fetch_ohlcv(sym, "15m", limit=BOOTSTRAP_LIMITS["15m"], client=client)
                if not df15.empty:
                    # _last_ts WS surekliligini takip eder: filtreden ONCEKI son bar.
                    self._last_ts[(sym, "15m")] = int(df15.index[-1].value // 1_000_000)
                    self.store.set_df(sym, "15m", fx_session.drop_dead_session(df15))
            except Exception as e:
                log.warning("15M bootstrap basarisiz %s: %s", sym, e)
        try:
            df1h = await fetch_ohlcv_deep(sym, "1h", bars=SESSION_1H_BARS, client=client)
            if df1h.empty:
                log.warning("1H derin bootstrap bos dondu: %s", sym)
                return
            self._last_ts[(sym, "1h")] = int(df1h.index[-1].value // 1_000_000)
            self.store.set_df(sym, "1h", df1h)
            self._rebuild_session_htf(sym, full=True)
        except Exception as e:
            log.warning("Seans bootstrap basarisiz %s: %s", sym, e)

    def _rebuild_session_htf(self, symbol: str, full: bool = False) -> None:
        """Seans sembolunun 4H/1D serilerini 1H'ten NY-hizali uret.

        `full=True` (bootstrap) tum seriyi bastan kurar. Aksi halde yalnizca
        icinde bulunulan forming kova yeniden hesaplanir: bu metot 1H akisinda
        her mesajda cagriliyor, tum seriyi resample etmek pahali olurdu.
        """
        df1h = self.store.get_df(symbol, "1h")
        if df1h is None or df1h.empty:
            return

        if full:
            for tf in ("4h", "1d"):
                out = fx_session.resample_from_1h(df1h, tf)
                if not out.empty:
                    self.store.set_df(symbol, tf, out)
            return

        last_ts = df1h.index[-1]
        if fx_session.is_dead_session(last_ts):
            return  # piyasa kapali: HTF mumu ilerlemez
        for tf in ("4h", "1d"):
            start = fx_session.bucket_start(last_ts, tf)
            bars = fx_session.drop_dead_session(df1h[df1h.index >= start])
            if bars.empty:
                continue
            row = {
                "open": float(bars.iloc[0]["open"]),
                "high": float(bars["high"].max()),
                "low": float(bars["low"].min()),
                "close": float(bars.iloc[-1]["close"]),
                "volume": float(bars["volume"].sum()),
            }
            self.store.upsert_candle(symbol, tf, int(start.value // 1_000_000), row)

    def _rebuild_forming_1d(self, symbol: str) -> None:
        """Ayni UTC gununun 1H mumlarindan forming 1D bari guncelle.

        Seans sembollerinde bunun yerine NY-hizali 4H+1D sentezi calisir.
        """
        if is_session_symbol(symbol):
            self._rebuild_session_htf(symbol)
            return
        df1h = self.store.get_df(symbol, "1h")
        if df1h is None or df1h.empty:
            return
        df_work = df1h.copy()
        idx = df_work.index
        if getattr(idx, "tz", None) is None:
            df_work.index = idx.tz_localize("UTC")
        else:
            df_work.index = idx.tz_convert("UTC")
        day_start = df_work.index[-1].normalize()
        day_bars = df_work[df_work.index.normalize() == day_start]
        if day_bars.empty:
            return
        row = {
            "open": float(day_bars.iloc[0]["open"]),
            "high": float(day_bars["high"].max()),
            "low": float(day_bars["low"].min()),
            "close": float(day_bars.iloc[-1]["close"]),
            "volume": float(day_bars["volume"].sum()),
        }
        day_ms = int(day_start.value // 1_000_000)
        self.store.upsert_candle(symbol, "1d", day_ms, row)

    async def _refresh_1d(self) -> None:
        """Gunluk (1D) veriyi yeniden cek (gunde bir kez).

        Seans sembolleri haric: onlarin 1D serisi 1H akisindan sentezleniyor,
        yani kendi kendini uzatiyor. Onlar icin REST yalnizca WS kacagini
        onarmak uzere 1H serisini tazeler.
        """
        all_syms = list(dict.fromkeys([*self._symbols, *self._d1h_symbols]))
        rest_syms = [s for s in all_syms if not is_session_symbol(s)]
        sess_syms = [s for s in all_syms if is_session_symbol(s)]
        async with httpx.AsyncClient(base_url=BINGX_REST_BASE, timeout=15.0) as client:
            for sym in rest_syms:
                try:
                    df1d = await fetch_ohlcv(sym, "1d", limit=BOOTSTRAP_LIMITS["1d"], client=client)
                    if not df1d.empty:
                        self.store.set_df(sym, "1d", df1d)
                except Exception as e:
                    log.warning("1D yenileme basarisiz %s: %s", sym, e)
            for sym in sess_syms:
                try:
                    df1h = await fetch_ohlcv_deep(sym, "1h", bars=SESSION_1H_BARS, client=client)
                    if not df1h.empty:
                        self.store.set_df(sym, "1h", df1h)
                        self._rebuild_session_htf(sym, full=True)
                except Exception as e:
                    log.warning("Seans 1H yenileme basarisiz %s: %s", sym, e)
        self._last_1d_day = datetime.now(timezone.utc).date()
        log.info("1D veri yenilendi (%d REST + %d seans sentezi).",
                 len(rest_syms), len(sess_syms))

    # ──────────────── WS yasam dongusu ────────────────

    async def start(self) -> None:
        self._running = True
        await self.bootstrap()
        self._tasks = [
            asyncio.create_task(self._ws_loop(), name="bingx-ws"),
            asyncio.create_task(self._processor_loop(), name="bingx-processor"),
        ]
        if MAINTENANCE_INTERVAL_SEC > 0:
            self._tasks.append(asyncio.create_task(self._maintenance_loop(), name="bingx-maintenance"))

    async def stop(self) -> None:
        self._running = False
        if self._connected:
            await record_event(
                "ws_disconnect",
                "Uygulama kapaniyor, BingX WS baglantisi kapatildi.",
                level="info",
            )
        self._connected = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []

    async def _ws_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(BINGX_WS_URL, max_size=None, ping_interval=None) as ws:
                    self._connected = True
                    await self._subscribe_all(ws)
                    log.info(
                        "BingX WS baglandi, %d sembol (4H) + %d (1h) + %d (5m).",
                        len(self._symbols), len(self._d1h_symbols), len(self._h1_5m_symbols),
                    )
                    await record_event(
                        "ws_connect",
                        (
                            f"BingX WS baglandi - {len(self._symbols)} sembol / "
                            f"{self._subscription_count()} abonelik "
                            f"(+{len(self._d1h_symbols)} x 1h, +{len(self._h1_5m_symbols)} x 5m)"
                        ),
                        level="success",
                    )
                    # Kopukluk sirasinda kacan TP/SL/BE icin hemen mutabakat.
                    try:
                        async with async_session() as session:
                            await scanner.reconcile_open_signals(session, store=self.store)
                    except Exception:
                        log.exception("post-reconnect reconcile failed")
                    while self._running:
                        raw = await ws.recv()
                        await self._handle_raw(ws, raw)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                log.warning("BingX WS koptu (%s). 5 sn sonra yeniden baglanilacak.", e)
                if self._running:
                    await record_event(
                        "ws_disconnect",
                        f"BingX WS koptu: {e}. 5 sn sonra yeniden baglanilacak.",
                        level="warning",
                    )
                await asyncio.sleep(5)
        self._connected = False

    def _core_tfs_for(self, symbol: str) -> list[str]:
        """Seans sembollerinde 4H aboneligi yok - seri 1H'ten sentezleniyor."""
        if is_session_symbol(symbol):
            return [tf for tf in _WS_CORE_TFS if tf != "4h"]
        return _WS_CORE_TFS

    def _subscription_count(self) -> int:
        return (
            sum(len(self._core_tfs_for(s)) for s in self._symbols)
            + len(self._d1h_symbols) * len(_WS_D1H_TFS)
            + len(self._h1_5m_symbols) * len(_WS_H1_TFS)
        )

    async def _subscribe_all(self, ws) -> None:
        for sym in self._symbols:
            for tf in self._core_tfs_for(sym):
                msg = {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{sym}@{_TF_SUFFIX[tf]}"}
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.01)
        for sym in self._d1h_symbols:
            for tf in _WS_D1H_TFS:
                msg = {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{sym}@{_TF_SUFFIX[tf]}"}
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.01)
        for sym in self._h1_5m_symbols:
            for tf in _WS_H1_TFS:
                msg = {"id": str(uuid.uuid4()), "reqType": "sub", "dataType": f"{sym}@{_TF_SUFFIX[tf]}"}
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.01)

    async def _handle_raw(self, ws, raw) -> None:
        if isinstance(raw, (bytes, bytearray)):
            try:
                text = gzip.decompress(raw).decode("utf-8")
            except Exception:
                return
        else:
            text = raw

        if text == "Ping":
            await ws.send("Pong")
            return

        self._last_message_at = datetime.now(timezone.utc)
        try:
            payload = json.loads(text)
        except Exception:
            return

        data_type = payload.get("dataType") or ""
        data = payload.get("data")
        if not data_type or not data:
            return

        try:
            symbol, suffix = data_type.split("@", 1)
        except ValueError:
            return
        tf = _SUFFIX_TF.get(suffix)
        if tf is None:
            return

        item = data[0] if isinstance(data, list) else data
        candle = {
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"],
            "volume": item.get("v", 0.0),
        }
        ts_ms = int(item["T"])

        session_sym = is_session_symbol(symbol)
        cur_dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
        # Olu seansta BingX fiyat uretmeye devam eder ama piyasa kapalidir:
        # ne seriye yazilir ne de TP/SL takibine beslenir.
        bar_dead = session_sym and fx_session.is_dead_session(cur_dt)

        if not bar_dead:
            self.store.upsert_candle(symbol, tf, ts_ms, candle)
            if tf == "1h":
                self._rebuild_forming_1d(symbol)

        key = (symbol, tf)
        prev_ts = self._last_ts.get(key)
        if prev_ts is not None and ts_ms > prev_ts:
            # Onceki mum kapandi.
            prev_dt = pd.to_datetime(prev_ts, unit="ms", utc=True)
            prev_dead = session_sym and fx_session.is_dead_session(prev_dt)
            if not prev_dead:
                await self._closed_queue.put((symbol, tf))
            if tf == "1h" and session_sym:
                if not prev_dead:
                    # Sentezlenmis HTF: kapanan 1H bari bir kovayi bitirdi mi?
                    for htf in ("4h", "1d"):
                        if fx_session.bucket_start(prev_dt, htf) != fx_session.bucket_start(cur_dt, htf):
                            await self._closed_queue.put((symbol, htf))
                    if fx_session.is_week_close_bar(prev_dt):
                        marker = prev_dt.strftime("%G-W%V")
                        if self._week_close_marker != marker:
                            self._week_close_marker = marker
                            self._week_close_pending = True
            elif tf == "1h":
                prev_day = datetime.fromtimestamp(prev_ts / 1000, tz=timezone.utc).date()
                new_day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
                if new_day > prev_day:
                    await self._closed_queue.put((symbol, "1d"))
        if prev_ts is None or ts_ms >= prev_ts:
            self._last_ts[key] = ts_ms

        # Intrabar fill/TP-SL: 15m -> 4H, 1h -> 1D, 5m -> 1H.
        if tf in ("15m", "1h", "5m") and not bar_dead:
            self._pending_price[(symbol, tf)] = {**candle, "ts_ms": ts_ms, "ltf": tf}

    async def _processor_loop(self) -> None:
        while self._running:
            try:
                # Once kapanan mumlar (setup tespiti).
                while not self._closed_queue.empty():
                    symbol, tf = self._closed_queue.get_nowait()
                    await scanner.on_candle_closed(symbol, tf, self.store)
                    await asyncio.sleep(0)

                # Cuma 17:00 NY: kripto-disi acik islemleri kapat, bekleyenleri
                # iptal et. Pozisyon hafta sonuna sarkmasin (Pazar acilisindaki
                # gap SL'yi asabilir).
                if self._week_close_pending:
                    self._week_close_pending = False
                    try:
                        async with async_session() as db:
                            await scanner.close_session_positions(db, self.store)
                    except Exception:
                        log.exception("Cuma kapanisi basarisiz")

                # Sonra forming fiyat guncellemeleri (coalesced).
                if self._pending_price:
                    pending = self._pending_price
                    self._pending_price = {}
                    for (symbol, ltf), candle in pending.items():
                        await scanner.on_price_update(symbol, candle, self.store, ltf=ltf)
                        await asyncio.sleep(0)

                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("processor loop error")
                await asyncio.sleep(1)

    async def _maintenance_loop(self) -> None:
        """Bakim dongusu.

        - Startup'tan ~10 sn sonra tek seferlik tarama (radar/gecmis setuplari).
        - Aktif sembol seti degisirse (haftaici<->haftasonu) yeniden bootstrap.
        - Gun degisince 1D veriyi yenile.
        - Her dongude acik sinyaller icin REST TP/SL/BE mutabakati (WS kacagina
          karsi guvenlik agi; metal/fx gibi seansli piyasalar icin kritik).
        Yeni setup tespiti WS olaylariyla (15m/4h kapanis) calisir.
        """
        warmup = True
        while self._running:
            try:
                await asyncio.sleep(10 if warmup else MAINTENANCE_INTERVAL_SEC)

                # Aktif sembol seti degistiyse yeniden bootstrap + abonelik.
                current = get_active_symbols_flat()
                current_d1h = get_d1h_symbols_flat()
                current_h1 = get_h1_5m_symbols()
                if (
                    set(current) != set(self._symbols)
                    or set(current_d1h) != set(self._d1h_symbols)
                    or set(current_h1) != set(self._h1_5m_symbols)
                ):
                    log.info("Aktif sembol seti degisti; bootstrap + yeniden abonelik.")
                    self._symbols = current
                    self._d1h_symbols = current_d1h
                    self._h1_5m_symbols = current_h1
                    await self.bootstrap()
                else:
                    # Gun degistiyse yalnizca 1D veriyi yenile.
                    today = datetime.now(timezone.utc).date()
                    if self._last_1d_day is None or today != self._last_1d_day:
                        await self._refresh_1d()

                if warmup:
                    warmup = False
                    async with async_session() as session:
                        await scanner.run_scan(session, source="scheduler", store=self.store)
                else:
                    # WS kacagina karsi acik sinyal mutabakati.
                    async with async_session() as session:
                        await scanner.reconcile_open_signals(session, store=self.store)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("maintenance loop error")

    # ──────────────── Durum ────────────────

    def _subscription_breakdown(self) -> dict:
        """Abonelikler tek WS baglantisinda paylasilir; stratejiye bolunemez.

        `1h` akisi ayni anda uc ise yariyor: 1D-1H'in LTF'si, 1H-5M'in HTF'si ve
        seans sembollerinin 4H/1D sentezi. Tek bir stratejiye yazmak cift sayim
        olurdu; bu yuzden kirilim akis bazinda verilir.
        """
        crypto_4h = [s for s in self._symbols if not is_session_symbol(s)]
        session_4h = [s for s in self._symbols if is_session_symbol(s)]
        return {
            "15m": len(self._symbols),
            "4h": len(crypto_4h),
            "1h": len(self._d1h_symbols),
            "5m": len(self._h1_5m_symbols),
            "session_symbols": len(session_4h),
        }

    def status(self) -> dict:
        return {
            "running": self._running,
            "connected": self._connected,
            "symbol_count": len(self._symbols),
            "d1h_symbol_count": len(self._d1h_symbols),
            "h1_symbol_count": len(self._h1_5m_symbols),
            "subscription_count": self._subscription_count(),
            "subscription_breakdown": self._subscription_breakdown(),
            "last_message_at": (
                self._last_message_at.astimezone(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S UTC+3")
                if self._last_message_at else None
            ),
        }


# Global instance (main.py lifespan tarafindan yonetilir)
market_data = BingXMarketData()
