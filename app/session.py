"""FX/metal/endeks/petrol seans takvimi ve NY-hizali mum sentezi.

BingX'in NC* sentetik kontratlari kripto gibi 7/24 islem gorur ve mumlari UTC
gece yarisina hizalidir. Gercek piyasa ise seansli ve broker/TradingView mumlari
seans acilisina hizalidir. Aradaki fark CRT'yi dogrudan bozar: C1/C2 bambaska
araliklar olur ve SL (`purge_extreme`) yanlis yere duser (XAU'da olculen fark
8-13 puan).

**Iki ayri seans takvimi var** (18.09.2026, ham 1H verisiyle olculdu; olcum
betikleri `tmp/tmp_session_anchor_probe*.py`):

- **Spot FX** (18 parite): gun donumu 17:00 NY. BingX bu paritelerde hafta sonu
  hic mum uretmiyor; son Cuma bari 20:00 UTC (21:00'de kapanir = 17:00 NY) ve
  hafta ici 21:00 UTC'de aktivite cukuru yok - interbank gunu kesintisiz.
- **CME urunleri** (XAUUSD, XAGUSD, US100, US500, OILWTI, OILBRENT): gun donumu
  **18:00 NY = 17:00 Chicago**. Pazar 22:00 UTC'de aktivite tabanin %110-184'une
  sicriyor (21:00'de %45-96) ve hafta ici her gun 21:00-22:00 UTC gunun en olu
  saati - CME'nin gunluk bakim arasi. TSI'de kovalar 01-05-09-13-17-21.

Cuma kapanisi ikisinde de ayni an: CME Cuma 16:00 Chicago = 17:00 NY = 21:00
UTC'de kapanir. Bu yuzden `is_week_close_bar` ve olu seansin Cuma dali sembole
bagli degildir; degisen yalnizca **kova anchor'i** ve olu seansin **Pazar** dali.

CME'nin gunluk 21:00-22:00 UTC arasi olu sayilmaz: BingX'in kontrati o saatte de
gercek fiyat basiyor (tabanin %37-54'u) ve 18:00 NY anchor'i o bari zaten biten
gunun son kovasina yaziyor.

Bu modul iki isi yapar:

1. **Olu seans elemesi** - Cuma 17:00 NY -> Pazar (17:00 | 18:00) NY
   arasindaki barlari atar. BingX o pencerede de fiyat uretir ama hacim yoktur: XAU 1D serisinin
   %28'i bu sahte barlardan olusuyordu ve 17'sinin 8'i onceki gunun H/L'sini
   asarak purge/swing tespitini tetikliyordu.
2. **Seans-hizali sentez** - 1H barlarini sembolun anchor'ina (17:00 ya da
   18:00 NY) hizali 4H ve 1D mumlarina cevirir.

Sentez kayipsizdir: NY ofseti tam saat oldugu icin (UTC-4/-5) 1H sinirlari
4H/1D sinirlarina tam oturur, hicbir mum ikiye bolunmez. Mevcut UTC anchor'inda
sentezleyip BingX'in kendi 4H'i ile karsilastirdik: 198/198 mum birebir esit
(09.09.2026).

Kolaylik: ABD yaz saati gecisleri Pazar 02:00 NY'de, yani **olu seansin
icinde** olur. Dolayisiyla bir islem haftasi boyunca NY ofseti hic degismez ve
hafta ici DST belirsizligi diye bir sorun yoktur.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")

# Seansin gun donumu (NY yerel saati): kova sinirini ve Pazar acilisini belirler.
SESSION_HOUR = 17       # spot FX: 17:00 NY (interbank konvansiyonu)
CME_SESSION_HOUR = 18   # CME urunleri: 18:00 NY = 17:00 Chicago

# CME'de listelenen sembollerin market'leri (bkz. app/exchange.py SYMBOLS_BY_MARKET).
CME_MARKETS = ("metal", "index", "oil")


def session_hour(symbol: str | None = None) -> int:
    """Sembolun gun donumu saati (NY). Sembol verilmezse FX varsayilani.

    `symbol` ham BingX sembolu ("NCCOGOLD2USD-USDT"). Bilinmeyen sembol FX
    sayilir - eski davranis.
    """
    if not symbol:
        return SESSION_HOUR
    try:
        from app.exchange import market_of
        return CME_SESSION_HOUR if market_of(symbol) in CME_MARKETS else SESSION_HOUR
    except Exception:
        return SESSION_HOUR

_OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


# ──────────────────── Olu seans ────────────────────


def _is_dead_ny(ts_ny: pd.Timestamp, open_hour: int = SESSION_HOUR) -> bool:
    """NY yerel saatiyle verilen bar basi olu seansa mi dusuyor?

    Cuma kapanisi her sembolde 17:00 NY (CME de 16:00 Chicago'da kapanir);
    degisen yalnizca Pazar acilisi.
    """
    dow = ts_ny.dayofweek  # Pzt=0 ... Paz=6
    if dow == 4 and ts_ny.hour >= SESSION_HOUR:   # Cuma 17:00 ve sonrasi
        return True
    if dow == 5:                                   # Cumartesi tam gun
        return True
    if dow == 6 and ts_ny.hour < open_hour:        # Pazar acilis oncesi
        return True
    return False


def is_dead_session(ts, symbol: str | None = None) -> bool:
    """UTC zaman damgasi olu seansa mi dusuyor? (tekil kontrol)"""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return _is_dead_ny(t.tz_convert(NY), session_hour(symbol))


def drop_dead_session(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Olu seansa dusen barlari eler. Index UTC olmali."""
    if df is None or df.empty:
        return df
    open_hour = session_hour(symbol)
    idx = _as_utc_index(df.index)
    ny = idx.tz_convert(NY)
    dow = ny.dayofweek
    hour = ny.hour
    dead = (
        ((dow == 4) & (hour >= SESSION_HOUR))
        | (dow == 5)
        | ((dow == 6) & (hour < open_hour))
    )
    out = df.loc[~dead]
    return out


def is_week_close_bar(ts) -> bool:
    """Bu 1H bari haftanin son canli bari mi? (Cuma 16:00-17:00 NY)

    Cuma kapanisi (kripto-disi acik islemleri duzlestirme) bu bar kapaninca
    tetiklenir; ayni an haftanin son 4H kovasinin da kapanisidir.

    Sembole bagli degil: CME urunleri de Cuma 16:00 Chicago = 17:00 NY'de
    kapanir, yani FX ile ayni an.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ny = t.tz_convert(NY)
    return ny.dayofweek == 4 and ny.hour == SESSION_HOUR - 1


def last_week_close(ts) -> pd.Timestamp | None:
    """`ts` anindan onceki (dahil) en son Cuma 17:00 NY kapanisi, UTC olarak.

    FX haftasi bu anda kapanir: acik islemler duzlestirilir, dolmamis setuplar
    silinir (bkz. scanner.close_session_positions). Sembole bagli degil, CME de
    ayni anda kapanir (bkz. is_week_close_bar).
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ny = t.tz_convert(NY)
    # Bu takvim haftasinin Cuma'si; saat NY yerelinde kurulur (DST gecisi Pazar
    # 02:00'de oldugu icin Cuma 17:00 her zaman tek ve gecerli bir andir).
    friday = (ny.normalize().tz_localize(None)
              - pd.Timedelta(days=int(ny.dayofweek))
              + pd.Timedelta(days=4))
    close = (friday + pd.Timedelta(hours=SESSION_HOUR)).tz_localize(NY)
    if close > t:                       # hafta henuz kapanmadi -> onceki hafta
        close = (friday - pd.Timedelta(days=7)
                 + pd.Timedelta(hours=SESSION_HOUR)).tz_localize(NY)
    return close.tz_convert("UTC")


# ──────────────────── NY-hizali sentez ────────────────────


def _as_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def bucket_start(ts, timeframe: str, symbol: str | None = None) -> pd.Timestamp:
    """Verilen anin ait oldugu seans-hizali kovanin baslangici (UTC)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    open_hour = session_hour(symbol)
    shifted = t.tz_convert(NY) - pd.Timedelta(hours=open_hour)
    if timeframe == "1d":
        anchor = shifted.normalize()
    else:
        anchor = shifted.floor(timeframe)
    return (anchor + pd.Timedelta(hours=open_hour)).tz_convert("UTC")


def resample_from_1h(df1h: pd.DataFrame, timeframe: str,
                     symbol: str | None = None) -> pd.DataFrame:
    """1H barlarindan seans-hizali 4H veya 1D serisi uret.

    Olu seans barlari onceden elenir. Index UTC kalir; yalnizca kova sinirlari
    sembolun anchor'ina (17:00 ya da 18:00 NY) hizalanir.
    """
    if df1h is None or df1h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    open_hour = session_hour(symbol)
    work = df1h.copy()
    work.index = _as_utc_index(work.index)
    work = drop_dead_session(work, symbol)
    if work.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # NY'ye cevir, anchor saatini gun basi yapacak sekilde kaydir, kovala.
    shifted = work.index.tz_convert(NY) - pd.Timedelta(hours=open_hour)
    if timeframe == "1d":
        anchors = shifted.normalize()
    else:
        anchors = shifted.floor(timeframe)

    work["_bucket"] = (anchors + pd.Timedelta(hours=open_hour)).tz_convert("UTC")
    out = work.groupby("_bucket", sort=True).agg(_OHLC_AGG)
    out.index.name = None
    return out
