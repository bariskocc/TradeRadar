"""FX/metal/endeks/petrol seans takvimi ve NY-hizali mum sentezi.

BingX'in NC* sentetik kontratlari kripto gibi 7/24 islem gorur ve mumlari UTC
gece yarisina hizalidir. Gercek FX piyasasi ise Cuma 17:00 NY'de kapanip Pazar
17:00 NY'de acilir; broker/TradingView mumlari da bu ana hizalidir. Aradaki
fark CRT'yi dogrudan bozar: C1/C2 bambaska araliklar olur ve SL (`purge_extreme`)
yanlis yere duser (XAU'da olculen fark 8-13 puan).

Bu modul iki isi yapar:

1. **Olu seans elemesi** - Cuma 17:00 NY -> Pazar 17:00 NY arasindaki barlari
   atar. BingX o pencerede de fiyat uretir ama hacim yoktur: XAU 1D serisinin
   %28'i bu sahte barlardan olusuyordu ve 17'sinin 8'i onceki gunun H/L'sini
   asarak purge/swing tespitini tetikliyordu.
2. **NY-hizali sentez** - 1H barlarini 17:00 NY anchor'ina hizali 4H ve 1D
   mumlarina cevirir.

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

# Seansin gun donumu: 17:00 NY. Hem haftalik kapanis/acilis hem de gunluk
# mumun sinirini belirler (broker konvansiyonu).
SESSION_HOUR = 17

_OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


# ──────────────────── Olu seans ────────────────────


def _is_dead_ny(ts_ny: pd.Timestamp) -> bool:
    """NY yerel saatiyle verilen bar basi olu seansa mi dusuyor?"""
    dow = ts_ny.dayofweek  # Pzt=0 ... Paz=6
    if dow == 4 and ts_ny.hour >= SESSION_HOUR:   # Cuma 17:00 ve sonrasi
        return True
    if dow == 5:                                   # Cumartesi tam gun
        return True
    if dow == 6 and ts_ny.hour < SESSION_HOUR:     # Pazar 17:00 oncesi
        return True
    return False


def is_dead_session(ts) -> bool:
    """UTC zaman damgasi olu seansa mi dusuyor? (tekil kontrol)"""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return _is_dead_ny(t.tz_convert(NY))


def drop_dead_session(df: pd.DataFrame) -> pd.DataFrame:
    """Olu seansa dusen barlari eler. Index UTC olmali."""
    if df is None or df.empty:
        return df
    idx = _as_utc_index(df.index)
    ny = idx.tz_convert(NY)
    dow = ny.dayofweek
    hour = ny.hour
    dead = (
        ((dow == 4) & (hour >= SESSION_HOUR))
        | (dow == 5)
        | ((dow == 6) & (hour < SESSION_HOUR))
    )
    out = df.loc[~dead]
    return out


def is_week_close_bar(ts) -> bool:
    """Bu 1H bari haftanin son canli bari mi? (Cuma 16:00-17:00 NY)

    Cuma kapanisi (kripto-disi acik islemleri duzlestirme) bu bar kapaninca
    tetiklenir; ayni an haftanin son 4H kovasinin da kapanisidir.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ny = t.tz_convert(NY)
    return ny.dayofweek == 4 and ny.hour == SESSION_HOUR - 1


# ──────────────────── NY-hizali sentez ────────────────────


def _as_utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def bucket_start(ts, timeframe: str) -> pd.Timestamp:
    """Verilen anin ait oldugu NY-hizali kovanin baslangici (UTC)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    shifted = t.tz_convert(NY) - pd.Timedelta(hours=SESSION_HOUR)
    if timeframe == "1d":
        anchor = shifted.normalize()
    else:
        anchor = shifted.floor(timeframe)
    return (anchor + pd.Timedelta(hours=SESSION_HOUR)).tz_convert("UTC")


def resample_from_1h(df1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """1H barlarindan NY-hizali 4H veya 1D serisi uret.

    Olu seans barlari onceden elenir. Index UTC kalir; yalnizca kova sinirlari
    NY 17:00'a hizalanir.
    """
    if df1h is None or df1h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    work = df1h.copy()
    work.index = _as_utc_index(work.index)
    work = drop_dead_session(work)
    if work.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # NY'ye cevir, 17:00'i gun basi yapacak sekilde kaydir, kovala.
    shifted = work.index.tz_convert(NY) - pd.Timedelta(hours=SESSION_HOUR)
    if timeframe == "1d":
        anchors = shifted.normalize()
    else:
        anchors = shifted.floor(timeframe)

    work["_bucket"] = (anchors + pd.Timedelta(hours=SESSION_HOUR)).tz_convert("UTC")
    out = work.groupby("_bucket", sort=True).agg(_OHLC_AGG)
    out.index.name = None
    return out
