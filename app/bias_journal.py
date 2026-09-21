"""1D bias tahmin karnesi: yon dogru muydu? (olcum; motor davranisini DEGISTIRMEZ)

Soru: `compute_daily_bias` yonu dogru tahmin ediyor mu? Motorda iki yerde belirleyici -- hard
filtre (LONG'a BULLISH/NEUTRAL, SHORT'a BEARISH/NEUTRAL) ve skorda +-2. Yanlissa hem kazananlari
eliyor hem yanlis setup'i one cikariyor.

ONEMLI -- yanlis olcum: "islem bias yonunde gitti mi". Hard filtre zaten yalnizca hizali setuplari
gecirdigi icin kontrol grubu yok (dongusel orneklem); ustelik TP/SL'ye gidisi entry/stop yerlesimi,
RR ve kismi kar belirliyor, bias'in payi ayiklanamaz. Dogrusu: bias'i ISLEMDEN BAGIMSIZ, hava
durumu tahmini gibi puanlamak -- her sembol her gun, ufuk sonunda fiyat ne yapti (ATR'ye bolunmus).

Yontem: seri tabanli, IDEMPOTENT yeniden hesap. Store'daki 1D serisinin her kapanmis bari icin
seri o bara kadar dilimlenir ve motorun kendi saf fonksiyonlari cagrilir (`htf_bias_with_age`,
`compute_ict_bias`, `compute_daily_bias`, `compute_weekly_bias`) -- yani olculen sey motorun o gun
soyleyecegi seyin ta kendisi. Bu yuzden GECMIS de ayni kod yoluyla dolar: replay degil, saf
fonksiyonun elde duran seriye uygulanmasi (emir simulasyonu ve BingX indirmesi YOK).

Ileriye donuk hareket (`fwd_1d/3d/5d`) ATR birimindedir ve ISARETLIDIR (pozitif = yukari). Isabet,
bias yonuyle bu isaretin uyusmasi. Referans cizgisi `momentum` ("dun ne yaptiysa bugun de onu
yapar") ayni satirda tutulur; bias bunu gecemiyorsa tahmin degeri yoktur.

Rapor: `python scripts/bias_stat.py`. Karar kurali: IZLEME.md "1D bias tahmin karnesi".
Bu modulun hicbir hatasi islem akisini bozmamali: disariya acilan her fonksiyon hatayi yutar/loglar.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.crt_engine import (
    STRUCTURE_STALE_DAYS,
    compute_atr,
    compute_daily_bias,
    compute_ict_bias,
    compute_weekly_bias,
    htf_bias_with_age,
)
from app.models import BiasJournal

log = logging.getLogger(__name__)

# Swing tespiti icin gereken en az kapali gun; bunun altindaki barlar icin satir yazilmaz
# (yoksa "veri yetmedi" NEUTRAL'leri gercek NEUTRAL sanilir).
WARMUP_BARS = 25
# Ileriye donuk ufuklar (kapali gun). TODO'daki "stratejinin tipik islem suresi" (1D'de medyan
# 12 saat) 1D serisinden olculemez -- gun alti ufuk bu tablonun disindadir, bilerek.
HORIZONS = (1, 3, 5)
ATR_PERIOD = 14
# Tahmin alanlari: bir kez yazilir, BIR DAHA DEGISMEZ. Gerekce: store 1D penceresi kayarken ayni
# gun daha az/cok gecmisle yeniden hesaplanabiliyor ve gecmise donuk "tahmin" degisiyordu -- karne
# boyle tutulmaz. Setup Journal'da seviyelerin dondurulmasiyla ayni ilke.
_PREDICTION_COLS = ("structure", "structure_age", "ict", "combined", "weekly", "stale_applied",
                    "momentum", "close", "atr")
# Sonuc alanlari: NULL iken dolar (ileri gunler geldikce).
_OUTCOME_COLS = tuple(f"fwd_{h}d" for h in (1, 3, 5))


def _label(prev_close: float, close: float) -> str:
    """Referans cizgisi: onceki gunun yonu (momentum)."""
    if close > prev_close:
        return "BULLISH"
    if close < prev_close:
        return "BEARISH"
    return "NEUTRAL"


def build_rows(symbol: str, market: str | None, df_1d: pd.DataFrame | None) -> list[dict]:
    """Serideki her kapanmis gun icin bir satir (saf hesap; DB'ye dokunmaz)."""
    if df_1d is None or getattr(df_1d, "empty", True) or len(df_1d) <= WARMUP_BARS + 1:
        return []
    df = df_1d.sort_index()
    # Olusan (kapanmamis) gunu dusur: son barin 24 saati dolmadiysa tahmin yarim mumla kurulur.
    now = pd.Timestamp.now(tz="UTC")
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    if last + pd.Timedelta(hours=24) > now:
        df = df.iloc[:-1]
    n = len(df)
    if n <= WARMUP_BARS + 1:
        return []
    try:
        atr_series = compute_atr(df, period=ATR_PERIOD)
    except Exception:
        return []
    closes = df["close"].astype(float).to_numpy()
    rows: list[dict] = []
    for i in range(WARMUP_BARS, n):
        sub = df.iloc[: i + 1]
        try:
            structure, age = htf_bias_with_age(sub)
        except Exception:
            structure, age = "NEUTRAL", None
        try:
            ict = compute_ict_bias(sub)
        except Exception:
            ict = "NEUTRAL"
        try:
            combined = compute_daily_bias(sub)
        except Exception:
            combined = "NEUTRAL"
        try:
            weekly = compute_weekly_bias(sub)
        except Exception:
            weekly = "NEUTRAL"
        # Karar ict'ye mi birakildi? (compute_daily_bias'in bayatlik dali)
        stale = structure == "NEUTRAL" or age is None or age > STRUCTURE_STALE_DAYS
        try:
            atr = float(atr_series.iloc[i])
        except Exception:
            atr = float("nan")
        if not atr or atr != atr:          # 0 ya da NaN -> normalize edilemez
            continue
        close = float(closes[i])
        row = {
            "day": sub.index[-1].to_pydatetime().replace(tzinfo=None),
            "symbol": symbol, "market_type": market,
            "structure": structure, "structure_age": age, "ict": ict,
            "combined": combined, "weekly": weekly, "stale_applied": bool(stale),
            "momentum": _label(float(closes[i - 1]), close),
            "close": close, "atr": atr,
        }
        for h in HORIZONS:
            row[f"fwd_{h}d"] = round((float(closes[i + h]) - close) / atr, 4) if i + h < n else None
        rows.append(row)
    return rows


def _fill_outcomes(existing: BiasJournal, row: dict) -> bool:
    """Yalniz bos sonuc alanlarini doldur; tahmin alanlarina DOKUNMA."""
    changed = False
    for c in _OUTCOME_COLS:
        if getattr(existing, c) is None and row.get(c) is not None:
            setattr(existing, c, row[c])
            changed = True
    return changed


def _needs_work(df, existing: dict) -> bool:
    """Bu sembol icin hesaplanacak yeni bir sey var mi? (bos yere 35 dilim hesaplamamak icin)

    Her restart'ta tum seri yeniden hesaplaniyordu: 71 sembol x ~35 dilim, bir cekirdegi
    dakikalarca dolduruyor ve WS isleyicisiyle yarisiyordu. Iki kosuldan biri varsa hesapla:
      - serinin son KAPANMIS gunu icin satirimiz yok       -> yeni gun gelmis
      - DOLDURULABILIR NULL fwd var: satirdan sonra en az `max(HORIZONS)` kapali bar birikmis

    ⚠️ "NULL fwd_5d var mi" demek YETMEZ: en yeni 5 gunun fwd_5d'si tanim geregi hep NULL'dur,
    o kosul her zaman dogru cikar ve atlama hic tetiklenmezdi.
    """
    try:
        closed = _closed_days(df)
        if not closed:
            return False
        if closed[-1] not in existing:
            return True                       # yeni kapanmis gun
        need = max(HORIZONS)
        for r in existing.values():
            if r.fwd_5d is None and sum(1 for d in closed if d > r.day) >= need:
                return True                   # ileri hareket artik hesaplanabilir
        return False
    except Exception:
        return True          # supheliyse hesapla


def _closed_days(df) -> list:
    """Serideki KAPANMIS gunlerin (naive UTC) sirali listesi; olusan gun dusulur."""
    idx = df.sort_index().index
    if len(idx) == 0:
        return []
    last = pd.Timestamp(idx[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    n = len(idx)
    if last + pd.Timedelta(hours=24) > pd.Timestamp.now(tz="UTC"):
        n -= 1                                # olusan gun
    return [pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo is None
            else pd.Timestamp(t).tz_convert("UTC").tz_localize(None)
            for t in idx[:n]]


async def refresh(session, store, symbols) -> dict:
    """Store'daki 1D serilerinden tabloyu tazele (upsert; tamamlanmis satirlar atlanir).

    `symbols`: (bingx_symbol, display_symbol, market) uclulerinden olusan liste.
    Agir kisim thread'de -- event loop'u tikamamak icin (bkz. store yazim performansi notu).

    ⚠️ SEMBOL BASINA COMMIT. Ilk surumde tum semboller tek transaction'da birikiyordu: SQLAlchemy
    bir sonraki sembolun SELECT'inde autoflush yapip yazma islemini basliyor ve commit'e kadar
    (ilk acilista ~2500 satir, dakikalar) SQLite yazma kilidini tutuyordu. Sonuc: canli
    `setup_journal.flush` "database is locked" ile basarisiz oluyordu (18.09 10:40). Kisa
    transaction'lar bu aclik sorununu ortadan kaldirir; ayrica ilerleme DB'de gorunur ve bir
    sembolun hatasi digerlerini goturmez.
    """
    stats = {"symbols": 0, "yeni": 0, "guncel": 0, "hata": 0, "atlandi": 0}
    try:
        for bingx, display, market in symbols:
            try:
                df = store.get_df(bingx, "1d") if store is not None else None
                if df is None or getattr(df, "empty", True):
                    continue
                existing = {
                    r.day: r for r in (await session.execute(
                        select(BiasJournal).where(BiasJournal.symbol == display)
                    )).scalars().all()
                }
                if existing and not _needs_work(df, existing):
                    stats["atlandi"] += 1
                    continue                      # hesaplanacak yeni sey yok
                df = df.copy()
                rows = await asyncio.to_thread(build_rows, display, market, df)
                if not rows:
                    continue
                stats["symbols"] += 1
                yeni = guncel = 0
                for row in rows:
                    cur = existing.get(row["day"])
                    if cur is None:
                        session.add(BiasJournal(**row))
                        yeni += 1
                    elif _fill_outcomes(cur, row):
                        guncel += 1
                if yeni or guncel:
                    await session.commit()          # kisa transaction: kilidi tutma
                    stats["yeni"] += yeni
                    stats["guncel"] += guncel
                else:
                    session.expunge_all()
                await asyncio.sleep(0)              # event loop'a nefes aldir
            except Exception:
                stats["hata"] += 1
                try:
                    await session.rollback()
                except Exception:
                    pass
                log.exception("bias_journal.refresh failed for %s", bingx)
        # HER ZAMAN yaz: "0 yeni" ile "hic calismadi" ayirt edilebilsin (18.09'da bu ayrim
        # olmadigi icin turun calisip calismadigi logdan anlasilamadi).
        log.info("Bias journal: %d sembol islendi, %d atlandi, %d yeni, %d guncellenen satir, %d hata.",
                 stats["symbols"], stats["atlandi"], stats["yeni"], stats["guncel"], stats["hata"])
    except Exception:
        log.exception("bias_journal.refresh failed")
    return stats
