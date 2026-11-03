"""Supply / Demand stratejisinin (`sd`, 03.11) canli baglantisi -- OLCUM MODU.

`scanner.on_candle_closed` 4H kapanisinda `evaluate_symbol`'u, 15m kapanisinda `journal.track_bar("sd", ...)`'i
cagirir (ikisi de run_scan'de de). Burada sinyal / Telegram yok: her canli bolge Setup Journal'a bir satir
(anahtar: strateji `sd` + sembol + yon + base'in ilk mumu), Radar'in S/D sekmesine sembol basina en iyi bolge.

Kapilar (olcum; Journal `best_stage`): sd_dead < sd_bias < sd_no_target < sd_low_rr < sd_ready.
Seviyeler (proximal / distal+tampon / hedef) kapidan bagimsiz Journal'a verilir ki kapinin eledigi bolgenin
sonrasi da olculsun (CRT'deki gibi). Olu bolge yalniz Journal'da zaten satiri varsa not edilir.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from app import sd_engine as sde
from app import setup_journal as journal
from app.crt_engine import compute_daily_bias, compute_weekly_bias
from app.exchange import market_of, to_display_symbol

log = logging.getLogger(__name__)

STRATEGY_SD = sde.STRATEGY_SD
MEASURE_ONLY = True   # sinyal acilmaz; canliya gecis ayri karar (kullanici)

# Iki olcum stratejisi (03.11, kullanici): 4H bolge (tum 4H evreni) ve 1H bolge (altcoin yok -- 1h akisi yalniz BTC/ETH +
# FX/metal/endeks/petrol'de var). Ikisi de 15m ile izlenir: 5m akisi 28.09'dan beri kapali, olcum icin 15m yeterli.
SD_CFG = {
    "sd":   {"htf": "4h", "delta": sde.HTF_DELTA},
    "sd1h": {"htf": "1h", "delta": timedelta(hours=1)},
}

# Radar siralamasi icin kapi onceligi (yuksek = daha ileri).
_STAGE_RANK = {"sd_dead": 0, "sd_bias": 1, "sd_no_target": 2, "sd_far": 3, "sd_low_rr": 4, "sd_ready": 5}


def _gate(z: sde.SDZone, plan: sde.SDPlan, bias: str | None) -> str:
    if z.dead:
        return "sd_dead"
    if not sde.bias_aligned(z.direction, bias):
        return "sd_bias"
    if plan.tp is None:
        return "sd_no_target"
    if plan.dist_r is not None and plan.dist_r > sde.MAX_RR:
        # 03.11 (kullanici: "fresh bolgeye TP yazmis"): giris fiyattan MAX_RR x risk'ten uzaksa 4R'lik TP fiyatin bu
        # tarafinda kalir -- Journal ilk mumda "TP before entry" sayiyordu (BTC 1H SHORT 8 ATR ustte, stop 0.3 ATR).
        # Boyle bolgeye emir de konmaz; seviye yazilmaz, izlenmez.
        return "sd_far"
    if plan.rr is None or plan.rr < sde.MIN_RR:
        return "sd_low_rr"
    return "sd_ready"


def _bars_since(df: pd.DataFrame, ts) -> int:
    try:
        return int(len(df) - 1 - df.index.get_loc(ts))
    except Exception:
        return 0


def evaluate_symbol(bingx_symbol: str, frames: dict, now=None, strategy: str = STRATEGY_SD) -> dict | None:
    """HTF kapanisinda: bolgeleri cikar, her canli bolgeyi Journal'a not et, Radar'a en iyisini yaz."""
    from app.scanner import _RADAR, _RADAR_META

    cfg = SD_CFG[strategy]
    display = to_display_symbol(bingx_symbol)
    market = market_of(bingx_symbol)
    now = now or datetime.now(timezone.utc)
    key = (strategy, display)
    try:
        df4 = frames.get(cfg["htf"])
        if df4 is None or len(df4) < sde.ATR_PERIOD + 10:
            _RADAR[key] = _radar_row(display, market, "no_data", now, strategy=strategy)
            return None
        closed = sde.closed_bars(df4, now, cfg["delta"])
        zones = sde.detect_zones(closed)
        df1d = frames.get("1d")
        bias = compute_daily_bias(df1d) if df1d is not None and not df1d.empty else "NEUTRAL"
        try:
            weekly = compute_weekly_bias(df1d) if df1d is not None and not df1d.empty else None
        except Exception:
            weekly = None
        last_close = float(closed["close"].iloc[-1])
        best = None
        n_live = 0
        for z in zones:
            age = _bars_since(closed, z.base_start)
            if age > sde.MAX_AGE_BARS:
                continue
            if z.dead and not journal.has_levels(strategy, display, z.direction, z.base_start):
                continue
            plan = sde.plan_zone(z, zones, closed, last_close)
            parts = sde.score_zone(z, bias)
            feats = sde.zone_features(z, last_close, age, plan)
            stage = _gate(z, plan, bias)
            has_levels = plan.tp is not None and stage != "sd_far"
            # Ilk gorus geri doldurmasi (03.11, kullanici: XAUUSD "neden Waiting? retest ve TP oldu"): restart / ilk calismada
            # bolge olusumundan sonra goruluyor; Journal yalniz donmadan SONRA kapanan 15m mumlarini izledigi icin aradaki
            # dolum/TP kayboluyordu. Yeni satirda seviyeler olusum anina (formed mumunun kapanisi) tarihlenir ve store'daki
            # kapanmis 15m mumlari o andan bugune yeniden oynatilir -- 15m gecmisi olusumu kapsiyorsa.
            df15 = frames.get("15m")
            if has_levels and not journal.has_record(strategy, display, z.direction, z.base_start) \
                    and df15 is not None and len(df15) and df15.index[0] <= z.formed_time + cfg["delta"]:
                # Ilk gorus geri doldurmasi: olusumdan bugune HTF mumu mumu (o anki engeller / bias / uzaklik) yuruyup
                # kapanmis 15m mumlarini oynat -- motor surekli calissaydi ne gorecekse o. Bugunku degerlendirme en sonda.
                _backfill_zone(strategy, display, market, z, closed, df15, d1 if (d1 := frames.get("1d")) is not None else None,
                               cfg["delta"], now, weekly)
            journal.note(
                strategy, display, market, stage,
                direction=z.direction, purge_time=z.base_start, crt_bar_time=z.base_start,
                score=parts["score"], bias=bias, weekly_bias=weekly, rr=plan.rr,
                entry=plan.entry if has_levels else None,
                sl=plan.sl if has_levels else None,
                tp=plan.tp if has_levels else None,
                model=z.zone_type, parts=parts, features=feats, now=now,
            )
            if z.dead:
                continue
            n_live += 1
            cand = (_STAGE_RANK[stage], -(feats.get("dist_atr") or 0.0))
            if best is None or cand > best[0]:
                best = (cand, z, plan, parts, feats, stage)
        if best is None:
            row = _radar_row(display, market, "no_setup", now, strategy=strategy)
        else:
            _, z, plan, parts, feats, stage = best
            row = _radar_row(
                display, market, stage, now, strategy=strategy,
                direction=z.direction, score=parts["score"], bias=bias, weekly_bias=weekly,
                rr=plan.rr, entry=plan.entry, sl=plan.sl, tp=plan.tp, model=z.zone_type,
                purge_time=z.base_start, crt_bar_time=z.base_start, parts=parts, features=feats,
                sd={   # JSON'a gidecek: numpy tipleri duz Python'a
                    "zone_type": z.zone_type, "touches": int(z.touches), "fresh": bool(z.fresh),
                    "in_zone": bool(z.in_zone), "legout_atr": float(z.legout_atr), "base_bars": int(z.base_bars),
                    "fvg": bool(z.fvg), "bos": bool(z.bos), "sweep": bool(z.sweep),
                    "dist_atr": feats.get("dist_atr"), "target_kind": plan.target_kind, "rr_struct": plan.rr_struct,
                    "zone_atr": round(float(z.height_atr), 2), "n_live": n_live, "n_zones": len(zones),
                },
            )
        row["sd"] = {**(row.get("sd") or {}), "n_live": n_live, "n_zones": len(zones)}
        _RADAR[key] = row
        _RADAR_META["last_update"] = now
        return row
    except Exception:
        log.exception("%s evaluate failed for %s", strategy, bingx_symbol)
        return None


def _backfill_zone(strategy, display, market, z, closed, df15, df1d, delta, now, weekly) -> None:
    """Yeni gorulen bolge icin gecmisi yeniden oynat: her HTF kapanisinda o anki plan/kapi ile `journal.note` (dolum oncesi
    TP guncellemesi dahil), aradaki kapanmis 15m mumlariyla `track_bar`. Bugunun kapanisi dahil degil (cagiran yazar)."""
    now_ts = _ts(now, closed.index[0])
    hts = [t for t in closed.index if t >= z.formed_time and t + delta < now_ts]
    prev_close = None
    for t in hts:
        past = closed[closed.index <= t]
        zones_at = sde.detect_zones(past)
        twin = next((q for q in zones_at if q.base_start == z.base_start and q.side == z.side), None)
        if twin is None:
            continue
        last_close = float(past["close"].iloc[-1])
        plan_at = sde.plan_zone(twin, zones_at, past, last_close)
        bias_at = "NEUTRAL"
        if df1d is not None and not df1d.empty:
            d1p = df1d[df1d.index <= t]
            bias_at = compute_daily_bias(d1p) if not d1p.empty else "NEUTRAL"
        stage_at = _gate(twin, plan_at, bias_at)
        lv = plan_at.tp is not None and stage_at != "sd_far"
        parts_at = sde.score_zone(twin, bias_at)
        feats_at = sde.zone_features(twin, last_close, _bars_since(past, twin.base_start), plan_at)
        t_close = (t + delta).to_pydatetime()
        journal.note(
            strategy, display, market, stage_at,
            direction=twin.direction, purge_time=twin.base_start, crt_bar_time=twin.base_start,
            score=parts_at["score"], bias=bias_at, weekly_bias=weekly, rr=plan_at.rr,
            entry=plan_at.entry if lv else None, sl=plan_at.sl if lv else None, tp=plan_at.tp if lv else None,
            model=twin.zone_type, parts=parts_at, features=feats_at, now=t_close,
        )
        lo = prev_close if prev_close is not None else t
        bars = df15[(df15.index + timedelta(minutes=15) > lo) & (df15.index + timedelta(minutes=15) <= t + delta)]
        for ts, b in bars.iterrows():
            journal.track_bar(strategy, display, ts, float(b["high"]), float(b["low"]), float(b["close"]))
        prev_close = t + delta
        if twin.dead:
            break


def _ts(now, like) -> pd.Timestamp:
    """`now`'u `like` ile ayni tz'de Timestamp yap (store indeksi tz'li, test verisi tz'siz olabilir)."""
    t = pd.Timestamp(now)
    if getattr(like, "tzinfo", None) is None:
        return t.tz_localize(None) if t.tzinfo is not None else t
    return t.tz_localize(timezone.utc) if t.tzinfo is None else t.tz_convert(like.tzinfo)


def _radar_row(display: str, market: str | None, state: str, now, **kw) -> dict:
    """Radar sozlugu: `scanner._set_radar`'in anahtarlari + `sd` alt sozlugu (S/D sekmesinin sutunlari)."""
    row = {
        "symbol": display, "market": market, "state": state,
        "direction": None, "score": None, "bias": None, "weekly_bias": None, "rr": None,
        "entry": None, "sl": None, "tp": None, "smt": None, "pd": None, "c2_closed": None,
        "ifvg": None, "bpr": None, "model": None, "purge_time": None, "crt_bar_time": None,
        "parts": None, "features": None, "sd": None,
        "strategy": STRATEGY_SD, "updated_at": now,
    }
    row.update(kw)
    return row
