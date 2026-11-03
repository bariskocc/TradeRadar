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
            note_now, df15, lv = now, frames.get("15m"), plan
            if has_levels and not journal.has_record(strategy, display, z.direction, z.base_start):
                formed_close = z.formed_time + cfg["delta"]
                if df15 is not None and len(df15) and df15.index[0] <= formed_close < _ts(now, df15.index[0]):
                    # Seviyeler OLUSUM ANINDAKI engellerle (o gunku bolgeler / swingler) hesaplanir; bugunku hedefle geriye
                    # gitmek sonradan olusan swing'i hedef sayip sahte "TP before entry" uretiyordu (XAUUSD 3978).
                    past = closed[closed.index <= z.formed_time]
                    zones_at = sde.detect_zones(past)
                    twin = next((q for q in zones_at if q.base_start == z.base_start and q.side == z.side), None)
                    plan_at = sde.plan_zone(twin, zones_at, past, float(past["close"].iloc[-1])) if twin is not None else None
                    if plan_at is not None and plan_at.tp is not None and plan_at.dist_r is not None \
                            and plan_at.dist_r <= sde.MAX_RR and plan_at.rr >= sde.MIN_RR:
                        note_now, lv = formed_close.to_pydatetime(), plan_at
            journal.note(
                strategy, display, market, stage,
                direction=z.direction, purge_time=z.base_start, crt_bar_time=z.base_start,
                score=parts["score"], bias=bias, weekly_bias=weekly, rr=lv.rr,
                entry=lv.entry if has_levels else None,
                sl=lv.sl if has_levels else None,
                tp=lv.tp if has_levels else None,
                model=z.zone_type, parts=parts, features=feats, now=note_now,
            )
            if note_now is not now:
                for ts, b in df15[df15.index >= note_now].iterrows():
                    if ts + timedelta(minutes=15) > _ts(now, ts):
                        break                                   # olusan mum
                    journal.track_bar(strategy, display, ts, float(b["high"]), float(b["low"]), float(b["close"]))
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
