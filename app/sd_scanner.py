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
_STAGE_RANK = {"sd_dead": 0, "sd_bias": 1, "sd_no_target": 2, "sd_low_rr": 3, "sd_ready": 4}


def _gate(z: sde.SDZone, plan: sde.SDPlan, bias: str | None) -> str:
    if z.dead:
        return "sd_dead"
    if not sde.bias_aligned(z.direction, bias):
        return "sd_bias"
    if plan.tp is None:
        return "sd_no_target"
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
            journal.note(
                strategy, display, market, stage,
                direction=z.direction, purge_time=z.base_start, crt_bar_time=z.base_start,
                score=parts["score"], bias=bias, weekly_bias=weekly, rr=plan.rr,
                entry=plan.entry if plan.tp is not None else None,
                sl=plan.sl if plan.tp is not None else None, tp=plan.tp,
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
