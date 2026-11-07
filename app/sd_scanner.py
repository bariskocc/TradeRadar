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

from app import sd_confirm
from app import sd_engine as sde
from app import setup_journal as journal
from app.crt_engine import compute_daily_bias, compute_weekly_bias
from app.exchange import market_of, to_display_symbol

log = logging.getLogger(__name__)

STRATEGY_SD = sde.STRATEGY_SD
MEASURE_ONLY = True   # sinyal acilmaz; canliya gecis ayri karar (kullanici)
# 1D bias hard kapisi (04.11, kullanici karari: KAPALI). CRT'den miras `sd_bias` XAUUSD 1H 04.11 06:00 TSI demand'ini (1D BEARISH,
# leg-out 4.8 ATR) eliyordu; S/D'de bolge zaten trende karsi donus olarak calisir. Bias yalniz skor kalemi (+2/-2); etkisi
# `/sd-izleme` sayaclarinda (features/parts "htf"). True yapmak eski davranisi geri getirir.
BIAS_GATE = False

# Iki olcum stratejisi (03.11, kullanici): 4H bolge (tum 4H evreni) ve 1H bolge (altcoin yok -- 1h akisi yalniz BTC/ETH +
# FX/metal/endeks/petrol'de var). Ikisi de 15m ile izlenir: 5m akisi 28.09'dan beri kapali, olcum icin 15m yeterli.
SD_CFG = {
    "sd":    {"htf": "4h", "delta": sde.HTF_DELTA},
    "sd1h":  {"htf": "1h", "delta": timedelta(hours=1)},
    # 03.11 (kullanici): 15m bolgeleri, yalniz set-and-forget (CHOCH kolu yok, izleme maddesi yok), 1D evreni, 5m izleme.
    "sd15m": {"htf": "15m", "delta": timedelta(minutes=15)},
}
ALL = tuple(SD_CFG)

# Radar siralamasi icin kapi onceligi (yuksek = daha ileri).
_STAGE_RANK = {"sd_dead": 0, "sd_bias": 1, "sd_no_target": 2, "sd_far": 3, "sd_low_rr": 4, "sd_near": 4.5, "sd_ready": 5}
NEAR_STAGE = "sd_near"   # 05.11: cikis govdesi NEAR_LEG_BODY_ATR..LEG_BODY_MIN_ATR -- kapilari gecse de Signals'a girmez (olcum)


def _near_zones(closed: pd.DataFrame, zones: list) -> list:
    """Yakin kacan bolgeler: gevsek cikis govdesiyle bulunup ana tespitte olmayanlar (ana kume degismez)."""
    main = {(z.side, z.base_start) for z in zones}
    return [q for q in sde.detect_zones(closed, leg_body_min=sde.NEAR_LEG_BODY_ATR)
            if q.legout_body_atr < sde.LEG_BODY_MIN_ATR and (q.side, q.base_start) not in main]


def week_close(store, now=None) -> int:
    """Cuma 17:00 NY (market_data, motorun hafta kapanisiyla ayni an): kripto disi S/D satirlarinin dolmus kollarini haftanin son
    fiyatiyla kapat (setup_journal.sd_week_close). Dolmamis bolge beklemeye devam eder. Kapanan satir sayisi."""
    from app.exchange import from_display_symbol, is_session_symbol
    from app.session import last_week_close
    ts = last_week_close(now or datetime.now(timezone.utc))
    n = 0
    for key, rec in list(journal._CACHE.items()):
        if key[0] not in SD_CFG or rec.get("levels_at") is None:
            continue
        try:
            bingx = from_display_symbol(key[1])[0]
            if not is_session_symbol(bingx):
                continue
            df = store.get_df(bingx, "1h") if store is not None else None
            px = float(df.iloc[-1]["close"]) if df is not None and not df.empty else None
        except Exception:
            continue
        if px is not None and journal.sd_week_close(rec, px, ts):
            journal._DIRTY.add(key)
            n += 1
    if n:
        log.info("SD WEEK CLOSE: %d S/D satiri haftanin son fiyatiyla kapandi", n)
    return n


def _gate(z: sde.SDZone, plan: sde.SDPlan, bias: str | None) -> str:
    if z.dead:
        return "sd_dead"
    if BIAS_GATE and not sde.bias_aligned(z.direction, bias):
        return "sd_bias"
    if plan.tp is None:
        return "sd_no_target"
    if plan.dist_r is not None and plan.dist_r > sde.MAX_DIST_R:
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
        for z, is_near in [(z, False) for z in zones] + [(q, True) for q in _near_zones(closed, zones)]:
            age = _bars_since(closed, z.base_start)
            if age > sde.MAX_AGE_BARS:
                continue
            plan = sde.plan_zone(z, zones, closed, last_close)
            parts = sde.score_zone(z, bias)
            feats = sde.zone_features(z, last_close, age, plan)
            stage = _gate(z, plan, bias)
            if is_near:
                feats["near"] = 1
                stage = NEAR_STAGE if stage == "sd_ready" else stage
            # Ilk gorus geri doldurmasi BUGUNKU kapidan bagimsiz (04.11): bolge bugun uzak/olu olsa da gecmiste hazir olup dolmus
            # olabilir (XAUUSD 1H: bugun 4R uzakta, dun TP). Yeni kayitsa HTF mumu mumu yurunur; seviye yalniz o anki sd_ready'de.
            # Izleme serisi: sd1h/sd15m'de 5m varsa 5m (SD_5M_ENABLED), yoksa 15m.
            df5 = frames.get("5m") if strategy in ("sd1h", "sd15m") else None
            df15, ltf_delta = (df5, timedelta(minutes=5)) if df5 is not None and len(df5) else (frames.get("15m"), timedelta(minutes=15))
            if not journal.has_record(strategy, display, z.direction, z.base_start)                     and df15 is not None and len(df15) and df15.index[0] <= z.formed_time + cfg["delta"]:
                _backfill_zone(strategy, display, market, z, closed, df15, d1 if (d1 := frames.get("1d")) is not None else None,
                               cfg["delta"], now, weekly, ltf_delta, near=is_near)
            if z.dead and not journal.has_levels(strategy, display, z.direction, z.base_start):
                continue
            if is_near and stage != NEAR_STAGE and not journal.has_record(strategy, display, z.direction, z.base_start):
                continue              # yakin kacan bolge yalniz kapilari gectiyse kaydedilir (Journal kapi sayilari karismasin)
            # 04.11: seviyeler YALNIZ butun kapilari gecince donar. Eskiden low_rr/no_target bolgelere de seviye veriliyordu
            # ("kapi koruyor mu" olcumu); NZDUSD 1H 04.11: low_rr aninda donan TP (RR 1.1) ile dolup sonra sd_ready oldu --
            # Signals'ta RR 1.1'lik "sinyal". Kapida kalan bolge artik seviyesiz (asama kaydi var, sonuc yok).
            has_levels = plan.tp is not None and stage in ("sd_ready", NEAR_STAGE)
            if has_levels:
                _seed_ltf_liq(strategy, feats, z, df15, now, ltf_delta)
            # Ilk gorus geri doldurmasi (03.11, kullanici: XAUUSD "neden Waiting? retest ve TP oldu"): restart / ilk calismada
            # bolge olusumundan sonra goruluyor; Journal yalniz donmadan SONRA kapanan 15m mumlarini izledigi icin aradaki
            # dolum/TP kayboluyordu. Yeni satirda seviyeler olusum anina (formed mumunun kapanisi) tarihlenir ve store'daki
            # kapanmis 15m mumlari o andan bugune yeniden oynatilir -- 15m gecmisi olusumu kapsiyorsa.
            journal.note(
                strategy, display, market, stage,
                direction=z.direction, purge_time=z.base_start, crt_bar_time=z.base_start,
                score=parts["score"], bias=bias, weekly_bias=weekly, rr=plan.rr,
                entry=plan.entry if has_levels else None,
                sl=plan.sl if has_levels else None,
                tp=plan.tp if has_levels else None,
                model=z.zone_type, parts=parts, features=feats, now=now,
                entries=_entry_cands(z, plan, last_close) if has_levels else None,
                c1=plan.sl_zone if has_levels else None,            # golge: bolge stopu (distal + tampon)
            )
            if z.dead or is_near:
                continue              # Radar yalniz ana bolgeleri gosterir
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
                    "fvg": bool(z.fvg), "bos": bool(z.bos), "mss": bool(z.mss), "sweep": bool(z.sweep), "sweep_kind": z.sweep_kind,
                    "dist_atr": feats.get("dist_atr"), "target_kind": plan.target_kind, "rr_struct": plan.rr_struct,
                    "zone_atr": round(float(z.height_atr), 2), "n_live": n_live, "n_zones": len(zones),
                    "ltf_sweep": _ltf_sweep_n(strategy, display, z),
                },
            )
        row["sd"] = {**(row.get("sd") or {}), "n_live": n_live, "n_zones": len(zones)}
        _RADAR[key] = row
        _RADAR_META["last_update"] = now
        return row
    except Exception:
        log.exception("%s evaluate failed for %s", strategy, bingx_symbol)
        return None


def _backfill_zone(strategy, display, market, z, closed, df15, df1d, delta, now, weekly,
                   ltf_delta=timedelta(minutes=15), near: bool = False) -> None:
    """Yeni gorulen bolge icin gecmisi yeniden oynat: her HTF kapanisinda o anki plan/kapi ile `journal.note` (dolum oncesi
    TP guncellemesi dahil), aradaki kapanmis 15m mumlariyla `track_bar`. Bugunun kapanisi dahil degil (cagiran yazar)."""
    now_ts = _ts(now, closed.index[0])
    hts = [t for t in closed.index if t >= z.formed_time and t + delta < now_ts]
    prev_close = None
    for t in hts:
        past = closed[closed.index <= t]
        zones_at = sde.detect_zones(past)
        pool = _near_zones(past, zones_at) if near else zones_at
        twin = next((q for q in pool if q.base_start == z.base_start and q.side == z.side), None)
        if twin is None:
            continue
        last_close = float(past["close"].iloc[-1])
        plan_at = sde.plan_zone(twin, zones_at, past, last_close)
        bias_at = "NEUTRAL"
        if df1d is not None and not df1d.empty:
            d1p = df1d[df1d.index <= t]
            bias_at = compute_daily_bias(d1p) if not d1p.empty else "NEUTRAL"
        stage_at = _gate(twin, plan_at, bias_at)
        if near:
            stage_at = NEAR_STAGE if stage_at == "sd_ready" else stage_at
        lv = plan_at.tp is not None and stage_at in ("sd_ready", NEAR_STAGE)
        parts_at = sde.score_zone(twin, bias_at)
        feats_at = sde.zone_features(twin, last_close, _bars_since(past, twin.base_start), plan_at)
        if near:
            feats_at["near"] = 1
        skip_note = near and stage_at != NEAR_STAGE and not journal.has_record(strategy, display, twin.direction, twin.base_start)
        t_close = (t + delta).to_pydatetime()
        if lv:
            _seed_ltf_liq(strategy, feats_at, twin, df15, t_close, ltf_delta)
        if not skip_note:
            journal.note(
                strategy, display, market, stage_at,
                direction=twin.direction, purge_time=twin.base_start, crt_bar_time=twin.base_start,
                score=parts_at["score"], bias=bias_at, weekly_bias=weekly, rr=plan_at.rr,
                entry=plan_at.entry if lv else None, sl=plan_at.sl if lv else None, tp=plan_at.tp if lv else None,
                model=twin.zone_type, parts=parts_at, features=feats_at, now=t_close,
                entries=_entry_cands(twin, plan_at, last_close) if lv else None,
                c1=plan_at.sl_zone if lv else None,
            )
        lo = prev_close if prev_close is not None else t
        bars = df15[(df15.index + ltf_delta > lo) & (df15.index + ltf_delta <= t + delta)]
        for ts, b in bars.iterrows():
            journal.track_bar(strategy, display, ts, float(b["high"]), float(b["low"]), float(b["close"]))
        prev_close = t + delta
        if twin.dead:
            break


def _ltf_sweep_n(strategy, display, z) -> int:
    """Radar: Journal satirinin LTF onay kolunda dokunustan sonra alinan LTF swing sayisi (yoksa 0)."""
    try:
        rec = journal._CACHE.get(journal._key(strategy, display, z.direction, z.base_start)) or {}
        return int(((rec.get("entries") or {}).get(sd_confirm.SDC_KEY) or {}).get("sw") or 0)
    except Exception:
        return 0


def _seed_ltf_liq(strategy, feats: dict, z, df_ltf, now, ltf_delta) -> None:
    """LTF onay kolunun sweep tohumu (05.11): base'ten bu ana kadar kapanmis LTF mumlarindaki swing dip / tepeler ve ilk alindiklari
    mum (`sd_confirm.ltf_liq`). Dokunusta 'alinmamis' olanlar, dip onlari alirsa LTF sweep sayilir."""
    if strategy not in sd_confirm.STRATEGIES or df_ltf is None or not len(df_ltf):
        return
    try:
        feats["ltf_liq"] = sd_confirm.ltf_liq(df_ltf, z.base_start, now - ltf_delta + timedelta(seconds=1),
                                              z.direction == "LONG")
    except Exception:
        log.exception("ltf_liq seed failed")


def _entry_cands(z, plan, last_close: float) -> dict:
    """Journal giris varyantlari (04.11): chosen = proximal (bugunku), zone_mid = bolge ortasi, zone_far = distal kenar;
    07.11 zone_wick = 'fitil-fitil' cizim, base'in karsi fitili (butun mum; doji base'de govde ortada kaliyordu).
    SL/TP satirinki; `_ref` = seviye andaki fiyat (Journal 'zaten gecilmis giris' isaretlemesi icin)."""
    return {"chosen": float(plan.entry), "zone_mid": float((z.proximal + z.distal) / 2), "zone_far": float(z.distal),
            **({"zone_wick": float(z.wick_prox)} if getattr(z, "wick_prox", None) is not None else {}),
            "_ref": float(last_close)}


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
