"""Setup Journal: her setup'in motor kapilarindaki yolu ve elendikten sonra fiyatin ne yaptigi.

Neden: eleme/silme nedenleri yalnizca log satiriydi ve hangi stratejiye ait oldugu yazmiyordu;
"sinyal neden gelmedi" sorusu her seferinde logdan elle eslestiriliyordu. Radar bellekte ve
restart'ta siliniyor.

Kayit anahtari: strateji + sembol + yon + purge (C2) zamani -> setup basina TEK satir.

Akis:
- `scanner._set_radar` setup iceren her durumda `note(...)` cagirir: bellekteki kayit guncellenir.
  DB'ye her degerlendirmede yazilmaz; `flush(session)` degisenleri tek commit'te yazar
  (on_candle_closed / on_price_update / run_scan / reconcile / hafta kapanisi sonunda).
- Silinen pending/waiting: `note_deleted(sig, reason)`.
- Sonuc izleme: sinyale donusmemis ve seviyeli kayit icin kapanan her LTF mumunda `track_bar(...)`:
  fiyat once entry'ye mi, TP'ye mi, SL'ye mi degdi? Seviyeler ILK goruldugu haliyle dondurulur
  (setup sonradan yeniden seviyelenirse izleme kaymasin). Ufuk strateji bazli (HORIZON).
- Restart: `ensure_loaded(session, store)` son KEEP gunun kayitlarini yukler ve izlenen kayitlari
  store'daki kapanmis LTF mumlariyla telafi eder.

Bu modulun hicbir hatasi islem akisini bozmamali: disariya acilan her fonksiyon hatayi yutar ve loglar.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models import SetupJournal

log = logging.getLogger(__name__)

# Motor kapilari, geriden ileriye. Kayit ulastigi EN ILERI asamayi tutar.
STAGES: list[tuple[str, str]] = [
    ("bias_mismatch", "1D bias opposite"),
    ("target_taken", "Target already taken"),
    ("low_quality", "Score < 7"),
    ("no_cisd", "No entry levels / MSS"),
    ("low_rr", "RR < 2"),
    ("score7", "Score-7 gate"),
    ("tight_stop", "Stop too tight"),
    ("past_tp", "TP before entry"),
    ("past_sl", "SL before fill"),
    ("invalidated", "CRT 60% crossed"),
    ("missed_quality", "Score < 7 at retest"),
    ("stale", "Retest too old"),
    ("cluster_limit", "Cluster limit"),
    ("corr_open", "Correlated pair open"),
    ("has_open", "Open signal exists"),
    ("duplicate", "Already saved"),
    ("c2_open", "Waiting for C2 close"),
    ("waiting", "Signal opened"),
    ("week_close", "Cancelled at week close"),
]
STAGE_LABELS = dict(STAGES)
_RANK = {code: i for i, (code, _) in enumerate(STAGES)}
SIGNAL_STAGE = "waiting"

OUTCOME_LABELS = {
    "pending": "Tracking (no entry yet)",
    "filled": "Tracking (entry touched)",
    "tp_before_entry": "TP before entry",
    "win": "Entry then TP",
    "loss": "Entry then SL",
    "no_touch": "No entry in horizon",
    "open": "Entry, no TP/SL in horizon",
    "ambiguous": "Same bar (order unknown)",
    "signal": "Became a signal",
}
TRACKING = ("pending", "filled")
HORIZON = {"4h": timedelta(hours=48), "1d": timedelta(hours=120), "1h": timedelta(hours=12)}
LTF_OF = {"4h": "15m", "1d": "1h", "1h": "5m"}
KEEP = timedelta(days=10)
# Yalnizca last_seen ilerlediyse bu kadar bekleyip yaz (her 15 dk'da her satiri yazmamak icin).
_TOUCH_EVERY = timedelta(minutes=30)

# Golge izleme (14.09): "stop daha yakin olsaydi?" sorusu icin geriye donuk replay yerine canli
# veri (once 1H, ayni gun 4H/1D). Seviyeleri olan her setup, SL = entry + k x (SL - entry) ile
# ayrica izlenir; k=1 bugunku kural (sinyale donusen setup'ta ana izleme durdugu icin temel cizgi
# burada tutulur). Duz TP/SL: 4H/1D'deki kismi kar + BE yok. Degerlendirme ve karar kurali:
# IZLEME.md "Dar stop — golge izleme".
SHADOW_SL = {"4h": (1.0, 0.75, 0.5), "1d": (1.0, 0.75, 0.5), "1h": (1.0, 0.75, 0.5)}
_SHADOW_TS = ("e", "at", "until")

_CACHE: dict[tuple, dict] = {}
_DIRTY: set[tuple] = set()
_STATE = {"loaded": False}


def _shadow_init(rec: dict) -> dict | None:
    fractions = SHADOW_SL.get(rec.get("strategy"))
    e, sl, tp = rec.get("entry"), rec.get("sl"), rec.get("tp")
    if not fractions or None in (e, sl, tp) or e == sl:
        return None
    risk = abs(e - sl)
    return {
        f"{k:g}": {"k": k, "sl": e + k * (sl - e), "rr": round(abs(tp - e) / (k * risk), 4),
                   "o": "pending", "e": None, "at": None, "until": None}
        for k in fractions
    }


def _shadow_dump(shadow: dict | None) -> str | None:
    if not shadow:
        return None
    out = {
        name: {f: (v[f].isoformat() if f in _SHADOW_TS and v.get(f) is not None else v.get(f)) for f in v}
        for name, v in shadow.items()
    }
    return json.dumps(out, separators=(",", ":"))


def _shadow_load(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        for v in data.values():
            for f in _SHADOW_TS:
                if v.get(f):
                    v[f] = datetime.fromisoformat(v[f])
        return data
    except Exception:
        return None


def _shadow_active(rec: dict, bar_ts=None) -> bool:
    shadow = rec.get("shadow")
    if not shadow or rec.get("levels_at") is None:
        return False
    if bar_ts is not None and bar_ts < rec["levels_at"]:
        return False
    return any(
        v["o"] in TRACKING and (bar_ts is None or v.get("until") is None or bar_ts > v["until"])
        for v in shadow.values()
    )


def _apply_shadow(rec: dict, bar_ts, high: float, low: float) -> None:
    """_apply_bar'in ayni kurallari, her SL varyanti icin ayri."""
    long = rec["direction"] == "LONG"
    e, tp = rec["entry"], rec["tp"]
    hit_e = low <= e if long else high >= e
    hit_tp = high >= tp if long else low <= tp
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    for v in rec["shadow"].values():
        if v["o"] not in TRACKING or (v.get("until") is not None and bar_ts <= v["until"]):
            continue
        hit_sl = low <= v["sl"] if long else high >= v["sl"]
        v["until"] = bar_ts
        if v["o"] == "pending":
            if hit_tp and not hit_e:
                v["o"], v["at"] = "tp_before_entry", bar_ts
            elif hit_e:
                v["e"] = bar_ts
                if hit_tp:
                    v["o"], v["at"] = "ambiguous", bar_ts
                elif hit_sl:
                    v["o"], v["at"] = "loss", bar_ts
                else:
                    v["o"] = "filled"
        elif v["o"] == "filled":
            if hit_tp and hit_sl:
                v["o"], v["at"] = "ambiguous", bar_ts
            elif hit_tp:
                v["o"], v["at"] = "win", bar_ts
            elif hit_sl:
                v["o"], v["at"] = "loss", bar_ts
        if v["o"] in TRACKING and bar_ts >= rec["levels_at"] + horizon:
            v["o"], v["at"] = ("no_touch" if v["o"] == "pending" else "open"), bar_ts

_COLUMNS = (
    "market_type", "crt_bar_time", "first_seen", "last_seen", "last_stage", "best_stage", "best_stage_at",
    "detail", "score", "htf_bias", "weekly_bias", "c2_closed", "model", "entry", "sl", "tp", "rr", "levels_at",
    "deleted_reason", "deleted_at", "outcome", "outcome_at", "entry_touched_at", "tracked_until",
)


def _naive_utc(ts):
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _now(now=None):
    return _naive_utc(now) if now is not None else datetime.now(timezone.utc).replace(tzinfo=None)


def _key(strategy, symbol, direction, purge_time) -> tuple:
    return (strategy or "4h", symbol, direction, _naive_utc(purge_time))


def stage_rank(stage: str | None) -> int:
    return _RANK.get(stage or "", -1)


def _detail(score, rr, bias, weekly, c2_closed) -> str | None:
    parts = []
    if score is not None:
        parts.append(f"score {int(score)}")
    if rr is not None:
        parts.append(f"RR {float(rr):.2f}")
    if bias:
        parts.append(f"1D {bias}")
    if weekly:
        parts.append(f"1W {weekly}")
    if c2_closed is False:
        parts.append("C2 open")
    return " · ".join(parts) or None


def note(
    strategy: str,
    symbol: str,
    market: str | None,
    stage: str,
    *,
    direction: str | None = None,
    purge_time=None,
    crt_bar_time=None,
    score=None,
    bias: str | None = None,
    weekly_bias: str | None = None,
    rr=None,
    entry=None,
    sl=None,
    tp=None,
    c2_closed: bool | None = None,
    model: str | None = None,
    now=None,
) -> None:
    """Bir degerlendirmenin sonucunu kayda isle (yalnizca bellek)."""
    try:
        if direction is None or purge_time is None or stage not in _RANK:
            return
        now = _now(now)
        key = _key(strategy, symbol, direction, purge_time)
        rec = _CACHE.get(key)
        changed = False
        if rec is None:
            rec = {"id": None, "strategy": key[0], "symbol": symbol, "direction": direction,
                   "purge_time": key[3], "first_seen": now, "best_stage": stage, "best_stage_at": now,
                   "outcome": None, "_flushed_seen": None}
            _CACHE[key] = rec
            changed = True
        rec["last_seen"] = now
        if market:
            rec["market_type"] = market
        if crt_bar_time is not None:
            rec["crt_bar_time"] = _naive_utc(crt_bar_time)
        if rec.get("last_stage") != stage:
            rec["last_stage"] = stage
            changed = True
        if stage_rank(stage) > stage_rank(rec.get("best_stage")):
            rec["best_stage"], rec["best_stage_at"] = stage, now
            changed = True
        for k, v in (("score", score), ("htf_bias", bias), ("weekly_bias", weekly_bias),
                     ("c2_closed", c2_closed), ("model", model)):
            if v is not None and rec.get(k) != v:
                rec[k] = v
                changed = True
        if entry is not None and sl is not None and tp is not None and rec.get("levels_at") is None:
            # Ilk gorulen seviyeler dondurulur: sonuc izleme bunlarla yapilir.
            rec.update(entry=float(entry), sl=float(sl), tp=float(tp),
                       rr=float(rr) if rr is not None else None, levels_at=now)
            rec["shadow"] = _shadow_init(rec)
            changed = True
        detail = _detail(score, rr, bias, weekly_bias, c2_closed)
        if detail and rec.get("detail") != detail:
            rec["detail"] = detail
            changed = True
        if stage == SIGNAL_STAGE and rec.get("outcome") != "signal":
            rec["outcome"] = "signal"
            changed = True
        elif rec.get("levels_at") is not None and rec.get("outcome") is None:
            rec["outcome"] = "pending"
            changed = True
        seen = rec.get("_flushed_seen")
        if changed or seen is None or now - seen >= _TOUCH_EVERY:
            _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.note failed for %s %s", strategy, symbol)


def note_deleted(sig, reason: str, now=None) -> None:
    """Silinen pending/waiting sinyali kayda isle."""
    try:
        if sig is None or getattr(sig, "purge_time", None) is None:
            return
        now = _now(now)
        strategy = getattr(sig, "timeframe", None) or "4h"
        key = _key(strategy, sig.symbol, sig.direction, sig.purge_time)
        if key not in _CACHE:
            stage = SIGNAL_STAGE if sig.status in ("waiting_entry", "active") else "c2_open"
            note(strategy, sig.symbol, sig.market_type, stage, direction=sig.direction, purge_time=sig.purge_time,
                 crt_bar_time=sig.crt_bar_time, score=sig.bias_score, bias=sig.htf_bias,
                 weekly_bias=sig.weekly_bias, rr=sig.planned_rr, entry=sig.entry_price, sl=sig.stop_loss,
                 tp=sig.take_profit, c2_closed=sig.c2_closed, model=sig.entry_model, now=now)
        rec = _CACHE.get(key)
        if rec is None:
            return
        rec["deleted_reason"], rec["deleted_at"] = reason, now
        _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.note_deleted failed for %s", getattr(sig, "symbol", "?"))


def _final(rec: dict, outcome: str, ts) -> None:
    rec["outcome"], rec["outcome_at"] = outcome, ts


def _apply_bar(rec: dict, bar_ts, high: float, low: float) -> None:
    long = rec["direction"] == "LONG"
    e, sl, tp = rec["entry"], rec["sl"], rec["tp"]
    hit_e = low <= e if long else high >= e
    hit_tp = high >= tp if long else low <= tp
    hit_sl = low <= sl if long else high >= sl
    rec["tracked_until"] = bar_ts
    if rec["outcome"] == "pending":
        if hit_tp and not hit_e:
            _final(rec, "tp_before_entry", bar_ts)
        elif hit_e:
            rec["entry_touched_at"] = bar_ts
            if hit_tp:
                # Ayni mumda entry ve TP: TP once mi (dolmazdi) sonra mi (kazanc) bilinmez.
                _final(rec, "ambiguous", bar_ts)
            elif hit_sl:
                # SL entry'nin otesinde; fiyat entry tarafindan geldigi icin once entry dolar, sonra SL.
                _final(rec, "loss", bar_ts)
            else:
                rec["outcome"] = "filled"
    elif rec["outcome"] == "filled":
        if hit_tp and hit_sl:
            _final(rec, "ambiguous", bar_ts)
        elif hit_tp:
            _final(rec, "win", bar_ts)
        elif hit_sl:
            _final(rec, "loss", bar_ts)
    horizon = HORIZON.get(rec["strategy"], timedelta(hours=48))
    if rec["outcome"] in TRACKING and bar_ts >= rec["levels_at"] + horizon:
        _final(rec, "no_touch" if rec["outcome"] == "pending" else "open", bar_ts)


def _is_tracking(rec: dict, bar_ts) -> bool:
    if rec.get("outcome") not in TRACKING or rec.get("levels_at") is None:
        return False
    if bar_ts < rec["levels_at"]:
        return False
    return rec.get("tracked_until") is None or bar_ts > rec["tracked_until"]


def track_bar(strategy: str, symbol: str, bar_ts, high: float, low: float) -> None:
    """Kapanan LTF mumunu, o sembol+stratejinin izlenen kayitlarina uygula."""
    try:
        bar_ts = _naive_utc(bar_ts)
        for key, rec in _CACHE.items():
            if key[0] != strategy or key[1] != symbol:
                continue
            touched = False
            if _is_tracking(rec, bar_ts):
                _apply_bar(rec, bar_ts, float(high), float(low))
                touched = True
            if _shadow_active(rec, bar_ts):
                _apply_shadow(rec, bar_ts, float(high), float(low))
                touched = True
            if touched:
                _DIRTY.add(key)
    except Exception:
        log.exception("setup_journal.track_bar failed for %s %s", strategy, symbol)


def _merge_row(rec: dict, row: SetupJournal) -> None:
    """DB satiri ile (restart oncesi) bellekteki yeni kaydi birlestir."""
    rec["id"] = row.id
    if row.first_seen and (rec.get("first_seen") is None or row.first_seen < rec["first_seen"]):
        rec["first_seen"] = row.first_seen
    if stage_rank(row.best_stage) > stage_rank(rec.get("best_stage")):
        rec["best_stage"], rec["best_stage_at"] = row.best_stage, row.best_stage_at
    if row.levels_at is not None and (rec.get("levels_at") is None or row.levels_at <= rec["levels_at"]):
        for k in ("entry", "sl", "tp", "rr", "levels_at", "outcome", "outcome_at", "entry_touched_at",
                  "tracked_until"):
            rec[k] = getattr(row, k)
        rec["shadow"] = _shadow_load(row.shadow) or _shadow_init(rec)
    for k in ("deleted_reason", "deleted_at", "crt_bar_time", "market_type"):
        if rec.get(k) is None and getattr(row, k) is not None:
            rec[k] = getattr(row, k)
    if rec.get("outcome") != "signal" and row.outcome == "signal":
        rec["outcome"] = "signal"


async def ensure_loaded(session, store=None, symbol_resolver=None) -> None:
    """Ilk cagrida son KEEP gunun kayitlarini yukle, izlenenleri store mumlariyla telafi et."""
    if _STATE["loaded"]:
        return
    _STATE["loaded"] = True
    try:
        since = _now() - KEEP
        rows = (await session.execute(select(SetupJournal).where(SetupJournal.last_seen >= since))).scalars().all()
        for row in rows:
            key = _key(row.strategy, row.symbol, row.direction, row.purge_time)
            rec = _CACHE.get(key)
            if rec is None:
                rec = {c: getattr(row, c) for c in _COLUMNS}
                rec.update(id=row.id, strategy=row.strategy, symbol=row.symbol, direction=row.direction,
                           purge_time=row.purge_time, _flushed_seen=row.last_seen)
                rec["shadow"] = _shadow_load(row.shadow)
                _CACHE[key] = rec
            else:
                _merge_row(rec, row)
                _DIRTY.add(key)
        caught = 0
        if store is not None and symbol_resolver is not None:
            for key, rec in _CACHE.items():
                if rec.get("levels_at") is None or (
                    rec.get("outcome") not in TRACKING and not _shadow_active(rec)
                ):
                    continue
                try:
                    bingx = symbol_resolver(rec["symbol"])
                    df = store.get_df(bingx, LTF_OF.get(rec["strategy"], "15m"))
                except Exception:
                    continue
                if df is None or len(df) < 2:
                    continue
                for ts, bar in df.iloc[:-1].iterrows():      # son satir forming
                    bar_ts = _naive_utc(ts)
                    hit = False
                    if _is_tracking(rec, bar_ts):
                        _apply_bar(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if _shadow_active(rec, bar_ts):
                        _apply_shadow(rec, bar_ts, float(bar["high"]), float(bar["low"]))
                        hit = True
                    if hit:
                        _DIRTY.add(key)
                        caught += 1
        log.info("Setup journal yuklendi: %d kayit, %d telafi mumu.", len(rows), caught)
    except Exception:
        log.exception("setup_journal.ensure_loaded failed")


async def flush(session) -> None:
    """Degisen kayitlari tek commit'te yaz."""
    if not _DIRTY:
        return
    keys = list(_DIRTY)
    _DIRTY.clear()
    try:
        for key in keys:
            rec = _CACHE.get(key)
            if rec is None:
                continue
            row = await session.get(SetupJournal, rec["id"]) if rec.get("id") else None
            if row is None:
                row = (await session.execute(select(SetupJournal).where(
                    SetupJournal.strategy == key[0], SetupJournal.symbol == key[1],
                    SetupJournal.direction == key[2], SetupJournal.purge_time == key[3],
                ))).scalars().first()
                if row is not None:
                    _merge_row(rec, row)
            if row is None:
                row = SetupJournal(strategy=key[0], symbol=key[1], direction=key[2], purge_time=key[3])
                session.add(row)
            for c in _COLUMNS:
                setattr(row, c, rec.get(c))
            row.shadow = _shadow_dump(rec.get("shadow"))
            rec["_flushed_seen"] = rec.get("last_seen")
        await session.commit()
        for key in keys:
            rec = _CACHE.get(key)
            if rec is not None and rec.get("id") is None:
                row = (await session.execute(select(SetupJournal.id).where(
                    SetupJournal.strategy == key[0], SetupJournal.symbol == key[1],
                    SetupJournal.direction == key[2], SetupJournal.purge_time == key[3],
                ))).scalar()
                rec["id"] = row
        _prune()
    except Exception:
        _DIRTY.update(keys)
        try:
            await session.rollback()
        except Exception:
            pass
        log.exception("setup_journal.flush failed (%d kayit tekrar denenecek)", len(keys))


def _prune() -> None:
    cutoff = _now() - KEEP
    for key in [k for k, r in _CACHE.items()
                if r.get("last_seen") and r["last_seen"] < cutoff and r.get("outcome") not in TRACKING
                and not _shadow_active(r) and k not in _DIRTY]:
        _CACHE.pop(key, None)
