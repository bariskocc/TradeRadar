"""Dar stop kapisi -- strateji bazinda olcum (22.10, kullanici istegi).

Soru: `min_stop_range_mult` (1.0) kapisi hangi stratejide dogru calisiyor, ve esik strateji bazinda farkli
(dinamik) mi olmali? 29.09'da kapanan eski madde yalniz kapida ELENEN setuplara bakiyordu ve verinin cogu
4H'ti. Burada kume: kapiya KADAR butun kapilari gecmis setuplar (`best_stage` sirasi `tight_stop` ve
otesi), stop orani (`features.stop_range_mult`) esigin altinda olan "dar", ustunde olan "normal". Ikisi
ayni kurallarla izleniyor; sinyale donusende golge k=1 (duz TP/SL, `score_outcome.outcome_of`).

Rapor `scripts/tight_stop_stat.py` ve `/izleme` sayaci ayni fonksiyonlari kullanir (iki yerde mantik yok).
Karar kurali IZLEME.md "Dar stop kapisi — strateji bazinda (4H / 1D / 1W)".
"""
from __future__ import annotations

import random

from app.score_outcome import outcome_of
from app.setup_journal import PRE_SETUP_STAGES, _RANK, analysis_parts, levels_final

# Kapi esigi (scanner STRATEGY_CFG `min_stop_range_mult`; dort stratejide de 1.0).
GATE_MULT = 1.0
# Tetik: dar grupta cozulmus setup (4H'te dar setup bol, 1W'de haftada ~1 setup).
MIN_RES = {"4h": 30, "1d": 20, "1w": 10}
# Olcum baslangici: 4H/1D dar stoplu bolge atlanmasi (29.09), 1W %80 stop (01.10).
SINCE = {"4h": "2026-09-29", "1d": "2026-09-29", "1w": "2026-10-01"}
BAND = 0.15
BUCKETS = ((0.0, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 99.0))


def _get(row, k):
    try:
        return row[k]
    except (KeyError, IndexError):
        return None


def collect(rows, strategy: str, broad: bool = False) -> list[dict]:
    """Kapiya kadar gecmis, seviyesi kesin, stop orani olculmus setuplar.

    `broad` (bilgi): kapi sirasi aranmaz, yalniz RR >= 2 -- 1D/1W'de skor kapisini gecen setup cok az
    oldugu icin esigin kendisi hakkinda erken isaret verir; karar dar kumeden (`broad=False`).
    """
    gate = _RANK["tight_stop"]
    since = SINCE.get(strategy, "2026-09-29")
    out = []
    for r in rows:
        if _get(r, "strategy") != strategy or str(_get(r, "first_seen") or "") < since:
            continue
        st = _get(r, "best_stage")
        if st in PRE_SETUP_STAGES or _get(r, "entry") is None:
            continue
        if not broad and _RANK.get(st, -1) < gate:
            continue
        if not levels_final(r):
            continue
        _pal, fe = analysis_parts(r)
        m = (fe or {}).get("stop_range_mult")
        if m is None:
            continue
        o, rr = outcome_of(r)
        if broad and float(rr or 0.0) < 2.0:
            continue
        out.append(dict(m=float(m), o=o, rr=float(rr or 0.0), dar=float(m) < GATE_MULT,
                        first_seen=str(_get(r, "first_seen")), stage=st))
    return out


def stats(sub) -> dict:
    w = [x for x in sub if x["o"] == "win"]
    l_ = [x for x in sub if x["o"] == "loss"]
    res = len(w) + len(l_)
    return dict(n=len(sub), res=res, w=len(w), l=len(l_),
                rps=((sum(x["rr"] for x in w) - len(l_)) / res) if res else None)


def _r(x):
    return x["rr"] if x["o"] == "win" else -1.0


def p_dar_worse(dar, normal, it=4000, seed=7):
    """Tek yonlu bootstrap: dar grubun normalden KOTU OLMAMA olasiligi (kucuk = kapi koruyucu)."""
    g = [_r(x) for x in dar if x["o"] in ("win", "loss")]
    b = [_r(x) for x in normal if x["o"] in ("win", "loss")]
    if not g or not b:
        return None
    rnd = random.Random(seed)
    n = 0
    for _ in range(it):
        mg = sum(rnd.choice(g) for _ in g) / len(g)
        mb = sum(rnd.choice(b) for _ in b) / len(b)
        n += mg - mb >= 0
    return n / it


def summary(sub, strategy: str) -> dict:
    """Tek strateji satiri: dar / normal / fark, yarilar, p ve karar."""
    dar = [x for x in sub if x["dar"]]
    nor = [x for x in sub if not x["dar"]]
    sd, sn = stats(dar), stats(nor)
    diff = (sd["rps"] - sn["rps"]) if sd["rps"] is not None and sn["rps"] is not None else None
    halves = []
    if sub:
        cut = sorted(x["first_seen"] for x in sub)[len(sub) // 2]
        for part in ([x for x in sub if x["first_seen"] < cut], [x for x in sub if x["first_seen"] >= cut]):
            a, b = stats([x for x in part if x["dar"]]), stats([x for x in part if not x["dar"]])
            halves.append((a["rps"] - b["rps"]) if a["rps"] is not None and b["rps"] is not None else None)
    p = p_dar_worse(dar, nor)
    need = MIN_RES.get(strategy, 20)
    if sd["res"] < need:
        verdict = f"VERI BIRIKIYOR (dar cozulmus {sd['res']}/{need})"
    elif diff is not None and diff <= -BAND and all(h is not None and h < 0 for h in halves) \
            and p is not None and p < 0.10:
        verdict = "KAPI DOGRU — kalir"
    elif diff is not None and diff > -BAND and (sd["rps"] or 0) > 0:
        verdict = "KAPI KAZANC KESIYOR — tek replay, sonra kullaniciya (kaldir / esigi dusur)"
    else:
        verdict = f"KARARSIZ — dar grupta {2 * need} cozulmuse uzat"
    return dict(dar=sd, normal=sn, diff=diff, halves=halves, p=p, need=need, verdict=verdict)


def buckets(sub) -> list[tuple[str, dict]]:
    """Dinamik esik icin stop orani kovalari (bilgi)."""
    return [(f"{lo:.2f}-{hi:.2f}" if hi < 99 else f">= {lo:.2f}", stats([x for x in sub if lo <= x["m"] < hi]))
            for lo, hi in BUCKETS]
