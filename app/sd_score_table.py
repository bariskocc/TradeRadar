"""S/D Skor Tablosu (04.11, kullanici: "S/D icin ayri score table"). Arayuz Turkce (CRT Skor Tablosu gibi).

Strateji (sd / sd1h / sd15m) basina: skor kalemleri (puan + kural, motor sabitlerinden) ve her kalem degerinin Journal sonucu
(cozulmus = win / loss, dolum sart), kapilar (kac bolge eledi, elenenler sonra TP mi SL mi). Hukumler: "Yetersiz veri" (taraf < 10),
"Ayiriyor" (WR farki >= 10 puan), "Ayirmiyor". Puan verilmeyen olcumler (BOS, sweep, bolge yuksekligi) de ayni tabloda "aday".
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import sd_engine as sde
from app.models import SetupJournal

TFS = [("sd", "4H"), ("sd1h", "1H"), ("sd15m", "15m")]
MIN_SIDE = 10
WR_GAP = 10

ITEMS = [
    ("strength", "Leg-out gücü", f"proximal'den ≥ {sde.LEG_OUT_MIN_ATR:g} ATR +1 · ≥ {sde.STRENGTH_STRONG_ATR:g} ATR +2", "+1 / +2"),
    ("base", "Base uzunluğu", "1 mum +2 · 2 mum +1 · 3–4 mum 0", "0 / +1 / +2"),
    ("fvg", "Leg-out FVG", "çıkış 3'lü boşluk bıraktı +1", "0 / +1"),
    ("fresh", "Tazelik", "0 dokunuş +2 · 1 dokunuş 0 (2+ ölü)", "0 / +2"),
    ("htf", "1D bias", "işlem yönünde +2 · ters −2 · NEUTRAL 0 (hard kapı yok, 04.11)", "−2 / 0 / +2"),
]
CANDIDATES = [
    ("sweep", "Likidite sweep'i (SSL / BSL)", "base asıl likiditeyi aldı: geniş swing (her yanda 5 mum), eşit dipler/tepeler ya da önceki gün ucu (05.11; yalnız ara swing sayılmaz)"),
    ("sweep_minor", "Yalnız ara swing sweep'i", "base yalnız küçük swing'i (her yanda 2 mum) aldı, asıl likidite alınmadı (05.11)"),
    ("mss", "MSS (dönüş kırılımı)", "dönüş bölgesi (DBR / RBD) ve leg-out base öncesi son swing'i kırdı (05.11)"),
    ("bos", "Kırılım (BOS + MSS)", "leg-out base öncesi son swing'i kırdı — bölge tipinden bağımsız"),
    ("pd_ok", "Premium / Discount uygun", f"demand son {sde.PD_RANGE_BARS} mumluk aralığın alt yarısında, supply üst yarısında (05.11)"),
    ("strong", "Çok sert çıkış (≥ 3 ATR)", "ilk okumada ters (05.11) — 'strength' +2'si buna veriliyor"),
]
GATES = [
    ("sd_dead", "Ölü bölge", "2+ dokunuş ya da distal kırıldı"),
    ("sd_bias", "1D bias ters", "KAPALI (04.11) — eski satırlar"),
    ("sd_no_target", "Hedef yok", "fiyatın ötesinde karşı bölge / alınmamış swing yok (son kapanıştan ≥ 0,5 ATR)"),
    ("sd_far", "Fiyata uzak", f"giriş > {sde.MAX_DIST_R:g}R uzak — seviye yazılmaz"),
    ("sd_low_rr", "RR düşük", f"RR < {sde.MIN_RR:g} (hedef en yakın engel, tavan {sde.MAX_RR:g}R)"),
    ("sd_near", "Yakın kaçan bölge", "çıkış gövdesi 0,8–1 ATR, kapıları geçti — ölçüm, Signals'a girmez (05.11)"),
    ("sd_ready", "Hazır (sinyal)", "bütün kapılar geçildi — seviye ve sonuç yalnız burada (04.11)"),
]


def _verdict(groups: dict[int, list[int]]) -> str:
    """Kalem degerleri arasinda WR farki: en dusuk ve en yuksek puanli taraflar, her biri >= MIN_SIDE cozulmus."""
    ok = {v: g for v, g in groups.items() if sum(g) >= MIN_SIDE}
    if len(ok) < 2:
        return "Yetersiz veri"
    lo, hi = min(ok), max(ok)
    wr_lo = ok[lo][0] / sum(ok[lo]); wr_hi = ok[hi][0] / sum(ok[hi])
    gap = round(100 * (wr_hi - wr_lo))
    if gap >= WR_GAP:
        return f"Ayırıyor (+{gap} puan)"
    if gap <= -WR_GAP:
        return f"Ters ({gap} puan)"
    return "Ayırmıyor"


async def build_sd_score_table(db: AsyncSession, tf: str) -> dict:
    tf = tf if tf in dict(TFS) else "sd"
    t = SetupJournal
    rows = (await db.execute(select(t).where(t.strategy == tf))).scalars().all()
    resolved = [r for r in rows if r.outcome in ("win", "loss")]
    item_rows = []
    for key, label, rule, pts in ITEMS + [(k, l, r_, "aday") for k, l, r_ in CANDIDATES]:
        groups: dict[int, list[int]] = {}
        for r in resolved:
            try:
                p = json.loads(r.parts_at_levels or r.score_parts or "{}")
                f = json.loads(r.features_at_levels or r.features or "{}")
            except Exception:
                continue
            if key == "strong":
                val = None if f.get("legout_atr") is None else int(float(f["legout_atr"]) >= sde.STRENGTH_STRONG_ATR)
            else:
                val = p.get(key) if key in ("strength", "base", "fvg", "fresh", "htf") else f.get(key)
            if val is None:
                continue
            g = groups.setdefault(int(val), [0, 0])
            g[0 if r.outcome == "win" else 1] += 1
        cells = [{"val": v, "tp": g[0], "sl": g[1], "wr": round(100 * g[0] / sum(g)) if sum(g) else None}
                 for v, g in sorted(groups.items())]
        item_rows.append({"key": key, "label": label, "rule": rule, "pts": pts, "cells": cells, "verdict": _verdict(groups),
                          "candidate": pts == "aday"})
    # toplam skor
    score_groups: dict[int, list[int]] = {}
    for r in resolved:
        try:
            p = json.loads(r.parts_at_levels or r.score_parts or "{}")
        except Exception:
            continue
        if p.get("score") is None:
            continue
        g = score_groups.setdefault(int(p["score"]), [0, 0])
        g[0 if r.outcome == "win" else 1] += 1
    lo_g = [sum(g[0] for v, g in score_groups.items() if v <= 3), sum(g[1] for v, g in score_groups.items() if v <= 3)]
    hi_g = [sum(g[0] for v, g in score_groups.items() if v >= 5), sum(g[1] for v, g in score_groups.items() if v >= 5)]
    total = {"cells": [{"val": v, "tp": g[0], "sl": g[1], "wr": round(100 * g[0] / sum(g))} for v, g in sorted(score_groups.items())],
             "low": lo_g, "high": hi_g, "verdict": _verdict({0: lo_g, 1: hi_g})}
    # kapilar
    gate_rows = []
    for code, label, rule in GATES:
        g = [r for r in rows if r.best_stage == code]
        w = sum(1 for r in g if r.outcome == "win"); l = sum(1 for r in g if r.outcome == "loss")
        gate_rows.append({"code": code, "label": label, "rule": rule, "n": len(g), "tp": w, "sl": l,
                          "wr": round(100 * w / (w + l)) if w + l else None})
    return {
        "tf": tf, "tfs": TFS, "tf_label": dict(TFS)[tf], "items": item_rows, "total": total, "gates": gate_rows,
        "n_rows": len(rows), "n_resolved": len(resolved),
        "plan": {"entry": "proximal kenar (limit); ölçümde %50 ve distal da izlenir", "sl": f"distal + {sde.SL_BUFFER_ATR:g} ATR, risk en az {sde.MIN_RISK_ATR:g} ATR (ATR = pariteye göre ölçek); gölgede ×0,67 / ×1,33, bölge stopu ve eski plan ({sde.OLD_MIN_RISK_ATR:g} ATR)",
                 "tp": f"en yakın engel (karşı bölge / alınmamış swing, son kapanıştan ≥ {sde.TARGET_MIN_BEYOND_ATR:g} ATR), engel RR < {sde.MIN_RR:g} ise TP = {sde.MIN_RR:g}R noktası, tavan {sde.MAX_RR:g}R; dolum öncesi güncellenir",
                 "base": f"≤ {sde.BASE_MAX_BARS} mum, gövde ≤ {sde.BASE_MAX_BODY_ATR:g} ATR", "legout": f"ilk mum gövdesi ≥ {sde.LEG_BODY_MIN_ATR:g} ATR, {sde.LEG_OUT_BARS} mumda ≥ {sde.LEG_OUT_MIN_ATR:g} ATR",
                 "age": f"base ≤ {sde.MAX_AGE_BARS} mum", "touch": f"dokunuş koşusu {sde.TOUCH_EXIT_ATR:g} ATR uzaklaşınca biter; 2+ dokunuş ölü"},
    }
