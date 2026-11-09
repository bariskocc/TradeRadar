"""S/D Skor Tablosu (04.11, kullanici: "S/D icin ayri score table"). Arayuz Turkce (CRT Skor Tablosu gibi).

Strateji (sd / sd1h / sd15m) basina: skor kalemleri (puan + kural, motor sabitlerinden) ve her kalem degerinin Journal sonucu
(cozulmus = win / loss, dolum sart), kapilar (kac bolge eledi, elenenler sonra TP mi SL mi). Hukumler: "Yetersiz veri" (taraf < 10),
"Ayiriyor" (WR farki >= 10 puan), "Ayirmiyor". Puan verilmeyen olcumler (BOS, sweep, bolge yuksekligi) de ayni tabloda "aday".
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import sd_confirm as sdc
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
    ("pd_ok_sw", "Premium / Discount (swing aralığı)", "aynı kural, aralık = base öncesi son geniş swing tepe-dip (her yanda 5 mum) — 07.11"),
    ("ltf_disp", "LTF displacement (yalnız 4H / 1H)", f"CHOCH mumunun gövdesi ≥ {sdc.CHOCH_DISP_MIN:g} × ort. LTF mum boyu — etiket, 07.11"),
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
# 4H / 1H LTF onay kolunun kapisi (07.11): satirin asamasi degil, entries.sdc sonucu. Elenenlerin sonrasi bolge limitinden.
CONFIRM_GATES = [
    ("no_sweep", "LTF sweep yok (4H / 1H)", "bölgenin altındaki LTF likiditesi alınmadan CHOCH aranmaz — elenenlerin bölge limiti sonucu (07.11)"),
    ("failed", "CHOCH'tan önce stop (4H / 1H)", "bölge stopu CHOCH gelmeden yendi — işlem yok"),
]


def _ents(r) -> dict:
    try:
        return json.loads(r.entries or "{}") or {}
    except Exception:
        return {}


def _result(r) -> str | None:
    """Kalem okumasinin sonucu = ANA ISLEM (07.11): 15m satirin bolge limiti; 4H / 1H LTF onay kolu (entries.sdc, 05.11'den beri ana
    giris). Hafta kapanisi R'nin isaretinden kazanc / kayip (S/D Signals kurali). Islem acilmadiysa None."""
    en = _ents(r)
    if r.strategy in sdc.STRATEGIES:
        st = en.get(sdc.SDC_KEY) or {}
        o = st.get("o")
        if o in ("win", "loss"):
            return o
        if o == "week_close":
            v = sdc.final_r(st)
            return None if not v else ("win" if v > 0 else "loss")
        return None
    if r.outcome in ("win", "loss"):
        return r.outcome
    if r.outcome == "week_close":
        v = sdc.week_close_r(r.entry, r.sl, (en.get("_wc") or {}).get("px"), r.direction)
        return None if not v else ("win" if v > 0 else "loss")
    return None


def _plan(tf: str) -> dict:
    """Sayfanin ust ozeti -- motor sabitlerinden (07.11 kurallari)."""
    tp = (f"en yakın engel (karşı bölge / alınmamış swing, son kapanıştan ≥ {sde.TARGET_MIN_BEYOND_ATR:g} ATR); RR < {sde.FLOOR_RR_MIN:g} "
          f"veren ara engel atlanır, ilk engel RR {sde.FLOOR_RR_MIN:g}–{sde.MIN_RR:g} ise TP = {sde.MIN_RR:g}R noktası; tavan {sde.MAX_RR:g}R"
          + (" (işlemin kendi riskiyle)" if tf in sdc.STRATEGIES else "")
          + "; yapısal hedef gölgede izlenir; dolum öncesi güncellenir")
    if tf in sdc.STRATEGIES:
        entry = (f"bölgeye dokunuş = uyarı (bölge ≥ {sde.MID_ENTRY_MIN_ATR:g} ATR ise orta noktaya); LTF'de bölgenin altındaki likidite (5'li dip / eşit dipler / base dibi) alınmalı → CHOCH "
                 "(son alınan dip ile sweep dibi arası tepe, kapanışla) → retest'te FVG / IFVG / CISD / MSS (RR ≥ 2, ilk dolan); "
                 "bölge limiti ölçüm kolu")
        sl = (f"bölge planı stopu: distal + {sde.SL_BUFFER_ATR:g} ATR, risk en az {sde.MIN_RISK_ATR:g} ATR (bölge kenarından); "
              "gölgede dip stopu, +1R'de BE, %50'de BE")
    else:
        entry = (f"proximal kenar = base'in karşı fitili (fitil-fitil, limit; 09.11); bölge ≥ {sde.MID_ENTRY_MIN_ATR:g} ATR ise orta nokta "
                 "(stop yerinde); ölçümde %50, distal ve eski gövde kenarı da izlenir")
        sl = (f"distal + {sde.SL_BUFFER_ATR:g} ATR, risk en az {sde.MIN_RISK_ATR:g} ATR (ATR = pariteye göre ölçek); gölgede ×0,67 / ×1,33, "
              f"bölge stopu ve eski plan ({sde.OLD_MIN_RISK_ATR:g} ATR)")
    return {"entry": entry, "sl": sl, "tp": tp,
            "base": f"≤ {sde.BASE_MAX_BARS} mum, gövde ≤ {sde.BASE_MAX_BODY_ATR:g} ATR", "legout": f"ilk mum gövdesi ≥ {sde.LEG_BODY_MIN_ATR:g} ATR, {sde.LEG_OUT_BARS} mumda ≥ {sde.LEG_OUT_MIN_ATR:g} ATR",
            "age": f"base ≤ {sde.MAX_AGE_BARS} mum", "touch": f"dokunuş koşusu {sde.TOUCH_EXIT_ATR:g} ATR uzaklaşınca biter; 2+ dokunuş ölü",
            "result": ("4H / 1H: sonuç LTF onay işleminden (ana giriş); hafta kapanışı R işaretiyle" if tf in sdc.STRATEGIES
                       else "sonuç bölge limitinden (ana giriş); hafta kapanışı R işaretiyle")}


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
    resolved = [(r, res) for r in rows if (res := _result(r)) is not None]
    item_rows = []
    for key, label, rule, pts in ITEMS + [(k, l, r_, "aday") for k, l, r_ in CANDIDATES]:
        if key == "ltf_disp" and tf not in sdc.STRATEGIES:
            continue                              # 15m'de LTF onay kolu yok
        groups: dict[int, list[int]] = {}
        for r, res in resolved:
            try:
                p = json.loads(r.parts_at_levels or r.score_parts or "{}")
                f = json.loads(r.features_at_levels or r.features or "{}")
            except Exception:
                continue
            if key == "strong":
                val = None if f.get("legout_atr") is None else int(float(f["legout_atr"]) >= sde.STRENGTH_STRONG_ATR)
            elif key == "ltf_disp":
                st = _ents(r).get(sdc.SDC_KEY) or {}
                val = None if st.get("bdisp") is None else int(sdc.is_disp(st))
            else:
                val = p.get(key) if key in ("strength", "base", "fvg", "fresh", "htf") else f.get(key)
            if val is None:
                continue
            g = groups.setdefault(int(val), [0, 0])
            g[0 if res == "win" else 1] += 1
        cells = [{"val": v, "tp": g[0], "sl": g[1], "wr": round(100 * g[0] / sum(g)) if sum(g) else None}
                 for v, g in sorted(groups.items())]
        item_rows.append({"key": key, "label": label, "rule": rule, "pts": pts, "cells": cells, "verdict": _verdict(groups),
                          "candidate": pts == "aday"})
    # toplam skor
    score_groups: dict[int, list[int]] = {}
    for r, res in resolved:
        try:
            p = json.loads(r.parts_at_levels or r.score_parts or "{}")
        except Exception:
            continue
        if p.get("score") is None:
            continue
        g = score_groups.setdefault(int(p["score"]), [0, 0])
        g[0 if res == "win" else 1] += 1
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
    if tf in sdc.STRATEGIES:
        for code, label, rule in CONFIRM_GATES:
            g = [r for r in rows if (_ents(r).get(sdc.SDC_KEY) or {}).get("o") == code]
            w = sum(1 for r in g if r.outcome == "win"); l = sum(1 for r in g if r.outcome == "loss")
            gate_rows.append({"code": code, "label": label, "rule": rule, "n": len(g), "tp": w, "sl": l,
                              "wr": round(100 * w / (w + l)) if w + l else None})
    return {
        "tf": tf, "tfs": TFS, "tf_label": dict(TFS)[tf], "items": item_rows, "total": total, "gates": gate_rows,
        "n_rows": len(rows), "n_resolved": len(resolved),
        "plan": _plan(tf),
    }
