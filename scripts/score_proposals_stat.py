"""Skor onerileri -- "su kalemi cikar / su puani degistir" Journal karsi-olgusu (03.10, kullanici).

Her oneri skoru yeniden hesaplar ve kumeyi ayni statik kapilardan gecirir; bugunku kural (A @7) ile karsilastirir.
Replay DEGIL: portfoy kapilari (kume limiti / acik islem / korele) ve kismi kar/BE yok, duz TP/SL. Indirme yok.
Yontem 28.09 paket karsi-olgusuyla ayni (`tmp/tmp_score_weight_cf.py`).

Kume: Journal, strateji (varsayilan 4H), parts_at_levels dolu, 4H'te C2 kapali donmus, best_stage skor kapisinda ya da
otesinde (CRT-oncesi, bias_mismatch, target_taken haric). Statik kapilar herkese ayni: donmus RR >= 2,
stop_range_mult >= 1.0, skor tam 7'de skor-7 kapisi (giris bolge degil ve key level var -- key level VARLIGI, puani degil).
Sonuc: elenen setupta Journal outcome, sinyale donende golge shadow["1"]; win = +RR, loss = -1, tp_before_entry/no_touch 0.
Karsi-olgu skoru: min(11, max(0, min(9, raw - smt + delta)) + smt') -- tavan SMT'den once (tavan tuzagi, CLAUDE.md).

Karar kurali: IZLEME.md "Skor önerileri — 4H" (03.10, veri gorulmeden; taze veri --since 2026-10-04).
Kullanim: python scripts/score_proposals_stat.py [--strategy 4h] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.setup_journal import analysis_parts, levels_final  # noqa: E402

ALL_LEVELS = "--all-levels" in sys.argv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent

EARLY = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "c2_second_sweep", "not_selected",
         "bias_mismatch", "target_taken"}
MIN_RR, TIGHT = 2.0, 1.0
# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler (03.10) ---
DECISION_SINCE = "2026-10-04"
MIN_A = 60            # A kolunun (bugunku kural) taze setup sayisi
# strateji -> (levels_at alt siniri, A kolu esigi). 1D (03.10): bugunku kural 3 haftada 9 setup aciyor, taze veriyle 60
# ~20 hafta -> 25.09'dan (1D skor kurallari degisti) butun veri, A 30; esikler 4H'ten kopyalandi, 1D verisine bakilmadan.
DECISION = {"4h": (DECISION_SINCE, MIN_A), "1d": ("2026-09-25", 30)}
MIN_GAIN_R = 3.0      # oneri A'ya gore toplam R'de en az bu kadar iyi
THR = {"4h": 7, "1d": 7, "1w": 6}


def _j(s):
    try:
        d = json.loads(s) if s else None
        return d if isinstance(d, dict) else None
    except Exception:
        return None


# Oneri: (aciklama, fonksiyon(p, f) -> (delta_nonsmt, smt_yeni), esik farki)
def _smt(p):
    return p.get("smt", 0)


PROPOSALS = {
    # 03.10 (kullanici): SMT onerileri ASKIDA -- "SMT onemli konfirme, cikarmak yerine duzeltelim"; tanim duzeltmesi
    # `scripts/smt_variants_stat.py`. Kollar bilgi icin basilir, KARAR blogu onlari aday saymaz.
    "SMT +2 -> +1 (ASKIDA)": lambda p, f: (0, min(_smt(p), 1)),
    "SMT +2 -> 0 (ASKIDA)": lambda p, f: (0, 0),
    "Key level FVG/OB +1 -> 0": lambda p, f: (-p.get("pd_struct", 0), _smt(p)),
    "Key level major +1 -> +2": lambda p, f: (p.get("pd_major", 0), _smt(p)),
    "C2 rengi: doji +2 / dolu +1 (yer degistir)": lambda p, f: (
        (1 if p.get("base") == 1 else -1 if p.get("base") == 2 else 0), _smt(p)),
    # 21.10: C1 araligi puan KARARI "Skor tablosu yeni adaylar" maddelerinde (aday c1_big, karar bandi); burada bilgi.
    "C1 araligi/ATR >= 1.3 -> -1": lambda p, f: (
        (-1 if isinstance(f.get("range_atr"), (int, float)) and f["range_atr"] >= 1.3 else 0), _smt(p)),
    "1W uyumu +1 -> 0": lambda p, f: (-p.get("weekly", 0), _smt(p)),
    "LTF IFVG +1 -> 0": lambda p, f: (-p.get("ifvg", 0), _smt(p)),
    "Wick esigi 0.10 -> 0.30": lambda p, f: (
        (-p.get("wick", 0) if p.get("wick", 0) > 0 and isinstance(f.get("wick_frac"), (int, float))
         and f["wick_frac"] < 0.30 else 0), _smt(p)),
    # 03.10: puan cikarmak butun skorlari dusurur -> esik de duser. Gorulen veride SMT 0 @6: setup sayisi bugunku kadar
    # (57 / 50), R/setup -0.05 / -0.18, iki yarida da arti; birlesik "sade paket" (SMT+1W+IFVG 0, C1 araligi -1) SMT'den iyi degil.
    "SMT +2 -> 0, esik 7 -> 6 (ASKIDA)": (lambda p, f: (0, 0), 6),
}


def load(db, strategy, since, until):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    recs = []
    for r in con.execute("select * from setup_journal where strategy=? and parts_at_levels is not null", (strategy,)):
        if r["best_stage"] in EARLY:
            continue
        if strategy != "4h" and not ALL_LEVELS and not levels_final(r):  # 03.10: 1D/1W'de yalniz seviyesi kesin satir (C2 acik donmus = kosan SL, SL yonune yanli); --all-levels eskiler dahil
            continue
        if strategy == "4h" and (_j(r["parts_at_levels"]) or {}).get("c2_closed") != 1:
            continue
        p, f = analysis_parts(r)   # 03.10: 1D/1W'de skor sinyal / C2 kapanisi anindan
        if not p or "raw" not in p:
            continue
        t = datetime.fromisoformat(str(r["levels_at"] or r["first_seen"])[:19])
        if (since and t < datetime.fromisoformat(since)) or (until and t >= datetime.fromisoformat(until)):
            continue
        rr, o = r["rr"], r["outcome"]
        if o == "signal":
            sh = (_j(r["shadow"]) or {}).get("1") or {}
            o, rr = sh.get("o"), sh.get("rr", rr)
        if o not in ("win", "loss", "tp_before_entry", "no_touch"):
            continue
        if rr is None or rr < MIN_RR or (f.get("stop_range_mult") is not None and f["stop_range_mult"] < TIGHT):
            continue
        pd_any = (p.get("pd_major", 0) + p.get("pd_monthly", 0) + p.get("pd_struct", 0)) > 0 \
            or (f.get("key_level_n") or 0) > 0
        recs.append(dict(p=p, f=f, t=t, o=o, R=rr if o == "win" else (-1.0 if o == "loss" else 0.0),
                         zone=(r["model"] or "cisd") in ("ifvg", "bpr"), pd_any=pd_any))
    return recs


def score(p, delta, smt_new):
    return min(11, max(0, min(9, p["raw"] - _smt(p) + delta)) + smt_new)


def passes(rec, sc, thr):
    if sc < thr:
        return False
    if sc == 7:   # skor-7 kapisi motorda skor TAM 7'de
        return (not rec["zone"]) and rec["pd_any"]
    return True


def arm(recs, fn, thr):
    out = []
    for x in recs:
        if fn is None:
            sc = x["p"]["score"]
        else:
            d, s2 = fn(x["p"], x["f"])
            sc = score(x["p"], d, s2)
        if passes(x, sc, thr):
            out.append(x)
    return out


def tally(rs):
    return len(rs), sum(x["o"] == "win" for x in rs), sum(x["o"] == "loss" for x in rs), sum(x["R"] for x in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="4h", choices=("4h", "1d", "1w"))
    ap.add_argument("--since", default=None)
    ap.add_argument("--all-levels", action="store_true", help="1D/1W: C2 acikken donmus (yanli) eski satirlar da")
    ap.add_argument("--until", default=None)
    A = ap.parse_args()
    thr = THR[A.strategy]

    def report(recs, title):
        print(f"\n### {title} -- {len(recs)} setup")
        if not recs:
            return {}
        half = sorted(x["t"] for x in recs)[len(recs) // 2]
        h1 = [x for x in recs if x["t"] < half]
        h2 = [x for x in recs if x["t"] >= half]
        a = tally(arm(recs, None, thr))
        a1, a2 = tally(arm(h1, None, thr))[3], tally(arm(h2, None, thr))[3]
        print(f"   {'kol':<44}{'n':>4}{'TP':>4}{'SL':>4}{'R':>8}{'fark':>8}{'1.yari':>8}{'2.yari':>8}")
        print(f"   {'A  bugunku kural @' + str(thr):<44}{a[0]:>4}{a[1]:>4}{a[2]:>4}{a[3]:>+8.1f}")
        res = {}
        for name, spec in PROPOSALS.items():
            fn, t2 = spec if isinstance(spec, tuple) else (spec, thr)   # (fn, esik) -> esigi de degistiren oneri
            b = tally(arm(recs, fn, t2))
            d1 = tally(arm(h1, fn, t2))[3] - a1
            d2 = tally(arm(h2, fn, t2))[3] - a2
            res[name] = (b[3] - a[3], d1, d2)
            print(f"   {name:<44}{b[0]:>4}{b[1]:>4}{b[2]:>4}{b[3]:>+8.1f}{b[3] - a[3]:>+8.1f}{d1:>+8.1f}{d2:>+8.1f}")
        return dict(res, _a=a)

    recs = load(A.db, A.strategy, A.since, A.until)
    print(f"Strateji {A.strategy} · esik {thr} · levels_at {A.since or 'hepsi'} .. {A.until or 'bugun'}")
    report(recs, "hepsi (gorulen veri -- KARAR DEGIL)")

    if A.strategy not in DECISION:
        print("\nKARAR: bu strateji icin kural yok (4H ve 1D); bilgi amacli")
        return
    since, min_a = DECISION[A.strategy]
    print("\n" + "=" * 92 + f"\nKARAR (03.10 kurali; {A.strategy}, levels_at >= {since}, A kolu >= {min_a} setup)\n"
          + "=" * 92)
    dec = load(A.db, A.strategy, since, None)
    res = report(dec, "taze veri")
    na = res.get("_a", (0,))[0]
    if na < min_a:
        print(f"   A kolu {na}/{min_a} -> veri birikiyor")
        return
    for name, (d, d1, d2) in res.items():
        if name == "_a" or "ASKIDA" in name:
            continue
        if d >= MIN_GAIN_R and d1 > 0 and d2 > 0:
            print(f"   ADAY: {name} ({d:+.1f}R, yarilar {d1:+.1f} / {d2:+.1f}) -> tek replay (portfoy + kismi kar)")
        elif d <= -MIN_GAIN_R and d1 < 0 and d2 < 0:
            print(f"   RED : {name} ({d:+.1f}R) -- bugunku puan daha iyi")
        else:
            print(f"   yok : {name} ({d:+.1f}R, yarilar {d1:+.1f} / {d2:+.1f})")


if __name__ == "__main__":
    main()
