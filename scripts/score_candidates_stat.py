"""Skor tablosu YENI ADAYLAR (21.10, kullanici): bugun skorda olmayan kalemler puan almali mi? -- strateji bazinda.

Adaylar `app/score_outcome.CANDIDATES`: seans (London / NY AM), CISD gucu, purge yakininda likidite, puansiz key level
(EQ / SSL), MSS kirilimi, C1 araligi buyuk, key level yok, C1 TP fitili, bolge yok. Puanlar ve esikler ilk okumadan
ONCE yazildi; okumada ayar yapilmaz. Filtre sorusu (>= 0.3 R) adayin eski maddesinde kalir, PUAN karari burada.

Kume: `score_outcome.collect` (motorun CRT saydigi, 4H C2 kapali donmus, 1D/1W seviyesi kesin, skor `analysis_parts`,
sonuc duz TP/SL; sinyalde golge k=1) + RR >= 2. KARAR BANDI: adayin puani sinyali degistirecek setuplar (odulde bugunku
raw [esik - v, esik - 1], cezada [esik, esik - v - 1]); ayni bantta aday var / yok.

KARAR KURALI (21.10 veri gorulmeden; 29.10 olcu WR'ye gecti, kullanici -- hicbir aday tetige varmadan;
IZLEME.md "Skor tablosu yeni adaylar — 4H / 1D / 1W"):
  tetik     : karar bandinda taraf basina 30 cozulmus
  PUAN ADAYI: lehte WR farki >= 5 puan + iki zaman yarisinda ayni yon + tek yonlu iki oran z-testi p < 0.10 -> tek replay
  RED       : lehte fark < 5 puan (ters dahil) -> aday o stratejide kapanir
  UZAT      : fark >= 5 puan ama yari / p tutmuyor -> taraf basina 60; 60'ta da tutmuyorsa RED
"Lehte": odulde aday VAR olan daha iyi, cezada aday var olan daha KOTU. WR = TP once mi SL once mi (giristen bagimsiz).
Neden WR: R/setup dolmadan TP'yi tam RR'yle sayiyor (TP'lerin ~%60'i), RR'si yuksek grup kendiliginden iyi gorunuyordu;
eski kuralin p'si ise dolum bazli R'dendi (iki ayri olcu). R (TP once / dolum bazli) yalniz bilgi olarak basilir.

Kullanim: python scripts/score_candidates_stat.py [--strategy 4h|1d|1w] [--since 2026-09-29] [--db traderadar.db]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app.score_outcome import (  # noqa: E402
    BANDS, CAND_DECIDED, CAND_MIN_RR, FILTER_GAP_WR, FILTER_HERE, CAND_MIN_SIDE, CANDIDATES, P_MAX, THR, WR_BAND, cand_flip,
    candidates_for, collect, stats, wr_test,
)

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--strategy", choices=("4h", "1d", "1w"), default="4h")
ap.add_argument("--since", default="2026-09-29", help="key level / C1 / likidite olcumleri 29.09'dan beri")
A = ap.parse_args()


def _fill_r(sub_) -> str:
    """Bilgi: dolum bazli R/islem (dolmadan TP haric)."""
    f = [x for x in sub_ if x["o"] in ("win", "loss")]
    if not f:
        return "-"
    r = (sum(x["rr"] for x in f if x["o"] == "win") - sum(x["o"] == "loss" for x in f)) / len(f)
    return f"{r:+.2f} (n {len(f)})"


def _fmt(st):
    if not st["res"]:
        return f"n{st['n']:4d} çözülmüş 0"
    return f"n{st['n']:4d} TP {st['w']:3d} SL {st['l']:3d} WR %{st['wr']:4.1f}"


con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute("select * from setup_journal where strategy = ? and first_seen >= ?", (A.strategy, A.since)).fetchall()
sub = collect(rows, A.strategy)
thr = THR[A.strategy]
ts = sorted(str(x["first_seen"]) for x in sub)
mid = ts[len(ts) // 2] if ts else ""
print(f"Strateji {A.strategy} · eşik {thr} · küme {len(sub)} setup (RR ≥ {CAND_MIN_RR:g} karar bandında) · "
      f"{A.since} → {ts[-1][:10] if ts else '-'} · yarı sınırı {mid[:10]}")
print(f"Tetik: karar bandında taraf başına {CAND_MIN_SIDE} çözülmüş · hüküm WR farkı ≥ {WR_BAND:g} puan + iki yarı + p < {P_MAX:g}\n")

for key, label, v, src, filt in candidates_for(A.strategy):   # 24.10: 1W-only adaylar (CAND_ONLY)
    g, b, d, (lo, hi), band = cand_flip(sub, key, thr)
    sign = 1 if v > 0 else -1
    measured = sum(1 for x in sub if x["cand"].get(key) is not None)
    have = sum(1 for x in sub if x["cand"].get(key) == 1)
    print(f"■ {label}  [{v:+d}]  · kaynak {src}" + (f" · filtre kararı: \"{filt}\"" if filt else ""))
    print(f"   ölçülen {measured} / {len(sub)} setup, adayı taşıyan {have}")
    print(f"   karar bandı (bugünkü raw {lo}–{hi}):  var {_fmt(g)}  |  yok {_fmt(b)}")
    # Bilgi (karar degil): butun skor bantlarinda ayni bantta var - yok, var olanin cozulmusuyle agirlikli.
    num = wt = 0.0
    for blo, bhi in BANDS:
        bb = [x for x in sub if x["rr"] >= CAND_MIN_RR and x["cand"].get(key) is not None
              and blo <= (x["pal"].get("raw") or 0) <= bhi]
        bg = stats([x for x in bb if x["cand"][key] == 1])
        bn = stats([x for x in bb if x["cand"][key] == 0])
        if bg["res"] and bn["res"]:
            num += bg["res"] * (bg["wr"] - bn["wr"])
            wt += bg["res"]
    if wt:
        print(f"   bilgi — tüm bantlar (0–4 / 5–6 / 7+), lehte WR farkı {(1 if v > 0 else -1) * num / wt:+.1f} puan "
              f"(aday çözülmüş {int(wt)}) — karar değil")
    wd, p_hi, p_lo = wr_test(g, b)
    if wd is None:
        print("   HÜKÜM: veri yok\n")
        continue
    good, p = sign * wd, (p_hi if v > 0 else p_lo)
    halves = []
    for part in ([x for x in band if str(x["first_seen"]) < mid], [x for x in band if str(x["first_seen"]) >= mid]):
        pg = stats([x for x in part if x["cand"][key] == 1])
        pb = stats([x for x in part if x["cand"][key] == 0])
        halves.append(sign * (pg["wr"] - pb["wr"]) if pg["res"] and pb["res"] else None)
    h = " / ".join("—" if x is None else f"{x:+.1f}" for x in halves)
    print(f"   lehte WR farkı {good:+.1f} puan · yarılar {h} · p {p:.3f}")
    print(f"   bilgi (karar değil) — TP önce R/setup farkı {d:+.2f} · dolum bazlı R/işlem: var "
          f"{_fill_r([x for x in band if x['cand'][key] == 1])} | yok {_fill_r([x for x in band if x['cand'][key] == 0])}")
    side = min(g["res"], b["res"])
    if side < CAND_MIN_SIDE:
        verdict = f"veri birikiyor ({side}/{CAND_MIN_SIDE})"
    elif good < WR_BAND:
        verdict = "RED — puan hak etmiyor (bu stratejide kapanır)"
    elif all(x is not None and x > 0 for x in halves) and p < P_MAX:
        verdict = "PUAN ADAYI — önce tek replay"
        if key in FILTER_HERE and good >= FILTER_GAP_WR:
            verdict += f" · fark ≥ {FILTER_GAP_WR:g} puan: FİLTRE seçeneği de kullanıcıya (bölge şartı)"
    elif side >= 2 * CAND_MIN_SIDE:
        verdict = f"RED — {2 * CAND_MIN_SIDE}'ta da yarı / p tutmuyor"
    else:
        verdict = f"UZAT — fark var ama yarı / p tutmuyor (taraf başına {2 * CAND_MIN_SIDE})"
    print(f"   HÜKÜM: {verdict}\n")

_done = [(k, txt) for (k, st), txt in CAND_DECIDED.items() if st == A.strategy]
if _done:   # 28.10: karara baglanan adaylar yukarida hesaplanmaz
    _lab = {c[0]: c[1] for c in CANDIDATES}
    print("Karara bağlanan adaylar (bu stratejide listeden çıktı):")
    for k, txt in _done:
        print(f"   ✓ {_lab.get(k, k)} — {txt}")
