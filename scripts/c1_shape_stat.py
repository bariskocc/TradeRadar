"""C1'in sekli sonucu etkiliyor mu? (Setup Journal'dan, terminal)

Soru (29.09, SEI 4H #133): motor C1'in rengine ve seklina hic bakmiyor. SEI'de C1 yesil bir
yukselis mumuydu, uzun ust fitilli; LONG'un TP'si o fitilin tepesiydi ve setup 1.6 saatte SL oldu.
Iki hipotez:
  H1  C1 ile C2 AYNI RENK olan setup, zit renkliden kotu.
  H2  C1'in TP tarafindaki fitili UZUN olan setup (hedef zaten reddedilmis tepe/dip) kotu.

Olcum: `features.c1_same_color`, `c1_body_frac`, `c1_tp_wick_frac` (29.09'dan beri; oncesinde yok).
Kume: motorun CRT saydigi setuplar (`not_selected` dahil -- gecerli CRT sekli), duz TP/SL sonucu
(win = entry -> TP, loss = entry -> SL). Gercek R degil, gruplar arasi fark okunur.

Karar kurali (onceden yazildi, IZLEME.md "C1 sekli"; 29.09 key level / purge otesi likidite maddeleriyle
ayni kalip): her grupta >= 20 cozulmus setup; "kotu" grup digerinden
  - >= 0.30 R/setup kotuyse     -> hard filtre adayi
  - 0.15-0.30 R/setup kotuyse   -> -2 skor cezasi adayi
  - < 0.15                      -> degisiklik yok
Iki degisiklik de once tek replay'den gecer. Zaman yarilari bilgi icin basilir (tutarsizsa kullaniciya sor).

Kullanim:
    python scripts/c1_shape_stat.py                  # tum stratejiler
    python scripts/c1_shape_stat.py --strategy 4h
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.setup_journal import PRE_SETUP_STAGES  # noqa: E402

NOT_CRT = tuple(PRE_SETUP_STAGES - {"not_selected"})
MIN_N = 20
LONG_TP_WICK = 0.40   # "uzun TP fitili" esigi; ilk bakistan ONCE kondu (SEI: 0.64)


def _load(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _r(row) -> float:
    return float(row["rr"] or 0.0) if row["outcome"] == "win" else -1.0


def _stat(sub):
    n = len(sub)
    if not n:
        return n, None, None
    w = sum(1 for r in sub if r["outcome"] == "win")
    return n, 100.0 * w / n, sum(_r(r) for r in sub) / n


def _cell(sub) -> str:
    n, wr, rs = _stat(sub)
    return "n=   0" if not n else f"n={n:4d}  win {wr:4.1f}%  R/setup {rs:+.3f}"


def _compare(title: str, bad, good, label_bad: str, label_good: str) -> None:
    print(f"\n== {title}")
    print(f"  {label_bad:26s} {_cell(bad)}")
    print(f"  {label_good:26s} {_cell(good)}")
    halves = []
    allr = sorted(bad + good, key=lambda r: r["first_seen"])
    if allr:
        cut = allr[len(allr) // 2]["first_seen"]
        for name, pred in (("1. yari", lambda r: r["first_seen"] < cut), ("2. yari", lambda r: r["first_seen"] >= cut)):
            b = [r for r in bad if pred(r)]
            g = [r for r in good if pred(r)]
            nb, wb, rb = _stat(b)
            ng, wg, rg = _stat(g)
            if nb and ng:
                halves.append((wg - wb, rg - rb))
                print(f"    {name}: fark win {wg - wb:+5.1f} puan, R/setup {rg - rb:+.3f}  (n {nb}/{ng})")
    nb, wb, rb = _stat(bad)
    ng, wg, rg = _stat(good)
    if nb < MIN_N or ng < MIN_N:
        print(f"  KARAR: veri birikiyor (grup basina {MIN_N}; {nb}/{ng}).")
        return
    gap = rg - rb
    split = len(halves) == 2 and (halves[0][1] >= 0.15) != (halves[1][1] >= 0.15)
    note = "  (zaman yarilari tutarsiz -- kullaniciya sor)" if split else ""
    if gap >= 0.30:
        print(f"  KARAR: fark {gap:+.3f} R/setup -> hard filtre adayi (once tek replay).{note}")
    elif gap >= 0.15:
        print(f"  KARAR: fark {gap:+.3f} R/setup -> -2 skor cezasi adayi (once tek replay).{note}")
    else:
        print(f"  KARAR: fark {gap:+.3f} R/setup (< 0.15) -> degisiklik yok.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d | 1w (bos = hepsi)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = ("SELECT strategy, outcome, rr, first_seen, features, features_at_levels, best_stage "
           "FROM setup_journal WHERE outcome IN ('win','loss') AND best_stage NOT IN ("
           + ",".join("?" * len(NOT_CRT)) + ")")
    params: list = list(NOT_CRT)
    if args.strategy:
        sql += " AND strategy = ?"
        params.append(args.strategy)
    rows = []
    for r in con.execute(sql, params):
        f = _load(r["features_at_levels"]) or _load(r["features"]) or {}
        if "c1_same_color" not in f:
            continue
        rows.append({"outcome": r["outcome"], "rr": r["rr"], "first_seen": str(r["first_seen"]), "f": f})
    con.close()

    print(f"C1 sekli -- strateji: {args.strategy or 'hepsi'}, olculmus cozulmus setup: {len(rows)}")
    if not rows:
        print("c1_* olcumu tasiyan cozulmus satir yok (29.09'da eklendi; sunucu restart'indan sonra birikir).")
        return

    same = [r for r in rows if r["f"]["c1_same_color"] == 1]
    opp = [r for r in rows if r["f"]["c1_same_color"] == 0]
    _compare("H1  C1 ile C2 ayni renk mi?", same, opp, "ayni renk", "zit renk / doji")

    wick = [r for r in rows if isinstance(r["f"].get("c1_tp_wick_frac"), (int, float))]
    long_w = [r for r in wick if r["f"]["c1_tp_wick_frac"] >= LONG_TP_WICK]
    short_w = [r for r in wick if r["f"]["c1_tp_wick_frac"] < LONG_TP_WICK]
    _compare(f"H2  C1'in TP tarafindaki fitil >= {LONG_TP_WICK:.2f} mi?", long_w, short_w,
             "uzun TP fitili", "kisa TP fitili")

    # Bilgi: ikisi birden (SEI tipi) ve C1 govde ceyrekleri
    both = [r for r in same if isinstance(r["f"].get("c1_tp_wick_frac"), (int, float))
            and r["f"]["c1_tp_wick_frac"] >= LONG_TP_WICK]
    print(f"\n== Bilgi: ayni renk + uzun TP fitili (SEI tipi)  {_cell(both)}")
    body = sorted((r for r in rows if isinstance(r["f"].get("c1_body_frac"), (int, float))),
                  key=lambda r: r["f"]["c1_body_frac"])
    if len(body) >= 40:
        q = len(body) // 4
        out = []
        for i in range(4):
            chunk = body[i * q:(i + 1) * q] if i < 3 else body[3 * q:]
            n, wr, _rs = _stat(chunk)
            out.append(f"[{chunk[0]['f']['c1_body_frac']:.2f}-{chunk[-1]['f']['c1_body_frac']:.2f}] {wr:4.1f}%")
        print("== Bilgi: C1 govde orani ceyrekleri (win%)  " + "  ".join(out))


if __name__ == "__main__":
    main()
