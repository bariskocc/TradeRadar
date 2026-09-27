"""4H'te C2 acikken CISD onayliysa dolum olsun mu? -- Setup Journal karsilastirmasi (27.09).

Bugunku kural: 4H'te C2 kapanmadan dolum yok (`require_c2_closed`). BCH 4H 27.09: CISD onayindan sonra C2
icinde 343.17'ye retest geldi, sinyal C2 kapaninca dogdu ve fiyat bir daha donmedi.

Veri (restart sonrasi dolar): 4H seviyeleri C2 kapanisinda yeniden dondurulur; C2-acik seviyeler ve onlarla
yapilan izleme `setup_journal.pre_c2`'de kalir. `pre_c2.stage_at_levels == "c2_open"` = seviyeler donarken
CISD onayli + butun kapilar gecilmisti (radar `c2_open`) -> temiz karsi-olgu. Ayni setupta iki kol:

  A) C2 acikken dolum: pre_c2.o  (C2 icinde dolmadiysa no_c2_fill -> 0R; C2 yeni uc yapip kosan SL'yi
     aldiysa loss; dolduysa C2 sonrasi da ayni seviyelerle TP/SL/ufuk)
  B) bugunku kural: C2 kapanisindaki seviyeler, k=1 golgesi (shadow["1"]), YALNIZ C2 kapanisindaki
     kapilari gecerse (best_stage waiting/week_close/portfoy kapilari); gecmezse 0R.

Duz TP/SL (kismi kar/BE yok), komisyon yok. Karar kurali: IZLEME.md -> "4H'te C2 acikken dolum olsun mu?".

Kullanim: python scripts/c2_open_fill_stat.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent

MIN_PAIRS = 30          # IZLEME.md: karar icin sonuclanmis cift
MIN_EDGE_R = 0.15       # A, B'yi bu kadar R/setup gecmeli
PASS_AFTER_C2 = {"waiting", "week_close", "cluster_limit", "corr_open", "has_open", "duplicate",
                 "missed_quality", "past_sl", "stale", "week_gap", "invalidated", "past_tp"}
RESOLVED = {"win", "loss", "open", "no_c2_fill"}


def r_of(o: str | None, rr) -> float | None:
    if o == "win":
        return float(rr or 0)
    if o == "loss":
        return -1.0
    if o in ("open", "no_c2_fill", "no_touch", "tp_before_entry"):
        return 0.0
    return None                      # pending / filled / ambiguous: sonuclanmadi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    A = ap.parse_args()
    con = sqlite3.connect(A.db)
    con.row_factory = sqlite3.Row
    cols = [r[1] for r in con.execute("PRAGMA table_info(setup_journal)")]
    if "pre_c2" not in cols:
        print("setup_journal.pre_c2 kolonu yok -- restart sonrasi dolmaya baslar.")
        return
    rows = con.execute("select * from setup_journal where strategy='4h' and pre_c2 is not null").fetchall()
    pairs, skipped = [], {"stage_at_levels": 0, "A sonuclanmadi": 0, "B sonuclanmadi": 0}
    for r in rows:
        p = json.loads(r["pre_c2"])
        if p.get("stage_at_levels") != "c2_open":
            skipped["stage_at_levels"] += 1
            continue
        ra = r_of(p.get("o"), p.get("rr"))
        if ra is None:
            skipped["A sonuclanmadi"] += 1
            continue
        if r["best_stage"] in PASS_AFTER_C2:
            sh = (json.loads(r["shadow"] or "{}").get("1") or {})
            rb = r_of(sh.get("o"), sh.get("rr"))
            if rb is None:
                skipped["B sonuclanmadi"] += 1
                continue
        else:
            rb = 0.0                  # C2 kapanisinda elendi: bugunku kural islem acmazdi
        pairs.append((r, p, ra, rb))

    print(f"4H yeniden dondurulmus satir: {len(rows)}  |  karsilastirilabilir cift: {len(pairs)}  "
          f"(atlanan: {skipped})")
    if not pairs:
        print("Veri birikiyor.")
        return
    n = len(pairs)
    fa = [x for x in pairs if x[1].get("o") != "no_c2_fill"]
    wins = sum(1 for x in pairs if x[1].get("o") == "win")
    loss = sum(1 for x in pairs if x[1].get("o") == "loss")
    ra, rb = sum(x[2] for x in pairs), sum(x[3] for x in pairs)
    print(f"\nA) C2 acikken dolum : dolan {len(fa)}/{n}  TP {wins}  SL {loss}   toplam {ra:+.1f}R   R/setup {ra / n:+.3f}")
    print(f"B) bugunku kural    : toplam {rb:+.1f}R   R/setup {rb / n:+.3f}")
    only_a = sum(1 for x in pairs if x[2] != 0 and x[3] == 0)
    print(f"   yalniz A'da islem olan setup: {only_a}   (BCH 27.09 turu: C2 icinde retest, sonra donmedi)")
    print("\n   ornekler (purge, UTC):")
    for r, p, a, b in sorted(pairs, key=lambda x: x[0]["purge_time"])[-12:]:
        print(f"     {r['purge_time'][5:16]} {r['symbol']:<14}{r['direction']:<6} A={p.get('o'):<11}{a:+.2f}  "
              f"B={r['best_stage']:<13}{b:+.2f}")
    edge = (ra - rb) / n
    print(f"\nKARAR: fark {edge:+.3f} R/setup (esik +{MIN_EDGE_R}), cift {n}/{MIN_PAIRS}")
    if n < MIN_PAIRS:
        print("  -> veri birikiyor, karar yok.")
    elif edge >= MIN_EDGE_R:
        print("  -> H1: C2 acikken dolum daha iyi. Kullaniciya 4H require_c2_closed=False onerilir.")
    else:
        print("  -> H2: fark yok/kotu. Kural kalir, madde kapanir.")


if __name__ == "__main__":
    main()
