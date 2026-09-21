"""Dar stop kapisi: eledigi setuplar stop olmadan TP'ye gidiyor mu?

Kapi (`scanner`, `min_stop_range_mult` = 1.0): entry ile C2 ucu arasindaki mesafe son 20 LTF
mumunun ortalama range'inin altindaysa setup acilmaz -- gerekce "stop o kadar dar ki gurultu
oldurur". Ama dar stop ayni zamanda YUKSEK RR demek; kapi gercekten kayip mi onluyor, yoksa
kazanc mi kesiyor? Journal her elenen setup'in sonrasini zaten izliyor (`best_stage`='tight_stop',
`outcome`), yani soru yeni veri toplamadan cevaplanir.

Karsilastirma taban orani: ayni donemde seviyesi olan TUM setuplar (kapidan gecen + elenen),
`tight_stop` satirlari cikarilarak. Kapi koruyucuysa eledigi setuplarin win orani tabandan
BELIRGIN dusuk olmali.

Karar kurali ve esikler IZLEME.md -> "Dar stop kapisi - elenen setup TP'ye gidiyor mu?"
icinde ONCEDEN yazildi; bu betik yalnizca oradaki kurali calistirir.

  H1 (karar): cozulmus >= 20 VE elenenlerin R/islem'i > 0  -> kapi kazanc kesiyor; kaldirmadan
              once tek dogrulama replay'i (kismi kar/BE/trail + portfoy kapilariyla).
  H2 (bilgi): elenenlerin win orani ile taban oran farki. Kapi koruyucu sayilmasi icin
              tabandan en az 5 puan dusuk olmali.

Kullanim:
    python scripts/tight_stop_stat.py
    python scripts/tight_stop_stat.py --strategy 4h
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esikler.
MIN_RESOLVED = 20        # H1: karar icin gereken cozulmus elenen setup
MIN_WR_GAP_PP = 5.0      # H2: kapi koruyucu sayilmasi icin taban orandan bu kadar puan dusuk olmali

GATE = "tight_stop"
WIN, LOSS = "win", "loss"


def fetch(con: sqlite3.Connection, strategy: str, market: str) -> tuple[list[sqlite3.Row], dict]:
    where = ["levels_at IS NOT NULL"]
    params: list = []
    if strategy:
        where.append("strategy = ?")
        params.append(strategy)
    if market:
        where.append("market_type = ?")
        params.append(market)
    cond = " AND ".join(where)
    rows = con.execute(
        f"SELECT strategy, symbol, direction, score, rr, outcome, first_seen, "
        f"json_extract(features, '$.stop_range_mult') AS mult "
        f"FROM setup_journal WHERE {cond} AND best_stage = ? ORDER BY first_seen",
        (*params, GATE),
    ).fetchall()
    base = dict(con.execute(
        f"SELECT SUM(outcome = 'win') AS w, SUM(outcome = 'loss') AS l "
        f"FROM setup_journal WHERE {cond} AND best_stage != ?", (*params, GATE)
    ).fetchone())
    return rows, base


def r_of(row: sqlite3.Row) -> float:
    if row["outcome"] == WIN:
        return float(row["rr"] or 0)
    return -1.0 if row["outcome"] == LOSS else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d | 1h")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows, base = fetch(con, args.strategy, args.market)

    print("=" * 78)
    print("  DAR STOP KAPISI - eledigi setuplar ne yapti?")
    print("=" * 78)
    if not rows:
        print("  Kapida elenmis setup yok.")
        return

    resolved = [r for r in rows if r["outcome"] in (WIN, LOSS)]
    zero = [r for r in rows if r["outcome"] in ("tp_before_entry", "no_touch")]
    open_ = [r for r in rows if r not in resolved and r not in zero]

    print(f"\n  elenen setup: {len(rows)}   cozulmus: {len(resolved)}"
          f"   TP entry'den once: {len(zero)}   hala izlemede: {len(open_)}")
    for r in rows:
        mult = f"{r['mult']:.2f}" if r["mult"] is not None else "  - "
        print(f"    {str(r['first_seen'])[:10]}  {r['strategy']:3} {r['symbol']:14} {r['direction']:5}"
              f" skor {r['score'] or 0:>4.0f}  RR {r['rr'] or 0:>5.2f}  mult {mult}"
              f"  -> {r['outcome'] or 'izlemede'}")

    if not resolved:
        print("\n  KARAR: HENUZ YOK - cozulmus elenen setup yok.")
        return

    w = sum(1 for r in resolved if r["outcome"] == WIN)
    total_r = sum(r_of(r) for r in resolved)
    wr = 100 * w / len(resolved)
    bw, bl = int(base.get("w") or 0), int(base.get("l") or 0)
    base_wr = 100 * bw / (bw + bl) if (bw + bl) else 0.0

    print(f"\n  ELENENLER    : cozulmus {len(resolved):>4}  win {w:>3}  WR %{wr:.1f}"
          f"  R {total_r:+.1f}  R/islem {total_r / len(resolved):+.3f}")
    print(f"  TABAN ORAN   : win {bw:>3} / loss {bl:>3}  WR %{base_wr:.1f}"
          f"   (ayni filtreler, kapida elenenler haric)")
    print(f"  fark         : {wr - base_wr:+.1f} puan   (H2: kapi koruyucu ise <= -{MIN_WR_GAP_PP:.0f} olmali)")

    print("\n  KARAR:", end=" ")
    if len(resolved) < MIN_RESOLVED:
        print(f"HENUZ YOK - cozulmus {len(resolved)} < {MIN_RESOLVED}. Veri birikiyor.")
    elif total_r > 0:
        print(f"H1 GECTI (R/islem {total_r / len(resolved):+.3f} > 0) - kapi kazanc kesiyor. "
              "Kaldirmadan once tek dogrulama replay'i.")
    elif wr - base_wr <= -MIN_WR_GAP_PP:
        print(f"KAPI KORUYUCU - elenenlerin WR'si tabandan {base_wr - wr:.1f} puan dusuk, R/islem "
              f"{total_r / len(resolved):+.3f}. Kapi kalir, madde done'a cekilir.")
    else:
        print(f"KARARSIZ - R/islem {total_r / len(resolved):+.3f} <= 0 ama WR farki "
              f"{wr - base_wr:+.1f} puan (esik -{MIN_WR_GAP_PP:.0f}). Kapi kalir, izleme surer.")

    print("\nKarar kurali ve sinirlar: IZLEME.md -> "
          "\"Dar stop kapisi - elenen setup TP'ye gidiyor mu?\"")


if __name__ == "__main__":
    main()
