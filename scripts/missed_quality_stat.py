"""Retest skor kapisi (`missed_quality`): eledigi setuplar sonradan TP'ye mi gidiyor?

Kapi (`scanner`): bekleyen setup'ta fiyat entry'ye dondugunde kalite skoru yeniden olculur;
o an < `MIN_QUALITY_SCORE` (7) ise setup silinir ("retest anindaki setup artik o setup degil").
Ama skor MUM ICINDE oynuyor: C2 kapanisi (+1), PD array ve SMT kalemleri zamanla geliyor.
Soru: kapi gercekten kotu setup'i mi eliyor, yoksa yalnizca ERKEN olctugu icin iyi setup'i mi?

Tetikleyen vaka ve olculen sonuclar calisma notlarinda (IZLEME.md -> "Retest skor kapisi -
kazanan mi eliyor?"): kapi elerken skorun mum ici oynakligi belirleyici olabiliyor -- ayni setup
dakikalar arayla farkli puan aliyor ve fatal okuma en dusuk olani oluyor.
NOT: bir satirda `last_stage='tight_stop'` gormek o setup'in dar stop kapisinda elendigi anlamina
GELMEZ; baglayici kapi `best_stage`'dir, `last_stage` yalnizca en son degerlendirmedir.

**Neden yeni veri gerekmiyor:** Journal elenen setup'in sonrasini zaten izliyor
(`best_stage`='missed_quality' satirinin `outcome`'u), seviyeler ilk goruldugu haliyle donuk.

**C2 zamanlama dilimi (H3):** retest, C2 mumu KAPANMADAN once olduysa skor "C2 kapali +1" ve
PD kalemlerini alamaz -- yapisal dezavantaj. C2 kapanis ani `purge_time + bar suresi`, retest ani
`entry_touched_at`; ikisi de Journal'da duruyor, yani bu dilim de birikmis veriden turer.

Karar kurali ve esikler IZLEME.md -> "Retest skor kapisi - kazanan mi eliyor?" icinde ONCEDEN
yazildi; bu betik yalnizca oradaki kurali calistirir.

Kullanim:
    python scripts/missed_quality_stat.py
    python scripts/missed_quality_stat.py --strategy 4h --market crypto
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esikler.
MIN_RESOLVED = 20        # H1: karar icin gereken cozulmus elenen setup
MIN_WR_GAP_PP = 5.0      # H2: kapi koruyucu sayilmasi icin taban orandan bu kadar puan dusuk olmali
MIN_SLICE = 10           # H3: her zamanlama diliminde gereken cozulmus setup
MIN_SLICE_GAP_PP = 10.0  # H3: dilimler arasi anlamli sayilan WR farki

GATE = "missed_quality"
WIN, LOSS = "win", "loss"
# C2 mumunun suresi: kapanis ani = purge_time + bu sure.
BAR = {"4h": timedelta(hours=4), "1d": timedelta(days=1), "1h": timedelta(hours=1)}


def _dt(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


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
        f"purge_time, entry_touched_at "
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


def c2_open_at_retest(row: sqlite3.Row) -> bool | None:
    """Retest, C2 mumu kapanmadan once mi oldu? Bilinemiyorsa None."""
    purge, touched = _dt(row["purge_time"]), _dt(row["entry_touched_at"])
    span = BAR.get(row["strategy"] or "")
    if purge is None or touched is None or span is None:
        return None
    return touched < purge + span


def _wr(rows: list[sqlite3.Row]) -> tuple[int, int, float, float]:
    """(cozulmus, win, WR%, R/islem)."""
    resolved = [r for r in rows if r["outcome"] in (WIN, LOSS)]
    if not resolved:
        return 0, 0, 0.0, 0.0
    w = sum(1 for r in resolved if r["outcome"] == WIN)
    total = sum(r_of(r) for r in resolved)
    return len(resolved), w, 100 * w / len(resolved), total / len(resolved)


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
    print("  RETEST SKOR KAPISI (missed_quality) - eledigi setuplar ne yapti?")
    print("=" * 78)
    if not rows:
        print("  Kapida elenmis setup yok.")
        return

    resolved = [r for r in rows if r["outcome"] in (WIN, LOSS)]
    zero = [r for r in rows if r["outcome"] in ("tp_before_entry", "no_touch")]
    open_ = [r for r in rows if r not in resolved and r not in zero]

    print(f"\n  elenen setup: {len(rows)}   cozulmus: {len(resolved)}"
          f"   TP entry'den once / dokunmadi: {len(zero)}   hala izlemede: {len(open_)}")
    for r in rows:
        early = c2_open_at_retest(r)
        mark = "C2 acik" if early else ("C2 kapali" if early is False else "   -   ")
        print(f"    {str(r['first_seen'])[:10]}  {r['strategy']:3} {r['symbol']:14} {r['direction']:5}"
              f" skor {r['score'] or 0:>4.0f}  RR {r['rr'] or 0:>5.2f}  retest {mark:9}"
              f"  -> {r['outcome'] or 'izlemede'}")

    if not resolved:
        print("\n  KARAR: HENUZ YOK - cozulmus elenen setup yok.")
        return

    n, w, wr, r_per = _wr(rows)
    bw, bl = int(base.get("w") or 0), int(base.get("l") or 0)
    base_wr = 100 * bw / (bw + bl) if (bw + bl) else 0.0

    print(f"\n  ELENENLER    : cozulmus {n:>4}  win {w:>3}  WR %{wr:.1f}"
          f"  R {r_per * n:+.1f}  R/islem {r_per:+.3f}")
    print(f"  TABAN ORAN   : win {bw:>3} / loss {bl:>3}  WR %{base_wr:.1f}"
          f"   (ayni filtreler, kapida elenenler haric)")
    print(f"  fark         : {wr - base_wr:+.1f} puan   (H2: kapi koruyucu ise <= -{MIN_WR_GAP_PP:.0f} olmali)")

    # H3: retest C2 kapanmadan once mi oldu?
    early = [r for r in rows if c2_open_at_retest(r) is True]
    late = [r for r in rows if c2_open_at_retest(r) is False]
    en, ew, ewr, er = _wr(early)
    ln, lw, lwr, lr = _wr(late)
    print(f"\n  ZAMANLAMA DILIMI (H3) - skor, C2 kapanmadan olculdugunde +1 ve PD kalemlerini alamaz")
    print(f"    retest C2 ACIKKEN  : cozulmus {en:>3}  win {ew:>3}  WR %{ewr:.1f}  R/islem {er:+.3f}")
    print(f"    retest C2 KAPALIYKEN: cozulmus {ln:>3}  win {lw:>3}  WR %{lwr:.1f}  R/islem {lr:+.3f}")
    if en and ln:
        print(f"    fark: {ewr - lwr:+.1f} puan   (H3 esigi: her dilimde >= {MIN_SLICE} cozulmus"
              f" ve fark >= {MIN_SLICE_GAP_PP:.0f} puan)")

    print("\n  KARAR:", end=" ")
    if n < MIN_RESOLVED:
        print(f"HENUZ YOK - cozulmus {n} < {MIN_RESOLVED}. Veri birikiyor.")
    elif r_per > 0:
        print(f"H1 GECTI (R/islem {r_per:+.3f} > 0) - kapi kazanc kesiyor. "
              "Degistirmeden once tek dogrulama replay'i.")
    elif wr - base_wr <= -MIN_WR_GAP_PP:
        print(f"KAPI KORUYUCU - elenenlerin WR'si tabandan {base_wr - wr:.1f} puan dusuk, "
              f"R/islem {r_per:+.3f}. Kapi kalir, madde done'a cekilir.")
    else:
        print(f"KARARSIZ - R/islem {r_per:+.3f} <= 0 ama WR farki {wr - base_wr:+.1f} puan "
              f"(esik -{MIN_WR_GAP_PP:.0f}). Kapi kalir, izleme surer.")

    if en >= MIN_SLICE and ln >= MIN_SLICE and (ewr - lwr) >= MIN_SLICE_GAP_PP:
        print("  H3 GECTI - bedel ZAMANLAMADAN geliyor: kapiyi kaldirmak degil, skoru C2 "
              "kapanisinda yeniden olcmek dogru cozum.")

    print("\nKarar kurali ve sinirlar: IZLEME.md -> \"Retest skor kapisi - kazanan mi eliyor?\"")


if __name__ == "__main__":
    main()
