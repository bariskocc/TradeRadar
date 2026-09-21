"""LONG'lar mi kotu, piyasa mi yonluydu? (canli islemler + Journal taban orani)

Soru (17.09.2026): canli sinyallerde LONG 2/15 (-8.24R), SHORT 7/12 (+8.21R) cikti. Bu motorun
SECIMINDEN mi geliyor, yoksa piyasa o donemde tek yonlu muydu?

Tek basina canli islemler bunu ayiramaz (n kucuk ve zaten kapilardan gecmis, secilmis bir kume).
Ayirt edici olcum: **Setup Journal taban orani** -- motorun gordugu ama elemis oldugu setuplar da
seviyeleriyle izleniyor (`outcome`: win = entry'ye gelip TP, loss = entry'ye gelip SL). Journal
kumesi kapilardan gecmedigi icin piyasanin o donemdeki yon egilimini tarafsiz gosterir:

  - Canlida fark BUYUK, Journal'da fark KUCUK  -> fark motorun secimindendir (kapilar incelenir)
  - Ikisinde de fark BUYUK                     -> piyasa rejimi; kural degismez
  - Ikisinde de fark KUCUK                     -> gurultuydu, baslik kapanir

Karar kurali ve esikler onceden yazildi: IZLEME.md -> "LONG/SHORT ayrismasi".

Kullanim:
    python scripts/direction_stat.py                    # 08.09.2026'dan bugune (izleme penceresi)
    python scripts/direction_stat.py --days 30
    python scripts/direction_stat.py --strategy 4h
    python scripts/direction_stat.py --market crypto
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esikler
MIN_N_PER_SIDE = 30      # her yon icin bu kadar kapali islem birikmeden karar yok
GAP_DECIDE = 20.0        # canli WR farki (puan) -- H1/H2 esigi
GAP_JOURNAL_SMALL = 8.0  # Journal farki bunun altindaysa piyasa yonlu degildi
GAP_NOISE = 10.0         # canli fark bunun altina inerse baslik kapanir
WINDOW_START = "2026-09-08 00:00:00"  # izleme penceresinin basi (ilk kapali islem)


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def pct(win: int, n: int) -> float:
    return (win / n * 100.0) if n else 0.0


def row(label: str, n: int, win: int, loss: int, net: float | None = None) -> None:
    """BE'yi kayip saymaz: n = W + BE + L, o yuzden loss ayrica gecilir."""
    extra = f"   net={net:+7.2f}R" if net is not None else ""
    be = n - win - loss
    bar = "#" * max(1, round(pct(win, n) / 4)) if n else ""
    print(f"  {label:<26} n={n:<4} W={win:<4} BE={be:<3} L={loss:<4} "
          f"oran {pct(win, n):>5.1f}%{extra}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0, help="son N gun (0 = izleme penceresi basindan)")
    ap.add_argument("--strategy", default="", help="4h | 1d | 1h")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    args = ap.parse_args()

    since = WINDOW_START
    if args.days:
        since = (datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # --- 1) Canli islemler (secilmis kume) -------------------------------------------------
    sql = ("SELECT direction, market_type, timeframe, result, rr_value, symbol, closed_at "
           "FROM signals WHERE closed_at IS NOT NULL AND closed_at >= ? AND rr_value IS NOT NULL")
    params: list = [since]
    if args.strategy:
        sql += " AND timeframe = ?"
        params.append(args.strategy)
    if args.market:
        sql += " AND market_type = ?"
        params.append(args.market)
    live = [dict(r) for r in con.execute(sql, params)]

    head(f"1) Canli islemler -- {len(live)} kapali islem ({since[:10]} ->)")
    if not live:
        print("  kayit yok")
        return
    agg: dict = defaultdict(lambda: [0, 0, 0.0, 0])
    for x in live:
        a = agg[x["direction"]]
        a[0] += 1
        a[1] += x["result"] == "win"
        a[2] += x["rr_value"] or 0.0
        a[3] += x["result"] == "loss"
    for d in ("LONG", "SHORT"):
        n, w, net, l = agg[d]
        row(d, n, w, l, net)
    live_gap = pct(agg["SHORT"][1], agg["SHORT"][0]) - pct(agg["LONG"][1], agg["LONG"][0])
    print(f"\n  SHORT - LONG kazanma orani farki: {live_gap:+.1f} puan")

    print("\n  -- kirilim (yon x strateji / market)")
    sub: dict = defaultdict(lambda: [0, 0, 0.0, 0])
    for x in live:
        for k in (f'{x["direction"]}/{x["timeframe"]}', f'{x["direction"]}/{x["market_type"]}'):
            s = sub[k]
            s[0] += 1
            s[1] += x["result"] == "win"
            s[2] += x["rr_value"] or 0.0
            s[3] += x["result"] == "loss"
    for k in sorted(sub, key=lambda k: -sub[k][0]):
        n, w, net, l = sub[k]
        row(k, n, w, l, net)

    # --- 2) Journal taban orani (secilmemis kume) ------------------------------------------
    jsql = ("SELECT direction, outcome FROM setup_journal WHERE outcome IN ('win', 'loss') "
            "AND entry_touched_at IS NOT NULL AND entry_touched_at >= ?")
    jparams: list = [since]
    if args.strategy:
        jsql += " AND strategy = ?"
        jparams.append(args.strategy)
    if args.market:
        jsql += " AND market_type = ?"
        jparams.append(args.market)
    jr = [dict(r) for r in con.execute(jsql, jparams)]

    head(f"2) Journal taban orani -- {len(jr)} setup entry'ye dokundu (duz TP/SL)")
    jagg: dict = defaultdict(lambda: [0, 0])
    for x in jr:
        a = jagg[x["direction"]]
        a[0] += 1
        a[1] += x["outcome"] == "win"
    for d in ("LONG", "SHORT"):
        n, w = jagg[d]
        row(d, n, w, n - w)
    j_gap = pct(jagg["SHORT"][1], jagg["SHORT"][0]) - pct(jagg["LONG"][1], jagg["LONG"][0])
    print(f"\n  SHORT - LONG TP orani farki: {j_gap:+.1f} puan")
    print("  (bu kume kapilardan GECMEDI -- piyasanin yon egilimini tarafsiz gosterir)")

    # --- 3) Onceden yazilmis karar kurali ---------------------------------------------------
    head("3) Karar (IZLEME.md 'LONG/SHORT ayrismasi' kurali)")
    n_long, n_short = agg["LONG"][0], agg["SHORT"][0]
    print(f"  ornek: LONG n={n_long}, SHORT n={n_short} (esik: her yon icin {MIN_N_PER_SIDE})")
    print(f"  canli fark {live_gap:+.1f} puan | Journal farki {j_gap:+.1f} puan")
    if n_long < MIN_N_PER_SIDE or n_short < MIN_N_PER_SIDE:
        print(f"\n  -> KARAR YOK. Ornek yetersiz; veri birikiyor. Altindaki isaret gurultudur.")
    elif abs(live_gap) < GAP_NOISE:
        print("\n  -> H3: canli fark kapandi. Baslik kapanir, gurultuydu.")
    elif abs(live_gap) >= GAP_DECIDE and abs(j_gap) < GAP_JOURNAL_SMALL:
        print("\n  -> H1: fark motorun SECIMINDEN geliyor (piyasa yonlu degildi).")
        print("     LONG tarafindaki kapilar incelenir: 1W/1D bias kalemleri, PD array, C2 rengi.")
    elif abs(live_gap) >= GAP_DECIDE and abs(j_gap) >= GAP_DECIDE:
        print("\n  -> H2: piyasa rejimi yonluydu. Kural degismez, 'yonlu rejim' olarak not edilir.")
    else:
        print("\n  -> ARA BOLGE. Kural bu hali icin yazilmadi; karar verme, veri biriktir.")


if __name__ == "__main__":
    main()
