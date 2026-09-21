"""1D bias 1H'te iki kez mi cezalandiriyor? (hard filtre + skor kalemi)

1D bias 1H setup'ina IKI kanaldan giriyor:

  (a) HARD FILTRE (`REQUIRE_HTF_BIAS_ALIGN`): yon uyusmuyorsa setup hic acilmaz
      -> Journal'da `best_stage` = 'bias_mismatch'.
  (b) SKOR KALEMI (`score_parts.htf`): hizali +2, NEUTRAL 0, karsi -2. 9 tavanli,
      7 esikli skorda +2 tek basina belirleyici.

Soru (20.09, BTC 1H vakasi): 1H'in HTF'i zaten 1H; 1D bias'in ayni setup'a hem veto hem
+-2 puan olarak girmesi 1H'te fazla mi? Vaka: BTC SHORT 20.09 01:00 TSI C1 -- 02:25'te
hard filtre eledi (daily BULLISH), 03:00'da bias NEUTRAL'a donunce bu kez skor kapisi
eledi (htf 0 -> skor 1). Fiyat TP'ye gitti.

Yeni izleme kodu GEREKMEDI: Journal elenen setup'in sonrasini zaten izliyor ve skor
kirilimi 16.09'dan beri kayitli. Bu betik iki kanali ayri okur ve IZLEME.md ->
"1D bias 1H'te iki kez mi eliyor?" icinde ONCEDEN yazilmis kurali calistirir.

  H1 (kapi):  cozulmus >= 25 VE R/islem > 0  -> hard filtre 1H'te kazanc kesiyor;
              1H icin bias'i skor kalemine indirme tartismasi acilir (kural degismeden
              once tek dogrulama replay'i).
  H2 (kalem): htf=+2 ile htf=0 kovalarinin WR farki. Her kovada n >= 25 iken fark
              < 5 puansa +2 kalemi 1H'te kanitsiz -- kalem kucultulur (dilim degeri
              kardes madde "Skor kalemleri" ile birlikte okunur).
  Kontrol:    ayni iki olcum 4H/1D icin de basilir. Sonuc her stratejide aynıysa sorun
              1H'e ozgu degil, genel bias tasarimindadir -> madde oraya devredilir.

Kullanim:
    python scripts/bias_1h_stat.py
    python scripts/bias_1h_stat.py --strategy 4h --days 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esikler.
MIN_RESOLVED = 25        # H1: hard filtre karari icin gereken cozulmus elenen setup
MIN_BUCKET = 25          # H2: htf kovasi basina gereken cozulmus setup
MIN_WR_GAP_PP = 5.0      # H2: +2 kaleminin kanitli sayilmasi icin gereken WR farki

# Motorun CRT saymadigi adaylar: kapi sirasina hic gelmediler, taban orani bozarlar.
PRE_STAGES = ("sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "not_selected")
WIN, LOSS = "win", "loss"
RESOLVED = (WIN, LOSS)


def r_of(outcome: str | None, rr: float | None) -> float:
    if outcome == WIN:
        return float(rr or 0)
    return -1.0 if outcome == LOSS else 0.0


def stats(rows: list[sqlite3.Row]) -> tuple[int, int, float, float]:
    """(n, win, WR, R/islem) -- yalnizca cozulmus satirlar."""
    res = [r for r in rows if r["outcome"] in RESOLVED]
    if not res:
        return 0, 0, 0.0, 0.0
    w = sum(1 for r in res if r["outcome"] == WIN)
    total = sum(r_of(r["outcome"], r["rr"]) for r in res)
    return len(res), w, 100 * w / len(res), total / len(res)


def fetch(con: sqlite3.Connection, strategy: str, days: int, market: str) -> list[sqlite3.Row]:
    where = ["levels_at IS NOT NULL", "strategy = ?"]
    params: list = [strategy]
    if days:
        where.append("first_seen >= datetime('now', ?)")
        params.append(f"-{days} days")
    if market:
        where.append("market_type = ?")
        params.append(market)
    return con.execute(
        f"SELECT symbol, direction, score, rr, outcome, best_stage, first_seen, score_parts "
        f"FROM setup_journal WHERE {' AND '.join(where)} ORDER BY first_seen",
        params,
    ).fetchall()


def report(con: sqlite3.Connection, strategy: str, days: int, market: str, *, full: bool) -> None:
    rows = fetch(con, strategy, days, market)
    engine = [r for r in rows if (r["best_stage"] or "") not in PRE_STAGES]
    gated = [r for r in engine if r["best_stage"] == "bias_mismatch"]
    rest = [r for r in engine if r["best_stage"] != "bias_mismatch"]

    gn, gw, gwr, gr = stats(gated)
    bn, bw, bwr, _ = stats(rest)
    head = f"  {strategy:3} | (a) HARD FILTRE: elenen {len(gated):>4}  cozulmus {gn:>4}"
    if gn:
        head += f"  win {gw:>3}  WR %{gwr:>5.1f}  R/islem {gr:+.3f}   [taban WR %{bwr:.1f}]"
    print(head)

    buckets: dict[int, list[sqlite3.Row]] = {2: [], 0: [], -2: []}
    for r in engine:
        if not r["score_parts"]:
            continue
        try:
            htf = int(json.loads(r["score_parts"]).get("htf", 0))
        except (ValueError, TypeError):
            continue
        buckets.setdefault(htf, []).append(r)
    line = f"  {strategy:3} | (b) SKOR KALEMI:"
    for htf in (2, 0, -2):
        n, w, wr, rpt = stats(buckets.get(htf, []))
        line += f"  htf{htf:+d}: n={n:<4}WR %{wr:>5.1f} R {rpt:+.3f} |"
    print(line)

    if not full:
        return

    if gated:
        print("\n  Hard filtrenin eledigi setuplar:")
        for r in gated:
            print(f"    {str(r['first_seen'])[:10]}  {r['symbol']:14} {r['direction']:5}"
                  f" skor {r['score'] or 0:>4.0f}  RR {r['rr'] or 0:>5.2f}  -> {r['outcome'] or 'izlemede'}")

    print("\n  KARAR (a) hard filtre:", end=" ")
    if gn < MIN_RESOLVED:
        print(f"HENUZ YOK - cozulmus {gn} < {MIN_RESOLVED}. Veri birikiyor.")
    elif gr > 0:
        print(f"H1 GECTI (R/islem {gr:+.3f} > 0) - filtre {strategy}'te kazanc kesiyor. "
              "Bias'i skor kalemine indirmeyi tartis; once tek dogrulama replay'i.")
    else:
        print(f"FILTRE KORUYUCU (R/islem {gr:+.3f} <= 0, elenen WR %{gwr:.1f} vs taban %{bwr:.1f}).")

    n2, _, wr2, r2 = stats(buckets.get(2, []))
    n0, _, wr0, r0 = stats(buckets.get(0, []))
    print("  KARAR (b) +2 kalemi :", end=" ")
    if min(n2, n0) < MIN_BUCKET:
        print(f"HENUZ YOK - kova basina n gerekli {MIN_BUCKET} (htf+2: {n2}, htf0: {n0}).")
    elif wr2 - wr0 < MIN_WR_GAP_PP:
        print(f"KANITSIZ - htf+2 WR %{wr2:.1f} (R {r2:+.3f}) vs htf0 %{wr0:.1f} ({r0:+.3f}), "
              f"fark {wr2 - wr0:+.1f} puan < {MIN_WR_GAP_PP:.0f}. Kalemi kucult.")
    else:
        print(f"KALEM KANITLI - fark {wr2 - wr0:+.1f} puan (htf+2 %{wr2:.1f} / htf0 %{wr0:.1f}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="1h", help="1h | 4h | 1d")
    ap.add_argument("--days", type=int, default=0, help="0 = tum zamanlar")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    print("=" * 96)
    print(f"  1D BIAS IKI KANALDA - {args.strategy.upper()}"
          f"{f' (son {args.days} gun)' if args.days else ''}"
          f"{f' - {args.market}' if args.market else ''}")
    print("=" * 96)
    report(con, args.strategy, args.days, args.market, full=True)

    others = [s for s in ("1h", "4h", "1d") if s != args.strategy]
    print("\n  KONTROL (ayni olcum, diger stratejiler):")
    for s in others:
        report(con, s, args.days, args.market, full=False)
    print("\n  Ayni sonuc her stratejideyse sorun 1H'e ozgu degil -> genel bias tasarimi.")
    print("\nKarar kurali ve sinirlar: IZLEME.md -> \"1D bias 1H'te iki kez mi eliyor?\"")


if __name__ == "__main__":
    main()
