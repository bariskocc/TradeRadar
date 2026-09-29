"""Gercek sinyallerde: fiyat TP yolunun %50'sini gectikten SONRA entry'ye donen islem stop mu oluyor?

Soru (kullanici, 29.09): dolumdan once TP yolunun buyuk kismini kosan sinyal, geri donup entry'yi
doldurdugunda SL'ye mi gidiyor? Journal `prefill` ayni soruyu tum setuplarda cevapladi (28.09, kapi
acilmadi) ama penceresi Journal'in KENDI dolumunda bitiyor; motorun gercek dolumunu goremedi
(APT #113, DOGE #114, AVAX #115 -- TODO 5q).

Olcum `signals.prefill_run` (29.09+, scanner `_measure_prefill_runs`): CISD onay mumundan motorun dolum
mumuna kadar (dolum mumu haric) TP yolunda en uzak nokta, 0 = entry, 1 = TP. `prefill_cov` = 0 ise
pencerenin basi store'da yoktu (deger alt sinir).

Sonuc gercek islem: kismi kar + BE dahil `rr_value`; BE = exit_reason 'be' ya da result 'breakeven'
(Dashboard `_trade_outcome` kurali).

Karar kurali (onceden yazildi, IZLEME.md "Gercek sinyallerde dolum oncesi %50+ kosu"): >= %50 kosan grupta
10, digerinde 20 kapali islem. >= %50 grubu digerinden
  - SL orani >= 15 puan yuksek VE R/islem >= 0.30 kotuyse -> kapi adayi ("TP yolunun %50'sini kosan
    bekleyen emir iptal") -- once tek replay
  - aksi -> degisiklik yok (28.09 Journal karariyla ayni yonde)

Kullanim: python scripts/prefill_signal_stat.py [--strategy 4h]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
RUN = 0.50
MIN_HI, MIN_LO = 10, 20


def _outcome(r) -> str:
    if r["result"] == "loss":
        return "L"
    if r["exit_reason"] == "be" or r["result"] == "breakeven":
        return "BE"
    return "W"


def _row(label: str, sub: list) -> str:
    n = len(sub)
    if not n:
        return f"  {label:22s} n=  0"
    c = {k: sum(1 for r in sub if r["out"] == k) for k in ("W", "BE", "L")}
    rs = sum(float(r["rr_value"] or 0.0) for r in sub) / n
    return (f"  {label:22s} n={n:3d}  W {c['W']:3d}  BE {c['BE']:3d}  L {c['L']:3d}  "
            f"SL% {100 * c['L'] / n:5.1f}  R/islem {rs:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d | 1w (bos = hepsi)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("PRAGMA table_info(signals)")}
    if "prefill_run" not in cols:
        print("signals.prefill_run kolonu yok -- sunucu bir kez yeniden baslayinca (init_db) eklenir.")
        return
    sql = ("SELECT id, symbol, timeframe, direction, result, exit_reason, rr_value, prefill_run, prefill_cov "
           "FROM signals WHERE prefill_run IS NOT NULL AND status != 'active' AND result IS NOT NULL")
    params: list = []
    if args.strategy:
        sql += " AND timeframe = ?"
        params.append(args.strategy)
    rows = [dict(r) | {"out": _outcome(r)} for r in con.execute(sql, params)]
    con.close()

    print(f"Gercek sinyaller, dolum oncesi kosu -- strateji: {args.strategy or 'hepsi'}, kapali islem: {len(rows)}")
    if not rows:
        print("prefill_run olculmus kapali sinyal yok (29.09'da eklendi; restart sonrasi dolan sinyallerle birikir).")
        return
    hi = [r for r in rows if r["prefill_run"] >= RUN]
    lo = [r for r in rows if r["prefill_run"] < RUN]
    print(_row(f">= %{RUN * 100:.0f} kostu", hi))
    print(_row(f"< %{RUN * 100:.0f}", lo))
    print(_row("  (0-25%)", [r for r in rows if r["prefill_run"] < 0.25]))
    part = sum(1 for r in rows if r["prefill_cov"] == 0)
    if part:
        print(f"  (pencere basi store'da olmayan: {part} -- deger alt sinir)")
    print("\n  >= %50 kosan islemler:")
    for r in sorted(hi, key=lambda r: -r["prefill_run"]):
        print(f"    #{r['id']:<4d} {r['symbol']:14s} {r['timeframe']:3s} {r['direction']:5s} kosu "
              f"%{r['prefill_run'] * 100:4.0f}  {r['out']:2s} {float(r['rr_value'] or 0):+.2f}R")

    if len(hi) < MIN_HI or len(lo) < MIN_LO:
        print(f"\n  KARAR: veri birikiyor (>= %50: {len(hi)}/{MIN_HI}, diger: {len(lo)}/{MIN_LO}).")
        return
    sl = lambda s: 100 * sum(1 for r in s if r["out"] == "L") / len(s)
    rs = lambda s: sum(float(r["rr_value"] or 0.0) for r in s) / len(s)
    d_sl, d_r = sl(hi) - sl(lo), rs(lo) - rs(hi)
    if d_sl >= 15 and d_r >= 0.30:
        print(f"\n  KARAR: >= %50 grubu SL +{d_sl:.1f} puan, R -{d_r:.3f} -> kapi adayi (once tek replay).")
    else:
        print(f"\n  KARAR: fark esikte degil (SL {d_sl:+.1f} puan, R {-d_r:+.3f}) -> degisiklik yok.")


if __name__ == "__main__":
    main()
