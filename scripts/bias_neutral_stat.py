"""1D bias NEUTRAL'i azaltmak -- "celismede taze structure" varyanti (28.10, kullanici). Motor DEGISMEZ.

Iki olcum:
  1) Bias karnesi (`bias_journal`): bugunku kural / varyant / yalniz ict icin yonlu pay ve 1-3-5 gun isabeti; ayrica
     bugun NEUTRAL olup varyantin yon verdigi gunlerde isabet ("eklenen yon yazi-tura mi?").
  2) Journal karsi-olgusu (`app/bias_variant.flips`): varyantla ACILACAK ve DUSECEK 4H / 1D setuplar, TP once mi + dolum
     bazli R. Yaklasik: dar stop, skor-7 kapisi, kume/korelasyon taklit edilmez -- kural degismeden once tek replay.

Karar kurali (IZLEME.md "1D bias NEUTRAL'ı azaltmak — çatışmada taze yapısal yön", 28.10, taze veri gorulmeden):
  tetik 4H'te levels_at >= 29.10, eklenen >= 40 ve dusen >= 15 cozulmus.
  K1 eklenen dolum R/islem > 0 VE (eklenen - dusen) R/setup >= +0.15 VE iki zaman yarisinda fark > 0 VE 1D'de fark
     <= -0.15 degil -> tek replay (4H + 1D), karar kullaniciya
  K2 fark <= -0.15 ya da eklenen dolum R/islem <= 0 -> kural ayni, madde kapanir
  K3 digerleri -> kullaniciya sor
Kullanim: python scripts/bias_neutral_stat.py [--since 2026-10-04] [--strategy 4h|1d]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app import bias_variant as bv  # noqa: E402

BAND = 0.15


def _fmt(s: dict) -> str:
    def f(v):
        return f"{v:+.2f}" if v is not None else "  -  "
    return (f"n {s['n']:3d} · TP {s['w']:3d} (dolmadan {s['tpb']:2d}) · SL {s['l']:3d} · "
            f"R/setup {f(s['rps'])} · dolum R/işlem {f(s['rpf'])}")


def karne(con, since: str) -> None:
    rows = [dict(r) for r in con.execute("select * from bias_journal where day >= ?", (since,))]
    d1 = {x[0] for x in con.execute("select distinct symbol from setup_journal where strategy='1d'")}
    kols = (("bugünkü kural", bv.bias_now),) + tuple((lab, fn) for _k, lab, fn in bv.VARIANTS)
    for uni, sub in (("1D evreni", [r for r in rows if r["symbol"] in d1]), ("bütün semboller", rows)):
        print(f"\n== Karne — {uni} ({since}+, {len(sub)} gün×sembol): yönlü pay · isabet 1g / 3g / 5g")
        for lab, fn in kols:
            dirn = [r for r in sub if bv._sg(fn(r))]
            hs = []
            for h in ("fwd_1d", "fwd_3d", "fwd_5d"):
                m = [bv._sg(fn(r)) * r[h] for r in dirn if r[h] is not None]
                hs.append(f"%{sum(x > 0 for x in m) / len(m) * 100:4.1f}" if m else "  -  ")
            print(f"  {lab:28s} yönlü %{len(dirn) / len(sub) * 100 if sub else 0:3.0f}  " + " / ".join(hs))
        neu = [r for r in sub if bv.bias_now(r) == "NEUTRAL"]
        for _k, lab, fn in bv.VARIANTS:
            m = [bv._sg(fn(r)) * r["fwd_1d"] for r in neu if bv._sg(fn(r)) and r["fwd_1d"] is not None]
            if m:
                print(f"    bugün NEUTRAL → {lab}: {len(m)} gün yön alır, 1g isabet %{sum(x > 0 for x in m) / len(m) * 100:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=bv.DECISION_SINCE.strftime("%Y-%m-%d"),
                    help="levels_at alt sınırı (karar: 2026-10-29; hipotez verisi 2026-10-04)")
    ap.add_argument("--karne-since", default="2026-07-01")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since)
    con = sqlite3.connect(REPO / "traderadar.db")
    con.row_factory = sqlite3.Row
    karne(con, args.karne_since)
    bmap = bv.build_map(dict(r) for r in con.execute("select * from bias_journal"))
    jr = [dict(r) for r in con.execute(
        "select * from setup_journal where strategy in ('4h','1d') and levels_at >= ?", (args.since,))]
    res = {}
    for _k, lab, fn in bv.VARIANTS:
        print(f"\n== Journal karşı-olgusu — {lab} (levels_at ≥ {args.since}, RR ≥ {bv.MIN_RR}, eşik {bv.THR})")
        for st in bv.STRATEGIES:
            a, d = bv.flips([r for r in jr if r["strategy"] == st], bmap, fn, since)
            sa, sd = bv.stats(a), bv.stats(d)
            print(f"  {st.upper()} EKLENEN {_fmt(sa)}")
            print(f"  {'':3s}DÜŞEN   {_fmt(sd)}")
            res[(_k, st)] = (a, d, sa, sd)

    a, d, sa, sd = res[("variant", "4h")]
    print(f"\n== KARAR (varyant, 4H; tetik eklenen {bv.NEED_ADDED} / düşen {bv.NEED_DROPPED} çözülmüş)")
    if sa["res"] < bv.NEED_ADDED or sd["res"] < bv.NEED_DROPPED:
        print(f"  veri birikiyor: eklenen {sa['res']}/{bv.NEED_ADDED} · düşen {sd['res']}/{bv.NEED_DROPPED}")
        return
    diff = sa["rps"] - sd["rps"]
    allx = sorted(a + d, key=lambda x: x["at"])
    mid = allx[len(allx) // 2]["at"]
    halves = []
    for part in ([x for x in allx if x["at"] < mid], [x for x in allx if x["at"] >= mid]):
        pa = bv.stats([x for x in part if x in a])
        pd = bv.stats([x for x in part if x in d])
        halves.append((pa["rps"] - pd["rps"]) if pa["res"] and pd["res"] else None)
    _a1, _d1, sa1, sd1 = res[("variant", "1d")]
    d1 = (sa1["rps"] - sd1["rps"]) if sa1["res"] and sd1["res"] else None
    print(f"  fark (eklenen − düşen) {diff:+.2f} R/setup · eklenen dolum R/işlem {sa['rpf']:+.2f} · yarılar {halves} · 1D fark {d1}")
    if diff <= -BAND or (sa["rpf"] is not None and sa["rpf"] <= 0):
        print("  K2 → kural aynı, madde kapanır")
    elif diff >= BAND and all(h is not None and h > 0 for h in halves) and not (d1 is not None and d1 <= -BAND):
        print("  K1 → tek replay (4H + 1D), karar kullanıcıya")
    else:
        print("  K3 → kullanıcıya sor")


if __name__ == "__main__":
    main()
