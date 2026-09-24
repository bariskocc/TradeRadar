"""1D bias DOGRU HESAPLANIYOR MU? (Setup Journal x bias karnesi, terminal)

Soru (kullanici, 24.09): bias'in yonu isabetli mi sorusu ayri (karne: `scripts/bias_stat.py`,
setup ufku: Journal). Bu rapor BASKA bir seye bakar: motor bir setup'i degerlendirdigi anda
kullandigi 1D bias, o an KAPANMIS son gunun bias'iyla ayni mi? Tanim geregi ayni olmali.
Farkliysa hesap yanlis veriyle yapilmistir (olusan mum, bayat gunluk seri, seans kovasi) --
gecmisteki bias hatalarinin hepsi bu turdendi (11.09 yarim mum, 18.09 gece yarisi PD kaymasi).

Iki kaynak:
  * `setup_journal.htf_bias`  -- motorun kullandigi bias. Her degerlendirmede guncellenir, yani
    `last_seen` anina aittir (onceki degerlendirmeler gorunmez -> hata sayisi ALT SINIRDIR).
  * `bias_journal.combined`   -- ayni sembol, `last_seen`'den once kapanan son gun. Karne gun
    kapandiktan SONRA, tam seriden, motorun kendi saf fonksiyonlariyla yazilir -> referans.

Ikinci kontrol, seviyeler donarken: `parts_at_levels["htf"]` (+2 / 0 / -2) referans bias'la
tutarli mi? Hizali -> +2, ters -> -2, NEUTRAL -> 0 ya da +2 (reversal-at-PD +2 verebilir).
`levels_at` anina gore ayni referans kullanilir.

Dilim: gun donumunden sonraki ilk 35 dk (kripto 00:00 UTC, seans sembolleri kendi anchor'i) vs
gunun geri kalani. Hata gun donumunde toplaniyorsa sebep veri tazeligidir, formul degil.

Kullanim:
    python scripts/bias_fidelity_stat.py            # tum Journal
    python scripts/bias_fidelity_stat.py --days 7
    python scripts/bias_fidelity_stat.py --examples 20
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BOUNDARY_MIN = 35   # gun donumunden sonraki ilk bakim turu (<= 30 dk) + pay
DAY = timedelta(hours=24)


def _dt(v) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v)[:19])
    except ValueError:
        return None


def _load_karne(con) -> dict[str, tuple[list[datetime], list[str]]]:
    out: dict[str, list] = {}
    for day, sym, comb in con.execute("SELECT day, symbol, combined FROM bias_journal"):
        d = _dt(day)
        if d is not None and comb:
            out.setdefault(sym, []).append((d, comb))
    res = {}
    for sym, rows in out.items():
        rows.sort()
        res[sym] = ([d for d, _ in rows], [c for _, c in rows])
    return res


def _ref(karne, sym: str, t: datetime):
    """`t`'den once KAPANMIS son gunun bias'i + o gunun kapanis ani. Karne o gunu hic
    gormediyse (seri baslangicindan once) None."""
    k = karne.get(sym)
    if not k:
        return None, None
    days, combs = k
    i = bisect.bisect_right(days, t - DAY) - 1      # day + 24s <= t
    if i < 0:
        return None, None
    return combs[i], days[i] + DAY


def _expected_htf(direction: str, ref: str) -> set[int]:
    aligned = (direction == "LONG" and ref == "BULLISH") or (direction == "SHORT" and ref == "BEARISH")
    contrary = (direction == "LONG" and ref == "BEARISH") or (direction == "SHORT" and ref == "BULLISH")
    if aligned:
        return {2}
    if contrary:
        return {-2}
    return {0, 2}                                   # NEUTRAL: reversal-at-PD +2 verebilir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    karne = _load_karne(con)
    sql = ("SELECT id, strategy, symbol, market_type, direction, last_seen, htf_bias, "
           "levels_at, parts_at_levels FROM setup_journal WHERE htf_bias IS NOT NULL")
    params: list = []
    if args.days:
        sql += " AND last_seen >= ?"
        params.append((datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S"))

    # (dilim, strateji) -> [ayni, farkli]
    live: dict[tuple[str, str], list[int]] = {}
    lev: dict[str, list[int]] = {}
    no_ref = 0
    bad_live, bad_lev = [], []
    for rid, strat, sym, mkt, direc, ls, hb, la, pal in con.execute(sql, params):
        t = _dt(ls)
        ref, closed_at = _ref(karne, sym, t) if t else (None, None)
        if ref is None:
            no_ref += 1
            continue
        edge = "gun donumu" if (t - closed_at) <= timedelta(minutes=BOUNDARY_MIN) else "gunun geri kalani"
        c = live.setdefault((edge, strat), [0, 0])
        ok = hb == ref
        c[0 if ok else 1] += 1
        if not ok:
            bad_live.append((rid, strat, sym, str(ls)[:16], int((t - closed_at).total_seconds() // 60), hb, ref))

        tl = _dt(la)
        if tl and pal:
            try:
                htf = json.loads(pal).get("htf")
            except ValueError:
                htf = None
            ref2, closed2 = _ref(karne, sym, tl)
            if htf is not None and ref2 is not None:
                edge2 = "gun donumu" if (tl - closed2) <= timedelta(minutes=BOUNDARY_MIN) else "gunun geri kalani"
                c2 = lev.setdefault(edge2, [0, 0])
                ok2 = int(htf) in _expected_htf(direc, ref2)
                c2[0 if ok2 else 1] += 1
                if not ok2:
                    bad_lev.append((rid, strat, sym, str(la)[:16], int((tl - closed2).total_seconds() // 60), direc, htf, ref2))

    tot_ok = sum(v[0] for v in live.values())
    tot_bad = sum(v[1] for v in live.values())
    print("=" * 78)
    print("  1D bias dogru hesaplaniyor mu?  (motorun kullandigi vs kapanmis gunun bias'i)")
    print("=" * 78)
    n = tot_ok + tot_bad
    if not n:
        print("  Karsilastirilacak setup yok (karne bos ya da Journal'da bias yok).")
        return
    print(f"  {n} setup karsilastirildi -- ayni {tot_ok} (%{100*tot_ok/n:.1f}), FARKLI {tot_bad}"
          f" (%{100*tot_bad/n:.1f})   [karnede referansi olmayan: {no_ref}]")
    print(f"  Not: Journal yalniz SON degerlendirmenin bias'ini tutar -> farkli sayisi alt sinirdir.\n")
    print(f"  {'dilim':20s} {'strateji':8s} {'ayni':>6s} {'farkli':>7s} {'hata %':>7s}")
    for (edge, strat), (a, b) in sorted(live.items()):
        print(f"  {edge:20s} {strat:8s} {a:6d} {b:7d} {100*b/(a+b):6.1f}%")

    if lev:
        print(f"\n  Seviyeler donarken skorun 1D kalemi (+2/0/-2) referansla tutarli mi:")
        for edge, (a, b) in sorted(lev.items()):
            print(f"  {edge:20s} tutarli {a:6d}  tutarsiz {b:5d}  (%{100*b/max(1,a+b):.1f})")

    if bad_live and args.examples:
        print(f"\n  Ornekler (motorun bias'i != kapanmis gunun bias'i), ilk {args.examples}:")
        print(f"  {'id':>6s} {'str':4s} {'sembol':14s} {'an (UTC)':16s} {'gun kapanisindan':>16s}  motor -> dogrusu")
        for rid, strat, sym, ls, mins, hb, ref in bad_live[: args.examples]:
            print(f"  {rid:6d} {strat:4s} {sym:14s} {ls:16s} {mins:13d} dk  {hb} -> {ref}")
    if bad_lev and args.examples:
        print(f"\n  Skor kalemi tutarsiz ornekler, ilk {args.examples}:")
        for rid, strat, sym, la, mins, direc, htf, ref in bad_lev[: args.examples]:
            print(f"  {rid:6d} {strat:4s} {sym:14s} {la:16s} {mins:6d} dk  {direc:5s} htf {htf:+d}  dogru bias {ref}")

    edge_bad = sum(v[1] for (e, _), v in live.items() if e == "gun donumu")
    print("\n  Yorum:")
    if tot_bad == 0:
        print("  Motorun kullandigi bias her setupta kapanmis gunun bias'iyla ayni. Hesap dogru.")
    elif edge_bad == tot_bad:
        print("  Hatalarin HEPSI gun donumunden sonraki ilk dakikalarda: formul degil VERI TAZELIGI.")
        print("  Gun kapanir kapanmaz gunluk seri henuz yenilenmemis, bayat kapanisla hesaplaniyor.")
    else:
        print(f"  {tot_bad - edge_bad} hata gun donumu DISINDA -- veri tazeligiyle aciklanamaz, incele.")
    print("  Karar kurali: IZLEME.md -> '1D bias hesap dogrulugu'.")


if __name__ == "__main__":
    main()
