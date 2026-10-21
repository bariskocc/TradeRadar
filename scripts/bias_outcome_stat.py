"""1D bias kazanani seciyor mu? -- setup sonucu bias iliskisine gore (03.10, kullanici: "bias en onemli konu").

Kume: Journal, CRT-oncesi asamalar haric, 4H'te seviyeler C2 kapali donmus (parts_at_levels.c2_closed = 1), sonuc
win/loss (sinyale donenlerde golge shadow["1"]). R duz TP/SL: win = +rr, loss = -1. Kalite kapilari UYGULANMAZ --
ters bias'li setup hard filtrede elendigi icin kontrol grubu ancak boyle var (elenen setup'in sonrasini Journal izliyor).

Bias: setup'in seviyeleri dondugu ana (`levels_at`) kadar KAPANMIS son gunun bias karnesi satiri (`bias_journal`).
Motorun o anda kullandigi bias budur (scripts/bias_fidelity_stat.py gun donumu disinda %99.9 ayni).
UYARI: `setup_journal.htf_bias` kolonu KULLANILMAZ -- her degerlendirmede guncellenir, last_seen anina aittir;
03.10'da setuplarin ~%16'sinda seviye anindaki bias'tan farkliydi.

Gruplar setup yonune gore: hizali (LONG+BULLISH / SHORT+BEARISH), NEUTRAL, ters.
Karsi-olgu bias tanimlari (motor degismez, olcum): yalniz ict, yalniz structure.

Karar kurali: IZLEME.md "1D bias 4H'te kazananı seçiyor mu?" (03.10, veri gorulmeden; taze veri --since 2026-10-04)
ve "1D bias 1D'de kazananı seçiyor mu?" (03.10, butun veri, grup basina 50).
Kullanim: python scripts/bias_outcome_stat.py [--strategy 4h] [--since 2026-10-04] [--until YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app.setup_journal import levels_final  # noqa: E402

ALL_LEVELS = "--all-levels" in sys.argv

# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler (03.10) ---------------------------------
DECISION_SINCE = "2026-10-04"   # 4H taze veri: 03.10 okumasindan sonra donan seviyeler
MIN_GROUP = 100                 # 4H: hizali ve ters gruplarin her biri, cozulmus setup
# strateji -> (levels_at alt siniri, grup basina esik). 1D (03.10, eski "+2 puan" maddesinden devralindi): 1D'de setup
# seyrek, taze veriyle 50 aylar surer -> butun veri; esikler 4H'ten kopyalandi, 1D verisine bakilarak secilmedi.
DECISION = {"4h": (DECISION_SINCE, MIN_GROUP), "1d": (None, 50)}
EDGE_R = 0.15                   # R/setup farki
PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected"}
ORDER = ("hizali", "NEUTRAL", "ters")


def ts(s) -> datetime:
    return datetime.fromisoformat(str(s)[:19])


def rel(direction: str, b) -> str:
    if b not in ("BULLISH", "BEARISH"):
        return "NEUTRAL"
    return "hizali" if (direction == "LONG") == (b == "BULLISH") else "ters"


def load(db: str, strategy: str, since: str | None, until: str | None) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    karne: dict[str, list] = defaultdict(list)
    for r in con.execute("select * from bias_journal order by day"):
        karne[r["symbol"]].append((ts(r["day"]), dict(r)))
    keys = {k: [d for d, _ in v] for k, v in karne.items()}

    def day_row(sym, t):
        ks = keys.get(sym)
        if not ks:
            return None
        i = bisect_right(ks, t - timedelta(hours=24)) - 1   # gun + 24s <= t -> kapanmis gun
        return karne[sym][i][1] if i >= 0 else None

    out = []
    for r in con.execute("select * from setup_journal where strategy=? and parts_at_levels is not null", (strategy,)):
        if r["best_stage"] in PRE:
            continue
        if strategy != "4h" and not ALL_LEVELS and not levels_final(r):  # 03.10: 1D/1W'de yalniz seviyesi kesin satir (C2 acik donmus = kosan SL, SL yonune yanli); --all-levels eskiler dahil
            continue
        p = json.loads(r["parts_at_levels"])
        if strategy == "4h" and p.get("c2_closed") != 1:
            continue
        o, rr = r["outcome"], r["rr"]
        if o == "signal":
            sh = (json.loads(r["shadow"]) if r["shadow"] else {}).get("1") or {}
            o, rr = sh.get("o"), sh.get("rr", rr)
        if o not in ("win", "loss") or rr is None:
            continue
        t = ts(r["levels_at"] or r["first_seen"])
        if since and t < ts(since):
            continue
        if until and t >= ts(until):
            continue
        b = day_row(r["symbol"], t)
        if b is None:
            continue
        d = r["direction"]
        out.append(dict(t=t, mkt="kripto" if r["market_type"] == "crypto" else "FX", rr=rr, win=o == "win",
                        R=rr if o == "win" else -1.0, wk=f"W{t.isocalendar()[1]}",
                        eng=rel(d, b["combined"]), ict=rel(d, b["ict"]), st=rel(d, b["structure"]),
                        stale=b["stale_applied"]))
    return out


def stats(rs):
    n = len(rs)
    if not n:
        return 0, None, None
    return n, 100 * sum(x["win"] for x in rs) / n, sum(x["R"] for x in rs) / n


def fmt(rs) -> str:
    n, wr, r = stats(rs)
    return "n=   0" if not n else f"n={n:4d}  WR %{wr:5.1f}  R/setup {r:+.3f}"


def block(title, rs, key="eng"):
    print(f"\n## {title}")
    for g in ORDER:
        print(f"   {g:<8} {fmt([x for x in rs if x[key] == g])}")


def diff(rs, key, a, b):
    """(R/setup a - b, WR a - b) -- iki grup da doluysa."""
    na, wa, ra = stats([x for x in rs if x[key] == a])
    nb, wb, rb = stats([x for x in rs if x[key] == b])
    if not na or not nb:
        return None, None
    return ra - rb, wa - wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="4h", choices=("4h", "1d", "1w"))
    ap.add_argument("--since", default=None, help=f"levels_at alt siniri (karar dilimi {DECISION_SINCE})")
    ap.add_argument("--all-levels", action="store_true", help="1D/1W: C2 acikken donmus (yanli) eski satirlar da")
    ap.add_argument("--until", default=None)
    A = ap.parse_args()

    recs = load(A.db, A.strategy, A.since, A.until)
    print(f"Strateji {A.strategy} · {len(recs)} cozulmus setup · levels_at {A.since or 'hepsi'} .. {A.until or 'bugun'}")
    block("1) Motorun bias'i (birlesik, seviye aninda)", recs)
    block("2) RR >= 2", [x for x in recs if x["rr"] >= 2])
    for m in ("kripto", "FX"):
        block(f"3) {m}", [x for x in recs if x["mkt"] == m])
    print("\n## 4) Haftalar (WR: hizali / NEUTRAL / ters)")
    for wk in sorted({x["wk"] for x in recs}):
        rs = [x for x in recs if x["wk"] == wk]
        cells = []
        for g in ORDER:
            n, wr, _ = stats([x for x in rs if x["eng"] == g])
            cells.append(f"{g} {wr:5.1f}% (n {n})" if n else f"{g} -")
        print(f"   {wk}  " + " | ".join(cells))
    block("5) Karsi-olgu: yalniz ict bias", recs, "ict")
    block("6) Karsi-olgu: yalniz structure bias", recs, "st")
    for sv, nm in ((0, "taze structure (AND dali)"), (1, "bayat structure -> karar ict'de")):
        block(f"7) {nm}", [x for x in recs if x["stale"] == sv])

    # --- KARAR (IZLEME '1D bias 4H'te kazananı seçiyor mu?', 03.10) ---
    if A.strategy not in DECISION:
        print("\nKARAR: bu strateji icin kural yok (yalniz 4H ve 1D); bilgi amacli")
        return
    since, need = DECISION[A.strategy]
    dec = load(A.db, A.strategy, since, None)
    print("\n" + "=" * 100 + f"\nKARAR (03.10 kurali; {A.strategy}, levels_at >= {since or 'hepsi'}, grup basina {need})\n"
          + "=" * 100)
    n_al = sum(x["eng"] == "hizali" for x in dec)
    n_op = sum(x["eng"] == "ters" for x in dec)
    print(f"   cozulmus setup {len(dec)} · hizali {n_al}/{need} · ters {n_op}/{need}")
    if n_al < need or n_op < need:
        print("   -> veri birikiyor")
        return
    half = sorted(x["t"] for x in dec)[len(dec) // 2]
    halves = ([x for x in dec if x["t"] < half], [x for x in dec if x["t"] >= half])

    def both_halves(key, a, b, sign):
        ds = [diff(h, key, a, b)[0] for h in halves]
        return all(d is not None and (d > 0 if sign > 0 else d < 0) for d in ds), ds

    d_op_al, w_op_al = diff(dec, "eng", "ters", "hizali")
    d_op_ne, _ = diff(dec, "eng", "ters", "NEUTRAL")
    ok_b1, h_b1 = both_halves("eng", "ters", "hizali", +1)
    ok_b2, h_b2 = both_halves("eng", "ters", "NEUTRAL", -1)
    print(f"   ters - hizali : {d_op_al:+.3f} R/setup, WR {w_op_al:+.1f} puan, yarilar "
          + " / ".join(f"{d:+.3f}" if d is not None else "-" for d in h_b1))
    print(f"   ters - NEUTRAL: {d_op_ne:+.3f} R/setup, yarilar "
          + " / ".join(f"{d:+.3f}" if d is not None else "-" for d in h_b2))
    if d_op_al >= EDGE_R and w_op_al >= 0 and ok_b1:
        print("   B1: TERS bias'li setup hizalidan iyi -> 4H hard filtre + htf kalemi icin TEK REPLAY (motor degismez)")
    elif d_op_ne is not None and d_op_ne <= -EDGE_R and ok_b2:
        print("   B2: ters bias'li setup NEUTRAL'dan kotu -> filtre dogru, kalir")
    else:
        print("   B3: bias 4H'te kazanani ayirt etmiyor -> degisiklik yok, kullaniciya sor")
    d_ict, _ = diff(dec, "ict", "hizali", "ters")
    ok_ict, h_ict = both_halves("ict", "hizali", "ters", +1)
    print(f"   ict-only: hizali - ters {d_ict:+.3f} R/setup, yarilar "
          + " / ".join(f"{d:+.3f}" if d is not None else "-" for d in h_ict))
    if d_ict is not None and d_ict >= EDGE_R and ok_ict:
        print("   C1: yalniz ict bias kazanani ayiriyor -> 'ict bias' adayi (replay ile)")
    else:
        print("   C1 yok: yalniz ict bias da ayirmiyor")


if __name__ == "__main__":
    main()
