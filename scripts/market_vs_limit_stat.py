"""Market vs limit giris -- komisyonsuz (brut) ve komisyonlu (net) (27.09, kullanici).

Soru: sinyal aninda market emriyle girmek, bugunku limit girisinden kac R fazla/eksik getiriyor -- once
komisyonsuz, sonra komisyon dusulunce. Limit SEVIYESI sorusu (CISD/MSS/IFVG/BPR) ayri madde:
IZLEME "Geri cekilme derinligi" / tmp/tmp_limit_variant_depth.py.

Kaynak `setup_journal.retrace` ekseni (mum indirmez): 0 = seviye dondugu andaki fiyat (`ref`, market girisi),
1 = SL. Limit = `d_of.chosen` (bugunku giris). Stop kesri k: SL = x + k(1-x) (%100 bugunku purge ucu);
market kollari k = %100 / %90 / %80 / %70 / %60.
  tp_first True : x <= d_tp ise dolar; d_tp < s -> TP, degilse SL; x > d_tp -> TP giristen once (0R)
  tp_first False: dolar, SL
  ufuk doldu    : d_max >= x -> dolu/sonuclanmamis (0R brut, komisyonun giris yarisi dusulur), degilse dolmadi
  RR(x,k) = (rr0 + x) / (k (1-x))
Komisyon R cinsinden: fee_R = gidis-donus % x fiyat / risk (risk = span x k x (1-x)). Stop yuzde olarak dar
oldugu icin ayni % komisyon dar stopta daha cok R yer. Varsayilan BingX standart: market (taker+taker) %0.10,
limit (maker giris + taker cikis) %0.07. Kendi oranin: --fee-market / --fee-limit.

RR kapisi: "RR>=2" = motorun bugunku kurali yeni giris/stopla; "kapi yok" = her setup alinir.
Duz TP/SL (kismi kar + BE yok). Karar yalniz first_bar'li satirlarda (26.09+): eski satirlar sinyal anindaki ilk
mumu gormez -> d_tp kucuk -> market (sig giris) ve dar stop LEHINE yanli.

Karar kurali: IZLEME.md "Market vs limit giriş — komisyonsuz / komisyonlu".
Kullanim: python scripts/market_vs_limit_stat.py [--fee-market 0.10] [--fee-limit 0.07] [--all]
  --all : kapi fark etmez tum gercek setuplar (varsayilan: kalite kapilarini gecmis + tum; ikisi de basilir)
"""
import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--fee-market", type=float, default=0.10, help="market gidis-donus komisyonu, yuzde")
ap.add_argument("--fee-limit", type=float, default=0.07, help="limit gidis-donus komisyonu, yuzde")
A = ap.parse_args()

PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected"}
QUAL = {"bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr", "score7", "tight_stop"}
MIN_RR = 2.0

con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
recs = []
for r in con.execute("select * from setup_journal where retrace is not null and parts_at_levels is not null"):
    if r["best_stage"] in PRE:
        continue
    pal = json.loads(r["parts_at_levels"])
    if r["strategy"] in ("4h", "1d") and pal.get("c2_closed") != 1:
        continue
    rt = json.loads(r["retrace"])
    if rt.get("amb") or not rt.get("span") or not rt.get("ref"):
        continue
    if rt.get("tp_first") is None and not rt.get("done"):
        continue
    d = (rt.get("d_of") or {}).get("chosen")
    if d is None or d >= 1:
        continue
    long = r["direction"] == "LONG"
    recs.append(dict(r=r, rt=rt, d=max(0.0, d), gp=r["best_stage"] not in QUAL, fb=r["first_bar"] is not None,
                     rr0=((r["tp"] - rt["ref"]) if long else (rt["ref"] - r["tp"])) / rt["span"],
                     span_pct=rt["span"] / rt["ref"] * 100))


def trade(rec, x, k, fee_pct):
    """(durum, brut R, net R, RR) -- dolmayan emir 0/0."""
    rt = rec["rt"]
    rr = (rec["rr0"] + x) / (k * (1 - x))
    s = x + k * (1 - x)
    fee_r = fee_pct / (rec["span_pct"] * k * (1 - x))
    dtp = rt.get("d_tp") or 0.0
    if rt.get("tp_first") is True:
        if x > dtp:
            return "miss", 0.0, 0.0, rr
        return ("tp", rr, rr - fee_r, rr) if dtp < s else ("sl", -1.0, -1.0 - fee_r, rr)
    if rt.get("tp_first") is False:
        return "sl", -1.0, -1.0 - fee_r, rr
    if (rt.get("d_max") or 0.0) >= x:
        return "open", 0.0, -fee_r / 2, rr
    return "miss", 0.0, 0.0, rr


ARMS = [("limit (bugunku)", "limit", 1.0, True)]
# %90 / %70 (28.09, kullanici): "market + dar stop limitten kotu mu" sorusunu ara noktalarla da gormek icin.
for k in (1.0, 0.9, 0.8, 0.7, 0.6):
    for gate in (True, False):
        ARMS.append((f"market stop %{int(k * 100)} " + ("RR>=2" if gate else "kapi yok"), "market", k, gate))


def arm(sub, kind, k, gate):
    n = tp = sl = 0
    gross = net = 0.0
    fees = []
    for rec in sub:
        x = rec["d"] if kind == "limit" else 0.0
        fee = A.fee_limit if kind == "limit" else A.fee_market
        o, g, nt, rr = trade(rec, x, k, fee)
        if gate and rr < MIN_RR:
            continue
        if o == "miss":
            continue
        n += 1
        tp += o == "tp"
        sl += o == "sl"
        gross += g
        net += nt
        fees.append(g - nt)
    return n, tp, sl, gross, net, (statistics.median(fees) if fees else 0.0)


def table(title, sub):
    print(f"\n### {title} -- {len(sub)} setup")
    if not sub:
        return
    half = sorted(x["r"]["first_seen"] for x in sub)[len(sub) // 2]
    h1 = [x for x in sub if x["r"]["first_seen"] < half]
    h2 = [x for x in sub if x["r"]["first_seen"] >= half]
    base = {}
    print(f"   {'kol':<26}{'islem':>6}{'TP':>4}{'SL':>4}{'brut R':>8}{'net R':>8}{'kom/islem':>10}"
          f"{'brut fark':>10}{'net fark':>9}{'1.yari net':>11}{'2.yari net':>11}")
    for name, kind, k, gate in ARMS:
        n, tp, sl, g, nt, fm = arm(sub, kind, k, gate)
        n1 = arm(h1, kind, k, gate)[4]
        n2 = arm(h2, kind, k, gate)[4]
        if kind == "limit":
            base = dict(g=g, nt=nt, n1=n1, n2=n2)
        dg, dn = g - base["g"], nt - base["nt"]
        print(f"   {name:<26}{n:>6}{tp:>4}{sl:>4}{g:>+8.1f}{nt:>+8.1f}{fm:>10.2f}"
              f"{dg:>+10.1f}{dn:>+9.1f}{n1 - base['n1']:>+11.1f}{n2 - base['n2']:>+11.1f}")
    print(f"   (fark = market kolu - limit, toplam R; per setup icin {len(sub)}'e bol)")


print(f"Komisyon varsayimi: market %{A.fee_market:.2f}, limit %{A.fee_limit:.2f} gidis-donus")
print(f"Stop mesafesi (ref -> SL) medyan %{statistics.median(x['span_pct'] for x in recs):.2f} fiyat")
for name, pop in (("KAPILARI GECMIS", [x for x in recs if x["gp"]]), ("TUM GERCEK SETUP (kapi fark etmez)", recs)):
    print("\n" + "=" * 110 + f"\n{name}\n" + "=" * 110)
    table("hepsi (26.09 oncesi satirlar market lehine yanli)", pop)
    table("first_bar (26.09+, KARAR DILIMI)", [x for x in pop if x["fb"]])
    table("kripto", [x for x in pop if x["r"]["market_type"] == "crypto"])
    table("kripto disi", [x for x in pop if x["r"]["market_type"] != "crypto"])
