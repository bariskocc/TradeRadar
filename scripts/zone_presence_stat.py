"""Bolgesiz (IFVG/BPR yok) setup daha cok mu stop oluyor? (27.09, kullanici sorusu)

Kaynak `setup_journal` (canli sinyal degil): motorun CRT saydigi her setup, kapidan gecsin gecmesin. Bolge,
seviyeler dondugu andaki `entries` varyantlarindan okunur (`bpr_near` varsa BPR, `ifvg_near` varsa IFVG, ikisi de
yoksa YOK); 18.09 oncesi satirlarda `parts_at_levels.ifvg` (yalniz `ifvg_checked`, BPR ayrilamaz -> "IFVG*").
Sonuc Journal'in `outcome` kolonu (secilen giris, duz TP/SL: kismi kar + BE yok). Skor/RR donmus halden
(`parts_at_levels.score`, `entries.chosen.rr`).

Dilimler:
  B  C2 kapaliyken donmus (1H her zaman; 4H/1D `parts_at_levels.c2_closed = 1`) -- temiz kume
  C  4H, C2 acikken donmus (27.09 oncesi; SL kosan purge ucu) -- yanli, ayri basilir
  KARAR dilimi = B'de skor >= 7 ve RR >= 2 (motorun alacagi), yalniz `entries` olan satirlar (18.09+). Ayrica ayni giris (`cisd` varyanti) ile kiyas:
  fark setup'tan mi girisin kendisinden mi geliyor.
En altta canli sinyaller (kismi kar + BE dahil, `signals.result`).

Karar kurali: IZLEME.md "Bölgesiz setup daha çok mu stop oluyor?".
Kullanim: python scripts/zone_presence_stat.py [--db traderadar.db] [--since 2026-09-16]
"""
import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--since", default="2000-01-01")
A = ap.parse_args()

# motorun CRT saymadigi adaylar (PRE_SETUP_STAGES)
PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected"}
MIN_SCORE, MIN_RR = 7, 2.0


def zone_of(r, pal, fal, en):
    if en is not None:
        return "BPR" if "bpr_near" in en else ("IFVG" if "ifvg_near" in en else "YOK")
    if not fal.get("ifvg_checked"):
        return None
    return "IFVG*" if pal.get("ifvg") else "YOK"


def load(con):
    out = []
    for r in con.execute("select * from setup_journal where entry is not null and parts_at_levels is not null "
                         "and first_seen >= ?", (A.since,)):
        if r["best_stage"] in PRE:
            continue
        pal = json.loads(r["parts_at_levels"])
        fal = json.loads(r["features_at_levels"]) if r["features_at_levels"] else {}
        en = json.loads(r["entries"]) if r["entries"] else None
        z = zone_of(r, pal, fal, en)
        if z is None:
            continue
        ch = (en or {}).get("chosen") or {}
        cis = (en or {}).get("cisd")
        out.append(dict(
            r=r, z=z, zb="YOK" if z == "YOK" else "VAR",
            c2=1 if r["strategy"] == "1h" else pal.get("c2_closed"),
            o=r["outcome"], rr=ch.get("rr") or r["rr"] or 0.0, sc=pal.get("score", r["score"] or 0),
            imm="imm" if ch.get("imm") else "limit",
            cis=(cis.get("o"), cis.get("rr") or 0.0) if cis else None,
        ))
    return out


def tally(sub, get):
    w = l = 0
    R = 0.0
    for x in sub:
        o, rr = get(x)
        if o == "win":
            w += 1
            R += rr
        elif o == "loss":
            l += 1
            R -= 1
    return w, l, R


def table(title, sub, key, get=lambda x: (x["o"], x["rr"])):
    print(f"\n## {title}")
    print(f"  {'grup':<22}{'n':>5}{'TP':>5}{'SL':>5}{'SL%':>6}{'R':>8}{'R/setup':>9}")
    g = collections.defaultdict(list)
    for x in sub:
        g[key(x)].append(x)
    for k in sorted(g):
        w, l, R = tally(g[k], get)
        n = w + l
        print(f"  {k:<22}{n:>5}{w:>5}{l:>5}{(100 * l / n if n else 0):>6.1f}{R:>+8.1f}{(R / n if n else 0):>+9.3f}")


def band(sc):
    return "0-4" if sc < 5 else ("5-6" if sc < 7 else "7+")


con = sqlite3.connect(A.db)
con.row_factory = sqlite3.Row
rows = load(con)
B = [x for x in rows if x["c2"] == 1]
C = [x for x in rows if x["c2"] == 0]
trade = lambda x: x["sc"] >= MIN_SCORE and x["rr"] >= MIN_RR

print(f"Journal satiri (motorun CRT saydigi, bolgesi okunabilen): {len(rows)}  B={len(B)}  C={len(C)}")
print("\n" + "=" * 70 + "\nB: C2 KAPALI donmus, kapi fark etmez\n" + "=" * 70)
table("bolge", B, lambda x: x["zb"])
table("bolge ayrinti", B, lambda x: x["z"])
table("skor bandi x bolge", B, lambda x: band(x["sc"]) + "/" + x["zb"])
table("strateji x bolge", B, lambda x: x["r"]["strategy"] + "/" + x["zb"])
table("RR<2 / RR>=2 x bolge", B, lambda x: ("RR>=2" if x["rr"] >= MIN_RR else "RR<2") + "/" + x["zb"])
table("giris aninda doluyor mu (imm) x bolge", [x for x in B if x["r"]["entries"]], lambda x: x["imm"] + "/" + x["zb"])
table("AYNI GIRIS (cisd varyanti) x bolge", [x for x in B if x["cis"]], lambda x: x["zb"], lambda x: x["cis"])

print("\n" + "=" * 70 + "\nKARAR DILIMI: B, skor >= 7 ve RR >= 2\n" + "=" * 70)
K = [x for x in B if trade(x) and x["r"]["entries"]]   # 18.09+: bolge kesin, watchlist sayaci ayni kume
table("bolge", K, lambda x: x["zb"])
table("strateji x bolge", K, lambda x: x["r"]["strategy"] + "/" + x["zb"])
table("AYNI GIRIS (cisd) x bolge", [x for x in K if x["cis"]], lambda x: x["zb"], lambda x: x["cis"])

print("\n" + "=" * 70 + "\nC: 4H C2 ACIKKEN donmus (yanli dilim, yalniz yon teyidi)\n" + "=" * 70)
table("bolge", C, lambda x: x["zb"])
table("skor >= 7 ve RR >= 2 x bolge", [x for x in C if trade(x)], lambda x: x["zb"])

print("\n" + "=" * 70 + "\nCanli sinyaller (kismi kar + BE dahil)\n" + "=" * 70)
print(f"  {'grup':<22}{'n':>5}{'win':>5}{'BE':>4}{'loss':>5}{'R':>8}{'R/islem':>9}")
q = """select case when bpr_low is not null then 'BPR' when ifvg_low is not null then 'IFVG' else 'YOK' end z,
       result, rr_value from signals where result is not null and created_at >= ?"""
g = collections.defaultdict(list)
for z, res, rv in con.execute(q, (A.since,)):
    g[z].append((res, rv or 0.0))
for k in ("BPR", "IFVG", "YOK"):
    v = g.get(k, [])
    R = sum(rv for _, rv in v)
    print(f"  {k:<22}{len(v):>5}{sum(r == 'win' for r, _ in v):>5}{sum(r == 'breakeven' for r, _ in v):>4}"
          f"{sum(r == 'loss' for r, _ in v):>5}{R:>+8.1f}{(R / len(v) if v else 0):>+9.3f}")
