"""Zayif C2 geri donusu cezasi (-4) alan setuplar gercekten stop oluyor mu? -- strateji bazinda (01.10, kullanici).

Kaynak `setup_journal` (canli sinyal degil): motorun CRT saydigi her setup, kapidan gecsin gecmesin. Ceza alan setup
skor kapisinda elendigi icin sinyale donusmez; Journal onun sonrasini yine izler -- soru tam olarak bu: "elenmeseydi
TP mi SL mi?". Ceza `parts_at_levels.reclaim` (seviyeler dondugu andaki kirilim), sonuc Journal `outcome` (secilen
giris, duz TP/SL: kismi kar + BE yok); setup sinyale donustuyse `shadow["1"]` (ayni kurallar, izleme surer).

Gruplar (strateji basina):
  ceza     reclaim = C2_RECLAIM_PENALTY (-4)
  KARAR dilimi: ham skor >= esik (4H/1D 7, 1W 6) -- cezayi kaldirmak/yumusatmak yalniz bunlari sinyale cevirir
  cezasiz  reclaim = 0
Adil kiyas AYNI HAM SKOR bandinda: ham skor = cezadan onceki skor, `min(cap, raw - smt - reclaim) + smt` (tavan
SMT'den once -- CLAUDE.md "Skor karsi-olgusunu raw'dan hesapla"). Ceza alan setup zaten dusuk skorlu olabilir; o zaman
kotu sonucu cezadan degil setup'in kendisinden gelir. "ceza olmasa gecerdi" = ham skor >= stratejinin esigi.

Temiz kume: 4H'te yalniz C2 kapaliyken donmus satirlar (`parts_at_levels.c2_closed = 1`; aciktayken donan SL kosan purge
ucu). 1D/1W 25.09'dan beri C2 acikken islem aciyor -> filtre yok.

Karar kurali: IZLEME.md "Zayıf C2 geri dönüşü cezası (−4) alan setup gerçekten stop oluyor mu?".
Kullanim: python scripts/reclaim_penalty_stat.py [--db traderadar.db] [--since 2026-09-19]
  (--since varsayilani 19.09: ceza o gun -2 -> -4 oldu; oncesinde ceza alan satirda reclaim = -2)
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app import crt_engine as CE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--since", default="2026-09-19")
A = ap.parse_args()

PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected"}
PEN = -abs(CE.C2_RECLAIM_PENALTY)
CAP = {"4h": 9, "1d": 9, "1w": 8}
MIN_SCORE = {"4h": 7, "1d": 7, "1w": 6}
MIN_RESOLVED = 20      # karar icin ceza grubunda cozulmus setup (strateji basina)
BAND = 0.15            # R/setup farki


def outcome_of(r):
    o = r["outcome"]
    rr = None
    if o == "signal" and r["shadow"]:
        sh = (json.loads(r["shadow"]) or {}).get("1") or {}
        o, rr = sh.get("o"), sh.get("rr")
    if rr is None and r["entries"]:
        rr = ((json.loads(r["entries"]) or {}).get("chosen") or {}).get("rr")
    return o, (rr if rr is not None else r["rr"])


con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
recs = []
for r in con.execute("select * from setup_journal where entry is not null and parts_at_levels is not null "
                     "and first_seen >= ?", (A.since,)):
    if r["best_stage"] in PRE or r["strategy"] not in CAP:
        continue
    pal = json.loads(r["parts_at_levels"])
    if r["strategy"] == "4h" and pal.get("c2_closed") != 1:
        continue
    rec = pal.get("reclaim")
    if rec not in (0, PEN):
        continue
    o, rr = outcome_of(r)
    smt = pal.get("smt") or 0
    raw = pal.get("raw", pal.get("score", 0))
    pre = min(11, max(0, min(CAP[r["strategy"]], raw - smt - rec)) + smt)
    recs.append(dict(s=r["strategy"], pen=rec == PEN, o=o, rr=rr or 0.0, pre=pre, gp=pre >= MIN_SCORE[r["strategy"]]))


def stats(sub):
    win = [x for x in sub if x["o"] == "win"]
    loss = [x for x in sub if x["o"] == "loss"]
    res = len(win) + len(loss)
    r = sum(x["rr"] for x in win) - len(loss)
    return dict(n=len(sub), res=res, w=len(win), l=len(loss),
                tpb=sum(x["o"] == "tp_before_entry" for x in sub), nt=sum(x["o"] in ("no_touch", "open", "pending", "filled") for x in sub),
                wr=(len(win) / res * 100) if res else None, rps=(r / res) if res else None)


def line(name, st):
    wr = f"{st['wr']:.1f}" if st["wr"] is not None else "-"
    sl = f"{st['l'] / st['res'] * 100:.1f}" if st["res"] else "-"
    rps = f"{st['rps']:+.3f}" if st["rps"] is not None else "-"
    print(f"   {name:<30}{st['n']:>6}{st['res']:>7}{st['w']:>5}{st['l']:>5}{st['tpb']:>9}{st['nt']:>8}"
          f"{wr:>7}{sl:>7}{rps:>9}")


HDR = (f"   {'grup':<30}{'setup':>6}{'cozulmus':>9}{'TP':>5}{'SL':>5}{'TP once':>9}{'acik':>6}"
       f"{'WR%':>7}{'SL%':>7}{'R/setup':>9}")
BANDS = [("ham skor 0-4", 0, 4), ("ham skor 5-6", 5, 6), ("ham skor 7+", 7, 99)]

print(f"Ceza = {PEN} (C2_RECLAIM_PENALTY), esik C2 geri donusu < %{CE.C2_RECLAIM_WEAK_PCT:.0f}; "
      f"since {A.since}; sonuc = secilen giris, duz TP/SL")
print("R/setup cozulmus (TP/SL) setup basina; 'TP once' = giristen once TP, 'acik' = sonuclanmadi / dokunmadi")
summary = []
for strat in ("4h", "1d", "1w"):
    sub = [x for x in recs if x["s"] == strat]
    print(f"\n### {strat.upper()} -- {len(sub)} setup")
    if not sub:
        continue
    print(HDR)
    pen, base = [x for x in sub if x["pen"]], [x for x in sub if not x["pen"]]
    line("ceza (-4)", stats(pen))
    line("cezasiz", stats(base))
    print("   -- ayni ham skor bandinda (ham = cezadan onceki skor) --")
    for name, lo, hi in BANDS:
        bp = stats([x for x in pen if lo <= x["pre"] <= hi])
        bb = stats([x for x in base if lo <= x["pre"] <= hi])
        if not bp["n"] and not bb["n"]:
            continue
        line(f"{name}: ceza", bp)
        line(f"{name}: cezasiz", bb)
    # karar dilimi: ham skoru esigi gecen -- cezayi degistirmek yalniz bunlarin kaderini degistirir
    gp_p, gp_b = stats([x for x in pen if x["gp"]]), stats([x for x in base if x["gp"]])
    line(f"KARAR ham >= {MIN_SCORE[strat]}: ceza", gp_p)
    line(f"KARAR ham >= {MIN_SCORE[strat]}: cezasiz", gp_b)
    d = (gp_p["rps"] - gp_b["rps"]) if gp_p["res"] and gp_b["res"] else None
    summary.append((strat, gp_p, gp_b, d))

print(f"\nKARAR (IZLEME.md kurali): ham skoru esigi gecen ceza grubunda >= {MIN_RESOLVED} cozulmus setup; "
      f"fark = ceza - cezasiz R/setup (ikisi de ham skor >= esik), bant {BAND}")
for strat, gp_p, gp_b, d in summary:
    res = gp_p["res"]
    if res < MIN_RESOLVED:
        v = f"veri birikiyor ({res}/{MIN_RESOLVED})"
    elif d is None:
        v = "cezasiz grupta cozulmus yok"
    elif d <= -BAND:
        v = "ceza KALIR (ceza alanlar belirgin kotu)"
    else:
        v = "ceza bu stratejide GEVSETME ADAYI (-2) -- replay oncesi kullaniciya"
    wr = lambda st: f"{st['wr']:.0f}%" if st["wr"] is not None else "-"
    ds = f"{d:+.3f}" if d is not None else "-"
    print(f"   {strat.upper():<4} cozulmus {res:>3}  WR ceza {wr(gp_p):>4} / cezasiz {wr(gp_b):>4}  fark {ds:>7}  -> {v}")
