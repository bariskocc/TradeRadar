"""Skor kalemi basina sonuc: bu puani alan setuplarin kaci TP, kaci SL? -- strateji bazinda (01.10, kullanici).

Soru (kullanici): "-4 cezasi dogru calisiyor mu, bu zamana kadar bu cezayi alanlarin kac tanesi stop olmus kac
tanesi TP olmus -- her kalem icin." Her kalem, aldigi her DEGERE gore ayri satir (ör. reclaim -4 / 0, htf +2 / 0 / -2):
setup, cozulmus, TP, SL, giristen once TP, acik, WR, R/setup.

Kaynak `setup_journal` (canli sinyal degil): motorun CRT saydigi her setup, kapidan gecsin gecmesin -- ceza alan setup
skor kapisinda elenir ama Journal sonrasini izler, soru tam olarak "elenmeseydi ne olurdu". Kalem degeri
`parts_at_levels` (seviyeler dondugu andaki kirilim), sonuc `outcome` (secilen giris, DUZ TP/SL: kismi kar + BE yok);
sinyale donustuyse `shadow["1"]`. 4H'te yalniz C2 kapaliyken donmus satirlar (aciktayken donan SL kosan purge ucu).

HUKUM (kalem basina, "puan alan" = deger != 0, "almayan" = 0; htf'de +2 ve -2 ayri ayri 0'a karsi).
HUKUM yalniz KARAR BANDINDAN (01.10): kalemin sinyali degistirdigi setuplar -- bu kalem haric skor oyle ki odul onu
esigin ustune tasir, ceza altina iter (4H/1D esik 7, 1W 6); ayni bantta puan alan vs almayan, taraf basina >= 10 cozulmus.
Eski "tum bantlar" ortalamasi bilgi olarak basilir; ilk denemede 4H -4 cezasinin bant 5-6 (+0.51) ve 7+ (-0.44) farklari
birbirini sifirliyordu, oysa karar yalniz 7+ bandini etkiler. Ham ve tum-bant fark: kalemler birbirine bagli (ceza alan setup zaten dusuk skorlu, SMT'li zaten yuksek), ham fark
yaniltir -- 01.10'da 4H -4 cezasi hamda "fark yok", ayni bantta belirgin kotu cikti. Bant = bu kalem HARIC skor
(raw - deger): 0-4 / 5-6 / 7+; fark her bantta (puan alan - almayan), puan alanin cozulmus sayisiyla agirlikli. Ham fark bilgi.
  odul (+)  : puan alanin R/setup'i almayandan >= 0.15 iyi -> "dogru calisiyor"; |fark| < 0.15 -> "fark yok";
              <= -0.15 -> "TERS" (puan verdigimiz setup daha kotu)
  ceza (-)  : ceza alanin R/setup'i >= 0.15 kotu -> "dogru calisiyor"; ... ayni bantlar ters yonde
  iki tarafta da >= MIN_RESOLVED cozulmus yoksa "veri az".
-4 cezasinin "ceza olmasa sinyal olurdu" dilimi ayrica `scripts/reclaim_penalty_stat.py`'de.
⚠️ Kural degisiklikleri: key level kalemleri (pd_*) 29.09'da yeniden tanimlandi -> `--since 2026-09-29`;
1D c2_closed/base 25.09 oncesi hep 1 / yanlis renk 1 -> `--since 2026-09-25`; reclaim 19.09 oncesi -2 (ayri satirda).

Kullanim: python scripts/score_item_outcome_stat.py [--strategy 4h|1d|1w] [--since 2026-09-16] [--db traderadar.db]
Karar kurali: IZLEME.md "Skor kalemleri — 4H" / "— 1D" / "— 1W".
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app.crt_engine import SCORE_PART_KEYS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--strategy", choices=("4h", "1d", "1w"))
ap.add_argument("--since", default="2026-09-16", help="kalem kirilimi 16.09'dan beri var")
A = ap.parse_args()

PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected"}
KEYS = [k for k in SCORE_PART_KEYS if k not in ("raw", "score")]
LABELS = {
    "base": "C2 rengi (dogru +2 / doji +1 / yanlis 0)",
    "htf": "Bias (1D; 1W'de aylik) +2 / ters -2",
    "weekly": "1W uyumu +1",
    "c2_closed": "C2 kapali +1",
    "pd_major": "Key level major +1",
    "pd_monthly": "Key level aylik +1",
    "pd_struct": "Key level FVG/OB +1",
    "wick": "Purge fitili +1",
    "ifvg": "LTF IFVG +1",
    "reclaim": "Zayif C2 geri donusu (ceza)",
    "smt": "SMT +2",
}
MIN_RESOLVED = 20
BAND = 0.15


def outcome_of(r):
    o, rr = r["outcome"], None
    if o == "signal" and r["shadow"]:
        sh = (json.loads(r["shadow"]) or {}).get("1") or {}
        o, rr = sh.get("o"), sh.get("rr")
    if rr is None and r["entries"]:
        rr = ((json.loads(r["entries"]) or {}).get("chosen") or {}).get("rr")
    return o, (rr if rr is not None else (r["rr"] or 0.0))


def stats(sub):
    w = [x for x in sub if x["o"] == "win"]
    l_ = [x for x in sub if x["o"] == "loss"]
    res = len(w) + len(l_)
    return dict(n=len(sub), res=res, w=len(w), l=len(l_),
                tpb=sum(x["o"] == "tp_before_entry" for x in sub),
                op=sum(x["o"] in ("open", "pending", "filled", "no_touch", "ambiguous", None) for x in sub),
                wr=(len(w) / res * 100) if res else None,
                rps=((sum(x["rr"] for x in w) - len(l_)) / res) if res else None)


def fmt(v, f):
    return format(v, f) if v is not None else "-"


BANDS = ((0, 4), (5, 6), (7, 99))
THR = {"4h": 7, "1d": 7, "1w": 6}
MIN_FLIP = 10          # karar bandinda taraf basina cozulmus


def flip_diff(sub, k, v, thr):
    """KARAR BANDI: kalemin sinyali degistirdigi setuplar -- bu kalem HARIC skor (raw - deger) oyle ki:
    odul (v > 0): pre in [thr - v, thr - 1]  (puan alirsa esigi gecer, almazsa gecemez)
    ceza (v < 0): pre in [thr, thr - v - 1]  (ceza almazsa gecer, alirsa duser)
    Ayni bantta deger = v olanlar ile 0 olanlar kiyaslanir. (got, base, fark) doner."""
    lo, hi = (thr - v, thr - 1) if v > 0 else (thr, thr - v - 1)
    band = [x for x in sub if lo <= (x["pal"].get("raw", 0) - (x["pal"].get(k) or 0)) <= hi]
    g = stats([x for x in band if x["pal"].get(k) == v])
    b = stats([x for x in band if x["pal"].get(k) == 0])
    d = (g["rps"] - b["rps"]) if g["res"] and b["res"] else None
    return g, b, d, (lo, hi)


def controlled_diff(sub, k, v):
    """Bu kalem HARIC skor bandinda (puan alan - almayan) R/setup farki, puan alanin cozulmusuyle agirlikli."""
    num = w = 0.0
    for lo, hi in BANDS:
        band = [x for x in sub if lo <= (x["pal"].get("raw", 0) - (x["pal"].get(k) or 0)) <= hi]
        g = stats([x for x in band if x["pal"].get(k) == v])
        b = stats([x for x in band if x["pal"].get(k) == 0])
        if g["res"] and b["res"]:
            num += g["res"] * (g["rps"] - b["rps"])
            w += g["res"]
    return (num / w) if w else None


def verdict(fg, fb, fd, penalty):
    """Hukum YALNIZ karar bandindan: kalemin sinyali degistirdigi setuplar (taraf basina >= MIN_FLIP cozulmus)."""
    if fg["res"] < MIN_FLIP or fb["res"] < MIN_FLIP or fd is None:
        return f"veri az (karar bandi {fg['res']}/{fb['res']}, taraf basina {MIN_FLIP})"
    good = -fd if penalty else fd
    if good >= BAND:
        return f"DOGRU CALISIYOR (karar bandi fark {fd:+.2f})"
    if good <= -BAND:
        return f"TERS (karar bandi fark {fd:+.2f})"
    return f"fark yok (karar bandi fark {fd:+.2f})"


con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = []
for r in con.execute("select * from setup_journal where parts_at_levels is not null and entry is not null "
                     "and first_seen >= ?", (A.since,)):
    if r["best_stage"] in PRE or r["strategy"] not in ("4h", "1d", "1w"):
        continue
    if A.strategy and r["strategy"] != A.strategy:
        continue
    pal = json.loads(r["parts_at_levels"])
    if r["strategy"] == "4h" and pal.get("c2_closed") != 1:
        continue
    o, rr = outcome_of(r)
    rows.append(dict(s=r["strategy"], pal=pal, o=o, rr=rr or 0.0))

print(f"since {A.since} · sonuc = secilen giris, duz TP/SL · R/setup cozulmus (TP/SL) basina · "
      f"'TP once' = giristen once TP (islem olmazdi)")
HDR = (f"   {'deger':>6}{'setup':>7}{'cozulmus':>9}{'TP':>5}{'SL':>5}{'TP once':>8}{'acik':>6}"
       f"{'WR%':>7}{'R/setup':>9}")
for strat in ("4h", "1d", "1w"):
    sub = [x for x in rows if x["s"] == strat]
    if not sub:
        continue
    print(f"\n{'=' * 78}\n{strat.upper()} -- {len(sub)} setup ({stats(sub)['res']} cozulmus)\n{'=' * 78}")
    for k in KEYS:
        vals = sorted({x["pal"].get(k) for x in sub if x["pal"].get(k) is not None}, reverse=True)
        if not vals:
            continue
        print(f"\n  {k} -- {LABELS.get(k, k)}")
        print(HDR)
        by = {v: stats([x for x in sub if x["pal"].get(k) == v]) for v in vals}
        for v in vals:
            st = by[v]
            print(f"   {v:>+6}{st['n']:>7}{st['res']:>9}{st['w']:>5}{st['l']:>5}{st['tpb']:>8}{st['op']:>6}"
                  f"{fmt(st['wr'], '.1f'):>7}{fmt(st['rps'], '+.3f'):>9}")
        if len(vals) == 1:
            print(f"   HUKUM: etkisi yok -- butun setuplar ayni degeri aliyor ({vals[0]:+})")
            continue
        zero = by.get(0)
        if zero is None:
            print("   HUKUM: 0 degeri yok, karsilastirma yapilamadi")
            continue
        for v in vals:
            if v == 0:
                continue
            d = controlled_diff(sub, k, v)
            raw_d = (by[v]["rps"] - zero["rps"]) if by[v]["res"] and zero["res"] else None
            fg, fb, fd, (lo, hi) = flip_diff(sub, k, v, THR[strat])
            print(f"   HUKUM {v:+} vs 0: {verdict(fg, fb, fd, v < 0)}")
            print(f"      karar bandi (bu kalem haric skor {lo}-{hi}): puan alan {fg['res']} cozulmus "
                  f"{fg['w']} TP / {fg['l']} SL, R/setup {fmt(fg['rps'], '+.2f')} · almayan {fb['res']} cozulmus "
                  f"{fb['w']} TP / {fb['l']} SL, R/setup {fmt(fb['rps'], '+.2f')}")
            print(f"      bilgi: ham fark {fmt(raw_d, '+.2f')} · tum bantlar kontrollu {fmt(d, '+.2f')}")
