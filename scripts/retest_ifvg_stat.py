"""CISD retest sonrasi IFVG -- ikinci konfirme mi? (03.10, kullanici)

Soru: CISD seviyesi retest edildikten sonra LTF'de yeni bir IFVG olusursa (LONG: dusus FVG'si, ustunde kapanis)
(1) ne siklikta olusuyor, (2) fiyat o bolgeye geri geliyor mu, (3) o girisler TP'ye gidiyor mu -- AYNI setupta
CISD girisiyle kiyasla. Kaynak Setup Journal `entries`: `cisd` varyanti + gec eklenen `rfvg_near|mid|far`
(`app/setup_journal.py` "CISD retest SONRASI IFVG"). Yalniz 03.10 restart'indan sonra seviyelenen satirlarda var.
Duz TP/SL (kismi kar + BE yok, komisyon yok); SL/TP ortak, yalniz giris degisir.

Iki kume (strateji basina ayri; 4H'te yalniz C2 kapaliyken donmus satir):
  A) KAPIDAN GECMIS: kalite kapilarini gecmis + CISD girisi RR >= 2 -- soru: "CISD'de dolma, retest sonrasi IFVG'yi
     bekle" daha mi iyi? Eslesik R/setup: B (yalniz rfvg_near; IFVG olusmadi / dolmadi / sonuclanmadi = 0) - A (cisd;
     sonuclanmadi = 0), CISD'si retest edilmis setuplarda.
  B) CISD RR < 2 (low_rr) ama diger kapilar gecmis -- ONDO 01.10 tipi: CISD acilmazdi; retest sonrasi IFVG'nin RR'si
     >= 2 ise islem acilsaydi R/islem ne olurdu? (Motor bunu bugun kismen yapiyor: en yakin bolge RR >= 2 ise.)

Onceden kayitli karar kurali (IZLEME.md "CISD retest sonrasi IFVG — ikinci konfirme mi?"):
  A: retest + IFVG olusmus >= 30 setup (strateji basina). B - A >= +0.15 R/setup -> "IFVG'yi bekle" kurali kullaniciya
     (replay sonrasi); <= -0.15 -> CISD kalir, madde kapanir; arada -> fark yok, CISD kalir.
  B: dolmus + sonuclanmis rfvg_near (RR >= 2) >= 20 islem. R/islem >= +0.15 -> bu girisler kazandiriyor (motorun
     bolge secimi bu tipi zaten aliyor; kural gerekmez, not); <= -0.15 -> bu tip girisi engelleme sorusu kullaniciya.
Bilgi (karar degil): retest sonrasi IFVG OLUSAN setuplarda CISD girisinin WR'si olusmayanlara gore -- IFVG, CISD
dolduktan sonra goruldugu icin filtre olamaz; yalniz "konfirme" sezgisini gosterir.

Kullanim: python scripts/retest_ifvg_stat.py [--strategy 4h|1d|1w] [--db traderadar.db]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--strategy", choices=("4h", "1d", "1w"))
A = ap.parse_args()

QUAL_FAIL = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected",
             "bias_mismatch", "target_taken", "low_quality", "no_cisd", "score7", "tight_stop"}
MIN_RR = 2.0
BAND = 0.15
MIN_A, MIN_B = 30, 20


def r_of(v):
    """Duz TP/SL R: win = +rr, loss = -1, digerleri 0. (sonuc, R)"""
    if not v:
        return None, 0.0
    o = v.get("o")
    if o == "win":
        return o, float(v.get("rr") or 0)
    if o == "loss":
        return o, -1.0
    return o, 0.0


def wr(xs):
    w = sum(1 for o, _ in xs if o == "win")
    lo = sum(1 for o, _ in xs if o == "loss")
    return w, lo, (w / (w + lo) * 100 if w + lo else None)


con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = []
for r in con.execute("select strategy, symbol, best_stage, entries, parts_at_levels from setup_journal "
                     "where entries like '%\"_rfvg\"%'"):
    if A.strategy and r["strategy"] != A.strategy:
        continue
    if r["best_stage"] in QUAL_FAIL:
        continue
    pal = json.loads(r["parts_at_levels"]) if r["parts_at_levels"] else {}
    if r["strategy"] == "4h" and pal.get("c2_closed") != 1:
        continue
    ents = json.loads(r["entries"]) or {}
    st, cv = ents.get("_rfvg") or {}, ents.get("cisd") or {}
    if st.get("imm") or cv.get("imm") or cv.get("o") == "invalid":
        continue                  # CISD seviyesi donarken zaten gecilmis: "retest" sorusu dogmuyor
    rows.append(dict(s=r["strategy"], st=st, cv=cv, near=ents.get("rfvg_near"),
                     mid=ents.get("rfvg_mid"), far=ents.get("rfvg_far"),
                     low_rr=r["best_stage"] == "low_rr" or (cv.get("rr") or 0) < MIN_RR))

print("CISD retest sonrasi IFVG -- duz TP/SL, ayni setupta CISD ile kiyas. 'acik' = durum henuz kapanmadi.")
for strat in ("4h", "1d", "1w"):
    sub = [x for x in rows if x["s"] == strat]
    if not sub:
        print(f"\n{strat.upper()}: veri yok (03.10 restart'indan sonra seviyelenen satir bekleniyor)")
        continue
    print(f"\n{'=' * 78}\n{strat.upper()} -- {len(sub)} setup (retest/IFVG takibi olan)\n{'=' * 78}")
    # Huni
    done = [x for x in sub if x["st"].get("done")]
    retested = [x for x in done if x["st"].get("rt")]
    formed = [x for x in retested if x["near"] and x["near"].get("e") is not None]
    why = {}
    for x in done:
        if not (x["near"] and x["near"].get("e") is not None):
            k = (x["near"] or {}).get("why") or "?"
            why[k] = why.get(k, 0) + 1
    filled = [x for x in formed if x["near"].get("t")]
    print(f"  huni: durum kapandi {len(done)} (acik {len(sub) - len(done)}) -> CISD retest {len(retested)} -> "
          f"retest sonrasi IFVG {len(formed)} -> near dolumu {len(filled)}")
    print(f"  IFVG cikmayan nedenler: " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    for name in ("near", "mid", "far"):
        vs = [r_of(x[name]) for x in formed]
        w, lo, p = wr(vs)
        rr = [x[name].get("rr") for x in formed if x[name].get("rr")]
        print(f"    rfvg_{name:<4}: dolan {sum(1 for x in formed if x[name].get('t'))}/{len(formed)}, "
              f"TP {w} / SL {lo}, WR {p:.1f}%" if p is not None else
              f"    rfvg_{name:<4}: dolan {sum(1 for x in formed if x[name].get('t'))}/{len(formed)}, TP {w} / SL {lo}",
              f"· ort. RR {sum(rr) / len(rr):.2f}" if rr else "")

    # A) kapidan gecmis, CISD RR >= 2: eslesik kiyas
    a = [x for x in retested if not x["low_rr"]]
    a_formed = [x for x in a if x["near"] and x["near"].get("e") is not None]
    if a:
        ca = [r_of(x["cv"]) for x in a]
        cb = [r_of(x["near"]) if x["near"] and x["near"].get("e") is not None else (None, 0.0) for x in a]
        ra, rb = sum(v for _, v in ca) / len(a), sum(v for _, v in cb) / len(a)
        print(f"\n  A) kapidan gecmis + CISD RR >= 2, CISD retest edilmis: {len(a)} setup (IFVG olusan {len(a_formed)})")
        print(f"     CISD girisi     : TP {wr(ca)[0]} / SL {wr(ca)[1]}, R/setup {ra:+.3f}")
        print(f"     yalniz rfvg_near: TP {wr(cb)[0]} / SL {wr(cb)[1]}, R/setup {rb:+.3f}  (IFVG yok/dolmadi = 0)")
        d = rb - ra
        if len(a_formed) < MIN_A:
            verdict = f"veri birikiyor ({len(a_formed)}/{MIN_A} IFVG olusan setup)"
        elif d >= BAND:
            verdict = f"IFVG'YI BEKLEMEK DAHA IYI ({d:+.3f}) -> kullaniciya, once replay"
        elif d <= -BAND:
            verdict = f"CISD KALIR ({d:+.3f}) -> madde kapanir"
        else:
            verdict = f"fark yok ({d:+.3f}) -> CISD kalir"
        print(f"     KARAR A: {verdict}")
        cf = [r_of(x["cv"]) for x in a_formed]
        cn = [r_of(x["cv"]) for x in a if x not in a_formed]
        pf, pn = wr(cf)[2], wr(cn)[2]
        print(f"     bilgi: CISD WR -- IFVG olusanlarda {pf:.1f}% ({len(cf)})" if pf is not None else
              f"     bilgi: CISD WR -- IFVG olusanlarda - ({len(cf)})",
              f"· olusmayanlarda {pn:.1f}% ({len(cn)})" if pn is not None else f"· olusmayanlarda - ({len(cn)})")
    # B) CISD RR < 2: ONDO tipi
    b = [x for x in sub if x["low_rr"] and x["near"] and x["near"].get("e") is not None
         and (x["near"].get("rr") or 0) >= MIN_RR and x["near"].get("o") in ("win", "loss")]
    if b or any(x["low_rr"] for x in sub):
        vs = [r_of(x["near"]) for x in b]
        rpt = sum(v for _, v in vs) / len(b) if b else None
        w, lo, p = wr(vs)
        print(f"\n  B) CISD RR < 2, retest sonrasi IFVG RR >= 2, dolup sonuclanan: {len(b)} islem, TP {w} / SL {lo}"
              + (f", R/islem {rpt:+.3f}" if rpt is not None else ""))
        if len(b) < MIN_B:
            verdict = f"veri birikiyor ({len(b)}/{MIN_B})"
        elif rpt >= BAND:
            verdict = f"bu girisler kazandiriyor ({rpt:+.3f}) -> not (motor bu tipi zaten aliyor)"
        elif rpt <= -BAND:
            verdict = f"bu girisler kaybettiriyor ({rpt:+.3f}) -> engelleme sorusu kullaniciya"
        else:
            verdict = f"basabas ({rpt:+.3f})"
        print(f"     KARAR B: {verdict}")
