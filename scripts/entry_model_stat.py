"""Entry modeli karsilastirmasi: ayni setupta hangi giris seviyesi daha iyi?

Veri: setup_journal.entries (18.09'dan itibaren seviyelenen setuplar). AYNI setup icin butun
aday entry'ler ayri ayri izlenir -- SL (purge ucu) ve TP (karsi C1 ucu) SABIT, yalniz entry
degisir. Boylece secilim etkisi yok: bugune kadar IFVG ile CISD'yi kiyaslamak FARKLI setuplari
kiyaslamak demekti (IFVG ancak CISD adayi kotuyken seciliyor, cunku bolgenin EN KOTU RR'li
noktasiyla yarisiyor).

Varyantlar:
  chosen                 motorun fiilen sectigi (temel cizgi)
  cisd / mss             iki yapisal aday
  ifvg_near/mid/far      IFVG bolgesinin ust-orta-alt noktasi (LONG'da near = ust = BUGUNKU)
  bpr_near/mid/far       BPR = iki FVG'nin KESISIMI (BPR'li setupta bugunku entry)
  bfvg_near/mid/far      BPR bacagi = kesisimi olusturan AYNI YONLU FVG'nin KENDISI
  dfvg_near/mid/far      purge -> onay arasinda birakilan FVG (seviye dondugu an bakilir)
  cfvg_near/mid/far      CISD kirilim FVG'si (22.09): onayi YAPAN hamlenin arkada biraktigi
                         bosluk. IFVG'ye de MSS'e de bakmaz. dfvg'den farki pencere (onay mumu
                         merkezli) ve zamanlama (seviye GEC eklenir) -- dfvg bu FVG'yi yapisal
                         olarak goremiyordu, 3 mumluk desen onay mumundan SONRA tamamlanir.

Cevaplanan sorular:
  1. IFVG'ye fiyat geliyor mu, bosuna mi bekliyoruz? -> ifvg_* dolum orani
  2. Bolgenin alt noktasindan da dolar miydi?        -> ifvg_far vs ifvg_near
  3. Kirilim FVG'si daha mi iyi?                     -> dfvg_* vs chosen
  4. BPR'de kesisim mi, bacagin kendisi mi?          -> bfvg_near vs bpr_near (yalniz BPR'li)
  5. CISD onayindan sonra FVG'ye retest gelir mi?    -> cfvg_* vs chosen (ESLENMIS)
  6. Seviyeler donarken C2 KAPALI miydi?             -> her varyant, iki dilimde ayri

R: win = +rr, loss = -1, tp_before_entry / no_touch = 0 (emir dolmadi, islem yok).
`ambiguous` ve `open` disarida, ayrica raporlanir.

ONEMLI: `imm` isaretli varyantlar (seviye dondugunda fiyat ZATEN adayin gerisindeydi -> limit
emri aninda dolardi) ayri tutulur. Onlari saymak sig varyantlari haksiz yere avantajli gosterir;
asil soru "fiyat geri geliyor mu".

Karar kurali: IZLEME.md "Entry modeli karsilastirmasi".

Kullanim:
    python scripts/entry_model_stat.py
    python scripts/entry_model_stat.py --days 30 --strategy 4h --market crypto
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler ---------------------------------
MIN_SETUPS = 60        # varyant basina karar verilebilir setup
MIN_EDGE_R = 0.15      # chosen'i R/setup olarak bu kadar gecmeli
MIN_BPR_SETUPS = 40    # BPR dilimi icin ayri taban (IZLEME "BPR entry modeli" ile ayni)
MIN_SLICE_SETUPS = 30  # bir dilimi (C2 kapali/acik) okumak icin en az setup

VARIANTS = ("chosen", "cisd", "mss",
            "ifvg_near", "ifvg_mid", "ifvg_far",
            "bpr_near", "bpr_mid", "bpr_far",
            "bfvg_near", "bfvg_mid", "bfvg_far",
            "dfvg_near", "dfvg_mid", "dfvg_far",
            "cfvg_near", "cfvg_mid", "cfvg_far")
ZERO = ("tp_before_entry", "no_touch")
SKIP = ("ambiguous", "open", "pending", "filled")


def head(t: str) -> None:
    print(f"\n{'=' * 86}\n  {t}\n{'=' * 86}")


def tally(rows: list[dict], name: str, drop_imm: bool) -> dict:
    n = w = l = zero = skipped = imm = invalid = 0
    total = 0.0
    rrs = []
    for r in rows:
        v = (r["_entries"] or {}).get(name)
        if not v:
            continue
        if v.get("o") == "invalid":
            invalid += 1
            continue
        if v.get("imm"):
            imm += 1
            if drop_imm:
                continue
        o = v.get("o")
        if o in SKIP:
            skipped += 1
            continue
        n += 1
        if o == "win":
            w += 1
            total += float(v.get("rr") or 0)
            rrs.append(float(v.get("rr") or 0))
        elif o == "loss":
            l += 1
            total -= 1
            rrs.append(float(v.get("rr") or 0))
        elif o in ZERO:
            zero += 1
            rrs.append(float(v.get("rr") or 0))
    return {"n": n, "w": w, "l": l, "zero": zero, "skip": skipped, "imm": imm,
            "invalid": invalid, "R": total, "ev": (total / n) if n else 0.0,
            "rr": (sum(rrs) / len(rrs)) if rrs else 0.0}


def c2_at_levels(row: dict):
    """Seviyeler DONDUGU AN C2 kapali miydi? (True/False/None)

    `setup_journal.c2_closed` KOLONU kullanilamaz: her degerlendirmede guncellenir, yani C2
    sonradan kapaninca True'ya doner -- olculen 1521 satirin 476'sinda (%31) kolon ile donmus
    hal farkli. Donmus kopya skor kiriliminda duruyor: `parts_at_levels["c2_closed"]` o kalemin
    puanidir (kapaliysa +1). Izleme ve entry seviyeleri o ana ait oldugu icin dogru dilim budur.
    """
    parts = row.get("_parts")
    if not parts or "c2_closed" not in parts:
        return None
    return bool(parts["c2_closed"])


def _r_of(v: dict | None, drop_imm: bool):
    """Tek varyantin R'si; karar verilemiyorsa None (tablodaki `tally` ile ayni kurallar)."""
    if not v or v.get("o") == "invalid":
        return None
    if v.get("imm") and drop_imm:
        return None
    o = v.get("o")
    if o in SKIP:
        return None
    if o == "win":
        return float(v.get("rr") or 0)
    if o == "loss":
        return -1.0
    return 0.0 if o in ZERO else None


def pair_tally(rows: list[dict], a: str, b: str, drop_imm: bool) -> dict:
    """AYNI setuplarda iki varyant. Havuzlar farkli oldugunda tablo yaniltici olabiliyor:
    varyantlarin karar verilebilir setup kumesi ayni degil (bir aday hic olusmamis olabilir)."""
    ra, rb = [], []
    for r in rows:
        ents = r["_entries"] or {}
        x, y = _r_of(ents.get(a), drop_imm), _r_of(ents.get(b), drop_imm)
        if x is None or y is None:
            continue
        ra.append(x)
        rb.append(y)
    n = len(ra)
    def side(vals):
        return {"n": n, "R": sum(vals), "ev": (sum(vals) / n) if n else 0.0,
                "w": sum(1 for v in vals if v > 0), "l": sum(1 for v in vals if v < 0),
                "z": sum(1 for v in vals if v == 0)}
    return {"n": n, a: side(ra), b: side(rb)}


def table(rows: list[dict], drop_imm: bool, title: str) -> dict:
    head(title)
    print(f"{'varyant':<12}{'karar':>7}{'dolan':>7}{'dolum%':>8}{'win':>6}{'loss':>6}"
          f"{'ort.RR':>8}{'toplam R':>10}{'R/setup':>9}{'imm':>6}{'gecersiz':>9}")
    out = {}
    for name in VARIANTS:
        t = tally(rows, name, drop_imm)
        out[name] = t
        if not t["n"] and not t["invalid"]:
            continue
        dolan = t["w"] + t["l"]
        pct = 100 * dolan / t["n"] if t["n"] else 0
        print(f"{name:<12}{t['n']:>7}{dolan:>7}{pct:>7.1f}%{t['w']:>6}{t['l']:>6}"
              f"{t['rr']:>8.2f}{t['R']:>10.1f}{t['ev']:>+9.3f}{t['imm']:>6}{t['invalid']:>9}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--strategy", default="")
    ap.add_argument("--market", default="")
    ap.add_argument("--c2", choices=("closed", "open"), default="",
                    help="yalniz seviyeler donarken C2'si kapali / acik olan setuplar")
    ap.add_argument("--keep-imm", action="store_true",
                    help="fiyatin gerisinde kalan adaylari da say (varsayilan: disla)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB bulunamadi: {args.db}")
        return
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("SELECT entries FROM setup_journal LIMIT 1")
    except sqlite3.OperationalError:
        print("setup_journal.entries kolonu yok. Sunucu acilista olusturur (init_db).")
        return
    sql = "SELECT * FROM setup_journal WHERE entries IS NOT NULL"
    params: list = []
    if args.days:
        sql += " AND levels_at >= ?"
        params.append((datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S"))
    if args.strategy:
        sql += " AND strategy = ?"
        params.append(args.strategy)
    if args.market:
        sql += " AND market_type = ?"
        params.append(args.market)
    rows = []
    for r in con.execute(sql, params):
        d = dict(r)
        try:
            d["_entries"] = json.loads(d["entries"]) or {}
        except Exception:
            continue
        try:
            d["_parts"] = json.loads(d["parts_at_levels"] or "null") or {}
        except Exception:
            d["_parts"] = {}
        rows.append(d)

    if not rows:
        print("Entry varyanti tasiyan satir yok. Varyantlar 18.09'dan itibaren SEVIYELENEN\n"
              "setuplara yazilir; eski satirlarda bulunmaz.")
        return

    if args.c2:
        want = args.c2 == "closed"
        rows = [r for r in rows if c2_at_levels(r) is want]
        if not rows:
            print(f"C2 {args.c2} dilimine giren setup yok.")
            return

    res = table(rows, not args.keep_imm,
                f"ENTRY MODELLERI | {len(rows)} setup"
                + ("  (fiyatin gerisinde kalanlar DISLANDI)" if not args.keep_imm else "  (hepsi)"))

    head("1) IFVG'ye fiyat geliyor mu? (ust / orta / alt nokta)")
    base = res.get("ifvg_near", {}).get("n", 0)
    if not base:
        print("  IFVG varyanti olan setup yok.")
    else:
        for name, lbl in (("ifvg_near", "ust kenar (BUGUNKU)"), ("ifvg_mid", "orta"),
                          ("ifvg_far", "alt kenar")):
            t = res[name]
            dolan = t["w"] + t["l"]
            print(f"  {lbl:<22} karar {t['n']:>4}  dolum %{100*dolan/t['n'] if t['n'] else 0:>5.1f}  "
                  f"ort.RR {t['rr']:>4.2f}  R/setup {t['ev']:>+6.3f}")
        n_, f_ = res["ifvg_near"], res["ifvg_far"]
        d_near = 100 * (n_["w"] + n_["l"]) / n_["n"] if n_["n"] else 0
        d_far = 100 * (f_["w"] + f_["l"]) / f_["n"] if f_["n"] else 0
        print(f"\n  Alt kenar ust kenara gore: dolum {d_far - d_near:+.1f} puan, "
              f"RR {f_['rr'] - n_['rr']:+.2f}, R/setup {f_['ev'] - n_['ev']:+.3f}")
        print("  (Alt kenardan da doluyorsa IFVG entry modeli derin kenara cekilebilir.)")

    head("2) BPR'li setuplar: kesisim mi, bacagin kendisi mi?")
    bpr_rows = [r for r in rows if (r["_entries"] or {}).get("bpr_near")]
    if not bpr_rows:
        print("  BPR varyanti tasiyan setup yok (BPR 21.09'da, bacak olcumu 22.09'da eklendi).")
    else:
        b = table(bpr_rows, not args.keep_imm, f"BPR'li setuplar ({len(bpr_rows)})")
        pair = []
        for name, lbl in (("bpr_near", "kesisim yakin kenar (BUGUNKU)"),
                          ("bfvg_near", "bacagin yakin kenari"),
                          ("ifvg_near", "IFVG yakin kenar"),
                          ("dfvg_near", "kirilim FVG'si")):
            t = b.get(name)
            if not t or not t["n"]:
                continue
            dolan = t["w"] + t["l"]
            pair.append((name, t))
            print(f"  {lbl:<30} karar {t['n']:>4}  dolum %{100*dolan/t['n']:>5.1f}  "
                  f"ort.RR {t['rr']:>4.2f}  R/setup {t['ev']:>+6.3f}")
        k, g = b.get("bpr_near"), b.get("bfvg_near")
        if k and g and k["n"] and g["n"]:
            d_k = 100 * (k["w"] + k["l"]) / k["n"]
            d_g = 100 * (g["w"] + g["l"]) / g["n"]
            print(f"\n  Bacak kesisime gore: dolum {d_g - d_k:+.1f} puan, "
                  f"RR {g['rr'] - k['rr']:+.2f}, R/setup {g['ev'] - k['ev']:+.3f}")
            if min(k["n"], g["n"]) < MIN_BPR_SETUPS:
                print(f"  -> KARAR YOK: karar verilebilir BPR setup'i "
                      f"{min(k['n'], g['n'])} < {MIN_BPR_SETUPS}, veri birikiyor.")
            elif g["ev"] - k["ev"] >= MIN_EDGE_R:
                print(f"  -> Bacak kesisimi >= {MIN_EDGE_R}R geciyor: BPR entry'si bacagin yakin")
                print("     kenarina tasinabilir -- once dogrulama replay'i.")
            else:
                print(f"  -> Bacak kesisimi {MIN_EDGE_R}R gecemedi. BPR entry'si kesisimde kalir.")
        else:
            print("  -> KARAR YOK: bacak (bfvg_*) olcumu 22.09'da eklendi, geriye donuk veri yok;")
            print("     seviyeleri BUNDAN SONRA donan BPR'li setuplarda birikir.")
        print("  (Bacak kesisimi KAPSAR: yakin kenari daima daha sig -> dolum yuksek, RR dusuk.)")

    head("3) CISD kirilim FVG'si: onaydan sonra retest geliyor mu?")
    cf = [r for r in rows if (r["_entries"] or {}).get("cfvg_near")]
    if not cf:
        print("  cfvg olcumu 22.09'da eklendi; seviyeleri BUNDAN SONRA donan setuplarda birikir.")
    else:
        why = {}
        for r in cf:
            v = r["_entries"]["cfvg_near"]
            if v.get("o") == "invalid":
                why[v.get("why") or "?"] = why.get(v.get("why") or "?", 0) + 1
        uygulanabilir = len(cf) - sum(why.values())
        print(f"  Onay penceresi kapanan setup: {len(cf)} -- FVG cikan {uygulanabilir}, "
              f"uygulanamayan {sum(why.values())} {why or ''}")
        for name, lbl in (("cfvg_far", "derin kenar (ilk aday)"),
                          ("cfvg_mid", "orta nokta"),
                          ("cfvg_near", "sig kenar")):
            t = res.get(name)
            if not t or not t["n"]:
                continue
            dolan = t["w"] + t["l"]
            print(f"  {lbl:<26} karar {t['n']:>4}  dolum %{100*dolan/t['n']:>5.1f}  "
                  f"ort.RR {t['rr']:>4.2f}  R/setup {t['ev']:>+6.3f}")
        for name in ("cfvg_far", "cfvg_mid", "cfvg_near"):
            pr = pair_tally(rows, "chosen", name, not args.keep_imm)
            if not pr["n"]:
                continue
            c_, v_ = pr["chosen"], pr[name]
            print()
            print(f"  ESLENMIS (ayni {pr['n']} setup) chosen vs {name}:")
            print(f"    chosen {c_['ev']:+.3f} R/setup (W{c_['w']} L{c_['l']} dolmayan {c_['z']})")
            print(f"    {name:<10} {v_['ev']:+.3f} R/setup (W{v_['w']} L{v_['l']} dolmayan {v_['z']})")
            if pr["n"] < MIN_SETUPS:
                print(f"    -> KARAR YOK: {pr['n']} < {MIN_SETUPS} setup, veri birikiyor.")
            elif v_["ev"] - c_["ev"] >= MIN_EDGE_R:
                print(f"    -> {name} chosen'i {MIN_EDGE_R}R gecti: entry modeli degisebilir")
                print("       -- once tek dogrulama replay'i (kismi kar/BE + portfoy kapilari).")
            else:
                print(f"    -> {MIN_EDGE_R}R gecemedi; mevcut CISD entry'si kalir.")

    head("4) Seviyeler donarken C2 kapali miydi?")
    print("  Journal setup'i GORDUGU an seviyeliyor -- 4H/1D'de C2 kapanmadan sinyal dogmaz ama")
    print("  radar 'c2_open' asamasinda da kayit acilir, 1H'te kural zaten C2 kapanisini beklemez.")
    print("  C2 acikken donan seviyeler KOSAN bir purge ucuna dayanir: SL/TP ve CISD/MSS")
    print("  seviyeleri C2 kapanana kadar daha degisebilir. Dilim bu bozulmayi olcer.")
    if args.c2:
        print(f"  (--c2 {args.c2} verildi: rapor zaten yalniz bu dilimi gosteriyor.)")
    else:
        slices = {"kapali": [r for r in rows if c2_at_levels(r) is True],
                  "acik": [r for r in rows if c2_at_levels(r) is False]}
        bilinmeyen = len(rows) - sum(len(v) for v in slices.values())
        dag = {k: dict(sorted(Counter(r["strategy"] for r in v).items())) for k, v in slices.items()}
        print()
        print(f"  C2 kapali {len(slices['kapali'])} {dag['kapali']} | "
              f"C2 acik {len(slices['acik'])} {dag['acik']}"
              + (f" | bilinmeyen {bilinmeyen}" if bilinmeyen else ""))
        out = {}
        for lbl, sub in slices.items():
            if len(sub) < MIN_SLICE_SETUPS:
                print(f"  C2 {lbl}: {len(sub)} setup -- {MIN_SLICE_SETUPS} altinda, "
                      "tablo basilmadi.")
                continue
            out[lbl] = table(sub, not args.keep_imm, f"C2 {lbl.upper()} ({len(sub)} setup)")
        a, b = out.get("kapali"), out.get("acik")
        if a and b:
            print()
            print("  Varyant bazinda R/setup farki (kapali - acik):")
            for name in VARIANTS:
                ta, tb = a[name], b[name]
                if min(ta["n"], tb["n"]) < MIN_SLICE_SETUPS:
                    continue
                d_a = 100 * (ta["w"] + ta["l"]) / ta["n"]
                d_b = 100 * (tb["w"] + tb["l"]) / tb["n"]
                print(f"    {name:<12} kapali {ta['ev']:+.3f} (n={ta['n']}, dolum %{d_a:.0f})  "
                      f"acik {tb['ev']:+.3f} (n={tb['n']}, dolum %{d_b:.0f})  "
                      f"fark {ta['ev'] - tb['ev']:+.3f}")
            print("  (Aradaki fark modelin iyiligi degil OLCUMUN sagligidir: C2 acik dilimde")
            print("   seviyeler sonradan kaymis olabilir. Buyuk ve tutarli bir fark, entry modeli")
            print("   kiyasinin C2 kapali dilimde yapilmasi gerektigini soyler.)")

    head("5) Kapi ustunde: skor >= 7 ve RR >= 2 olan setuplar")
    g = [r for r in rows if (r["score"] or 0) >= 7 and (r["rr"] or 0) >= 2]
    if len(g) >= 10:
        table(g, not args.keep_imm, f"score>=7 & RR>=2 ({len(g)} setup)")
    else:
        print(f"  yalniz {len(g)} setup -- anlamli degil, veri birikiyor.")

    head("KARAR (IZLEME.md kurali; veri gorulmeden yazildi)")
    if not args.c2:
        kap = [r for r in rows if c2_at_levels(r) is True]
        ack = [r for r in rows if c2_at_levels(r) is False]
        if len(kap) >= MIN_SLICE_SETUPS and len(ack) >= MIN_SLICE_SETUPS:
            a = tally(kap, "chosen", not args.keep_imm)
            b = tally(ack, "chosen", not args.keep_imm)
            if a["n"] and b["n"] and abs(a["ev"] - b["ev"]) >= MIN_EDGE_R:
                print(f"  ! Havuz iki dilimi KARISTIRIYOR: chosen C2 kapali {a['ev']:+.3f} "
                      f"(n={a['n']}) vs C2 acik {b['ev']:+.3f} (n={b['n']}).")
                print("    C2 acikken donan seviyeler kosan purge ucuna dayanir; temiz okuma icin")
                print("    ayni raporu `--c2 closed` ile de calistir (bkz. bolum 4).")
    ch = res["chosen"]
    print(f"  temel cizgi (chosen): n={ch['n']}  R/setup {ch['ev']:+.3f}")
    if ch["n"] < MIN_SETUPS:
        print(f"  -> KARAR YOK: chosen'da karar verilebilir setup {ch['n']} < {MIN_SETUPS}.")
    else:
        kazanan = []
        for name in VARIANTS:
            if name == "chosen":
                continue
            t = res[name]
            if t["n"] >= MIN_SETUPS and t["ev"] - ch["ev"] >= MIN_EDGE_R:
                kazanan.append((t["ev"] - ch["ev"], name, t))
        kazanan.sort(reverse=True)
        if not kazanan:
            print(f"  -> Hicbir varyant chosen'i >= {MIN_EDGE_R}R gecmedi. Mevcut entry modeli kalir.")
        else:
            for edge, name, t in kazanan:
                print(f"  -> {name}: {edge:+.3f}R (n={t['n']}) -- kural HEMEN degismez, once tek")
                print("     dogrulama replay'i (kismi kar/BE/trail + portfoy kapilariyla).")
    print("\nSinir: SL/TP sabit tutuldu (entry ve stop ayni anda degistirilmedi); stop tarafi")
    print("ayri izlemede (dar stop + C1 ucu). Golge R gercek R degildir -- varyantlar arasi kiyas icin.")


if __name__ == "__main__":
    main()
