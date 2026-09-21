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
  dfvg_near/mid/far      kirilim sonrasi birakilan FVG (yeni model)

Cevaplanan sorular:
  1. IFVG'ye fiyat geliyor mu, bosuna mi bekliyoruz? -> ifvg_* dolum orani
  2. Bolgenin alt noktasindan da dolar miydi?        -> ifvg_far vs ifvg_near
  3. Kirilim FVG'si daha mi iyi?                     -> dfvg_* vs chosen

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
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler ---------------------------------
MIN_SETUPS = 60        # varyant basina karar verilebilir setup
MIN_EDGE_R = 0.15      # chosen'i R/setup olarak bu kadar gecmeli

VARIANTS = ("chosen", "cisd", "mss",
            "ifvg_near", "ifvg_mid", "ifvg_far",
            "bpr_near", "bpr_mid", "bpr_far",
            "dfvg_near", "dfvg_mid", "dfvg_far")
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
        rows.append(d)

    if not rows:
        print("Entry varyanti tasiyan satir yok. Varyantlar 18.09'dan itibaren SEVIYELENEN\n"
              "setuplara yazilir; eski satirlarda bulunmaz.")
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

    head("2) Kapi ustunde: skor >= 7 ve RR >= 2 olan setuplar")
    g = [r for r in rows if (r["score"] or 0) >= 7 and (r["rr"] or 0) >= 2]
    if len(g) >= 10:
        table(g, not args.keep_imm, f"score>=7 & RR>=2 ({len(g)} setup)")
    else:
        print(f"  yalniz {len(g)} setup -- anlamli degil, veri birikiyor.")

    head("KARAR (IZLEME.md kurali; veri gorulmeden yazildi)")
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
