"""Geri cekilme derinligi: fiyat nereye kadar geri geliyor, limit emri hangi seviyeden dolar?

Veri: setup_journal.retrace (18.09'dan itibaren seviyelenen setuplar).
Olcek: 0.0 = sinyal dogdugu andaki fiyat (ref), 1.0 = SL (purge ucu). Her aday entry bu
aralikta bir noktaya duser, otesi zaten stop.

NEDEN KARAR GIRDISI KAZANANLAR:
    E[R](d) = SUM(dolan kazananlar) RR(d)  -  (dolan kaybedenler)
Kaybeden bir setupta fiyat SL'ye (d = 1.0) kadar gider, yani YOLDA HER DERINLIGI gecer ->
her derinlikte dolar, her biri -1R. Kayip terimi d'den BAGIMSIZ sabittir; optimumun yerini
yalniz kazananlarin d_tp dagilimi belirler. Kaybedenler egrinin SEVIYESINI belirler (sistem
karda mi) ve varsayimin denetimidir (asagida "kontrol" satiri).

RR(d) analitiktir, olcmeye gerek yok:
    entry(d) = ref -+ d*span   ->   RR(d) = |tp - entry(d)| / (span * (1 - d))

SINIRLAR: (a) TP'nin gerceklestigi mumun dibi d_tp'ye KATILMAZ (mum ici sira bilinmiyor);
`--tp-bar` ile duyarlilik alinir. (b) Sabit-kayip ozelligi SL/TP sabit ve R normalize oldugu
icin gecerlidir; canli motorda kismi kar + BE tetigi entry'ye gore olculdugunden BOZULUR --
bu yuzden kural degismeden once tek dogrulama replay'i zorunlu.

Kullanim:
    python scripts/retrace_stat.py
    python scripts/retrace_stat.py --strategy 4h --market crypto --tp-bar
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler ---------------------------------
MIN_WINNERS = 60       # cozulmus kazanan setup; altinda karar yok
MIN_EDGE_R = 0.15      # optimum, bugunku entry derinligini bu kadar gecmeli
GRID = [round(0.05 * i, 2) for i in range(0, 19)]      # 0.00 .. 0.90
NAMED = ("chosen", "cisd", "mss", "ifvg_near", "ifvg_mid", "ifvg_far",
         "bpr_near", "bpr_mid", "bpr_far",
         "dfvg_near", "dfvg_mid", "dfvg_far")


def head(t: str) -> None:
    print(f"\n{'=' * 84}\n  {t}\n{'=' * 84}")


def rr_at(row: dict, d: float) -> float | None:
    """RR(d) = |tp - entry(d)| / (span * (1-d)); entry(d) ref'ten d kadar geriye."""
    r = row["_r"]
    span, ref, tp = r.get("span"), r.get("ref"), row.get("tp")
    if not span or ref is None or tp is None or d >= 1:
        return None
    entry = ref - d * span if row["direction"] == "LONG" else ref + d * span
    risk = span * (1 - d)
    if risk <= 0:
        return None
    return abs(tp - entry) / risk


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--strategy", default="")
    ap.add_argument("--market", default="")
    ap.add_argument("--tp-bar", action="store_true",
                    help="TP mumunun dibini de derinlige kat (duyarlilik; varsayilan haric)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB bulunamadi: {args.db}")
        return
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("SELECT retrace FROM setup_journal LIMIT 1")
    except sqlite3.OperationalError:
        print("setup_journal.retrace kolonu yok. Sunucu acilista olusturur (init_db).")
        return
    sql = "SELECT * FROM setup_journal WHERE retrace IS NOT NULL"
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
            d["_r"] = json.loads(d["retrace"]) or {}
        except Exception:
            continue
        if d["_r"].get("amb"):
            continue                       # ayni mumda TP+SL: sira bilinmiyor
        rows.append(d)
    if not rows:
        print("retrace verisi yok. 18.09'dan itibaren seviyelenen setuplara yazilir.")
        return

    def depth(row: dict) -> float:
        r = row["_r"]
        d = float(r.get("d_tp") or 0.0)
        if args.tp_bar and r.get("d_tp_bar") is not None:
            d = max(d, float(r["d_tp_bar"]))
        return d

    win = [r for r in rows if r["_r"].get("tp_first") is True]
    loss = [r for r in rows if r["_r"].get("tp_first") is False]
    acik = [r for r in rows if r["_r"].get("tp_first") is None]
    head(f"GERI CEKILME DERINLIGI | {len(rows)} setup | kazanan {len(win)} · kaybeden {len(loss)}"
         f" · sonuclanmamis {len(acik)}")
    print("  olcek: 0.00 = sinyal aninda ki fiyat, 1.00 = SL")

    if not win:
        print("\n  Henuz cozulmus kazanan setup yok; veri birikiyor.")
        return

    wd = [depth(r) for r in win]
    ld = [float(r["_r"].get("d_max") or 0) for r in loss]
    head("1) KAZANANLAR nereye kadar geri cekildi? (karar girdisi)")
    print(f"  n={len(wd)}  medyan {pct(wd,0.5):.2f}  ort {st.mean(wd):.2f}")
    print(f"  %10 {pct(wd,0.10):.2f} | %25 {pct(wd,0.25):.2f} | %50 {pct(wd,0.50):.2f} | "
          f"%75 {pct(wd,0.75):.2f} | %90 {pct(wd,0.90):.2f}")
    if ld:
        print(f"\n  kontrol -- KAYBEDENLERIN d_max: medyan {pct(ld,0.5):.2f}, "
              f"{100*sum(1 for x in ld if x >= 0.999)/len(ld):.0f}% SL'ye ulasti "
              f"(sabit-kayip varsayimi bu oranda gecerli)")

    head("2) Limit emri hangi derinlikten dolar? (kazananlar icinde)")
    print(f"{'derinlik':>9}{'dolum%':>9}{'ort.RR':>9}{'E[R] kazanan':>14}{'E[R] tum':>11}")
    best = None
    ntot = len(win) + len(loss)
    for d in GRID:
        dolan = [r for r, dd in zip(win, wd) if dd >= d]
        rrs = [rr_at(r, d) for r in dolan]
        rrs = [x for x in rrs if x is not None]
        if not rrs and d > 0:
            print(f"{d:>9.2f}{0.0:>8.1f}%{'-':>9}{'-':>14}{'-':>11}")
            continue
        toplam = sum(rrs)
        ev_win = toplam / len(win) if win else 0
        ev_all = (toplam - len(loss)) / ntot if ntot else 0
        print(f"{d:>9.2f}{100*len(dolan)/len(win):>8.1f}%{(sum(rrs)/len(rrs) if rrs else 0):>9.2f}"
              f"{ev_win:>+14.3f}{ev_all:>+11.3f}")
        if best is None or ev_win > best[1]:
            best = (d, ev_win, len(dolan))

    head("3) Bugunku modeller bu eksenin neresinde?")
    for name in NAMED:
        ds = [r["_r"].get("d_of", {}).get(name) for r in rows]
        ds = [float(x) for x in ds if x is not None]
        if not ds:
            continue
        wds = [r["_r"]["d_of"][name] for r in win if r["_r"].get("d_of", {}).get(name) is not None]
        fill = 100 * sum(1 for r, dd in zip(win, wd)
                         if r["_r"].get("d_of", {}).get(name) is not None
                         and dd >= r["_r"]["d_of"][name]) / len(win) if wds else 0
        print(f"  {name:<12} ort. derinlik {st.mean(ds):>5.2f}  (medyan {pct(ds,0.5):>4.2f})  "
              f"kazananlarda dolum %{fill:>5.1f}")

    head("KARAR (IZLEME.md 'Geri cekilme derinligi' kurali; veri gorulmeden yazildi)")
    if len(win) < MIN_WINNERS:
        print(f"  -> KARAR YOK: cozulmus kazanan {len(win)} < {MIN_WINNERS}. Veri birikiyor.")
    elif best:
        d_ch = [r["_r"]["d_of"]["chosen"] for r in rows if r["_r"].get("d_of", {}).get("chosen") is not None]
        ref_d = st.mean(d_ch) if d_ch else None
        ev_ref = None
        if ref_d is not None:
            dolan = [r for r, dd in zip(win, wd) if dd >= ref_d]
            rrs = [x for x in (rr_at(r, ref_d) for r in dolan) if x is not None]
            ev_ref = sum(rrs) / len(win)
        print(f"  optimum derinlik {best[0]:.2f}  E[R] {best[1]:+.3f}  (o derinlikte dolan kazanan {best[2]})")
        if ev_ref is not None:
            print(f"  bugunku entry ort. derinlik {ref_d:.2f}  E[R] {ev_ref:+.3f}")
            edge = best[1] - ev_ref
            print(f"  fark {edge:+.3f}R  (esik >= +{MIN_EDGE_R})")
            if edge >= MIN_EDGE_R:
                print("  -> ADAY: kural HEMEN degismez; once tek dogrulama replay'i (kismi kar/BE/")
                print("     trail + portfoy kapilariyla), cunku sabit-kayip ozelligi canlida bozulur.")
            else:
                print("  -> Mevcut entry derinligi yeterince iyi; degisiklik gerekmiyor.")
    print("\nNot: 'E[R] kazanan' optimumun YERINI, 'E[R] tum' seviyeyi verir. Ikisinin tepe")
    print("noktasi aynidir (kayip terimi d'den bagimsiz sabittir).")


if __name__ == "__main__":
    main()
