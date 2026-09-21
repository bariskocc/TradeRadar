"""SL'yi C2 ucundan ne kadar iceri cekebiliriz? -- surekli k egrisi.

Bugunku SL = C2 ucu (purge_extreme). Gölge kollari yalniz uc noktayi olcuyor (k = 1 / 0.75 / 0.5)
ve C1 varyanti setup basina degisen bir k veriyor. Asil soru "EN FAZLA ne kadar" oldugu icin
cozunurluk yetmiyor: egrinin tepesi 0.75-1.0 arasinda olabilir ve orada hic olcum noktasi yok.

Bu betik ayni cevabi **yeni kol eklemeden** verir. `setup_journal.retrace` her setup icin
0.0 = seviye dondugu andaki fiyat (`ref`), 1.0 = SL ekseninde **TP'den once ulasilan en derin
noktayi** (`d_tp`) ve adaylarin eksendeki yerini (`d_of`) tutuyor. Entry `d_of["chosen"]`
oldugundan, kazanan bir islemin hayatta kalmasi icin gereken MINIMUM stop kesri:

    k_min = (d_tp - d_entry) / (1 - d_entry)        [0 = entry, 1 = bugunku SL]

Yani SL'yi k > k_min'e cekersek islem yine TP yapar, k < k_min'e cekersek stop olur. Kaybeden
islem her k'da kaybeder (fiyat SL'ye kadar gitti, yolda her derinligi gecti) ve R normalize
oldugu icin kaybi her k'da -1R'dir. Bu yuzden egri tek bir sayidan analitik cikar:

    E[R](k) = [ SUM(kazanan, k_min <= k) rr/k  -  #(kazanan, k_min > k)  -  #(kaybeden) ] / n

`rr` bugunku stopa gore RR; stop k katina inince RR 1/k katina cikar.

⚠️ Mum ici sira bilinmedigi icin TP mumunun kendi dibi `d_tp`'ye KATILMAZ (`on_fill_bar`
disiplini). `--tp-bar` o dibi de sayar: kotumser sinir. Gercek deger ikisinin arasindadir --
iyimser tarafta cikan bir kazanc, kotumser tarafta da pozitif olmadikca karara girmez.

⚠️ Duz TP/SL: 4H/1D'de kismi kar + BE, 1H'te trail yok. BE tetigi entry'ye baglidir, stop
degisince o da degisir -- bu yuzden kural degismeden once dogrulama replay'i zorunlu.

Karar kurali: IZLEME.md -> "Dar stop — gölge izleme, 4H/1D/1H" icindeki
"Surekli k egrisi" alt basligi.

Kullanim:
    python scripts/stop_width_stat.py
    python scripts/stop_width_stat.py --strategy 4h --tp-bar
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esikler.
MIN_WINNERS = 60         # dilimde cozulmus kazanan (retrace'li) sayisi
MIN_EDGE_R = 0.15        # en iyi k, k=1'i bu kadar R/islem gecmeli
GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)

ZERO = ("tp_before_entry", "no_touch")


def k_min_of(rt: dict, tp_bar: bool) -> float | None:
    """Kazananin hayatta kalmasi icin gereken minimum stop kesri; hesaplanamazsa None."""
    d_entry = (rt.get("d_of") or {}).get("chosen")
    d_tp = rt.get("d_tp")
    if d_entry is None or d_tp is None:
        return None
    d_entry = float(d_entry)
    if d_entry >= 1.0:          # entry SL'nin otesinde: eksen anlamsiz
        return None
    depth = float(d_tp)
    if tp_bar and rt.get("d_tp_bar") is not None:
        depth = max(depth, float(rt["d_tp_bar"]))
    k = (depth - d_entry) / (1.0 - d_entry)
    return max(0.0, k)


def load(db: str, strategy: str, market: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    where = ["retrace IS NOT NULL", "outcome IS NOT NULL"]
    params: list = []
    if strategy:
        where.append("strategy = ?")
        params.append(strategy)
    if market:
        where.append("market_type = ?")
        params.append(market)
    out = []
    for r in con.execute(f"SELECT strategy, symbol, direction, score, rr, outcome, retrace "
                         f"FROM setup_journal WHERE {' AND '.join(where)}", params):
        try:
            rt = json.loads(r["retrace"]) or {}
        except Exception:
            continue
        out.append({"strategy": r["strategy"], "symbol": r["symbol"], "score": r["score"] or 0,
                    "rr": float(r["rr"] or 0), "outcome": r["outcome"], "rt": rt})
    return out


def curve(rows: list[dict], tp_bar: bool) -> tuple[dict[float, float], int, int, int]:
    """(k -> R/islem, kazanan, kaybeden, 0R) -- kazananin k_min'i hesaplanamazsa satir atlanir."""
    wins: list[tuple[float, float]] = []      # (k_min, rr)
    losses = zeros = 0
    for r in rows:
        o = r["outcome"]
        if o == "win":
            k = k_min_of(r["rt"], tp_bar)
            if k is None:
                continue
            wins.append((k, r["rr"]))
        elif o == "loss":
            losses += 1
        elif o in ZERO:
            zeros += 1
    n = len(wins) + losses + zeros
    if not n:
        return {}, 0, 0, 0
    res = {}
    for k in GRID:
        total = sum(rr / k for km, rr in wins if km <= k)
        total -= sum(1 for km, _ in wins if km > k)
        total -= losses
        res[k] = total / n
    return res, len(wins), losses, zeros


def show(rows: list[dict], label: str, tp_bar: bool) -> None:
    res, w, l, z = curve(rows, tp_bar)
    if not res:
        print(f"\n  {label:24} veri yok")
        return
    n = w + l + z
    base = res[1.0]
    best_k = max(res, key=lambda k: res[k])
    print(f"\n  {label:24} n={n}  (kazanan {w} · kaybeden {l} · 0R {z})")
    print("     k      " + "".join(f"{k:>8.2f}" for k in GRID))
    print("     R/islem" + "".join(f"{res[k]:>+8.3f}" for k in GRID))
    alive = {k: sum(1 for km, _ in ((k_min_of(r["rt"], tp_bar), r["rr"]) for r in rows
                                    if r["outcome"] == "win")
                    if km is not None and km <= k) for k in GRID}
    print("     kazanan" + "".join(f"{alive[k]:>8}" for k in GRID) + "   (hayatta kalan)")
    gain = res[best_k] - base
    verdict = "ESIGI ASIYOR" if (gain >= MIN_EDGE_R and w >= MIN_WINNERS) else (
        f"kazanan {w} < {MIN_WINNERS}" if w < MIN_WINNERS else f"kazanc {gain:+.3f} < +{MIN_EDGE_R}")
    print(f"     en iyi k = {best_k:.2f}  ({res[best_k]:+.3f} R/islem, k=1'e gore {gain:+.3f})"
          f"  ->  {verdict}")
    if best_k in (GRID[0], GRID[-1]):
        print(f"     ! optimum izgaranin kenarinda ({best_k:.2f}) -- gercek tepe disarida olabilir")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d | 1h")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    ap.add_argument("--tp-bar", action="store_true",
                    help="TP mumunun kendi dibini de say (kotumser sinir)")
    args = ap.parse_args()

    rows = load(args.db, args.strategy, args.market)
    print("=" * 78)
    print("  SL'yi C2 ucundan ne kadar iceri cekebiliriz? (surekli k egrisi)")
    print(f"  {'TP mumu DAHIL (kotumser)' if args.tp_bar else 'TP mumu haric (iyimser)'}"
          f"   |  k=1.00 = bugunku kural")
    print("=" * 78)
    if not rows:
        print("\n  `retrace` tasiyan satir yok. Kolon 18.09'da eklendi; oncesinde bulunmaz.")
        return

    show(rows, "hepsi", args.tp_bar)
    show([r for r in rows if r["score"] >= 7], "skor >= 7", args.tp_bar)
    show([r for r in rows if r["score"] >= 7 and r["rr"] >= 2], "skor >= 7 ve RR >= 2", args.tp_bar)
    if not args.strategy:
        for s in ("4h", "1d", "1h"):
            sub = [r for r in rows if r["strategy"] == s]
            if sub:
                show(sub, f"{s.upper()} (hepsi)", args.tp_bar)

    print("\n  Karar: en iyi k, k=1'i >= +%.2fR gecmeli VE dilimde >= %d cozulmus kazanan olmali;"
          % (MIN_EDGE_R, MIN_WINNERS))
    print("  ayrica --tp-bar (kotumser) tarafinda da pozitif kalmali. Sonra tek dogrulama replay'i.")
    print("  Kural: IZLEME.md -> \"Dar stop — gölge izleme\" / \"Surekli k egrisi\"")


if __name__ == "__main__":
    main()
