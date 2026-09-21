"""Dolum oncesi kosu -> sonuc: "hareket emir dolmadan once olduysa islem kotulesiyor mu?"

Soru (18.09, ENA 4H #71'den): fiyat TP yolunun buyuk kismini emir dolmadan once
yurur, sonra entry'ye geri gelip doldurur ve doner. `target_taken` kapisi bunu
yakalamaz -- o kapi TAM TP'nin tuketilmesine bakar; %50-%100 arasinda duran hareket
kapidan gecer.

Olcek: 0.0 = entry, 1.0 = TP. `f_max` dolum mumundan ONCEKI mumlarin en iyisi
(dolum mumu mum ici sira bilinmedigi icin haric; `f_bar`'da ayri durur).

Calistirma: repo kokunden
    python scripts/prefill_stat.py [--days N] [--strategy 4h] [--market crypto] [--signals-only]

⚠️ Journal seviyeleri ILK goruldugu haliyle dondurur; motor entry'yi sonradan
tasidiysa f_max o ilk seviyelere gore olculur. Istatistik icin tanim tutarli,
tek bir gercek islemin sayisini birebir vermez.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traderadar.db")

# f_max kovalari. Sinir noktalari kasitli: %50 bugunku BE/kismi kar tetigi, %100 ise
# `target_taken` kapisinin fiilen baktigi yer -- aradaki bosluk olculuyor.
BUCKETS = [(0.00, 0.25, "0-25%"), (0.25, 0.50, "25-50%"),
           (0.50, 0.75, "50-75%"), (0.75, 1.00, "75-100%")]


def bucket_of(f: float) -> str | None:
    for lo, hi, name in BUCKETS:
        if lo <= f < hi:
            return name
    return "75-100%" if f >= 0.75 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--strategy", default=None, help="4h | 1d | 1h")
    ap.add_argument("--market", default=None, help="crypto | fx | metal | index | oil")
    ap.add_argument("--signals-only", action="store_true",
                    help="yalniz sinyale donusmus setuplar (kucuk ornek, secilimli)")
    a = ap.parse_args()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    if "prefill" not in {r[1] for r in con.execute("PRAGMA table_info(setup_journal)")}:
        # Kolon `init_db` -> `_MIGRATIONS` ile aciliyor; sunucu yeniden baslamadan yok.
        print("prefill kolonu henuz yok — sunucu yeniden baslatildiginda olusur (18.09).")
        return 0
    where = ["prefill IS NOT NULL", "first_seen >= datetime('now', ?)"]
    args: list = [f"-{a.days} day"]
    if a.strategy:
        where.append("strategy = ?")
        args.append(a.strategy)
    if a.market:
        where.append("market_type = ?")
        args.append(a.market)
    if a.signals_only:
        where.append("outcome = 'signal'")
    rows = con.execute(
        f"SELECT strategy, symbol, direction, outcome, rr, prefill, entry_touched_at "
        f"FROM setup_journal WHERE {' AND '.join(where)}", args,
    ).fetchall()

    print(f"Setup Journal — dolum oncesi kosu ({a.days} gun"
          + (f", {a.strategy}" if a.strategy else "")
          + (f", {a.market}" if a.market else "") + ")")
    print(f"prefill tasiyan satir: {len(rows)}")
    if not rows:
        print("\nHenuz veri yok. Olcum 18.09.2026'da eklendi; oncesi NULL.")
        return 0

    # Populasyon: emir DOLMUS ve sonucu belli olanlar. Dolmayanlar ayri sayilir --
    # "cok kostu, hic donmedi" de bir cevap, ama farkli bir soruya.
    tab: dict[str, dict[str, int]] = {}
    no_fill: dict[str, int] = {}
    amb = 0
    for r in rows:
        try:
            p = json.loads(r["prefill"])
        except Exception:
            continue
        f = p.get("f_max")
        if f is None:
            continue
        b = bucket_of(float(f))
        if b is None:
            continue
        if not p.get("hit"):
            no_fill[b] = no_fill.get(b, 0) + 1
            continue
        o = r["outcome"]
        if o == "ambiguous":
            amb += 1
            continue
        if o not in ("win", "loss"):
            continue
        tab.setdefault(b, {"win": 0, "loss": 0})[o] += 1

    print(f"\n{'f_max kovasi':>14} | {'dolan':>6} | {'win':>5} | {'loss':>5} | {'win%':>6} | {'dolmadi':>8}")
    print("-" * 66)
    toplam_w = toplam_l = 0
    satirlar = []
    for _, _, name in BUCKETS:
        c = tab.get(name, {"win": 0, "loss": 0})
        n = c["win"] + c["loss"]
        wr = (c["win"] / n * 100) if n else 0.0
        toplam_w += c["win"]
        toplam_l += c["loss"]
        satirlar.append((name, n, wr))
        print(f"{name:>14} | {n:6d} | {c['win']:5d} | {c['loss']:5d} | "
              f"{wr:5.1f}% | {no_fill.get(name, 0):8d}")
    tn = toplam_w + toplam_l
    print("-" * 66)
    print(f"{'TOPLAM':>14} | {tn:6d} | {toplam_w:5d} | {toplam_l:5d} | "
          f"{(toplam_w / tn * 100) if tn else 0:5.1f}% | {sum(no_fill.values()):8d}")
    if amb:
        print(f"\n({amb} setup 'ambiguous' — ayni mumda entry+TP, sira bilinmiyor; disarida)")

    # ── Onceden yazilmis karar kurali (IZLEME.md "Dolum oncesi kosu") ──
    dus = next((s for s in satirlar if s[0] == "0-25%"), None)
    yuk = [s for s in satirlar if s[0] in ("50-75%", "75-100%")]
    yuk_n = sum(s[1] for s in yuk)
    yuk_wr = (sum(s[1] * s[2] for s in yuk) / yuk_n) if yuk_n else 0.0
    print("\n" + "=" * 66)
    print("KARAR KURALI (olcumden once yazildi):")
    print("  Kapi ancak su ikisi birden saglanirsa acilir:")
    print("    1) 0-25% ve >=50% kovalarinin HER BIRINDE >= 100 dolan setup")
    print("    2) >=50% kovasinin win orani, 0-25% kovasindan >= 10 puan dusuk")
    if dus is None or dus[1] < 100 or yuk_n < 100:
        print(f"\n  -> ORNEKLEM YETERSIZ (0-25%: {dus[1] if dus else 0}, >=50%: {yuk_n}; esik 100).")
        print("     Veri birikiyor; tahmin yurutme.")
    else:
        fark = dus[2] - yuk_wr
        print(f"\n  0-25% win {dus[2]:.1f}%  vs  >=50% win {yuk_wr:.1f}%   fark {fark:+.1f} puan")
        print("  -> " + ("KAPI ACILSIN: hipotez dogrulandi." if fark >= 10
                         else "KAPI ACILMASIN: fark esigin altinda, mevcut davranis kalsin."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
