"""Kismi kar (%50'de yari) vs sadece BE: hangisi daha cok kazandirirdi? (canli kayitlardan)

Neden hesaplanabiliyor: iki varyant AYNI tetikte devreye giriyor (TP yolunun %50'si), yani fiyat
yolu ayni; tek fark pozisyon boyutu. Kapanista `rr_value` agirlikli yazildigi icin
(`partial_size x partial_rr + (1 - partial_size) x kalan R`) kalan yarinin R'si geri cozulebilir:

    kalan R = (rr_value - f x partial_rr) / (1 - f)

BE-only varyanti = tam boyutla ayni yol => R = kalan R. Simetrik sonuc:
  - kalan yari BE'de kapandiysa kismi kar  +0.25 x RR  kazandirir
  - kalan yari TP'ye gittiyse kismi kar    -0.25 x RR  kaybettirir
Yani kural tek satir: **%50'ye cikip girise donen isler, TP'ye gidenlerden (RR agirlikli) fazlaysa
kismi kar kazandirir.** Karar kurali ve esik: IZLEME.md -> "Kismi kar vs BE-only".

Kullanim:
    python scripts/partial_vs_be.py                  # tum kayitlar
    python scripts/partial_vs_be.py --days 60        # son 60 gun (kapanisa gore)
    python scripts/partial_vs_be.py --strategy 4h
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# IZLEME.md'deki onceden yazilmis esik: bu kadar "kismi kar almis" islem birikmeden karar verilmez.
MIN_TRADES = 25


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0, help="son N gun (kapanis tarihine gore; 0 = hepsi)")
    ap.add_argument("--strategy", default="", help="4h | 1d (1h'te kismi kar yok)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = ("SELECT id, symbol, timeframe, market_type, direction, result, exit_reason, planned_rr, "
           "rr_value, partial_size, partial_price, partial_rr, closed_at FROM signals "
           "WHERE status = 'expired' AND rr_value IS NOT NULL AND partial_size IS NOT NULL "
           "AND partial_rr IS NOT NULL")
    params: list = []
    if args.days:
        sql += " AND closed_at >= ?"
        params.append((datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S"))
    if args.strategy:
        sql += " AND timeframe = ?"
        params.append(args.strategy)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY closed_at", params)]
    con.close()

    head("Kismi kar (%50 yari + BE) vs sadece BE")
    if not rows:
        print("  Kismi kar almis kapanmis islem yok. Kismi kar 11.09.2026'da canliya alindi (4H/1D).")
        return

    print(f"  {'id':>4} {'sembol':11s} {'tf':3s} {'cikis':10s} {'gercek R':>9} {'BE-only R':>10} {'fark':>7}")
    tot_act = tot_be = 0.0
    kind = {"be": 0, "tp": 0, "diger": 0}
    for r in rows:
        f, pr, act = float(r["partial_size"]), float(r["partial_rr"]), float(r["rr_value"])
        if f >= 1.0:
            continue
        be_only = (act - f * pr) / (1.0 - f)        # kalan yarinin R'si = tam boyut BE-only
        tot_act += act
        tot_be += be_only
        ex = (r["exit_reason"] or "").lower()
        kind["be" if ex == "be" else ("tp" if ex == "tp" else "diger")] += 1
        print(f"  {r['id']:4d} {r['symbol']:11s} {(r['timeframe'] or ''):3s} {ex:10s} "
              f"{act:+9.2f} {be_only:+10.2f} {be_only - act:+7.2f}")

    n = len(rows)
    print(f"\n  TOPLAM  gercek (kismi + BE): {tot_act:+.2f}R    BE-only: {tot_be:+.2f}R    "
          f"kismi karin katkisi: {tot_act - tot_be:+.2f}R")
    print(f"  Kalan yari: BE {kind['be']} / TP {kind['tp']} / diger {kind['diger']}  "
          f"(BE > TP ise kismi kar lehine)")
    if n:
        print(f"  Islem basina fark: {(tot_act - tot_be) / n:+.3f}R")

    head("Karar")
    if n < MIN_TRADES:
        print(f"  {n}/{MIN_TRADES} islem — KARAR VERME. Esik dolmadan okunan fark gurultudur:")
        print("  tek bir yuksek RR'li islem isareti cevirir (replay'de donemden doneme +8.5R / -0.2R).")
        print(f"  Dashboard 'Dikkat' panosu {MIN_TRADES}'e ulasinca kendiliginden hatirlatir.")
    else:
        print(f"  {n} islem birikti — IZLEME.md 'Kismi kar vs BE-only' karar kuralini uygula:")
        print("  isaret + islem basina fark + BE/TP sayaci birlikte okunur; sonucu oraya yaz.")
    print("  Sinir: 1H'te kismi kar yok. Trail acik olan islemlerde (1H) bu hesap gecerli degil.")
    print("  'diger' cikislar (week_close / trail) simetrik +-0.25xRR kuralinin disinda kalir.")


if __name__ == "__main__":
    main()
