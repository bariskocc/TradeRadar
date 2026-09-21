"""Skor kalemleri gercekten kazandiriyor mu? (Setup Journal'dan, terminal)

Soru: skor tablosunda +1 / +2 verdigimiz kalemler gercekten kazanan setuplari mi isaret
ediyor? Toplam skorun ayristirdigi zaten olculuyordu; 16.09'dan itibaren Journal her
setup'in KALEM KALEM kirilimini de saklıyor (`setup_journal.score_parts`), bu rapor onu okur.

Olculen sonuc: setup elendikten/dolmadan sonra fiyatin ne yaptigi (`outcome`):
  win  = entry'ye gelip TP  |  loss = entry'ye gelip SL
Duz TP/SL'dir: kismi kar / BE / trail yoktur, yani gercek R degil, kalem karsilastirmasi icin
ortak bir cetveldir. Sinyale donusen setuplar izlemeden cikar (`outcome='signal'`) -- yani
ust skor bandi secilmis bir alt kumedir, mutlak win% degil KALEMLER ARASI fark okunmalidir.

Kullanim:
    python scripts/score_parts_stat.py                    # tum Journal
    python scripts/score_parts_stat.py --days 30          # son 30 gun
    python scripts/score_parts_stat.py --strategy 4h      # tek strateji
    python scripts/score_parts_stat.py --min-n 40         # kucuk orneklemleri gizle
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

from app.crt_engine import SCORE_PART_KEYS  # noqa: E402

# Kalem -> baslik (raw/score kalem degil, toplam)
LABELS = {
    "base": "C2 rengi: dogru +2 / doji +1",
    "htf": "1D bias (+2 / ters -2)",
    "weekly": "1W uyumu (+1)",
    "c2_closed": "C2 kapali (+1)",
    "pd_major": "PD major PDH/PDL/PWH/PWL (+1)",
    "pd_monthly": "PD aylik PMH/PML (+1)",
    "pd_struct": "PD yapisal FVG/OB (+1)",
    "wick": "Purge rejection wick (+1)",
    "ifvg": "LTF IFVG (+1)",
    "reclaim": "Zayif C2 geri donusu (-2)",
    "smt": "SMT divergence (+2)",
}
PARTS = [k for k in SCORE_PART_KEYS if k not in ("raw", "score")]


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def rate(rows) -> tuple[int, float | None]:
    n = len(rows)
    if not n:
        return 0, None
    return n, 100.0 * sum(1 for r in rows if r["outcome"] == "win") / n


def line(label: str, got, missed, min_n: int) -> None:
    n1, w1 = rate(got)
    n0, w0 = rate(missed)
    if n1 < min_n or n0 < min_n:
        print(f"  {label:34s}  az ornek (puanli {n1}, puansiz {n0})")
        return
    diff = w1 - w0
    flag = "  <-- kalem lehine" if diff >= 5 else ("  <-- kalem ALEYHINE" if diff <= -5 else "")
    print(f"  {label:34s}  puanli {w1:5.1f}% (n={n1:4d})   puansiz {w0:5.1f}% (n={n0:4d})   "
          f"fark {diff:+5.1f}{flag}")


def _load(raw):
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _split(rows, key):
    """Kalemi 'puan aldi' / 'almadi' diye ikiye ayir.

    base her setupta en az +1 oldugu icin +2 (dogru renk) vs +1 (doji/ayni renk) karsilastirilir;
    reclaim bir CEZA kalemi, orada 'puanli' = cezayi yiyen taraf; htf'in -2'si ayri satirda.
    """
    if key == "base":
        return [r for r in rows if r["parts"].get("base", 0) == 2],                [r for r in rows if r["parts"].get("base", 0) == 1]
    if key == "reclaim":
        return [r for r in rows if r["parts"].get("reclaim", 0) < 0],                [r for r in rows if r["parts"].get("reclaim", 0) == 0]
    return [r for r in rows if r["parts"].get(key, 0) > 0],            [r for r in rows if r["parts"].get(key, 0) == 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0, help="son N gun (0 = hepsi)")
    ap.add_argument("--strategy", default="", help="4h | 1d | 1h (bos = hepsi)")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    ap.add_argument("--min-n", type=int, default=25, help="altinda 'az ornek' yazilir")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("PRAGMA table_info(setup_journal)")}
    if "score_parts" not in cols:
        con.close()
        print("setup_journal.score_parts kolonu yok - migrasyon uygulanmamis. Sunucu bir kez "
              "yeniden baslayinca (init_db) kolon eklenir ve yeni setuplar kirilimi yazmaya baslar.")
        return
    has_snap = "parts_at_levels" in cols
    sql = ("SELECT strategy, symbol, market_type, score, score_parts, outcome, first_seen"
           + (", parts_at_levels, features, features_at_levels" if has_snap else "")
           + " FROM setup_journal WHERE score_parts IS NOT NULL AND outcome IN ('win','loss')")
    params: list = []
    if args.days:
        sql += " AND first_seen >= ?"
        params.append((datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S"))
    if args.strategy:
        sql += " AND strategy = ?"
        params.append(args.strategy)
    if args.market:
        sql += " AND market_type = ?"
        params.append(args.market)
    rows = []
    frozen = 0
    for r in con.execute(sql, params):
        # Seviyeler DONDUGU andaki kirilim tercih edilir: sonuc o seviyelerden izleniyor,
        # sonradan degisen skorla (C2 kapanisi / SMT) eslestirmek yaniltici olur.
        snap = _load(r["parts_at_levels"]) if has_snap else None
        parts = snap or _load(r["score_parts"])
        if not parts:
            continue
        frozen += 1 if snap else 0
        feats = (_load(r["features_at_levels"]) or _load(r["features"])) if has_snap else {}
        rows.append({"outcome": r["outcome"], "score": (snap or {}).get("score", r["score"]),
                     "parts": parts, "features": feats or {},
                     "strategy": r["strategy"], "symbol": r["symbol"]})
    con.close()

    if not rows:
        print("Kirilimli ve sonuclanmis kayit yok. score_parts 16.09'da eklendi; oncesindeki "
              "satirlarda NULL. Once veri biriksin.")
        return

    n, w = rate(rows)
    head(f"Skor kalemleri - {n} sonuclanmis setup, genel win {w:.1f}%")
    print("  win = entry'ye gelip TP, loss = entry'ye gelip SL (duz TP/SL; kismi kar/BE yok).")
    print("  Sinyale donusenler izlemeden cikar -> mutlak oran degil, KALEMLER ARASI fark okunur.")

    head("1) Toplam skor (kontrol: kirilim olmadan da bildigimiz)")
    for sc in sorted({int(r["score"]) for r in rows if r["score"] is not None}):
        sub = [r for r in rows if r["score"] is not None and int(r["score"]) == sc]
        m, ww = rate(sub)
        print(f"  skor {sc:2d}   n={m:4d}   win {ww:5.1f}%   {'#' * max(1, round(ww / 4))}")

    head("2) Kalem kalem - puan aldi mi / almadi mi")
    for k in PARTS:
        pool = rows
        if k == "smt":
            elig = [r for r in rows if r["features"].get("smt_possible") == 1]
            pool = elig or rows      # bayrak yoksa (eski satir) hepsi
            if elig:
                print(f"  {'(SMT yalniz korele paritesi olanlarda)':34s}  n={len(elig)}")
        if k == "ifvg":
            pool = [r for r in rows if r["features"].get("ifvg_checked", 1) == 1] or rows
        got, missed = _split(pool, k)
        line(LABELS[k], got, missed, args.min_n)
    neg = [r for r in rows if r["parts"].get("htf", 0) < 0]
    if neg:
        n2, w2 = rate(neg)
        print(f"  {'1D bias TERS (-2)':34s}  n={n2:4d}   win {w2:5.1f}%")

    head("3) Ayni toplam skor bandinda (confounding kontrolu)")
    print("  Yuksek skorlu setup zaten cok kalemi birden tasir; asagisi kalemi ayni bant icinde")
    print("  karsilastirir - kalemin KENDI katkisina en yakin okuma budur.\n")
    for lo, hi in ((0, 4), (5, 6), (7, 11)):
        band = [r for r in rows if r["score"] is not None and lo <= r["score"] <= hi]
        m, ww = rate(band)
        if not m:
            continue
        print(f"  --- skor {lo}-{hi}: n={m}, win {ww:.1f}% ---")
        for k in PARTS:
            got, missed = _split(band, k)
            line(LABELS[k], got, missed, max(10, args.min_n // 2))
        print()

    feat_rows = [r for r in rows if r["features"]]
    if feat_rows:
        head("4) Surekli olcumler - esik dogru yerde mi? (ceyreklik win%)")
        print("  Kalemin ikili esigi yerine olcumun kendisi: win% ceyrekler arasinda artiyorsa")
        print("  esik anlamli, duzse kalem ikili olarak bilgi tasimiyor demektir.")
        print("")
        for f, label, thr in (("wick_frac", "purge wick / C1 range", "esik 0.30"),
                              ("c2_body_frac", "C2 govde / range", "doji esigi"),
                              ("reclaim_pct", "C2 geri donus %", "ceza < 25"),
                              ("ifvg_gap_frac", "IFVG bosluk / LTF range", "esik 0.15"),
                              ("range_atr", "C1 range / ATR", "band 0.8-2.5"),
                              ("stop_range_mult", "stop mesafesi / LTF range", "dar stop esigi 1.0")):
            vals = sorted(((r["features"][f], r) for r in feat_rows
                           if isinstance(r["features"].get(f), (int, float))),
                          key=lambda t: t[0])      # esitlikte kayitlari kiyaslamaya calismasin
            if len(vals) < 4 * max(8, args.min_n // 4):
                print(f"  {label:28s} ({thr})  az ornek (n={len(vals)})")
                continue
            q = len(vals) // 4
            parts_out = []
            for i in range(4):
                chunk = vals[i * q:(i + 1) * q] if i < 3 else vals[3 * q:]
                m, ww = rate([r for _v, r in chunk])
                lo_v, hi_v = chunk[0][0], chunk[-1][0]
                parts_out.append(f"[{lo_v:.2f}-{hi_v:.2f}] {ww:4.1f}%")
            print(f"  {label:28s} ({thr})  " + "  ".join(parts_out))

    head("Yorum notu")
    print("  Tek bir kalemin farki, orneklem kucukken rahatlikla +-10 puan salinir. Karar icin")
    print("  kalem basina her iki tarafta da >= 50 sonuclanmis setup ve tercihen iki ayri hafta")
    print("  bekleyin; ayrica 'ayni bant' tablosunda da ayni yonde olsun. Degerlendirme tarihi ve")
    print("  onceden yazilmis karar kurali: IZLEME.md -> 'Skor kalemleri'.")
    if has_snap:
        print(f"  Seviye anindaki dondurulmus kirilimle okunan satir: {frozen}/{len(rows)}"
              " (kalani en guncel kirilim).")


if __name__ == "__main__":
    main()
