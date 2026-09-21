"""1D bias tahmin karnesi: yon dogru muydu? (tablo: bias_journal)

Bias bir YON IDDIASIDIR; islemden bagimsiz, hava durumu tahmini gibi puanlanir. "Islem bias
yonunde gitti mi" diye BAKMIYORUZ: hard filtre zaten yalnizca hizali setuplari geciriyor (kontrol
grubu yok) ve TP/SL'ye gidisi entry/stop yerlesimi belirliyor.

Olculen:
  - ISABET: bias yonu ile ileriye donuk hareketin isareti uyusuyor mu (yonlu satirlarda).
  - BUYUKLUK: tahmin yonundeki ortalama hareket, ATR biriminde (yon dogru ama hareket sifirsa
    tahminin ticari degeri yoktur).
  - REFERANS: `momentum` ("dun ne yaptiysa bugun de onu yapar") ve yazi-tura (%50).
    Bias bunlari gecemiyorsa hard filtre de skordaki +-2 de dayanaksiz kalir.
  - BILESEN: structure-only / ict-only / birlesik; ve STRUCTURE_STALE_DAYS esiginin etkisi.

UYARI (istatistik): ardisik gunler ayni ileri pencereyi paylasir ve semboller birbirine korele
(kripto toplu hareket eder), yani asagidaki standart hata GERCEKTEN OLDUGUNDAN KUCUKTUR. Bu yuzden
karar kurali ciplak "anlamlilik" degil, MARJ (>= 3 puan) arar. Sembol bazli dagilim da basilir.

Kullanim:
    python scripts/bias_stat.py
    python scripts/bias_stat.py --days 60 --market crypto
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler ---------------------------------
MIN_ROWS = 500          # ufuk basina yonlu satir; altinda karar verilmez
MIN_EDGE_PTS = 3.0      # isabette hem %50'yi hem momentum'u bu kadar puan gecmeli
MIN_HORIZONS = 2        # 3 ufkun en az bu kadarinda saglanmali
MIN_MOVE_ATR = 0.10     # tahmin yonundeki ortalama hareket (ATR birimi)
HORIZONS = ("fwd_1d", "fwd_3d", "fwd_5d")
DIRECTIONAL = ("BULLISH", "BEARISH")


def head(t: str) -> None:
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


def sign(label: str) -> int:
    return 1 if label == "BULLISH" else (-1 if label == "BEARISH" else 0)


def score(rows: list[dict], field: str, horizon: str):
    """(n, isabet%, tahmin yonundeki ort. hareket ATR, se) -- yalniz yonlu satirlar."""
    moves = []
    for r in rows:
        s = sign(r.get(field) or "")
        v = r.get(horizon)
        if not s or v is None:
            continue
        moves.append(s * float(v))          # pozitif = tahmin yonunde
    if not moves:
        return 0, 0.0, 0.0, 0.0
    hit = 100 * sum(1 for m in moves if m > 0) / len(moves)
    mean = st.mean(moves)
    se = (st.pstdev(moves) / (len(moves) ** 0.5)) if len(moves) > 1 else 0.0
    return len(moves), hit, mean, se


def abs_move(rows: list[dict], field: str, horizon: str, labels) -> tuple[int, float]:
    vals = [abs(float(r[horizon])) for r in rows
            if (r.get(field) or "") in labels and r.get(horizon) is not None]
    return (len(vals), st.mean(vals)) if vals else (0, 0.0)


def table(rows: list[dict], fields: list[tuple[str, str]]) -> dict:
    out: dict = {}
    print(f"{'tahmin':<22}" + "".join(f"{h[4:]:>26}" for h in HORIZONS))
    print(f"{'':22}" + "".join(f"{'n / isabet / hareket':>26}" for _ in HORIZONS))
    for field, label in fields:
        cells = []
        for h in HORIZONS:
            n, hit, mean, se = score(rows, field, h)
            out[(field, h)] = (n, hit, mean, se)
            cells.append(f"{n:>6} {hit:>5.1f}% {mean:>+6.3f}ATR" if n else "-")
        print(f"{label:<22}" + "".join(f"{c:>26}" for c in cells))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0, help="son N gun (0 = hepsi)")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    ap.add_argument("--symbol", default="")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB bulunamadi: {args.db}")
        return
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("SELECT 1 FROM bias_journal LIMIT 1")
    except sqlite3.OperationalError:
        print("bias_journal tablosu yok. Sunucu acilista olusturur (init_db) ve doldurur "
              "(app/bias_journal.py -> refresh).")
        return
    sql = "SELECT * FROM bias_journal WHERE 1=1"
    params: list = []
    if args.days:
        sql += " AND day >= ?"
        params.append((datetime.utcnow() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S"))
    if args.market:
        sql += " AND market_type = ?"
        params.append(args.market)
    if args.symbol:
        sql += " AND symbol = ?"
        params.append(args.symbol)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY day", params)]
    if not rows:
        print("bias_journal bos. Sunucu acilista ve her gun kapanisinda doldurur "
              "(app/bias_journal.py -> refresh).")
        return

    days = sorted({str(r["day"])[:10] for r in rows})
    syms = sorted({r["symbol"] for r in rows})
    head(f"1D BIAS TAHMIN KARNESI | {len(rows)} satir | {len(syms)} sembol | {days[0]} - {days[-1]}")

    res = table(rows, [
        ("combined", "birlesik (motorun)"),
        ("structure", "structure"),
        ("ict", "ict"),
        ("weekly", "haftalik (1W)"),
        ("momentum", "REFERANS momentum"),
    ])

    head("NEUTRAL gercekten yonsuz mu? (ort. MUTLAK hareket, ATR)")
    for field in ("combined", "structure", "ict"):
        cells = []
        for h in HORIZONS:
            n, m = abs_move(rows, field, h, ("NEUTRAL",))
            _, dm = abs_move(rows, field, h, DIRECTIONAL)
            cells.append(f"{n:>5} {m:>6.3f} (yonlu {dm:.3f})")
        print(f"  {field:<12}" + "  ".join(cells))

    head("STRUCTURE_STALE_DAYS esigi (karar ict'ye birakildi mi) -- birlesik bias isabeti")
    for flag, lbl in ((1, "bayat (karar ict'ye birakildi)"), (0, "taze (structure VE ict)")):
        sub = [r for r in rows if (r["stale_applied"] or 0) == flag]
        cells = []
        for h in HORIZONS:
            n, hit, mean, _ = score(sub, "combined", h)
            cells.append(f"{n:>5} {hit:>5.1f}% {mean:>+6.3f}" if n else "-")
        print(f"  {lbl:<32}" + "  ".join(f"{c:>21}" for c in cells))

    head("Sembol bazli dagilim (birlesik, fwd_3d) -- korelasyon uyarisi icin")
    per = []
    for s in syms:
        sub = [r for r in rows if r["symbol"] == s]
        n, hit, mean, _ = score(sub, "combined", "fwd_3d")
        if n >= 10:
            per.append((hit, s, n))
    per.sort()
    if per:
        hits = [h for h, _, _ in per]
        print(f"  {len(per)} sembol | isabet medyan %{st.median(hits):.1f} | "
              f"%50'nin ustunde olan: {sum(1 for h in hits if h > 50)}/{len(hits)}")
        print("  en dusuk :", ", ".join(f"{s} %{h:.0f}" for h, s, _ in per[:3]))
        print("  en yuksek:", ", ".join(f"{s} %{h:.0f}" for h, s, _ in per[-3:]))

    head("KARAR (IZLEME.md '1D bias tahmin karnesi' kurali; veri gorulmeden yazildi)")
    passed = []
    for h in HORIZONS:
        cn, chit, cmean, cse = res[("combined", h)]
        mn, mhit, _, _ = res[("momentum", h)]
        ok_n = cn >= MIN_ROWS
        ok_hit = chit >= 50 + MIN_EDGE_PTS and chit >= mhit + MIN_EDGE_PTS
        ok_move = cmean >= MIN_MOVE_ATR and cse > 0 and cmean / cse >= 2
        passed.append(ok_n and ok_hit and ok_move)
        print(f"  {h}: n={cn} (>={MIN_ROWS}: {'E' if ok_n else 'H'}) | "
              f"isabet %{chit:.1f} vs momentum %{mhit:.1f} ve %50 (+{MIN_EDGE_PTS}p: {'E' if ok_hit else 'H'}) | "
              f"hareket {cmean:+.3f} ATR (>={MIN_MOVE_ATR} ve 2xSE: {'E' if ok_move else 'H'})")
    n_ok = sum(passed)
    print()
    if min(res[("combined", h)][0] for h in HORIZONS) < MIN_ROWS:
        print(f"  -> KARAR YOK: ornek yetersiz (ufuk basina >= {MIN_ROWS} yonlu satir gerekli).")
    elif n_ok >= MIN_HORIZONS:
        print(f"  -> BIAS TAHMIN DEGERI TASIYOR ({n_ok}/3 ufuk). Hard filtre ve skordaki +-2 yerinde kalir.")
    else:
        print(f"  -> BIAS REFERANSI GECEMEDI ({n_ok}/3 ufuk). Hard filtreyi GEVSETME ADAYI -- kaldirma degil:")
        print("     Setup Journal huni olcumu bias kapisinin koruyucu oldugunu soyluyor (13-16.09'da")
        print("     -16.7R'lik setup eledi). Celiski varsa once sebebi arastirilir (bkz. IZLEME.md).")
    print("\nSinir: gun alti ufuk (1D'de medyan 12 saatlik islem suresi) bu tablodan olculemez.")


if __name__ == "__main__":
    main()
