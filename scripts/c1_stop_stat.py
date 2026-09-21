"""C1 ucu stop golge izlemesi: SL purge ucu yerine C1 ucu olsa ayni isler hayatta kalir miydi?

Veri: setup_journal.shadow -> "c1" varyanti (17.09'dan beri seviyelenen satirlar). Karsilastirma
ESLESMIS kumede yapilir: ayni satirin shadow["1"] (bugunku kural) sonucu ile shadow["c1"] sonucu.
Kumeler, esikler ve karar kurali IZLEME.md -> "C1 ucu stop - golge izleme" icinde ONCEDEN yazildi;
bu betik yalnizca oradaki kurali calistirir.

  C  = shadow["c1"], o != invalid VE (mult >= 1.0, mult yoksa k >= 0.15)   [dar stop kapisi]
  A  = ayni satirlarin shadow["1"] sonucu
  R  : win = +rr, loss = -1, tp_before_entry / no_touch = 0; ambiguous/open ayri raporlanir
  H1 : C'nin R/islem'i A'yi >= 0.15R gecer VE C'de dolan >= 40 islem  -> dogrulama replay'i
  H2 : invalid + (mult < 1.0) orani > %20 -> kural fallback'siz uygulanamaz (bilgi)
  H3 : best_stage = low_rr VE C1 RR >= 2 -> C1 stopu bu setuplari RR kapisindan gecirirdi,
       yani EK sinyal olurdu; sonuclari ayri raporlanir (bilgi; H1 kumesini degistirmez)

Kullanim:
    python scripts/c1_stop_stat.py
    python scripts/c1_stop_stat.py --days 30 --strategy 4h
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

# IZLEME.md'deki onceden yazilmis esikler.
MIN_TRADES = 40          # C kumesinde entry'ye dokunan islem
MIN_EDGE_R = 0.15        # C, A'yi bu kadar R/islem gecmeli
MIN_MULT = 1.0           # dar stop kapisi (mult = k x stop_range_mult)
MIN_K_FALLBACK = 0.15    # mult olculememisse
MAX_UNUSABLE_PCT = 20    # H2 bilgi esigi
MIN_RR_GATE = 2.0        # motorun RR kapisi (scanner.MIN_RR_RATIO / CRYPTO_ / FX_ -- ucu de 2.0)

ZERO = ("tp_before_entry", "no_touch")
SKIP = ("ambiguous", "open", "pending", "filled", "signal")


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def score(variants: list[dict]) -> tuple[int, int, int, float]:
    """(dolan, win, loss, toplam R) - ZERO sonuclar 0R sayilir, SKIP disarida."""
    w = l = 0
    total = 0.0
    for v in variants:
        o = v.get("o")
        if o == "win":
            w += 1
            total += float(v.get("rr") or 0)
        elif o == "loss":
            l += 1
            total -= 1
        elif o in ZERO:
            continue
    return w + l, w, l, total


def line(label: str, n: int, w: int, l: int, total: float) -> str:
    wr = f"%{100 * w / n:.1f}" if n else "  -  "
    ev = f"{total / n:+.3f}" if n else "  -   "
    return f"{label:<26} dolan {n:>4}  win {w:>4}  loss {l:>4}  WR {wr:>6}  R {total:>8.1f}  R/islem {ev}"


RESOLVED = ("win", "loss", "tp_before_entry", "no_touch")


def r_of(v: dict) -> float:
    """Bir varyantin R'si: win = +rr, loss = -1, digerleri 0."""
    o = v.get("o")
    if o == "win":
        return float(v.get("rr") or 0)
    return -1.0 if o == "loss" else 0.0


def matrix_and_slices(pairs: list[tuple[dict, dict, float, float]]) -> None:
    """Gecis matrisi + dilimler. `pairs` = (c1, a, score, rr), IKI kolda da cozulmus satirlar.

    Neden ayri kume: C kolu daha SIK ve daha ERKEN cozulur (dar stop once vurur), bu yuzden
    "dolan" sayilari esit degil ve ham R/islem karsilastirmasi A'nin aleyhine kayar. Karar
    girdisi olarak ikisinde de sonucu belli olan satirlar kullanilir.
    """
    if not pairs:
        return
    cells: dict[tuple[str, str], int] = {}
    for c, a, _, _ in pairs:
        key = (a.get("o") or "?", c.get("o") or "?")
        cells[key] = cells.get(key, 0) + 1
    print(f"\n  GECIS MATRISI (ikisinde de cozulmus n={len(pairs)})"
          " -- satir: bugunku SL (C2 ucu), kolon: C1 ucu")
    kinds = [k for k in RESOLVED if any(a == k for a, _ in cells)]
    print("      " + f"{'A -> C':<18}" + "".join(f"{k[:9]:>11}" for k in RESOLVED))
    for a_o in kinds:
        row = [cells.get((a_o, c_o), 0) for c_o in RESOLVED]
        print("      " + f"{a_o:<18}" + "".join(f"{v:>11}" for v in row))
    a_win = sum(v for (a_o, _), v in cells.items() if a_o == "win")
    c_loss_of_a_win = cells.get(("win", "loss"), 0)
    c_loss = sum(v for (_, c_o), v in cells.items() if c_o == "loss")
    if a_win:
        print(f"      > TP olanlar C1 ile: {a_win - c_loss_of_a_win} yine TP, {c_loss_of_a_win} stop"
              f"  (kazananlarin %{100 * c_loss_of_a_win / a_win:.0f}'i olur)")
    if c_loss:
        print(f"      > C1 ile stop olanlarin %{100 * cells.get(('loss', 'loss'), 0) / c_loss:.0f}'i"
              f" bugunku SL ile de stop olacakti (C1'in EK zarari: {c_loss - cells.get(('loss', 'loss'), 0)})")

    print("\n  DILIMLER (ayni kume; kalite dilimi kararin asil girdisi)")
    for label, sel in (("hepsi", lambda sc, rr: True),
                       ("skor >= 7", lambda sc, rr: sc >= 7),
                       ("RR >= 2", lambda sc, rr: rr >= 2),
                       ("skor >= 7 VE RR >= 2", lambda sc, rr: sc >= 7 and rr >= 2)):
        sub = [(c, a) for c, a, sc, rr in pairs if sel(sc, rr)]
        if not sub:
            print(f"      {label:<22} n=   0")
            continue
        ar = sum(r_of(a) for _, a in sub) / len(sub)
        cr = sum(r_of(c) for c, _ in sub) / len(sub)
        aw = sum(1 for _, a in sub if a.get("o") == "win")
        cw = sum(1 for c, _ in sub if c.get("o") == "win")
        flag = "  <-- esigi asiyor" if (cr - ar) >= MIN_EDGE_R and len(sub) >= MIN_TRADES else ""
        print(f"      {label:<22} n={len(sub):>4}   A: WR %{100 * aw / len(sub):>4.1f} R/i {ar:+.3f}"
              f"   C: WR %{100 * cw / len(sub):>4.1f} R/i {cr:+.3f}   fark {cr - ar:+.3f}{flag}")
    print(f"      (fark esigi >= +{MIN_EDGE_R} R/islem ve dilimde >= {MIN_TRADES} islem gerekir)")


def report(rows: list[dict], label: str) -> None:
    head(label)
    usable, unusable, too_tight = [], [], []
    opened, opened_tight = [], []        # H3: RR kapisinin acilacagi setuplar
    matched: list[tuple[dict, dict, float, float]] = []   # ikisinde de cozulmus (matris + dilimler)
    for r in rows:
        sh = r["_shadow"]
        c1 = sh.get("c1")
        if not c1:
            continue
        if c1.get("o") == "invalid":
            unusable.append(c1)
            continue
        mult, k = c1.get("mult"), c1.get("k")
        ok = (mult >= MIN_MULT) if mult is not None else ((k or 0) >= MIN_K_FALLBACK)
        a = sh.get("1") or {}
        pair = (c1, a)
        (usable if ok else too_tight).append(pair)
        if ok and a.get("o") in RESOLVED and c1.get("o") in RESOLVED:
            matched.append((c1, a, float(r.get("score") or 0), float(r.get("rr") or 0)))
        # Motorun RR kapisinda eledigi setup, C1 stopunun yukselttigi RR ile gecerdi mi?
        if r.get("best_stage") == "low_rr" and (c1.get("rr") or 0) >= MIN_RR_GATE:
            (opened if ok else opened_tight).append(pair)

    total_seen = len(usable) + len(unusable) + len(too_tight)
    if not total_seen:
        print("  C1 varyanti olan satir yok (17.09 oncesi kayitlarda bu varyant bulunmaz).")
        return

    cn, cw, cl, cr = score([c for c, _ in usable])
    an, aw, al, ar = score([a for _, a in usable])
    print(line("A  bugunku SL (purge)", an, aw, al, ar))
    print(line("C  C1 ucu SL", cn, cw, cl, cr))
    if cn and an:
        edge = cr / cn - ar / an
        print(f"\n  fark: {edge:+.3f} R/islem   (esik >= +{MIN_EDGE_R})")
    else:
        edge = None

    flipped = sum(1 for c, a in usable if a.get("o") == "win" and c.get("o") == "loss")
    saved = sum(1 for c, a in usable if a.get("o") == "loss" and c.get("o") == "win")
    print(f"  A'da kazanip C'de stop olan: {flipped}   |   C'de kurtulan: {saved}")

    pct = round(100 * (len(unusable) + len(too_tight)) / total_seen)
    why: dict[str, int] = {}
    for v in unusable:
        why[v.get("why") or "?"] = why.get(v.get("why") or "?", 0) + 1
    print(f"\n  H2 - kural uygulanamayan: %{pct} ({len(unusable) + len(too_tight)}/{total_seen})"
          f"  |  {why or '-'}  |  asiri dar (mult < {MIN_MULT}): {len(too_tight)}")
    if pct > MAX_UNUSABLE_PCT:
        print(f"       > %{MAX_UNUSABLE_PCT}: C1 stopu fallback'siz uygulanamaz, bu maliyet karara girer.")

    matrix_and_slices(matched)

    if opened or opened_tight:
        on, ow, ol, orr = score([c for c, _ in opened])
        pn, pw, pl, pr = score([a for _, a in opened])
        print(f"\n  H3 - C1 stopu RR kapisini actigi icin EK sinyal olacak setuplar"
              f" (low_rr, C1 RR >= {MIN_RR_GATE}): {len(opened)}")
        print("  " + line("   bu setuplar C1 stopuyla", on, ow, ol, orr))
        print("  " + line("   ayni setuplar purge SL ile", pn, pw, pl, pr))
        if opened_tight:
            print(f"     dar stop kapisinda geri verilen: {len(opened_tight)}"
                  f" (RR gecti ama mult < {MIN_MULT})")
        print("     Bilgi: sinyal sayisinin artmasi tek basina lehte delil degil -- H1 kumesi ayni.")

    skipped = sum(1 for c, _ in usable if c.get("o") in SKIP)
    if skipped:
        print(f"  raporlanmayan (ambiguous/open/izleniyor): {skipped}")

    print("\n  KARAR:", end=" ")
    if cn < MIN_TRADES:
        print(f"HENUZ YOK - C kumesinde dolan {cn} < {MIN_TRADES}. IZLEME.md: 02.11.2026'ya uzat.")
    elif edge is not None and edge >= MIN_EDGE_R:
        print(f"H1 GECTI ({edge:+.3f} R) - kural HEMEN degismez; once tek dogrulama replay'i "
              "(kismi kar/BE/trail + portfoy kapilariyla).")
    else:
        print("H1 GECMEDI - fikir kapanir. Sonucu IZLEME.md'ye yaz, maddeyi done'a cek.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--days", type=int, default=0, help="son N gun (levels_at'e gore; 0 = hepsi)")
    ap.add_argument("--strategy", default="", help="4h | 1d | 1h")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = ("SELECT strategy, symbol, market_type, direction, rr, score, best_stage, levels_at, shadow "
           "FROM setup_journal WHERE shadow IS NOT NULL AND shadow LIKE '%\"c1\":%'")
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
            d["_shadow"] = json.loads(d["shadow"]) or {}
        except Exception:
            continue
        rows.append(d)

    if not rows:
        print("C1 varyanti tasiyan satir yok. Varyant 17.09'dan itibaren SEVIYELENEN setuplara yazilir;\n"
              "eski satirlarda bulunmaz (izleme ortasinda stop eklemek sonucu bozardi).")
        return

    report(rows, f"TUM STRATEJILER (n={len(rows)})")
    if not args.strategy:
        for s in ("4h", "1d", "1h"):
            sub = [r for r in rows if r["strategy"] == s]
            if sub:
                report(sub, f"STRATEJI {s.upper()} (n={len(sub)})")
    print("\nKarar kurali ve sinirlar: IZLEME.md -> \"C1 ucu stop - golge izleme\"")


if __name__ == "__main__":
    main()
