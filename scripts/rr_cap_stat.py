"""RR tavani: yuksek RR'li setupta 3-4R civarindaki mevcut girise gecmek (01.10, kullanici onerisi).

Kullanici: "8RR falan olan setuplarin entry'e gelme ihtimali cok dusuk, daha gercekci 3-4 RR seviyesinde bir entry
secmesi daha iyi olur" -> ayni gun duzeltme: "girisi 3-4R'a cekmek degil, bu civardaki entry hangisiyse ona".
Kural: bugunku girisin RR'si 4'un USTUNDEYSE, motorun bildigi aday girislerden (CISD, MSS, IFVG yakin kenar, BPR yakin
kenar) RR'si bant icinde (3-4) olani secilir; birden coksa bandin en derini (en yuksek RR). Bantta aday yoksa giris
DEGISMEZ. SL ve TP ayni. Genis bant (2.5-5) yalniz bilgi icin basilir.

Mum indirmez, replay degil: `setup_journal.retrace` (0 = seviye dondugu andaki fiyat `ref`, 1 = SL) adaylarin eksendeki
yerini (`d_of`) ve her derinlik x icin duz TP/SL sonucunu verir (`tmp/tmp_gate_passed_depth.py` ile ayni kural):
  tp_first True : x <= d_tp ise dolar ve TP; x > d_tp ise dolmadi (0R)
  tp_first False: fiyat SL'ye gitti, her x dolar, -1R
  ufuk doldu    : 0R
  RR(x) = (rr0 + x) / (1 - x); aday fiyatin gerisindeyse (x < 0) market: 0

Kume (analiz kurali 27.09): kalite kapilarini gecmis, 4H'te seviyeler C2 kapaliyken donmus, bugunku girisin RR'si
>= 2, sonuclanmis, ilk goruldugunde TP'si gecilmemis. KARAR yalniz first_bar'li satirlarda (26.09 oncesi satirlar
ilk mumu gormez -> sig giris LEHINE yanli) ve yalniz kuralin GIRISI DEGISTIRDIGI setuplarda (bugunku RR > 4 ve bantta
aday var) okunur -- degismeyen setup iki kolda ayni, farki sulandirir.

SINIR: duz TP/SL. Canlida kismi kar + BE tetigi girise bagli -> karar oncesi mum re-track'i zorunlu.

Kullanim: python scripts/rr_cap_stat.py [--strategy 4h|1d|1w] [--all]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent

# --- IZLEME.md'ye (01.10) yazilan esikler ------------------------------------------
AFFECTED_MIN_RR = 4.0      # etkilenen dilim: bugunku RR bunun ustu
BAND = (3.0, 4.0)          # karar bandi (kullanici)
WIDE_BAND = (2.5, 5.0)     # yalniz bilgi
CANDIDATES = ("cisd", "mss", "ifvg_near", "bpr_near")   # motorun gercek giris adaylari (yakin kenar = bugunku davranis)
MIN_CHANGED = 20           # first_bar'li, sonuclanmis, girisi degisen setup (havuz)
MIN_CHANGED_TF = 10        # tek stratejiye ozel karar icin
MIN_EDGE_R = 0.15          # yeni giris, ayni setuplarda bugunku girisi bu kadar gecmeli (R/setup)
MIN_RR = 2.0

QUAL_FAIL = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected",
             "bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr", "score7", "tight_stop"}


def load(db: str, strategy: str = "", first_bar_only: bool = True) -> list[dict]:
    """Kapilari gecmis, sonuclanmis setuplar; her biri d (bugunku giris derinligi), rr0, rr ile."""
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute("select * from setup_journal where retrace is not null order by first_seen"):
        if r["best_stage"] in QUAL_FAIL or (strategy and r["strategy"] != strategy):
            continue
        if first_bar_only and r["first_bar"] is None:
            continue
        rt = json.loads(r["retrace"])
        pal = json.loads(r["parts_at_levels"]) if r["parts_at_levels"] else {}
        if r["strategy"] == "4h" and pal.get("c2_closed") != 1:
            continue
        d = (rt.get("d_of") or {}).get("chosen")
        span, ref = rt.get("span"), rt.get("ref")
        if d is None or not span or ref is None or d >= 1 or r["tp"] is None:
            continue
        if rt.get("amb") or (rt.get("tp_first") is None and not rt.get("done")):
            continue
        rr0 = ((r["tp"] - ref) if r["direction"] == "LONG" else (ref - r["tp"])) / span
        if rr0 <= 0:
            continue                                   # ilk goruldugunde TP zaten gecilmis
        d = max(0.0, d)                                # giris fiyatin gerisinde -> aninda dolar
        rr = (rr0 + d) / (1 - d)
        if rr < MIN_RR:
            continue
        out.append(dict(strategy=r["strategy"], symbol=r["symbol"], first_seen=r["first_seen"],
                        rt=rt, d=d, rr0=rr0, rr=rr))
    return out


def outcome(x: float, rec: dict) -> tuple[str, float]:
    rt = rec["rt"]
    rr = (rec["rr0"] + x) / (1 - x)
    if rt.get("tp_first") is True:
        if x > 0 and x > (rt.get("d_tp") or 0.0):
            return "miss", 0.0
        return "tp", rr
    if rt.get("tp_first") is False:
        return "sl", -1.0
    return ("open" if (rt.get("d_max") or 0.0) >= x else "miss"), 0.0


def pick(rec: dict, band: tuple[float, float]) -> tuple[float, str] | None:
    """Bantta aday giris varsa (derinlik, ad); bugunku RR <= 4 ya da bantta aday yoksa None (giris degismez)."""
    if rec["rr"] <= AFFECTED_MIN_RR:
        return None
    lo, hi = band
    best = None
    for name in CANDIDATES:
        d = (rec["rt"].get("d_of") or {}).get(name)
        if d is None or d >= 1:
            continue
        d = max(0.0, d)
        rr = (rec["rr0"] + d) / (1 - d)
        if lo <= rr <= hi and rr < rec["rr"] and (best is None or rr > best[0]):
            best = (rr, d, name)
    return (best[1], best[2]) if best else None


def r_sum(sub: list[dict], band: tuple[float, float] | None) -> float:
    tot = 0.0
    for x in sub:
        p = pick(x, band) if band else None
        tot += outcome(p[0] if p else x["d"], x)[1]
    return tot


def bucket_table(sub: list[dict]) -> None:
    print(f"\n  Bugunku giris, RR kovasina gore (n={len(sub)})")
    print(f"  {'RR':<8}{'setup':>6}{'dolan':>6}{'dolum':>7}{'TP':>4}{'SL':>4}{'R':>8}{'R/setup':>9}")
    for lo, hi in ((2, 3), (3, 4), (4, 6), (6, 99)):
        b = [x for x in sub if lo <= x["rr"] < hi]
        if not b:
            continue
        o = [outcome(x["d"], x) for x in b]
        f = sum(k != "miss" for k, _ in o)
        R = sum(v for _, v in o)
        print(f"  {f'{lo}-{hi}':<8}{len(b):>6}{f:>6}{100 * f / len(b):>6.0f}%{sum(k == 'tp' for k, _ in o):>4}"
              f"{sum(k == 'sl' for k, _ in o):>4}{R:>+8.1f}{R / len(b):>+9.3f}")


def band_table(aff: list[dict], band: tuple[float, float], label: str) -> list[dict]:
    ch = [x for x in aff if pick(x, band)]
    names: dict[str, int] = {}
    for x in ch:
        names[pick(x, band)[1]] = names.get(pick(x, band)[1], 0) + 1
    print(f"\n  {label} {band[0]:g}-{band[1]:g}R: RR > {AFFECTED_MIN_RR:g} olan {len(aff)} setupun {len(ch)}'inde bantta aday var "
          f"({', '.join(f'{k} {v}' for k, v in sorted(names.items())) or '-'})")
    if not ch:
        return ch
    print(f"  {'giris':<10}{'dolan':>6}{'TP':>4}{'SL':>4}{'R':>8}{'R/setup':>9}")
    for nm, b in (("bugunku", None), ("bant", band)):
        o = []
        for x in ch:
            p = pick(x, b) if b else None
            o.append(outcome(p[0] if p else x["d"], x))
        R = sum(v for _, v in o)
        print(f"  {nm:<10}{sum(k != 'miss' for k, _ in o):>6}{sum(k == 'tp' for k, _ in o):>4}"
              f"{sum(k == 'sl' for k, _ in o):>4}{R:>+8.1f}{R / len(ch):>+9.3f}")
    return ch


def verdict(ch: list[dict], need: int, label: str) -> None:
    if len(ch) < need:
        print(f"  KARAR ({label}): veri birikiyor -- {len(ch)}/{need} girisi degisen setup")
        return
    half = len(ch) // 2
    edge = lambda sub: (r_sum(sub, BAND) - r_sum(sub, None)) / len(sub)  # noqa: E731
    e, ha, hb = edge(ch), edge(ch[:half]), edge(ch[half:])
    ok = e >= MIN_EDGE_R and ha > 0 and hb > 0
    print(f"  bant {BAND[0]:g}-{BAND[1]:g}R: fark {e:+.3f} R/setup · yarilar {ha:+.3f} / {hb:+.3f}")
    print(f"  KARAR ({label}): " + ("aday -> mum re-track'i (kismi kar + BE), sonra kullaniciya" if ok
                                   else "bugunku giris KALIR"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="")
    ap.add_argument("--all", action="store_true", help="first_bar oncesi satirlar da (yanli, karar degil)")
    A = ap.parse_args()

    recs = load(A.db, A.strategy, first_bar_only=not A.all)
    title = "TUM satirlar (26.09 oncesi yanli -- karar degil)" if A.all else "first_bar'li satirlar (karar dilimi)"
    print(f"=== RR tavani -- {title}{' · ' + A.strategy.upper() if A.strategy else ''} ===")
    bucket_table(recs)
    aff = [x for x in recs if x["rr"] > AFFECTED_MIN_RR]
    ch = band_table(aff, BAND, "KARAR bandi")
    band_table(aff, WIDE_BAND, "genis bant (bilgi)")
    if A.all:
        return
    print()
    if A.strategy:
        verdict(ch, MIN_CHANGED_TF, A.strategy.upper())
        return
    verdict(ch, MIN_CHANGED, "havuz")
    for s in ("4h", "1d", "1w"):
        sub = [x for x in ch if x["strategy"] == s]
        if sub:
            print(f"  -- {s.upper()}: {len(sub)} girisi degisen setup")
            if len(sub) >= MIN_CHANGED_TF:
                verdict(sub, MIN_CHANGED_TF, s.upper())


if __name__ == "__main__":
    main()
