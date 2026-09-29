"""Sadece CISD onayli setup mi, CISD + MSS onayli setup mi daha cok stop oluyor? (Setup Journal, terminal)

Soru (kullanici, 29.09): motor CISD onayiyla calisiyor, MSS (purge oncesi son swing'in kapanisla kirilmasi)
beklenmiyor. MSS kirildiginda CISD de kirilmis olur. MSS'i henuz kirilmamis setuplar daha cok mu stop oluyor?

Olcum A -- ASIL (29.09+ satirlar, `retrace.brk`): CISD ve MSS seviyesinin purge'den sonra KAPANISLA ilk kirildigi
LTF mumu (scanner `_break_seed` + Journal `_apply_retrace`). Dolum = `entries.chosen.t` (dolum mumunun acilisi).
Kirilim dolumdan ONCEKI bir mumda olmali (ayni mum: sira bilinmez -> "once" sayilmaz). Gruplar:
  sadece CISD   : dolumdan once CISD kirilmis, MSS kirilmamis
  CISD + MSS    : ikisi de dolumdan once kirilmis
  (sadece MSS / onaysiz dolum: bilgi -- onaysiz = Journal seviyeleri onaydan once dondurup doldurmus)

Olcum B -- ESKI satirlar (brk yok), tek an: seviyeler donarken `retrace.ref` (o anki CANLI fiyat -- oluşan mumun
son fiyati, kapanis degil) seviyenin otesinde mi. Onaydan once donmus satirlar "hicbiri"ne duser. Yalniz ilk okuma.

Sonuc `entries.chosen` (motorun sectigi giris, duz TP/SL; sinyale donusen setupta da izlenir):
  win = giris -> TP, loss = giris -> SL. SL% = loss / (win + loss).

Karar kurali (onceden yazildi, IZLEME.md "CISD onayi mi, CISD + MSS onayi mi?"): OLCUM A, kapilardan gecmis kume,
her grupta >= 20 cozulmus setup; "sadece CISD" grubu "CISD + MSS"ten
  - >= 0.30 R/setup kotuyse  -> MSS sarti (hard) adayi
  - 0.15-0.30 kotuyse        -> -2 skor cezasi adayi
  - < 0.15                   -> degisiklik yok
Ikisi de once tek replay'den gecer (MSS beklemek dolumu geciktirir/kacirir -- Journal bunu olcemez).

Kumeler:
  genis     : motorun CRT saydigi setuplar, 4H/1D'de seviyeler C2 kapaliyken donmus
  kapilar   : ayrica kalite kapilarini gecmis (bias/target/skor/no_cisd/low_rr/score7/tight_stop) ve RR >= 2
  first_bar : kapilar kumesinin 26.09+ satirlari (ilk mum da izleniyor; oncesi sig giris lehine yanli)

Kullanim: python scripts/cisd_mss_stat.py [--strategy 4h] [--market crypto]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent

NOT_CRT = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color"}
QUAL_FAIL = NOT_CRT | {"not_selected", "bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr",
                       "score7", "tight_stop"}
MIN_RR = 2.0
GROUPS = ("sadece CISD", "CISD + MSS", "sadece MSS", "hicbiri")
GROUPS_A = ("sadece CISD", "CISD + MSS", "sadece MSS", "onaysiz dolum")
MIN_N = 20


def _j(raw):
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _broken(long: bool, ref: float, level) -> bool | None:
    if level is None:
        return None
    return ref > float(level) if long else ref < float(level)


def _row(label: str, sub: list) -> str:
    n = len(sub)
    if not n:
        return f"  {label:14s} n=   0"
    w = sum(1 for r in sub if r["o"] == "win")
    rs = sum(r["rr"] if r["o"] == "win" else -1.0 for r in sub) / n
    rr = sum(r["rr"] for r in sub) / n
    return (f"  {label:14s} n={n:4d}  TP {w:4d}  SL {n - w:4d}  SL% {100 * (n - w) / n:5.1f}  "
            f"R/setup {rs:+.3f}  ort. RR {rr:.2f}")


def _rs(sub: list) -> float:
    return sum(r["rr"] if r["o"] == "win" else -1.0 for r in sub) / len(sub)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d | 1w | 1h (bos = hepsi)")
    ap.add_argument("--market", default="", help="crypto | fx | metal | index | oil")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = {"genis": [], "kapilar": [], "first_bar": []}
    rows_a = {"genis": [], "kapilar": []}
    same_level = 0
    for r in con.execute("SELECT * FROM setup_journal WHERE entries IS NOT NULL AND retrace IS NOT NULL"):
        if r["best_stage"] in NOT_CRT:
            continue
        if args.strategy and r["strategy"] != args.strategy:
            continue
        if args.market and r["market_type"] != args.market:
            continue
        pal = _j(r["parts_at_levels"]) or {}
        if r["strategy"] in ("4h", "1d") and pal.get("c2_closed") != 1:
            continue                                   # C2 acikken donmus: SL/TP kosan purge ucunda
        ents, rt = _j(r["entries"]) or {}, _j(r["retrace"]) or {}
        ch = ents.get("chosen") or {}
        if ch.get("o") not in ("win", "loss") or ch.get("rr") is None or rt.get("ref") is None:
            continue
        long = r["direction"] == "LONG"
        gated = r["best_stage"] not in QUAL_FAIL and float(ch["rr"]) >= MIN_RR
        brk = rt.get("brk")
        if isinstance(brk, dict):
            fill = ch.get("t")
            if not fill:
                continue
            fill = str(fill)[:19]
            ca = brk.get("cisd") and str(brk["cisd"])[:19] < fill
            ma = brk.get("mss") and str(brk["mss"])[:19] < fill
            g = GROUPS_A[0] if ca and not ma else GROUPS_A[1] if ca and ma else GROUPS_A[2] if ma else GROUPS_A[3]
            rec = {"g": g, "o": ch["o"], "rr": float(ch["rr"])}
            rows_a["genis"].append(rec)
            if gated:
                rows_a["kapilar"].append(rec)
            continue
        c = _broken(long, float(rt["ref"]), (ents.get("cisd") or {}).get("e"))
        m = _broken(long, float(rt["ref"]), (ents.get("mss") or {}).get("e"))
        if c is None or m is None:
            continue
        if (ents.get("cisd") or {}).get("e") == (ents.get("mss") or {}).get("e"):
            same_level += 1
        g = GROUPS[0] if c and not m else GROUPS[1] if c and m else GROUPS[2] if m else GROUPS[3]
        rec = {"g": g, "o": ch["o"], "rr": float(ch["rr"]), "model": r["model"]}
        rows["genis"].append(rec)
        if gated:
            rows["kapilar"].append(rec)
            if r["first_bar"] is not None:
                rows["first_bar"].append(rec)
    con.close()

    print(f"CISD vs CISD + MSS -- strateji: {args.strategy or 'hepsi'}, market: {args.market or 'hepsi'}")
    print("Sonuc: motorun sectigi giris, duz TP/SL (kismi kar / BE yok).")
    print("\n###### OLCUM A -- dolumdan once kapanisla kirilim (29.09+ satirlar) ######")
    for key, title in (("genis", "Genis kume (bilgi)"),
                       ("kapilar", f"Kapilardan gecmis (kalite kapilari + RR >= {MIN_RR:g}) -- KARAR KUMESI")):
        sub = rows_a[key]
        print(f"\n== {title}")
        for g in GROUPS_A:
            print(_row(g, [r for r in sub if r["g"] == g]))
    ka = rows_a["kapilar"]
    only = [r for r in ka if r["g"] == GROUPS_A[0]]
    both = [r for r in ka if r["g"] == GROUPS_A[1]]
    if len(only) < MIN_N or len(both) < MIN_N:
        print(f"\n  KARAR: veri birikiyor (grup basina {MIN_N}; sadece CISD {len(only)} / CISD + MSS {len(both)}).")
    else:
        gap = _rs(both) - _rs(only)
        verdict = ("MSS sarti (hard) adayi" if gap >= 0.30 else "-2 skor cezasi adayi" if gap >= 0.15
                   else "degisiklik yok")
        print(f"\n  KARAR: sadece CISD {gap:+.3f} R/setup geride -> {verdict}"
              + (" (once tek replay)." if gap >= 0.15 else "."))

    print("\n###### OLCUM B -- eski satirlar, tek an (canli fiyat), yalniz ilk okuma ######")
    titles = {"genis": "Genis kume (motorun CRT saydigi, C2 kapali donmus)",
              "kapilar": f"Kapilardan gecmis (kalite kapilari + RR >= {MIN_RR:g})",
              "first_bar": "Kapilardan gecmis, 26.09+ (first_bar)"}
    for key, sub in rows.items():
        print(f"\n== {titles[key]}")
        for g in GROUPS:
            print(_row(g, [r for r in sub if r["g"] == g]))
    print(f"\n(CISD ve MSS seviyesi ayni olan setup: {same_level} -- 'CISD + MSS' grubuna duser)")


if __name__ == "__main__":
    main()
