"""SMT'nin hangi tanimi kazanani ayiriyor? -- bugunku / HTF (C1 ucu) / seviyeye bagli LTF (03.10, kullanici).

Kullanici: "SMT onemli bir konfirme; cikarmak yerine duzeltelim. Altcoin-BTC arasinda olmuyorsa ayri degerlendir; EURUSD/GBPUSD,
XAUUSD/XAGUSD, US100/US500 gibi ciftlerde cok iyi konfirme oluyor."

Tanimlar (setup basina, Journal):
  bugun : score_parts.smt > 0 -- C2 cevresindeki iki LTF mini-swing (`crt_engine.check_smt_divergence`)
  htf   : features.smt_htf -- ana sembol C1 ucunu supurdu, korele sembol kendi C1 ucunu C2 sonuna kadar supurmedi
  lvl   : features.smt_lvl -- ana sembolun supurme anina kadar korele kendi C1 ucunu supurmedi
smt_htf / smt_lvl 03.10'da eklendi (`crt_engine.smt_level_features`) -> yalniz restart sonrasi satirlarda var.
Cift turu (features.smt_kind): alt_btc (altcoin -> BTC) / fx (FX, metal, endeks, petrol ciftleri) / major (BTC <-> ETH).

Kume: Journal, CRT-oncesi asamalar haric, features'ta smt_htf olan (korele cift var), 4H'te C2 kapali donmus, sonuc win/loss
(sinyalde golge). Duz TP/SL. Iki okuma: (a) tum setuplar, (b) KARAR BANDI -- bugunku SMT haric skor esigin 1-2 altinda (+2
sinyali acardi: 4H/1D 5-6, 1W 4-5), tanimi tasiyan vs tasimayan.

Karar kurali: IZLEME.md "SMT doğru tanımla — bugünkü / HTF / seviyeye bağlı" (03.10, veri gorulmeden).
Kullanim: python scripts/smt_variants_stat.py [--strategy 4h] [--since YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.setup_journal import analysis_parts, levels_final  # noqa: E402

ALL_LEVELS = "--all-levels" in sys.argv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "c2_second_sweep", "not_selected"}
THR = {"4h": 7, "1d": 7, "1w": 6}
# --- IZLEME.md'ye veri GORULMEDEN yazilan esikler (03.10) ---
MIN_SIDE = 10      # karar bandinda tanimi tasiyan ve tasimayan, cozulmus
EDGE = 0.15        # R/setup farki
DEFS = (("bugun", "bugunku SMT (+2 alan)"), ("htf", "HTF SMT (C1 ucu, C2 sonuna kadar)"),
        ("lvl", "seviyeye bagli LTF SMT (supurme aninda)"))
KINDS = (("fx", "FX / metal / endeks / petrol ciftleri"), ("alt_btc", "altcoin -> BTC"), ("major", "BTC <-> ETH"))


def _j(s):
    try:
        d = json.loads(s) if s else None
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def load(db, strategy, since):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute("select * from setup_journal where strategy=? and parts_at_levels is not null", (strategy,)):
        if r["best_stage"] in PRE:
            continue
        if strategy != "4h" and not ALL_LEVELS and not levels_final(r):  # 03.10: 1D/1W'de yalniz seviyesi kesin satir (C2 acik donmus = kosan SL, SL yonune yanli); --all-levels eskiler dahil
            continue
        if strategy == "4h" and (_j(r["parts_at_levels"]) or {}).get("c2_closed") != 1:
            continue
        p, f = analysis_parts(r)   # 03.10: 1D/1W'de skor sinyal / C2 kapanisi anindan
        p, f = p or {}, dict(f or {})
        f2 = _j(r["features"]) or {}
        for k in ("smt_htf", "smt_lvl", "smt_kind", "smt_div"):   # seviye donduktan sonra eklenen olcum de sayilsin
            if f.get(k) is None and f2.get(k) is not None:
                f[k] = f2[k]
        if f.get("smt_htf") is None:
            continue
        t = datetime.fromisoformat(str(r["levels_at"] or r["first_seen"])[:19])
        if since and t < datetime.fromisoformat(since):
            continue
        o, rr = r["outcome"], r["rr"]
        if o == "signal":
            sh = (_j(r["shadow"]) or {}).get("1") or {}
            o, rr = sh.get("o"), sh.get("rr", rr)
        if o not in ("win", "loss") or rr is None:
            continue
        smt_now = int((p.get("smt") or 0) > 0 or f.get("smt_div") == 1)   # 29.10: 4H altcoin SMT puansiz, smt_div
        out.append(dict(t=t, kind=f.get("smt_kind") or "?", win=o == "win", R=rr if o == "win" else -1.0,
                        bugun=smt_now, htf=int(f.get("smt_htf") or 0), lvl=int(f.get("smt_lvl") or 0),
                        wo=(p.get("score") or 0) - (p.get("smt") or 0)))
    return out


def st(rs):
    n = len(rs)
    return (n, 100 * sum(x["win"] for x in rs) / n, sum(x["R"] for x in rs) / n) if n else (0, None, None)


def fmt(s):
    return "n=   0" if not s[0] else f"n={s[0]:4d} WR %{s[1]:5.1f} R/setup {s[2]:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="4h", choices=("4h", "1d", "1w"))
    ap.add_argument("--since", default=None)
    ap.add_argument("--all-levels", action="store_true", help="1D/1W: C2 acikken donmus (yanli) eski satirlar da")
    A = ap.parse_args()
    thr = THR[A.strategy]
    recs = load(A.db, A.strategy, A.since)
    print(f"Strateji {A.strategy} · SMT olcumu olan cozulmus setup {len(recs)} (olcum 03.10 restart'indan sonra baslar)")
    if not recs:
        print("   veri yok -- olcum restart'tan sonra birikir")
    for kind, kname in KINDS:
        rs = [x for x in recs if x["kind"] == kind]
        print(f"\n## {kname} -- {len(rs)} setup")
        if not rs:
            continue
        band = [x for x in rs if thr - 2 <= x["wo"] <= thr - 1]
        half = sorted(x["t"] for x in band)[len(band) // 2] if band else None
        for d, dname in DEFS:
            a, b = st([x for x in rs if x[d]]), st([x for x in rs if not x[d]])
            ba, bb = st([x for x in band if x[d]]), st([x for x in band if not x[d]])
            print(f"   {dname}")
            print(f"      tum setup  : tasiyan {fmt(a)} · tasimayan {fmt(b)}")
            print(f"      karar bandi: tasiyan {fmt(ba)} · tasimayan {fmt(bb)}", end="")
            if ba[0] >= MIN_SIDE and bb[0] >= MIN_SIDE:
                diff = ba[2] - bb[2]
                hs = []
                for h in ([x for x in band if x["t"] < half], [x for x in band if x["t"] >= half]):
                    ha, hb = st([x for x in h if x[d]]), st([x for x in h if not x[d]])
                    hs.append(ha[2] - hb[2] if ha[0] and hb[0] else None)
                both = all(v is not None and v > 0 for v in hs)
                verdict = ("KAZANANI AYIRIYOR" if diff >= EDGE and both else
                           "TERS" if diff <= -EDGE else "fark yok")
                print(f" · fark {diff:+.3f}, yarilar " + " / ".join(f"{v:+.2f}" if v is not None else "-" for v in hs)
                      + f" -> {verdict}")
            else:
                print(f" · veri az (taraf basina {MIN_SIDE})")
    print("\nKarar kurali: IZLEME.md 'SMT doğru tanımla — bugünkü / HTF / seviyeye bağlı'. Motor puani bugunku tanimdan.")


if __name__ == "__main__":
    main()
