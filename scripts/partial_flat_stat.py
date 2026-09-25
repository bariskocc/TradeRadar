"""Kismi kar + BE'yi kapatsak mi? -- %50'ye gelen islemler sonunda TP mi, SL mi? (Setup Journal)

Canli sinyalde BE kalan yariyi kapattigi icin "BE olmasaydi ne olurdu" gorulmuyor. Setup Journal'in
k=1 golgesi ise duz TP/SL ile izliyor ve sinyale donusse de devam ediyor; 24.09'dan beri ayni
golgeye motorun kuralini da isletiyor (`setup_journal._apply_partial_arm`):
    pa  -> TP yolunun %50'sine degildi (motordaki tetik)
    cur -> motorun kuraliyla kalan yari: "tp" | "be" | "amb"
    o   -> duz TP/SL sonucu: "win" | "loss" | ...
Ikisi AYNI setupta, ayni seviyelerle -> fark secilim etkisinden arinmis.

R cetveli (motorla ayni, 4H/1D: yari %50'de kapanir, kalan yari SL=entry):
    bugunku kural = 0.5 x (0.5 x RR) + 0.5 x (RR if cur == "tp" else 0)
    duz TP/SL     = RR if o == "win" else -1
%50'ye gelmeyen islemde iki kural ayni sonucu verir (fark 0) -> kiyas yalniz tetiklenenlerde.
Basabas: tetiklenenlerin duz TP orani > (1 + 0.25 RR) / (1 + 0.5 RR) ise kapatmak kazandirir
(RR 2'de %75, RR 3'te %70) -- BE'den sonra TP'ye giden olmadigi varsayimiyla; rapor ayrica sayar.

Karar kurali: IZLEME.md "Kısmi kâr + BE kapatılsın mı?".

Kullanim:
    python scripts/partial_flat_stat.py
    python scripts/partial_flat_stat.py --strategy 4h --market crypto
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.setup_journal import PRE_SETUP_STAGES  # noqa: E402

# IZLEME.md'deki onceden yazilmis esik (ana dilimde, tetiklenmis ve iki kolu da cozulmus setup).
MIN_ARMED = 100
BAND = 0.15            # R / tetiklenen setup; |fark| bunun altindaysa "fark yok"


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def cur_r(rr: float, cur: str) -> float:
    return 0.5 * (0.5 * rr) + 0.5 * (rr if cur == "tp" else 0.0)


def flat_r(rr: float, o: str) -> float:
    return rr if o == "win" else -1.0


def load(db: str, strategy: str, market: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = ("SELECT strategy, market_type, symbol, direction, purge_time, best_stage, shadow, "
           "parts_at_levels FROM setup_journal WHERE shadow IS NOT NULL")
    params: list = []
    if strategy:
        sql += " AND strategy = ?"
        params.append(strategy)
    else:
        sql += " AND strategy IN ('4h', '1d')"          # 1H'te kismi kar yok (trail var)
    if market == "crypto":
        sql += " AND market_type = 'crypto'"
    elif market == "fx":
        sql += " AND market_type != 'crypto'"
    out = []
    for r in con.execute(sql, params):
        k1 = (json.loads(r["shadow"]) or {}).get("1") or {}
        if not k1.get("pt"):
            continue                                    # 24.09 oncesi / dolumdan sonra izlemeye alinmis
        pal = json.loads(r["parts_at_levels"]) if r["parts_at_levels"] else {}
        out.append({
            "strategy": r["strategy"], "symbol": r["symbol"], "direction": r["direction"],
            "purge_time": r["purge_time"], "pre": r["best_stage"] in PRE_SETUP_STAGES,
            "signal": r["best_stage"] == "waiting",
            "score": pal.get("score"), "c2": bool(pal.get("c2_closed")),
            "rr": k1.get("rr"), "o": k1.get("o"), "pa": k1.get("pa"), "cur": k1.get("cur"),
        })
    con.close()
    return out


def tally(rows: list[dict]) -> dict:
    """Tetiklenmis ve iki kolu da cozulmus satirlarda sayim + R."""
    t = {"filled": 0, "armed": 0, "open": 0, "amb": 0, "n": 0, "flat": 0.0, "cur": 0.0,
         "tp": 0, "be": 0, "be_win": 0, "be_loss": 0, "flat_win": 0, "pstar": 0.0}
    for r in rows:
        if r["o"] in ("filled", "win", "loss", "open", "ambiguous"):
            t["filled"] += 1
        if not r["pa"]:
            continue
        t["armed"] += 1
        if r["cur"] == "amb" or r["o"] == "ambiguous":
            t["amb"] += 1
            continue
        if r["o"] not in ("win", "loss") or r["cur"] not in ("tp", "be") or not r["rr"]:
            t["open"] += 1                              # ufukta cozulmedi
            continue
        rr = float(r["rr"])
        t["n"] += 1
        t["flat"] += flat_r(rr, r["o"])
        t["cur"] += cur_r(rr, r["cur"])
        t["flat_win"] += r["o"] == "win"
        t["pstar"] += (1 + 0.25 * rr) / (1 + 0.5 * rr)
        if r["cur"] == "tp":
            t["tp"] += 1
        else:
            t["be"] += 1
            t["be_win"] += r["o"] == "win"
            t["be_loss"] += r["o"] == "loss"
    return t


def show(name: str, t: dict) -> None:
    n = t["n"]
    if not n:
        print(f"  {name:34s} tetiklenen+cozulmus 0  (dolan {t['filled']}, tetiklenen {t['armed']})")
        return
    d = (t["flat"] - t["cur"]) / n
    print(f"  {name:34s} n={n:4d}  duz TP %{100 * t['flat_win'] / n:4.0f} (basabas %{100 * t['pstar'] / n:3.0f})"
          f"  bugun {t['cur'] / n:+.3f}  duz {t['flat'] / n:+.3f}  fark {d:+.3f} R/tetik")
    print(f"  {'':34s} kalan yari: TP {t['tp']} / BE {t['be']}  ->  BE'dekiler sonra: TP {t['be_win']} / SL {t['be_loss']}"
          f"   (cozulmemis {t['open']}, ayni mum {t['amb']})")


def live_block(db: str) -> None:
    """Canli sinyaller: kismi kar almis islemi Journal'in duz sonucuyla esle (dogrulama, karar degil)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, symbol, timeframe, direction, exit_reason, planned_rr, rr_value, purge_time FROM signals "
        "WHERE status = 'expired' AND partial_size IS NOT NULL AND rr_value IS NOT NULL ORDER BY closed_at").fetchall()
    n = act = flat = 0.0
    kind = {"tp": 0, "be_win": 0, "be_loss": 0, "be_unk": 0, "diger": 0}
    for s in rows:
        j = con.execute("SELECT shadow FROM setup_journal WHERE strategy=? AND symbol=? AND direction=? "
                        "AND purge_time=?", (s["timeframe"], s["symbol"], s["direction"], s["purge_time"])).fetchone()
        k1 = ((json.loads(j["shadow"]) or {}).get("1") or {}) if j and j["shadow"] else {}
        ex = (s["exit_reason"] or "").lower()
        rr = float(s["planned_rr"] or 0)
        if ex == "tp":
            kind["tp"] += 1
            f = rr
        elif ex == "be" and k1.get("o") in ("win", "loss"):
            kind["be_win" if k1["o"] == "win" else "be_loss"] += 1
            f = flat_r(rr, k1["o"])
        else:
            kind["be_unk" if ex == "be" else "diger"] += 1
            continue
        n += 1
        act += float(s["rr_value"])
        flat += f
    con.close()
    print(f"  kismi kar almis kapali islem {len(rows)}; eslesen {int(n)}")
    print(f"  kalan yari TP {kind['tp']} · BE {kind['be_win'] + kind['be_loss'] + kind['be_unk']}"
          f" (sonra TP {kind['be_win']} / SL {kind['be_loss']} / golge yok {kind['be_unk']}) · diger {kind['diger']}")
    if n:
        print(f"  eslesenlerde: gercek {act:+.2f}R  duz TP/SL {flat:+.2f}R  fark {(flat - act) / n:+.3f} R/islem")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "traderadar.db"))
    ap.add_argument("--strategy", default="", help="4h | 1d (bos = ikisi)")
    ap.add_argument("--market", default="", help="crypto | fx")
    args = ap.parse_args()

    rows = load(args.db, args.strategy, args.market)
    setup = [r for r in rows if not r["pre"]]

    head("Kismi kar + BE vs duz TP/SL  (Setup Journal, k=1 golgesi)")
    print("  fark = duz - bugunku; POZITIF fark = kismi kar + BE'yi kapatmak kazandirirdi")
    main_t = tally(setup)
    show("ANA: motorun setup'lari", main_t)
    show("  C2 kapaliyken donmus", tally([r for r in setup if r["c2"]]))
    show("  RR >= 2", tally([r for r in setup if (r["rr"] or 0) >= 2]))
    show("  skor >= 7", tally([r for r in setup if (r["score"] or 0) >= 7]))
    show("  sinyale donusen", tally([r for r in setup if r["signal"]]))
    for st in ("4h", "1d"):
        show(f"  {st}", tally([r for r in setup if r["strategy"] == st]))
    show("motorun CRT saymadigi adaylar", tally([r for r in rows if r["pre"]]))

    head("Canli sinyaller (dogrulama)")
    live_block(args.db)

    head("Karar")
    n = main_t["n"]
    if n < MIN_ARMED:
        print(f"  ANA dilim {n}/{MIN_ARMED} -- KARAR VERME. 03.10'da hala eksikse kullaniciya sor.")
        return
    d = (main_t["flat"] - main_t["cur"]) / n
    if d > BAND:
        print(f"  H1: fark {d:+.3f} > +{BAND} -> kismi kar + BE'yi KAPATMA ADAYI. RR >= 2 ve C2 kapali")
        print("  dilimleri ayni isarette mi bak; degisiklikten once tek dogrulama replay'i (CLAUDE.md kural 4).")
    elif d < -BAND:
        print(f"  H2: fark {d:+.3f} < -{BAND} -> kismi kar + BE KALIR (getiri gerekcesiyle).")
    else:
        print(f"  H3: |fark| {abs(d):.3f} <= {BAND} -> getiri farki yok, mevcut hal KALIR (daha yuksek WR).")
    print("  Sonucu IZLEME.md 'Kısmi kâr + BE kapatılsın mı?' maddesine + watchlist registry'ye yaz.")


if __name__ == "__main__":
    main()
