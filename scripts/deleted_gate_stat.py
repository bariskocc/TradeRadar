"""`waiting`e ulasip SONRADAN silinen setuplarin sonucu (19.09).

Soru (kullanici, 19.09): skor kapisi kazanan sinyalleri eliyor mu? `low_quality` ile
BASTAN elenen setuplar Journal'da zaten izleniyordu; olculmeyen tek kose, skoru gecip
`waiting`e ulastiktan SONRA retest'te skoru dusup silinen setuplardi (`missed_quality`).
`note_deleted` artik izlemeyi geri aciyor (setup_journal._resume_tracking), bu rapor da
onu okur: silme anindan itibaren fiyat entry'ye geldi mi, geldiyse TP mi SL mi.

Defter diger kapilarla AYNI: win = entry sonrasi TP (+rr), loss = entry sonrasi SL (-1),
tp_before_entry = emir dolmazdi (kacan islem DEGIL, 0), no_touch = entry'ye gelmedi (0).

UYARI: Journal izlemesi duz TP/SL -- 4H/1D'deki kismi kar ve BE yok. Isaret ve buyukluk
sirasi guvenilir, kesin R degil (golge izlemedeki ayni sinir).

Calistirma: python scripts/deleted_gate_stat.py [--days N] [--strategy 4h] [--reason missed_quality]
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traderadar.db")
RESOLVED = ("win", "loss")
# Karar esigi: dar stop kapisiyla ayni konvansiyon (IZLEME.md).
MIN_RESOLVED = 20


def r_of(outcome, rr):
    if outcome == "win":
        return float(rr or 0)
    if outcome == "loss":
        return -1.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="yalniz son N gun")
    ap.add_argument("--strategy", help="4h / 1d / 1h")
    ap.add_argument("--reason", help="tek bir silme nedeni")
    ap.add_argument("--min-rr", type=float, default=None, help="yalniz RR >= X (alinabilir kesit)")
    a = ap.parse_args()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    q = ["select * from setup_journal where best_stage = 'waiting' and deleted_reason is not null"]
    p = []
    if a.days:
        q.append("and first_seen >= ?")
        p.append((datetime.utcnow() - timedelta(days=a.days)).strftime("%Y-%m-%d %H:%M:%S"))
    if a.strategy:
        q.append("and strategy = ?")
        p.append(a.strategy)
    if a.reason:
        q.append("and deleted_reason = ?")
        p.append(a.reason)
    rows = con.execute(" ".join(q) + " order by deleted_at", p).fetchall()
    if a.min_rr is not None:
        rows = [r for r in rows if (r["rr"] or 0) >= a.min_rr]

    if not rows:
        print("Kayit yok. (Izleme 19.09'da acildi; oncesindeki silinmis satirlarda outcome='signal' kalir.)")
        return

    print(f"{len(rows)} setup `waiting`e ulasip sonradan silinmis.\n")
    per = defaultdict(lambda: defaultdict(int))
    per_r = defaultdict(float)
    for r in rows:
        per[r["deleted_reason"]][r["outcome"] or "?"] += 1
        if r["outcome"] in RESOLVED:
            per_r[r["deleted_reason"]] += r_of(r["outcome"], r["rr"])

    print(f"{'silme nedeni':<18} {'n':>4} {'TP':>4} {'SL':>4} {'win%':>6} {'R':>8} {'R/islem':>8}  izleniyor")
    print("-" * 74)
    tot_n = tot_w = tot_l = 0
    tot_r = 0.0
    for reason in sorted(per, key=lambda k: -sum(per[k].values())):
        d = per[reason]
        w, l = d.get("win", 0), d.get("loss", 0)
        n = w + l
        live = d.get("pending", 0) + d.get("filled", 0)
        rr = per_r[reason]
        tot_n += n
        tot_w += w
        tot_l += l
        tot_r += rr
        wr = f"{100 * w / n:.0f}%" if n else "-"
        rpt = f"{rr / n:+.2f}" if n else "-"
        print(f"{reason:<18} {n:>4} {w:>4} {l:>4} {wr:>6} {rr:>+8.1f} {rpt:>8}  {live}")
    print("-" * 74)
    wr = f"{100 * tot_w / tot_n:.0f}%" if tot_n else "-"
    rpt = f"{tot_r / tot_n:+.2f}" if tot_n else "-"
    print(f"{'TOPLAM':<18} {tot_n:>4} {tot_w:>4} {tot_l:>4} {wr:>6} {tot_r:>+8.1f} {rpt:>8}")

    other = defaultdict(int)
    for r in rows:
        if r["outcome"] not in RESOLVED:
            other[r["outcome"] or "?"] += 1
    if other:
        print("\nCozulmemis / alinamaz:", " · ".join(f"{k}={v}" for k, v in sorted(other.items())))
        print("  (tp_before_entry = fiyat entry'ye ugramadan TP'yi gordu -> limit emri dolmazdi)")

    print("\n--- Karar kurali (IZLEME.md \"Silinen waiting setup\") ---")
    if tot_n < MIN_RESOLVED:
        print(f"  Ornekem yetersiz: {tot_n}/{MIN_RESOLVED} cozulmus. Veri birikiyor, yorum yok.")
        return
    print(f"  R/islem {tot_r / tot_n:+.2f} ({tot_n} cozulmus).")
    if tot_r > 0:
        print("  >>> POZITIF: silinen setuplar toplamda kazandiriyor -- silme kapilari kazanc kesiyor.")
        print("      Once NEDEN bazina bak (ustteki tablo); tek bir neden tasiyorsa yalniz onu gevset.")
        print("      Kural degistirmeden once dogrulama replay'i zorunlu (kismi kar + BE burada yok).")
    else:
        print("  >>> NEGATIF: silme kapilari kayip onluyor, degisiklik gerekmiyor.")


if __name__ == "__main__":
    main()
