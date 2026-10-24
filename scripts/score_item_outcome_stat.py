"""Skor kalemi basina sonuc: bu puani alan setuplarin kaci TP, kaci SL? -- strateji bazinda (01.10, kullanici).

Soru (kullanici): "-4 cezasi dogru calisiyor mu, bu zamana kadar bu cezayi alanlarin kac tanesi stop olmus kac
tanesi TP olmus -- her kalem icin." Her kalem, aldigi her DEGERE gore ayri satir (ör. reclaim -4 / 0, htf +2 / 0 / -2):
setup, cozulmus, TP, SL, giristen once TP, acik, WR, R/setup.

Kaynak `setup_journal` (canli sinyal degil): motorun CRT saydigi her setup, kapidan gecsin gecmesin -- ceza alan setup
skor kapisinda elenir ama Journal sonrasini izler, soru tam olarak "elenmeseydi ne olurdu". Kalem degeri
`parts_at_levels` (seviyeler dondugu andaki kirilim), sonuc `outcome` (secilen giris, DUZ TP/SL: kismi kar + BE yok);
sinyale donustuyse `shadow["1"]`. 4H'te yalniz C2 kapaliyken donmus satirlar (aciktayken donan SL kosan purge ucu).

HUKUM (kalem basina, "puan alan" = deger != 0, "almayan" = 0; htf'de +2 ve -2 ayri ayri 0'a karsi).
HUKUM yalniz KARAR BANDINDAN (01.10): kalemin sinyali degistirdigi setuplar -- bu kalem haric skor oyle ki odul onu
esigin ustune tasir, ceza altina iter (4H/1D esik 7, 1W 6); ayni bantta puan alan vs almayan, taraf basina >= 10 cozulmus.
Eski "tum bantlar" ortalamasi bilgi olarak basilir; ilk denemede 4H -4 cezasinin bant 5-6 (+0.51) ve 7+ (-0.44) farklari
birbirini sifirliyordu, oysa karar yalniz 7+ bandini etkiler. Ham ve tum-bant fark: kalemler birbirine bagli (ceza alan setup zaten dusuk skorlu, SMT'li zaten yuksek), ham fark
yaniltir -- 01.10'da 4H -4 cezasi hamda "fark yok", ayni bantta belirgin kotu cikti. Bant = bu kalem HARIC skor
(raw - deger): 0-4 / 5-6 / 7+; fark her bantta (puan alan - almayan), puan alanin cozulmus sayisiyla agirlikli. Ham fark bilgi.
  odul (+)  : puan alanin R/setup'i almayandan >= 0.15 iyi -> "dogru calisiyor"; |fark| < 0.15 -> "fark yok";
              <= -0.15 -> "TERS" (puan verdigimiz setup daha kotu)
  ceza (-)  : ceza alanin R/setup'i >= 0.15 kotu -> "dogru calisiyor"; ... ayni bantlar ters yonde
  iki tarafta da >= MIN_RESOLVED cozulmus yoksa "veri az".
-4 cezasinin "ceza olmasa sinyal olurdu" dilimi ayrica `scripts/reclaim_penalty_stat.py`'de.
⚠️ Kural degisiklikleri: key level kalemleri (pd_*) 29.09'da yeniden tanimlandi -> `--since 2026-09-29`;
1D c2_closed/base 25.09 oncesi hep 1 / yanlis renk 1 -> `--since 2026-09-25`; reclaim 19.09 oncesi -2 (ayri satirda).

En altta SMT segment bazinda: altcoin / BTC-ETH / FX (02.10), yalniz SMT alabilecek setuplar.
Kullanim: python scripts/score_item_outcome_stat.py [--strategy 4h|1d|1w] [--since 2026-09-16] [--db traderadar.db]
Karar kurali: IZLEME.md "Skor kalemleri — 4H" / "— 1D" / "— 1W".
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app.crt_engine import SCORE_PART_KEYS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=str(REPO / "traderadar.db"))
ap.add_argument("--strategy", choices=("4h", "1d", "1w"))
ap.add_argument("--since", default="2026-09-16", help="kalem kirilimi 16.09'dan beri var")
ap.add_argument("--all-levels", action="store_true", help="1D/1W: C2 acikken donmus (yanli) eski satirlar da")
A = ap.parse_args()

PRE = {"sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "c2_second_sweep", "not_selected"}
KEYS = [k for k in SCORE_PART_KEYS if k not in ("raw", "score")]
LABELS = {
    "base": "C2 rengi (dogru +2 / doji +1 / yanlis 0)",
    "htf": "Bias (1D; 1W'de aylik) +2 / ters -2",
    "weekly": "1W uyumu +1 (1W stratejisinde 1D bias uyumu, 24.10)",
    "c2_closed": "C2 kapali +1",
    "pd_major": "Key level major +1",
    "pd_monthly": "Key level aylik +1 (1W stratejisinde guclu C2 donusu, 24.10)",
    "pd_struct": "Key level FVG/OB +1",
    "wick": "Purge fitili +1",
    "ifvg": "LTF IFVG +1",
    "reclaim": "Zayif C2 geri donusu (ceza)",
    "smt": "SMT +2",
}
# 04.10: hesap `app/score_outcome.py`'de -- /skor-tablosu ayni fonksiyonlari kullanir.
from app.score_outcome import (  # noqa: E402
    MIN_FLIP, THR, collect, controlled_diff, flip_diff, stats, verdict,
)


def fmt(v, f):
    return format(v, f) if v is not None else "-"


SEGMENTS = (("alt", "Kripto altcoin (korele: BTC)"), ("major", "BTC / ETH"), ("fx", "FX / metal / endeks / petrol"))

con = sqlite3.connect(f"file:{A.db}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = collect(con.execute("select * from setup_journal where parts_at_levels is not null and entry is not null "
                           "and first_seen >= ?", (A.since,)),
               strategy=A.strategy, all_levels=A.all_levels)

print(f"since {A.since} · sonuc = secilen giris, duz TP/SL · R/setup cozulmus (TP/SL) basina · "
      f"'TP once' = giristen once TP (islem olmazdi)")
HDR = (f"   {'deger':>6}{'setup':>7}{'cozulmus':>9}{'TP':>5}{'SL':>5}{'TP once':>8}{'acik':>6}"
       f"{'WR%':>7}{'R/setup':>9}")
for strat in ("4h", "1d", "1w"):
    sub = [x for x in rows if x["s"] == strat]
    if not sub:
        continue
    print(f"\n{'=' * 78}\n{strat.upper()} -- {len(sub)} setup ({stats(sub)['res']} cozulmus)\n{'=' * 78}")
    for k in KEYS:
        vals = sorted({x["pal"].get(k) for x in sub if x["pal"].get(k) is not None}, reverse=True)
        if not vals:
            continue
        print(f"\n  {k} -- {LABELS.get(k, k)}")
        print(HDR)
        by = {v: stats([x for x in sub if x["pal"].get(k) == v]) for v in vals}
        for v in vals:
            st = by[v]
            print(f"   {v:>+6}{st['n']:>7}{st['res']:>9}{st['w']:>5}{st['l']:>5}{st['tpb']:>8}{st['op']:>6}"
                  f"{fmt(st['wr'], '.1f'):>7}{fmt(st['rps'], '+.3f'):>9}")
        if len(vals) == 1:
            print(f"   HUKUM: etkisi yok -- butun setuplar ayni degeri aliyor ({vals[0]:+})")
            continue
        zero = by.get(0)
        if zero is None:
            print("   HUKUM: 0 degeri yok, karsilastirma yapilamadi")
            continue
        for v in vals:
            if v == 0:
                continue
            d = controlled_diff(sub, k, v)
            raw_d = (by[v]["rps"] - zero["rps"]) if by[v]["res"] and zero["res"] else None
            fg, fb, fd, (lo, hi) = flip_diff(sub, k, v, THR[strat])
            print(f"   HUKUM {v:+} vs 0: {verdict(fg, fb, fd, v < 0)}")
            print(f"      karar bandi (bu kalem haric skor {lo}-{hi}): puan alan {fg['res']} cozulmus "
                  f"{fg['w']} TP / {fg['l']} SL, R/setup {fmt(fg['rps'], '+.2f')} · almayan {fb['res']} cozulmus "
                  f"{fb['w']} TP / {fb['l']} SL, R/setup {fmt(fb['rps'], '+.2f')}")
            print(f"      bilgi: ham fark {fmt(raw_d, '+.2f')} · tum bantlar kontrollu {fmt(d, '+.2f')}")


# SMT -- segment bazinda (02.10, kullanici: "altcoinlerde hatali calisiyor olabilir, FX'te dogru"). Kontrol grubu
# yalniz SMT ALABILECEK setuplar (`smt_possible`; korele paritesi olmayan sembol 0 alir, kiyasi bozar). Hukum ayni
# karar bandi kuraliyla (taraf basina >= MIN_FLIP cozulmus), segment kucuk oldugu icin ham fark da basilir.
print(f"\n{'=' * 78}\nSMT +2 -- SEGMENT BAZINDA (yalniz smt_possible=1 setuplar)\n{'=' * 78}")
for strat in ("4h", "1d", "1w"):
    for seg, label in SEGMENTS:
        sub = [x for x in rows if x["s"] == strat and x["seg"] == seg and x["smt_ok"] != 0]
        if not sub:
            continue
        # 24.10: 4H altcoin SMT bonusu +1 (once +2). Eski satirlar bugunku bonusa normalize edilir (SMT haric raw ayni).
        v = 1 if (strat == "4h" and seg == "alt") else 2
        sub = [dict(x, pal=dict(x["pal"], smt=v, raw=(x["pal"].get("raw") or 0) - x["pal"]["smt"] + v))
               if (x["pal"].get("smt") or 0) > 0 else x for x in sub]
        g_all, b_all = stats([x for x in sub if x["pal"].get("smt") == v]), stats([x for x in sub if x["pal"].get("smt") == 0])
        fg, fb, fd, (lo, hi) = flip_diff(sub, "smt", v, THR[strat])
        raw_d = (g_all["rps"] - b_all["rps"]) if g_all["res"] and b_all["res"] else None
        print(f"\n  {strat.upper()} · {label} -- {len(sub)} setup")
        print(f"   SMT'li    {g_all['n']:>5} setup, {g_all['res']:>4} cozulmus {g_all['w']:>3} TP / {g_all['l']:>3} SL, "
              f"WR {fmt(g_all['wr'], '.1f')}, R/setup {fmt(g_all['rps'], '+.3f')}")
        print(f"   SMT'siz   {b_all['n']:>5} setup, {b_all['res']:>4} cozulmus {b_all['w']:>3} TP / {b_all['l']:>3} SL, "
              f"WR {fmt(b_all['wr'], '.1f')}, R/setup {fmt(b_all['rps'], '+.3f')}")
        print(f"   HUKUM: {verdict(fg, fb, fd, False)}")
        print(f"      karar bandi (SMT haric skor {lo}-{hi}): SMT'li {fg['res']} cozulmus {fg['w']} TP / {fg['l']} SL, "
              f"R/setup {fmt(fg['rps'], '+.2f')} · SMT'siz {fb['res']} cozulmus {fb['w']} TP / {fb['l']} SL, "
              f"R/setup {fmt(fb['rps'], '+.2f')} · ham fark {fmt(raw_d, '+.2f')}")
