"""TradeRadar log raporu (terminal). Web karsiligi: /logs?view=report

Kullanim:
    python scripts/logstat.py                 # tum log
    python scripts/logstat.py --hours 24      # son 24 saat
    python scripts/logstat.py --file X.log    # donen yedek (traderadar.log.1)

Toplama mantigi app/log_report.py icinde; burasi yalnizca yazdirir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.log_report import DEFAULT_LOG, build_report  # noqa: E402


def head(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def bar(count: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    return "#" * max(1, round(count / total * width))


def main() -> int:
    ap = argparse.ArgumentParser(description="TradeRadar log raporu")
    ap.add_argument("--file", default=str(DEFAULT_LOG), help="log dosyasi")
    ap.add_argument("--hours", type=float, default=None, help="son N saati raporla")
    args = ap.parse_args()

    r = build_report(args.file, args.hours)
    if not r["exists"]:
        print(f"Log bulunamadi: {r['file']}", file=sys.stderr)
        return 1
    if not r["lines"]:
        print("Bu pencerede kayit yok.")
        return 0

    span = (f"{r['start']:%d.%m %H:%M} - {r['end']:%d.%m %H:%M}"
            if r["start"] and r["end"] else "-")
    head("OZET")
    print(f"  Dosya      : {r['file']}  ({r['size_kb']} KB)")
    print(f"  Kayit      : {r['lines']} satir")
    print(f"  Aralik     : {span}" + (f"   (son {r['hours']} saat)" if r["hours"] else ""))
    print(f"  Oturum     : {r['sessions']} baslatma")

    if r["skipped"]:
        head(f"SETUP NEDEN SINYALE DONUSMEDI  ({r['skipped_total']} eleme)")
        for s in r["skipped"]:
            top = ", ".join(s["top"][:3])
            print(f"  {s['count']:>5}  {s['pct']:>5.1f}%  "
                  f"{bar(s['count'], r['skipped_total']):<28} {s['reason']:<16} {top}")

    lc = r["lifecycle"]
    if lc:
        head("SINYAL YASAM DONGUSU")
        kinds = "  ".join(f"{k}={v}" for k, v in lc["created"].items())
        print(f"  Olusturulan : {lc['created_total']}  {kinds}")
        if lc["rr_avg"] is not None:
            print(f"  Plan RR     : ort {lc['rr_avg']}  min {lc['rr_min']}  max {lc['rr_max']}")
        print(f"  Dolan       : {lc['filled']}"
              + (f"   ({', '.join(lc['filled_symbols'])})" if lc["filled_symbols"] else ""))
        print(f"  SMT bonusu  : {lc['smt']}"
              + (f"   ({', '.join(lc['smt_symbols'])})" if lc["smt_symbols"] else ""))
        print(f"  BE acilan   : {len(lc['be_arm'])}"
              + (f"   ({', '.join(lc['be_arm'])})" if lc["be_arm"] else ""))
        print(f"  Trail acilan: {len(lc['trail_arm'])}"
              + (f"   ({', '.join(lc['trail_arm'])})" if lc["trail_arm"] else ""))

    if r["closes"]:
        cs = r["close_stats"]
        head(f"KAPANAN ISLEMLER  ({len(r['closes'])})")
        print(f"  {'zaman':<12} {'sembol':<14} {'yon':<6} {'nasil':<10} {'R':>7}")
        for c in r["closes"]:
            when = f"{c['ts']:%d.%m %H:%M}" if c["ts"] else "-"
            print(f"  {when:<12} {c['symbol']:<14} {c['direction']:<6} "
                  f"{c['how']:<10} {c['rr']:>+7.2f}")
        print(f"\n  Toplam R    : {cs['total_r']:+.2f}      Ortalama R: {cs['avg_r']:+.2f}")
        print(f"  Win/Loss/BE : {cs['wins']}/{cs['losses']}/{cs['be']}"
              f"      Win rate: {cs['win_rate']}%")
        print("  Cikis turu  : " + "  ".join(f"{k}={v}" for k, v in cs["by_how"].items()))

    mon = r["monitoring"]
    if mon:
        head("IZLEME METRIKLERI (TODO: Izleme)")
        bf = mon["backfills"]
        print(f"  BACKFILL FILL tetiklenme : {len(bf)}"
              + ("   -> hic tetiklemediyse TODO 'IFVG fallback' gereksiz" if not bf else ""))
        for b in bf:
            when = f"{b['ts']:%d.%m %H:%M}" if b["ts"] else "-"
            print(f"      {when}  {b['symbol']} {b['direction']}")
        p = mon["protection"]
        if r["closes"]:
            print(f"  Koruma etkisi            : TP={p['tp']}  trail={p['trail']}  "
                  f"BE={p['be']}  SL={p['sl']}")
            if mon["trail_avg_r"] is not None:
                print(f"      trail cikislarinin ort. R = {mon['trail_avg_r']:+.2f}"
                      "   (dusukse trail cok erken/dar demektir)")
        c60 = mon["crt60"]
        if c60["total"]:
            print(f"  CRT %60 elemeleri        : {c60['total']}  "
                  f"(fill once={c60['fill_first']}, hic fill yok={c60['no_fill']})")

    b = r.get("bias") or {}
    if b.get("blocks"):
        head(f"1D BIAS TAKIBI  ({b['blocks']} eleme)")
        tops = "  ".join(f"{t['symbol']}={t['count']}" for t in b["top_symbols"])
        print(f"  En cok bloklanan  : {tops}")
        if b["with_parts"]:
            pct = b["disagree"] / b["with_parts"] * 100
            print(f"  structure vs ict  : {b['disagree']}/{b['with_parts']} ayrisiyor"
                  f" ({pct:.0f}%)   -> ayrisinca daily NEUTRAL, setup +2 puan kaybeder")
        if b["weekly_contradicts_daily"]:
            print(f"  weekly != daily   : {b['weekly_contradicts_daily']}"
                  "   (weekly filtre DEGIL, yalnizca skor/bilgi)")
        if b["samples"]:
            print("  Son ornekler:")
            for s in b["samples"]:
                parts = f"structure={s['structure']} ict={s['ict']}" if s["structure"] else ""
                print(f"      {s['symbol']:<13} {s['direction']:<5} "
                      f"daily={s['daily']:<8} {parts}")

    h = r["health"]
    if h:
        head("BAGLANTI / SAGLIK")
        print(f"  WS baglanti : {h['ws_up']} baglandi / {h['ws_down']} koptu")
        print(f"  Tarama      : {h['scans']} tam tarama")
        print(f"  Uyari/Hata  : {h['warnings']} warning, {h['errors']} error")
        for i in h["recent_issues"]:
            when = f"{i['ts']:%d.%m %H:%M}" if i["ts"] else "-"
            print(f"      {when}  {i['msg'][:90]}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
