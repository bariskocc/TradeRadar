"""TradeRadar log raporu — logs/traderadar.log uzerinden isleyis ozeti.

Kullanim:
    python scripts/logstat.py                 # tum log
    python scripts/logstat.py --hours 24      # son 24 saat
    python scripts/logstat.py --file X.log    # baska dosya (donen yedekler icin)

TODO madde 6'daki izleme sorularini cevaplamak icin yazildi:
  - Uretilen sinyal sayisi/kalitesi
  - SKIPPED dagilimi (setup neden sinyale donusmuyor)
  - BACKFILL FILL hic tetikliyor mu (tetiklemiyorsa TODO madde 1'e gerek yok)
  - BE/trail korumasi islemleri erken mi boguyor
  - WS baglanti sagligi
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOG = BASE_DIR / "logs" / "traderadar.log"
TSI = timezone(timedelta(hours=3))

# "08.09 23:18:09 INFO    app.scanner | mesaj"  (yil opsiyonel: 08.09.26 ...)
LINE_RE = re.compile(
    r"^(?P<ts>\d{2}\.\d{2}(?:\.\d{2,4})? \d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>\w+)\s+(?P<logger>[\w.]+)\s*\|\s*(?P<msg>.*)$"
)

SKIPPED_RE = re.compile(r"^SKIPPED \((?P<reason>[^)]+)\):\s*(?P<sym>\S+)?\s*(?P<dir>LONG|SHORT)?")
NEW_RE = re.compile(r"^NEW (?P<kind>WAITING|PENDING|ACTIVE[^:]*):\s*(?P<sym>\S+) (?P<dir>LONG|SHORT).*?RR:(?P<rr>[-\d.]+)")
PROMOTE_RE = re.compile(r"^PROMOTE WAITING:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT).*?RR:(?P<rr>[-\d.]+)")
CLOSED_RE = re.compile(r"^CLOSED \((?P<how>[A-Z_]+)\):\s*(?P<sym>\S+) (?P<dir>LONG|SHORT) RR (?P<rr>[-\d.]+)")
FILLED_RE = re.compile(r"^FILLED:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
BACKFILL_RE = re.compile(r"^BACKFILL FILL:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
BE_RE = re.compile(r"^BE ARM:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
TRAIL_ARM_RE = re.compile(r"^TRAIL ARM:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
SMT_RE = re.compile(r"^SMT\+\d+:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
CRT60_RE = re.compile(r"breach=(?P<breach>\S+ \S+) fill=(?P<fill>\S+?)\)?\.?$")


def parse_ts(raw: str, year_hint: int) -> datetime | None:
    """Zaman damgasini coz. Yilsiz eski format (%d.%m) da desteklenir."""
    for fmt, needs_year in (("%d.%m.%Y %H:%M:%S", False), ("%d.%m.%y %H:%M:%S", False),
                            ("%d.%m %H:%M:%S", True)):
        try:
            with warnings.catch_warnings():
                # Yilsiz eski satirlar: 3.15 uyarisi bizim icin anlamsiz,
                # yil zaten year_hint ile tamamlaniyor.
                warnings.simplefilter("ignore", DeprecationWarning)
                dt = datetime.strptime(raw, fmt)
            if needs_year:
                dt = dt.replace(year=year_hint)
            return dt.replace(tzinfo=TSI)
        except ValueError:
            continue
    return None


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

    path = Path(args.file)
    if not path.exists():
        print(f"Log bulunamadi: {path}", file=sys.stderr)
        return 1

    year_hint = datetime.now().year
    cutoff = (datetime.now(TSI) - timedelta(hours=args.hours)) if args.hours else None

    rows: list[tuple[datetime | None, str, str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        ts = parse_ts(m["ts"], year_hint)
        if cutoff and ts and ts < cutoff:
            continue
        rows.append((ts, m["level"], m["logger"], m["msg"]))

    if not rows:
        print("Bu pencerede kayit yok.")
        return 0

    stamps = [t for t, *_ in rows if t]
    span = f"{stamps[0]:%d.%m %H:%M} - {stamps[-1]:%d.%m %H:%M}" if stamps else "-"

    head("OZET")
    print(f"  Dosya      : {path}  ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  Kayit      : {len(rows)} satir")
    print(f"  Aralik     : {span}" + (f"   (son {args.hours} saat)" if args.hours else ""))
    print(f"  Oturum     : {sum(1 for _, _, lg, m in rows if lg.endswith('logging_config'))} baslatma")

    # ── Setup neden sinyale donusmedi ───────────────────────────────────
    skipped = Counter()
    skipped_syms: dict[str, Counter] = defaultdict(Counter)
    for _, _, _, msg in rows:
        m = SKIPPED_RE.match(msg)
        if m:
            skipped[m["reason"]] += 1
            if m["sym"]:
                skipped_syms[m["reason"]][m["sym"]] += 1
    if skipped:
        total = sum(skipped.values())
        head(f"SETUP NEDEN SINYALE DONUSMEDI  ({total} eleme)")
        for reason, n in skipped.most_common():
            top = ", ".join(s for s, _ in skipped_syms[reason].most_common(3))
            print(f"  {n:>5}  {n / total * 100:>5.1f}%  {bar(n, total):<28} {reason:<16} {top}")

    # ── Sinyal yasam dongusu ────────────────────────────────────────────
    created = Counter()
    created_rr: list[float] = []
    filled = Counter()
    backfills: list[tuple[datetime | None, str]] = []
    be_arm: set[str] = set()
    trail_arm: set[str] = set()
    smt: Counter = Counter()
    closes: list[tuple[datetime | None, str, str, str, float]] = []

    for ts, _, _, msg in rows:
        if m := NEW_RE.match(msg):
            created[m["kind"].split()[0]] += 1
            try:
                created_rr.append(float(m["rr"]))
            except ValueError:
                pass
        elif m := PROMOTE_RE.match(msg):
            created["PROMOTE"] += 1
            try:
                created_rr.append(float(m["rr"]))
            except ValueError:
                pass
        elif m := FILLED_RE.match(msg):
            filled[m["sym"]] += 1
        elif m := BACKFILL_RE.match(msg):
            backfills.append((ts, f'{m["sym"]} {m["dir"]}'))
        elif m := BE_RE.match(msg):
            be_arm.add(m["sym"])
        elif m := TRAIL_ARM_RE.match(msg):
            trail_arm.add(m["sym"])
        elif m := SMT_RE.match(msg):
            smt[m["sym"]] += 1
        elif m := CLOSED_RE.match(msg):
            closes.append((ts, m["sym"], m["dir"], m["how"], float(m["rr"])))

    head("SINYAL YASAM DONGUSU")
    print(f"  Olusturulan : {sum(created.values())}  " +
          "  ".join(f"{k.lower()}={v}" for k, v in created.most_common()))
    if created_rr:
        print(f"  Plan RR     : ort {sum(created_rr) / len(created_rr):.2f}  "
              f"min {min(created_rr):.2f}  max {max(created_rr):.2f}")
    print(f"  Dolan       : {sum(filled.values())}"
          + (f"   ({', '.join(filled)})" if filled else ""))
    print(f"  SMT bonusu  : {sum(smt.values())}"
          + (f"   ({', '.join(s for s, _ in smt.most_common(5))})" if smt else ""))
    print(f"  BE acilan   : {len(be_arm)}"
          + (f"   ({', '.join(sorted(be_arm))})" if be_arm else ""))
    print(f"  Trail acilan: {len(trail_arm)}"
          + (f"   ({', '.join(sorted(trail_arm))})" if trail_arm else ""))

    # ── Kapanan islemler ────────────────────────────────────────────────
    if closes:
        head(f"KAPANAN ISLEMLER  ({len(closes)})")
        print(f"  {'zaman':<12} {'sembol':<14} {'yon':<6} {'nasil':<10} {'R':>7}")
        for ts, sym, d, how, rr in closes:
            when = f"{ts:%d.%m %H:%M}" if ts else "-"
            print(f"  {when:<12} {sym:<14} {d:<6} {how.lower():<10} {rr:>+7.2f}")
        rs = [rr for *_, rr in closes]
        wins = [r for r in rs if r > 0.05]
        losses = [r for r in rs if r < -0.05]
        be = [r for r in rs if -0.05 <= r <= 0.05]
        print(f"\n  Toplam R    : {sum(rs):+.2f}      Ortalama R: {sum(rs) / len(rs):+.2f}")
        print(f"  Win/Loss/BE : {len(wins)}/{len(losses)}/{len(be)}"
              f"      Win rate: {len(wins) / len(rs) * 100:.0f}%")
        by_how = Counter(how for *_, how, _ in closes)
        print(f"  Cikis turu  : " + "  ".join(f"{k.lower()}={v}" for k, v in by_how.most_common()))

    # ── TODO madde 6 metrikleri ─────────────────────────────────────────
    head("IZLEME METRIKLERI (TODO madde 6)")
    print(f"  BACKFILL FILL tetiklenme : {len(backfills)}"
          + ("   -> hic tetiklemediyse TODO madde 1'e gerek yok" if not backfills else ""))
    for ts, who in backfills:
        print(f"      {ts:%d.%m %H:%M} {who}" if ts else f"      {who}")

    be_exits = [c for c in closes if c[3] == "HIT_BE"]
    trail_exits = [c for c in closes if c[3] == "HIT_TRAIL"]
    tp_exits = [c for c in closes if c[3] == "HIT_TP"]
    sl_exits = [c for c in closes if c[3] == "HIT_SL"]
    if closes:
        print(f"  Koruma etkisi            : TP={len(tp_exits)}  trail={len(trail_exits)}  "
              f"BE={len(be_exits)}  SL={len(sl_exits)}")
        if trail_exits:
            avg = sum(c[4] for c in trail_exits) / len(trail_exits)
            print(f"      trail cikislarinin ort. R = {avg:+.2f}"
                  "   (dusukse trail cok erken/dar demektir)")

    # CRT %60: fill ihlalden once miydi (kronoloji)
    crt_rows = [msg for _, _, _, msg in rows if msg.startswith("SKIPPED (CRT 60%)")]
    if crt_rows:
        fill_first = 0
        no_fill = 0
        for msg in crt_rows:
            m = CRT60_RE.search(msg)
            if not m:
                continue
            if m["fill"] == "None":
                no_fill += 1
            elif m["fill"] < m["breach"]:
                fill_first += 1
        print(f"  CRT %60 elemeleri        : {len(crt_rows)}  "
              f"(fill once={fill_first}, hic fill yok={no_fill})")
        if fill_first:
            print("      fill once olanlar bayat oldugu icin diriltilmedi"
                  " (max_backfill_fill_bars)")

    # ── Baglanti sagligi ────────────────────────────────────────────────
    ws_up = sum(1 for _, _, _, m in rows if "WS baglandi" in m)
    ws_down = sum(1 for _, _, _, m in rows if "WS koptu" in m)
    scans = sum(1 for _, _, _, m in rows if m.startswith("Scan complete"))
    errors = [(ts, m) for ts, lv, _, m in rows if lv in ("ERROR", "CRITICAL")]
    warns = [(ts, m) for ts, lv, _, m in rows if lv == "WARNING"]
    head("BAGLANTI / SAGLIK")
    print(f"  WS baglanti : {ws_up} baglandi / {ws_down} koptu")
    print(f"  Tarama      : {scans} tam tarama")
    print(f"  Uyari/Hata  : {len(warns)} warning, {len(errors)} error")
    for ts, m in (errors + warns)[-5:]:
        when = f"{ts:%d.%m %H:%M}" if ts else "-"
        print(f"      {when}  {m[:90]}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
