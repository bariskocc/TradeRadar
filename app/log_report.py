"""Motor logundan (logs/traderadar.log) ozet rapor uretir.

Hem `/logs?view=report` sayfasi hem `scripts/logstat.py` bu modulu kullanir;
toplama mantigi tek yerde durur. Dosyayi yalnizca OKUR — DB'ye/sunucuya
dokunmaz, sunucu kapaliyken de calisir.

IZLEME.md'deki sorular icin: setup'lar hangi kapida eleniyor,
BACKFILL hic tetikliyor mu, koruma (BE/trail) islemleri erken mi boguyor.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import BASE_DIR

TSI = timezone(timedelta(hours=3))
DEFAULT_LOG = BASE_DIR / "logs" / "traderadar.log"

# "09.09.26 00:12:14 INFO    app.scanner | mesaj"  (eski satirlarda yil yok)
LINE_RE = re.compile(
    r"^(?P<ts>\d{2}\.\d{2}(?:\.\d{2,4})? \d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>\w+)\s+(?P<logger>[\w.]+)\s*\|\s*(?P<msg>.*)$"
)
SKIPPED_RE = re.compile(r"^SKIPPED \((?P<reason>[^)]+)\):\s*(?P<sym>\S+)?")
NEW_RE = re.compile(
    r"^NEW (?P<kind>WAITING|PENDING|ACTIVE[^:]*):\s*(?P<sym>\S+) "
    r"(?P<dir>LONG|SHORT).*?RR:(?P<rr>[-\d.]+)"
)
PROMOTE_RE = re.compile(r"^PROMOTE WAITING:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT).*?RR:(?P<rr>[-\d.]+)")
CLOSED_RE = re.compile(
    r"^CLOSED \((?P<how>[A-Z_]+)\):\s*(?P<sym>\S+) (?P<dir>LONG|SHORT) RR (?P<rr>[-\d.]+)"
)
FILLED_RE = re.compile(r"^FILLED:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
BACKFILL_RE = re.compile(r"^BACKFILL FILL:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
BE_RE = re.compile(r"^BE ARM:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
TRAIL_ARM_RE = re.compile(r"^TRAIL ARM:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
SMT_RE = re.compile(r"^SMT\+\d+:\s*(?P<sym>\S+) (?P<dir>LONG|SHORT)")
CRT60_RE = re.compile(r"breach=(?P<breach>\S+ \S+) fill=(?P<fill>\S+?)\)?\.?$")
# "SKIPPED (BIAS): XAUUSD SHORT filter=B daily=B structure=B ict=B weekly=B"
BIAS_RE = re.compile(
    r"^SKIPPED \(BIAS\):\s*(?P<sym>\S+) (?P<dir>LONG|SHORT) "
    r"filter=(?P<filter>\w+) daily=(?P<daily>\w+)"
    r"(?: structure=(?P<structure>\w+))?(?: ict=(?P<ict>\w+))?"
    r"(?: weekly=(?P<weekly>\w+))?"
)


def _parse_ts(raw: str, year_hint: int) -> datetime | None:
    """Zaman damgasini coz. Yilsiz eski format (%d.%m) da desteklenir."""
    for fmt, needs_year in (
        ("%d.%m.%Y %H:%M:%S", False),
        ("%d.%m.%y %H:%M:%S", False),
        ("%d.%m %H:%M:%S", True),
    ):
        try:
            with warnings.catch_warnings():
                # Yilsiz satirlar: 3.15 uyarisi anlamsiz, yili biz tamamliyoruz.
                warnings.simplefilter("ignore", DeprecationWarning)
                dt = datetime.strptime(raw, fmt)
            if needs_year:
                dt = dt.replace(year=year_hint)
            return dt.replace(tzinfo=TSI)
        except ValueError:
            continue
    return None


def build_report(path: str | Path | None = None, hours: float | None = None) -> dict:
    """Log dosyasini ayristirip ozet sozlugu dondur. Dosya yoksa `exists: False`."""
    p = Path(path) if path else DEFAULT_LOG
    if not p.exists():
        return {"exists": False, "file": str(p), "hours": hours}

    year_hint = datetime.now().year
    cutoff = (datetime.now(TSI) - timedelta(hours=hours)) if hours else None

    rows: list[tuple[datetime | None, str, str, str]] = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        ts = _parse_ts(m["ts"], year_hint)
        if cutoff and ts and ts < cutoff:
            continue
        rows.append((ts, m["level"], m["logger"], m["msg"]))

    stamps = [t for t, *_ in rows if t]
    report: dict = {
        "exists": True,
        "file": str(p),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "lines": len(rows),
        "hours": hours,
        "start": stamps[0] if stamps else None,
        "end": stamps[-1] if stamps else None,
        "sessions": sum(1 for _, _, lg, _ in rows if lg.endswith("logging_config")),
    }
    if not rows:
        report.update(skipped=[], skipped_total=0, lifecycle={}, closes=[],
                      close_stats={}, monitoring={}, bias={}, health={})
        return report

    # ── Setup neden sinyale donusmedi ───────────────────────────────
    skipped: Counter = Counter()
    skipped_syms: dict[str, Counter] = defaultdict(Counter)
    for _, _, _, msg in rows:
        if m := SKIPPED_RE.match(msg):
            skipped[m["reason"]] += 1
            if m["sym"]:
                skipped_syms[m["reason"]][m["sym"]] += 1
    sk_total = sum(skipped.values())
    report["skipped_total"] = sk_total
    report["skipped"] = [
        {
            "reason": reason,
            "count": n,
            "pct": round(n / sk_total * 100, 1) if sk_total else 0.0,
            "top": [s for s, _ in skipped_syms[reason].most_common(4)],
        }
        for reason, n in skipped.most_common()
    ]

    # ── Yasam dongusu ───────────────────────────────────────────────
    created: Counter = Counter()
    created_rr: list[float] = []
    filled: Counter = Counter()
    backfills: list[dict] = []
    be_arm: set[str] = set()
    trail_arm: set[str] = set()
    smt: Counter = Counter()
    closes: list[dict] = []

    for ts, _, _, msg in rows:
        if m := NEW_RE.match(msg):
            created[m["kind"].split()[0].lower()] += 1
            try:
                created_rr.append(float(m["rr"]))
            except ValueError:
                pass
        elif m := PROMOTE_RE.match(msg):
            created["promote"] += 1
            try:
                created_rr.append(float(m["rr"]))
            except ValueError:
                pass
        elif m := FILLED_RE.match(msg):
            filled[m["sym"]] += 1
        elif m := BACKFILL_RE.match(msg):
            backfills.append({"ts": ts, "symbol": m["sym"], "direction": m["dir"]})
        elif m := BE_RE.match(msg):
            be_arm.add(m["sym"])
        elif m := TRAIL_ARM_RE.match(msg):
            trail_arm.add(m["sym"])
        elif m := SMT_RE.match(msg):
            smt[m["sym"]] += 1
        elif m := CLOSED_RE.match(msg):
            closes.append({
                "ts": ts, "symbol": m["sym"], "direction": m["dir"],
                "how": m["how"].lower(), "rr": float(m["rr"]),
            })

    report["lifecycle"] = {
        "created": dict(created),
        "created_total": sum(created.values()),
        "rr_avg": round(sum(created_rr) / len(created_rr), 2) if created_rr else None,
        "rr_min": round(min(created_rr), 2) if created_rr else None,
        "rr_max": round(max(created_rr), 2) if created_rr else None,
        "filled": sum(filled.values()),
        "filled_symbols": list(filled),
        "smt": sum(smt.values()),
        "smt_symbols": [s for s, _ in smt.most_common(6)],
        "be_arm": sorted(be_arm),
        "trail_arm": sorted(trail_arm),
    }

    # ── Kapanan islemler ────────────────────────────────────────────
    report["closes"] = closes
    if closes:
        rs = [c["rr"] for c in closes]
        wins = [r for r in rs if r > 0.05]
        losses = [r for r in rs if r < -0.05]
        be = [r for r in rs if -0.05 <= r <= 0.05]
        report["close_stats"] = {
            "total_r": round(sum(rs), 2),
            "avg_r": round(sum(rs) / len(rs), 2),
            "wins": len(wins), "losses": len(losses), "be": len(be),
            "win_rate": round(len(wins) / len(rs) * 100),
            "by_how": dict(Counter(c["how"] for c in closes)),
        }
    else:
        report["close_stats"] = {}

    # ── Izleme metrikleri (IZLEME.md) ────────────────────────────
    by_how = Counter(c["how"] for c in closes)
    trail_exits = [c["rr"] for c in closes if c["how"] == "hit_trail"]
    crt_rows = [m for _, _, _, m in rows if m.startswith("SKIPPED (CRT 60%)")]
    fill_first = no_fill = 0
    for msg in crt_rows:
        if m := CRT60_RE.search(msg):
            if m["fill"] == "None":
                no_fill += 1
            elif m["fill"] < m["breach"]:
                fill_first += 1
    report["monitoring"] = {
        "backfills": backfills,
        "protection": {
            "tp": by_how.get("hit_tp", 0),
            "trail": by_how.get("hit_trail", 0),
            "be": by_how.get("hit_be", 0),
            "sl": by_how.get("hit_sl", 0),
        },
        "trail_avg_r": round(sum(trail_exits) / len(trail_exits), 2) if trail_exits else None,
        "crt60": {"total": len(crt_rows), "fill_first": fill_first, "no_fill": no_fill},
    }

    # ── 1D bias takibi ──────────────────────────────────────────────
    # daily = structure VE ict; ayrisirlarsa NEUTRAL olur ve setup +2 kalite
    # puanini kaybeder. Bayat structure (haftalar once kirilmis) taze ict'yi
    # vetolayabiliyor -- bu bolum onu gorunur kilar.
    bias_blocks: list[dict] = []
    for _, _, _, msg in rows:
        if m := BIAS_RE.match(msg):
            bias_blocks.append({
                "symbol": m["sym"], "direction": m["dir"],
                "daily": m["daily"], "structure": m["structure"],
                "ict": m["ict"], "weekly": m["weekly"],
            })
    bias_syms = Counter(b["symbol"] for b in bias_blocks)
    # structure/ict ayrisan kayitlar (yeni format; eski satirlarda ict yok)
    with_parts = [b for b in bias_blocks if b["structure"] and b["ict"]]
    disagree = [b for b in with_parts if b["structure"] != b["ict"]]
    wk_vs_daily = [
        b for b in bias_blocks
        if b["weekly"] and b["daily"] not in (None, "NEUTRAL")
        and b["weekly"] not in (None, "NEUTRAL") and b["weekly"] != b["daily"]
    ]
    report["bias"] = {
        "blocks": len(bias_blocks),
        "top_symbols": [{"symbol": s, "count": n} for s, n in bias_syms.most_common(6)],
        "with_parts": len(with_parts),
        "disagree": len(disagree),
        "weekly_contradicts_daily": len(wk_vs_daily),
        "samples": bias_blocks[-6:],
    }

    # ── Baglanti / saglik ───────────────────────────────────────────
    issues = [
        {"ts": ts, "level": lv, "msg": m}
        for ts, lv, _, m in rows if lv in ("WARNING", "ERROR", "CRITICAL")
    ]
    report["health"] = {
        "ws_up": sum(1 for _, _, _, m in rows if "WS baglandi" in m),
        "ws_down": sum(1 for _, _, _, m in rows if "WS koptu" in m),
        "scans": sum(1 for _, _, _, m in rows if m.startswith("Scan complete")),
        "warnings": sum(1 for i in issues if i["level"] == "WARNING"),
        "errors": sum(1 for i in issues if i["level"] in ("ERROR", "CRITICAL")),
        "recent_issues": issues[-6:],
    }
    return report
