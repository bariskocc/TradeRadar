"""Calendar sayfasi (24.10, kullanici istegi): motor sinyallerinin aylik R takvimi -- Paper Trades takviminin
(`paper_trades.calendar_month`) aynisi, dolar yerine R.

Kurallar Dashboard'la ayni: islem KAPANDIGI gune sayilir (`Signal.closed_at`, TSI), R = `rr_value` (kismi kar dahil,
agirlikli), W / L / BE `main._trade_outcome` kuraliyla (kismi karli BE kapanisi BE sayilir). Aktif (dolmus, kapanmamis)
sinyaller dolum gununde sari isaret, R'ye girmez. Saf hesap; DB'yi cagiran okur.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

TSI = timezone(timedelta(hours=3))
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_EPS = 1e-9


def _day(dt: datetime | None) -> date | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TSI).date()


def build_month(closed, active, first_day: date, last_day: date, outcome_fn) -> dict:
    """closed: kapanmis sinyaller (closed_at + rr_value), active: dolmus acik sinyaller (entry_filled_time).
    outcome_fn(signal) -> 'win' | 'loss' | 'breakeven' | None (main._trade_outcome)."""
    by_day: dict[date, dict] = {}
    for s in closed:
        d = _day(s.closed_at)
        if d is None:
            continue
        cell = by_day.setdefault(d, {"n": 0, "wins": 0, "losses": 0, "be": 0, "r": 0.0})
        cell["n"] += 1
        cell["r"] = round(cell["r"] + float(s.rr_value or 0.0), 2)
        o = outcome_fn(s)
        if o == "win":
            cell["wins"] += 1
        elif o == "loss":
            cell["losses"] += 1
        elif o == "breakeven":
            cell["be"] += 1
    max_abs = max((abs(c["r"]) for c in by_day.values()), default=0.0)

    open_by_day: dict[date, int] = {}
    for s in active or ():
        d = _day(getattr(s, "entry_filled_time", None))
        if d is not None:
            open_by_day[d] = open_by_day.get(d, 0) + 1

    weeks = []
    day = first_day - timedelta(days=first_day.weekday())
    grid_end = last_day + timedelta(days=6 - last_day.weekday())
    while day <= grid_end:
        cells = []
        for _ in range(7):
            c = by_day.get(day)
            v = c["r"] if c else None
            tone = "flat"
            if v is not None and abs(v) > _EPS:
                tone = "win" if v > 0 else "loss"
            cells.append({
                "day": day.day, "iso": day.isoformat(), "in_month": first_day <= day <= last_day,
                "n": c["n"] if c else 0, "r": v,
                "wins": c["wins"] if c else 0, "losses": c["losses"] if c else 0, "be": c["be"] if c else 0,
                "open_n": open_by_day.get(day, 0), "tone": tone,
                # 0.10-0.45: en kucuk gun de secilsin, en buyugu ekrani yakmasin (Paper Trades ile ayni).
                "alpha": round(0.10 + 0.35 * (abs(v) / max_abs), 2) if v is not None and max_abs else 0.0,
            })
            day += timedelta(days=1)
        wm = [c for c in cells if c["in_month"] and c["n"]]
        weeks.append({
            "cells": cells,
            "r": round(sum(c["r"] for c in wm), 2) if wm else None,
            "n": sum(c["n"] for c in wm),
            "wins": sum(c["wins"] for c in wm), "losses": sum(c["losses"] for c in wm), "be": sum(c["be"] for c in wm),
            "open_n": sum(c["open_n"] for c in cells if c["in_month"]),
        })

    days = [(d, c) for d, c in by_day.items() if first_day <= d <= last_day]
    total = round(sum(c["r"] for _, c in days), 2) if days else None
    wins = sum(c["wins"] for _, c in days)
    losses = sum(c["losses"] for _, c in days)
    best = max(days, key=lambda x: x[1]["r"]) if days else None
    worst = min(days, key=lambda x: x[1]["r"]) if days else None
    return {
        "weeks": weeks, "weekdays": WEEKDAYS,
        "trading_days": len(days),
        "n": sum(c["n"] for _, c in days),
        "wins": wins, "losses": losses, "be": sum(c["be"] for _, c in days),
        # Win rate = W / (W + L), BE paydaya girmez (Dashboard kurali, 22.10).
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
        "open_n": sum(v for d, v in open_by_day.items() if first_day <= d <= last_day),
        "total_r": total,
        "win_days": sum(1 for _, c in days if c["r"] > _EPS),
        "loss_days": sum(1 for _, c in days if c["r"] < -_EPS),
        "avg_day_r": round(total / len(days), 2) if days else None,
        "best": {"day": best[0].day, "r": best[1]["r"], "iso": best[0].isoformat()} if best else None,
        "worst": {"day": worst[0].day, "r": worst[1]["r"], "iso": worst[0].isoformat()} if worst else None,
    }
