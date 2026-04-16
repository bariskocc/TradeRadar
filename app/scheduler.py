"""APScheduler entegrasyonu – periyodik CRT taraması."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.database import async_session
from app.scanner import run_scan

log = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="UTC")

SCAN_SCHEDULE = {
    "4h_candles": {
        "trigger": CronTrigger(minute=5, timezone="UTC"),
        "timeframe": "4h",
        "description": "Her saat 5. dakikada",
    },
}


async def _scheduled_scan(timeframe: str) -> None:
    log.info("Scheduled scan starting (timeframe=%s)...", timeframe)
    try:
        async with async_session() as session:
            result = await run_scan(session, timeframe=timeframe, source="scheduler")
            log.info(
                "Scheduled scan done: %d setup(s), %d activated, %d closed, %d breakeven at %s",
                len(result["new_setups"]),
                len(result["activated"]),
                len(result["closed"]),
                len(result["breakeven"]),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            )
    except Exception:
        log.exception("Scheduled scan failed")


def start_scheduler() -> None:
    for job_id, cfg in SCAN_SCHEDULE.items():
        scheduler.add_job(
            _scheduled_scan,
            trigger=cfg["trigger"],
            kwargs={"timeframe": cfg["timeframe"]},
            id=job_id,
            replace_existing=True,
            name=cfg["description"],
        )

    scheduler.start()
    log.info(
        "Scheduler started with %d job(s): %s",
        len(SCAN_SCHEDULE),
        ", ".join(SCAN_SCHEDULE.keys()),
    )


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("Scheduler stopped.")


def get_scheduler_status() -> dict:
    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time
        next_run_tsi = (
            next_run.astimezone(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M UTC+3")
            if next_run
            else "paused"
        )
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run_tsi,
        })
    return {
        "running": scheduler.running,
        "jobs": jobs,
    }
