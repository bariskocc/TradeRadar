"""Olay bazli log yardimcisi.

Event-driven mimaride periyodik/bos tarama loglari yerine yalnizca gercek
olaylar kaydedilir. Cagiran bir session verirse ayni transaction'a eklenir
(commit cagirana birakilir); vermezse kendi session'ini acip commit eder.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models import EventLog

log = logging.getLogger(__name__)


async def record_event(
    event_type: str,
    message: str,
    *,
    symbol: str | None = None,
    direction: str | None = None,
    market_type: str | None = None,
    level: str = "info",
    session: AsyncSession | None = None,
) -> None:
    row = EventLog(
        event_type=event_type,
        level=level,
        symbol=symbol,
        direction=direction,
        market_type=market_type,
        message=message,
    )
    try:
        if session is not None:
            # Cagiranin transaction'ina ekle; commit cagirana ait.
            session.add(row)
            return
        async with async_session() as own:
            own.add(row)
            await own.commit()
    except Exception:
        log.warning("record_event failed: %s %s", event_type, message)
