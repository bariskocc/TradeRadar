"""Sol menu rozetleri (04.11, kullanici): Watchlist / S/D Watchlist yaninda "aksiyon bekleyen" madde sayisi.

`build_watchlist` CRT grubunda ~12 sn (40 madde, onlarca sorgu); her sayfa isteginde calistirilamaz. Sayilar bellekte tutulur,
her sayfa isteginde (`schedule_refresh`, HTTP middleware) TTL gectiyse ARKA PLANDA yeniden hesaplanir -- sayfa beklemez, gosterilen
sayi en fazla bir onceki yuklemeden kalir. Izleme sayfalari kendi hesabini `set_from` ile aninda yazar.
"""
from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger(__name__)

TTL_SEC = 60
_cache: dict = {"crt": None, "sd": None, "at": 0.0, "running": False}


def get() -> dict:
    """Sablon icin: {"crt": n | None, "sd": n | None}."""
    return {"crt": _cache["crt"], "sd": _cache["sd"]}


def set_from(group: str, counts: dict) -> None:
    if group in ("crt", "sd") and isinstance(counts, dict) and "action" in counts:
        _cache[group] = int(counts["action"])
        _cache["at"] = time.time()


async def refresh() -> None:
    if _cache["running"]:
        return
    _cache["running"] = True
    try:
        from app.database import async_session
        from app.watchlist import build_watchlist
        for group in ("crt", "sd"):
            async with async_session() as s:
                data = await build_watchlist(s, group=group)
            set_from(group, data.get("counts") or {})
    except Exception:
        log.exception("watch badge refresh failed")
    finally:
        _cache["running"] = False


def schedule_refresh(force: bool = False) -> None:
    """Sayfa isteginde cagrilir: TTL gectiyse arka planda yenile (sayfa beklemez)."""
    if _cache["running"]:
        return
    if not force and time.time() - _cache["at"] < TTL_SEC:
        return
    try:
        asyncio.get_running_loop().create_task(refresh())
    except RuntimeError:
        pass
