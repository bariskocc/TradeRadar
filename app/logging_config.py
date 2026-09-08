"""Uygulama log yapilandirmasi.

Uygulama hicbir yerde `logging.basicConfig` cagirmadigi icin scanner/crt_engine
icindeki tum `log.info` satirlari (NEW WAITING, SKIPPED (...), BACKFILL FILL,
TRAIL ARM ...) goruntulenmiyordu; yalnizca warning/exception Python'un
lastResort handler'i uzerinden stderr'e dusuyordu. Bu modul kok logger'a
konsol + donen dosya handler'i takar.

Zaman damgasi TSI (UTC+3) — UI ve Telegram ile ayni saat dilimi.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

from app.config import BASE_DIR, LOG_FILE_MAX_BYTES, LOG_BACKUP_COUNT, LOG_LEVEL

TSI = timezone(timedelta(hours=3))

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# Yil dahil: donen yedekler aylara yayilabiliyor, yilsiz damga belirsiz kalir.
_DATEFMT = "%d.%m.%y %H:%M:%S"

# REST/WS kutuphaneleri INFO'da her istegi yazar; bootstrap'ta yuzlerce satir.
_NOISY = ("httpx", "httpcore", "websockets", "asyncio")

_configured = False


class _TsiFormatter(logging.Formatter):
    """Zaman damgasini TSI (UTC+3) olarak yazar; makine saat diliminden bagimsiz."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=TSI)
        return dt.strftime(datefmt or _DATEFMT)


def setup_logging() -> None:
    """Kok logger'a konsol + donen dosya handler'i tak (bir kez)."""
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    formatter = _TsiFormatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / "traderadar.log",
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    # Yeniden yuklemede handler cogaltma.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(console)
    root.addHandler(file_handler)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Log yapilandirildi: seviye=%s dosya=%s",
        LOG_LEVEL, log_dir / "traderadar.log",
    )
