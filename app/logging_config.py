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
from collections import deque
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


class _IssueBuffer(logging.Handler):
    """Son WARNING/ERROR kayitlarini bellekte tutar (Dashboard saglik karti).

    Dashboard'un her yuklemede 3+ MB log dosyasini okumamasi icin; surec basindan beri
    gecerli, restart'ta sifirlanir.
    """

    def __init__(self, maxlen: int = 30) -> None:
        super().__init__(level=logging.WARNING)
        self.items: deque[dict] = deque(maxlen=maxlen)
        self.counts = {"WARNING": 0, "ERROR": 0}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = "ERROR" if record.levelno >= logging.ERROR else "WARNING"
            self.counts[level] += 1
            self.items.append({
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc),
                "level": level,
                "logger": record.name,
                "msg": record.getMessage()[:240],
            })
        except Exception:
            pass


_ISSUES = _IssueBuffer()


def recent_issues(hours: float = 24.0) -> dict:
    """Son `hours` saatteki uyari/hatalar (en yeni sonda) + surec basindan beri sayilar."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items = [i for i in list(_ISSUES.items) if i["ts"] >= cutoff]
    return {
        "items": items,
        "warnings": sum(1 for i in items if i["level"] == "WARNING"),
        "errors": sum(1 for i in items if i["level"] == "ERROR"),
        "total_warnings": _ISSUES.counts["WARNING"],
        "total_errors": _ISSUES.counts["ERROR"],
    }


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
    root.addHandler(_ISSUES)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Log yapilandirildi: seviye=%s dosya=%s",
        LOG_LEVEL, log_dir / "traderadar.log",
    )
