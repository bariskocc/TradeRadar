from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} .env icinde tanimli olmali")
    return value


SECRET_KEY = _require_env("SECRET_KEY")
ADMIN_USERNAME = _require_env("ADMIN_USERNAME")
ADMIN_PASSWORD = _require_env("ADMIN_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR / 'traderadar.db'}")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 saat

# ──────────────────── BingX veri kaynagi ────────────────────
BINGX_REST_BASE = os.getenv("BINGX_REST_BASE", "https://open-api.bingx.com")
BINGX_WS_URL = os.getenv("BINGX_WS_URL", "wss://open-api-swap.bingx.com/swap-market")

# Bootstrap'ta cekilecek gecmis mum sayilari (timeframe basina)
BOOTSTRAP_LIMITS = {
    "4h": 120,
    "1d": 60,
    "15m": 200,
    "1h": 200,
    "5m": 300,
}

# Bakim dongusu araligi (saniye): aktif sembol seti degisim kontrolu + gunluk
# 1D yenileme burada yapilir. Periyodik REST TARAMA YOKTUR; tespit tamamen WS
# olaylariyla (15m/4h kapanis) calisir. 0 => bakim dongusu kapali.
MAINTENANCE_INTERVAL_SEC = int(os.getenv("MAINTENANCE_INTERVAL_SEC", "1800"))
