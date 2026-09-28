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

# ──────────────────── Loglama ────────────────────
# Scanner/CRT karar loglari (NEW WAITING, SKIPPED, BACKFILL FILL ...) INFO'dadir.
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", str(5 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

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

# Seans sembolleri (FX/metal/endeks/petrol) icin 4H ve 1D serileri BingX'ten
# degil, 1H'ten NY-hizali sentezlenir (bkz. app/session.py). Olu seans (haftanin
# 48 saati) elendigi icin takvim saatinin ~5/7'si canli kalir:
#   60 islem gunu = 1440 canli saat ~= 2016 takvim saati
# BingX tek istekte en fazla 1000 mum dondurdugunden sayfalanarak cekilir.
SESSION_1H_BARS = int(os.getenv("SESSION_1H_BARS", "2100"))
SESSION_1H_PAGE = 1000

# 1W-4H stratejisi (28.09): haftalik HTF 1D'den sentezlenir ve CRT tespiti >= 16 haftalik mum
# + ATR14 ister; aylik bias/PMH-PML icin 2 kapali ay gerekir. Bugunku seriler yetmiyor (kripto 1D
# 60 bar ~ 8 hafta, seans 1H 2100 bar ~ 12 hafta). Bu yuzden 1W evrenine AYRI, uzun bir 1D
# gecmisi (store "1d_deep") tutulur; 1h/4h/1d serileri eskisiyle birebir ayni kalir.
# - Kripto (BTC/ETH): ayni 1D istegi daha buyuk limitle (ek istek YOK), kuyrugu "1d"ye yazilir.
# - Seans: derin 1H sayfasi yalniz "1d_deep" yoksa (acilis) cekilir (~5 sayfa, 2100'un 3'u yerine).
W1_DAILY_BARS = int(os.getenv("W1_DAILY_BARS", "400"))
W1_SESSION_1H_BARS = int(os.getenv("W1_SESSION_1H_BARS", "4400"))

# 1H-5M stratejisi 28.09'da KAPATILDI (kullanici karari; yerine 1W-4H). Kod duruyor: True yapmak
# 5m aboneliklerini, bootstrap'i ve 1H tespitini geri acar (scanner.STRATEGY_CFG'deki yorumlu
# 1H blogu da acilmali).
H1_5M_ENABLED = False

# Bakim dongusu araligi (saniye): aktif sembol seti degisim kontrolu + gunluk
# 1D yenileme burada yapilir. Periyodik REST TARAMA YOKTUR; tespit tamamen WS
# olaylariyla (15m/4h kapanis) calisir. 0 => bakim dongusu kapali.
MAINTENANCE_INTERVAL_SEC = int(os.getenv("MAINTENANCE_INTERVAL_SEC", "1800"))
