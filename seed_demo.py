"""Demo verileri ekler. Bir kez çalıştırın: python seed_demo.py"""
import asyncio
import random
from datetime import datetime, timedelta, timezone
from app.database import engine, init_db, async_session, Base
from app.models import Signal

SYMBOLS_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "AVAXUSDT", "APTUSDT", "BCHUSDT", "AAVEUSDT", "LINKUSDT", "ADAUSDT"]
SYMBOLS_FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
SYMBOLS_INDEX = ["US100", "US500", "US30", "GER40"]
SYMBOLS_METAL = ["XAUUSD", "XAGUSD"]

ALL_SYMBOLS = (
    [(s, "crypto") for s in SYMBOLS_CRYPTO]
    + [(s, "fx") for s in SYMBOLS_FX]
    + [(s, "index") for s in SYMBOLS_INDEX]
    + [(s, "metal") for s in SYMBOLS_METAL]
)


def random_signal(i: int, now: datetime) -> Signal:
    symbol, market = random.choice(ALL_SYMBOLS)
    direction = random.choice(["LONG", "SHORT"])
    purge_type = "LOW" if direction == "LONG" else "HIGH"
    result = random.choices(["win", "loss", "breakeven"], weights=[45, 35, 20])[0]
    rr = round(random.uniform(0.5, 3.5), 2) if result == "win" else (round(-random.uniform(0.5, 1.0), 2) if result == "loss" else 0.0)
    bias_map = {"win": ("BULLISH" if direction == "LONG" else "BEARISH"), "loss": ("BEARISH" if direction == "LONG" else "BULLISH"), "breakeven": "NEUTRAL"}
    bias = bias_map[result] if random.random() > 0.3 else random.choice(["BULLISH", "BEARISH", "NEUTRAL"])
    score = random.randint(1, 5) if bias == ("BULLISH" if direction == "LONG" else "BEARISH") else random.randint(-5, -1) if bias == ("BEARISH" if direction == "LONG" else "BULLISH") else 0
    offset = timedelta(hours=4 * i)
    crt_time = now - offset - timedelta(hours=4)
    purge_time = crt_time + timedelta(hours=random.uniform(0.5, 3.0))
    duration = round(random.uniform(1.0, 12.0), 1)

    if i < 8:
        status, result_val, rr_val = "active", None, None
    else:
        status = "hit" if result == "win" else "expired"
        result_val = result
        rr_val = rr

    return Signal(
        symbol=symbol, direction=direction, purge_type=purge_type,
        bias=bias, bias_score=score, market_type=market, timeframe="4h",
        crt_bar_time=crt_time, purge_time=purge_time,
        result=result_val, rr_value=rr_val, duration_hours=duration,
        status=status, created_at=now - offset,
    )


async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    random.seed(42)
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        for i in range(80):
            session.add(random_signal(i, now))
        await session.commit()
    print("80 demo sinyal eklendi.")


if __name__ == "__main__":
    asyncio.run(seed())
