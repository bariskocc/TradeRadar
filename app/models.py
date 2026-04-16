from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime, timezone

from app.database import Base


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    direction = Column(String, nullable=False)            # LONG / SHORT
    purge_type = Column(String, nullable=False)            # HIGH / LOW
    bias = Column(String, nullable=True)                   # BULLISH / BEARISH / NEUTRAL
    bias_score = Column(Float, nullable=True)

    # CRT key levels (4H)
    key_level_high = Column(Float, nullable=True)
    key_level_low = Column(Float, nullable=True)
    crt_bar_time = Column(DateTime, nullable=True)
    purge_time = Column(DateTime, nullable=True)

    # Trade levels
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    invalidation_level = Column(Float, nullable=True)      # CRT mumunun %50'si
    reached_50pct = Column(Boolean, default=False)         # Fiyat %50'ye ulasti mi?

    # CISD confirmation (15M)
    cisd_confirmed = Column(Boolean, default=False)
    cisd_time = Column(DateTime, nullable=True)
    cisd_price = Column(Float, nullable=True)

    # Result tracking
    result = Column(String, nullable=True)                 # win / loss / breakeven
    rr_value = Column(Float, nullable=True)
    duration_hours = Column(Float, nullable=True)

    # HTF Bias (daily/weekly trend)
    htf_bias = Column(String, nullable=True)               # BULLISH / BEARISH / NEUTRAL

    # Status: pending_cisd → active → expired (win/loss/invalidated) | breakeven
    status = Column(String, default="pending_cisd")
    market_type = Column(String, default="crypto")
    timeframe = Column(String, default="4h")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
