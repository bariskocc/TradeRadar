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
    initial_stop_loss = Column(Float, nullable=True)       # Fill anindaki orijinal SL (R hesabi)
    invalidation_level = Column(Float, nullable=True)      # CRT mumunun %60'i (bilgi; BE artik buna bagli degil)
    reached_50pct = Column(Boolean, default=False)         # Legacy: BE korumasi aktif
    partial_hit = Column(Boolean, default=False)           # BE: SL->entry cekildi
    trail_active = Column(Boolean, default=False)          # MFE trail acildi
    mfe_price = Column(Float, nullable=True)               # Aktifken en iyi lehine fiyat
    # BE/trail korumasinin devreye girdigi an. Cekilmis stop, bu andan ONCEKI
    # mumlara uygulanmaz; aksi halde reconcile gecmisi tekrar oynatirken fill
    # mumunun kendi low'u (LONG'da dogal olarak entry'nin altinda) BE stopunu
    # tetikliyor ve kazanan islem "breakeven" yaziliyordu.
    protection_armed_time = Column(DateTime, nullable=True)

    # CISD confirmation (15M)
    cisd_confirmed = Column(Boolean, default=False)
    cisd_time = Column(DateTime, nullable=True)
    cisd_price = Column(Float, nullable=True)
    # MSS'in kirdigi onceki swing mumunun acilis zamani (15M)
    mss_ref_time = Column(DateTime, nullable=True)

    # Limit-order giris: fiyat CISD/entry seviyesine donunce dolan an
    entry_filled_time = Column(DateTime, nullable=True)

    # Result tracking
    result = Column(String, nullable=True)                 # win / loss / breakeven
    # Cikisin SEBEBI: tp / sl / be / trail / week_close.
    # `result` R'nin isaretine gore yazilmaya devam eder (istatistikler bozulmasin);
    # bu kolon "nasil cikildi" sorusunu ayri tutar. week_close = Cuma 17:00 NY'de
    # kripto-disi pozisyonun duzlestirilmesi.
    exit_reason = Column(String, nullable=True)
    planned_rr = Column(Float, nullable=True)              # Potansiyel R:R (olusturulunca; degismez)
    rr_value = Column(Float, nullable=True)                # Gerceklesen R (kapanista); acikken plan ile ayni olabilir
    duration_hours = Column(Float, nullable=True)

    # HTF Bias (1D) — trade filtresinde kullanilir
    htf_bias = Column(String, nullable=True)               # BULLISH / BEARISH / NEUTRAL
    # Haftalik bias — yalnizca bilgi; trade kararina etki ETMEZ
    weekly_bias = Column(String, nullable=True)            # BULLISH / BEARISH / NEUTRAL

    # SMT divergence bulunan korele parite (gosterim adi); yoksa None
    smt_pair = Column(String, nullable=True)

    # Purge (C2) mumunun dokundugu PD array etiketleri (or. 'PDH,FVG'); yoksa None
    pd_array = Column(String, nullable=True)

    # Purge (C2) 4H mumu setup aninda KAPANMIS miydi? (kapali = daha guvenilir)
    c2_closed = Column(Boolean, default=False)

    # Telegram: aktif sinyal mesajinin message_id'si (sonucu buna reply atmak icin)
    tg_message_id = Column(Integer, nullable=True)

    # Entry modeli: cisd (MSS seviyesi) | ifvg (LTF IFVG %50)
    entry_model = Column(String, nullable=True)
    ifvg_low = Column(Float, nullable=True)
    ifvg_high = Column(Float, nullable=True)

    # Status: pending_cisd → waiting_entry → active → expired (win/loss) | breakeven
    status = Column(String, default="waiting_entry")
    market_type = Column(String, default="crypto")
    timeframe = Column(String, default="4h")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)                # manual / scheduler
    timeframe = Column(String, nullable=False)             # 4h, 1h...
    status = Column(String, nullable=False, default="success")  # success / failed

    new_setups = Column(Integer, default=0)
    activated = Column(Integer, default=0)
    closed = Column(Integer, default=0)
    breakeven = Column(Integer, default=0)

    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(String, nullable=True)


class EventLog(Base):
    """Olay bazli log (event-driven mimari).

    Periyodik/bos tarama kaydi tutulmaz; yalnizca gercek olaylar yazilir:
    new_setup, filled, closed_win, closed_loss, breakeven, cancelled,
    ws_connect, ws_disconnect, scan, error.
    """
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    level = Column(String, nullable=False, default="info")   # info / success / warning / error
    symbol = Column(String, nullable=True, index=True)
    direction = Column(String, nullable=True)                # LONG / SHORT
    market_type = Column(String, nullable=True)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
