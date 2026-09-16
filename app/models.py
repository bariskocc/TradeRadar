from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, UniqueConstraint
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
    # Kismi kar (4H/1D): BE esiginde pozisyonun `partial_size` kesri `partial_price`
    # seviyesinden kapatilir. rr_value kapanista agirlikli yazilir:
    # partial_size x partial_rr + (1 - partial_size) x kalan kismin R'si.
    partial_size = Column(Float, nullable=True)
    partial_price = Column(Float, nullable=True)
    partial_rr = Column(Float, nullable=True)
    partial_time = Column(DateTime, nullable=True)

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
    # Kapanis ani (TP/SL/BE/trail mumu ya da hafta kapanisi). Donem ozetleri (Dashboard
    # haftalik/aylik, Analytics haftalik) islemi KAPANDIGI doneme sayar. 14.09 oncesi
    # kayitlar init_db'de dolum (yoksa CISD) + duration_hours ile dolduruldu.
    closed_at = Column(DateTime, nullable=True)

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
    # Telegram: EN SON 1D "POTANSIYEL CRT" mesaji (sonraki C2 durumu / iptal / ACTIVE buna reply)
    # ve en son hangi C2 durumuyla bildirildigi ('open' | 'closed'); tekrar gondermez.
    tg_potential_id = Column(Integer, nullable=True)
    tg_potential_state = Column(String, nullable=True)

    # Entry modeli: cisd (MSS seviyesi) | ifvg (LTF IFVG %50)
    entry_model = Column(String, nullable=True)
    ifvg_low = Column(Float, nullable=True)
    ifvg_high = Column(Float, nullable=True)

    # Status: pending_cisd → waiting_entry → active → expired (win/loss) | breakeven
    status = Column(String, default="waiting_entry")
    market_type = Column(String, default="crypto")
    timeframe = Column(String, default="4h")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SetupJournal(Base):
    """Setup basina tek satir: motor kapilarindaki yolu + elendikten sonra fiyatin ne yaptigi.

    Anahtar: strateji + sembol + yon + purge (C2) zamani. Yazan: app/setup_journal.py.
    entry/sl/tp/rr ilk goruldugu haliyle dondurulur; outcome bunlarla izlenir.
    """
    __tablename__ = "setup_journal"
    __table_args__ = (
        UniqueConstraint("strategy", "symbol", "direction", "purge_time", name="uq_setup_journal_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String, nullable=False, index=True)       # 4h / 1d / 1h
    symbol = Column(String, nullable=False, index=True)
    market_type = Column(String, nullable=True)
    direction = Column(String, nullable=False)
    purge_time = Column(DateTime, nullable=False)
    crt_bar_time = Column(DateTime, nullable=True)

    first_seen = Column(DateTime, nullable=True, index=True)
    last_seen = Column(DateTime, nullable=True, index=True)
    last_stage = Column(String, nullable=True)                  # son degerlendirmedeki asama
    best_stage = Column(String, nullable=True)                  # ulastigi en ileri asama
    best_stage_at = Column(DateTime, nullable=True)
    detail = Column(String, nullable=True)                      # "score 6 · RR 1.54 · 1D BEARISH"
    score = Column(Float, nullable=True)
    htf_bias = Column(String, nullable=True)
    weekly_bias = Column(String, nullable=True)
    c2_closed = Column(Boolean, nullable=True)
    model = Column(String, nullable=True)

    entry = Column(Float, nullable=True)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    rr = Column(Float, nullable=True)
    levels_at = Column(DateTime, nullable=True)                 # seviyelerin ilk goruldugu an

    deleted_reason = Column(String, nullable=True)              # pending/waiting silindiyse neden
    deleted_at = Column(DateTime, nullable=True)

    # pending/filled (izleniyor) | tp_before_entry | win | loss | no_touch | open | ambiguous | signal
    outcome = Column(String, nullable=True)
    outcome_at = Column(DateTime, nullable=True)
    entry_touched_at = Column(DateTime, nullable=True)
    tracked_until = Column(DateTime, nullable=True)
    # Golge izleme (14.09, dar stop sorusu; 4H/1D/1H): ayni setup farkli SL kesirleriyle izlenir.
    # JSON: {"1": {...}, "0.75": {...}, "0.5": {...}} -> sl, rr, o (outcome), e (entry), at, until.
    shadow = Column(String, nullable=True)


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
