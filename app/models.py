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

    # Entry modeli: cisd (MSS seviyesi) | mss | ifvg (LTF IFVG kenari) | bpr (iki FVG kesisimi)
    entry_model = Column(String, nullable=True)
    ifvg_low = Column(Float, nullable=True)
    ifvg_high = Column(Float, nullable=True)
    # BPR bolgesi (21.09): invert olmus FVG ile donus hamlesinin FVG'sinin KESISIMI.
    # IFVG kolonlari gibi "bolge var mi" bilgisidir; entry olarak kullanilip
    # kullanilmadigini entry_model soyler.
    bpr_low = Column(Float, nullable=True)
    bpr_high = Column(Float, nullable=True)

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
    # Kalite skorunun kalem kalem kirilimi (crt_engine.SCORE_PART_KEYS), JSON:
    # {"base":2,"htf":2,...,"raw":11,"score":9}. 16.09'da eklendi -- oncesindeki
    # satirlarda NULL. Yalniz OLCUM icin: hangi +1/+2 gercekten kazandiriyor?
    score_parts = Column(String, nullable=True)
    # Kalemlerin arkasindaki surekli olcumler + uygunluk bayraklari
    # (crt_engine.SCORE_FEATURE_KEYS): wick orani, C2 govde orani, geri donus %, IFVG bosluk
    # orani, C1 range/ATR, dar stop carpani, smt_possible... Esik kalibrasyonu icin.
    features = Column(String, nullable=True)
    # Seviyeler DONDUGU andaki skor kirilimi ve olcumler. Sonuc izleme ilk gorulen seviyelerle
    # yapildigi icin, sonradan degisen skorla (C2 kapanisi / SMT) sonucu eslestirmek yaniltici.
    parts_at_levels = Column(String, nullable=True)
    features_at_levels = Column(String, nullable=True)

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
    # Entry modeli karsilastirmasi (18.09, kullanici sorusu): AYNI setupta butun aday entry'ler
    # ayri ayri izlenir -- SL ve TP sabit, yalniz entry degisir. JSON:
    # {"cisd": {...}, "mss": {...}, "ifvg_near|mid|far": {...}, "dfvg_near|mid|far": {...},
    #  "chosen": {...}} -> e (entry), rr, o (outcome), at, until.
    # Cevaplanan sorular: (a) IFVG'ye fiyat geliyor mu, bosuna mi bekliyoruz; (b) bolgenin ust/orta/
    # alt noktasindan limit emri dolar miydi (bugun UST kenar kullaniliyor, RR'si en dusuk nokta);
    # (c) kirilim sonrasi birakilan FVG daha mi iyi. Secilim etkisi yok: hepsi ayni setupta.
    entries = Column(String, nullable=True)

    # Geri cekilme derinligi (18.09): "fiyat nereye kadar geri cekiliyor, limit emri hangi
    # seviyeden doluyor?" Olcek: 0.0 = sinyal dogdugu andaki fiyat (ref), 1.0 = SL (purge ucu).
    # JSON: ref, d_max, d_tp (TP'den ONCE ulasilan en derin nokta), d_tp_bar (TP mumunun dibi --
    # mum ici sira bilinmedigi icin d_tp'ye KATILMAZ, duyarlilik icin ayri), tp_first, d_of
    # (her aday entry'nin bu olcekteki yeri), done.
    # Neden ayrica: 9 ayrik varyant yerine SUREKLI egri -> olculmemis seviyeler icin de cevap
    # (dolum orani p(d), RR(d) analitik, E[R](d) argmax). Karar girdisi KAZANAN islemlerin d_tp
    # dagilimidir: kaybeden setupta fiyat SL'ye kadar gittigi icin her derinlik dolar, yani kayip
    # terimi d'den bagimsiz sabittir; optimumun yerini yalniz kazananlar belirler.
    retrace = Column(String, nullable=True)
    prefill = Column(String, nullable=True)      # dolum oncesi TP yolunda gidilen en uzak nokta

    # Golge izleme (14.09, dar stop sorusu; 4H/1D/1H): ayni setup farkli SL kesirleriyle izlenir.
    # JSON: {"1": {...}, "0.75": {...}, "0.5": {...}} -> sl, rr, o (outcome), e (entry), at, until.
    shadow = Column(String, nullable=True)


class BiasJournal(Base):
    """1D bias'in TAHMIN KARNESI: her sembol, her kapanmis gun icin bir satir.

    Neden ayri tablo: Setup Journal setup basina yaziliyor, burada **setup cikmayan gunler de**
    olcumun parcasi (bias bir yon iddiasidir, islem olsun olmasin tutar ya da tutmaz).

    Neden "islem bias yonunde gitti mi" diye BAKMIYORUZ: hard filtre zaten yalnizca hizali
    setuplari geciriyor, yani kontrol grubu yok (dongusel orneklem); ayrica TP/SL'ye gidisi
    entry/stop yerlesimi ve kismi kar belirliyor, bias'in katkisi ayiklanamaz. Bunun yerine bias
    hava durumu tahmini gibi puanlanir: yon dogru muydu, hareket ATR'nin kac katiydi.

    Bilesenler AYRI tutulur (structure / ict / birlesik) ki "structure tek basina mi daha iyi",
    "STRUCTURE_STALE_DAYS = 7 dogru mu", "NEUTRAL gercekten yonsuz mu" sorulari YENI VERI
    BEKLEMEDEN cevaplanabilsin. `momentum` referans cizgisi: "dun ne yaptiysa bugun de onu yapar".
    Bias bunu gecemiyorsa tahmin degeri yoktur.
    """
    __tablename__ = "bias_journal"
    __table_args__ = (UniqueConstraint("day", "symbol", name="uq_bias_journal_key"),)

    id = Column(Integer, primary_key=True, index=True)
    day = Column(DateTime, nullable=False, index=True)      # 1D barin acilis zamani (kapanmis bar)
    symbol = Column(String, nullable=False, index=True)     # gorunum adi
    market_type = Column(String, nullable=True)

    structure = Column(String, nullable=True)               # htf_bias_with_age yonu
    structure_age = Column(Integer, nullable=True)          # son yapisal kirilimdan bu yana kapali gun
    ict = Column(String, nullable=True)                     # compute_ict_bias
    combined = Column(String, nullable=True)                # compute_daily_bias (motorun kullandigi)
    weekly = Column(String, nullable=True)                  # compute_weekly_bias
    # structure bayat sayilip karar ict'ye mi birakildi? (STRUCTURE_STALE_DAYS esigi o gun devrede miydi)
    stale_applied = Column(Boolean, nullable=True)
    momentum = Column(String, nullable=True)                # referans cizgisi: onceki gunun yonu

    close = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)

    # Ileriye donuk hareket, ATR'ye bolunmus (semboller kiyaslanabilsin) ve ISARETLI:
    # pozitif = fiyat yukari gitti. Isabet, bias yonuyle bu isaretin uyusmasidir.
    fwd_1d = Column(Float, nullable=True)
    fwd_3d = Column(Float, nullable=True)
    fwd_5d = Column(Float, nullable=True)

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


class PotentialNotice(Base):
    """Potansiyel 1D CRT bildiriminin kalici durumu (setup basina tek satir).

    Kapida elenen setup'in Signal kaydi yoktur; tekrar gonderimi engelleyecek
    `tg_potential_state` tasiyacak nesne de yok. Bu tablo o rolu ustlenir:
    anahtar (strategy, symbol, direction, purge_time) — Setup Journal'la ayni
    anahtar. Restart'ta bildirim tekrar gitmesin diye DB'de tutulur.

    `state` degeri "<open|closed>:<gated|pass>"; degistiginde zincire yeni bir
    reply gonderilir (C2 kapandi, ya da setup artik tum kapilardan geciyor).
    """
    __tablename__ = "potential_notices"
    __table_args__ = (
        UniqueConstraint("strategy", "symbol", "direction", "purge_time", name="uq_potential_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    purge_time = Column(DateTime, nullable=True)
    crt_bar_time = Column(DateTime, nullable=True)

    tg_message_id = Column(Integer, nullable=True)      # zincirin SON mesaji (reply hedefi)
    state = Column(String, nullable=True)               # open:gated / closed:gated / open:pass / ...
    gate = Column(String, nullable=True)                # son bildirilen kapi (bias_mismatch, low_rr, ...)
    cancelled = Column(Boolean, default=False)          # IPTAL reply'i gitti

    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PaperTrade(Base):
    """Deneme (paper) islem gunlugu: ELLE tutulan islem kaydi.

    Neden Signal tablosuna yazilmiyor: Dashboard, Analytics, Engine Report ve butun
    scripts/*_stat.py "motor ne yapti" sorusunu olcuyor; elle acilan islem oraya
    girerse motorun karnesi kirlenir. Bu tablo INSAN kararini olcer:
      - bot sinyalini aldim mi (`source='bot'`), motorun eledigi setup'i elle aldim mi
        (`source='gated'`, kapi adi `gate`), yoksa kendi setup'im mi (`source='manual'`)
      - plana uydum mu, hangi hatayi yaptim, o hata kac R'ye mal oldu

    R'nin paydasi DAIMA `stop_loss` (ilk SL) -- motorun `initial_stop_loss` ayrimiyla ayni
    kural, yoksa BE'ye cekilen stop R'yi bozar. Kismi cikis varsa R motordaki gibi
    AGIRLIKLI yazilir: `partial_fraction x kismi R + (1 - fraction) x kalan R`.
    """
    __tablename__ = "paper_trades"
    __table_args__ = (UniqueConstraint("ext_id", name="uq_paper_trade_ext"),)

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)     # gorunum adi (exchange.py); serbest de olabilir
    market_type = Column(String, nullable=True)             # crypto/fx/index/metal/oil/other (sembolden turer)
    direction = Column(String, nullable=False)              # LONG / SHORT
    strategy = Column(String, nullable=True)                # 4h / 1d / 1h / other

    # Karar kaynagi + bagli motor kaydi (ikisi de opsiyonel; tamamen manuel islem de olur)
    source = Column(String, nullable=False, default="manual")   # bot / gated / manual
    signal_id = Column(Integer, nullable=True, index=True)      # Signal.id
    journal_id = Column(Integer, nullable=True, index=True)     # SetupJournal.id
    # Elenen setup'in kapisi. Bagli satir degisse/silinse de karne bozulmasin diye KOPYALANIR.
    gate = Column(String, nullable=True)

    # Plan (acarken)
    entry_price = Column(Float, nullable=False)
    # ILK SL; R'nin paydasi, degismez. TradingView ice aktarmasinda emir dosyasi yoksa (ya da
    # islem stop emirsiz acildiysa) BOS kalabilir -> o kayitta R hesaplanmaz, sonuc paradan yazilir.
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    planned_rr = Column(Float, nullable=True)               # acilista hesaplanip SAKLANIR
    entered_at = Column(DateTime, nullable=True, index=True)
    confidence = Column(Integer, nullable=True)             # 1-5

    # Kismi cikis (opsiyonel; motorun kismi kar davranisiyla ayni cetvel)
    partial_price = Column(Float, nullable=True)
    partial_fraction = Column(Float, nullable=True)
    partial_at = Column(DateTime, nullable=True)

    # Kapanis
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)             # tp/sl/be/trail/manual/week_close
    closed_at = Column(DateTime, nullable=True, index=True)
    rr_value = Column(Float, nullable=True)                 # gerceklesen R (agirlikli)
    result = Column(String, nullable=True)                  # win / loss / breakeven (R'nin isaretinden)
    duration_hours = Column(Float, nullable=True)

    # Ogrenme (kapanista sorulur -- o an hatirlanir)
    followed_plan = Column(Boolean, nullable=True)
    mistakes = Column(String, nullable=True)                # virgullu etiket listesi (MISTAKE_TAGS)
    note = Column(String, nullable=True)
    chart_url = Column(String, nullable=True)

    # Para. R motorla kiyaslanabilen tek birim, ama gercek sonuc para; ikisi birlikte tutulur.
    # Ice aktarmada TradingView'in kendi rakamlari yazilir (komisyon dahil), elle girilen islemde
    # `qty` verilmisse fiyat farkindan hesaplanir.
    currency = Column(String, nullable=True, default="USD")
    qty = Column(Float, nullable=True)                      # Boyut (miktar)
    notional = Column(Float, nullable=True)                 # Boyut (deger)
    leverage = Column(Float, nullable=True)
    margin = Column(Float, nullable=True)                   # Teminat
    fees = Column(Float, nullable=True)                     # Komisyon
    pnl_amount = Column(Float, nullable=True)               # Net K/Z (komisyon dahil)
    return_pct = Column(Float, nullable=True)               # Getiri %

    status = Column(String, nullable=False, default="open")  # open / closed

    # Ice aktarma (TradingView paper trading CSV'si) -- elle girilen kayitta NULL.
    ext_source = Column(String, nullable=True)              # 'tv'
    ext_id = Column(String, nullable=True)                  # TV emir no; mukerrer kaydi engeller

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
