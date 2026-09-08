# CLAUDE.md

TradeRadar — CRT/ICT tabanlı kişisel trade sinyal radarı ve dashboard'u.

Açık işler: **[TODO.md](TODO.md)** — madde tamamlanınca o dosyadan silinir.

## Çalışma kuralları (her görevden önce)

`.cursor/rules/project-workflow.mdc` bağlayıcıdır:

1. **Token-yoğun işlemden önce izin al** — büyük taramalar, geniş refactor, çok
   dosya okuma/düzenleme, uzun analizler. Tahmini maliyeti **M (milyon) token**
   olarak belirt. Onay yoksa yapma.
2. **Sunucuyu izinsiz yeniden başlatma** — bir görev bitince kullanıcı istemeden
   `python run.py` restart etme. Gerekiyorsa önce sor.
3. **1M token'ı aşarsan dur ve sor** — devam onayı gelmezse durdur.

Kod stili: çevre kodla aynı — Türkçe yorumlar, mevcut isimlendirme ve deyim.

## Çalıştırma

```powershell
python run.py            # uvicorn, http://0.0.0.0:8000
```

- Python 3.14, bağımlılıklar `requirements.txt` (venv yok, global kurulu).
- **Bash tool bu makinede bozuk** (`ls`/`grep`/`python` → exit 127). Terminal
  işleri için **PowerShell tool** kullan. Sunucuyu arka planda başlatmak için:
  `Start-Process python -ArgumentList 'run.py' -RedirectStandardOutput server.log -RedirectStandardError server.err.log -NoNewWindow -PassThru`
- Giriş: tek admin, kullanıcı adı/parola `.env` (`ADMIN_USERNAME`/`ADMIN_PASSWORD`).
- `.env` anahtarları: `SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`,
  `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DATABASE_URL` (opsiyonel).
- Tailwind: `tools/tailwindcss.exe` ile `app/static/css/input.css` →
  `app/static/css/app.css` derlenir (CDN yok, yerel).
- DB: `traderadar.db` (SQLite, WAL). Şema `init_db()` içinde `create_all` +
  `_MIGRATIONS` (elle additive `ALTER TABLE` listesi, `app/database.py`).
- `seed_demo.py` DB'yi **siler** ve sahte veri basar — yalnızca geliştirme.
- `tmp_*.py` / `tmp_*.json` gitignore'lu geçici analiz dosyaları.
- Test yok.

### Loglama

`app/logging_config.py` → `setup_logging()`, `app/main.py` import edilirken çağrılır
(hangi giriş noktası olursa olsun geçerli). Kök logger'a **konsol + dönen dosya**
handler'ı takar: `logs/traderadar.log` (5 MB × 5 yedek). Zaman damgası **TSİ
(UTC+3)**, makine saat diliminden bağımsız. `httpx`/`httpcore`/`websockets`/
`asyncio` WARNING'e kısılır (bootstrap'ta yüzlerce satır yazıyorlardı).

Ayarlar `.env`: `LOG_LEVEL` (varsayılan INFO), `LOG_FILE_MAX_BYTES`, `LOG_BACKUP_COUNT`.

Motorun karar satırları burada: `NEW WAITING` / `NEW PENDING` / `NEW ACTIVE
(backfill)`, `PROMOTE WAITING`, `SKIPPED (...)` (BIAS / LOW RR / TIGHT STOP /
LOW QUALITY / SCORE7 / PAST TP / PAST SL / CRT 60% / TARGET TAKEN /
MISSED QUALITY), `FILLED`, `BACKFILL FILL`, `BE ARM`, `TRAIL ARM`, `TRAIL`,
`CLOSED (...)`, `SMT+2`. Uvicorn'un kendi logları `server.err.log`'a gider.

## Mimari

Tek veri kaynağı: **BingX USDT-M perpetual swap**. FX/metal/endeks/petrol de
BingX'in `NC*` sentetik kontratlarına eşlenir (`app/exchange.py`).

| Dosya | Görev |
|---|---|
| `app/main.py` | FastAPI route'ları, Jinja filtreleri, `RADAR_STATE_META`, dashboard istatistikleri |
| `app/config.py` | `.env` yükleme, BingX URL'leri, `BOOTSTRAP_LIMITS`, `MAINTENANCE_INTERVAL_SEC` |
| `app/auth.py` | JWT cookie (`access_token`), `verify_credentials` (düz karşılaştırma) |
| `app/database.py` | async engine, `Base`, `_MIGRATIONS`, WAL pragmaları |
| `app/models.py` | `Signal`, `ScanLog` (kullanılmıyor gibi), `EventLog` |
| `app/exchange.py` | Sembol listeleri, ham↔görünüm dönüşümü, gün/haftasonu aktif evren, SMT korelasyon çiftleri, REST OHLCV (`fetch_ohlcv`) |
| `app/market_data.py` | **Tek kalıcı BingX WebSocket.** REST bootstrap → canlı 15m/4h (+1h 1D evreni, +5m 1H evreni). 1D barı 1h'ten sentezlenir. Processor loop olayları scanner'a iletir. Maintenance loop: sembol seti değişince re-bootstrap, günlük 1D yenileme, WS kaçağına karşı REST reconcile |
| `app/scheduler.py` | Eski APScheduler kaldırıldı; sadece WS durumunu scanner sayfası formatına çevirir |
| `app/crt_engine.py` | Saf hesap motoru — CRT setup tespiti, CISD/MSS onayı, kalite skoru, PD array, LTF IFVG, 1D/1W bias, SMT divergence |
| `app/scanner.py` | Sinyal yaşam döngüsü, radar state makinesi, fill/TP/SL/BE/trail, reconcile, `run_scan` |
| `app/telegram.py` | Aktif sinyal mesajı + sonucu ona reply |
| `app/event_log.py` | `record_event` — olay bazlı log (periyodik/boş tarama kaydı yok) |

### Olay akışı

Periyodik REST tarama **yoktur**. Tespit tamamen mum kapanışı olayıyla:

- `market_data._processor_loop` → `scanner.on_candle_closed(symbol, tf, store)`
  → `detect_and_create_waiting` (her strateji) + kapanan LTF mumuyla
  `manage_symbol_on_price`.
- Forming mum güncellemesi → `scanner.on_price_update` → yalnızca aktif
  TP/SL/trail (fill yok).
- `reconcile_open_signals` (reconnect + maintenance) → WS kaçağına karşı REST
  TP/SL/BE güvenlik ağı (özellikle seanslı FX/metal).
- `run_scan` → manuel `/api/scan` ve maintenance warmup; aynı mantığı REST/store
  ile tek seferlik çalıştırır.

### Üç strateji (`STRATEGY_CFG`, `app/scanner.py`)

| Kod | HTF/LTF | Evren |
|---|---|---|
| `4h` | 4H CRT + 15m CISD | Tüm kripto listesi + hafta içi XAU |
| `1d` | 1D CRT + 1h CISD | BTC/ETH + FX/metal/endeks/petrol |
| `1h` | 1H CRT + 5m CISD | XAU, EURUSD, US100, BTC |

UI'da her yerde `?tf=4h|1d|1h` ile geçiş.

**Koruma eşikleri** — üçünde de trail = **TP yolunun %75'i**; sabit R
tetikleyicileri (`be_arm_r`, `trail_arm_r`) kapalı. BE yalnızca **4H**'te açık:
`be_arm_tp_fraction = 0.50` (TP yolunun %50'si). Sabit +1R BE / +1.5R trail,
5R'lik bir işlemde yolun %20'sinde tetikleyip işlemi erken boğuyordu; TP kesri
hedefe göre ölçekleniyor (min RR 2.0'da TP %50 zaten +1R'ye denk). 1D/1H'te BE
kapalı — 1h/5m mum boyu 1R'ye yakın olduğu için entry'deki stop iğneye açık
(bkz. NZDUSD notu, [TODO.md](TODO.md) madde 7).

**1H-5M istisnası**: `require_c2_closed = False`. C2 yalnızca 1 saat olduğu için
kapanışını beklemek CISD onayından sonra 45 dk'ya kadar ölü bekleme demek ve
retest tipik olarak o pencerede oluşuyor. Ters dönüş kanıtı zaten CISD onayı;
kapanmamış C2'nin skorda −1 bedeli var. **IFVG girişi C2 kapanmadan hâlâ kapalı**
(`ifvg_requires_c2_closed`), yani C2 penceresinde daima CISD/MSS kullanılır.
Skor-7 kapısı da açık kalır ama içindeki C2 şartı `require_c2_closed`'a bağlı
olduğu için 1H'te düşer (CISD entry + PD array şartı aynen durur).
Takas: daha çok fırsat ↔ C2 içinde yeni dip/tepe olursa tamponsuz SL yenebilir.

### Sinyal yaşam döngüsü

`pending_cisd → waiting_entry → active → expired (win/loss) | breakeven`

1. HTF CRT pattern (C1 → purge/sweep → C2 içeri kapanış) + LTF MSS/CISD onayı.
2. Kapalı C2 + onay olmadan `waiting`/fill **yok** (o ana kadar `pending_cisd`,
   radar `c2_open`) — **1H-5M hariç** (`require_c2_closed = False`, bkz. yukarı).
3. Fill: yalnızca CISD onay mumundan **sonraki kapanmış** LTF mumunda, fiyat
   entry'ye retest edip SL'ye gitmediyse → `active`.

**Entry seviyesi seçimi** (`_maybe_ifvg_entry` + `check_cisd_confirmation` →
`_pick_wider_stop`; sistem "limit emri" gibi davranır):

- **LTF IFVG varsa** (LTF'de purge sonrası invert olmuş, mid CRT C1 aralığında,
  henüz mitigate edilmemiş FVG) → giriş **IFVG %50 (mid)**, `entry_model='ifvg'`.
  Ön koşullar: `ifvg_requires_c2_closed` (C2 kapanmadan IFVG girişi devre dışı),
  IFVG mid'i SL–TP arasında olmalı, boşluk ≥ `MIN_IFVG_GAP_RANGE_FRAC` (0.15) ×
  son 20 LTF mumunun ort. range'i (gürültü filtresi), **ve IFVG RR'si CISD
  adayınınkinden düşük olmamalı** — LONG'da mid CISD entry'sinin üstünde
  (SHORT'ta altında) kalırsa stop genişler, RR düşer; öyle bir IFVG kullanılmaz.
  Kullanılmasa da `ifvg_low`/`ifvg_high` yazılır (UI rozeti).
- **IFVG yoksa** → **CISD** (purge öncesi LTF'deki son yükseliş/düşüş mum
  bloğunun ilk mumunun açılış fiyatı) ile **MSS** (purge öncesi son swing
  high/low) adaylarından **hangisi daha yüksek RR veriyorsa** (entry SL'ye daha
  yakın = dar stop) o seçilir. `_pick_wider_stop` adı yanıltıcı — gövdesi doğru:
  daha iyi RR'li / SL'ye yakın adayı döndürür.
- **CISD blok başlangıcı**: likiditeyi süpüren LTF mumu **kendisi** temiz bir
  zıt gövde ise (geniş wick'li doji değil) bloğa **dahildir** ve açılışı CISD
  seviyesidir. Doji/wick-rejection ise bir önceki bloğa düşülür. (`_check_*_cisd`,
  `run_end = ext_iloc if _is_*_body(...)`.)
- **SL**: doğrudan purge ucu (`setup.purge_extreme`). Eski LTF-range tamponu
  **kaldırıldı** — C2 kapanmadan fill olmadığı için C2 sonrası yeni bir ekstrem
  beklenmez. Aşırı dar stop hâlâ `min_stop_range_mult` (1.5×) ile `tight_stop`.
- **Bilinen açık**: IFVG girişi seçildiğinde fiyat IFVG'yi hiç etiketlemezse
  CISD'ye **geri dönüş yok** (durum tutmayı gerektirir). Ve `_price_past_crt_mid`
  (invalidation) setup **oluşturulmadan** kontrol edilir; daha önce geçerli bir
  retest/fill olmuş olsa bile setup doğmaz (#4). Bkz. US100 1H-5M vakası.
4. Fill öncesi iptaller: CRT %60 geçildi (`invalidated`), TP'ye ulaşıldı
   (`missed`), SL geçildi (`past_sl`), retest anında skor < 7 (`missed_quality`).

**Kronoloji / geçmiş fill (backfill)**: CRT %60 kuralı bir stop değil, *"fiyat
bizi almadan kaçtı"* kontrolüdür — yalnızca emir dolmamışken anlamlıdır. Motor
`fill_ts` (CISD sonrası entry'ye değen ilk kapanmış LTF mumu, önce SL'ye
gitmemiş) ile `breach_ts` (%60'ı geçen ilk kapanış) zamanlarını karşılaştırır:

- `fill_ts <= breach_ts` (veya ihlal yok) → limit emri dolmuş sayılır; sinyal
  **doğrudan `active`** doğar (`entry_filled_time = fill_ts`), sonra
  `_replay_after_fill` fill'den bugüne LTF mumlarını `manage_symbol_on_price`'tan
  geçirir (aradaki TP/SL/BE/trail yakalanır) ve Telegram aktif mesajı gider.
- Aksi (önce ihlal, sonra dönüş) → `invalidated` (eski davranış).

Bayat emir dirilmesin diye pencere sınırlı: `max_backfill_fill_bars` (6) —
fill'den sonra en fazla 6 kapanmış LTF mumu olabilir (5m→30dk, 15m→90dk,
1h→6sa). `require_c2_closed` olan stratejilerde fill C2 kapanışından önceyse
sayılmaz. Pratikte yalnızca **uygulama kapalıyken/restart sonrası** ve **bir gate
sonradan açıldığında** devreye girer; normal akışta `waiting_entry` zaten canlı
yazılıyor.
5. Hard filter: **1D bias hizası** (`REQUIRE_HTF_BIAS_ALIGN`) — LONG için
   BULLISH/NEUTRAL, SHORT için BEARISH/NEUTRAL. 1W yalnızca bilgi/skor.
6. Kalite skoru (`_calc_live_setup_bias`): baz (doğru C2 rengi +2 / doji +1) +
   1D hiza veya reversal-at-PD (+2 / karşı −2) + 1W uyum (+1) + C2 kapalı (+1) +
   PD major (+1) + PD aylık (+1) + HTF FVG/OB (+1) + purge wick (+1) + LTF IFVG
   (+1). Tavan 9. **SMT divergence** korele pariteyle bulunursa +2 → max 11 =
   "Premium". **Skor < 7 açılmaz** (`MIN_QUALITY_SCORE`).
7. Kripto aynı yön küme limiti: açık + son N saatte 2 (BTC/ETH muaf).
8. Min RR: tüm marketler 2.0 (`_min_rr_for_market`).
9. Aşırı dar stop: `stop_dist < min_stop_range_mult (1.0) × ort. LTF range` →
   `tight_stop`. Ölçülen şey **entry ↔ C2 ucu mesafesi** (SL'nin yeri değil —
   SL her zaman `purge_extreme`). Eşik `113d4a0`'da SL tamponuyla birlikte
   1.0→1.5 çıkarılmıştı; tampon kaldırılınca (#7) kalibrasyon bozuldu ve en dar
   stoplu = en yüksek RR'li setup'ları eliyordu, tampon öncesi 1.0'a döndü.

### Radar

`scanner._RADAR` bellek içi sözlük `(strategy, display_symbol) → durum`.
`/api/radar?tf=` bunu okur; `RADAR_STATE_META` (main.py) etiket/renk/sıra verir.
`/radar` sayfası 30 sn'de bir poll eder. State'ler: `waiting`, `c2_open`,
`no_cisd`, `low_rr`, `tight_stop`, `missed`, `missed_quality`, `invalidated`,
`past_sl`, `stale`, `bias_mismatch`, `cluster_limit`, `has_open`, `corr_open`,
`duplicate`, `low_quality`, `no_setup`, `no_data`.

## Sayfalar / API

- `/` — dashboard, kripto vs global (fx/index/metal/oil) ayrı istatistik blokları
- `/signals`, `/signals/crypto`, `/signals/global` — filtre + tab (all/active/waiting/closed) + sayfalama
- `/analytics` — Chart.js (CDN), equity curve + dağılımlar + haftalık + top semboller
- `/logs` — `EventLog` tablosu
- `/radar` + `/api/radar` — canlı radar
- `/scanner` — manuel `/api/scan`, scheduler durumu, Telegram testi, taranan semboller
- `/api/scan-status`, `/api/recalc-scores`, `/api/telegram-test`

## Bilinen açık konular

Devralınan kapalı kod parçaları — ayrıntı ve karar notları [TODO.md](TODO.md)
madde 4'te:

- `crt_engine.py` içinde strong-close margin CISD onayında **kapalı** (FX'te
  onayı geciktirip CRT %60 ile çatışıyordu).
- `color_opposite` / "same color" eski hard filter yorum satırında; artık skorda
  baz +1/+2 farkı olarak var.
- `.cursor/rules/acik-sinyal-duzeltmeleri.mdc`: NZDUSD 1D sahte-trail ve GBPCHF
  MSS/CISD notları (ikisi de kodda düzeltilmiş görünüyor).

### US100 1H-5M LONG incelemesi (2026-09-08)

Radar'da "CRT 60% crossed" (`invalidated`) görünen, kullanıcının "active olmalıydı"
dediği vaka. C1=16:00, purge=17:00 TSİ, TP 29655, 60% seviyesi 29569.

**Bulgular (replay):**
1. Motor CISD'yi **29501.57**'ye koymuştu (16:50 kırmızı mumun açılışı). Doğrusu
   likiditeyi süpüren **17:00 mumunun açılışı = 29485.74**. → **#6 düzeltildi.**
2. 1.39 puanlık minik IFVG (29470.76) gürültüydü, seçiliyordu. → **#2: min boşluk
   filtresi (`MIN_IFVG_GAP_RANGE_FRAC`) eklendi.**
3. SL tamponu (1.5× 5m range ≈ 28pt) RR'yi 2.46 → 1.74'e düşürüyordu. → **#7: SL
   tamponu kaldırıldı.**

4. **1H'te `require_c2_closed` kaldırıldı** (+ skor-7 kapısındaki C2 şartı buna
   bağlandı). Replay: 17:20'de `waiting_entry`, 17:25 retest → fill.

Bu dört düzeltmeyle US100 CISD entry 29485.74 / SL 29416.80 / RR **2.46** ve
17:25'te dolarak `active` oluyor (SL hiç test edilmedi; 20:00'de trail armed).

5. **#1: IFVG-RR koruması** — `_maybe_ifvg_entry` artık IFVG'yi ancak RR'yi
   kötüleştirmiyorsa kullanıyor. Tüm evrende 6 setup `ifvg → cisd`'ye döndü,
   hepsi daha iyi RR ile (INJ 4h 1.25→3.34, BNB 4h 1.43→2.03, APT 4h 0.37→3.10,
   NZDUSD 1d 0.86→1.62, UNI 4h 0.34→1.25, LDO 4h 0.69→1.70). Regresyon yok;
   IFVG gerçekten daha iyiyken korunuyor (XAU 1h 1.95, XMR 4h 4.34).

6. **#4: kronoloji / backfill** — `fill_ts <= breach_ts` ise sinyal `active`
   doğuyor, `max_backfill_fill_bars` (6) penceresiyle sınırlı. Bkz. yaşam
   döngüsü bölümü.

**Açık kalan:** bkz. [TODO.md](TODO.md).
