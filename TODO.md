# Yapılacaklar

Tamamlanan madde bu dosyadan **silinir** — kararı verilmiş ama ileride tekrar
bakılacak konular "açık gözden geçirme" olarak kalır (tetikleyicisiyle birlikte).
Her madde, yeniden araştırma gerektirmeyecek kadar bağlam içerir. Çapraz
referanslar numara değil **isim** kullanır; numaralar madde silindikçe kayıyor.

Son güncelleme: 2026-09-09

---

## 1. IFVG dolmazsa CISD/MSS'e fallback

**Durum:** #1'in yarısı yapıldı (IFVG, CISD'den kötü RR veriyorsa artık
seçilmiyor). Kalan yarı bu.

**Sorun:** IFVG entry modeli seçildikten sonra fiyat IFVG bölgesine hiç
gelmezse (etiketlemezse) sinyal orada asılı kalır ve fiyat CRT %60'ı geçince
`invalidated` olur. CISD/MSS entry'sine geri dönüş yok — oysa fiyat oraya
retest vermiş olabilir.

**Yapılacak:** Bekleyen sinyalde "IFVG N mumdur dolmadı → entry'yi CISD'ye
çevir" mantığı. Durum tutmayı gerektiriyor:
- ya `Signal`'a yeni kolon (+ `app/database.py` `_MIGRATIONS` satırı)
- ya `cisd_time`'dan mum sayarak türetme

**Maliyet tahmini:** ~0,3–0,5M token (migration dahil).

**Not:** En seyrek tetiklenen madde. Canlı loglarda gerçekten ne sıklıkta
olduğunu görmeden yapmaya değmeyebilir.

---

## 2. 1D BIAS hesabı — yapısal bileşen bayatlıyor (öncelikli)

**Bu filtre elemelerin ~%30'unu tek başına yapıyor** — en sık tetikleyen kapı.
Ve yanlış çalıştığında hem sinyali bloklar hem kalite puanını düşürür.

### Nasıl çalışıyor

`compute_daily_bias` = **`compute_htf_bias` (structure) VE `compute_ict_bias`**
aynı yönde ise o yön, değilse **NEUTRAL**.

- **structure** — son kapanışın kırdığı teyitli swing high/low yönü. **Kalıcı
  durum**: ters bir kırılım olana kadar eski yönü taşır. Bayatlama sınırı YOK.
- **ict** — yalnızca **son 2 mum**: close > önceki high → BULLISH, close <
  önceki low → BEARISH, failed high/low → ters yön, inside/outside → NEUTRAL.
  Çok reaktif, her gün dönebilir.

Yani **yavaş + hızlı** iki sinyalin AND'i alınıyor.

### Ölçüm (2026-09-09, 1D-1H evreni, 26 sembol)

| | |
|---|---|
| Yönlü `daily` bias | 18/26 |
| **NEUTRAL** | **8/26 (%31)** |
| Yapısal kırılım yaşı | ortanca **5 gün**, max **17 gün** |
| ≥10 gün bayat | 6/26 (%23) |

### XAUUSD vakası (kullanıcının tespiti — doğrulandı)

```
07.09  O4426.65 H4433.66 L4385.14 C4427.16
08.09  O4427.16 H4447.39 L4350.14 C4353.73   close < önceki LOW
```
- **ict = BEARISH** ✅ (Sept 8 kapanışı Sept 7 low'unun altında — kullanıcının
  gözlemi motorda doğru okunuyor)
- **structure = BULLISH** — ama kırılım **25.08**, yani **14 gün önce**. Altın
  o tarihten beri 4287–4700 aralığında; yeni kırılım olmadığı için eski boğa
  yönü duruyor.
- **daily = NEUTRAL**

### İki ayrı zarar

1. **NEUTRAL, sinyali bloklamaz** (`_bias_aligned` NEUTRAL'de `True` döner) ama
   `_calc_live_setup_bias` içinde `htf_score` **+2 yerine 0** olur. 9 tavanlı,
   7 eşikli bir skorda bu belirleyici.
2. **Daha kötüsü:** bayat structure + tek günlük ict sapması *yanlış yönde*
   `daily` üretebiliyor ve o zaman **hard blok** olur. 08.09 logu:
   ```
   SKIPPED (BIAS): XAUUSD SHORT filter=BULLISH daily=BULLISH weekly=BEARISH structure=BULLISH
   ```
   O gün ict, 07.09'un "failed low" dalından BULLISH gelmişti; structure zaten
   bayat BULLISH'ti → geçerli bir short vetolandı. **weekly BEARISH'ti ama
   weekly filtre olarak kullanılmıyor.**

### Değerlendirilecek seçenekler (karar senin)

| | Fikir | Not |
|---|---|---|
| a | **structure'a bayatlama sınırı**: kırılım N günden eskiyse NEUTRAL say | En küçük değişiklik; sadece bayat vetoyu kaldırır |
| b | **AND yerine ağırlık**: ict ve structure ayrışırsa NEUTRAL yerine daha taze olanı tercih et | Daha çok yönlü bias, daha çok sinyal |
| c | **weekly'yi karara kat** — bugün yalnızca bilgi. XAU'da weekly doğruydu | 3 bileşenin çoğunluğu? |
| d | **1D PD array / FVG'yi bias'a kat** — kullanıcının bahsettiği IFVG + bearish FVG iğnesi bugün bias'ta hiç kullanılmıyor (yalnızca setup skorunda) | En büyük iş |
| e | Bias'ı hard filtre olmaktan çıkar, yalnızca skora bırak | Radikal; `REQUIRE_HTF_BIAS_ALIGN = False` |

### Takip (kuruldu)

`SKIPPED (BIAS)` log satırına **`ict=`** eklendi; artık NEUTRAL'in sebebi
görülebiliyor. Rapor sayfasında **"1D Bias takibi"** bölümü var:
`/logs` → Engine Report → bias ile eleme sayısı, `structure ↔ ict` ayrışma
oranı, `weekly ≠ daily` sayısı ve son örnekler.

**İzlenecek:** ayrışma oranı kalıcı olarak yüksekse (>%40) AND kuralı fazla
katı demektir.

---

## 3. `TARGET TAKEN` filtresi fazla katı mı? (ticaret kararı)

**Gözlem (2026-09-08 taraması):** 62 setup'ın **10'unda** tetikledi — en sık
üçüncü filtre.

**Ne yapıyor:** `detect_crt_setup` içinde `target_consumed` — C1'in hedef (TP)
tarafı, C1'den sonraki bir mum tarafından zaten süpürülmüşse setup geçersiz
sayılıyor (hedef likidite tüketilmiş).

- SHORT (TP = C1 low): C1'den sonra bir mumun low'u C1 low altına inmişse
- LONG (TP = C1 high): C1'den sonra bir mumun high'ı C1 high üstüne çıkmışsa

**Karar gereken:** Meşru bir ICT kuralı, ama bu kadar sık tetiklemesi normal mi?
Örneğin "tamamen süpürülmüş" yerine "%X'inden fazlası süpürülmüş" gibi bir
tolerans anlamlı olur mu?

---

## 4. Signals + Radar sayfalarına "Entry Model" sütunu

**İstenen:** Her sinyalde girişin hangi modelden geldiği görünsün: `IFVG`,
`CISD`, `MSS`.

**⚠️ Bu salt UI işi değil — motor şu an MSS'i ayrı kaydetmiyor.**
`Signal.entry_model` yalnızca iki değer alıyor: `"cisd"` | `"ifvg"`.
`check_cisd_confirmation` içinde iki aday var (`app/crt_engine.py`):

- **CISD** = purge öncesi düşüş/yükseliş bloğunun ilk mumunun açılışı
- **MSS** = purge öncesi son swing high/low

`_pick_wider_stop` hangisi daha iyi RR veriyorsa onu seçiyor **ama hangisinin
kazandığını döndürmüyor** — `CISDConfirmation.entry_model` dataclass
varsayılanı olarak her hâlükârda `"cisd"` kalıyor. Yani bugün "cisd" yazan
kayıtların bir kısmı aslında MSS.

**Yapılacaklar:**

1. `_pick_wider_stop` kazanan adayı da döndürsün (`"cisd"` / `"mss"`).
2. `_check_bullish_cisd` / `_check_bearish_cisd` bunu
   `CISDConfirmation.entry_model`'e yazsın. `_maybe_ifvg_entry` zaten IFVG
   seçilince `"ifvg"` ile eziyor — sıralama korunmalı.
3. **Signals sayfası** (`app/templates/signals.html`): yeni sütun.
   **Karar verildi: IFVG sütunu KALIYOR.** İki sütun iki farklı soruyu
   cevaplıyor:
   - **IFVG** = setup'ta IFVG *var mı?* (varlık)
   - **Entry Model** = giriş *hangi modelden geldi?* (IFVG / CISD / MSS)

   #1 düzeltmesinden sonra bunlar gerçekten ayrışıyor: IFVG var ama RR'yi
   kötüleştirdiği için kullanılmamış olabilir → `IFVG ✓` + `Entry Model: CISD`.
4. **Radar** (`app/templates/radar.html` + `/api/radar`): radar entry model'i
   hiç taşımıyor. Zincir: `_preview_trade_levels` dönüşüne `model` eklensin →
   `_set_radar(...)` yeni parametre → `api_radar` (`app/main.py`) JSON'a koysun
   → template'te rozet. Radar'daki mevcut `ifvg` alanı da bool.

**Migration gerekmiyor** — `entry_model` kolonu DB'de mevcut. Ama geçmiş
kayıtlar geriye dönük düzelmez (hepsi `"cisd"` yazıyor); istenirse
`/api/recalc-scores` benzeri bir yeniden hesaplama gerekir.

### ⚠️ Bonus hata: mevcut IFVG sütunu "var mı"yı doğru göstermiyor

Kullanıcı bu sütunu "setup'ta IFVG var mı" diye okuyor, ama her iki kaynak da
**"IFVG var VE kullanılabilir"** anlamına geliyor:

- `_maybe_ifvg_entry` (signals kaynağı, `app/scanner.py`):
  ```python
  if not allow_ifvg:
      return cisd, planned      # detect_ltf_ifvg HİÇ çağrılmıyor
  ...
  cisd.ifvg_low = zone.low      # ancak buraya gelirse yazılıyor
  ```
- `_preview_trade_levels` (radar kaynağı): zone tespit ediliyor ama
  `ifvg = zone is not None and allow_ifvg` ile AND'leniyor.

`_ifvg_allowed` C2 kapanmamışken (çok yaygın — `c2_open` / `no_cisd` durumları)
veya skor tam 7 iken `False` döner. Yani **C2 formasyondayken IFVG sütunu, IFVG
fiilen var olsa bile hep "-" gösteriyor.**

**Düzeltme:** varlık tespitini `allow_ifvg`'den ayır — zone'u her hâlükârda
tespit et, `allow_ifvg`'yi yalnızca *entry olarak kullanma* kararında kullan.
`_maybe_ifvg_entry`'de erken `return`'ü zone tespitinden sonraya al.

**Maliyet notu:** bu, IFVG'nin izinli olmadığı her setup'ta bir ek
`detect_ltf_ifvg` çağrısı demek. Sıcak yolda; `_cpu()` ile thread'e atılıyor,
muhtemelen sorun değil ama ölçmeye değer.

---

## 5. Eski açık başlıklar (devralınan)

Bunlar 2026-09-08 oturumundan önce de açıktı; hâlâ geçerli mi teyit edilmeli.

**a) CISD onayında `strong_close_margin` kapalı**
`app/crt_engine.py` `_check_bullish_cisd` / `_check_bearish_cisd` içinde MSS
onayı için "güçlü kapanış marjı" kontrolü yorum satırında:
```python
# margin = _strong_break_margin(work, i, break_level)
# if float(cur["close"]) > break_level + margin:
if float(cur["close"]) > break_level:
```
Gerekçe (kod yorumu): FX'te onayı geciktirip CRT %60 ile çatışıyordu.
Sabitler duruyor: `CISD_STRONG_CLOSE_PCT`, `CISD_STRONG_CLOSE_RANGE_MULT`,
`CISD_MARGIN_LOOKBACK`. Kalıcı olarak silinsin mi, yoksa geri mi açılsın?

**b) `color_opposite` / "same color" hard filter kapalı**
`app/scanner.py` içinde yorum satırında; artık skorda baz +2 yerine +1 farkı
olarak yaşıyor (`_calc_live_setup_bias`). Radar state'i `same_color` da
`RADAR_STATE_META` içinde yorumda. Temizlensin mi?

**c) `.cursor/rules/acik-sinyal-duzeltmeleri.mdc`**
NZDUSD 1D sahte-trail ve GBPCHF MSS/CISD notları. İkisi de kodda düzeltilmiş
görünüyor (1D/1H trail %75 + aynı-mum MFE koruması; doji artık blok bölmüyor).
Doğrulanıp bu dosya silinebilir.

---

## 6. İzleme (madde değil, süreç)

2026-09-08'de 7 düzeltme canlıya alındı ve loglama kuruldu. Bir süre
`logs/traderadar.log` izlenmeli:

- Üretilen `waiting` sinyal sayısı ve kalitesi (düzeltmeler öncesi **sıfırdı**)
- `SKIPPED (...)` dağılımının zaman içinde değişimi
- 1H-5M'de C2 kapanmadan fill'in gerçek etkisi (daha çok fırsat mı, daha çok
  −1R mi?)
- `BACKFILL FILL` hiç tetikliyor mu (tetiklemiyorsa "IFVG fallback" gereksiz)
- Tamponsuz SL'in stop-out oranı
- Kripto küme limiti — bkz. bir sonraki madde

---

## 7. Kripto küme limiti — gerekirse yeniden ayarla (açık gözden geçirme)

**Karar verildi, madde kapanmadı.** Sinyal akışı fazlalaşırsa güncel veriyle
tekrar bakılacak; o yüzden bağlam burada duruyor.

**Mevcut ayar** (`app/scanner.py`):

| Parametre | Değer | Yer |
|---|---|---|
| `cluster_open` | 2 | `_RISK_GATES` |
| `cluster_recent` | 2 | `_RISK_GATES` |
| `cluster_window_hours` | 4 (1D'de 24) | `_RISK_GATES` / `STRATEGY_1D` |
| `CLUSTER_EXEMPT_SYMBOLS` | BTC, ETH | modül sabiti |
| `CLUSTER_EXEMPT_MIN_SCORE` | **9** | modül sabiti |

Yalnızca **kripto**; FX/metal/oil/index bu limite hiç dahil değil.

**2026-09-09'da yapılan:** Skoru ≥9 olan setup'lar limiti delebiliyor. Sebep:
ETH/LINK/SUI SHORT açılınca kota doldu ve ADA (RR 4.47), BNB, TAO, XRP —
**hepsi tam 9** — bloklandı. Eşik 10 olsaydı 40 setup'ta yalnızca 1'i geçerdi
(SMT'siz skor tavanı 9, SMT +2 ekliyor).

Muaf setup limiti **deler ama sayıma dâhildir** (BTC/ETH gibi sayımdan
çıkmaz) — yoksa SMT'li bir piyasa hareketinde sınırsız korele pozisyon açılırdı.

**Tetikleyici — şu olursa buraya dön:**
- Aynı yönde eşzamanlı açık kripto sinyal sayısı rahatsız edici olursa
- Aynı 4 saatlik pencerede çok sayıda korele sinyal gelirse
- `logstat` raporunda `CLUSTER` elemesi neredeyse sıfırlanırsa (= limit artık
  hiçbir şey yapmıyor demektir)

**Kısabileceğimiz kollar:** `CLUSTER_EXEMPT_MIN_SCORE`'u 10'a çekmek · muaf
setup'a ayrı bir tavan koymak · `cluster_window_hours`'ı kısaltmak ·
`cluster_open`/`cluster_recent`'ı ayrıştırmak.

**Ölçüm:** `/logs` → Engine Report → "Setup neden sinyale dönüşmedi" içinde
`CLUSTER` satırı, ve aynı anda açık sinyallerin yön dağılımı.

---

## 8. 1D/1H'te BE açılsın mı? + kısmi kâr alma fikri

**Bağlam:** 2026-09-08'de SUI 4H SHORT, MFE +1.48R'ye gitti ama trail TP %50'de
(+1.356R) açılıp 1R geride durduğu için sadece ~0.36R kilitliyordu; ilk geri
çekilme işlemi **+0.48R**'de kesti (planned 2.71R). Arm eşiği ile trail mesafesi
neredeyse eşitti.

**Yapıldı (4H):** BE → TP %50, trail → TP %75, sabit R tetikleyicileri
(`be_arm_r`, `trail_arm_r`) kapatıldı. Yeni anahtar `be_arm_tp_fraction`.
SUI verisiyle doğrulandı: trail açılmaz, SL entry'de kalır, **işlem açık kalırdı.**

### Kalan karar: 1D/1H'te BE

Şu an **kapalı** (`be_arm_tp_fraction: None`). Kasıtlı — koddaki gerekçe:

```python
# 1D: 1s range 1R'yi yer (NZDUSD +1.31R sahte trail). Eski %75.
```

1 saatlik mumun boyu 1R'ye yakın olduğu için SL'yi entry'ye çekmek iğneye açık
hale getiriyor. BE'yi geç tetiklemek bunu **çözmez** — armanma zamanı değişir,
stop yine tam entry'de durur.

`min_stop_range_mult: 1.0` artık 1R ≥ 1 ortalama LTF mumu garanti ediyor (ama
*ancak* eşit), ve aynı-mum MFE koruması var. Yine de risk gerçek.

**Karar:** 4H'te BE @ TP %50'nin sonucunu bir süre izle, sonra 1D/1H'e de
açılsın mı karar ver. Ölçülecek: BE tetiklenen işlemlerin kaçı 0R'de kapandı,
kaçı TP'ye yürüdü.

### Yeni fikir: %50'de BE + işlemin yarısını kapat

TP yolunun %50'sine gelince:
1. **İşlemin yarısını kapat** → kâr realize edilir
2. Kalan yarı devam eder, **SL = BE (entry)**

Böylece en kötü senaryo "yarım pozisyondan alınan kâr + kalan yarıda 0R" olur;
SUI gibi vakalarda hem kâr cebe girer hem de TP'ye yürüme ihtimali korunur.

**Gereken altyapı (bugün yok):** sistem pozisyon büyüklüğü tutmuyor. `Signal`
tek bir `rr_value` yazıyor; kısmi çıkış için en az şunlar gerekir:
- `Signal`'a kısım büyüklüğü / kısmi çıkış fiyatı + zamanı kolonları
  (+ `_MIGRATIONS`)
- `rr_value` hesabının ağırlıklı hale gelmesi
  (`0.5 × partial_R + 0.5 × final_R`)
- Dashboard/analytics R toplamlarının bu ağırlığa uyması
- Telegram mesajında kısmi çıkış bildirimi
- `partial_hit` alanı bugün "BE aktif" anlamında kullanılıyor; adı doğru ama
  anlamı değişecek, karışıklığa dikkat

---

## 9. Mimari analizi: tek kullanıcılık bir sistem için web yapısı doğru mu?

> **Öncelik: EN DÜŞÜK.** Acelesi yok, diğer maddelerin hepsi bitince bakılacak.
> **Çıktı bir rapor/öneri**, doğrudan refactor değil.

**Soru:** Sistem web sitesi olarak tasarlandı ama tek kullanıcı var (sen).
Farklı bir yapı daha mı uygun?

### Web olduğu için var olan, tek kullanıcıda karşılığı olmayan şeyler

- JWT + cookie + login sayfası (`app/auth.py`, `templates/login.html`) —
  `verify_credentials` zaten `.env`'deki tek kullanıcıyla düz string
  karşılaştırması
- Tailwind derleme adımı (`tools/tailwindcss.exe` → `app/static/css/app.css`)
- Sayfalama (sinyaller 20/sayfa, loglar 50/sayfa) — veri seti çok küçük
- `/api/*` katmanı, yalnızca kendi template'leri besliyor

### Asıl mimari koku (analizin ana konusu)

**Trade motoru web sunucusunun `lifespan`'i içinde yaşıyor**
(`app/main.py` → `market_data.start()`). Yani:

- Uvicorn yeniden başlarsa/çökerse **WS ölür, sinyal tespiti durur**
- Motorun çalışma süresi, sadece izlemek için var olan UI'ın çalışma süresine
  bağlı
- Bir template düzenlemesi + reload, canlı motoru etkiler

Oysa **değerli olan motor**, UI sadece bir görüntüleyici.

### Değerlendirilecek seçenekler

1. **Motoru ayrı process'e al** (en küçük değişiklik, en büyük kazanç):
   headless daemon SQLite'a yazar; web app yalnızca okur. UI çökse de motor
   çalışır. Windows Task Scheduler / NSSM ile servisleştirilebilir.
2. **Telegram-öncelikli, UI opsiyonel**: Telegram zaten aktif sinyal + sonuç
   gönderiyor. Radar da bir `/radar` komutuyla gelebilir mi? Web app yalnızca
   geçmiş/analitik için kalır.
3. **Olduğu gibi bırak**: tek kullanıcıda bu karmaşıklık zaten zarar vermiyor;
   sadece auth ve derleme adımı sadeleştirilir.
4. Masaüstü uygulaması (Tauri/Electron) — muhtemelen zahmete değmez, tamlık
   için listede.

### Analiz başlamadan cevaplanması gereken

- **UI'a başka bir cihazdan / makine dışından erişiyor musun?** Uygulama
  `0.0.0.0:8000`'e bağlanıyor, yani LAN'a açık. Cevap "hayır" ise auth ve web
  katmanının büyük kısmı gereksiz. "Evet" ise (telefondan bakmak gibi) web
  yapısı korunmalı — o zaman Telegram bunu zaten karşılıyor mu?
- Motor ile UI'ın aynı makinede kalması şart mı?

---

## Tamamlananlar (2026-09-08)

Referans için; ayrıntı [CLAUDE.md](CLAUDE.md) "US100 1H-5M LONG incelemesi".

- ~~#2 IFVG minimum boşluk (`MIN_IFVG_GAP_RANGE_FRAC = 0.15`)~~
- ~~#6 CISD bloğu süpüren mumu içeriyor~~
- ~~#7 SL tamponu kaldırıldı~~
- ~~#5 1H-5M `require_c2_closed = False`~~
- ~~#1 (yarısı) IFVG RR'yi kötüleştiriyorsa kullanılmıyor~~
- ~~#4 Invalidation kronolojisi + pencereli backfill~~
- ~~`min_stop_range_mult` 1.5 → 1.0~~
- ~~Loglama yapılandırması~~
