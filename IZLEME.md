# İzleme

Kararı **verilmiş ve canlıya alınmış** konular. Burada duruyorlar çünkü etkileri
zamanla ölçülecek ve gerekirse geri dönülecek. Her başlıkta **tetikleyici**
(ne olursa buraya dön) ve **nereden ölçülür** yazılı.

Yapılacak işler için → [TODO.md](TODO.md)

Son güncelleme: 2026-09-09

**Ana ölçüm aracı:** `/logs` → **Engine Report** sekmesi
(terminal karşılığı `python scripts/logstat.py --hours 24`)

---

## 1. Genel: 2026-09-08/09 düzeltmelerinin etkisi

Dokuz motor düzeltmesi canlıya alındı ve loglama kuruldu. Düzeltmeler öncesi
radarda **hiç `waiting` sinyal yoktu**, hepsi `low_rr`'de takılıydı.

**İzlenecekler:**

- Üretilen `waiting` sinyal sayısı ve kalitesi
- `SKIPPED (...)` dağılımının zaman içinde değişimi
- 1H-5M'de C2 kapanmadan fill'in gerçek etkisi (daha çok fırsat mı, daha çok
  −1R mi?)
- `BACKFILL FILL` hiç tetikliyor mu → tetiklemiyorsa TODO'daki
  **"IFVG fallback"** maddesi gereksiz demektir
- Tamponsuz SL'in stop-out oranı (SL tamponu 09.08'de kaldırıldı)
- 4H'te yeni BE eşiği (TP %50) işlemleri erken boğuyor mu

---

## 2. 1D BIAS — bayatlık düzeltmesi

**✅ Yapıldı (2026-09-09):** `STRUCTURE_STALE_DAYS = 7`. Yapısal kırılım 7 kapalı
günden eskiyse structure yön belirtmiyor, karar taze okumaya (ict) bırakılıyor.

XAUUSD artık **BEARISH** (eskiden NEUTRAL). Evrende yönlü bias **18/26 → 20/26**.

### Nasıl çalışıyor

`compute_daily_bias` iki bileşenden:
- **structure** (`htf_bias_with_age`) — son kapanışın kırdığı teyitli swing
  high/low yönü. Kalıcı durum, o yüzden **yaşı** da dönüyor.
- **ict** (`compute_ict_bias`) — yalnızca son 2 günlük mum. Çok reaktif.

Kırılım taze (≤7 gün) → ikisi aynı yön olmalı, yoksa NEUTRAL.
Kırılım bayat → structure susar, ict karar verir.

### Ölçülen varyantlar (26 sembol, 09.09.2026)

| Kural | XAUUSD | Yönlü | NEUTRAL |
|---|---|---|---|
| Eski (koşulsuz AND) | NEUTRAL | 18/26 | 8 |
| (a) bayat → NEUTRAL, yine AND | NEUTRAL ❌ | 15/26 ⬇ | 11 |
| **(a′) bayat → ict'ye düş** ✅ | **BEARISH** | **20/26** | 6 |
| (c) structure/ict/weekly çoğunluğu | BEARISH | 25/26 | 1 |

Düz (a) reddedildi: `NEUTRAL AND BEARISH` yine NEUTRAL verdiği için XAU
düzelmiyordu, üstüne structure'ı bayat ama ict ile hemfikir olan
ETH/US100/USDCAD gibi doğru okumaları da öldürüyordu.

### ⚠️ Kalan sorun: ict tek-mumluk

Bayatlık düzeltmesi **07.09'daki vetoyu engellemezdi**. O gün structure BULLISH
(bayat) **ve ict de BULLISH**'ti — 07.09 mumu önceki low'u süpürüp içeri
kapattı, bu kitaba göre "failed low" yani boğa sinyali:

```
SKIPPED (BIAS): XAUUSD SHORT filter=BULLISH daily=BULLISH weekly=BEARISH structure=BULLISH
```

İkisi de aynı yönde olduğu için hiçbir varyant farklı sonuç vermezdi. Motor
"yanlış" değildi; piyasa tek günlük boğa sinyali verip ertesi gün sert döndü.
Ama **weekly BEARISH'ti ve doğruydu** — weekly karara girmiyor.

### Hâlâ değerlendirilebilir

| | Fikir | Ölçülen etki |
|---|---|---|
| c | **weekly'yi karara kat** (üç bileşen çoğunluğu) | 25/26 yönlü — çok iddialı; NEUTRAL izin verdiği için *daha çok* bloklama demek |
| d | **1D PD array / FVG'yi bias'a kat** — 1D IFVG / bearish FVG bugün bias'ta hiç kullanılmıyor, yalnızca setup skorunda | En büyük iş |
| e | Bias'ı hard filtre olmaktan çıkar, yalnızca skora bırak | Radikal; `REQUIRE_HTF_BIAS_ALIGN = False` |

### Neden önemli

Bias **elemelerin ~%30'unu** tek başına yapıyor — en sık tetikleyen kapı.
NEUTRAL **bloklamaz** (`_bias_aligned` iki yöne de izin verir) ama
`_calc_live_setup_bias` içinde `htf_score` **+2 yerine 0** olur; 9 tavanlı,
7 eşikli skorda belirleyici.

**Tetikleyici:** rapordaki `structure ↔ ict` ayrışma oranı kalıcı olarak
**>%40** olursa, ya da bias yine en sık eleme sebebi olup gözle bakınca yön
yanlış görünürse.

**Ölçüm:** Engine Report → "1D Bias takibi". Log satırı yapısal yaşı taşıyor:
`structure=BULLISH(14)`.

---

## 3. Kripto küme limiti

**✅ Yapıldı (2026-09-09):** `CLUSTER_EXEMPT_MIN_SCORE = 9` — skoru ≥9 olan
kripto setup'lar aynı-yön küme limitine takılmıyor.

**Mevcut ayar** (`app/scanner.py`):

| Parametre | Değer | Yer |
|---|---|---|
| `cluster_open` | 2 | `_RISK_GATES` |
| `cluster_recent` | 2 | `_RISK_GATES` |
| `cluster_window_hours` | 4 (1D'de 24) | `_RISK_GATES` / `STRATEGY_1D` |
| `CLUSTER_EXEMPT_SYMBOLS` | BTC, ETH | modül sabiti |
| `CLUSTER_EXEMPT_MIN_SCORE` | **9** | modül sabiti |

Yalnızca **kripto**; FX/metal/oil/index bu limite hiç dahil değil.

**Neden 9:** ETH/LINK/SUI SHORT açılınca kota doldu ve ADA (RR 4.47), BNB, TAO,
XRP — **hepsi tam 9** — bloklandı. Eşik 10 olsaydı 40 setup'ta yalnızca 1'i
geçerdi (SMT'siz skor tavanı 9, SMT +2 ekliyor).

Muaf setup limiti **deler ama sayıma dâhildir** (BTC/ETH gibi sayımdan çıkmaz)
— yoksa SMT'li bir piyasa hareketinde sınırsız korele pozisyon açılırdı.

**Tetikleyici:**
- Aynı yönde eşzamanlı açık kripto sinyal sayısı rahatsız edici olursa
- Aynı 4 saatlik pencerede çok sayıda korele sinyal gelirse
- Raporda `CLUSTER` elemesi neredeyse sıfırlanırsa (= limit artık hiçbir şey
  yapmıyor demektir)

**Kısabileceğimiz kollar:** `CLUSTER_EXEMPT_MIN_SCORE`'u 10'a çekmek · muaf
setup'a ayrı bir tavan koymak · `cluster_window_hours`'ı kısaltmak ·
`cluster_open`/`cluster_recent`'ı ayrıştırmak.

**Ölçüm:** Engine Report → "Setup neden sinyale dönüşmedi" içinde `CLUSTER`
satırı, ve aynı anda açık sinyallerin yön dağılımı.

---

## 4. Trail arm eşiği %90 — kazananları kesiyor muydu?

**✅ Yapıldı (2026-09-09):** `trail_arm_tp_fraction` **0.75 → 0.90** (üç
stratejide de; 1D/1H'teki gereksiz override'lar kaldırıldı).

**Neden:** trail mesafesi `max(1R, 1.3 × ort. LTF range)` tipik olarak **1–1.5R**.
%75 arm'da hedefe kalan yol 3R'lik bir işlemde yalnızca **0.75R** — yani stop,
kalan mesafeden daha geniş. Fiyatın TP'ye varması için, trail mesafesinden daha
küçük bir geri çekilme yapması gerekiyor; bu yarış yapısal olarak kaybediliyor.

**ARB 4H LONG (09.09) — ölçülen:**

| Varyant | Sonuç |
|---|---|
| arm %75 (eski) | TRAIL **+1.13R** @ 08:30 |
| arm %75, offset 0.5R | TRAIL +2.12R |
| **arm %85 / %90** ✅ | **TP +3.16R** @ 09:00 |
| trail yok (sadece BE) | TP +3.16R |

ARB'nin MFE'si TP yolunun **%82.6**'sına çıktı — %75 arm'ı tetikledi, %85+'ı
tetiklemedi. Fiyat trail çıkışından 30 dk sonra TP'yi vurup **+5.63R**'ye gitti.
Offset'i genişletmek işe yaramadı (+0.91R): geri çekilme yine yetti ve daha az
kilitledi.

SUI (08.09) her varyantta BE — fiyat sonradan entry'ye döndü, trail ayarı
sonucu değiştirmiyor.

**Tetikleyici:** raporda `trail` çıkışlarının ortalama R'si düşükse (<1R) veya
`TP` çıkış sayısına göre `trail` çıkışları çok fazlaysa. Ters yönde: %90 fazla
gevşek kalıp işlemler +2R'den 0R'ye (BE) dönüyorsa eşik geri çekilmeli.

**Ölçüm:** Engine Report → "Çıkış türü" (TP / trail / BE / SL) ve
"Trail çıkış ort. R".

---

## 5. 1D/1H'te BE açılsın mı?

**Bağlam:** SUI 4H SHORT (08.09), MFE +1.48R'ye gitti ama trail TP %50'de
(+1.356R) açılıp 1R geride durduğu için sadece ~0.36R kilitliyordu; ilk geri
çekilme işlemi **+0.48R**'de kesti (planned 2.71R).

**✅ Yapıldı (4H):** BE → TP %50 (`be_arm_tp_fraction`), sabit R
tetikleyicileri (`be_arm_r`, `trail_arm_r`) kapatıldı. SUI verisiyle
doğrulandı: bu ayarla trail açılmaz, SL entry'de kalır. (Trail eşiği sonradan %90'a çekildi — bkz. bir üstteki madde.)

### Kalan karar

1D/1H'te BE **kapalı** (`be_arm_tp_fraction: None`). Kasıtlı — koddaki gerekçe:

```python
# 1D: 1s range 1R'yi yer (NZDUSD +1.31R sahte trail). Eski %75.
```

1 saatlik mumun boyu 1R'ye yakın olduğu için SL'yi entry'ye çekmek iğneye açık
hale getiriyor. BE'yi geç tetiklemek bunu **çözmez** — arm zamanı değişir, stop
yine tam entry'de durur.

`min_stop_range_mult: 1.0` artık 1R ≥ 1 ortalama LTF mumu garanti ediyor (ama
*ancak* eşit), ve aynı-mum MFE koruması var. Yine de risk gerçek.

**Tetikleyici:** 4H'te BE @ TP %50 yeterli veri biriktirince karar ver.
**Ölçülecek:** BE tetiklenen işlemlerin kaçı 0R'de kapandı, kaçı TP'ye yürüdü.

**Ölçüm:** Engine Report → "Çıkış türü" (TP / trail / BE / SL) ve
"Trail çıkış ort. R".

---

## 6. Not

Kısmi kâr alma (%50'de yarı kapat + BE) fikri bir **yapılacak iş**, izleme
konusu değil — [TODO.md](TODO.md) içinde.
