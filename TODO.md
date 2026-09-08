# Yapılacaklar

Tamamlanan madde bu dosyadan **silinir**. Her madde, yeniden araştırma
gerektirmeyecek kadar bağlam içerir.

Son güncelleme: 2026-09-08

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

## 2. `TARGET TAKEN` filtresi fazla katı mı? (ticaret kararı)

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

## 3. Kripto cluster limiti fazla kısıtlayıcı mı? (ticaret kararı)

**Gözlem (2026-09-08):** ETH/LINK/SUI SHORT açılınca 4 saatlik pencerede kota
doldu ve **4 setup daha bloklandı** — hepsi `open=1/2 recent=2/2` ile:

| Sembol | Skor | RR |
|---|---|---|
| ADA | 9 | 4.47 |
| BNB | 9 | 2.03 |
| TAO | 9 | 1.42 |
| XRP | 8 | — |

**Mevcut ayar** (`_RISK_GATES` / `STRATEGY_CFG`, `app/scanner.py`):
`cluster_open: 2`, `cluster_recent: 2`, `cluster_window_hours: 4` (1D'de 24).
Yalnızca kripto; BTC/ETH muaf (`CLUSTER_EXEMPT_SYMBOLS`).

**Karar gereken:** Piyasa geneli tek yöne giderken (ki CRT'de sık olur) bu limit
en kaliteli setup'ları da kesiyor. Korelasyon riski gerçek — ama 2 çok mu az?
Alternatif: limiti skora bağlamak (skor 9+ muaf) veya pencereyi kısaltmak.

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
- `BACKFILL FILL` hiç tetikliyor mu (tetiklemiyorsa madde 1'e gerek yok)
- Tamponsuz SL'in stop-out oranı

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
