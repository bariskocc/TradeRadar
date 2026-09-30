"""Ekonomik takvim + haber bildirimi (ayri Telegram botu).

Motorla hic konusmaz: sinyal/skor/kapi kararlarina dokunmaz, yalniz bilgi verir.

Kaynaklar (30.09'da denendi):
- **Takvim: ForexFactory haftalik JSON feed'i** (`nfs.faireconomy.media`). Saat, renk
  (High = kirmizi, Medium = turuncu, Holiday = tatil), beklenti ve onceki deger verir.
  **Aciklanan deger (actual) YOK**; forexfactory.com sitesi de 403 (Cloudflare) donuyor.
- **Sonuc + ani haber: FinancialJuice RSS.** Veri basliklari aciklamadan saniyeler sonra
  "US Core PCE Price Index MoM Actual 0.2% (Forecast 0.3%, Previous 0.2%)" biciminde dusuyor.
  Cloudflare siki: ard arda iki istek 429 + Retry-After ~1 dk → 2 dk'da bir, Retry-After'a uyulur.

Turkce baslik / aciklama / piyasa etkisi **sabit sozluk + kural** ile yazilir (LLM yok,
kullanici karari 30.09). Ani haber basliklari Ingilizce kalir, kategoriye gore Turkce not eklenir.

Akis (TSI): 09:00 gunun ozeti (haber yoksa "onemli haber yok") · haberden 15 dk once on-bildirim
(ayni saatteki haberler tek mesaj) · aciklaninca on-bildirime reply: sonuc + etki tahmini ·
konusma/karar metni icin pencere sonunda one cikan basliklar · takvim disi onemli basliklar.
"""

from __future__ import annotations

import asyncio
import email.utils
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import (
    NEWS_ALERT_MINUTES,
    NEWS_CURRENCIES,
    NEWS_ENABLED,
    NEWS_MORNING_HOUR,
    NEWS_TELEGRAM_BOT_TOKEN,
    NEWS_TELEGRAM_CHAT_ID,
)

log = logging.getLogger(__name__)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FJ_URL = "https://www.financialjuice.com/feed.ashx?xy=rss"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

TSI = timezone(timedelta(hours=3))
UTC = timezone.utc

_TICK_SEC = 20
_CAL_REFRESH_SEC = 3600          # feed saatlik yeterli (FF sik istege kizar)
_FJ_POLL_SEC = 120               # FinancialJuice 429 sinirinin guvenli tarafi
_ALERT_BEFORE = timedelta(minutes=NEWS_ALERT_MINUTES)
_RESULT_TIMEOUT = timedelta(minutes=20)   # bu surede sonuc basligi gelmezse "bulunamadi"
_RESULT_MAX_AGE = timedelta(minutes=60)   # restart sonrasi eski sonuclari gonderme
_TALK_WINDOW = timedelta(minutes=30)      # konusma: ilk 30 dk'nin basliklari
_TEXT_WINDOW = timedelta(minutes=15)      # karar metni / tutanak
_MORNING_GRACE = timedelta(minutes=30)    # 09:00-09:30 arasi restart olsa da ozet gider
_SUDDEN_COOLDOWN = timedelta(minutes=90)  # ayni kategoride yeni mesaj yerine toplu reply
_SUDDEN_CHAIN = timedelta(hours=6)        # bu sureden sonra yeni kok mesaj
_HEADLINE_KEEP = timedelta(hours=48)

IMPACTS = {"High": ("🔴", "kırmızı"), "Medium": ("🟠", "turuncu")}
_TR_DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
_COUNTRY_TR = {"USD": "ABD", "EUR": "Euro Bölgesi", "GBP": "İngiltere", "JPY": "Japonya",
               "AUD": "Avustralya", "NZD": "Yeni Zelanda", "CAD": "Kanada", "CHF": "İsviçre",
               "CNY": "Çin"}


# ---------------------------------------------------------------------------
# Sozluk: FF basligi -> Turkce ad, anlam, yon kurali
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventInfo:
    tr: str
    meaning: str = ""
    sign: int = 0          # +1: beklentiden yuksek = USD lehine · -1: tersi · 0: yon yorumu yok
    weight: int = 1        # ayni saatteki verilerin net okumasinda agirlik
    fj: str | None = None  # data: FJ veri adi ("US " ile " Actual" arasi) · text: baslik regex'i
    kind: str = "data"     # data | oil | talk (konusma) | text (karar metni, tutanak)
    tone: str = ""         # talk/text: konusma sonrasi reply'daki okuma rehberi


_INFL = "Yüksek gelirse Fed'in faiz indirmesi zorlaşır (USD lehine)."
_EVENTS: list[tuple[str, EventInfo]] = [
    # --- istihdam ---
    (r"Non-Farm Employment Change", EventInfo(
        "Tarım Dışı İstihdam — NFP (aylık yeni iş sayısı)",
        "Tarım dışı sektörlerde bir ayda eklenen iş sayısı. ABD ekonomisinin nabzı; Fed'in faiz kararını en çok etkileyen veri.",
        +1, 3, r"^non-?farm (payrolls|employment change)(?!.*private)")),
    (r"ADP Non-Farm Employment Change", EventInfo(
        "ADP İstihdam (özel sektörde aylık yeni iş sayısı)",
        "Özel sektörde eklenen iş sayısının ADP ölçümü; Cuma'daki NFP'nin öncüsü sayılır.",
        +1, 1, r"^adp (employment change|non-?farm)")),
    (r"Unemployment Rate", EventInfo(
        "İşsizlik Oranı (iş arayıp bulamayanların oranı)",
        "İş arayıp bulamayanların oranı. Yükselmesi ekonominin soğuduğunu, Fed'in faiz indirebileceğini gösterir.",
        -1, 2, r"^unemployment rate")),
    (r"Average Hourly Earnings m/m", EventInfo(
        "Ortalama Saatlik Kazanç (aylık ücret artışı)",
        "Ücretlerdeki aylık artış. Ücret enflasyonu yüksekse Fed faizleri yüksek tutar.",
        +1, 2, r"^average (hourly )?earnings mom")),
    (r"Unemployment Claims", EventInfo(
        "Haftalık İşsizlik Başvuruları (yeni işsiz sayısı)",
        "Geçen hafta ilk kez işsizlik maaşına başvuranların sayısı. Artış iş piyasasının zayıfladığını gösterir.",
        -1, 1, r"^initial jobless claims")),
    (r"JOLTS Job Openings", EventInfo(
        "JOLTS (açık iş ilanı sayısı)",
        "İşverenlerin doldurmak istediği açık pozisyon sayısı; iş gücü talebinin göstergesi.",
        +1, 1, r"^jolts")),
    (r"Challenger Job Cuts y/y", EventInfo(
        "Challenger (şirketlerin işten çıkarma planları, yıllık)",
        "Şirketlerin açıkladığı işten çıkarma planları. Artış iş piyasası için olumsuz.",
        -1, 1, r"^challenger")),
    (r"Employment Cost Index q/q", EventInfo(
        "İstihdam Maliyet Endeksi (ücret + yan hak artışı, çeyreklik)",
        "Ücret + yan hakların çeyreklik değişimi; Fed'in izlediği ücret enflasyonu ölçüsü.",
        +1, 1, r"^employment cost")),
    # --- enflasyon ---
    (r"CPI m/m", EventInfo("TÜFE (aylık enflasyon)", "Tüketici fiyatlarındaki aylık değişim, yani enflasyon. " + _INFL,
                           +1, 3, r"^cpi mom")),
    (r"CPI y/y", EventInfo("TÜFE (yıllık enflasyon)", "Tüketici enflasyonunun yıllık hızı. " + _INFL, +1, 2, r"^cpi yoy")),
    (r"Core CPI m/m", EventInfo("Çekirdek TÜFE (gıda-enerji hariç aylık enflasyon)",
                                "Gıda ve enerji hariç enflasyon; Fed'in asıl baktığı ölçülerden. " + _INFL,
                                +1, 3, r"^core cpi mom")),
    (r"Core CPI y/y", EventInfo("Çekirdek TÜFE (gıda-enerji hariç yıllık enflasyon)", "Gıda ve enerji hariç yıllık enflasyon. " + _INFL,
                                +1, 2, r"^core cpi yoy")),
    (r"PPI m/m", EventInfo("ÜFE (üretici enflasyonu, aylık)", "Üretici fiyatları; tüketici enflasyonunun öncüsü. " + _INFL,
                           +1, 1, r"^ppi mom")),
    (r"Core PPI m/m", EventInfo("Çekirdek ÜFE (gıda-enerji hariç üretici enflasyonu, aylık)", "Gıda ve enerji hariç üretici fiyatları. " + _INFL,
                                +1, 1, r"^core ppi mom")),
    (r"Core PCE Price Index m/m", EventInfo(
        "Çekirdek PCE (aylık TÜFE — gıda-enerji hariç, Fed'in ana ölçüsü)",
        "Fed'in enflasyon hedefinde kullandığı ana ölçü (gıda ve enerji hariç). " + _INFL,
        +1, 3, r"^core pce price index mom")),
    (r"Core PCE Price Index y/y", EventInfo("Çekirdek PCE (yıllık TÜFE — gıda-enerji hariç, Fed'in ana ölçüsü)",
                                            "Fed'in hedef enflasyon ölçüsünün yıllık hızı. " + _INFL,
                                            +1, 2, r"^core pce price index yoy")),
    (r"PCE Price Index m/m", EventInfo("PCE (aylık TÜFE — Fed'in ölçüsü)", "Kişisel tüketim harcaması enflasyonu. " + _INFL,
                                       +1, 1, r"^pce price index mom")),
    (r"PCE Price Index y/y", EventInfo("PCE (yıllık TÜFE — Fed'in ölçüsü)", "Kişisel tüketim enflasyonunun yıllık hızı. " + _INFL,
                                       +1, 1, r"^pce price index yoy")),
    (r"Import Prices m/m", EventInfo("İthalat Fiyatları (ithal mal enflasyonu, aylık)", "İthal malların fiyat değişimi; enflasyon baskısının göstergesi.",
                                     +1, 1, r"^import prices mom")),
    (r"(Prelim |Revised )?UoM Inflation Expectations", EventInfo(
        "Michigan Enflasyon Beklentisi (tüketicinin 1 yıllık enflasyon tahmini)",
        "Tüketicilerin 1 yıllık enflasyon beklentisi. Yükselmesi Fed'i tedirgin eder (USD lehine).",
        +1, 1, r"(michigan|uom).*inflation")),
    # --- buyume / talep ---
    (r"(Advance|Prelim|Final) GDP q/q", EventInfo(
        "GSYH (ekonomik büyüme, çeyreklik)",
        "Ekonominin çeyreklik büyümesi (yıllıklandırılmış). Güçlü büyüme USD lehine.",
        +1, 2, r"^gdp (qoq|annualized|growth)")),
    (r"(Advance|Prelim|Final) GDP Price Index q/q", EventInfo(
        "GSYH Fiyat Endeksi (büyüme verisindeki enflasyon, çeyreklik)", "Büyüme verisinin içindeki enflasyon ölçüsü. " + _INFL,
        +1, 1, r"^gdp (price index|deflator)")),
    (r"Retail Sales m/m", EventInfo("Perakende Satışlar (mağaza harcamaları, aylık)",
                                    "Mağaza satışlarındaki aylık değişim; tüketici harcamasının ana göstergesi.",
                                    +1, 2, r"^retail sales mom")),
    (r"Core Retail Sales m/m", EventInfo("Çekirdek Perakende Satışlar (otomobil hariç harcama, aylık)",
                                         "Otomobil hariç perakende satışlar; harcamanın daha temiz ölçüsü.",
                                         +1, 2, r"^retail sales ex[- ]?autos? mom")),
    (r"Personal Spending m/m", EventInfo("Kişisel Harcamalar (hanehalkı harcaması, aylık)", "Hanehalkı harcamalarının aylık değişimi.",
                                         +1, 1, r"^(personal|consumer) spending mom")),
    (r"Personal Income m/m", EventInfo("Kişisel Gelir (hanehalkı geliri, aylık)", "Hanehalkı gelirinin aylık değişimi.",
                                       +1, 1, r"^personal income mom")),
    (r"Durable Goods Orders m/m", EventInfo("Dayanıklı Mal Siparişleri (makine/uçak gibi fabrika siparişleri, aylık)",
                                            "Uzun ömürlü mal (makine, uçak vb.) siparişleri; yatırım iştahının göstergesi.",
                                            +1, 1, r"^durable goods orders mom")),
    (r"Core Durable Goods Orders m/m", EventInfo("Çekirdek Dayanıklı Mal Siparişleri (ulaşım hariç fabrika siparişleri, aylık)",
                                                 "Ulaşım hariç dayanıklı mal siparişleri.",
                                                 +1, 1, r"^durable goods (orders )?ex[- ]?transport")),
    (r"Industrial Production m/m", EventInfo("Sanayi Üretimi (fabrika/maden/enerji üretimi, aylık)", "Fabrika, maden ve enerji üretiminin değişimi.",
                                             +1, 1, r"^industrial production mom")),
    (r"Trade Balance", EventInfo("Dış Ticaret Dengesi (ihracat − ithalat)", "İhracat eksi ithalat. Açığın daralması USD lehine.",
                                 +1, 1, r"^trade balance")),
    (r"Goods Trade Balance", EventInfo("Mal Ticareti Dengesi (mal ihracatı − ithalatı, öncü)", "Mal ihracatı eksi ithalatı. Açığın daralması USD lehine.",
                                       +1, 1, r"goods trade balance")),
    # --- anketler ---
    (r"ISM Manufacturing PMI", EventInfo(
        "ISM İmalat PMI (sanayi anketi; 50 üstü büyüme)",
        "İmalat sektörü satın alma yöneticileri endeksi; 50 üstü büyüme, altı daralma.",
        +1, 2, r"^ism manufacturing pmi")),
    (r"ISM Services PMI", EventInfo(
        "ISM Hizmet PMI (hizmet sektörü anketi; 50 üstü büyüme)",
        "Hizmet sektörü satın alma yöneticileri endeksi (ekonominin ~%70'i); 50 üstü büyüme.",
        +1, 2, r"^ism (services|non-?manufacturing) pmi")),
    (r"ISM Manufacturing Prices", EventInfo("ISM İmalat Fiyatları (sanayide ödenen fiyatlar)", "İmalatta ödenen fiyatlar; enflasyonun öncüsü. " + _INFL,
                                            +1, 1, r"^ism manufacturing prices")),
    (r"ISM Services Prices", EventInfo("ISM Hizmet Fiyatları (hizmette ödenen fiyatlar)", "Hizmette ödenen fiyatlar; enflasyonun öncüsü. " + _INFL,
                                       +1, 1, r"^ism (services|non-?manufacturing) prices")),
    (r"Flash Manufacturing PMI", EventInfo("S&P Öncü İmalat PMI (ay içi sanayi anketi; 50 üstü büyüme)", "İmalat sektörünün ay içi öncü anketi; 50 üstü büyüme.",
                                           +1, 1, r"^(s&p global )?manufacturing pmi")),
    (r"Flash Services PMI", EventInfo("S&P Öncü Hizmet PMI (ay içi hizmet anketi; 50 üstü büyüme)", "Hizmet sektörünün ay içi öncü anketi; 50 üstü büyüme.",
                                      +1, 1, r"^(s&p global )?services pmi")),
    (r"Chicago PMI", EventInfo("Chicago PMI (bölgesel iş anketi; 50 üstü büyüme)", "Chicago bölgesi iş aktivitesi anketi; 50 üstü büyüme.",
                               +1, 1, r"chicago pmi")),
    (r"Empire State Manufacturing Index", EventInfo("NY Empire State (New York sanayi anketi; 0 üstü büyüme)",
                                                    "New York bölgesi imalat anketi; sıfır üstü büyüme.",
                                                    +1, 1, r"empire state")),
    (r"Philly Fed Manufacturing Index", EventInfo("Philly Fed (Philadelphia sanayi anketi; 0 üstü büyüme)",
                                                  "Philadelphia bölgesi imalat anketi; sıfır üstü büyüme.",
                                                  +1, 1, r"philadelphia fed|philly fed")),
    (r"CB Consumer Confidence", EventInfo("CB Tüketici Güveni (hanehalkının ekonomiye güveni)",
                                          "Hanehalkının ekonomiye güveni; harcamanın öncüsü.",
                                          +1, 1, r"conference board|cb consumer confidence")),
    (r"(Prelim |Revised )?UoM Consumer Sentiment", EventInfo("Michigan Tüketici Güveni (hanehalkı hissiyat anketi)",
                                                             "Michigan Üniversitesi tüketici hissiyatı anketi.",
                                                             +1, 1, r"(michigan|uom).*sentiment")),
    # --- konut ---
    (r"Building Permits", EventInfo("İnşaat İzinleri (yeni konut izni sayısı)", "Yeni konut inşaatı için verilen izinler.", +1, 1, r"^building permits")),
    (r"Housing Starts", EventInfo("Konut Başlangıçları (yapımına başlanan konut)", "Yapımına başlanan yeni konut sayısı.", +1, 1, r"^housing starts")),
    (r"New Home Sales", EventInfo("Yeni Konut Satışları (sıfır konut)", "Satılan yeni konut sayısı.", +1, 1, r"^new home sales")),
    (r"Existing Home Sales", EventInfo("Mevcut Konut Satışları (ikinci el konut)", "İkinci el konut satışları.", +1, 1, r"^existing home sales")),
    (r"Pending Home Sales m/m", EventInfo("Bekleyen Konut Satışları (sözleşmesi imzalanan konut, aylık)", "Sözleşmesi imzalanmış konut satışları.",
                                          +1, 1, r"^pending home sales")),
    # --- enerji ---
    (r"Crude Oil Inventories", EventInfo(
        "Ham Petrol Stokları (haftalık ABD petrol stoğu, EIA)",
        "Haftalık ABD ham petrol stok değişimi. Beklentiden fazla artış petrol fiyatı için olumsuz.",
        -1, 1, r"(eia )?crude oil (stocks|inventories)", "oil")),
    # --- Fed ---
    (r"Federal Funds Rate", EventInfo(
        "Fed Faiz Kararı (politika faizi)",
        "Fed'in politika faizi. Beklentiden yüksek (şahin) karar USD'yi güçlendirir, altın ve borsayı baskılar.",
        +1, 3, r"(interest rate decision|fed(eral)? funds)")),
    (r"FOMC Statement", EventInfo(
        "FOMC Karar Metni (faiz kararıyla yayımlanan açıklama)",
        "Faiz kararıyla yayımlanan metin; gelecek adımlara dair ipucu için okunur.",
        kind="text", fj=r"\bFOMC\b|\bFed statement\b")),
    (r"FOMC Economic Projections", EventInfo(
        "FOMC Projeksiyonları (Fed'in faiz/büyüme/enflasyon tahminleri)",
        "Fed üyelerinin faiz, büyüme ve enflasyon tahminleri (nokta grafiği).",
        kind="text", fj=r"\bdot plot\b|\bprojections?\b|\bFOMC\b")),
    (r"FOMC Press Conference", EventInfo(
        "FOMC Basın Toplantısı (Fed Başkanı karar sonrası konuşuyor)",
        "Fed Başkanı karar sonrası soruları yanıtlıyor; piyasa en çok bu konuşmada oynar.",
        kind="talk", fj=r"\bFed Chair\b|\bPowell\b|\bWarsh\b")),
    (r"FOMC Meeting Minutes", EventInfo(
        "FOMC Tutanakları (önceki Fed toplantısının notları)",
        "Önceki toplantının tutanakları; üyelerin faiz konusunda ne kadar şahin/güvercin olduğunu gösterir.",
        kind="text", fj=r"\bminutes\b|\bFOMC\b")),
    (r"Beige Book", EventInfo("Bej Kitap (Fed'in bölgesel ekonomi raporu)", "Fed'in 12 bölgeden topladığı ekonomik durum raporu.",
                              kind="text", fj=r"\bBeige Book\b")),
    (r"Treasury Currency Report", EventInfo("Hazine Kur Raporu (kur manipülasyonu raporu)",
                                            "ABD Hazinesi'nin kur manipülasyonu raporu.",
                                            kind="text", fj=r"\bcurrency report\b|\bmanipulat")),
    (r"\d+-y Bond Auction", EventInfo("Tahvil İhalesi (Hazine borçlanması; getiri yüksekse USD lehine)", "Hazine tahvil ihalesi; getiri yüksek çıkarsa USD lehine.",
                                      +1, 1, r"(note|bond) auction (high )?yield")),
]
_EVENTS_RE = [(re.compile(p, re.I), info) for p, info in _EVENTS]

# "Fed Chair Powell Speaks", "FOMC Member Waller Speaks", "President Trump Speaks", ...
_SPEAKS_RE = re.compile(r"^(?P<role>.*?)\s*(?P<name>[A-Z][A-Za-z'\-]+)\s+Speaks$")
_ROLE_TR = [
    (r"Fed Chair", "Fed Başkanı"),
    (r"Fed Vice Chair", "Fed Başkan Yardımcısı"),
    (r"FOMC Member", "Fed üyesi"),
    (r"President", "ABD Başkanı"),
    (r"Treasury Sec", "ABD Hazine Bakanı"),
]
_TALK_MEANING = {
    "Fed Başkanı": "Fed Başkanı'nın faiz ve enflasyon mesajları USD, altın ve Nasdaq'ı sert oynatabilir.",
    "ABD Başkanı": "Trump'ın tarife, Fed ve jeopolitik söylemleri USD, altın ve endekslerde ani oynaklık yaratabilir.",
    "ABD Hazine Bakanı": "Hazine Bakanı'nın dolar, tahvil ve tarife mesajları piyasayı oynatabilir.",
}
_TALK_MEANING_DEFAULT = "Faize dair ipucu arayan piyasa konuşmayı izler; şahin mesaj USD lehine, güvercin mesaj aleyhine."
_TALK_TONE = ("Okuma rehberi: faizleri yüksek tutma / artırma (şahin) mesajı → USD güçlü, altın ve Nasdaq aşağı; "
              "faiz indirimine sıcak (güvercin) mesaj → tersi.")
_TALK_TONE_ROLE = {
    "ABD Başkanı": ("Okuma rehberi: tarifede sertleşme → Nasdaq ve kripto aşağı, altın yukarı; Fed'e faiz indirimi "
                    "baskısı → USD aşağı, altın yukarı; ticaret anlaşması / yumuşama → Nasdaq yukarı."),
    "ABD Hazine Bakanı": ("Okuma rehberi: güçlü dolar / tahvil piyasasını sakinleştiren mesaj → USD yukarı; "
                          "tarife ya da mali risk vurgusu → Nasdaq aşağı, altın yukarı."),
}


def _info_for(title: str) -> tuple[EventInfo, str | None]:
    """FF basligi -> (sozluk kaydi, konusmacinin soyadi). Bilinmeyen baslik Ingilizce adla doner."""
    for rx, info in _EVENTS_RE:
        if rx.fullmatch(title.strip()):
            return info, (_speaker_of(title) if info.kind == "talk" else None)
    m = _SPEAKS_RE.match(title.strip())
    if m:
        name = m.group("name")
        role_tr = next((tr for rx, tr in _ROLE_TR if re.search(rx, m.group("role"), re.I)), m.group("role").strip())
        tr = f"{role_tr} {name} konuşacak" if role_tr else f"{name} konuşacak"
        return EventInfo(tr, _TALK_MEANING.get(role_tr, _TALK_MEANING_DEFAULT), kind="talk",
                         fj=rf"\b{re.escape(name)}\b", tone=_TALK_TONE_ROLE.get(role_tr, _TALK_TONE)), name
    return EventInfo(title), None


def _speaker_of(title: str) -> str | None:
    m = _SPEAKS_RE.match(title.strip())
    return m.group("name") if m else None


# ---------------------------------------------------------------------------
# Veri modeli
# ---------------------------------------------------------------------------

@dataclass
class CalEvent:
    title: str
    country: str
    impact: str          # High | Medium | Low | Holiday
    at: datetime         # UTC, aware
    forecast: str = ""
    previous: str = ""
    info: EventInfo = field(default_factory=lambda: EventInfo(""))
    speaker: str | None = None

    @property
    def key(self) -> str:
        return f"{self.at:%Y%m%dT%H%MZ}|{self.country}|{self.title}"

    @property
    def tr(self) -> str:
        return self.info.tr or self.title


@dataclass
class Headline:
    guid: str
    title: str           # "FinancialJuice: " oneki atilmis
    at: datetime         # UTC


@dataclass
class DataResult:
    actual: str
    forecast: str | None
    previous: str | None
    at: datetime
    headline: str


@dataclass
class _SuddenState:
    root_id: int | None
    root_at: datetime
    last_sent_at: datetime
    pending: list[Headline] = field(default_factory=list)


_events: list[CalEvent] = []      # NEWS_CURRENCIES, High/Medium
_holidays: list[CalEvent] = []    # NEWS_CURRENCIES, Holiday
_cal_fetched_at: float = 0.0
_cal_ok_at: datetime | None = None
_cal_error: str | None = None
_headlines: dict[str, Headline] = {}
_fj_next_at: float = 0.0
_fj_ok_at: datetime | None = None
_fj_error: str | None = None
_results: dict[str, DataResult] = {}
_sent: dict[str, int | None] = {}
_sudden: dict[str, _SuddenState] = {}
_started_at: datetime = datetime.now(UTC)
_task: asyncio.Task | None = None


def is_configured() -> bool:
    return bool(NEWS_TELEGRAM_BOT_TOKEN and NEWS_TELEGRAM_CHAT_ID)


# ---------------------------------------------------------------------------
# Deger bicimleme / karsilastirma
# ---------------------------------------------------------------------------

_VAL_RE = re.compile(r"^\s*([-+]?)\s*(\d+(?:\.\d+)?)\s*([kKmMbBtT%]?)\s*$")
_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*([kKmMbBtT%]?)")
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_TR_UNIT = {"k": " bin", "m": " milyon", "b": " milyar", "t": " trilyon"}


def _clean(v: str | None) -> str | None:
    v = (v or "").strip().rstrip(",")
    return None if v in ("", "-", "--", "N/A") else v


def tr_value(v: str | None) -> str:
    """'90K' -> '90 bin', '0.3%' -> '%0,3', '-132.6B' -> '-132,6 milyar'."""
    v = _clean(v)
    if v is None:
        return "-"
    m = _VAL_RE.match(v)
    if not m:
        return v
    sign, num, unit = m.group(1), m.group(2).replace(".", ","), m.group(3).lower()
    if unit == "%":
        return f"{sign}%{num}"
    return f"{sign}{num}{_TR_UNIT.get(unit, '')}"


def _num(v: str | None) -> float | None:
    v = _clean(v)
    if v is None:
        return None
    m = _NUM_RE.search(v.replace(",", ""))
    if not m:
        return None
    return float(m.group(1)) * _MULT.get(m.group(2).lower(), 1.0)


# ---------------------------------------------------------------------------
# Takvim (ForexFactory)
# ---------------------------------------------------------------------------

def parse_calendar(rows: list[dict]) -> tuple[list[CalEvent], list[CalEvent]]:
    events, holidays = [], []
    for r in rows:
        country = (r.get("country") or "").upper()
        if country not in NEWS_CURRENCIES:
            continue
        impact = r.get("impact") or ""
        try:
            at = datetime.fromisoformat(r["date"]).astimezone(UTC)
        except Exception:
            continue
        title = (r.get("title") or "").strip()
        ev = CalEvent(title, country, impact, at, (r.get("forecast") or "").strip(), (r.get("previous") or "").strip())
        if impact == "Holiday":
            ev.info = EventInfo(title)
            holidays.append(ev)
        elif impact in IMPACTS:
            ev.info, ev.speaker = _info_for(title)
            events.append(ev)
    events.sort(key=lambda e: (e.at, 0 if e.impact == "High" else 1))
    return events, holidays


async def _refresh_calendar(force: bool = False) -> None:
    global _events, _holidays, _cal_fetched_at, _cal_ok_at, _cal_error
    if not force and time.monotonic() - _cal_fetched_at < _CAL_REFRESH_SEC and _cal_ok_at is not None:
        return
    _cal_fetched_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=20, headers=_HEADERS) as client:
            resp = await client.get(FF_URL)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        _events, _holidays = parse_calendar(resp.json())
        _cal_ok_at = datetime.now(UTC)
        if _cal_error:
            log.info("NEWS calendar recovered (%d events)", len(_events))
        _cal_error = None
    except Exception as e:
        if _cal_error is None:
            log.warning("NEWS calendar fetch failed: %s", e)
        _cal_error = str(e) or type(e).__name__


# ---------------------------------------------------------------------------
# FinancialJuice RSS
# ---------------------------------------------------------------------------

_FJ_DATA_RE = re.compile(r"^US\s+(?P<name>.+?)\s+Actual\s+(?P<actual>\S+)\s*(?:\((?P<rest>.*)\))?\s*$")


def parse_rss(text: str) -> list[Headline]:
    out = []
    root = ET.fromstring(text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        title = re.sub(r"^FinancialJuice:\s*", "", title)
        guid = (it.findtext("guid") or title).strip()
        try:
            at = email.utils.parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(UTC)
        except Exception:
            continue
        out.append(Headline(guid, re.sub(r"\s+", " ", title), at))
    return out


def parse_data_headline(title: str) -> tuple[str, str, str | None, str | None] | None:
    """'US Core PCE Price Index MoM Actual 0.2% (Forecast 0.3%, Previous 0.2%)' -> (ad, actual, forecast, previous)."""
    m = _FJ_DATA_RE.match(title)
    if not m:
        return None
    rest = m.group("rest") or ""
    fc = re.search(r"Forecast\s+([^,)\s]+)", rest)
    pv = re.search(r"Previous\s+([^,)\s]+)", rest)
    return (m.group("name").strip(), m.group("actual").rstrip(","),
            _clean(fc.group(1)) if fc else None, _clean(pv.group(1)) if pv else None)


async def _poll_fj() -> list[Headline]:
    """Yeni basliklari dondurur (daha once gorulmemis guid)."""
    global _fj_next_at, _fj_ok_at, _fj_error
    now_m = time.monotonic()
    if now_m < _fj_next_at:
        return []
    _fj_next_at = now_m + _FJ_POLL_SEC
    try:
        async with httpx.AsyncClient(timeout=20, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(FJ_URL)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After") or 60)
            _fj_next_at = now_m + max(_FJ_POLL_SEC, wait + 30)
            raise RuntimeError(f"429 (Retry-After {wait})")
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        items = parse_rss(resp.text)
    except Exception as e:
        if _fj_error is None:
            log.info("NEWS FinancialJuice fetch failed: %s", e)
        _fj_error = str(e) or type(e).__name__
        return []
    if _fj_error:
        log.info("NEWS FinancialJuice recovered")
    _fj_error = None
    _fj_ok_at = datetime.now(UTC)
    fresh = [h for h in items if h.guid not in _headlines]
    for h in fresh:
        _headlines[h.guid] = h
    cutoff = datetime.now(UTC) - _HEADLINE_KEEP
    for g in [g for g, h in _headlines.items() if h.at < cutoff]:
        del _headlines[g]
    return sorted(fresh, key=lambda h: h.at)


# ---------------------------------------------------------------------------
# Sonuc eslestirme
# ---------------------------------------------------------------------------

_TOK_SYN = (("m/m", " mom "), ("y/y", " yoy "), ("q/q", " qoq "))
_TOK_DROP = {"prelim", "final", "advance", "adv", "flash", "revised", "sa", "nsa", "the", "of", "change", "index", "us"}


def _tokens(s: str) -> set[str]:
    s = s.lower()
    for a, b in _TOK_SYN:
        s = s.replace(a, b)
    return set(re.findall(r"[a-z0-9]+", s)) - _TOK_DROP


def _name_matches(ev: CalEvent, name: str) -> bool:
    if ev.info.fj:
        return re.search(ev.info.fj, name, re.I) is not None
    a, b = _tokens(ev.title), _tokens(name)
    if not a:
        return False
    for t in ("core", "mom", "yoy", "qoq"):
        if (t in a) != (t in b):
            return False
    return len(a & b) / len(a) >= 0.75


def match_result(ev: CalEvent, headlines) -> DataResult | None:
    """FF olayinin aciklanan degerini FJ basliklarindan bulur (aciklama ±pencere, ilk eslesen)."""
    lo, hi = ev.at - timedelta(minutes=2), ev.at + timedelta(minutes=30)
    for h in sorted(headlines, key=lambda h: h.at):
        if not (lo <= h.at <= hi):
            continue
        parsed = parse_data_headline(h.title)
        if parsed is None:
            continue
        name, actual, fc, pv = parsed
        if _name_matches(ev, name):
            return DataResult(actual, fc, pv, h.at, h.title)
    return None


def talk_headlines(ev: CalEvent, headlines, window: timedelta) -> list[Headline]:
    if not ev.info.fj:
        return []
    lo, hi = ev.at - timedelta(minutes=2), ev.at + window
    return [h for h in sorted(headlines, key=lambda h: h.at)
            if lo <= h.at <= hi and re.search(ev.info.fj, h.title, re.I)
            and "FJElite" not in h.title and _FJ_DATA_RE.match(h.title) is None]


def _effect(ev: CalEvent, res: DataResult) -> tuple[int, str]:
    """(USD etkisi -1/0/+1, 'beklentiden yuksek' gibi aciklama). Beklenti: FJ'nin (ayni bicim) yoksa FF'nin."""
    a = _num(res.actual)
    ref_label = "beklenti"
    ref = _num(res.forecast) if res.forecast else _num(ev.forecast)
    if ref is None:
        ref, ref_label = (_num(res.previous) if res.previous else _num(ev.previous)), "önceki"
    if a is None or ref is None:
        return 0, "karşılaştırılamadı"
    if abs(a - ref) < 1e-12:
        return 0, f"{ref_label}yle aynı"
    d = 1 if a > ref else -1
    word = "yüksek" if d > 0 else "düşük"
    return d * ev.info.sign, f"{ref_label}den {word}"


def _market_text(net: int, mixed: bool) -> str:
    if net > 0:
        body = ("USD güçlenebilir → altın (XAUUSD), EURUSD ve GBPUSD aşağı, USDJPY yukarı baskı olabilir. "
                "Faiz indirimi beklentisi zayıflayacağı için Nasdaq (US100) ve kripto için de aşağı yönlü risk.")
        head = "Veri USD lehine."
    elif net < 0:
        body = ("USD zayıflayabilir → altın (XAUUSD), EURUSD ve GBPUSD yukarı, USDJPY aşağı yönlü olabilir. "
                "Faiz indirimi beklentisi arttığı için Nasdaq (US100) ve kripto için olumlu.")
        head = "Veri USD aleyhine."
    else:
        return "Veriler beklentiye yakın ya da birbirini dengeliyor; belirgin bir yön sinyali yok, fiyat teknik seviyelere kalır."
    if mixed:
        head = "Karışık sonuç; ağırlıklı okuma " + ("USD lehine." if net > 0 else "USD aleyhine.")
    return f"{head} {body}"


# ---------------------------------------------------------------------------
# Mesaj metinleri
# ---------------------------------------------------------------------------

def _e(s: str) -> str:
    return html.escape(s or "", quote=False)


def _tsi(dt: datetime) -> datetime:
    return dt.astimezone(TSI)


def _fp_line(ev: CalEvent) -> str:
    parts = []
    if _clean(ev.forecast):
        parts.append(f"Beklenti {tr_value(ev.forecast)}")
    if _clean(ev.previous):
        parts.append(f"Önceki {tr_value(ev.previous)}")
    return " · ".join(parts)


def _impact_tag(ev: CalEvent) -> str:
    emoji, name = IMPACTS.get(ev.impact, ("⚪", ev.impact.lower()))
    return f"{emoji} ({name})"


def group_events(events: list[CalEvent]) -> dict[datetime, list[CalEvent]]:
    groups: dict[datetime, list[CalEvent]] = {}
    for ev in events:
        groups.setdefault(ev.at, []).append(ev)
    return groups


def format_morning(now: datetime, events: list[CalEvent], holidays: list[CalEvent], cal_error: str | None = None) -> str:
    n = _tsi(now)
    start = n.replace(hour=NEWS_MORNING_HOUR, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    todays = [e for e in events if start <= _tsi(e.at) < end]
    hols = [h for h in holidays if _tsi(h.at).date() == n.date()]
    cur = "/".join(NEWS_CURRENCIES)
    lines = [f"☀️ <b>{n:%d.%m.%Y} {_TR_DAYS[n.weekday()]} — günün {cur} takvimi</b>"]
    if cal_error and not events:
        lines.append(f"⚠️ ForexFactory takvimi alınamadı ({_e(cal_error)}).")
        return "\n".join(lines)
    for h in hols:
        country = _COUNTRY_TR.get(h.country, h.country)
        if "holiday" in h.title.lower():
            lines.append(f"🏦 <b>Bank Holiday</b> — {country}: bankalar kapalı, likidite düşük; spreadler açılabilir.")
        else:   # FF "Holiday" etkisiyle baska takvim notlari da veriyor (yaz saati gecisi gibi)
            lines.append(f"🕐 <b>{_e(h.title)}</b> — {country}")
    if not todays:
        lines.append("✅ Bugün önemli (kırmızı/turuncu) haber yok.")
        return "\n".join(lines)
    lines.append("")
    for e in todays:
        t = _tsi(e.at)
        day = "" if t.date() == n.date() else f"{t:%d.%m} "
        fp = _fp_line(e)
        lines.append(f"{IMPACTS[e.impact][0]} <b>{day}{t:%H:%M}</b> {_e(e.tr)}" + (f"\n      <i>{fp}</i>" if fp else ""))
    red = sum(1 for e in todays if e.impact == "High")
    orange = len(todays) - red
    lines.append("")
    lines.append(f"Toplam: {red} kırmızı, {orange} turuncu. Her haberden {NEWS_ALERT_MINUTES} dk önce ayrıca bildirim gelir.")
    return "\n".join(lines)


def format_alert(at: datetime, group: list[CalEvent], now: datetime) -> str:
    t = _tsi(at)
    mins = max(1, round((at - now).total_seconds() / 60))
    verb = "başlayacak:" if all(e.info.kind == "talk" for e in group) else "açıklanacak:"
    head = (f"⏰ <b>{t:%d.%m.%Y %H:%M} TSİ</b> — {mins} dk sonra "
            + (verb if len(group) == 1 else f"{len(group)} {group[0].country} haberi {verb}"))
    lines = [head, ""]
    for ev in group:
        lines.append(f"{_impact_tag(ev)} <b>{_e(ev.tr)}</b>")
        fp = _fp_line(ev)
        if fp:
            lines.append(f"   {fp}")
        if ev.info.meaning:
            lines.append(f"   <i>{_e(ev.info.meaning)}</i>")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_result(at: datetime, group: list[CalEvent], results: dict[str, DataResult]) -> str:
    t = _tsi(at)
    lines = [f"📢 <b>Açıklandı — {t:%d.%m %H:%M} TSİ</b>", ""]
    net, pos, neg, oil_lines = 0, False, False, []
    for ev in group:
        res = results.get(ev.key)
        if res is None:
            lines.append(f"{IMPACTS[ev.impact][0]} {_e(ev.tr)}: <i>sonuç bulunamadı</i>")
            continue
        fc = res.forecast or _clean(ev.forecast)
        pv = res.previous or _clean(ev.previous)
        eff, how = _effect(ev, res)
        lines.append(f"{IMPACTS[ev.impact][0]} <b>{_e(ev.tr)}</b>: <b>{tr_value(res.actual)}</b>")
        lines.append(f"   Beklenti {tr_value(fc)} · Önceki {tr_value(pv)} → {how}")
        if ev.info.kind == "oil":
            if eff:
                oil_lines.append("Petrol (OILWTI/OILBRENT): stok " + ("beklentiden az → yukarı baskı."
                                                                       if eff > 0 else "beklentiden fazla → aşağı baskı."))
            continue
        net += eff * ev.info.weight
        pos, neg = pos or eff > 0, neg or eff < 0
    usd_events = [e for e in group if e.info.kind == "data" and e.key in results]
    lines.append("")
    if usd_events:
        lines.append(f"📈 <b>Piyasaya olası etki (tahmin):</b> {_e(_market_text(net, pos and neg))}")
    for ol in oil_lines:
        lines.append(f"🛢 {_e(ol)}")
    return "\n".join(lines).rstrip()


def format_talk(at: datetime, group: list[CalEvent], found: dict[str, list[Headline]]) -> str:
    t = _tsi(at)
    lines = [f"🎙 <b>{t:%d.%m %H:%M} TSİ — {', '.join(_e(e.tr) for e in group)}</b>", ""]
    any_hl = False
    for ev in group:
        hls = found.get(ev.key) or []
        if not hls:
            continue
        any_hl = True
        lines.append("Öne çıkan başlıklar (FinancialJuice, İngilizce):")
        for h in hls[:8]:
            lines.append(f"• {_tsi(h.at):%H:%M} <i>{_e(h.title)}</i>")
        if len(hls) > 8:
            lines.append(f"  +{len(hls) - 8} başlık daha")
    if not any_hl:
        lines.append("<i>Konuşma/metinden başlık düşmedi (FinancialJuice'ta eşleşen yok).</i>")
    lines.append("")
    lines.extend(f"<i>{_e(t)}</i>" for t in dict.fromkeys(e.info.tone or _TALK_TONE for e in group))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Takvim disi (ani) haberler — anahtar kelime, LLM yok
# ---------------------------------------------------------------------------

_PEOPLE = r"(Trump|Fed Chair|Powell|Warsh|Bessent|Vance|White House|Treasury Sec(?:retary)?)"
SUDDEN_CATS: list[tuple[str, str, re.Pattern, str]] = [
    ("konusma", "Konuşma / açıklama duyurusu",
     re.compile(rf"\b{_PEOPLE}\b.*\b(to|will) (speak|deliver|address|hold|give|make|announce)\b"
                rf"|\b{_PEOPLE}\b.*\b(press conference|address to the nation|prime[- ]?time address|oval office address)\b", re.I),
     "Takvimde olmayan bir konuşma/açıklama duyuruldu. Bu tür konuşmalar USD, altın ve Nasdaq'ta ani oynaklık "
     "yaratabilir; konuşma saatinde pozisyon riskine dikkat."),
    ("fed_acil", "Fed'den olağanüstü adım",
     re.compile(r"\b(emergency|inter-?meeting|unscheduled)\b.*\b(rate|meeting|cut|hike|Fed|FOMC)\b"
                r"|\b(Fed|FOMC)\b.*\b(emergency|unscheduled)\b", re.I),
     "Fed'in takvim dışı faiz/toplantı hamlesi en sert piyasa hareketlerinden birini tetikler; "
     "indirim → USD aşağı, altın ve Nasdaq yukarı; artırım → tersi."),
    ("jeopolitik", "Jeopolitik şok",
     re.compile(r"\b(declares? war|invasion|invades?|nuclear (strike|attack|test)|missile (strike|attack)s?"
                r"|air ?strikes? (on|against)|martial law|coup)\b", re.I),
     "Savaş/saldırı haberleri risk iştahını düşürür: altın yukarı, Nasdaq ve kripto aşağı; "
     "USD genelde güvenli liman olarak güçlenir, petrol yükselebilir."),
    ("tarife", "Tarife haberi",
     re.compile(r"\b(Trump|White House)\b.*\btariffs?\b|\btariffs?\b.*\b(Trump|White House)\b", re.I),
     "Yeni tarife açıklamaları ticaret savaşı endişesini artırır: Nasdaq ve kripto aşağı, altın yukarı baskı; "
     "USD yönü karışık olabilir."),
    ("abd_mali", "ABD mali risk",
     re.compile(r"\b(government shutdown|debt ceiling|default on|downgrades? (the )?U\.?S\.?|U\.?S\.? (credit )?rating)\b", re.I),
     "Hükümetin kapanması, borç tavanı ya da not indirimi haberleri USD'yi zayıflatabilir, altını destekler."),
]


def sudden_category(title: str) -> tuple[str, str, str] | None:
    if "FJElite" in title or _FJ_DATA_RE.match(title):
        return None
    for cat, label, rx, meaning in SUDDEN_CATS:
        if rx.search(title):
            return cat, label, meaning
    return None


def format_sudden(label: str, meaning: str, h: Headline) -> str:
    return (f"⚡ <b>Takvim dışı: {_e(label)}</b> — {_tsi(h.at):%d.%m %H:%M} TSİ\n\n"
            f"<i>{_e(h.title)}</i>\n\n{_e(meaning)}")


def format_digest(label: str, hls: list[Headline]) -> str:
    lines = [f"⚡ <b>{_e(label)} — devam eden başlıklar</b>"]
    for h in hls[:6]:
        lines.append(f"• {_tsi(h.at):%H:%M} <i>{_e(h.title)}</i>")
    if len(hls) > 6:
        lines.append(f"  +{len(hls) - 6} başlık daha")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram + kalicilik
# ---------------------------------------------------------------------------

async def send(text: str, reply_to: int | None = None) -> int | None:
    if not is_configured():
        return None
    payload = {"chat_id": NEWS_TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to
        payload["allow_sending_without_reply"] = True
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TELEGRAM_API.format(token=NEWS_TELEGRAM_BOT_TOKEN), json=payload)
        if resp.status_code == 200:
            return resp.json().get("result", {}).get("message_id")
        log.warning("NEWS Telegram error %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        log.warning("NEWS Telegram send failed: %s", e)
    return None


async def _send_once(key: str, kind: str, text: str, reply_to: int | None = None) -> int | None:
    """Gonderir ve kaydeder; basarisizsa kaydetmez (sonraki turda yeniden denenir)."""
    msg_id = await send(text, reply_to)
    if msg_id is None:
        return None
    _sent[key] = msg_id
    log.info("NEWS %s sent (%s)", kind.upper(), key)
    try:
        from app.database import async_session
        from app.models import NewsNotice
        async with async_session() as s:
            s.add(NewsNotice(key=key, kind=kind, tg_message_id=msg_id,
                             sent_at=datetime.now(UTC).replace(tzinfo=None)))
            await s.commit()
    except IntegrityError:
        pass
    except Exception:
        log.exception("NEWS notice save failed (%s)", key)
    return msg_id


async def _load_sent() -> None:
    try:
        from app.database import async_session
        from app.models import NewsNotice
        since = (datetime.now(UTC) - timedelta(days=4)).replace(tzinfo=None)
        async with async_session() as s:
            rows = (await s.execute(select(NewsNotice).where(NewsNotice.sent_at >= since)
                                    .order_by(NewsNotice.sent_at))).scalars().all()
    except Exception:
        log.exception("NEWS notice load failed")
        return
    chain_since = datetime.now(UTC) - _SUDDEN_CHAIN
    for r in rows:
        _sent[r.key] = r.tg_message_id
        at = r.sent_at.replace(tzinfo=UTC)
        if r.kind in ("sudden", "digest") and at >= chain_since:
            cat = r.key.split("|")[1]
            st = _sudden.get(cat)
            if r.kind == "sudden":
                _sudden[cat] = _SuddenState(r.tg_message_id, at, at)
            elif st is not None:
                st.last_sent_at = at


# ---------------------------------------------------------------------------
# Dongu
# ---------------------------------------------------------------------------

def _gkey(at: datetime) -> str:
    return f"{at:%Y%m%dT%H%MZ}"


async def _tick(now: datetime) -> None:
    await _refresh_calendar()
    fresh = await _poll_fj()
    all_hl = list(_headlines.values())

    # Aciklanan degerler (Telegram olmasa da: web kutusu kullanir)
    for ev in _events:
        if ev.info.kind in ("data", "oil") and ev.key not in _results and ev.at <= now <= ev.at + timedelta(hours=24):
            res = match_result(ev, all_hl)
            if res is not None:
                _results[ev.key] = res

    if not is_configured():
        return

    # 09:00 ozeti
    n = _tsi(now)
    morning_at = n.replace(hour=NEWS_MORNING_HOUR, minute=0, second=0, microsecond=0)
    mkey = f"morning|{n:%Y-%m-%d}"
    if morning_at <= n < morning_at + _MORNING_GRACE and mkey not in _sent:
        if _cal_ok_at is None or now - _cal_ok_at > timedelta(minutes=30):
            await _refresh_calendar(force=True)
        await _send_once(mkey, "morning", format_morning(now, _events, _holidays, _cal_error))

    for at, group in group_events(_events).items():
        gk = _gkey(at)
        pre_key = f"pre|{gk}"
        # On-bildirim
        if timedelta(0) < at - now <= _ALERT_BEFORE + timedelta(seconds=_TICK_SEC) and pre_key not in _sent:
            await _send_once(pre_key, "pre", format_alert(at, group, now))
        if now < at or now - at > _RESULT_MAX_AGE + _TALK_WINDOW:
            continue
        reply_to = _sent.get(pre_key)
        # Veri sonucu
        data = [e for e in group if e.info.kind in ("data", "oil")]
        rkey = f"result|{gk}"
        if data and rkey not in _sent and now - at <= _RESULT_MAX_AGE:
            if all(e.key in _results for e in data) or now - at >= _RESULT_TIMEOUT:
                await _send_once(rkey, "result", format_result(at, data, _results), reply_to)
        # Konusma / karar metni
        talk = [e for e in group if e.info.kind in ("talk", "text")]
        tkey = f"talk|{gk}"
        if talk and tkey not in _sent:
            window = _TALK_WINDOW if any(e.info.kind == "talk" for e in talk) else _TEXT_WINDOW
            if window <= now - at <= window + _RESULT_MAX_AGE:
                found = {e.key: talk_headlines(e, all_hl, window) for e in talk}
                await _send_once(tkey, "talk", format_talk(at, talk, found), reply_to)

    await _handle_sudden(now, fresh)


def _scheduled_talk_covers(h: Headline) -> bool:
    """Takvimdeki bir konusmanin penceresindeki baslik ani haber sayilmaz (talk reply'i zaten verir)."""
    for ev in _events:
        if ev.info.kind in ("talk", "text") and ev.info.fj and re.search(ev.info.fj, h.title, re.I):
            if ev.at - timedelta(hours=1) <= h.at <= ev.at + timedelta(minutes=90):
                return True
    return False


async def _handle_sudden(now: datetime, fresh: list[Headline]) -> None:
    for h in fresh:
        if h.at < _started_at - timedelta(minutes=5):
            continue
        hit = sudden_category(h.title)
        if hit is None or _scheduled_talk_covers(h):
            continue
        cat, label, meaning = hit
        st = _sudden.get(cat)
        if st is None or now - st.root_at > _SUDDEN_CHAIN:
            msg_id = await _send_once(f"sudden|{cat}|{h.guid}", "sudden", format_sudden(label, meaning, h))
            if msg_id is not None:
                _sudden[cat] = _SuddenState(msg_id, now, now)
        else:
            st.pending.append(h)
    for cat, st in _sudden.items():
        if st.pending and now - st.last_sent_at >= _SUDDEN_COOLDOWN:
            label = next(l for c, l, _, _ in SUDDEN_CATS if c == cat)
            msg_id = await _send_once(f"digest|{cat}|{now:%Y%m%dT%H%M%S}", "digest",
                                      format_digest(label, st.pending), st.root_id)
            if msg_id is not None:
                st.pending.clear()
                st.last_sent_at = now


async def _loop() -> None:
    await _load_sent()
    while True:
        try:
            await _tick(datetime.now(UTC))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("NEWS tick failed")
        await asyncio.sleep(_TICK_SEC)


async def start() -> None:
    global _task, _started_at
    if not NEWS_ENABLED or _task is not None:
        return
    _started_at = datetime.now(UTC)
    _task = asyncio.create_task(_loop(), name="news")
    log.info("NEWS started (currencies=%s, telegram=%s)", ",".join(NEWS_CURRENCIES),
             "on" if is_configured() else "OFF — NEWS_TELEGRAM_BOT_TOKEN/CHAT_ID yok")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None


# ---------------------------------------------------------------------------
# Web (Open Signals haber kutusu)
# ---------------------------------------------------------------------------

def _in_text(delta: timedelta) -> str:
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"in {mins}m"
    h, m = divmod(mins, 60)
    return f"in {h}h {m:02d}m" if h < 24 else f"in {h // 24}d {h % 24}h"


def web_context(now: datetime | None = None) -> dict:
    """Bugunun (TSI) henuz aciklanmamis haberleri + bugunku tatiller (kullanici istegi 30.09:
    gecmis haber ve yarin gosterilmez). Arayuz metinleri Ingilizce."""
    now = now or datetime.now(UTC)
    n = _tsi(now)
    rows = []
    for ev in _events:
        t = _tsi(ev.at)
        if t.date() != n.date() or ev.at <= now:
            continue
        rows.append({
            "time": t.strftime("%H:%M"),
            "impact": ev.impact,
            "title": ev.title,
            "forecast": _clean(ev.forecast) or "",
            "previous": _clean(ev.previous) or "",
            "in": _in_text(ev.at - now),
            "soon": ev.at - now <= timedelta(hours=1),
        })
    hols = [{"country": h.country, "title": h.title} for h in _holidays if _tsi(h.at).date() == n.date()]
    return {
        "enabled": NEWS_ENABLED,
        "rows": rows,
        "holidays": hols,
        "currencies": "/".join(NEWS_CURRENCIES),
        "error": _cal_error if _cal_ok_at is None else None,
        "loaded": _cal_ok_at is not None,
        "telegram": is_configured(),
    }
