"""Izleme konularinin kaydi -- /izleme sayfasinin veri kaynagi.

IZLEME.md gerekce dosyasi olarak kalir (1000+ satir duz yazi, gitignore'lu); bu modul onun
INDEKSI: her basligin **statusu**, **tetikleyicisi** ve markdown'in veremedigi tek sey olan
**canli ilerlemesi** (tetige ne kadar kaldi). Sayfa detay metnini yine IZLEME.md'den okur --
prosa tek yerde kalsin diye. Eslesme `md` alanindaki BASLIK METNI uzerinden yapilir.

Yeni izleme basligi acarken iki yere birden yaz:
  1. IZLEME.md -> gerekce, olcum, karar kurali, sinirlar (uzun hali)
  2. buraya    -> bir WatchItem satiri (status + tetik + varsa ilerleme fonksiyonu)
Baslik metnini sonradan degistirirsen `md` alanini da guncelle; sayfa eslesmeyen basligi
"detay bulunamadi" diye gosterir, sessizce yanlis metin basmaz.

Karar verilip bir baslik kapandiginda: `status="done"` + `result` (tek cumle sonuc) yaz.
Kapanmis basliklar sayfada AYRI tabloda durur, acik olanlarla karismaz.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

from markupsafe import Markup
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BiasJournal, PotentialNotice, SetupJournal, Signal
from app.setup_journal import PRE_SETUP_STAGES

IZLEME_PATH = Path(__file__).resolve().parent.parent / "IZLEME.md"

# Statu -> (etiket, renk sinifi). "action" statusu registry'de yazilmaz; tetigi dolan
# acik baslik sayfada kendiliginden oraya tasinir.
STATUS_META = {
    "open": ("İzleniyor", "text-accent-blue", "bg-accent-blue/10"),
    "done": ("Karara bağlandı", "text-gray-400", "bg-dark-700"),
}


# Onem = maddenin cevabi motoru/parayi ne kadar degistirebilir (kullanici istegi 21.09:
# "onemsizler alt siralarda olsun"). Tetige yakinlik ikincil anahtar oldu -- bir bilgi maddesi
# tetigi dolu diye bir kural maddesinin ustune cikmasin.
#   1 = bir KURALI degistirebilir (giris/stop/kapi/skor) ya da para dogrudan etkilenir
#   2 = dogrulama: canliya alinmis bir degisiklik tuttu mu, kapi ne eliyor
#   3 = bilgi / saglik / bildirim ergonomisi -- motor karari degismez
ONEM_META: dict[int, tuple[str, str]] = {
    # 0 (01.10, kullanici): 21 acik madde "Yuksek"ti, sira bir sey soylemiyordu. Cok yuksek = kullanicinin
    # actigi VE canli bir kurali degistirebilecek madde (cogu orneklemi dolmus). Az tut -- hepsi cok yuksekse yine ayni sorun.
    0: ("Çok yüksek", "text-accent-red font-bold"),
    1: ("Yüksek", "text-accent-yellow"),
    2: ("Orta", "text-accent-blue"),
    3: ("Düşük", "text-gray-500"),
}


@dataclass
class Bar:
    """Tek bir ilerleme cubugu: 'LONG 15/30' gibi."""
    label: str
    current: int
    target: int

    @property
    def pct(self) -> int:
        if self.target <= 0:
            return 100
        return max(0, min(100, round(self.current / self.target * 100)))

    @property
    def ready(self) -> bool:
        return self.current >= self.target


@dataclass
class Progress:
    bars: list[Bar] = field(default_factory=list)
    due: Optional[date] = None          # takvime bagli basliklar (⏰ tarih)
    note: str = ""                      # serbest bilgi satiri (ort. R, sayac vb.)
    # Tetik "TARIH YA DA ORNEKLEM" (trigger metninde "ya da"): hangisi once dolarsa. 28.09'a kadar
    # yoktu; 7 madde "ya da" yazdigi halde VE ile degerlendiriliyor, ornegi dolan madde 03.10'u bekliyordu.
    either: bool = False
    # 22.10: surekli izleme maddesinin tablosu (ör. strateji x TP/SL). Ilk satir baslik; cubuk/tetik degildir.
    table: list[list[str]] = field(default_factory=list)

    @property
    def days_left(self) -> Optional[int]:
        if not self.due:
            return None
        return (self.due - datetime.now(timezone.utc).date()).days

    @property
    def ready(self) -> bool:
        """Tetik doldu mu: TUM cubuklar dolmali ve (varsa) tarih gelmis olmali.

        `either`: tarih gelmesi YA DA butun cubuklarin dolmasi yeter.
        """
        if self.either:
            date_ok = self.due is not None and (self.days_left or 0) <= 0
            bars_ok = bool(self.bars) and all(b.ready for b in self.bars)
            return date_ok or bars_ok
        if self.due is not None and (self.days_left or 0) > 0:
            return False
        if self.bars:
            return all(b.ready for b in self.bars)
        return self.due is not None


@dataclass
class WatchItem:
    key: str
    title: str
    status: str                 # "open" | "done"
    started: str                # "09.09" gibi
    trigger: str                # ne olursa bu basliga donulecek
    md: str                     # IZLEME.md'deki BASLIK METNI (## isareti olmadan)
    measure: str = ""           # calistirilacak komut / bakilacak sayfa
    result: str = ""            # status="done" ise tek cumle sonuc
    onem: int = 2               # 0 cok yuksek / 1 yuksek / 2 orta / 3 dusuk -- ONEM_META, siralamanin BIRINCIL anahtari
    pinned: bool = False        # kullanici "en uste" dedi (29.09): kumesinin basinda, onemden bile once
    pin_order: int = 9          # 03.10: sabitlenenler arasi sira (kucuk once); bias en onemli konu -> 0
    continuous: bool = False    # 22.10: tetigi yok, kapanmaz (sistem takibi) -- kartta "%" yerine "sürekli"
    progress_fn: Optional[Callable[[AsyncSession], Awaitable[Progress]]] = None


# ---------------------------------------------------------------------------------------
# Canli ilerleme hesaplari. Hepsi tek tek sayim -- SQLite'ta onemsiz maliyet.
# Hata yutulur (asagida `_safe`): izleme sayfasi asla sayfayi cokertmesin.
# ---------------------------------------------------------------------------------------

async def _closed_count(db: AsyncSession, *where) -> int:
    return await db.scalar(
        select(func.count(Signal.id)).where(Signal.closed_at.is_not(None), *where)
    ) or 0


async def _p_noncrypto(db: AsyncSession) -> Progress:
    n = await _closed_count(db, Signal.market_type != "crypto")
    return Progress(bars=[Bar("kripto-dışı kapalı işlem", n, 20)])


async def _p_partial(db: AsyncSession) -> Progress:
    n = await _closed_count(db, Signal.partial_size.is_not(None), Signal.partial_rr.is_not(None))
    # 03.10 (kullanici): esik 25 -> 100 (fark islem basina cok oynak; ~1 islem/gun -> ~80 gun)
    return Progress(bars=[Bar("kısmi kârlı kapalı işlem", n, 100)])


async def _p_partial_flat(db: AsyncSession) -> Progress:
    """%50'ye gelip iki kolu da cozulmus motor setup'u; `scripts/partial_flat_stat.py` KARAR (temiz) dilimiyle
    ayni sayar: pt temiz, ayni mum/ufuk disi haric, 4H'te seviyeleri C2 kapaliyken donmus + 1D.

    28.09 (kullanici onayi): eski sayac tum 4H/1D'yi sayiyordu (118), %77'si C2 acikken donmus 4H satiriydi
    (kosan SL) ve temiz dilim (27) ters isaret veriyordu. Esik 100 -> 200, tarih yok (orneklem tetigi)."""
    t = SetupJournal.__table__
    k1 = lambda f: func.json_extract(t.c.shadow, f'$."1".{f}')  # noqa: E731
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.shadow.is_not(None), t.c.strategy.in_(("4h", "1d")),
            t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
            # 03.10: 1D'de yalniz seviyesi kesin satir (eskiden 1D'nin hepsi temiz sayiliyordu, %88'i C2 acik donmustu)
            or_(and_(t.c.strategy == "1d", _final_levels_cond(t, "1d")),
                and_(t.c.strategy == "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1)),
            k1("pt") == 1, k1("pa").is_not(None), k1("cur").in_(("tp", "be")), k1("o").in_(("win", "loss")))
    ) or 0
    return Progress(bars=[Bar("%50'ye gelip çözülmüş temiz setup", n, 200)])


async def _p_direction(db: AsyncSession) -> Progress:
    rows = await db.execute(
        select(Signal.direction, func.count(Signal.id))
        .where(Signal.closed_at.is_not(None), Signal.closed_at >= datetime(2026, 9, 8),
               Signal.rr_value.is_not(None))
        .group_by(Signal.direction)
    )
    c = {d: n for d, n in rows.all()}
    return Progress(bars=[Bar("LONG", c.get("LONG", 0), 30), Bar("SHORT", c.get("SHORT", 0), 30)])


async def _p_trail(db: AsyncSession) -> Progress:
    n = await _closed_count(db, Signal.exit_reason == "trail")
    avg = await db.scalar(
        select(func.avg(Signal.rr_value)).where(Signal.exit_reason == "trail",
                                                Signal.rr_value.is_not(None))
    )
    note = f"trail çıkışlarının ort. R: {avg:+.2f}" if avg is not None else "henüz trail çıkışı yok"
    return Progress(bars=[Bar("trail ile kapanmış işlem", n, 10)], note=note)


async def _p_c2_penalty(db: AsyncSession) -> Progress:
    n = await _closed_count(db, Signal.closed_at >= datetime(2026, 9, 9))
    return Progress(bars=[Bar("09.09'dan beri kapalı işlem", n, 30)])


async def _p_be_1d_1h(db: AsyncSession) -> Progress:
    n = await _closed_count(db, Signal.timeframe == "4h", Signal.protection_armed_time.is_not(None))
    return Progress(bars=[Bar("4H'te koruma devreye girmiş işlem", n, 20)])


def _journal_table():
    return SetupJournal.__table__


def _journal_col(name: str):
    return _journal_table().c[name]


async def _p_shadow(db: AsyncSession) -> Progress:
    """Gölge izleme + alt başlık "Sürekli k eğrisi"nin tetiği (60 çözülmüş kazanan).

    Sayı `bars` DEĞİL `note`: `Progress.ready` hem tarihi hem TÜM çubukları ister, çubuk eklemek
    28.09 hatırlatmasını dolmamış eşiğin arkasına saklardı. Kardeş "retrace" maddesinin çubuğu bu
    iş için kullanılamaz — o `tp_first`'i sayar, içinde entry hiç dolmamış (`tp_before_entry`)
    satırlar da var; eğri yalnız GERÇEKTEN dolup TP'ye giden işlemi kullanır.
    """
    t = _journal_table()
    n = await db.scalar(select(func.count()).select_from(t).where(t.c.shadow.is_not(None))) or 0
    egri = await db.scalar(
        select(func.count()).select_from(t)
        .where(t.c.retrace.is_not(None), t.c.outcome == "win")
    ) or 0
    return Progress(note=f"{n} setup gölge izlemede · sürekli k eğrisi için çözülmüş kazanan {egri}/60")


async def _p_retrace(db: AsyncSession, strategy: Optional[str] = None, target: int = 30) -> Progress:
    """first_bar'li (26.09+, yanlissiz) bugunku girisle DOLUP TP'ye giden setup (limit seviyesi karar dilimi).

    01.10: strateji bazinda uc maddeye bolundu (`strategy`, kullanici: "30 az"); C2 kapali donmus sarti yalniz 4H'te
    (1D/1W 25.09'dan beri C2 acikken islem aciyor; rapor `tmp/tmp_limit_variant_depth.py --strategy` ayni kural).
    Tetik yalniz orneklem -- tarih yok.
    """
    t = SetupJournal.__table__
    base = [
        t.c.retrace.is_not(None), t.c.first_bar.is_not(None), t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
        or_(t.c.strategy != "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
    ]
    if strategy:
        base.append(t.c.strategy == strategy)
    n = await db.scalar(select(func.count()).select_from(t).where(*base)) or 0
    # 28.09: kazanan = bugunku girisle DOLUP TP'ye giden (d_of.chosen <= d_tp). Eskiden `tp_first` sayiliyordu;
    # dolmadan TP'ye gidenler de girdigi icin sayac 67/30 gosterirken esli tablolarda varyant basina 5-8 islem
    # doluyordu -- madde erken "aksiyon bekliyor"a dustu (sayac karar kuralinin saydigini saymali).
    d_ch = func.json_extract(t.c.retrace, "$.d_of.chosen")
    win = await db.scalar(
        select(func.count()).select_from(t).where(
            *base, func.json_extract(t.c.retrace, "$.tp_first") == 1, d_ch >= 0,
            d_ch <= func.coalesce(func.json_extract(t.c.retrace, "$.d_tp"), 0),
        )
    ) or 0
    return Progress(bars=[Bar("first_bar'lı dolmuş kazanan", win, target)], note=f"{n} first_bar'lı setup ölçekte")


def _p_retrace_for(strategy: str, target: int):
    async def fn(db: AsyncSession) -> Progress:
        return await _p_retrace(db, strategy, target)
    return fn


_CORE_SINCE = datetime(2026, 10, 4)     # 03.10 bulgulari gorulen veriden -> karar yalniz taze veriyle


async def _core_4h_rows(db: AsyncSession) -> list[dict]:
    """03.10 cekirdek maddeleri (min RR / skor sirasi / segment) icin ortak kume: TAZE (levels_at >= 04.10) motor
    setup'u (CRT-oncesi aday degil), 4H'te C2 kapali donmus, `entries.chosen` win/loss ile cozulmus, `imm` degil,
    RR >= 2. Kapilardan gecmis olmasi SART DEGIL -- skor ve bias sorusu kapida elenenleri de gormek zorunda.
    Satir: market, rr, skor (donmus kopya), htf kalemi, gate (kalite kapilarini gecti mi), R (duz TP/SL)."""
    t = SetupJournal.__table__
    rows = await db.execute(
        select(t.c.market_type, t.c.best_stage, t.c.score, t.c.entries, t.c.parts_at_levels).where(
            t.c.strategy == "4h", t.c.entries.is_not(None), t.c.levels_at >= _CORE_SINCE,
            t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
        )
    )
    out = []
    for mk, stage, score, ent_raw, pal_raw in rows.all():
        pal = json.loads(pal_raw) if pal_raw else {}
        if pal.get("c2_closed") != 1:
            continue
        ch = (json.loads(ent_raw) or {}).get("chosen") or {}
        if ch.get("o") not in ("win", "loss") or ch.get("imm") or (ch.get("rr") or 0) < 2:
            continue
        out.append(dict(mk=mk, rr=ch["rr"], score=pal.get("score", score), htf=pal.get("htf"),
                        gate=stage not in ("bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr",
                                           "score7", "tight_stop"),
                        R=ch["rr"] if ch["o"] == "win" else -1.0))
    return out


def _r_note(groups: dict[str, list[dict]]) -> str:
    parts = []
    for name, v in groups.items():
        if v:
            parts.append(f"{name} n {len(v)} R/setup {sum(x['R'] for x in v) / len(v):+.2f}")
    return " · ".join(parts)


async def _p_min_rr(db: AsyncSession) -> Progress:
    rows = [x for x in await _core_4h_rows(db) if x["gate"]]
    lo = [x for x in rows if x["rr"] < 2.5]
    hi = [x for x in rows if x["rr"] >= 2.5]
    return Progress(bars=[Bar("taze 4H, RR 2–2.5", len(lo), 50), Bar("taze 4H, RR ≥ 2.5", len(hi), 50)],
                    note=_r_note({"RR 2–2.5": lo, "RR ≥ 2.5": hi}))


async def _p_score_order(db: AsyncSession) -> Progress:
    rows = [x for x in await _core_4h_rows(db) if x["score"] is not None]
    lo = [x for x in rows if x["score"] < 7]
    hi = [x for x in rows if x["score"] >= 7]
    return Progress(bars=[Bar("taze 4H, skor < 7", len(lo), 60), Bar("taze 4H, skor ≥ 7", len(hi), 60)],
                    note=_r_note({"skor < 7": lo, "skor ≥ 7": hi}))


async def _p_segment(db: AsyncSession) -> Progress:
    rows = [x for x in await _core_4h_rows(db) if x["gate"]]
    cr = [x for x in rows if x["mk"] == "crypto"]
    fx = [x for x in rows if x["mk"] != "crypto"]
    return Progress(bars=[Bar("taze 4H kripto, kapı geçmiş", len(cr), 60), Bar("taze 4H FX/metal/endeks", len(fx), 30)],
                    note=_r_note({"kripto": cr, "FX": fx}))


async def _p_rr_cap(db: AsyncSession) -> Progress:
    """RR tavani maddesi: first_bar'li, kapilari gecmis, sonuclanmis, GIRISI DEGISEN setup (havuz, esik 20) --
    bugunku RR > 4 ve 3-4R bandinda motorun bildigi bir aday giris (CISD/MSS/IFVG/BPR yakin kenar) var.

    Sayac `scripts/rr_cap_stat.py` karar dilimini birebir sayar; RR retrace ekseninden hesaplandigi icin (giris
    fiyatin gerisindeyse market, TP'si gecilmis satir haric) SQL yerine Python'da.
    """
    t = SetupJournal.__table__
    rows = await db.execute(
        select(t.c.strategy, t.c.direction, t.c.tp, t.c.retrace, t.c.parts_at_levels).where(
            t.c.retrace.is_not(None), t.c.first_bar.is_not(None), t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
            t.c.best_stage.not_in(("bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr", "score7",
                                   "tight_stop")),
        )
    )
    per: dict[str, int] = {}
    aff = 0
    for strat, direction, tp, rt_raw, pal_raw in rows.all():
        rt = json.loads(rt_raw)
        if strat == "4h" and (json.loads(pal_raw) if pal_raw else {}).get("c2_closed") != 1:
            continue
        d, span, ref = (rt.get("d_of") or {}).get("chosen"), rt.get("span"), rt.get("ref")
        if d is None or not span or ref is None or d >= 1 or tp is None or rt.get("amb"):
            continue
        if rt.get("tp_first") is None and not rt.get("done"):
            continue
        rr0 = ((tp - ref) if direction == "LONG" else (ref - tp)) / span
        if rr0 <= 0:
            continue
        rr = lambda x: (rr0 + max(0.0, x)) / (1 - max(0.0, x))  # noqa: E731
        cur = rr(d)
        if cur < 2:
            continue
        aff += cur > 4
        if cur > 4 and any(3 <= rr(c) <= 4 and rr(c) < cur for c in
                           ((rt.get("d_of") or {}).get(k) for k in ("cisd", "mss", "ifvg_near", "bpr_near"))
                           if c is not None and c < 1):
            per[strat] = per.get(strat, 0) + 1
    note = " · ".join(f"{k.upper()} {v}" for k, v in sorted(per.items())) or "henüz yok"
    return Progress(bars=[Bar("first_bar'lı girişi değişen çözülmüş setup", sum(per.values()), 20)],
                    note=f"RR > 4 olan {aff} setupun bandında aday olanlar · strateji: {note}")


async def _p_market_fee(db: AsyncSession, strategy: Optional[str] = None, target: int = 400,
                        note: str = "") -> Progress:
    """first_bar'li (26.09+) cozulmus gercek setup (market vs limit karar dilimi).

    28.09: esik 100 -> 400 (kullanici "iyice emin olmak istiyorum"); 100'de okunan sonuc ara okuma sayildi.
    01.10: setup ilk goruldugunde TP'si gecilmis satirlar sayilmaz (114 satir; sayac 452 gosterip erken tetikledi).
    01.10: strateji bazinda uc maddeye bolundu (`strategy`); C2 kapali donmus sarti yalniz 4H'te -- 1D/1W 25.09'dan
    beri C2 acikken islem aciyor (rapor `--strategy` ile ayni kural).
    Tetik yalniz orneklem: tarih yok, either yok -- tarih tek basina tetiklemesin.
    """
    t = SetupJournal.__table__
    where = [
        t.c.retrace.is_not(None), t.c.first_bar.is_not(None), t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
        or_(t.c.strategy != "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
        or_(t.c.retrace.like('%"tp_first":true%'), t.c.retrace.like('%"tp_first":false%'),
            t.c.retrace.like('%"done":true%')),
        # 01.10: ilk goruldugunde TP'si zaten gecilmis satir islem degil (rapor da haric tutar)
        case((t.c.direction == "LONG", t.c.tp > func.json_extract(t.c.retrace, "$.ref")),
             else_=t.c.tp < func.json_extract(t.c.retrace, "$.ref")),
    ]
    if strategy:
        where.append(t.c.strategy == strategy)
    n = await db.scalar(select(func.count()).select_from(t).where(*where)) or 0
    return Progress(bars=[Bar("first_bar'lı çözülmüş setup", n, target)], note=note)


async def _p_market_gated(db: AsyncSession, strategy: str, note: str) -> Progress:
    """03.10 (kullanici): 4H / 1D / 1W market vs limit kapilari gecmis + first_bar diliminde okunur; market %90 ve %80
    RR >= 2 kollari her biri 50 cozulmus islem (TP+SL). Hesap `scripts/market_vs_limit_stat.py` KARAR blogunun aynisi
    (`trade`): market girisi x = 0, stop SL' = k x span, RR = rr0 / k; burada yalniz islem sayilir, fark rapordan.
    C2 kapali donmus sarti yalniz 4H'te (rapor gibi)."""
    t = SetupJournal.__table__
    qual = ("bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr", "score7", "tight_stop")
    where = [t.c.strategy == strategy, t.c.retrace.is_not(None), t.c.first_bar.is_not(None),
             t.c.parts_at_levels.is_not(None), t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES) + qual)]
    if strategy == "4h":
        where.append(func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1)
    rows = (await db.execute(select(t.c.direction, t.c.tp, t.c.retrace).where(*where))).all()
    counts = {0.9: 0, 0.8: 0}
    for direction, tp, raw in rows:
        try:
            rt = json.loads(raw)
        except Exception:
            continue
        span, ref = rt.get("span"), rt.get("ref")
        if rt.get("amb") or not span or not ref or rt.get("tp_first") is None:
            continue
        d = (rt.get("d_of") or {}).get("chosen")
        rr0 = ((tp - ref) if direction == "LONG" else (ref - tp)) / span
        if d is None or d >= 1 or rr0 <= 0:
            continue
        for k in counts:
            if rr0 / k >= 2.0:  # cozulmus = tp_first True (TP ya da SL) / False (SL); market x=0 her zaman dolar
                counts[k] += 1
    return Progress(bars=[Bar("market %90 RR ≥ 2 çözülmüş işlem", counts[0.9], 50),
                          Bar("market %80 RR ≥ 2 çözülmüş işlem", counts[0.8], 50)], note=note)


def _p_market_gated_for(strategy: str, note: str):
    async def fn(db: AsyncSession) -> Progress:
        return await _p_market_gated(db, strategy, note)
    return fn


async def _p_bias_outcome(db: AsyncSession, strategy: str, since: Optional[datetime], need: int,
                          note: str) -> Progress:
    """03.10 (kullanici: "bias en onemli konu"): temiz setuplarda bias'i hizali ve ters cozulmus setup, her biri `need`
    (4H taze levels_at >= 04.10 / 100; 1D butun veri / 50). Bias = seviye aninda KAPANMIS son gunun karne bias'i
    (bias_journal.combined) -- `setup_journal.htf_bias` last_seen anina ait, kullanilmaz. C2 kapali donmus sarti yalniz
    4H'te. Kural ve hesap `scripts/bias_outcome_stat.py` ile ayni."""
    t = SetupJournal.__table__
    where = [t.c.strategy == strategy, t.c.levels_at.is_not(None), t.c.parts_at_levels.is_not(None),
             t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)), t.c.outcome.in_(("win", "loss", "signal"))]
    if since is not None:
        where.append(t.c.levels_at >= since)
    if strategy == "4h":
        where.append(func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1)
    where.append(_final_levels_cond(t, strategy))
    rows = (await db.execute(select(t.c.symbol, t.c.direction, t.c.levels_at, t.c.outcome, t.c.shadow)
                             .where(*where))).all()
    kq = select(BiasJournal.symbol, BiasJournal.day, BiasJournal.combined).order_by(BiasJournal.day)
    if since is not None:
        kq = kq.where(BiasJournal.day >= since - timedelta(days=3))
    karne: dict[str, list] = {}
    for sym, day, comb in (await db.execute(kq)).all():
        karne.setdefault(sym, []).append((day, comb))
    n = {"hizali": 0, "ters": 0}
    for sym, direction, lv, outcome, shadow in rows:
        if outcome == "signal":
            try:
                outcome = (json.loads(shadow) if shadow else {}).get("1", {}).get("o")
            except Exception:
                continue
            if outcome not in ("win", "loss"):
                continue
        closed = [c for d, c in karne.get(sym, []) if d + timedelta(hours=24) <= lv]
        b = closed[-1] if closed else None
        if b in ("BULLISH", "BEARISH"):
            n["hizali" if (direction == "LONG") == (b == "BULLISH") else "ters"] += 1
    label = "taze 4H" if strategy == "4h" else strategy.upper()
    return Progress(bars=[Bar(f"{label}, bias hizalı", n["hizali"], need), Bar(f"{label}, bias ters", n["ters"], need)],
                    note=note)


def _p_bias_outcome_for(strategy: str, since: Optional[datetime], need: int, note: str):
    async def fn(db: AsyncSession) -> Progress:
        return await _p_bias_outcome(db, strategy, since, need, note)
    return fn


async def _p_reclaim_penalty(db: AsyncSession) -> Progress:
    """Zayif C2 geri donusu cezasi (-4) alan, ham skoru esigi gecen, cozulmus Journal setupu -- strateji basina 20.

    Rapor `scripts/reclaim_penalty_stat.py` ile ayni kume: 19.09+ (ceza o gun -4 oldu), 4H'te C2 kapali donmus,
    ham skor = raw - reclaim >= esik (tavan SMT'den once uygulandigi icin bu esitlik tavan durumunda da dogru).
    Sonuc `outcome` win/loss; setup sinyale donustuyse `shadow["1"].o`.
    """
    t = SetupJournal.__table__
    pal = t.c.parts_at_levels
    res = or_(t.c.outcome.in_(("win", "loss")),
              (t.c.outcome == "signal") & func.json_extract(t.c.shadow, '$."1".o').in_(("win", "loss")))
    bars = []
    for strat, thr in (("4h", 7), ("1d", 7), ("1w", 6)):
        if strat == "4h":
            def part(k):
                return func.json_extract(pal, f"$.{k}")
        else:
            # 03.10: 1D/1W'de skor `analysis_parts` ile ayni -- parts_final (sinyal / C2 kapanisi), yoksa son skor;
            # yalniz seviyesi kesin satir (`_final_levels_cond`). Eskiden parts_at_levels: 1D "21/20" sayiyordu, dogrusu 1.
            def part(k):
                return func.coalesce(func.json_extract(t.c.parts_final, f"$.parts.{k}"),
                                     func.json_extract(t.c.score_parts, f"$.{k}"))
        rec = part("reclaim")
        raw = func.coalesce(part("raw"), part("score"))
        n = await db.scalar(select(func.count()).select_from(t).where(
            t.c.strategy == strat, t.c.first_seen >= datetime(2026, 9, 19), t.c.entry.is_not(None),
            t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)), rec == -4, raw - rec >= thr, res,
            or_(t.c.strategy != "4h", func.json_extract(pal, "$.c2_closed") == 1),
            _final_levels_cond(t, strat),
        )) or 0
        bars.append(Bar(f"{strat.upper()} ceza alan, ham skor ≥ {thr}", n, 20))
    return Progress(bars=bars, note="03.10: 4H okundu (ceza kalır, 58 çözülmüş) · 1D/1W yalnız seviyesi kesin satırla — "
                                    "veri birikiyor")


async def _p_entry_models(db: AsyncSession) -> Progress:
    """Entry varyanti tasiyan setup sayisi (karar esigi varyant basina 60)."""
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(_journal_col("entries").is_not(None))
    ) or 0
    return Progress(bars=[Bar("entry varyantli setup", n, 60)])


async def _p_bpr(db: AsyncSession) -> Progress:
    """BPR bolgesi olan setup sayisi (karar esigi 40 cozulmus setup)."""
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(_journal_col("entries").like('%bpr_near%'))
    ) or 0
    return Progress(bars=[Bar("BPR'li setup", n, 40)])


async def _p_bpr_leg(db: AsyncSession) -> Progress:
    """Karar verilebilir `bfvg_near` sayisi (karar esigi 40).

    `bfvg_near` TASIYAN satiri saymak yetmiyor (24.09): cogu `imm` (seviye donarken fiyat
    adayin gerisinde) ya da henuz sonuclanmamis -- 96 satirda karar verilebilir yalniz 33 vardi
    ve madde erken "aksiyon bekliyor"a dustu. Kurallar `entry_model_stat.tally` ile ayni.
    Gerekce: IZLEME.md "BPR bacagi".
    """
    t = _journal_table()
    rows = (await db.execute(
        select(_journal_col("entries")).select_from(t)
        .where(_journal_col("entries").like('%bfvg_near%'))
    )).scalars().all()
    n = 0
    for raw in rows:
        try:
            v = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("bfvg_near") or {}
        except (ValueError, AttributeError):
            continue
        if v.get("o") in ("win", "loss", "tp_before_entry", "no_touch") and not v.get("imm"):
            n += 1
    return Progress(bars=[Bar("karar verilebilir bacak setup'ı", n, 40)])


async def _p_cfvg(db: AsyncSession) -> Progress:
    """CISD kirilim FVG'si olcumu tasiyan setup sayisi (karar esigi 60 eslenmis setup).

    22.09'dan itibaren yazilir. Pencere FVG'siz kapanirsa da satir yazilir (`no_fvg`), yani
    sayac "model uygulanabildi mi" degil "olcum tamamlandi mi" sayar; karar verilebilir
    (eslenmis) alt kume raporda. Gerekce: IZLEME.md "CISD kirilim FVG'si".
    """
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(_journal_col("entries").like('%cfvg_near%'))
    ) or 0
    return Progress(bars=[Bar("kırılım FVG'si ölçülen setup", n, 60)])


async def _p_1d_c2_open(db: AsyncSession) -> Progress:
    """25.09 sonrasi C2 acikken (purge + 24s'ten once) dolup kapanan 1D islem."""
    n = await _closed_count(
        db, Signal.timeframe == "1d", Signal.entry_filled_time >= datetime(2026, 9, 25),
        Signal.purge_time.is_not(None),
        func.julianday(Signal.entry_filled_time) < func.julianday(Signal.purge_time) + 1.0,
    )
    return Progress(bars=[Bar("C2 açıkken dolan kapalı 1D işlem", n, 10)])


async def _p_purge_threshold(db: AsyncSession) -> Progress:
    """Restart sonrasi esik alti supurmeli (features.sweep_pct < 0.05), sonuclanmis Journal satiri (esik 30)."""
    t = SetupJournal.__table__
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.first_seen >= datetime(2026, 9, 28),
            func.json_extract(t.c.features, "$.sweep_pct") < 0.05,
            t.c.outcome.in_(("win", "loss")),
        )
    ) or 0
    return Progress(bars=[Bar("eşik altı sonuçlanmış setup", n, 30)])


async def _p_nearest_zone(db: AsyncSession) -> Progress:
    """Restart sonrasi (28.09+) bolge girisli (ifvg/bpr) kapali sinyal (karar esigi 10)."""
    n = await _closed_count(
        db, Signal.entry_model.in_(("ifvg", "bpr")), Signal.created_at >= datetime(2026, 9, 28),
    )
    return Progress(bars=[Bar("bölge girişli kapalı sinyal (28.09+)", n, 10)])


async def _kl_count(db: AsyncSession, cond) -> int:
    """29.09+ Journal satiri, key level olcumu dolu, sonuclanmis (win/loss), KARAR KUMESI + ek kosul.

    01.10: karar kumesi IZLEME.md'deki gibi kapilari gecmis + RR >= 2 + 4H'te C2 kapali donmus. Eskiden butun
    satirlari (CRT sayilmayan adaylar ve kapida elenenler dahil) sayiyordu; madde 407/177 ile "aksiyon bekliyor"a
    dustu, karar kumesinde 9 setup vardi.
    """
    t = SetupJournal.__table__
    qual_fail = tuple(PRE_SETUP_STAGES | {"bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr",
                                          "score7", "tight_stop"})
    return await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.first_seen >= datetime(2026, 9, 29),
            func.json_extract(t.c.features, "$.key_level_n").is_not(None),
            t.c.outcome.in_(("win", "loss")),
            t.c.best_stage.not_in(qual_fail), t.c.rr >= 2.0,
            or_(t.c.strategy != "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
            cond(t),
        )
    ) or 0


async def _p_key_level_filter(db: AsyncSession) -> Progress:
    """Key level'siz (key_level_n = 0) ve key level'li sonuclanmis setup; her grupta 20."""
    none = await _kl_count(db, lambda t: func.json_extract(t.c.features, "$.key_level_n") == 0)
    some = await _kl_count(db, lambda t: func.json_extract(t.c.features, "$.key_level_n") > 0)
    return Progress(bars=[Bar("key level'siz", none, 30), Bar("key level'li", some, 30)])


async def _p_liq_beyond(db: AsyncSession) -> Progress:
    """Purge otesinde 0.5 ATR icinde alinmamis likidite olan / olmayan sonuclanmis setup; her grupta 20."""
    near = await _kl_count(db, lambda t: func.json_extract(t.c.features, "$.liq_beyond_atr") <= 0.5)
    far = await _kl_count(db, lambda t: func.coalesce(func.json_extract(t.c.features, "$.liq_beyond_atr"), 99) > 0.5)
    return Progress(bars=[Bar("yakında (≤ 0.5 ATR)", near, 20), Bar("uzak / yok", far, 20)])


async def _p_liq_near_bonus(db: AsyncSession) -> Progress:
    """Purge otesinde <= 0.5 ATR alinmamis likidite olan / olmayan sonuclanmis setup, YALNIZ 02.10+ (her grupta 50).

    Hipotez 29.09-01.10 verisinden cikti; karar ayni veriyle verilirse sonuc sisik olur, o yuzden sayac taze veriden
    baslar. Kume: motorun CRT saydigi + RR >= 2 + 4H'te C2 kapali donmus.
    """
    t = SetupJournal.__table__
    notcrt = tuple(PRE_SETUP_STAGES - {"not_selected"})
    atr = func.json_extract(t.c.features, "$.liq_beyond_atr")

    async def _n(cond) -> int:
        return await db.scalar(select(func.count()).select_from(t).where(
            t.c.first_seen >= datetime(2026, 10, 2),
            func.json_extract(t.c.features, "$.key_level_n").is_not(None),
            t.c.outcome.in_(("win", "loss")), t.c.best_stage.not_in(notcrt), t.c.rr >= 2.0,
            or_(t.c.strategy != "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
            cond,
        )) or 0
    near = await _n(atr <= 0.5)
    other = await _n(func.coalesce(atr, 99) > 0.5)
    return Progress(bars=[Bar("yakın (≤ 0.5 ATR)", near, 50), Bar("uzak / yok", other, 50)])


async def _p_c1_shape(db: AsyncSession) -> Progress:
    """TP fitili >= 0.40 / kisa C1 sonuclanmis setup (motorun CRT saydigi, RR >= 2); fitilli grupta 100.

    21.10: H1 (C1-C2 ayni renk) kapandi, sayac H2'ye gecti. C1 olcumu donmus kopyadan (`features_at_levels`):
    30.09 oncesi `features` baska C1'le ezilmis olabilir.
    """
    t = SetupJournal.__table__
    notcrt = tuple(PRE_SETUP_STAGES - {"not_selected"})
    col = func.json_extract(func.coalesce(t.c.features_at_levels, t.c.features), "$.c1_tp_wick_frac")

    async def _n(cond) -> int:
        return await db.scalar(select(func.count()).select_from(t).where(
            col.is_not(None), cond, t.c.outcome.in_(("win", "loss")), t.c.best_stage.not_in(notcrt), t.c.rr >= 2.0,
        )) or 0
    return Progress(bars=[Bar("TP fitili ≥ 0.40", await _n(col >= 0.40), 100), Bar("kısa fitil", await _n(col < 0.40), 100)])


async def _p_cisd_mss(db: AsyncSession) -> Progress:
    """Kapilardan gecmis, dolumdan once kapanisla kirilim olculmus cozulmus setup: sadece CISD / CISD + MSS (20)."""
    import json as _json
    t = SetupJournal.__table__
    qual_fail = tuple(PRE_SETUP_STAGES | {"bias_mismatch", "target_taken", "low_quality", "no_cisd", "low_rr",
                                          "score7", "tight_stop"})
    rows = (await db.execute(select(t.c.entries, t.c.retrace).where(
        func.json_extract(t.c.retrace, "$.brk").is_not(None), t.c.best_stage.not_in(qual_fail),
        func.json_extract(t.c.entries, "$.chosen.o").in_(("win", "loss")),
        func.json_extract(t.c.entries, "$.chosen.rr") >= 2,
    ))).all()
    only = both = 0
    for ents, rt in rows:
        try:
            fill = str(_json.loads(ents)["chosen"].get("t") or "")[:19]
            brk = _json.loads(rt)["brk"]
        except Exception:
            continue
        ca = fill and brk.get("cisd") and str(brk["cisd"])[:19] < fill
        ma = fill and brk.get("mss") and str(brk["mss"])[:19] < fill
        only += 1 if ca and not ma else 0
        both += 1 if ca and ma else 0
    return Progress(bars=[Bar("sadece CISD", only, 20), Bar("CISD + MSS", both, 20)])


async def _p_prefill_signal(db: AsyncSession) -> Progress:
    """Kosusu olculmus kapali gercek sinyal: >= %50 kosan / digerleri -- 03.10'dan beri grup basina 100 (kullanici:
    "10 setup cok az"; 03.10'da 10/10 okumasi kapatilip geri acildi). Hiz ~2 islem/gun/grup -> ~50 gun."""
    hi = await _closed_count(db, Signal.prefill_run >= 0.5, Signal.result.is_not(None))
    lo = await _closed_count(db, Signal.prefill_run < 0.5, Signal.result.is_not(None))
    return Progress(bars=[Bar("≥ %50 koşup dolan", hi, 100), Bar("diğerleri", lo, 100)],
                    note="03.10 ara okuma (10/10): ≥ %50 koşan daha iyi — SL %40 / %90, +0.30 / −0.65 R")


async def _p_ifvg_from_c2(db: AsyncSession) -> Progress:
    """29.09+ bolge girisli (ifvg/bpr) kapali sinyal (esik 10); 28.09 'en yakin bolge' maddesiyle ayni oturumda okunur."""
    n = await _closed_count(
        db, Signal.entry_model.in_(("ifvg", "bpr")), Signal.created_at >= datetime(2026, 9, 29),
    )
    return Progress(bars=[Bar("bölge girişli kapalı sinyal (29.09+)", n, 10)])


async def _p_c2_refreeze(db: AsyncSession) -> Progress:
    """C2 kapanisinda yeniden dondurulan 4H satiri (restart sonrasi dolmaya baslar)."""
    t = SetupJournal.__table__
    n = await db.scalar(
        select(func.count()).select_from(t).where(t.c.strategy == "4h", t.c.pre_c2.is_not(None))
    ) or 0
    # 28.09: cubuk kaldirildi -- yeniden dondurulan satir sayisi (104) karar kumesini (kapilari gecmis kullanilabilir
    # 4H satiri, 28.09'da 3) saymiyordu, madde erken "aksiyon"a dustu. Sayi note'ta, tetik yalniz tarih.
    # 03.10 (kullanici: tarih tetigi yok, yalniz yeterli veri): karar kumesi = kapilari gecmis, yeniden dondurulmus,
    # cozulmus 4H satiri (28.09'daki "tum yeniden dondurulmus satir" sayaci kumeyi saymiyordu).
    usable = await db.scalar(select(func.count()).select_from(t).where(
        t.c.strategy == "4h", t.c.pre_c2.is_not(None),
        t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES | {"bias_mismatch", "target_taken", "low_quality", "no_cisd",
                                                         "low_rr", "score7", "tight_stop"})),
        t.c.outcome.in_(("win", "loss", "signal")),
    )) or 0
    return Progress(bars=[Bar("kapıları geçmiş, yeniden dondurulmuş çözülmüş 4H", usable, 30)],
                    note=f"toplam {n} yeniden dondurulmuş 4H satırı")


async def _p_c2_open_fill(db: AsyncSession) -> Progress:
    """4H: C2 acikken CISD onayli + kapilari gecmis donmus, C2-acik kolu sonuclanmis satir (esik 30)."""
    t = SetupJournal.__table__
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.strategy == "4h", t.c.pre_c2.is_not(None),
            func.json_extract(t.c.pre_c2, "$.stage_at_levels") == "c2_open",
            func.json_extract(t.c.pre_c2, "$.o").in_(("win", "loss", "open", "no_c2_fill")),
        )
    ) or 0
    return Progress(bars=[Bar("karşılaştırılabilir 4H setup", n, 30)])


async def _p_zone_presence(db: AsyncSession) -> Progress:
    """Karar dilimi (C2 kapali donmus, skor >= 7, RR >= 2) cozulmus Journal setupu: bolgeli / bolgesiz (esik 30)."""
    t = SetupJournal.__table__
    pal = func.json_extract(t.c.parts_at_levels, "$.score")
    base = [
        t.c.entries.is_not(None), t.c.outcome.in_(("win", "loss")), t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
        pal >= 7, func.json_extract(t.c.entries, "$.chosen.rr") >= 2,
        or_(t.c.strategy == "1h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
    ]
    has_zone = or_(func.json_extract(t.c.entries, "$.bpr_near").is_not(None),
                   func.json_extract(t.c.entries, "$.ifvg_near").is_not(None))
    var = await db.scalar(select(func.count()).select_from(t).where(*base, has_zone)) or 0
    yok = await db.scalar(select(func.count()).select_from(t).where(*base, ~has_zone)) or 0
    return Progress(bars=[Bar("bölgeli (IFVG/BPR)", var, 30), Bar("bölgesiz", yok, 30)])


async def _p_week_gap(db: AsyncSession) -> Progress:
    """Hafta boslugu kapisinda elenen setup sayisi (karar esigi 10)."""
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            or_(t.c.best_stage == "week_gap", t.c.deleted_reason == "week_gap")
        )
    ) or 0
    return Progress(bars=[Bar("week_gap elemesi", n, 10)])


async def _p_pd_midnight(db: AsyncSession) -> Progress:
    """Duzeltmeden SONRA 00:00-00:30 UTC penceresinde dogan setup sayisi.

    Taban ve hedef oran calisma notlarinda: IZLEME.md -> "Gece yarisi PDH/PDL kaymasi duzeltildi".
    """
    t = SetupJournal.__table__
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.score_parts.is_not(None),
            t.c.first_seen >= datetime(2026, 9, 18, 9, 0),
            func.strftime("%H:%M", t.c.first_seen) < "00:30",
        )
    ) or 0
    return Progress(bars=[Bar("düzeltmeden sonra 00:00–00:30 penceresinde setup", n, 100)])


async def _p_bias_journal(db: AsyncSession) -> Progress:
    """Karnede kac satir var, kac tanesi yonlu + sonucu gelmis (karar esigi ufuk basina 500)."""
    t = BiasJournal.__table__
    n = await db.scalar(select(func.count()).select_from(t)) or 0
    yonlu = await db.scalar(
        select(func.count()).select_from(t)
        .where(t.c.combined.in_(("BULLISH", "BEARISH")), t.c.fwd_3d.is_not(None))
    ) or 0
    # Tarih de kosul: tarih olmadan karne 500 satiri ilk gun astigi icin madde 9 gun erken
    # "aksiyon" diye bagiriyordu. Baslik "ilk okuma 28.09" diyor ama o okuma 21.09'da yapildi
    # (hard filtre kaldi); kalan iki alt sorunun tarihi 03.10 -- trigger ile ayni (24.09).
    return Progress(bars=[Bar("yönlü satır (fwd_3d dolu)", yonlu, 500)],
                    note=f"{n} satır karnede")


_NONSIGNAL_TFS = (("4h", "4H"), ("1d", "1D"), ("1w", "1W"), ("1h", "1H (kapalı)"))


async def _p_nonsignal_outcomes(db: AsyncSession) -> Progress:
    """22.10 (kullanici): aktif sinyale DONUSMEMIS butun Journal setuplari kac TP / kac SL. Surekli izleme, tetik yok.

    "Donusmemis" = `outcome != 'signal'` (sinyale donusen satirin izlemesi durur; waiting'e ulasip dolmadan silinen
    setup izlemeye geri doner ve burada sayilir). TP = `win` (giris doldu -> TP), SL = `loss` (giris doldu -> SL);
    `tp_before_entry`, `ambiguous` (ayni mumda giris+TP), dolmadi/izleniyor ayri. Strateji x {CRT sayilan, motorun CRT
    saymadigi aday (PRE_SETUP_STAGES)}. Seviyeler ilk goruldugu haliyle donmus Journal seviyeleri -- duz TP/SL,
    kismi kar/BE yok; 03.10 oncesi 1D/1W satirlarinin cogu C2 acikken donmus (SL yonune yanli)."""
    t = SetupJournal.__table__
    pre = t.c.best_stage.in_(tuple(PRE_SETUP_STAGES))
    week_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)

    def _n(cond):
        return func.sum(case((cond, 1), else_=0))

    rows = (await db.execute(select(
        t.c.strategy, pre.label("pre"), func.count(),
        _n(t.c.outcome == "win"), _n(t.c.outcome == "loss"),
        _n(t.c.outcome == "tp_before_entry"), _n(t.c.outcome == "ambiguous"),
        _n(and_(t.c.outcome == "win", t.c.first_seen >= week_ago)),
        _n(and_(t.c.outcome == "loss", t.c.first_seen >= week_ago)),
    ).where(or_(t.c.outcome.is_(None), t.c.outcome != "signal")).group_by(t.c.strategy, pre))).all()
    agg = {(r[0], bool(r[1])): [int(x or 0) for x in r[2:]] for r in rows}

    def _wr(tp: int, sl: int) -> str:
        return f"%{tp / (tp + sl) * 100:.1f}" if tp + sl else "—"

    table = [["Strateji", "Küme", "Setup", "TP", "SL", "Win %", "Girişsiz TP", "Aynı mum", "Son 7 gün TP / SL"]]
    tot = [0] * 7
    for code, label in _NONSIGNAL_TFS:
        for is_pre, kume in ((False, "CRT sayılan"), (True, "CRT sayılmayan aday")):
            v = agg.get((code, is_pre))
            if not v or (code == "1h" and not v[0]):
                continue
            n, tp, sl, tbe, amb, tp7, sl7 = v
            tot = [a + b for a, b in zip(tot, v)]
            table.append([label, kume, str(n), str(tp), str(sl), _wr(tp, sl), str(tbe), str(amb), f"{tp7} / {sl7}"])
    n, tp, sl, tbe, amb, tp7, sl7 = tot
    table.append(["Toplam", "", str(n), str(tp), str(sl), _wr(tp, sl), str(tbe), str(amb), f"{tp7} / {sl7}"])
    return Progress(table=table,
                    note="TP / SL = giriş doldu, sonra TP / SL (düz, kısmi kâr yok) · Girişsiz TP = fiyat girişe "
                         "dönmeden TP'ye gitti · Aynı mum = giriş ve TP aynı mumda · son 7 gün = ilk görülme")


async def _p_bias_fidelity(db: AsyncSession) -> Progress:
    """03.10: surekli izleme (kullanici: "bias cok onemli, hata varsa duzeltilmeli"). Calisan surecin baslangicindan
    (03.10 15m/4h yaris duzeltmesi bu surecte canli) beri gun donumundeki (00:00-00:35 UTC) 4H kripto satirlari: motorun
    bias'i (`htf_bias`, last_seen) vs `last_seen`'e kadar kapanmis gunun karne bias'i. Hedef 50, yanlis 0 olmali.
    Gun ici farklar olcum yanilgisi olabilir (30.09 C1 kimligi -- htf_bias bayat kalir); kesin kontrol log'daki
    `SKIPPED (BIAS) ... daily=` satirlari. Ayrintili rapor `scripts/bias_fidelity_stat.py`."""
    try:
        from app.market_data import market_data
        since = market_data.status().get("started_at")
    except Exception:
        since = None
    if since is None:
        return Progress(bars=[Bar("restart sonrası gün dönümü 4H değerlendirmesi", 0, 50)],
                        note="motor çalışmıyor — sayaç çalışan sürecin başlangıcından sayar")
    since = since.replace(tzinfo=None)
    t = SetupJournal.__table__
    rows = (await db.execute(select(t.c.symbol, t.c.last_seen, t.c.htf_bias).where(
        t.c.strategy == "4h", t.c.market_type == "crypto", t.c.last_seen >= since, t.c.htf_bias.is_not(None),
    ))).all()
    karne: dict[str, list] = {}
    for sym, day, comb in (await db.execute(select(BiasJournal.symbol, BiasJournal.day, BiasJournal.combined)
                                            .where(BiasJournal.day >= since - timedelta(days=3))
                                            .order_by(BiasJournal.day))).all():
        karne.setdefault(sym, []).append((day, comb))
    n = bad = 0
    for sym, ls, hb in rows:
        if not (ls.hour == 0 and ls.minute < 35):
            continue
        closed = [c for d, c in karne.get(sym, []) if d + timedelta(hours=24) <= ls]
        if not closed:
            continue
        n += 1
        bad += hb != closed[-1]
    return Progress(bars=[Bar("restart sonrası gün dönümü 4H değerlendirmesi", n, 50)],
                    note=f"yanlış bias {bad}/{n} (0 olmalı) · süreç başlangıcı {since:%d.%m %H:%M} UTC")


async def _p_prefill(db: AsyncSession) -> Progress:
    """Kova basina dolan setup sayisi; karar esigi her iki kovada da 100."""
    t = SetupJournal.__table__
    col = t.c.prefill
    n = await db.scalar(select(func.count()).select_from(t).where(col.is_not(None))) or 0
    # f_max JSON'da; SQLite json_extract ile kovalara ayir (yalniz dolanlar: "hit" dolu).
    dolan = select(func.count()).select_from(t).where(
        col.is_not(None), func.json_extract(col, "$.hit").is_not(None))
    dus = await db.scalar(dolan.where(func.json_extract(col, "$.f_max") < 0.25)) or 0
    yuk = await db.scalar(dolan.where(func.json_extract(col, "$.f_max") >= 0.5)) or 0
    return Progress(
        bars=[Bar("0-25% kovasi (dolan)", dus, 100), Bar(">=50% kovasi (dolan)", yuk, 100)],
        note=f"{n} setup olcekte",
    )


async def _p_gated_potential(db: AsyncSession) -> Progress:
    """Kac "elenen setup" bildirimi gitti; karar girdisi gunluk mesaj sayisi (tahmin ~4)."""
    t = PotentialNotice.__table__
    gated = await db.scalar(
        select(func.count()).select_from(t).where(t.c.gate.is_not(None))
    ) or 0
    ilk = await db.scalar(select(func.min(t.c.created_at)).select_from(t))
    # created_at naive UTC saklaniyor; karsilastirmayi naive tarafta yap.
    simdi = datetime.now(timezone.utc).replace(tzinfo=None)
    gun = max(1.0, (simdi - ilk).total_seconds() / 86400.0) if ilk else 1.0
    return Progress(
        bars=[Bar("elenen setup bildirimi", gated, 40)],
        note=f"günde ~{gated / gun:.1f} mesaj (tahmin ~4; 8'i aşarsa kapsam daraltılacak)",
    )


# c1 varyantinin canliya alindigi an: bundan SONRA seviyelenen her setup'ta bulunmasi gerekir
# (oncesinde yok ve geriye donuk doldurulmaz -- izleme ortasinda stop eklemek sonucu bozar).
_C1_SINCE = datetime(2026, 9, 17, 18, 0)


async def _p_c1_stop(db: AsyncSession) -> Progress:
    """C1 varyanti olan satirlar + kuralin uygulanamadigi (invalid) oran + KAPSAMA boslugu."""
    col = _journal_col("shadow")
    n = await db.scalar(
        select(func.count()).select_from(_journal_table()).where(col.like('%"c1":%'))
    ) or 0
    bad = await db.scalar(
        select(func.count()).select_from(_journal_table()).where(col.like('%"why":%'))
    ) or 0
    # Seviyesi olup c1'i olmayan satir = izleme boslugu; beklenen 0.
    gap = await db.scalar(
        select(func.count()).select_from(_journal_table())
        .where(col.is_not(None), ~col.like('%"c1":%'), _journal_col("levels_at") >= _C1_SINCE)
    ) or 0
    note = f"{n} setup C1 varyantiyla izleniyor"
    if n:
        note += f" · {bad} tanesinde kural uygulanamadi (%{round(100 * bad / n)})"
    note += " · kapsama tam" if not gap else f" · ⚠️ {gap} satırda seviye var ama c1 yok"
    return Progress(note=note)


async def _p_tight_stop_gate(db: AsyncSession) -> Progress:
    """Dar stop kapisinda elenip SONUCU BELLI olan setup sayisi (karar esigi 20) + R/islem."""
    t = _journal_table()
    rows = await db.execute(
        select(t.c.outcome, t.c.rr).where(t.c.best_stage == "tight_stop",
                                          t.c.outcome.in_(("win", "loss")))
    )
    got = rows.all()
    total = sum((float(rr or 0) if o == "win" else -1.0) for o, rr in got)
    note = f"{len(got)} elenen setup çözüldü"
    if got:
        w = sum(1 for o, _ in got if o == "win")
        note += f" · {w} TP · R/işlem {total / len(got):+.2f}"
    return Progress(bars=[Bar("çözülmüş elenen setup", len(got), 20)], note=note)


def _p_tight_stop_tf(strategy: str):
    """22.10 (kullanici): dar stop kapisi strateji bazinda. Kume ve karar `app/tight_stop_outcome` (rapor
    `scripts/tight_stop_stat.py` ile ayni fonksiyonlar): kapiya kadar gecmis setuplarda stop orani < 1.0 = dar.
    Cubuk: dar grupta cozulmus setup (tetik 4H 30 / 1D 20 / 1W 10)."""
    async def _fn(db: AsyncSession) -> Progress:
        from app import tight_stop_outcome as tso
        t = SetupJournal.__table__
        res = await db.execute(select(t).where(t.c.strategy == strategy, t.c.entry.is_not(None),
                                               t.c.first_seen >= datetime.fromisoformat(tso.SINCE[strategy])))
        sub = tso.collect(res.mappings().all(), strategy)
        sm = tso.summary(sub, strategy)
        d, n = sm["dar"], sm["normal"]

        def _r(st):
            return f"{st['w']} TP / {st['l']} SL, {st['rps']:+.2f} R/setup" if st["res"] else "çözülmüş yok"
        note = f"dar: {_r(d)} · normal: {_r(n)}"
        if sm["diff"] is not None:
            note += f" · fark {sm['diff']:+.2f}"
        return Progress(bars=[Bar("dar stoplu çözülmüş setup", d["res"], sm["need"])],
                        note=note + f" — rapor: python scripts/tight_stop_stat.py")
    return _fn


async def _p_missed_quality(db: AsyncSession) -> Progress:
    """Retest skor kapisinda elenip SONUCU BELLI olan setup, C2 zamanlama dilimine gore (dilim basina 30).

    01.10'dan beri iki cubuk: karar (H3) iki dilimin BIRDEN dolmasini istiyor, `Progress.ready` de
    tam bunu yapiyor. Kume raporla ayni: aktif strateji + RR >= 2 (`missed_quality_stat`).
    """
    t = _journal_table()
    rows = (await db.execute(
        select(t.c.outcome, t.c.rr, t.c.strategy, t.c.purge_time, t.c.entry_touched_at)
        .where(t.c.best_stage == "missed_quality", t.c.outcome.in_(("win", "loss")),
               t.c.levels_at.is_not(None),
               t.c.strategy.in_(("4h", "1d", "1w")), t.c.rr >= 2.0)
    )).all()
    span = {"4h": timedelta(hours=4), "1d": timedelta(days=1), "1w": timedelta(weeks=1)}
    early = [r for r in rows if r.purge_time and r.entry_touched_at
             and r.entry_touched_at < r.purge_time + span[r.strategy]]
    late = [r for r in rows if r.purge_time and r.entry_touched_at
            and r.entry_touched_at >= r.purge_time + span[r.strategy]]
    total = sum((float(r.rr or 0) if r.outcome == "win" else -1.0) for r in rows)
    note = f"{len(rows)} elenen setup çözüldü (RR ≥ 2, 4H/1D/1W)"
    if rows:
        w = sum(1 for r in rows if r.outcome == "win")
        ew = sum(1 for r in early if r.outcome == "win")
        lw = sum(1 for r in late if r.outcome == "win")
        note += (f" · {w} TP · R/işlem {total / len(rows):+.2f}"
                 f" · retest C2 açıkken {ew}/{len(early)} TP, C2 kapalıyken {lw}/{len(late)} TP")
    return Progress(bars=[Bar("retest C2 açıkken", len(early), 30),
                          Bar("retest C2 kapalıyken", len(late), 30)], note=note)


async def _p_sticky_delete(db: AsyncSession) -> Progress:
    """02.10+ silinen (kalici silme canli), sonucu belli, RR >= 2 setup; past_sl/past_tp haric (tanim geregi SL/TP)."""
    t = _journal_table()
    rows = (await db.execute(
        select(t.c.outcome, t.c.rr).where(
            t.c.best_stage == "waiting", t.c.deleted_reason.is_not(None),
            t.c.deleted_reason.not_in(("past_sl", "past_tp", "no_levels", "week_close")),
            t.c.deleted_at >= datetime(2026, 10, 2), t.c.rr >= 2.0, t.c.outcome.in_(("win", "loss")),
        )
    )).all()
    note = f"{len(rows)} engellenen setup çözüldü"
    if rows:
        w = sum(1 for o, _ in rows if o == "win")
        total = sum((float(rr or 0) if o == "win" else -1.0) for o, rr in rows)
        note += f" · {w} TP · R/işlem {total / len(rows):+.2f}"
    return Progress(bars=[Bar("çözülmüş engellenen setup", len(rows), 30)], note=note)


async def _p_stale_days(db: AsyncSession) -> Progress:
    """02.10+ temiz kume (tmp_stale_days_cf ile ayni suzgec, yaklasik): CRT sayilan, skor kirilimi var, 4H'te C2 kapali,
    sonuc belli (win/loss/tp_before_entry/no_touch), RR >= 2, dar stop degil. Hedef 300 (01.10 kumesi 294)."""
    t = SetupJournal.__table__
    srm = func.json_extract(t.c.features, "$.stop_range_mult")
    n = await db.scalar(select(func.count()).select_from(t).where(
        t.c.levels_at >= datetime(2026, 10, 2),
        t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
        func.json_extract(t.c.parts_at_levels, "$.raw").is_not(None),
        or_(t.c.strategy != "4h", func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1),
        t.c.outcome.in_(("win", "loss", "tp_before_entry", "no_touch")),
        t.c.rr >= 2.0, or_(srm.is_(None), srm >= 1.0),
    )) or 0
    return Progress(bars=[Bar("taze temiz setup (02.10+)", n, 300)],
                    note="kesin sayım ve kollar: python tmp/tmp_stale_days_cf.py")


async def _p_cluster_recheck(db: AsyncSession) -> Progress:
    """Kume limitinin eledigi, islem acilabilir (RR >= 2), sonucu belli setup (hedef 60). Hepsi 4H kripto."""
    t = _journal_table()
    rows = (await db.execute(select(t.c.outcome, t.c.rr).where(
        t.c.best_stage == "cluster_limit", t.c.rr >= 2.0, t.c.outcome.in_(("win", "loss")),
    ))).all()
    note = f"{len(rows)} çözülmüş"
    if rows:
        w = sum(1 for o, _ in rows if o == "win")
        total = sum((float(rr or 0) if o == "win" else -1.0) for o, rr in rows)
        note += f" · {w} TP · R/işlem {total / len(rows):+.2f}"
    return Progress(bars=[Bar("elenen RR ≥ 2 setup", len(rows), 60)], note=note)


async def _p_bias_1h(db: AsyncSession) -> Progress:
    """1H'te bias kapisinda elenip SONUCU BELLI olan setup (karar esigi 25) + skor kalemi kovalari.

    Ikinci sayi `note`ta: `Progress.ready` TUM cubuklari birden ister, kalem kovasini cubuk
    yapmak kapi hatirlatmasini dolmamis esigin arkasina saklardi (kardes `_p_shadow` notu).
    """
    t = _journal_table()
    rows = await db.execute(
        select(t.c.outcome, t.c.rr).where(t.c.strategy == "1h",
                                          t.c.levels_at.is_not(None),
                                          t.c.best_stage == "bias_mismatch",
                                          t.c.outcome.in_(("win", "loss")))
    )
    got = rows.all()
    total = sum((float(rr or 0) if o == "win" else -1.0) for o, rr in got)
    note = f"{len(got)} elenen setup çözüldü"
    if got:
        w = sum(1 for o, _ in got if o == "win")
        note += f" · {w} TP · R/işlem {total / len(got):+.2f}"
    # H2 kovalari (htf +2 / 0) -- kalem karari icin kova basina 25 gerekiyor.
    # Kume raporunkiyle AYNI olmali (scripts/bias_1h_stat.py): seviyesi olan ve motorun CRT
    # saydigi setuplar. On-eleme adaylari sayilirsa sayac esigi rapordan once doldurur.
    pre = ("sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak", "not_selected")
    htf = func.json_extract(t.c.score_parts, "$.htf")
    buckets = await db.execute(
        select(htf, func.count()).where(t.c.strategy == "1h",
                                        t.c.levels_at.is_not(None),
                                        t.c.best_stage.not_in(pre),
                                        t.c.score_parts.is_not(None),
                                        t.c.outcome.in_(("win", "loss"))).group_by(htf)
    )
    by = {int(k): n for k, n in buckets.all() if k is not None}
    note += f" · skor kalemi kovalari htf+2: {by.get(2, 0)}/25, htf0: {by.get(0, 0)}/25"
    return Progress(bars=[Bar("çözülmüş bias elemesi", len(got), 25)], note=note)


async def _p_deleted_gate(db: AsyncSession) -> Progress:
    """`waiting`e ulasip silinen setuplardan SONUCU BELLI olanlar (karar esigi 20).

    Skorla ilgili nedenleri (low_quality / missed_quality / score7_gate) ayrica sayar:
    kullanicinin asil sorusu "skor kapisi kazanan sinyal mi eliyor" (19.09).
    """
    t = _journal_table()
    rows = await db.execute(
        select(t.c.outcome, t.c.rr, t.c.deleted_reason).where(
            t.c.best_stage == "waiting", t.c.deleted_reason.is_not(None)
        )
    )
    got = rows.all()
    resolved = [(o, rr, d) for o, rr, d in got if o in ("win", "loss")]
    total = sum((float(rr or 0) if o == "win" else -1.0) for o, rr, _ in resolved)
    score_gates = ("low_quality", "missed_quality", "score7_gate")
    sc_res = [x for x in resolved if x[2] in score_gates]
    sc_all = [x for x in got if x[2] in score_gates]
    note = f"{len(got)} silinmis setup · {len(sc_all)} tanesi skor kapisindan"
    if resolved:
        w = sum(1 for o, _, _ in resolved if o == "win")
        note += f" · çözülmüş {len(resolved)}: {w} TP · R/işlem {total / len(resolved):+.2f}"
    if sc_res:
        note += f" · skor kapısı çözülmüş {len(sc_res)}"
    return Progress(bars=[Bar("çözülmüş silinen setup", len(resolved), 20)], note=note)


def _final_levels_cond(t, strategy: str):
    """03.10: 1D/1W'de seviyesi KESIN satir (`setup_journal.levels_final` ile ayni) -- pre_c2 dolu (sinyalde / C2
    kapanisinda yeniden donduruldu) ya da seviyeler C2 kapandiktan sonra dondu. 4H'te kosul yok (cagiran C2 kapali
    donmus sartini ayrica koyar)."""
    if strategy == "4h":
        return True
    bar = "+7 days" if strategy == "1w" else "+1 day"
    return or_(t.c.pre_c2.is_not(None), t.c.levels_at >= func.datetime(t.c.purge_time, bar))


async def _score_parts_resolved(db: AsyncSession, strategy: str, since: datetime | None = None) -> int:
    """Stratejinin kirilimli ve sonuclanmis (win/loss) Journal satiri -- raporun okuma birimi."""
    t = _journal_table()
    # 03.10: CRT-oncesi adaylar sayilmaz (raporlar onlari disarida birakir; 1D'de 32 satirin 24'u oydu).
    where = [t.c.strategy == strategy, t.c.score_parts.is_not(None), t.c.outcome.in_(("win", "loss")),
             t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)), _final_levels_cond(t, strategy)]
    if since is not None:
        where.append(t.c.first_seen >= since)
    return await db.scalar(select(func.count()).select_from(t).where(*where)) or 0


async def _p_score_parts_4h(db: AsyncSession) -> Progress:
    """21.10 (kullanici): paket (PD major/aylik 0 + wick 0.30, esik 6) kararini kaderi DEGISEN setuplar verir -- eşik
    6'da giren + paketle dusen. Tetik 150 (0.15 R/setup farki p < 0.10'la ayirmak icin ~150 gerekiyor; 03.10'daki
    "A kolu 30" 0.3'luk farki ancak gorurdu). Kume ve skor `tmp/tmp_score_weight_cf.py` ile birebir: C2 kapali donmus,
    CRT-oncesi / bias / target_taken haric, sinyalde golge sonucu, RR >= 2, dar stop degil, skor-7 kapisi."""
    t = _journal_table()
    early = tuple(PRE_SETUP_STAGES | {"bias_mismatch", "target_taken"})
    rows = (await db.execute(select(
        t.c.parts_at_levels, t.c.features_at_levels, t.c.features, t.c.outcome, t.c.rr, t.c.shadow, t.c.model,
    ).where(t.c.strategy == "4h", t.c.parts_at_levels.is_not(None), t.c.best_stage.not_in(early),
            func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1))).all()

    def _j(v):
        if isinstance(v, dict):
            return v
        try:
            d = json.loads(v) if v else None
        except Exception:
            return None
        return d if isinstance(d, dict) else None

    n = w_in = 0
    total = 0.0
    for pal, fal, fe, o, rr, sh, model in rows:
        p_ = _j(pal) or {}
        if "raw" not in p_:
            continue
        f_ = _j(fal) or _j(fe) or {}
        if o == "signal":
            s1 = (_j(sh) or {}).get("1") or {}
            o, rr = s1.get("o"), s1.get("rr", rr)
        if o not in ("win", "loss", "tp_before_entry", "no_touch"):
            continue
        srm = f_.get("stop_range_mult")
        if rr is None or rr < 2.0 or (srm is not None and srm < 1.0):
            continue
        smt = p_.get("smt", 0)
        d = -p_.get("pd_major", 0) - p_.get("pd_monthly", 0)
        wf = f_.get("wick_frac")
        if p_.get("wick", 0) > 0 and isinstance(wf, (int, float)) and wf < 0.30:
            d -= p_["wick"]
        new_sc = min(11, max(0, min(9, p_["raw"] - smt + d)) + smt)
        pd_any = (p_.get("pd_major", 0) + p_.get("pd_monthly", 0) + p_.get("pd_struct", 0)) > 0
        zone = (model or "cisd") in ("ifvg", "bpr")

        def _passes(sc: int, thr: int) -> bool:
            if sc < thr:
                return False
            if sc == 7:  # skor-7 kapisi motorda skor TAM 7'de (esik 6'da da)
                return (not zone) and pd_any
            return True
        now_ok, pkg_ok = _passes(p_.get("score", 0), 7), _passes(new_sc, 6)
        if now_ok == pkg_ok:
            continue
        R = rr if o == "win" else (-1.0 if o == "loss" else 0.0)
        n += 1
        total += R if pkg_ok else -R
        w_in += pkg_ok
    note = f"paketin farkı {total:+.1f}R · setup başına {total / n:+.2f}" if n else ""
    return Progress(bars=[Bar("paketin kaderini değiştirdiği 4H setup (giren + düşen)", n, 150)],
                    note=(note + f" · giren {w_in} / düşen {n - w_in}" if n else note))


def _p_score_candidates(strategy: str):
    """21.10 (kullanici): "Skor tablosu yeni adaylar" -- bugun skorda olmayan 9 aday, strateji bazinda. Kume ve karar
    bandi `app/score_outcome` (rapor `scripts/score_candidates_stat.py` ile ayni fonksiyonlar). Cubuk: karar bandinda
    AZ olan tarafi en cok birikmis aday (tetik taraf basina 30) -- herhangi bir aday dolunca madde aksiyona duser."""
    async def _fn(db: AsyncSession) -> Progress:
        from app import score_outcome as so
        t = SetupJournal.__table__
        res = await db.execute(select(t).where(
            t.c.strategy == strategy, t.c.parts_at_levels.is_not(None), t.c.entry.is_not(None),
            t.c.first_seen >= datetime(2026, 9, 29),
        ))
        sub = so.collect(res.mappings().all(), strategy=strategy)
        prog = so.cand_progress(sub, so.THR[strategy])
        labels = {k: lab for k, lab, *_ in so.CANDIDATES}
        best_k, best_n = prog[0]
        note = " · ".join(f"{labels[k].split(' (')[0]} {n}" for k, n in prog[:4])
        return Progress(bars=[Bar(f"en ileri aday: {labels[best_k].split(' (')[0]}", best_n, so.CAND_MIN_SIDE)],
                        note=f"karar bandı, az taraf: {note} — rapor: python scripts/score_candidates_stat.py "
                             f"--strategy {strategy}")
    return _fn


async def _proposals_a_count(db: AsyncSession, strategy: str, since: datetime) -> int:
    """`scripts/score_proposals_stat.py` A koluna (bugunku kural @7) yaklasan sayim: CRT-oncesi / bias_mismatch /
    target_taken haric, skor >= 7, RR >= 2, sonuclanmis ya da dolmamis; 4H'te C2 kapali donmus. Kesin sayi rapordan."""
    t = _journal_table()
    early = tuple(PRE_SETUP_STAGES | {"bias_mismatch", "target_taken"})
    # Skor raporla ayni andan (`analysis_parts`): 4H donmus, 1D/1W son skor (parts_final restart sonrasi kolon;
    # sayac eski DB'de de calissin diye score_parts -- kesin sayi rapordan).
    score_col = t.c.parts_at_levels if strategy == "4h" else t.c.score_parts
    where = [t.c.strategy == strategy, t.c.levels_at >= since, t.c.best_stage.not_in(early),
             func.json_extract(score_col, "$.score") >= 7, t.c.rr >= 2.0,
             t.c.outcome.in_(("win", "loss", "tp_before_entry", "no_touch", "signal"))]
    if strategy == "4h":
        where.append(func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1)
    where.append(_final_levels_cond(t, strategy))
    return await db.scalar(select(func.count()).select_from(t).where(*where)) or 0


async def _p_score_proposals_1d(db: AsyncSession) -> Progress:
    """03.10: 1D kardesi -- 25.09'dan (1D skor kurallari degisti) butun veri, A kolu 30 (1D'de bugunku kural ~1 setup/hafta)."""
    n = await _proposals_a_count(db, "1d", datetime(2026, 9, 25))
    return Progress(bars=[Bar("1D 25.09+, bugünkü kural @7 (A kolu)", n, 30)],
                    note="1D'de bugünkü kural haftada ~1 setup açıyor; 03.10'dan beri yalnız seviyesi kesin satır "
                         "(sinyalde / C2 kapanışında) sayılır — örneklem baştan birikiyor")


async def _p_score_proposals_4h(db: AsyncSession) -> Progress:
    """03.10 (kullanici): skor onerileri karsi-olgusu, taze (levels_at >= 04.10) A kolu (bugunku kural @7) 60 setup."""
    n = await _proposals_a_count(db, "4h", datetime(2026, 10, 4))
    return Progress(bars=[Bar("taze 4H, bugünkü kural @7 (A kolu)", n, 60)],
                    note="SMT önerileri askıda (SMT düzeltiliyor); diğerleri küçük ya da eşiğe göre işaret değiştiriyor — "
                         "karar taze veri + replay")


async def _p_smt_variants(db: AsyncSession) -> Progress:
    """03.10 (kullanici: "SMT onemli konfirme, cikarmak yerine duzeltelim"): C1 ucuna bagli SMT olcumu (smt_htf / smt_lvl)
    tasiyan cozulmus 4H setup, cift turune gore. Kaba hedef: FX ciftleri 100, altcoin->BTC 150 -- hukum karar bandinda
    taraf basina 10 ile `scripts/smt_variants_stat.py`'den. 4H'te C2 kapali donmus sarti raporla ayni."""
    t = _journal_table()

    def kind_is(k):
        return or_(func.json_extract(t.c.features_at_levels, "$.smt_kind") == k,
                   func.json_extract(t.c.features, "$.smt_kind") == k)

    base = [t.c.strategy == "4h", t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)),
            func.json_extract(t.c.parts_at_levels, "$.c2_closed") == 1,
            t.c.outcome.in_(("win", "loss", "signal"))]
    n_fx = await db.scalar(select(func.count()).select_from(t).where(*base, kind_is("fx"))) or 0
    n_alt = await db.scalar(select(func.count()).select_from(t).where(*base, kind_is("alt_btc"))) or 0
    return Progress(bars=[Bar("4H FX çiftleri, SMT ölçümlü", n_fx, 100), Bar("4H altcoin→BTC, SMT ölçümlü", n_alt, 150)],
                    note="ölçüm restart'tan sonra başlar · 1D/1W raporda (--strategy)")


async def _p_score_parts_1w(db: AsyncSession) -> Progress:
    """1W (01.10): kirilimli + sonuclanmis 1W setup. Hukum kalem basina karar bandinda taraf basina 10 ister
    (`scripts/score_item_outcome_stat.py`); 60 sonuclanmis kaba bir hedef -- hukum raporun kendisinden okunur."""
    n = await _score_parts_resolved(db, "1w")
    return Progress(bars=[Bar("sonuçlanmış 1W setup", n, 60)])


async def _p_smt_segment(db: AsyncSession) -> Progress:
    """SMT segment bazinda -- 21.10'dan beri YALNIZ TAZE veri (first_seen >= 02.10): 4H altcoin karar bandinda TERS (-0.25)
    cikti ama tum kotuluk 02.10 sonrasindan, 1. yari tersineydi. Kume `score_outcome.collect` (rapor ile ayni); karar
    bandi SMT haric skor 5-6, smt_possible != 0. Segment x strateji basina AZ olan taraf; tetik 60 (21.10 kullanici: 30'da
    okundu, kural tuttu ama karar daha cok veriye birakildi) -- herhangi biri dolunca."""
    from app import score_outcome as so
    t = SetupJournal.__table__
    out = {}
    for st in ("4h", "1d"):
        res = await db.execute(select(t).where(
            t.c.strategy == st, t.c.parts_at_levels.is_not(None), t.c.entry.is_not(None),
            t.c.first_seen >= datetime(2026, 10, 2)))
        sub = [x for x in so.collect(res.mappings().all(), strategy=st) if x["smt_ok"] != 0]
        for seg in ("alt", "major", "fx"):
            band = [x for x in sub if x["seg"] == seg
                    and 5 <= ((x["pal"].get("raw") or 0) - (x["pal"].get("smt") or 0)) <= 6]
            g = so.stats([x for x in band if x["pal"].get("smt") == 2])
            b = so.stats([x for x in band if x["pal"].get("smt") == 0])
            out[(st, seg)] = min(g["res"], b["res"])
    names = {"alt": "altcoin", "major": "BTC/ETH", "fx": "FX"}
    best = max(out, key=out.get)
    note = " · ".join(f"{k[0].upper()} {names[k[1]]} {v}" for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v)
    return Progress(bars=[Bar(f"taze karar bandı, {best[0].upper()} {names[best[1]]} (az taraf)", out[best], 60)],
                    note=(note or "taze veride karar bandı boş") + " — 02.10+ yalnız")


_RFVG_QUAL_FAIL = ("sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "c1_weak",
                   "not_selected", "bias_mismatch", "target_taken", "low_quality", "no_cisd", "score7", "tight_stop")


async def _p_retest_ifvg(db: AsyncSession) -> Progress:
    """CISD retest sonrasi IFVG (03.10): karar A'nin birimi -- kapidan gecmis, CISD RR >= 2, retest sonrasi IFVG
    OLUSMUS setup (strateji basina 30). Kural `scripts/retest_ifvg_stat.py` ile ayni (4H'te C2 kapali, imm haric)."""
    t = _journal_table()
    rows = (await db.execute(select(t.c.strategy, t.c.best_stage, t.c.entries, t.c.parts_at_levels).where(
        t.c.entries.like('%"_rfvg"%')))).all()
    n = {"4h": 0, "1d": 0, "1w": 0}
    b = {"4h": 0, "1d": 0, "1w": 0}
    for r in rows:
        if r.strategy not in n or r.best_stage in _RFVG_QUAL_FAIL:
            continue
        pal = json.loads(r.parts_at_levels) if r.parts_at_levels else {}
        if r.strategy == "4h" and pal.get("c2_closed") != 1:
            continue
        ents = json.loads(r.entries) or {}
        st, cv, near = ents.get("_rfvg") or {}, ents.get("cisd") or {}, ents.get("rfvg_near") or {}
        if st.get("imm") or cv.get("imm") or near.get("e") is None:
            continue
        if r.best_stage == "low_rr" or (cv.get("rr") or 0) < 2.0:
            b[r.strategy] += (near.get("rr") or 0) >= 2.0 and near.get("o") in ("win", "loss")
        else:
            n[r.strategy] += 1
    return Progress(bars=[Bar("4H IFVG oluşan setup (A)", n["4h"], 30)],
                    note=f"1D A {n['1d']}/30 · 1W A {n['1w']}/30 · B (CISD RR < 2) 4H {b['4h']} / 1D {b['1d']} / "
                         f"1W {b['1w']} (eşik 20)")


async def _p_score_parts_1d(db: AsyncSession) -> Progress:
    """03.10: tarih doldu, hicbir kalemin karar bandi taraf basina 10'a ulasmadi -> madde ACIK, tetik orneklem
    (kullanici). 300 sonuclanmis 25.09+ setup kaba hedef (03.10'da ~100; karar bandi sayilari ~3 katina cikar);
    hukum kalem kalem raporun kendisinden (`scripts/score_item_outcome_stat.py --strategy 1d --since 2026-09-25`)."""
    n = await _score_parts_resolved(db, "1d")
    n25 = await _score_parts_resolved(db, "1d", datetime(2026, 9, 25))
    return Progress(bars=[Bar("sonuçlanmış 1D setup (25.09+)", n25, 300)],
                    note=f"toplam {n} · 03.10: yalnız −4 cezası hüküm verdi (doğru çalışıyor), diğer kalemler karar "
                         "bandında < 10")


async def _p_w1_live(db: AsyncSession) -> Progress:
    """1W-4H (28.09): Journal'da gorulen 1W setup + acilan/kapanan 1W sinyal (tarihli madde: sayilar note'ta)."""
    t = _journal_table()
    seen = await db.scalar(select(func.count()).select_from(t).where(
        t.c.strategy == "1w", t.c.best_stage.not_in(tuple(PRE_SETUP_STAGES)))) or 0
    sigs = await db.scalar(select(func.count()).select_from(Signal).where(Signal.timeframe == "1w")) or 0
    closed = await _closed_count(db, Signal.timeframe == "1w")
    wk = await _closed_count(db, Signal.timeframe == "1w", Signal.exit_reason == "week_close")
    return Progress(bars=[Bar("kapanan 1W sinyal", closed, 3)],
                    note=f"{seen} 1W setup görüldü · {sigs} sinyal · {wk} Cuma kapanışıyla")


async def _p_htf_bonus(db: AsyncSession) -> Progress:
    """W40 (28.09+), skor bandi 0-6, sonuclanmis: TF basina htf +2 / htf 0 kovalari (karar haftasi, esik 50/50).

    Tarihli madde: cubuk eklemek hatirlatmayi saklar -- sayilar note'ta (_p_score_parts_1d deseni)."""
    t = _journal_table()
    htf = func.json_extract(t.c.parts_at_levels, "$.htf")
    sc = func.json_extract(t.c.parts_at_levels, "$.score")
    parts = []
    # 1H 28.09'da kapandi (W40'ta veri gelmez); 1W'nin `htf` kalemi aylik bias -- ayri soru.
    for tf in ("4h", "1d"):
        base = [t.c.strategy == tf, t.c.parts_at_levels.is_not(None), t.c.outcome.in_(("win", "loss")),
                t.c.first_seen >= datetime(2026, 9, 28), sc <= 6]
        n2 = await db.scalar(select(func.count()).select_from(t).where(*base, htf == 2)) or 0
        n0 = await db.scalar(select(func.count()).select_from(t).where(*base, htf == 0)) or 0
        parts.append(f"{tf.upper()} {n2}/{n0}")
    # 1D havuz (28.09, kullanici onayi): hafta sarti yok, tum haftalar -- tek haftada 50/50 hic dolmuyor.
    b1 = [t.c.strategy == "1d", t.c.parts_at_levels.is_not(None), t.c.outcome.in_(("win", "loss")), sc <= 6]
    p2 = await db.scalar(select(func.count()).select_from(t).where(*b1, htf == 2)) or 0
    p0 = await db.scalar(select(func.count()).select_from(t).where(*b1, htf == 0)) or 0
    parts.append(f"1D havuz {p2}/{p0}")
    return Progress(note="W40 bant 0–6, htf +2 / htf 0 (her biri 50): " + " · ".join(parts))


async def _p_journal_live(db: AsyncSession) -> Progress:
    n = await db.scalar(select(func.count()).select_from(_journal_table())) or 0
    return Progress(note=f"{n} Journal satırı")


async def _p_dead_candidate(db: AsyncSession) -> Progress:
    """Olu aday slotu biraktiktan sonra dogan sinyaller (18.09 degisikligi).

    Vekil olcum: elenmis bir kardes aday (`not_selected`) canliyken sinyale donen setup sayisi --
    kardesin olu mu yoksa sadece dar range'li mi oldugunu Journal ayirt etmiyor, ust sinir.
    """
    t = _journal_table()
    a, b = t.alias("a"), t.alias("b")
    sib = select(1).select_from(b).where(
        b.c.strategy == a.c.strategy, b.c.symbol == a.c.symbol, b.c.id != a.c.id,
        b.c.best_stage == "not_selected",
        b.c.first_seen <= a.c.first_seen, b.c.last_seen >= a.c.first_seen,
    ).exists()
    n = await db.scalar(
        select(func.count()).select_from(a)
        .where(a.c.first_seen >= datetime(2026, 9, 18, 12, 0), a.c.best_stage == "waiting", sib)
    ) or 0
    return Progress(bars=[Bar("yeni yolla doğan sinyal (üst sınır)", n, 10)])


async def _p_cluster(db: AsyncSession) -> Progress:
    n = await db.scalar(
        select(func.count()).select_from(_journal_table())
        .where(_journal_col("best_stage") == "cluster_limit")
    ) or 0
    return Progress(bars=[Bar("küme limitine takılmış setup", n, 25)])


# ---------------------------------------------------------------------------------------
# KAYIT. Sira = sayfada gorunecek sira (acik olanlar yeniden eskiye).
# ---------------------------------------------------------------------------------------

_CME_SYMBOLS = ("XAUUSD", "XAGUSD", "US100", "US500", "OILWTI", "OILBRENT")


async def _p_session_anchor(db: AsyncSession) -> Progress:
    """Yeni izgarada (18:00 NY) dogan CME setup'lari: kac tanesi sonuclandi?"""
    t = SetupJournal.__table__
    n = await db.scalar(
        select(func.count()).select_from(t).where(
            t.c.symbol.in_(_CME_SYMBOLS),
            t.c.first_seen >= datetime(2026, 9, 19),
            t.c.outcome.in_(("win", "loss", "tp_before_entry", "no_touch", "signal")),
        )
    ) or 0
    return Progress(bars=[Bar("yeni ızgarada sonuçlanmış CME setup", n, 200)])


ITEMS: list[WatchItem] = [
    WatchItem(
        key="session_anchor", status="done", started="18.09",
        result="(03.10) KAPANDI, kullanıcı onayı: 18:00 NY anchor'ı KALIYOR — FX kontrol grubuna göre setup sıklığı aynı "
               "oranda (×0.69 / ×0.76), c2_breakout payı ~5 puan daha az; 4H R/setup FX'e göre −0.33 (42/72, ~1 SE, 1D ters yönde) "
               "— belirgin bozulma yok.",
        title="CME seans anchor'ı — endeks/metal/petrol 18:00 NY",
        trigger="yeni ızgarada 200 sonuçlanmış CME setup: bu 6 sembolde setup sayısı/outcome bozuldu mu, c2_breakout payı düştü mü",
        measure="python tmp/tmp_session_anchor_check.py · /setup-journal (CME sembolleri)",
        md="CME seans anchor'ı — endeks/metal/petrol 18:00 NY'ye geçti (18.09.2026, restart bekliyor)",
        progress_fn=_p_session_anchor,
    ),
    WatchItem(
        key="retrace", status="done", started="18.09", onem=1,
        title="Limit giriş seviyesi — CISD/MSS mi IFVG/BPR mi, stop %100/%80/%60",
        trigger="30 first_bar'lı dolmuş kazanan: bir varyant/stop bugünkü girişi +0.15 R/setup geçerse mum re-track'i",
        result="(01.10) STRATEJİ BAZINDA ÜÇ MADDEYE BÖLÜNDÜ (kullanıcı: 30 az): 4H 150 / 1D 40 / 1W 10 dolmuş kazanan; "
               "bölünürken 4H 62, 1D 4, 1W 0. Karar okuması yapılmadı.",
        measure="python tmp/tmp_limit_variant_depth.py --strategy 4h|1d|1w",
        md="Geri çekilme derinliği — limit emri hangi seviyeden dolar? (başladı 18.09.2026, ⏰ ilk okuma 28.09.2026)",
        progress_fn=_p_retrace,
    ),
    WatchItem(
        key="retrace_4h", status="done", started="01.10", onem=1,
        title="Limit giriş seviyesi — 4H",
        trigger="150 first_bar'lı dolmuş kazanan: bir limit varyantı (CISD/MSS/IFVG/BPR) ya da stop (%80/%60) aynı setuplarda bugünkü girişi +0.15 R/setup geçerse mum re-track'i; geçmezse giriş ve stop kalır",
        measure="python tmp/tmp_limit_variant_depth.py --strategy 4h · python scripts/retrace_stat.py",
        md="Limit giriş seviyesi — 4H",
        result="(21.10) GİRİŞ VE STOP KALIYOR (kullanıcı onayı): first_bar'lı eşli setuplarda hiçbir varyant/stop bugünkü girişi "
               "+0.15 geçmedi (IFVG'li 172: +0.001 · bölgesiz 323: MSS %80 +0.04 · BPR'li 222: +0.01). Kapıları geçmişte stop %80 "
               "~+0.10 (eşik altı; 'Market vs limit — 4H'de sürüyor). Yan bulgu: kapı geçmiş BPR girişi 18 dolumda 1 TP → "
               "'Skor tablosu yeni adaylar'a aday bpr_entry.",
        progress_fn=_p_retrace_for("4h", 150),
    ),
    WatchItem(
        key="retrace_1d", status="open", started="01.10", onem=1,
        title="Limit giriş seviyesi — 1D",
        trigger="40 first_bar'lı dolmuş kazanan: bir limit varyantı (CISD/MSS/IFVG/BPR) ya da stop (%80/%60) aynı setuplarda bugünkü girişi +0.15 R/setup geçerse mum re-track'i; geçmezse giriş ve stop kalır",
        measure="python tmp/tmp_limit_variant_depth.py --strategy 1d",
        md="Limit giriş seviyesi — 1D",
        progress_fn=_p_retrace_for("1d", 40),
    ),
    WatchItem(
        key="retrace_1w", status="open", started="01.10", onem=1,
        title="Limit giriş seviyesi — 1W",
        trigger="10 first_bar'lı dolmuş kazanan: bir limit varyantı (CISD/MSS/IFVG/BPR) ya da stop (%80/%60) aynı setuplarda bugünkü girişi +0.15 R/setup geçerse mum re-track'i; geçmezse giriş ve stop kalır. 01.10'dan beri 1W girişi yalnız CISD + %80 stop (kullanıcı kararı): bugünkü giriş = o kural. 10'da fark iki zaman yarısında da aynı yöndeyse karar, değilse eşik 20'ye uzar",
        measure="python tmp/tmp_limit_variant_depth.py --strategy 1w",
        md="Limit giriş seviyesi — 1W",
        progress_fn=_p_retrace_for("1w", 10),
    ),
    WatchItem(
        key="rr_cap", status="open", started="01.10", onem=1,
        title="Yüksek RR'li setupta 3–4R civarındaki girişe geçmek (RR tavanı)",
        trigger="20 first_bar'lı çözülmüş, girişi değişen setup (bugünkü RR > 4 ve 3–4R bandında CISD/MSS/IFVG/BPR adayı "
                "var; havuz): bant girişi aynı setuplarda bugünkü girişi +0.15 R/setup ve iki zaman yarısında geçerse mum "
                "re-track'i (kısmi kâr + BE), sonra kullanıcıya; geçmezse giriş kalır. Tek strateji 10 setupla ayrıca okunabilir",
        measure="python scripts/rr_cap_stat.py [--strategy 4h|1d|1w]",
        md="Yüksek RR'li setupta 3–4R civarındaki girişe geçmek (RR tavanı)",
        progress_fn=_p_rr_cap,
    ),
    WatchItem(
        key="score_order_4h", status="open", started="03.10", onem=1,
        title="Toplam skor kazananı sıralıyor mu? (eşik 7 anlamlı mı) — 4H",
        trigger="taze (levels_at ≥ 04.10) temiz 4H setupta (C2 kapalı, RR ≥ 2, kapıda elenenler dahil) skor < 7 ve ≥ 7 "
                "her biri 60 çözülmüş: S1 skor ≥ 7 − < 7 ≥ +0.15 R/setup + iki yarı → skor çalışıyor, kalem maddelerine "
                "dön · S2 fark ≤ 0 iki yarıda → skor kapısı kazananı seçmiyor: tek replay, sonra kullanıcıya (skoru "
                "yeniden kur / eşiği kaldır) · S3 arada → değişiklik yok",
        measure="/izleme sayacı (not satırı R/setup) · python scripts/score_parts_stat.py --strategy 4h --since 2026-10-04",
        md="Toplam skor kazananı sıralıyor mu? (03.10.2026)",
        progress_fn=_p_score_order,
    ),
    WatchItem(
        key="min_rr_25", status="open", started="03.10", onem=1,
        title="Min RR 2.0 → 2.5 mi? (alt bant 2–2.5 zararda mı)",
        trigger="taze (levels_at ≥ 04.10) temiz, kapıyı geçmiş 4H setupta RR 2–2.5 ve ≥ 2.5 her biri 50 çözülmüş: "
                "alt bant ≥ 0.3 R/setup kötü + iki yarı aynı yön → kısmi kâr + BE ile tek replay, sonra kullanıcıya · "
                "aksi halde min RR 2.0 kalır",
        measure="/izleme sayacı (not satırı R/setup)",
        md="Min RR 2.0 → 2.5 mi? (03.10.2026)",
        progress_fn=_p_min_rr,
    ),
    WatchItem(
        key="segment_4h", status="open", started="03.10", onem=1,
        title="Hangi segment kazanıyor: 4H kripto mu, 4H FX mi?",
        trigger="taze (levels_at ≥ 04.10) temiz, kapıyı geçmiş 4H setupta kripto 60 / FX-metal-endeks 30 çözülmüş: "
                "bir segment düz TP/SL'de ≤ −0.2 R/setup ve canlı kapalı işlemde de eksideyse o segmenti kısma / "
                "kapatma kullanıcıya sorulur; ikisi de artıdaysa değişiklik yok",
        measure="/izleme sayacı · Dashboard strateji kartları (4H Crypto / 4H FX)",
        md="Hangi segment kazanıyor: 4H kripto mu, 4H FX mi? (03.10.2026)",
        progress_fn=_p_segment,
    ),
    WatchItem(
        key="entry_models", status="done", started="18.09",
        title="Entry modeli karşılaştırması",
        trigger="Varyant başına 60 karar verilebilir setup (ilk okuma 28.09.2026)",
        measure="python scripts/entry_model_stat.py",
        md="Entry modeli karşılaştırması (18.09.2026 — kapandı 21.09)",
        result="Bugünkü entry modeli KALIYOR (21.09): 230 karar verilebilir setupta hiçbir varyant "
               "chosen'ı 0.15R geçmedi (chosen -0.281, en iyisi ifvg_far -0.253 R/setup); H2 de "
               "geçmedi (alt kenar dolum -4.5 puan, R/setup yalnız +0.025). DFVG modeli reddedildi. "
               "Açık not: skor>=7 & RR>=2 diliminde ifvg_far +0.463 vs chosen -0.259 ama n=15.",
        progress_fn=_p_entry_models,
    ),
    WatchItem(
        key="bpr", status="done", started="21.09", onem=1,
        title="BPR entry modeli",
        trigger="40 BPR'li çözülmüş setup: bpr_near, ifvg_near'ı 0,15 R/setup geçiyor mu",
        measure="python scripts/entry_model_stat.py",
        md="BPR entry modeli (21.09.2026 — kapandı 24.09)",
        result="BPR KALIYOR, skor ve skor-7 kapısı DEĞİŞMİYOR (24.09): aynı 54 setupta bpr_near "
               "+0.015 vs ifvg_near -0.031 R/setup, fark +0.046 -- bandın içinde. C2 kapalı dilimde "
               "de aynı yön (+0.094, n=32).",
        progress_fn=_p_bpr,
    ),
    WatchItem(
        key="bpr_leg", status="done", started="22.09", onem=1,
        title="BPR bacağı — kesişim yerine bacağın kendisinden mi girmeli?",
        trigger="40 bacak ölçümlü setup: bfvg_near, bpr_near'ı 0,15 R/setup geçiyor mu",
        result="KESİŞİM KALIYOR (28.09): aynı 143 setupta bacak −0.054 vs kesişim −0.051 R/setup (fark −0.004, "
               "bant ±0.15); bacak 2 kazanç fazla dolduruyor ama düşük RR bunu siliyor. bfvg ölçüm olarak duruyor.",
        measure="python scripts/entry_model_stat.py (bölüm 2)",
        md="BPR bacağı — kesişim yerine bacağın kendisinden mi girmeli? (22.09.2026 — kapandı 28.09)",
        progress_fn=_p_bpr_leg,
    ),
    WatchItem(
        key="cfvg", status="done", started="22.09", onem=1,
        title="CISD kırılım FVG'si — onaydan sonra bırakılan boşluktan giriş",
        trigger="60 eşlenmiş setup: cfvg_* varyantlarından biri chosen'ı 0,15 R/setup geçiyor mu",
        measure="python scripts/entry_model_stat.py (bölüm 3)",
        md="CISD kırılım FVG'si — onaydan sonra bırakılan boşluktan giriş (22.09.2026 — kapandı 24.09)",
        result="CISD entry'si KALIYOR (24.09): eşlenmiş 98-106 setupta üç cfvg varyantı da chosen'ın "
               "0.06-0.08 R/setup altında -- bandın içinde, geçen yok. C2 kapalı dilimde de aynı "
               "(far -0.084 vs +0.010, n=61). Kırılım setupların %39'unda FVG bırakmıyor. cfvg "
               "ölçüm olarak duruyor.",
        progress_fn=_p_cfvg,
    ),
    WatchItem(
        key="week_gap", status="open", started="21.09", onem=1,
        title="Hafta boşluğu kapısı — hafta sonunu aşan setup dirilmesin",
        trigger="10 week_gap elemesi: elenenlerin WR'si taban WR'yi 5 puan geçerse kural gevşer",
        measure="python scripts/deleted_gate_stat.py · /setup-journal (kapı week_gap)",
        md="Hafta boşluğu kapısı — hafta sonunu aşan setup dirilmesin (canlıya alındı 21.09.2026)",
        progress_fn=_p_week_gap,
    ),
    WatchItem(
        key="market_vs_limit_fee", status="done", started="27.09", onem=1,
        title="Market vs limit giriş — komisyonsuz / komisyonlu",
        trigger="400 first_bar'lı setup: market kolu NET limiti +0.15 R/setup ve iki zaman yarısında geçerse mum re-track'i",
        result="(01.10) STRATEJİ BAZINDA ÜÇ MADDEYE BÖLÜNDÜ (kullanıcı): 4H / 1D / 1W ayrı tetikle. Bölünmeden önceki "
               "ara okumada (354) her market kolu limitin altındaydı; 114 'TP'si doğmadan geçilmiş' satır dışlandı.",
        measure="python scripts/market_vs_limit_stat.py --strategy 4h|1d|1w",
        md="Market vs limit giriş — komisyonsuz / komisyonlu (27.09.2026, ⏰ 03.10)",
        progress_fn=_p_market_fee,
    ),
    WatchItem(
        key="market_vs_limit_4h", status="open", started="01.10", onem=1,
        title="Market vs limit giriş — 4H",
        trigger="03.10'dan beri kapıları geçmiş + first_bar dilimi: market %90 / %80 RR ≥ 2 kolu 50 çözülmüş işlem → "
                "komisyonlu VE komisyonsuz hüküm ayrı (+0.15 R/setup, iki yarı); ikisi de kullanıcıya söylenir, karar onun",
        measure="python scripts/market_vs_limit_stat.py --strategy 4h",
        md="Market vs limit giriş — 4H",
        progress_fn=_p_market_gated_for("4h", "kapıları geçmiş + first_bar · 03.10: %90 +0.125 R/setup (14 işlem), "
                                              "%80 +0.02 (23)"),
    ),
    WatchItem(
        key="market_vs_limit_1d", status="open", started="01.10", onem=1,
        title="Market vs limit giriş — 1D",
        trigger="03.10'dan beri 4H gibi: kapıları geçmiş + first_bar dilimi, market %90 / %80 RR ≥ 2 kolu 50 çözülmüş "
                "işlem → komisyonlu VE komisyonsuz hüküm ayrı (+0.15 R/setup, iki yarı); ikisi de kullanıcıya söylenir, "
                "karar onun",
        measure="python scripts/market_vs_limit_stat.py --strategy 1d (en alttaki KARAR bloğu)",
        md="Market vs limit giriş — 1D",
        progress_fn=_p_market_gated_for("1d", "kapıları geçmiş + first_bar · 03.10 (10 setup): %90 −0.47 / %80 −0.56 "
                                              "R/setup, komisyonsuz da eksi"),
    ),
    WatchItem(
        key="market_vs_limit_1w", status="open", started="01.10", onem=1,  # 03.10: 0 -> 1, kapiyi gecen 1W setup yok, aylar
        title="Market vs limit giriş — 1W",
        trigger="03.10'dan beri 4H/1D gibi: kapıları geçmiş + first_bar dilimi, market %90 / %80 RR ≥ 2 kolu 50 çözülmüş "
                "işlem → komisyonlu VE komisyonsuz hüküm ayrı (+0.15 R/setup, iki yarı), karar kullanıcının. 01.10'dan beri "
                "1W limiti = yalnız CISD + %80 stop (kullanıcı kararı); market kolları o SL'den ölçülür",
        measure="python scripts/market_vs_limit_stat.py --strategy 1w (en alttaki KARAR bloğu)",
        md="Market vs limit giriş — 1W",
        progress_fn=_p_market_gated_for("1w", "kapıları geçmiş + first_bar · 03.10: kapıyı geçen 1W setup yok (tüm "
                                              "kümede 25 setup, limit 2 TP / 16 SL)"),
    ),
    WatchItem(
        key="reclaim_penalty_by_tf", status="open", started="01.10", onem=0,
        title="Zayıf C2 geri dönüşü cezası (−4) alan setup gerçekten stop oluyor mu?",
        trigger="strateji başına 20 çözülmüş ceza alan setup (ham skor ≥ eşik): ceza alanın R/setup'ı cezasızdan "
                "0.15'ten az kötüyse o stratejide ceza gevşetme adayı (−2, replay öncesi kullanıcıya); değilse kalır",
        measure="python scripts/reclaim_penalty_stat.py",
        md="Zayıf C2 geri dönüşü cezası (−4) alan setup gerçekten stop oluyor mu?",
        progress_fn=_p_reclaim_penalty,
    ),
    WatchItem(
        key="zone_presence", status="done", started="27.09", onem=1,
        title="Bölgesiz setup daha çok mu stop oluyor?",
        trigger="her grupta 30 çözülmüş (C2 kapalı, skor ≥ 7, RR ≥ 2): bölgesizin SL oranı "
                "≥ 10 puan ve R/setup ≥ 0.25 kötüyse bölge şartı önerilir (21.10: ceza/puan sorusu 'Skor tablosu yeni "
                "adaylar'da, aday no_zone)",
        measure="python scripts/zone_presence_stat.py · setup_journal.entries",
        md="Bölgesiz setup daha çok mu stop oluyor? (ölçüm 27.09.2026 — kapandı 21.10)",
        result="(21.10) FİLTRE YOK, konu 'Skor tablosu yeni adaylar'a devredildi (kullanıcı): karar dilimi bölgeli 47 · +0.06 vs "
               "bölgesiz 38 · −0.18 R/setup, SL oranı farkı 1.8 puan (H1 10 ister) — fark tamamen 1D'nin 10 setupundan, 4H'te "
               "−0.08 / −0.08. Puan + filtre sorusu aday no_zone'da (≥ 0.3 R kötüyse bölge şartı seçeneği de).",
        progress_fn=_p_zone_presence,
    ),
    WatchItem(
        key="c2_open_fill_4h", status="open", started="27.09", onem=1,
        title="4H'te C2 açıkken dolum olsun mu?",
        trigger="30 çift: C2 açıkken CISD onaylı dolum (A) bugünkü kuralı (B) +0.15 R/setup "
                "geçerse 4H require_c2_closed=False önerilir",
        measure="python scripts/c2_open_fill_stat.py · setup_journal.pre_c2",
        md="4H'te C2 açıkken dolum olsun mu? (27.09.2026, restart bekliyor)",
        progress_fn=_p_c2_open_fill,
    ),
    WatchItem(
        key="c2_refreeze", status="done", started="27.09", onem=2,
        result="(03.10) KAPANDI, kullanıcı onayı: çalışıyor — kapıları geçmiş 4H satırlarında kullanılabilir (C2 kapalı seviye) "
               "%11 → %96 (24/211 → 77/80), birikim 2.5 → 13.3/gün; C2-açık kalan 3 satır beklenen istisna (setup C2 kapanmadan "
               "düştü / C2 hâlâ açık).",
        title="Journal: 4H setup'ta SL/TP seviyeleri C2 kapanışında tekrar güncellenir",
        trigger="kapıları geçmiş, yeniden dondurulmuş 30 çözülmüş 4H satırı: kullanılabilir küme büyüdü mü, yeni 4H "
                "satırlarında C2 açık seviyeyle kalan 'geçmiş' satır var mı (28.09 ara okuma: mekanizma çalışıyor)",
        measure="python tmp/tmp_gate_passed_depth.py · setup_journal.pre_c2",
        md="Journal: 4H setup'ta SL/TP seviyeleri C2 kapanışında tekrar güncellenir (27.09.2026)",
        progress_fn=_p_c2_refreeze,
    ),
    WatchItem(
        key="purge_threshold", status="done", started="28.09", onem=1,
        title="Purge eşiği kaldırıldı (%0.05 → 0)",
        trigger="restart sonrası 30 sonuçlanmış eşik altı (sweep_pct < 0.05) setup: "
                "eşik altı ≥ 0.3 R/setup kötüyse eşik geri gelir (C1 aralığına oranlı)",
        measure="setup_journal.features.sweep_pct · outcome (düz TP/SL)",
        md="Purge eşiği kaldırıldı (28.09.2026 — kapandı 01.10)",
        result="(01.10) EŞİK KALDIRILMIŞ KALIYOR (H2): eşik altı süpürmeler kötü değil — tüm adaylarda 136 vs 795 setup, "
               "fark −0.05 R/setup (%90 aralık −0.22…+0.14, 0.3'lük kötülük dışlandı); CRT + RR ≥ 2'de −0.07 (n 36/106), "
               "FX'te eşik altı tersine daha iyi. Kapı koşullu küme 3/15 — karar kuralının izin verdiği geniş kümeden. "
               "Yeniden okuma 21.10 (taze 02.10+): karar tutuyor — CRT + RR ≥ 2'de fark −0.10 (61/265), iki yarıda aynı yön.",
        progress_fn=_p_purge_threshold,
    ),
    WatchItem(
        key="bias_outcome_4h", status="open", started="03.10", onem=0, pinned=True, pin_order=0,
        title="1D bias kazananı seçiyor mu? — 4H",
        trigger="taze (levels_at ≥ 04.10) temiz 4H setupta bias hizalı ve ters her biri 100 çözülmüş: B1 ters − hizalı "
                "≥ +0.15 R/setup + WR ≥ 0 + iki yarı → filtre/+2 için tek replay · B2 ters − NEUTRAL ≤ −0.15 + iki yarı → "
                "filtre kalır · B3 → değişiklik yok, kullanıcıya sor; ayrıca yalnız-ict karşı-olgusu (C1)",
        measure="python scripts/bias_outcome_stat.py (en alttaki KARAR bloğu) · python scripts/bias_stat.py",
        md="1D bias 4H'te kazananı seçiyor mu? (03.10.2026)",
        progress_fn=_p_bias_outcome_for("4h", datetime(2026, 10, 4), 100,
                                        "03.10 görülen veri (482): hizalı %28.7 · NEUTRAL %30.8 · ters %36.1 WR — "
                                        "karar taze veriyle"),
    ),
    WatchItem(
        key="bias_outcome_1d", status="open", started="03.10", onem=0, pinned=True, pin_order=0,
        title="1D bias kazananı seçiyor mu? — 1D",
        trigger="temiz 1D setupta (bütün veri) bias hizalı ve ters her biri 50 çözülmüş: 4H ile aynı kural — B1 ters − "
                "hizalı ≥ +0.15 R/setup + WR ≥ 0 + iki yarı → tek replay · B2 ters − NEUTRAL ≤ −0.15 + iki yarı → filtre "
                "kalır · B3 → kullanıcıya sor · C1 yalnız-ict karşı-olgusu",
        measure="python scripts/bias_outcome_stat.py --strategy 1d (en alttaki KARAR bloğu)",
        md="1D bias 1D'de kazananı seçiyor mu? (03.10.2026)",
        progress_fn=_p_bias_outcome_for("1d", None, 50,
                                        "03.10'dan beri yalnız seviyesi kesin 1D satırı (eski 130'un çoğu C2 açık "
                                        "donmuş, koşan SL) — örneklem baştan birikiyor"),
    ),
    WatchItem(
        key="key_level_filter", status="open", started="29.09", onem=0, pinned=True,
        title="Key level teması yoksa işlem açılmasın mı? (+ EQ/SSL puan almalı mı)",
        trigger="29.09 sonrası her grupta 30 sonuçlanmış setup (kapı geçmiş, RR ≥ 2; 01.10'da 20 → 30) + fark tek yönlü "
                "bootstrap p < 0.10: key level'siz grup ≥ 0.3 R/setup kötüyse hard filtre, "
                "0.15–0.3 → puan sorusu 'Skor tablosu yeni adaylar'da (21.10), < 0.15 ise değişiklik yok (önce tek replay); EQ/SSL puanı da yeni adaylarda; EQ/SSL'li grup PDL/PWL'li kadar "
                "iyiyse +1 puan adayı",
        measure="setup_journal.features.key_level_n (EQ/SSL dahil etiket sayısı) · outcome (düz TP/SL)",
        md="Key level teması yoksa işlem açılmasın mı? (başladı 29.09.2026, ⏰ tetik: kapı geçmiş setupta grup başına 30)",
        progress_fn=_p_key_level_filter,
    ),
    WatchItem(
        key="liq_beyond", status="done", started="29.09", onem=0,
        title="Purge ötesinde alınmamış likidite varsa setup daha çok stop mu oluyor?",
        trigger="29.09 sonrası her grupta 20 sonuçlanmış setup: 0.5 ATR içinde alınmamış likidite olan grup "
                "≥ 0.3 R/setup kötüyse kapı/ceza adayı (önce tek replay)",
        measure="setup_journal.features.liq_beyond_atr / liq_beyond_frac · outcome (düz TP/SL)",
        md="Purge ötesinde alınmamış likidite (29.09.2026 — kapandı 01.10)",
        result="(01.10) KAPI YOK — hipotezin tersi: 0.5 ATR içinde alınmamış likidite olan setup daha İYİ gidiyor "
               "(CRT + RR ≥ 2: yakın 24 setup WR %45.8 / +0.95 R, uzak/yok 55 setup %12.7 / −0.53 R); her gün × yön "
               "hücresinde aynı yönde. Veri 2 günlük — puan adayı olarak ayrı soru, karar değil. Yeniden okuma 21.10 (taze "
               "02.10+): kapı kararı tutuyor (yakın +0.14 vs uzak −0.20, 91/235); 'yakın daha iyi' etkisi küçüldü, son haftada ters.",
        progress_fn=_p_liq_beyond,
    ),
    WatchItem(
        key="liq_near_bonus", status="done", started="01.10", onem=1,
        title="Purge yakınında alınmamış likidite: +1 puan hak ediyor mu?",
        trigger="02.10 sonrası her grupta 50 sonuçlanmış setup (CRT + RR ≥ 2): yakın grup ≥ 0.3 R/setup iyi, iki yarıda "
                "aynı yön ve tek yönlü bootstrap p < 0.10 ise +1 puan adayı (önce tek replay)",
        measure="setup_journal.features.liq_beyond_atr ≤ 0.5 · outcome (düz TP/SL) · okuma tmp/tmp_kl_liq_read.py",
        md="Purge yakınında alınmamış likidite: +1 puan hak ediyor mu? (başladı 01.10.2026, ⏰ tetik: grup başına 50 taze setup)",
        result="(21.10) DEVREDİLDİ (kullanıcı onayı): puan sorusu \"Skor tablosu yeni adaylar — 4H / 1D / 1W\" maddelerinde "
               "(aday liq_near, aynı 0.5 ATR eşiği, karar bandı yöntemiyle). Devirde taze 02.10+ veri: yakın +0.14 vs uzak "
               "−0.20 (91/235), p 0.11, ikinci yarıda ters — eski kuralla +1 adayı değildi.",
        progress_fn=_p_liq_near_bonus,
    ),
    WatchItem(
        key="c1_shape", status="open", started="29.09", onem=0,
        title="C1'in şekli: TP'si C1'in uzun fitilinin ucunda olan setup daha kötü mü?",
        trigger="fitilli grupta (TP fitili ≥ 0.40) 100 sonuçlanmış RR ≥ 2 setup + fark tek yönlü bootstrap p < 0.10: "
                "fitilli grup ≥ 0.3 R/setup kötüyse hard filtre, 0.15–0.3 → puan sorusu 'Skor tablosu yeni adaylar'da (21.10), < 0.15 ise değişiklik yok "
                "(önce tek replay). 21.10: H1 (aynı renk) kapandı — fark yok, filtre geri gelmiyor",
        measure="python scripts/c1_shape_stat.py · setup_journal.features.c1_same_color / c1_body_frac / c1_tp_wick_frac"
                " · 30.09: 4H'te C1 gövdesi < %30 filtrelendi (c1_weak, kullanıcı kararı)",
        md="C1'in şekli — fitil tepesine TP (başladı 29.09.2026; aynı renk kapandı 21.10, ⏰ tetik: fitilli grupta 100)",
        progress_fn=_p_c1_shape,
    ),
    WatchItem(
        key="cisd_mss", status="open", started="29.09", onem=1,
        title="Sadece CISD onaylı setup, CISD + MSS onaylıdan daha çok mu stop oluyor?",
        trigger="kapılardan geçmiş (RR ≥ 2) her grupta 20 çözülmüş setup: sadece CISD grubu ≥ 0.3 R/setup kötüyse "
                "MSS şartı, 0.15–0.3 → puan sorusu 'Skor tablosu yeni adaylar'da (21.10), < 0.15 ise değişiklik yok (önce tek replay)",
        measure="python scripts/cisd_mss_stat.py · setup_journal.retrace.brk (CISD/MSS kapanışla ilk kırılım) · entries.chosen",
        md="CISD onayı mı, CISD + MSS onayı mı? (ilk okuma 29.09.2026)",
        progress_fn=_p_cisd_mss,
    ),
    WatchItem(
        key="prefill_signal", status="open", started="29.09", onem=1,
        title="Gerçek sinyallerde: TP yolunun %50'sini koşup entry'ye dönen işlem stop mu oluyor?",
        trigger="grup başına 100 kapalı işlem (03.10, kullanıcı: 10 çok az): ≥ %50 grubu SL oranı ≥ 15 puan yüksek VE "
                "R/işlem ≥ 0.30 kötüyse kapı adayı (önce tek replay), aksi değişiklik yok",
        measure="python scripts/prefill_signal_stat.py · signals.prefill_run / prefill_cov",
        md="Gerçek sinyallerde dolum öncesi %50+ koşu (29.09.2026, restart bekliyor)",
        progress_fn=_p_prefill_signal,
    ),
    WatchItem(
        key="ifvg_from_c2", status="done", started="29.09", onem=2,
        title="IFVG'nin FVG'si C2 içinde oluşmalı (C1'deki boşluklar aday değil)",
        trigger="restart sonrası 10 kapalı bölge girişli sinyal: 21–28.09'a göre R/işlem 0.3R'den "
                "fazla düşmediyse kural kalır; 'en yakın bölge' maddesiyle aynı oturumda okunur (ayrıştırılamaz)",
        measure="signals (entry_model ifvg/bpr, created_at ≥ 29.09) + dar stop elemeleri (setup_journal best_stage)",
        md="IFVG'nin FVG'si C2 içinde oluşmalı (29.09.2026, restart bekliyor)",
        result="(21.10) KURAL KALIYOR (kullanıcı onayı, kural lafzen tutmadı): bölge girişli R/işlem 21–28.09 +0.54 (17) → 29.09+ "
               "−0.14 (12), düşüş 0.68 > 0.3 — ama tamamı BPR'den (7 işlem 1 TP / 6 SL); IFVG girişi +0.39 (5). CISD/MSS aynı "
               "dönemde −0.22 → +0.16. BPR sorunu 'Skor tablosu yeni adaylar' aday bpr_entry'de. Dar stoplu bölge silmesi 0.",
        progress_fn=_p_ifvg_from_c2,
    ),
    WatchItem(
        key="nearest_zone", status="done", started="27.09", onem=2,
        result="(03.10) KAPANDI, kullanıcı onayı: giriş kuralları KALIYOR — önceden yazılmış dönem kuralı lafzen H2 (bölge "
               "girişli R/işlem +0.72 → −0.10, 12/11 işlem, dönem karışık) ama aynı setupta seçilen bölge vs CISD: kapıları geçmiş 13 setupta "
               "+0.54 / +0.21, 28.09 kuralının değiştirdiği 8 setupta +0.52 / −0.25; dolum %50 → %58. Dönem kuralı bilerek aşıldı.",
        title="Birden çok IFVG'de fiyata en yakın bölge (+28.09: RR ≥ 2 bölge CISD'den sığ olsa da seçilir)",
        trigger="restart sonrası 10 kapalı bölge girişli sinyal: dolum oranı arttı ve R/işlem "
                "0.3R'den fazla düşmediyse kural kalır (iki değişiklik birlikte okunur)",
        measure="signals (entry_model ifvg/bpr, created_at ≥ 28.09) vs 21–27.09",
        md="Birden çok IFVG'de fiyata en yakın bölge (27.09.2026, restart bekliyor)",
        progress_fn=_p_nearest_zone,
    ),
    WatchItem(
        key="rules_0925", status="open", started="25.09", onem=2,  # 28.09: 1 -> 2, canliya alinan degisikligin dogrulamasi
        title="1D C2 hard filtreleri kalktı + CISD bloğu + IFVG C1 şartı",
        trigger="C2 açıkken dolan 10 kapalı 1D işlem: bu işlemler diğer 1D işlemlerden "
                "0.3R+ kötüyse require_c2_closed geri açılır",
        measure="signals (timeframe=1d, entry_filled_time < purge_time + 24s) · setup_journal 1d base=0",
        md="1D C2 hard filtreleri kalktı + CISD bloğu + IFVG C1 şartı (25.09.2026, restart bekliyor)",
        progress_fn=_p_1d_c2_open,
    ),
    WatchItem(
        key="pd_midnight", status="done", started="18.09",
        result="(03.10) KAPANDI, kullanıcı onayı: düzeltme tuttu — 18–29.09 (eski tanım) 00:00–00:30 UTC penceresinde "
               "pd_major %37 (73) vs dışarıda %26 (1175); düzeltme öncesi %14 vs %30. Yeni tanımda (29.09+) %3 vs %8, n 37, anlamsız.",
        title="Gece yarısı PDH/PDL kayması düzeltildi",
        trigger="düzeltmeden sonra pencerede 100 setup: pencere içi pd_major oranı %44.7'ye yakınsadı mı (yoksa %30 altı = tutmadı)",
        measure="setup_journal.score_parts — first_seen 00:00-00:30 UTC penceresi",
        md="Gece yarısı PDH/PDL kayması düzeltildi (canlıya alındı 18.09.2026)",
        progress_fn=_p_pd_midnight,
    ),
    WatchItem(
        key="bias_journal", status="done", started="18.09", onem=1,
        title="1D bias tahmin karnesi",
        # 21.09 ara okumasi: H1 battı (0/3 ufuk) ama hard filtre YERINDE KALDI -- karne 1-5 gun
        # olcuyor, setuplar medyan 1.2 saat yasiyor. Maddenin kalan isi iki alt soru.
        trigger="03.10: htf +2 ödülü hak edilmiş mi (hizalı −0.461 vs NEUTRAL −0.406) · "
                "STRUCTURE_STALE_DAYS düşsün mü (bayat dal %52.0 vs taze AND %46.1)",
        result="KAPANDI (28.09): STRUCTURE_STALE_DAYS = 7 KALIYOR — Journal karşı-olgusunda (3 strateji) hiçbir eşik "
               "(0–∞, yalnız ict) +5R şartına yaklaşmadı; eşik setupların yalnız 0–1'ini (C2 açık dahil ≤14) "
               "değiştiriyor. htf +2 sorusu 'Skor kalemleri — 4H'e devredildi (1H'te +2 kanıtlı). Karne ölçümü sürüyor.",
        measure="python scripts/bias_stat.py",
        md="1D bias tahmin karnesi (18.09.2026 — kapandı 28.09)",
        progress_fn=_p_bias_journal,
    ),
    WatchItem(
        key="stale_days_recheck", status="done", started="01.10", onem=1,
        title="STRUCTURE_STALE_DAYS 7 → 2 mi? (28.09 kararının denetimi)",
        trigger="02.10 sonrası 300 temiz setup: N=2 kolu taze veride A'dan (N=7) > 0 R, 4H'te negatif değil VE tüm veride "
                "≥ +5R + iki yarı > 0 → tek replay; aksi halde 7 kalır",
        measure="python tmp/tmp_stale_days_cf.py (taze dilim: --since 2026-10-02)",
        md="STRUCTURE_STALE_DAYS 7 → 2 mi? (denetim 01.10.2026 — kapandı 21.10)",
        result="(21.10) 7 KALIYOR (kullanıcı onayı): taze 02.10+ 478 setupta N=2 kolu A'dan −0.91R (giren 5: 1 TP / 4 SL) — "
               "eski verideki +5R tekrarlamadı; 1D'de de N=2 kötü (+0.07 vs +2.07).",
        progress_fn=_p_stale_days,
    ),
    WatchItem(
        key="nonsignal_outcomes", status="open", started="22.10", onem=3, pinned=True, pin_order=-1, continuous=True,
        title="Sinyale dönüşmeyen setuplar — kaçı TP, kaçı SL?",
        trigger="SÜREKLİ İZLEME (22.10, kullanıcı): tetik yok, madde kapanmaz — sistemin elediği setupların sonrasını takip",
        measure="/eliminated (filtre bazında TP/SL + bütün setuplar)",
        md="Sinyale dönüşmeyen setuplar — kaçı TP, kaçı SL? (başladı 22.10.2026, sürekli izleme)",
        progress_fn=_p_nonsignal_outcomes,
    ),
    WatchItem(
        key="bias_fidelity", status="open", started="24.09", onem=0, pinned=True, pin_order=0,  # 03.10: bias en onemli konu (kullanici)
        title="1D bias hesap doğruluğu — motor doğru günün bias'ını mı kullanıyor?",
        trigger="SÜREKLİ İZLEME (03.10, kullanıcı): restart sonrası gün dönümünde 50 4H değerlendirmesi → yanlış bias 0 "
                "olmalı (03.10 15m/4h yarış düzeltmesi); hata çıkarsa incele ve düzelt, madde kapanmaz",
        measure="python scripts/bias_fidelity_stat.py",
        md="1D bias hesap doğruluğu (başladı 24.09.2026, ⏰ 03.10.2026)",
        progress_fn=_p_bias_fidelity,
    ),
    WatchItem(
        key="prefill", status="done", started="18.09", onem=3,
        title="Dolum öncesi koşu",
        trigger="0-25% ve >=50% kovalarının her birinde 100 dolan setup (ilk okuma 28.09.2026)",
        result="KAPI AÇILMIYOR (28.09): hipotezin tersi — dolumdan önce TP yolunun ≥%50'sini koşan setup %41.6 "
               "kazanıyor, 0-25% kovası %22.2 (n 166 / 2727). ENA #71 tek vaka olarak kalır.",
        measure="python scripts/prefill_stat.py",
        md="Dolum öncesi koşu (18.09.2026 — kapandı 28.09)",
        progress_fn=_p_prefill,
    ),
    WatchItem(
        key="gated_potential", status="done", started="18.09", onem=3,
        title="Elenen 1D setup bildirimi",
        trigger="02.10.2026: kapsam daraltıldıktan sonra günlük mesaj ~4'e indi mi (8'i aşarsa RR tabanı 1.8)",
        measure="potential_notices — gate dolu satırlar",
        md="Elenen 1D setup bildirimi (18.09.2026, restart bekliyor)",
        result="(29.09) Bildirim kaldırıldı, ölçüm yapılmadı: kullanıcı potansiyel bildirimlerini iptal etti, "
               "tüm stratejilerde yalnız WAITING ENTRY gidiyor (scanner.POTENTIAL_NOTICES_ENABLED = False).",
        progress_fn=_p_gated_potential,
    ),
    WatchItem(
        key="dead_candidate", status="open", started="18.09",
        title="Ölü CRT adayı slotu tutmasın",
        trigger="yeni yolla doğan 10 sinyal: sonuçları + past_sl radar durumu seyrekleşti mi",
        measure="setup_journal — best_stage='waiting' satırları, kardeş not_selected ile",
        md="Ölü CRT adayı slotu tutmasın (18.09.2026, restart bekliyor)",
        progress_fn=_p_dead_candidate,
    ),
    WatchItem(
        key="direction", status="done", started="17.09",
        title="LONG/SHORT ayrışması + 16.09 FOMC etkisi",
        trigger="Her iki yönde de 30 kapalı işlem birikmesi",
        result="KAPANDI (28.09, kullanıcı kararı — kural uygulanmadan): açılıştaki LONG 2/15 vs SHORT 7/12 farkı kapandı "
               "(LONG 31 / SHORT 29 işlem, fark +2.7 puan, 4H'te ikisi de %40). Journal taban oranında SHORT 8 puan geride "
               "= piyasa yönü. Dashboard hatırlatması kaldırıldı; rapor scripts/direction_stat.py duruyor.",
        measure="python scripts/direction_stat.py",
        md="LONG/SHORT ayrışması + 16.09 FOMC etkisi (17.09.2026 — kapandı 28.09)",
        progress_fn=_p_direction,
    ),
    WatchItem(
        key="partial_vs_be", status="open", started="17.09", onem=1,
        title="Kısmi kâr vs BE-only",
        trigger="100 kısmi kârlı kapalı işlem (03.10'da 25 → 100, kullanıcı)",
        measure="python scripts/partial_vs_be.py",
        md="Kısmi kâr vs BE-only — canlı takip (başladı 17.09.2026, ⏰ tetik: 25 işlem)",
        progress_fn=_p_partial,
    ),
    WatchItem(
        key="partial_flat", status="done", started="24.09", onem=1,
        title="Kısmi kâr + BE kapatılsın mı?",
        trigger="200 temiz setup %50'ye gelip çözülünce (Journal; 4H'te C2 kapalıyken donmuş + 1D) — 28.09'da "
                "100 + 03.10'dan değişti, kullanıcı onayı",
        measure="python scripts/partial_flat_stat.py",
        md="Kısmi kâr + BE kapatılsın mı? (ölçüm başladı 24.09.2026 — kapandı 21.10)",
        result="(21.10) KISMİ KÂR + BE KALIYOR (H3, kullanıcı onayı): 350 temiz setupta düz TP/SL −0.015 R/tetik; RR ≥ 2 +0.05, "
               "skor ≥ 7 +0.09, 1D −0.13 — hepsi ±0.15 içinde; canlı 23 işlemde +0.03 R/işlem. BE'de kapanan kalan yarı sonra "
               "TP 79 / SL 117 — getiri aynı, WR daha yüksek.",
        progress_fn=_p_partial_flat,
    ),
    WatchItem(
        key="tight_stop_gate", status="done", started="18.09", onem=1,
        title="Dar stop kapısı — elenen setup TP'ye gidiyor mu?",
        trigger="20 çözülmüş elenen setup (kapı haftada ~1–2 eliyor)",
        measure="python scripts/tight_stop_stat.py",
        md="Dar stop kapısı — elenen setup TP'ye gidiyor mu? (18.09.2026 — kapandı 29.09)",
        result="(29.09) Kapı KALIYOR: elenen 10 çözülmüş setupun 10'u kayıp (−10R), taban WR %22.4. Eşik (20) "
               "dolmadan kapandı — dar stoplu bölge artık atlandığı için elenen akışı azalıyor. Denetim 01.10: 12/12 SL, "
               "aç-kapa sızıntısı kalıcı silmeyle kapandı. "
               "22.10: strateji bazında yeniden açıldı → \"Dar stop kapısı — 4H / 1D / 1W\".",
        progress_fn=_p_tight_stop_gate,
    ),
    WatchItem(
        key="tight_stop_4h", status="open", started="22.10", onem=1,
        title="Dar stop kapısı — 4H",
        trigger="dar grupta çözülmüş 30 (kapıya kadar bütün kapıları geçmiş setuplar, stop oranı < 1.0 = dar, düz TP/SL): dar − normal ≤ −0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → KAPI DOĞRU, kalır; dar normalden 0.15'ten az kötü VE dar R/setup > 0 → kapı kazanç kesiyor → tek replay → kullanıcıya (kaldır / eşiği düşür); arada 60'ye uzat. Dinamik eşik: stop oranı kovaları (bilgi)",
        measure="python scripts/tight_stop_stat.py",
        md="Dar stop kapısı — strateji bazında (4H / 1D / 1W)",
        progress_fn=_p_tight_stop_tf("4h"),
    ),
    WatchItem(
        key="tight_stop_1d", status="open", started="22.10", onem=1,
        title="Dar stop kapısı — 1D",
        trigger="dar grupta çözülmüş 20 (kapıya kadar bütün kapıları geçmiş setuplar, stop oranı < 1.0 = dar, düz TP/SL): dar − normal ≤ −0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → KAPI DOĞRU, kalır; dar normalden 0.15'ten az kötü VE dar R/setup > 0 → kapı kazanç kesiyor → tek replay → kullanıcıya (kaldır / eşiği düşür); arada 40'ye uzat. Dinamik eşik: stop oranı kovaları (bilgi)",
        measure="python scripts/tight_stop_stat.py",
        md="Dar stop kapısı — strateji bazında (4H / 1D / 1W)",
        progress_fn=_p_tight_stop_tf("1d"),
    ),
    WatchItem(
        key="tight_stop_1w", status="open", started="22.10", onem=1,
        title="Dar stop kapısı — 1W",
        trigger="dar grupta çözülmüş 10 (kapıya kadar bütün kapıları geçmiş setuplar, stop oranı < 1.0 = dar, düz TP/SL): dar − normal ≤ −0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → KAPI DOĞRU, kalır; dar normalden 0.15'ten az kötü VE dar R/setup > 0 → kapı kazanç kesiyor → tek replay → kullanıcıya (kaldır / eşiği düşür); arada 20'ye uzat. Dinamik eşik: stop oranı kovaları (bilgi) (1W'de kapıyı geçen setup ayda birkaç tane — rapordaki geniş küme erken işaret verir)",
        measure="python scripts/tight_stop_stat.py",
        md="Dar stop kapısı — strateji bazında (4H / 1D / 1W)",
        progress_fn=_p_tight_stop_tf("1w"),
    ),
    WatchItem(
        key="missed_quality", status="open", started="21.09", onem=0,
        title="Retest skor kapısı — kazanan mı eliyor?",
        trigger="her C2 zamanlama diliminde 30 çözülmüş elenen setup (RR ≥ 2, 4H/1D/1W; 01.10'da 20 → 30/dilim)",
        measure="python scripts/missed_quality_stat.py",
        md="Retest skor kapısı — kazanan mı eliyor? (başladı 21.09.2026, ⏰ tetik: dilim başına 30 çözülmüş setup)",
        progress_fn=_p_missed_quality,
    ),
    WatchItem(
        key="c1_stop", status="done", started="17.09",
        title="C1 ucu stop — gölge izleme",
        trigger="28.09.2026 değerlendirmesi (kardeş madde “Dar stop” ile aynı oturumda)",
        measure="python scripts/c1_stop_stat.py",
        md="C1 ucu stop — gölge izleme (17.09.2026 — kapandı 21.09)",
        result="C1 ucu stop KULLANILMIYOR (21.09): H1 her dilimde battı (havuz -0.096, 4H -0.082, "
               "1H -0.160 R/işlem). Bugünkü SL'le TP olan 136 setupun 45'i C1'le stop olurdu, C1'in "
               "kurtardığı sıfır. Fikri açık tutan skor>=7 dilimi n=102'de -0.009'a oturdu; kural "
               "setupların %52'sinde zaten uygulanamıyor.",
        progress_fn=_p_c1_stop,
    ),
    WatchItem(
        key="deleted_gate", status="done", started="19.09", onem=1,
        title="Silinen waiting setup — kapı kazanan sinyal mi eliyor?",
        trigger="20 çözülmüş silinen setup (~7/gün siliniyor; skor kapıları ~1,3/gün)",
        measure="python scripts/deleted_gate_stat.py",
        md="Silinen waiting setup — kapı kazanan sinyal mi eliyor? (19.09.2026 — kapandı 01.10)",
        result="(01.10) Skor elemesi kazanan kesmiyor: skor nedenleriyle silinen 11 setup 1 TP / 10 SL (−0.70 R/işlem); "
               "kapılar kalıyor. H3: silinip geri doğan 13 işlem 2 TP / 11 SL (−0.68), kalan 54 işlem +0.17 → "
               "KALICI SİLME canlıya alındı (kullanıcı kararı): bir kez silinen CRT yeniden sinyal açmaz. "
               "Doğrulaması 'Kalıcı silme' maddesinde.",
        progress_fn=_p_deleted_gate,
    ),
    WatchItem(
        key="sticky_delete", status="open", started="01.10", onem=2,
        title="Kalıcı silme — engellenen setup TP'ye gidiyor mu?",
        trigger="02.10 sonrası silinen 30 çözülmüş setup (RR ≥ 2, past_sl/past_tp hariç): R/işlem > +0.3 ve "
                "bootstrap p < 0.10 ise kural kazanç kesiyor → kullanıcıya geri alma önerisi",
        measure="python scripts/deleted_gate_stat.py · log SKIPPED (DELETED BEFORE) · radar deleted_before",
        md="Kalıcı silme — engellenen setup TP'ye gidiyor mu? (başladı 01.10.2026, ⏰ tetik: 30 çözülmüş setup)",
        progress_fn=_p_sticky_delete,
    ),
    WatchItem(
        key="bias_1h", status="done", started="20.09", onem=1,
        title="1D bias 1H'te iki kez mi eliyor?",
        trigger="25 çözülmüş bias elemesi (1H, ~2/gün) — “Skor kalemleri” ile aynı oturumda",
        result="İKİ KANAL DA KALIYOR (28.09): 1H hard filtresinin elediği 40 çözülmüş setup −0.417 R/işlem "
               "(koruyucu); htf +2 ile 0 farkı 1H'te +8.3 puan (n 51/38, kalem kanıtlı). 4H'te +2 vs 0 farkı "
               "+3.0 — orada karar 'Skor kalemleri — 4H' maddesinde. ⚠️ +8.3 bütün skor bantlarından; seçilimsiz bant 0–6'da "
               "+1.6 — +2 sorusu 'hizalı setup'a +2' maddesinde yeniden okunuyor.",
        measure="python scripts/bias_1h_stat.py",
        md="1D bias 1H'te iki kez mi eliyor? (20.09.2026 — kapandı 28.09)",
        progress_fn=_p_bias_1h,
    ),
    WatchItem(
        key="score_parts_4h", status="open", started="16.09", onem=1,
        title="Skor kalemleri — 4H",
        trigger="paketin kaderini değiştirdiği 150 setup (21.10, kullanıcı; eskiden A kolu 30): paket (PD major/aylık 0 + "
                "wick 0.30, eşik 6) değişen setuplarda ≥ +0.15 R/setup, iki yarıda aynı yön ve tek yönlü p < 0.10 → tek replay; "
                "değilse paket reddedilir. Gerekirse kalem kalem okuma (score_item_outcome_stat) aynı oturumda yenilenir",
        measure="python scripts/score_item_outcome_stat.py --strategy 4h (kalem başına TP/SL) · "
                "python scripts/score_parts_stat.py --strategy 4h",
        md="Skor kalemleri — 4H (ölçüm 16.09.2026, ilk okuma 28.09, ⏰ 03.10.2026)",
        progress_fn=_p_score_parts_4h,
    ),
    WatchItem(
        key="smt_variants", status="open", started="03.10", onem=0,  # 03.10: kullanici "SMT cok onemli konfirme"
        title="SMT doğru tanımla — bugünkü / HTF / seviyeye bağlı",
        trigger="çift türüne göre ayrı (FX/metal/endeks ↔ altcoin→BTC): karar bandında taraf başına 10 çözülmüş; bir tanım "
                "SMT'li − SMT'siz ≥ +0.15 R/setup ve iki yarıda > 0 ise 'kazananı ayırıyor' → o tür için bugünkü tanımın "
                "yerine aday (tek replay); bugünkü tanım hiçbir türde ayırmıyorsa kullanıcıyla konuşulur",
        measure="python scripts/smt_variants_stat.py [--strategy 4h|1d|1w]",
        md="SMT doğru tanımla — bugünkü / HTF / seviyeye bağlı (03.10.2026)",
        progress_fn=_p_smt_variants,
    ),
    WatchItem(
        key="score_proposals_4h", status="open", started="03.10", onem=1,
        title="Skor önerileri — 4H",
        trigger="taze (levels_at ≥ 04.10) A kolu 60 setup: öneri A'ya göre ≥ +3R ve iki yarıda > 0 → tek replay adayı; "
                "≤ −3R ve iki yarıda < 0 → red. Öneriler: SMT +1 / 0, FVG-OB 0, major +2, C2 doji/dolu yer değiştir, "
                "C1 aralığı/ATR ≥ 1.3 −1, 1W 0, IFVG 0, wick 0.30. SMT önerileri ASKIDA (03.10, kullanıcı: SMT önemli, "
                "çıkarmak yerine düzelt → 'SMT doğru tanımla')",
        measure="python scripts/score_proposals_stat.py (en alttaki KARAR bloğu)",
        md="Skor önerileri — 4H (03.10.2026)",
        progress_fn=_p_score_proposals_4h,
    ),
    WatchItem(
        key="score_proposals_1d", status="open", started="03.10", onem=1,
        title="Skor önerileri — 1D",
        trigger="1D 25.09+ bütün veri, A kolu (bugünkü kural @7) 30 setup: 4H ile aynı kural — öneri ≥ +3R ve iki yarıda > 0 "
                "→ tek replay adayı; ≤ −3R → red. SMT önerileri askıda",
        measure="python scripts/score_proposals_stat.py --strategy 1d (en alttaki KARAR bloğu)",
        md="Skor önerileri — 1D (03.10.2026)",
        progress_fn=_p_score_proposals_1d,
    ),
    WatchItem(
        key="score_parts_1d", status="open", started="16.09", onem=1,
        title="Skor kalemleri — 1D",
        trigger="300 sonuçlanmış 1D setup (25.09+; 03.10'da tarih doldu, kalemlerin karar bandı < 10 → örneklem "
                "tetiği): her kalem karar bandında taraf başına 10 çözülmüş, ±0.15 R/setup — hüküm raporda",
        measure="python scripts/score_item_outcome_stat.py --strategy 1d (kalem başına TP/SL) · "
                "python scripts/score_parts_stat.py --strategy 1d  (C2/renk: --since 2026-09-25)",
        md="Skor kalemleri — 1D (ölçüm 16.09.2026, ⏰ okuma 03.10.2026)",
        progress_fn=_p_score_parts_1d,
    ),
    WatchItem(
        key="score_parts_1w", status="open", started="01.10", onem=1,
        title="Skor kalemleri — 1W",
        trigger="kalem başına karar bandında (kalemin sinyali değiştirdiği setuplar) taraf başına 10 çözülmüş: puan alan "
                "almayandan 0.15 R/setup kötüyse (cezada iyiyse) kalem TERS → kullanıcıya; aylık bias, haftalık, PMH/PML 1W'ye özgü",
        measure="python scripts/score_item_outcome_stat.py --strategy 1w",
        md="Skor kalemleri — 1W",
        progress_fn=_p_score_parts_1w,
    ),
    WatchItem(
        key="score_candidates_4h", status="open", started="21.10", onem=1,
        title="Skor tablosu yeni adaylar — 4H",
        trigger="bir adayda karar bandında (adayın puanı sinyali değiştirecek setuplar, RR ≥ 2) taraf başına 30 çözülmüş: lehte fark ≥ 0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → tek replay → kullanıcıya; < 0.15 RED; arada taraf başına 60'a uzat. Filtre (≥ 0.3 R) adayın eski maddesinde",
        measure="python scripts/score_candidates_stat.py --strategy 4h · adaylar app/score_outcome.CANDIDATES (seans, CISD "
                "gücü, likidite, EQ/SSL, MSS, C1 aralığı, key level yok, C1 TP fitili, IFVG/BPR bölgesi oluşmamış, BPR girişi)",
        md="Skor tablosu yeni adaylar — 4H",
        progress_fn=_p_score_candidates("4h"),
    ),
    WatchItem(
        key="score_candidates_1d", status="open", started="21.10", onem=1,
        title="Skor tablosu yeni adaylar — 1D",
        trigger="bir adayda karar bandında (adayın puanı sinyali değiştirecek setuplar, RR ≥ 2) taraf başına 30 çözülmüş: lehte fark ≥ 0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → tek replay → kullanıcıya; < 0.15 RED; arada taraf başına 60'a uzat. Filtre (≥ 0.3 R) adayın eski maddesinde",
        measure="python scripts/score_candidates_stat.py --strategy 1d · adaylar app/score_outcome.CANDIDATES (seans, CISD "
                "gücü, likidite, EQ/SSL, MSS, C1 aralığı, key level yok, C1 TP fitili, IFVG/BPR bölgesi oluşmamış, BPR girişi)",
        md="Skor tablosu yeni adaylar — 1D",
        progress_fn=_p_score_candidates("1d"),
    ),
    WatchItem(
        key="score_candidates_1w", status="open", started="21.10", onem=3,
        title="Skor tablosu yeni adaylar — 1W",
        trigger="bir adayda karar bandında (adayın puanı sinyali değiştirecek setuplar, RR ≥ 2) taraf başına 30 çözülmüş: lehte fark ≥ 0.15 R/setup + iki yarıda aynı yön + tek yönlü p < 0.10 → tek replay → kullanıcıya; < 0.15 RED; arada taraf başına 60'a uzat. Filtre (≥ 0.3 R) adayın eski maddesinde "
                "(1W pasif: kapıyı geçen setup neredeyse yok, rapor istenince okunur)",
        measure="python scripts/score_candidates_stat.py --strategy 1w · adaylar app/score_outcome.CANDIDATES (seans, CISD "
                "gücü, likidite, EQ/SSL, MSS, C1 aralığı, key level yok, C1 TP fitili, IFVG/BPR bölgesi oluşmamış, BPR girişi)",
        md="Skor tablosu yeni adaylar — 1W",
    ),
    WatchItem(
        key="retest_ifvg", status="open", started="03.10", onem=1,
        title="CISD retest sonrası IFVG — ikinci konfirme mi?",
        trigger="A: strateji başına 30 kapıdan geçmiş (CISD RR ≥ 2) setupta retest sonrası IFVG oluşmuş — yalnız "
                "rfvg_near girişi CISD girişinden ≥ 0.15 R/setup iyiyse 'IFVG'yi bekle' kuralı kullanıcıya (replay "
                "sonrası); B: CISD RR < 2 iken IFVG RR ≥ 2 olan 20 sonuçlanmış giriş — R/işlem işareti",
        measure="python scripts/retest_ifvg_stat.py [--strategy 4h|1d|1w]",
        md="CISD retest sonrası IFVG — ikinci konfirme mi?",
        progress_fn=_p_retest_ifvg,
    ),
    WatchItem(
        key="smt_segment", status="open", started="02.10", onem=1,
        title="SMT +2 — segment bazında (altcoin / BTC-ETH / FX)",
        trigger="21.10'dan beri YALNIZ 02.10+ veri: segment × strateji karar bandında (SMT hariç skor 5–6) taraf başına 60 (30'da okundu 21.10) "
                "çözülmüş + iki zaman yarısı aynı yön + tek yönlü p < 0.10: SMT'li ≥ 0.15 R/setup kötüyse o segmentte "
                "puanı kaldırma önerisi (önce 'SMT doğru tanımla' sonucuna bakılır); |fark| < 0.15 etkisiz",
        measure="python scripts/score_item_outcome_stat.py  (en alttaki SMT segment bölümü)",
        md="SMT +2 — segment bazında (altcoin / BTC-ETH / FX)",
        progress_fn=_p_smt_segment,
    ),
    WatchItem(
        key="w1_live", status="open", started="28.09", onem=2,
        title="1W-4H stratejisi canlıda (1H-5M'in yerine)",
        trigger="3 kapanan 1W sinyal: no_data yok mu (veri derinliği), kaç setup/sinyal, Cuma kapanışı kaç 1W "
                "işlemi kesti; skor eşiği 6 ve aylık bias +2 için yalnız 'veri birikiyor' notu",
        measure="/radar?tf=1w · /setup-journal?strategy=1w · log: “1w” satırları",
        md="1W-4H stratejisi canlıda (28.09.2026, restart bekliyor, ⏰ 03.10)",
        progress_fn=_p_w1_live,
    ),
    WatchItem(
        key="htf_bonus", status="done", started="28.09", onem=1,
        title="1D bias: hizalı setup'a +2 puan hak ediliyor mu?",
        result="(03.10) KAPANDI, devredildi (kullanıcı onayı): 4H W40 +0.5 → H3 (fark ~0); soru \"1D bias kazananı "
               "seçiyor mu? — 4H / — 1D\" maddelerinde doğru yöntemle sürüyor. Bu maddenin ölçümü hatalıydı: CRT-öncesi "
               "adaylar dahil ve +2 kovası NEUTRAL'da PD dönüşünü de sayıyordu (1D hizalı 47 → doğrusu 21).",
        trigger="03.10.2026: W40'ta (bant 0–6) htf +2 ile htf 0 farkı, TF bazında (4H/1D) × market — 4H W38 +5.8 / "
                "W39 −7.5 (salınım kriptodan); 1H 28.09'da kapandı (W38–39: −0.6 / +5.1, karar dışı); "
                "1D hafta şartsız havuz 50/50 (28.09: 27/36). "
                "Ters (−2) = hard filtre doğru mu da aynı tabloda (4H filtre kripto dışında güçlü); kural IZLEME.md'de",
        measure="python tmp/tmp_htf_bonus_weeks.py  (--md: IZLEME tablosu)",
        md="1D bias: hizalı setup'a +2 puan hak ediliyor mu? (28.09.2026, ⏰ 03.10)",
        progress_fn=_p_htf_bonus,
    ),
    WatchItem(
        key="ws_watchdog", status="open", started="16.09", onem=3,
        title="WS bekçisi — 60 sn veri yoksa yeniden bağlan",
        trigger="Bekçi günde birkaç kez GEREKSİZ tetiklenirse (normal gün 0–1)",
        measure="Dashboard sağlık şeridi → WS kopma sayısı; log'da “60 sn veri yok”",
        md="WS bekçisi — 60 sn veri yoksa yeniden bağlan (canlıya alındı 16.09.2026, `658424e`)",
    ),
    WatchItem(
        key="shadow_stop", status="done", started="14.09",
        title="Dar stop — gölge izleme (4H/1D/1H)",
        trigger="28.09.2026 değerlendirmesi",
        measure="python scripts/stop_width_stat.py · scripts/c1_stop_stat.py · /setup-journal → shadow",
        md="Dar stop — gölge izleme, 4H/1D/1H (14.09.2026 — kapandı 21.09)",
        result="SL purge ucunda KALIYOR (21.09): 162 çözülmüş kazananla eğri monoton düşüyor "
               "(k=1 -0.472 → k=0.75 -0.567) ve tepe k=1.00, ızgaranın kenarında — kazanç varsa daha "
               "GENİŞ stop tarafında. 18.09'un 'tepe 0.75-0.85' okuması (21 kazanan) yanlışlandı. "
               "Gölge kollarında 4H B_0.5'in geçmesi daraltmadan değil eklenen low_rr setuplarından "
               "geliyor; işlem başına R A'nın altında (+0.141 vs +0.168).",
        progress_fn=_p_shadow,
    ),
    WatchItem(
        key="journal_live", status="done", started="14.09", onem=3,
        title="Setup Journal canlıda",
        trigger="2 hafta sonra kapı × sonuç özeti (özellikle “1D bias opposite” satırı)",
        result="ÇALIŞIYOR (28.09): günde ~260–470 satır, flush hatası yalnız 14.09/18.09'da birkaç dakikalık iki "
               "pakette; bias kapısının elediği 452 çözülmüş setup %23 win / −0.48 R (taban %29 / −0.36). "
               "Yan bulgu: Pazartesi evren geçişinde WS aboneliği gitmiyordu (FX 28.09 03:18'den beri kör) — düzeltildi.",
        measure="/setup-journal",
        md="Setup Journal canlıda — izlenecek (commit `c2c9fc2`, restart bekliyor)",
        progress_fn=_p_journal_live,
    ),
    WatchItem(
        key="partial_live", status="done", started="11.09",
        title="Kısmi kâr + bias düzeltmesi canlıda",
        trigger="Kısmi kâr alan işlemlerin dağılımı; bias düzeltmesinin yönlü bias oranına etkisi",
        measure="scripts/partial_vs_be.py · scripts/bias_stat.py (devredildi)",
        md="Kısmi kâr + bias düzeltmesi canlıda (11.09) — izlenecek",
        result="Devredildi (19.09): iki yarısı da sayaçlı maddelere geçti — kısmi kâr "
               "“Kısmi kâr vs BE-only” (17.09), bias “1D bias tahmin karnesi” (18.09).",
    ),
    WatchItem(
        key="be_1d_1h", status="done", started="09.09", onem=1,
        result="(28.09) SORU KALMADI: 1D'de BE + kısmi kâr 11.09'dan beri açık, yeni 1W-4H'te de açık; "
               "1H-5M 28.09'da kapatıldı. 1H geri açılırsa madde yeniden açılır.",
        title="1D/1H'te BE açılsın mı?",
        trigger="4H'te BE @ TP %50 yeterli veri biriktirince karar ver",
        measure="BE tetiklenen işlemlerin kaçı 0R'de kapandı, kaçı TP'ye yürüdü",
        md="6. 1D/1H'te BE açılsın mı?",
        progress_fn=_p_be_1d_1h,
    ),
    WatchItem(
        key="c2_reclaim", status="done", started="09.09",
        result="Ceza −2 → −4 (19.09): zayıf geri dönüş (<%25) win %7.8 / −0.800R, güçlü %37.5 / "
               "−0.152R; eşiğin altına düşen 31 setup 4 win / 20 loss = −15.3R.",
        title="C2 geri dönüş cezası (−2)",
        trigger="Birkaç düzine işlem sonra: ceza alan setuplar gerçekten daha mı kötü",
        measure="python scripts/score_parts_stat.py → “Zayıf C2 geri dönüşü”",
        md="5. C2 geri dönüş cezası (−2) — INJ'yi engellemiyor, dikkat",
        progress_fn=_p_c2_penalty,
    ),
    WatchItem(
        key="trail_90", status="done", started="09.09", onem=1,
        result="(28.09) SORU KALMADI: trail yalnız 1H-5M'de açıktı, strateji 28.09'da kapatıldı; 4H/1D/1W'de "
               "trail kapalı. 1H geri açılırsa madde yeniden açılır.",
        title="Trail arm eşiği %90 — kazananları kesiyor mu?",
        trigger="Trail çıkışlarının ortalama R'si < 1R olursa",
        measure="python scripts/logstat.py → çıkış türü dağılımı",
        md="4. Trail arm eşiği %90 — kazananları kesiyor muydu?",
        progress_fn=_p_trail,
    ),
    WatchItem(
        key="cluster", status="done", started="09.09",
        result="Limit KORUYUCU (19.09): elediği 49 sonuçlanmış setup 12 win / 37 loss = "
               "−18.1R. Gevşetme adayı değil; ayarlar aynen kalıyor. Denetim 01.10: −18R'nin çoğu RR < 2 setuplardan; "
               "RR ≥ 2'de elenenler −0.02 R/işlem (n 36) → 'koruyucu' desteklenmiyor, yeniden izleniyor.",
        title="Kripto küme limiti",
        trigger="Küme limiti kazananları eliyorsa (kapı hunisinde gevşetme adayı)",
        measure="/setup-journal → best_stage = cluster_limit",
        md="3. Kripto küme limiti",
        progress_fn=_p_cluster,
    ),
    WatchItem(
        key="cluster_recheck", status="open", started="01.10", onem=1,
        title="Küme limiti — işlem açılabilir setuplarda kazanan mı eliyor?",
        trigger="60 çözülmüş elenen RR ≥ 2 setup: R/işlem ≥ +0.15 ve %90 alt sınır > 0 ise gevşetme önerisi; "
                "aksi halde limit risk kontrolü olarak kalır",
        measure="/setup-journal → best_stage = cluster_limit, rr ≥ 2 · düz TP/SL",
        md="Küme limiti — işlem açılabilir setuplarda kazanan mı eliyor? (başladı 01.10.2026, ⏰ tetik: 60 çözülmüş setup)",
        progress_fn=_p_cluster_recheck,
    ),
    WatchItem(
        key="daily_bias", status="done", started="09.09",
        title="1D bias — bayatlık düzeltmesi",
        trigger="structure ↔ ict ayrışma oranı kalıcı olarak yükselirse",
        measure="python scripts/bias_stat.py (devredildi)",
        md="2. 1D BIAS — bayatlık düzeltmesi",
        result="Devredildi (19.09): ölçümü /logs'un satır sayan metriğine dayanıyordu; "
               "aynı soruyu 1D bias karnesi (bias_journal) 2700+ satırla doğru ölçüyor.",
    ),
    WatchItem(
        key="fixes_general", status="done", started="08.09",
        title="Genel: 08–09.09 motor düzeltmelerinin etkisi",
        trigger="Üretilen waiting sinyal sayısı/kalitesi ve SKIPPED dağılımının değişimi",
        measure="/setup-journal · “Kapı hunisi” maddesi (devredildi)",
        md="1. Genel: 2026-09-08/09 düzeltmelerinin etkisi",
        result="Devredildi (19.09): “SKIPPED dağılımı” sorusunu Setup Journal kapı bazında "
               "ölçüyor; toplu cevabı karara bağlanmış “Kapı hunisi” maddesi verdi (16.09).",
    ),
    WatchItem(
        key="crt60_off", status="open", started="10.09", onem=3,
        title="CRT %60 invalidation KAPATILDI",
        trigger="Breach sonrası fill'le açılan işlemlerin sonucu kötüyse kuralı geri açmayı tartış",
        measure="python scripts/logstat.py → BACKFILL etkisi",
        md="0c. CRT %60 invalidation KAPATILDI (10.09)",
    ),
    WatchItem(
        key="session_align", status="done", started="09.09",
        title="NY-hizalı HTF + seans filtresi + Cuma kapanışı",
        trigger="İlk 20 kapalı kripto-dışı işlem, ya da aynı yönde 3+ korele FX sinyali",
        measure="/signals?market=fx · week_close çıkışlarının R dağılımı",
        md="0. NY-hizalı HTF + seans filtresi + Cuma kapanışı (09.09)",
        result="(21.10) ÇALIŞIYOR, değişiklik yok (kullanıcı onayı): 21 kapalı kripto-dışı işlem +3.7R (+0.18/işlem; kripto 62 "
               "işlem +0.01); Cuma kapanışıyla kapanan işlem 0 (ölçülemedi); aynı yön 3+ korele FX hiç olmadı (en çok 2), "
               "corr_open 39 setup eledi. Pazartesi gap'i 'Hafta boşluğu kapısı'nda.",
        progress_fn=_p_noncrypto,
    ),

    # ---------------- karara baglananlar ----------------
    WatchItem(
        key="gate_funnel", status="done", started="16.09",
        title="Kapı hunisi: eleme kazandırıyor mu?",
        trigger="—", md="Kapı hunisi: eleme kazandırıyor mu? (ilk ölçüm 16.09.2026)",
        result="Eleme ~80R kayıp önlemiş, ~4R kazanç kaçırmış. Gevşetme adayı yalnız low_rr ve cluster_limit.",
    ),
    WatchItem(
        key="limit_vs_market", status="done", started="14.09",
        title="1D limit vs market giriş",
        trigger="—", md="1D limit vs market giriş (14.09) — limit korunur",
        result="Limit + purge SL korunur; market girişi kaybı 124 → 183'e çıkarıyordu. Dar stop fikri kapandı.",
    ),
    WatchItem(
        key="funnel_1d", status="done", started="14.09",
        title="1D sinyal hunisi — geçen hafta neden sinyal az?",
        trigger="—", md="1D sinyal hunisi (13–14.09) — geçen hafta neden sinyal az? (26 sembol, FX dahil)",
        result="Sinyal azlığı kural değişiklikleri + kesintilerle açıklandı; huni taban değerlerinin içinde.",
    ),
    WatchItem(
        key="partial_replay_4h", status="done", started="11.09",
        title="Kısmi kâr — 4H replay",
        trigger="Kurallar değişirse replay yeniden koşulmalı",
        md="Kısmi kâr — 4H replay (11.09): getiri gerekçesi yok; eski 4H replay fazla iyimserdi",
        result="Getiri gerekçesi yok (+0.06R/işlem, sıfırı içeriyor). Uygulanırsa gerekçe psikolojik/risk tercihi.",
    ),
    WatchItem(
        key="partial_replay_1d", status="done", started="11.09",
        title="Kısmi kâr — 1D replay",
        trigger="1D kuralları değişirse replay yeniden koşulmalı",
        md="Kısmi kâr — 1D replay (11.09): yarı kâr + BE getiriyi değiştirmiyor",
        result="Yarı kâr + BE getiriyi değiştirmiyor; kazanma oranı yükseliyor, büyük kazançlar yarılanıyor.",
    ),
    WatchItem(
        key="long_filter", status="done", started="10.09",
        title="4H kripto LONG kayıp serisi — filtre araması",
        trigger="—", md="0e. 4H kripto LONG kayıp serisi — replay + örneklem dışı doğrulama: FİLTRE YOK (10.09)",
        result="Hiçbir filtre eklenmedi: in-sample güçlü görünen kesitler örneklem dışında tutmadı.",
    ),
    WatchItem(
        key="ghost", status="done", started="10.09",
        title="Hayalet işlemler — tespit ve düzeltme",
        trigger="Log'da aynı saniyede FILLED + CLOSED görülürse",
        md="0d. Hayalet işlemler — tespit ve düzeltme (10.09)",
        result="Üç koruma eklendi (kayıt zamanı / C2 zamanı / STALE FILL); izole DB'de 11/11 test geçti.",
    ),
    WatchItem(
        key="ifvg_fallback", status="done", started="09.09",
        title="IFVG fallback — ölçüldü, YAPILMADI",
        trigger="CISD RR'si ≥ 2.0 olan bir setup IFVG yüzünden kaçarsa maddeyi geri aç",
        md="0b. IFVG fallback — ölçüldü, YAPILMADI (09.09)",
        result="Yapılmadı; gerekçe ölçüme dayanıyor. IFVG-RR koruması sorunun çoğunu zaten çözdü.",
    ),
]


# ---------------------------------------------------------------------------------------
# IZLEME.md'den bolum metni + kucuk markdown render'i (yeni bagimlilik eklemeden)
# ---------------------------------------------------------------------------------------

_md_cache: dict = {"mtime": None, "sections": {}}


def _load_sections() -> dict[str, str]:
    """IZLEME.md'yi '## ' basliklarindan bol; baslik metni -> govde. mtime ile onbellekli."""
    try:
        mtime = IZLEME_PATH.stat().st_mtime
    except OSError:
        return {}
    if _md_cache["mtime"] == mtime:
        return _md_cache["sections"]
    try:
        text = IZLEME_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    sections: dict[str, str] = {}
    head, body = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if head is not None:
                sections[head] = "\n".join(body).strip()
            head, body = line[3:].strip(), []
        elif head is not None:
            body.append(line)
    if head is not None:
        sections[head] = "\n".join(body).strip()
    _md_cache.update(mtime=mtime, sections=sections)
    return sections


_INLINE = (
    (re.compile(r"`([^`]+)`"), r'<code class="px-1 py-0.5 rounded bg-dark-900 text-accent-blue text-[11px]">\1</code>'),
    # Icerikte tek `*` gecebilir (ör. `tmp_session_anchor_probe*.py`): [^*]+ olsaydi o kalin
    # hic donusmez, sayfada ham ** kalirdi. Tembel eslesme ayri kalinlari birlestirmez.
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong class='text-white'>\1</strong>"),
    (re.compile(r"~~([^~]+)~~"), r"<del class='text-gray-600'>\1</del>"),
    (re.compile(r"(?<![*\w])\*([^*]+)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r"<span class='text-gray-400'>\1</span>"),  # yerel dosya linki: duz metin
)


def _inline(s: str) -> str:
    s = html.escape(s)
    for pat, rep in _INLINE:
        s = pat.sub(rep, s)
    return s


_BULLET_RE = re.compile(r"^\s*(-|\*|\d+\.)\s+")


def _starts_block(ln: str) -> bool:
    """Satir YENI bir blok mu basliyor? Degilse onceki paragrafin/maddenin devamidir.

    IZLEME.md satirlari ~110 karakterde sarilir ve `**kalin**` bir satir sonunu asabilir
    (dosyada 58 yerde oyle). Markdown bunlari tek paragraf sayar; her satiri ayri <p>
    yapmak kalin isaretini ORTADAN bolup sayfada ham `**` gosteriyordu.
    """
    s = ln.strip()
    if not s or s.startswith("```") or s in ("---", "***"):
        return True
    if ln.startswith("### ") or ln.startswith("#### "):
        return True
    return bool(_BULLET_RE.match(ln)) or s.startswith("|")


def render_section(md_heading: str) -> Markup:
    """Bir IZLEME.md bolumunu HTML'e cevir. Desteklenen: ###/####, liste, tablo, kod blogu, hr."""
    body = _load_sections().get(md_heading)
    if body is None:
        return Markup('<p class="text-gray-500 text-xs">Detay bulunamadı — '
                      'IZLEME.md\'deki başlık değişmiş olabilir.</p>')
    return Markup(_render_md(body))


def _render_md(body: str) -> str:
    """render_section'in govdesi, ayri fonksiyon: dogrulama betigi ham markdown verebilsin."""
    out: list[str] = []
    lines = body.splitlines()
    i, in_code = 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            if not in_code:
                out.append('<pre class="my-2 p-2 rounded bg-dark-900 overflow-x-auto '
                           'text-[11px] text-gray-300"><code>')
            else:
                out.append("</code></pre>")
            in_code = not in_code
            i += 1
            continue
        if in_code:
            out.append(html.escape(ln) + "\n")
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("#### "):
            out.append(f'<h5 class="mt-3 mb-1 text-xs font-semibold text-gray-300">{_inline(ln[5:])}</h5>')
        elif ln.startswith("### "):
            out.append(f'<h4 class="mt-3 mb-1 text-sm font-semibold text-white">{_inline(ln[4:])}</h4>')
        elif ln.strip() in ("---", "***"):
            out.append('<hr class="my-3 border-dark-600">')
        elif ln.lstrip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            rows = []
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            header, data = rows[0], rows[2:]
            out.append('<div class="my-2 overflow-x-auto"><table class="w-full text-[11px]">'
                       '<thead><tr class="text-gray-500 border-b border-dark-600">')
            out.append("".join(f'<th class="py-1 pr-3 text-left font-medium">{_inline(c)}</th>' for c in header))
            out.append("</tr></thead><tbody>")
            for r in data:
                out.append('<tr class="border-b border-dark-700/50">'
                           + "".join(f'<td class="py-1 pr-3 align-top">{_inline(c)}</td>' for c in r)
                           + "</tr>")
            out.append("</tbody></table></div>")
            i = j
            continue
        elif _BULLET_RE.match(ln):
            items = []
            j = i
            while j < len(lines):
                if _BULLET_RE.match(lines[j]):
                    items.append(_BULLET_RE.sub("", lines[j]))
                elif items and not _starts_block(lines[j]):
                    items[-1] += " " + lines[j].strip()   # sarilmis satir: ayni maddenin devami
                else:
                    break
                j += 1
            out.append('<ul class="my-1.5 ml-4 space-y-0.5 list-disc text-gray-400">'
                       + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            i = j
            continue
        else:
            para = [ln]
            j = i + 1
            while j < len(lines) and not _starts_block(lines[j]):
                para.append(lines[j].strip())   # sarilmis satir: ayni paragrafin devami
                j += 1
            out.append('<p class="my-1.5 text-gray-400 leading-relaxed">'
                       f'{_inline(" ".join(para))}</p>')
            i = j
            continue
        i += 1
    if in_code:
        out.append("</code></pre>")
    return "".join(out)


# ---------------------------------------------------------------------------------------
# Sayfa verisi
# ---------------------------------------------------------------------------------------

_DUE_HORIZON_DAYS = 30      # tarihli maddede "yakinlik" bu pencereye gore olculur
PASSIVE = -1.0              # sayaci olmayan izleme: siralanamaz, ayri kumede durur


def closeness(prog: Optional[Progress]) -> float:
    """Tetige yakinlik, 0..1 (1 = dolmus) -- sayfayi ONCELIGE gore sirlamak icin.

    Birden fazla kisit varsa **en geride olani** belirler, cunku `Progress.ready` hepsini
    birden ister (iki cubuk + tarih ise hepsi dolmali). Sayaci olmayan madde `PASSIVE` doner:
    tetigi bir GOZLEM oldugu icin (ör. "bekci gereksiz tetiklenirse") ilerleme diye bir sey yok,
    listede yukari cikmasi da anlamsiz.
    """
    if prog is None:
        return PASSIVE
    vals: list[float] = []
    for b in prog.bars:
        vals.append(min(1.0, b.current / b.target) if b.target > 0 else 1.0)
    if prog.due is not None:
        left = prog.days_left if prog.days_left is not None else 0
        vals.append(1.0 if left <= 0 else max(0.0, 1.0 - min(left, _DUE_HORIZON_DAYS) / _DUE_HORIZON_DAYS))
    return min(vals) if vals else PASSIVE


def _decided_key(result: str) -> tuple[int, int]:
    """Sonuc metnindeki ilk 'gg.aa' (karar tarihi, ör. "KAPANDI (28.09)") -> (ay, gun); yoksa en sona."""
    m = re.search(r"\b(\d{2})\.(\d{2})\b", result or "")
    return (int(m.group(2)), int(m.group(1))) if m else (0, 0)


def _started_key(started: str) -> tuple[int, int]:
    """"18.09" -> (9, 18); esitlikte EN ESKI izleme one gelsin diye."""
    m = re.match(r"(\d{1,2})\.(\d{1,2})", started or "")
    return (int(m.group(2)), int(m.group(1))) if m else (99, 99)


# Ayni isi strateji bazinda yapan maddeler (01.10, kullanici "alt alta yaz"): baslik "<aile> — 4H|1D|1W" kalibinda.
_FAMILY_RE = re.compile(r"^(.*?) — (4H|1D|1W|1H)$")
_TF_ORDER = {"4H": 0, "1D": 1, "1W": 2, "1H": 3}


def _family(item: WatchItem) -> tuple[str, int]:
    """(aile adi, strateji sirasi); kaliba uymayan madde kendi basina bir aile."""
    m = _FAMILY_RE.match(item.title or "")
    return (m.group(1), _TF_ORDER[m.group(2)]) if m else (item.key, 0)


def _group_families(rows: list[dict]) -> list[dict]:
    """Siralanmis kumede aile uyelerini alt alta topla: aile, en iyi siradaki uyesinin yerinde durur, uyeler 4H/1D/1W.

    Kume siniri asilmaz: tetigi dolan uye tek basina "Aksiyon bekliyor"a cikar (o kume "tetik doldu" demek).
    """
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        fam = _family(r["item"])[0]
        if fam in seen:
            continue
        seen.add(fam)
        out.extend(sorted((x for x in rows if _family(x["item"])[0] == fam), key=lambda x: _family(x["item"])[1]))
    return out


async def build_watchlist(db: AsyncSession, show_done: bool = False) -> dict:
    """Dort kume: aksiyon bekleyen / acik (oncelige gore) / pasif / tamamlanan.

    Tamamlananlar 28.09'dan beri HER ZAMAN doner ve sayfanin en altinda KATLANMIS bir baslikta durur
    (kullanici istegi; 18.09'da sayfa karisiyor diye hic basilmiyordu). `show_done` artik yalniz o
    basligin acik gelip gelmeyecegini soyler (`?done=1`). En yeni karar ustte (`_decided_key`).
    """
    action, open_items, passive, done_items = [], [], [], []
    for item in ITEMS:
        prog: Optional[Progress] = None
        if item.status == "open" and item.progress_fn is not None:
            try:
                prog = await item.progress_fn(db)
            except Exception:  # izleme sayfasi asla sayfayi cokertmesin
                prog = None
        row = {
            "item": item,
            "progress": prog,
            "detail": render_section(item.md),
            "status_label": STATUS_META[item.status][0],
            "closeness": closeness(prog),
            "onem_label": ONEM_META[item.onem][0],
            "onem_class": ONEM_META[item.onem][1],
        }
        if item.status == "done":
            done_items.append(row)
        elif prog is not None and prog.ready:
            action.append(row)
        elif row["closeness"] == PASSIVE:
            passive.append(row)
        else:
            open_items.append(row)

    # Siralama (21.09, kullanici istegi "onemsizler alt siralarda olsun"): BIRINCIL anahtar
    # ONEM, ikincil tetige yakinlik, esitlikte en eski izleme. Eskiden yalniz yakinlik vardi ve
    # bir bildirim maddesi sayaci doldu diye kural maddesinin ustune cikabiliyordu.
    # Sabitlenen maddeler (29.09, kullanici "cok onemli, uste yaz"): hangi kumede olursa olsun sayfanin EN
    # USTUNDE ayri bolum ("Sabitlenenler"); diger kumelerden cikarilir. Tetigi dolmussa kartta yazilir.
    pinned = [r for r in action + open_items + passive if r["item"].pinned]
    action = [r for r in action if not r["item"].pinned]
    open_items = [r for r in open_items if not r["item"].pinned]
    passive = [r for r in passive if not r["item"].pinned]
    pinned.sort(key=lambda r: (r["item"].pin_order, r["item"].onem, _started_key(r["item"].started)))
    open_items.sort(key=lambda r: (not r["item"].pinned, r["item"].onem, -r["closeness"],
                                   _started_key(r["item"].started)))
    action.sort(key=lambda r: (not r["item"].pinned, r["item"].onem, _started_key(r["item"].started)))
    passive.sort(key=lambda r: (not r["item"].pinned, r["item"].onem, _started_key(r["item"].started)))
    pinned, action, open_items, passive = (_group_families(x) for x in (pinned, action, open_items, passive))

    # "Ne bekliyor" ozeti: ayni tarihi bekleyen maddeler IZLEME.md'de zaten "ayni oturumda oku"
    # diye bagli (kardes maddeler ayni golge verisini okuyor). 18 satirlik listede bunu gormek
    # imkansizdi; ust satirda kac maddenin hangi tarihi/esigi bekledigini yaziyoruz.
    by_due: dict[date, int] = {}
    sample_only = 0
    for r in open_items:
        due = r["progress"].due if r["progress"] else None
        if due is None:
            sample_only += 1
        else:
            by_due[due] = by_due.get(due, 0) + 1
    waiting = [{"due": d, "count": n, "days_left": (d - datetime.now(timezone.utc).date()).days}
               for d, n in sorted(by_due.items())]
    return {
        "waiting": waiting,
        "sample_only": sample_only,
        "pinned": pinned,
        "action": action,
        "open_items": open_items,
        "passive_items": passive,
        "done_items": sorted(done_items, key=lambda r: _decided_key(r["item"].result), reverse=True),
        "show_done": show_done,
        "md_ok": bool(_load_sections()),
        "counts": {"action": len(action), "open": len(open_items),
                   "passive": len(passive), "done": len(done_items)},
    }
