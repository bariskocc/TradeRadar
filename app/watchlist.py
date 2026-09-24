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
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BiasJournal, PotentialNotice, SetupJournal, Signal

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
    1: ("Yüksek", "text-accent-red"),
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

    @property
    def days_left(self) -> Optional[int]:
        if not self.due:
            return None
        return (self.due - datetime.now(timezone.utc).date()).days

    @property
    def ready(self) -> bool:
        """Tetik doldu mu: TUM cubuklar dolmali ve (varsa) tarih gelmis olmali."""
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
    onem: int = 2               # 1 yuksek / 2 orta / 3 dusuk -- ONEM_META, siralamanin BIRINCIL anahtari
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
    return Progress(bars=[Bar("kısmi kârlı kapalı işlem", n, 25)])


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
    return Progress(due=date(2026, 9, 28),
                    note=f"{n} setup gölge izlemede · sürekli k eğrisi için çözülmüş kazanan {egri}/60")


async def _p_retrace(db: AsyncSession) -> Progress:
    """Cozulmus KAZANAN setup sayisi (karar girdisi; esik 60)."""
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(_journal_col("retrace").is_not(None))
    ) or 0
    win = await db.scalar(
        select(func.count()).select_from(t)
        .where(_journal_col("retrace").like('%"tp_first":true%'))
    ) or 0
    return Progress(bars=[Bar("çözülmüş kazanan setup", win, 60)],
                    due=date(2026, 9, 28), note=f"{n} setup ölçekte")


async def _p_entry_models(db: AsyncSession) -> Progress:
    """Entry varyanti tasiyan setup sayisi (karar esigi varyant basina 60)."""
    t = _journal_table()
    n = await db.scalar(
        select(func.count()).select_from(t).where(_journal_col("entries").is_not(None))
    ) or 0
    return Progress(bars=[Bar("entry varyantli setup", n, 60)], due=date(2026, 9, 28))


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
    return Progress(due=date(2026, 10, 2), note=f"düzeltmeden sonra pencerede {n} setup")


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
                    due=date(2026, 10, 3), note=f"{n} satır karnede")


async def _p_bias_fidelity(db: AsyncSession) -> Progress:
    """Tarihli madde; sayi raporda (`scripts/bias_fidelity_stat.py`) -- karsilastirma karne ile
    Journal'i gun gun eslestiriyor, izleme sayfasinda her istekte yapmaya degmez."""
    return Progress(due=date(2026, 10, 3), note="24.09: gün dönümünde 4H'te %18 yanlıştı; düzeltildi, restart'ta geçerli")


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
        due=date(2026, 9, 28), note=f"{n} setup olcekte",
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
    return Progress(due=date(2026, 9, 28), note=note)


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


async def _p_missed_quality(db: AsyncSession) -> Progress:
    """Retest skor kapisinda elenip SONUCU BELLI olan setup (karar esigi 20) + C2 zamanlama dilimi.

    Dilim sayilari `note`'a yazilir, ayri CUBUK olarak DEGIL: `Progress.ready` butun cubuklarin
    dolmasini istiyor, dilim cubugu eklemek ana hatirlatmayi saklardi (CLAUDE.md uyarisi).
    """
    t = _journal_table()
    rows = (await db.execute(
        select(t.c.outcome, t.c.rr, t.c.strategy, t.c.purge_time, t.c.entry_touched_at)
        .where(t.c.best_stage == "missed_quality", t.c.outcome.in_(("win", "loss")))
    )).all()
    total = sum((float(r.rr or 0) if r.outcome == "win" else -1.0) for r in rows)
    note = f"{len(rows)} elenen setup çözüldü"
    if rows:
        w = sum(1 for r in rows if r.outcome == "win")
        note += f" · {w} TP · R/işlem {total / len(rows):+.2f}"
        span = {"4h": timedelta(hours=4), "1d": timedelta(days=1), "1h": timedelta(hours=1)}
        early = [r for r in rows if r.purge_time and r.entry_touched_at and span.get(r.strategy or "")
                 and r.entry_touched_at < r.purge_time + span[r.strategy]]
        late = [r for r in rows if r.purge_time and r.entry_touched_at and span.get(r.strategy or "")
                and r.entry_touched_at >= r.purge_time + span[r.strategy]]
        if early or late:
            ew = sum(1 for r in early if r.outcome == "win")
            lw = sum(1 for r in late if r.outcome == "win")
            note += (f" · retest C2 açıkken {ew}/{len(early)} TP,"
                     f" C2 kapalıyken {lw}/{len(late)} TP (H3 için her dilimde 10 gerek)")
    return Progress(bars=[Bar("çözülmüş elenen setup", len(rows), 20)], note=note)


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
    pre = ("sweep_small", "range_atr", "c1_stale", "c2_breakout", "c2_wrong_color", "not_selected")
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


async def _p_score_parts(db: AsyncSession) -> Progress:
    n = await db.scalar(
        select(func.count()).select_from(_journal_table())
        .where(_journal_col("score_parts").is_not(None))
    ) or 0
    return Progress(due=date(2026, 9, 28), note=f"{n} setup'ta skor kırılımı kayıtlı")


async def _p_journal_live(db: AsyncSession) -> Progress:
    n = await db.scalar(select(func.count()).select_from(_journal_table())) or 0
    return Progress(due=date(2026, 9, 28), note=f"{n} Journal satırı")


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
    return Progress(due=date(2026, 10, 2), note=f"değişiklikten sonra {n} sinyal (üst sınır)")


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
    return Progress(due=date(2026, 10, 2), note=f"yeni ızgarada {n} sonuçlanmış CME setup'ı")


ITEMS: list[WatchItem] = [
    WatchItem(
        key="session_anchor", status="open", started="18.09",
        title="CME seans anchor'ı — endeks/metal/petrol 18:00 NY",
        trigger="02.10.2026: bu 6 sembolde setup sayısı/outcome bozuldu mu, c2_breakout payı düştü mü",
        measure="python tmp/tmp_session_anchor_check.py · /setup-journal (CME sembolleri)",
        md="CME seans anchor'ı — endeks/metal/petrol 18:00 NY'ye geçti (18.09.2026, restart bekliyor)",
        progress_fn=_p_session_anchor,
    ),
    WatchItem(
        key="retrace", status="open", started="18.09", onem=1,
        title="Geri çekilme derinliği — limit emri nereden dolar",
        trigger="60 çözülmüş kazanan setup (ilk okuma 28.09.2026)",
        measure="python scripts/retrace_stat.py",
        md="Geri çekilme derinliği — limit emri hangi seviyeden dolar? (başladı 18.09.2026, ⏰ ilk okuma 28.09.2026)",
        progress_fn=_p_retrace,
    ),
    WatchItem(
        key="entry_models", status="done", started="18.09",
        title="Entry modeli karşılaştırması",
        trigger="Varyant başına 60 karar verilebilir setup (ilk okuma 28.09.2026)",
        measure="python scripts/entry_model_stat.py",
        md="Entry modeli karşılaştırması (başladı 18.09.2026, ⏰ ilk okuma 28.09.2026)",
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
        md="BPR entry modeli (canlıya alındı 21.09.2026, ⏰ tetik: 40 BPR'li çözülmüş setup)",
        result="BPR KALIYOR, skor ve skor-7 kapısı DEĞİŞMİYOR (24.09): aynı 54 setupta bpr_near "
               "+0.015 vs ifvg_near -0.031 R/setup, fark +0.046 -- bandın içinde. C2 kapalı dilimde "
               "de aynı yön (+0.094, n=32).",
        progress_fn=_p_bpr,
    ),
    WatchItem(
        key="bpr_leg", status="open", started="22.09", onem=1,
        title="BPR bacağı — kesişim yerine bacağın kendisinden mi girmeli?",
        trigger="40 bacak ölçümlü setup: bfvg_near, bpr_near'ı 0,15 R/setup geçiyor mu",
        measure="python scripts/entry_model_stat.py (bölüm 2)",
        md="BPR bacağı — kesişim yerine bacağın kendisinden mi girmeli? (ölçüm başladı 22.09.2026)",
        progress_fn=_p_bpr_leg,
    ),
    WatchItem(
        key="cfvg", status="done", started="22.09", onem=1,
        title="CISD kırılım FVG'si — onaydan sonra bırakılan boşluktan giriş",
        trigger="60 eşlenmiş setup: cfvg_* varyantlarından biri chosen'ı 0,15 R/setup geçiyor mu",
        measure="python scripts/entry_model_stat.py (bölüm 3)",
        md="CISD kırılım FVG'si — onaydan sonra bırakılan boşluktan giriş (ölçüm başladı 22.09.2026)",
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
        key="pd_midnight", status="open", started="18.09",
        title="Gece yarısı PDH/PDL kayması düzeltildi",
        trigger="02.10.2026: pencere içi pd_major oranı %44.7'ye yakınsadı mı (yoksa %30 altı = tutmadı)",
        measure="setup_journal.score_parts — first_seen 00:00-00:30 UTC penceresi",
        md="Gece yarısı PDH/PDL kayması düzeltildi (canlıya alındı 18.09.2026)",
        progress_fn=_p_pd_midnight,
    ),
    WatchItem(
        key="bias_journal", status="open", started="18.09", onem=1,
        title="1D bias tahmin karnesi",
        # 21.09 ara okumasi: H1 battı (0/3 ufuk) ama hard filtre YERINDE KALDI -- karne 1-5 gun
        # olcuyor, setuplar medyan 1.2 saat yasiyor. Maddenin kalan isi iki alt soru.
        trigger="03.10: htf +2 ödülü hak edilmiş mi (hizalı −0.461 vs NEUTRAL −0.406) · "
                "STRUCTURE_STALE_DAYS düşsün mü (bayat dal %52.0 vs taze AND %46.1)",
        measure="python scripts/bias_stat.py",
        md="1D bias tahmin karnesi (başladı 18.09.2026, ⏰ ilk okuma 28.09.2026)",
        progress_fn=_p_bias_journal,
    ),
    WatchItem(
        key="bias_fidelity", status="open", started="24.09", onem=1,
        title="1D bias hesap doğruluğu — motor doğru günün bias'ını mı kullanıyor?",
        trigger="03.10: restart sonrası gün dönümünde yanlış bias 0 mı (düzeltme 24.09)",
        measure="python scripts/bias_fidelity_stat.py",
        md="1D bias hesap doğruluğu (başladı 24.09.2026, ⏰ 03.10.2026)",
        progress_fn=_p_bias_fidelity,
    ),
    WatchItem(
        key="prefill", status="open", started="18.09", onem=3,
        title="Dolum öncesi koşu",
        trigger="0-25% ve >=50% kovalarının her birinde 100 dolan setup (ilk okuma 28.09.2026)",
        measure="python scripts/prefill_stat.py",
        md="Dolum öncesi koşu (18.09.2026, restart bekliyor)",
        progress_fn=_p_prefill,
    ),
    WatchItem(
        key="gated_potential", status="open", started="18.09", onem=3,
        title="Elenen 1D setup bildirimi",
        trigger="02.10.2026: kapsam daraltıldıktan sonra günlük mesaj ~4'e indi mi (8'i aşarsa RR tabanı 1.8)",
        measure="potential_notices — gate dolu satırlar",
        md="Elenen 1D setup bildirimi (18.09.2026, restart bekliyor)",
        progress_fn=_p_gated_potential,
    ),
    WatchItem(
        key="dead_candidate", status="open", started="18.09",
        title="Ölü CRT adayı slotu tutmasın",
        trigger="02.10.2026: yeni yolla doğan sinyallerin sonucu + past_sl radar durumu seyrekleşti mi",
        measure="setup_journal — best_stage='waiting' satırları, kardeş not_selected ile",
        md="Ölü CRT adayı slotu tutmasın (18.09.2026, restart bekliyor)",
        progress_fn=_p_dead_candidate,
    ),
    WatchItem(
        key="direction", status="open", started="17.09",
        title="LONG/SHORT ayrışması + 16.09 FOMC etkisi",
        trigger="Her iki yönde de 30 kapalı işlem birikmesi",
        measure="python scripts/direction_stat.py",
        md="LONG/SHORT ayrışması + 16.09 FOMC etkisi (ilk ölçüm 17.09.2026, ⏰ tetik: her yönde 30 işlem)",
        progress_fn=_p_direction,
    ),
    WatchItem(
        key="partial_vs_be", status="open", started="17.09", onem=1,
        title="Kısmi kâr vs BE-only",
        trigger="25 kısmi kârlı kapalı işlem",
        measure="python scripts/partial_vs_be.py",
        md="Kısmi kâr vs BE-only — canlı takip (başladı 17.09.2026, ⏰ tetik: 25 işlem)",
        progress_fn=_p_partial,
    ),
    WatchItem(
        key="tight_stop_gate", status="open", started="18.09", onem=1,
        title="Dar stop kapısı — elenen setup TP'ye gidiyor mu?",
        trigger="20 çözülmüş elenen setup (kapı haftada ~1–2 eliyor)",
        measure="python scripts/tight_stop_stat.py",
        md="Dar stop kapısı — elenen setup TP'ye gidiyor mu? (başladı 18.09.2026, ⏰ tetik: 20 çözülmüş setup)",
        progress_fn=_p_tight_stop_gate,
    ),
    WatchItem(
        key="missed_quality", status="open", started="21.09", onem=1,
        title="Retest skor kapısı — kazanan mı eliyor?",
        trigger="20 çözülmüş elenen setup (H3 için ayrıca her zamanlama diliminde 10)",
        measure="python scripts/missed_quality_stat.py",
        md="Retest skor kapısı — kazanan mı eliyor? (başladı 21.09.2026, ⏰ tetik: 20 çözülmüş setup)",
        progress_fn=_p_missed_quality,
    ),
    WatchItem(
        key="c1_stop", status="done", started="17.09",
        title="C1 ucu stop — gölge izleme",
        trigger="28.09.2026 değerlendirmesi (kardeş madde “Dar stop” ile aynı oturumda)",
        measure="python scripts/c1_stop_stat.py",
        md="C1 ucu stop — gölge izleme (başladı 17.09.2026, ⏰ değerlendirme 28.09.2026)",
        result="C1 ucu stop KULLANILMIYOR (21.09): H1 her dilimde battı (havuz -0.096, 4H -0.082, "
               "1H -0.160 R/işlem). Bugünkü SL'le TP olan 136 setupun 45'i C1'le stop olurdu, C1'in "
               "kurtardığı sıfır. Fikri açık tutan skor>=7 dilimi n=102'de -0.009'a oturdu; kural "
               "setupların %52'sinde zaten uygulanamıyor.",
        progress_fn=_p_c1_stop,
    ),
    WatchItem(
        key="deleted_gate", status="open", started="19.09", onem=1,
        title="Silinen waiting setup — kapı kazanan sinyal mi eliyor?",
        trigger="20 çözülmüş silinen setup (~7/gün siliniyor; skor kapıları ~1,3/gün)",
        measure="python scripts/deleted_gate_stat.py",
        md="Silinen waiting setup — kapı kazanan sinyal mi eliyor? (başladı 19.09.2026, ⏰ tetik: 20 çözülmüş setup)",
        progress_fn=_p_deleted_gate,
    ),
    WatchItem(
        key="bias_1h", status="open", started="20.09", onem=1,
        title="1D bias 1H'te iki kez mi eliyor?",
        trigger="25 çözülmüş bias elemesi (1H, ~2/gün) — “Skor kalemleri” ile aynı oturumda",
        measure="python scripts/bias_1h_stat.py",
        md="1D bias 1H'te iki kez mi eliyor? (başladı 20.09.2026, ⏰ tetik: 25 çözülmüş elenen setup)",
        progress_fn=_p_bias_1h,
    ),
    WatchItem(
        key="score_parts", status="open", started="16.09", onem=1,
        title="Skor kalemleri — kırılım izleme",
        trigger="28.09.2026 tarihinde ilk okuma (skor bandı 0–6'dan)",
        measure="python scripts/score_parts_stat.py",
        md="Skor kalemleri — kırılım izleme (başladı 16.09.2026, ⏰ ilk okuma 28.09.2026)",
        progress_fn=_p_score_parts,
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
        md="Dar stop — gölge izleme, 4H/1D/1H (başladı 14.09.2026, ⏰ değerlendirme 28.09.2026)",
        result="SL purge ucunda KALIYOR (21.09): 162 çözülmüş kazananla eğri monoton düşüyor "
               "(k=1 -0.472 → k=0.75 -0.567) ve tepe k=1.00, ızgaranın kenarında — kazanç varsa daha "
               "GENİŞ stop tarafında. 18.09'un 'tepe 0.75-0.85' okuması (21 kazanan) yanlışlandı. "
               "Gölge kollarında 4H B_0.5'in geçmesi daraltmadan değil eklenen low_rr setuplarından "
               "geliyor; işlem başına R A'nın altında (+0.141 vs +0.168).",
        progress_fn=_p_shadow,
    ),
    WatchItem(
        key="journal_live", status="open", started="14.09", onem=3,
        title="Setup Journal canlıda",
        trigger="2 hafta sonra kapı × sonuç özeti (özellikle “1D bias opposite” satırı)",
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
        key="be_1d_1h", status="open", started="09.09",
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
        key="trail_90", status="open", started="09.09", onem=3,
        title="Trail arm eşiği %90 — kazananları kesiyor mu?",
        trigger="Trail çıkışlarının ortalama R'si < 1R olursa",
        measure="/logs → Engine Report → çıkış türü dağılımı",
        md="4. Trail arm eşiği %90 — kazananları kesiyor muydu?",
        progress_fn=_p_trail,
    ),
    WatchItem(
        key="cluster", status="done", started="09.09",
        result="Limit KORUYUCU (19.09): elediği 49 sonuçlanmış setup 12 win / 37 loss = "
               "−18.1R. Gevşetme adayı değil; ayarlar aynen kalıyor.",
        title="Kripto küme limiti",
        trigger="Küme limiti kazananları eliyorsa (kapı hunisinde gevşetme adayı)",
        measure="/setup-journal → best_stage = cluster_limit",
        md="3. Kripto küme limiti",
        progress_fn=_p_cluster,
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
        measure="/logs → Engine Report → BACKFILL etkisi",
        md="0c. CRT %60 invalidation KAPATILDI (10.09)",
    ),
    WatchItem(
        key="session_align", status="open", started="09.09",
        title="NY-hizalı HTF + seans filtresi + Cuma kapanışı",
        trigger="İlk 20 kapalı kripto-dışı işlem, ya da aynı yönde 3+ korele FX sinyali",
        measure="/signals?market=fx · week_close çıkışlarının R dağılımı",
        md="0. NY-hizalı HTF + seans filtresi + Cuma kapanışı (09.09)",
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


def _started_key(started: str) -> tuple[int, int]:
    """"18.09" -> (9, 18); esitlikte EN ESKI izleme one gelsin diye."""
    m = re.match(r"(\d{1,2})\.(\d{1,2})", started or "")
    return (int(m.group(2)), int(m.group(1))) if m else (99, 99)


async def build_watchlist(db: AsyncSession, show_done: bool = False) -> dict:
    """Dort kume: aksiyon bekleyen / acik (oncelige gore) / pasif / karara baglanmis.

    Karara baglananlar sayfada VARSAYILAN OLARAK gosterilmez (kullanici istegi 18.09: sayfa
    karisiyordu) -- `?done=1` ile geri gelir, kayit kaybolmaz. Pasif kume de katlanmis durur.
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
    open_items.sort(key=lambda r: (r["item"].onem, -r["closeness"],
                                   _started_key(r["item"].started)))
    action.sort(key=lambda r: (r["item"].onem, _started_key(r["item"].started)))
    passive.sort(key=lambda r: (r["item"].onem, _started_key(r["item"].started)))

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
        "action": action,
        "open_items": open_items,
        "passive_items": passive,
        "done_items": done_items if show_done else [],
        "show_done": show_done,
        "md_ok": bool(_load_sections()),
        "counts": {"action": len(action), "open": len(open_items),
                   "passive": len(passive), "done": len(done_items)},
    }
