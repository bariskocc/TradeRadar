"""1D bias NEUTRAL'i azaltan kural secenekleri -- karsi-olgu olcumu (28.10, kullanici: "cok fazla NEUTRAL olmamali,
bias BULLISH ya da BEARISH belirleyebilmeli"). Motor DEGISMEZ; saf hesap, DB okumaz.

Bugunku kural (`crt_engine.compute_daily_bias`): yapisal kirilim TAZE (<= STRUCTURE_STALE_DAYS) ise structure VE ict ayni
yon olmali, aksi halde NEUTRAL; bayat ise karar ict'nin. NEUTRAL iki yerden gelir: ict NEUTRAL, ya da taze structure ile
ict celisiyor. 28.10 olcumu: 1D evreninde gunlerin ~%50'si NEUTRAL.

Hipotez (VARIANT, 28.10, 04.10-28.10 4H Journal'dan cikti): celismede TAZE structure'in yonu alinir (bayatsa yine ict).
Ayni veride 4H'te 39 setup ekliyor (23 TP / 15 SL, dolum +0.80 R/islem), 13 dusuruyor (3 TP / 10 SL). Hipotezi uretmis
veri karar verisi DEGIL -- karar `DECISION_SINCE` sonrasi donan seviyelerden (IZLEME.md "1D bias NEUTRAL'ı azaltmak —
çatışmada taze yapısal yön").

Karsi-olgu (yaklasik; kural degisikligi oncesi tek replay sart): setup'in seviye anindaki bias'i (levels_at'e kadar
KAPANMIS son gunun karne satiri) bugunku kural ve varyantla okunur. Ters -> hard filtre eler; hizali -> `htf` = +2;
NEUTRAL -> `htf` aynen (reversal-at-PD +2 olabilir). Skor `min(11, min(9, raw - smt) + smt)` ile yeniden kurulur,
esik 7. `target_taken` ve CRT-oncesi asamalar acilmaz; RR >= 2 RR kapisini taklit eder. Taklit EDILMEYEN: dar stop,
skor-7 kapisi, kume/korelasyon, retest skoru.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta

from app.crt_engine import STRUCTURE_STALE_DAYS
from app.score_outcome import outcome_of
from app.setup_journal import PRE_SETUP_STAGES, analysis_parts, levels_final

DECISION_SINCE = datetime(2026, 10, 29)   # hipotez 28.10'a kadarki veriden cikti; karar bundan sonraki seviyelerden
THR = 7
MIN_RR = 2.0
NEED_ADDED = 40      # 4H: varyantin ekledigi, cozulmus (TP once mi)
NEED_DROPPED = 15    # 4H: varyantin dusurdugu, cozulmus
STRATEGIES = ("4h", "1d")   # 1W'de hard filtre yok, htf kalemi aylik bias


def _sg(b) -> int:
    return 1 if b == "BULLISH" else (-1 if b == "BEARISH" else 0)


def _fresh(r) -> bool:
    age = r.get("structure_age")
    return r.get("structure") in ("BULLISH", "BEARISH") and age is not None and age <= STRUCTURE_STALE_DAYS


def bias_now(r) -> str:
    """Bugunku kural, karne alanlarindan (karnenin `combined`'i ile ayni olmali)."""
    return r.get("combined") or "NEUTRAL"


def bias_variant(r) -> str:
    """Hipotez: taze structure varsa onun yonu (ict celisse de NEUTRAL degil); bayatsa ict."""
    return r["structure"] if _fresh(r) else (r.get("ict") or "NEUTRAL")


def bias_ict_only(r) -> str:
    """Referans kol: yalniz ict (28.10 olcumunde 4H'te eklenenler dolum bazli eksi)."""
    return r.get("ict") or "NEUTRAL"


VARIANTS = (("variant", "çatışmada taze structure", bias_variant), ("ict_only", "yalnız ict (referans)", bias_ict_only))


def build_map(rows) -> dict:
    """bias_journal satirlari (dict) -> {sembol: ([gun acilisi], [satir])}, siralı."""
    m: dict = {}
    for r in rows:
        d = r["day"]
        d = d if isinstance(d, datetime) else datetime.fromisoformat(str(d)[:19])
        m.setdefault(r["symbol"], []).append((d.replace(tzinfo=None), r))
    out = {}
    for sym, v in m.items():
        v.sort(key=lambda x: x[0])
        out[sym] = ([d for d, _ in v], [r for _, r in v])
    return out


def bias_row_at(bmap: dict, symbol, levels_at):
    """levels_at'e kadar KAPANMIS son gunun karne satiri (gun acilisi + 24s <= levels_at)."""
    days, rows = bmap.get(symbol, ((), ()))
    if not days or levels_at is None:
        return None
    la = levels_at if isinstance(levels_at, datetime) else datetime.fromisoformat(str(levels_at)[:19])
    i = bisect_right(days, la.replace(tzinfo=None) - timedelta(hours=24))
    return rows[i - 1] if i else None


def _score_with(pal: dict, htf: int) -> int:
    smt = pal.get("smt") or 0
    raw = pal["raw"] - (pal.get("htf") or 0) + htf
    return min(11, max(0, min(9, raw - smt)) + smt)


def _opens(pal: dict, d: int, b: str, stage) -> bool:
    if stage == "target_taken":
        return False
    s = _sg(b)
    if s == -d:
        return False
    return _score_with(pal, 2 if s == d else (pal.get("htf") or 0)) >= THR


def flips(journal_rows, bmap: dict, fn=bias_variant, since: datetime | None = None) -> tuple[list, list]:
    """(eklenen, dusen): varyantla acilip bugun acilmayan / bugun acilip varyantla acilmayan setuplar.
    Her oge dict(o, rr, at, sym, d). Satirlar Journal kolonlarini tasiyan dict / Row."""
    added, dropped = [], []
    for r in journal_rows:
        st = r["strategy"]
        if st not in STRATEGIES or r["best_stage"] in PRE_SETUP_STAGES or r["entry"] is None:
            continue
        if not levels_final(r):
            continue
        la = r["levels_at"]
        la = la if isinstance(la, datetime) else (datetime.fromisoformat(str(la)[:19]) if la else None)
        if la is None or (since and la.replace(tzinfo=None) < since):
            continue
        pal, _f = analysis_parts(r)
        if not pal or "raw" not in pal:
            continue
        o, rr = outcome_of(r)
        if (rr or 0) < MIN_RR:
            continue
        br = bias_row_at(bmap, r["symbol"], la)
        if br is None:
            continue
        d = 1 if r["direction"] == "LONG" else -1
        now, new = _opens(pal, d, bias_now(br), r["best_stage"]), _opens(pal, d, fn(br), r["best_stage"])
        if now == new:
            continue
        (added if new else dropped).append(dict(o=o, rr=rr or 0.0, at=la, sym=r["symbol"], d=r["direction"], s=st))
    return added, dropped


def stats(sub) -> dict:
    """TP once mi (win + tp_before_entry = TP) ve dolum bazli (yalniz win/loss) R."""
    w = [x for x in sub if x["o"] in ("win", "tp_before_entry")]
    wf = [x for x in sub if x["o"] == "win"]
    l_ = [x for x in sub if x["o"] == "loss"]
    res, resf = len(w) + len(l_), len(wf) + len(l_)
    return dict(n=len(sub), res=res, w=len(w), tpb=len(w) - len(wf), l=len(l_),
                rps=((sum(x["rr"] for x in w) - len(l_)) / res) if res else None,
                rpf=((sum(x["rr"] for x in wf) - len(l_)) / resf) if resf else None)
