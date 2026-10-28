"""Elenen setuplar sayfasi (22.10, kullanici istegi) -- `/eliminated`.

Soru: motorun gordugu ama AKTIF SINYALE DONUSMEYEN butun setuplar (CRT saymadigi adaylar dahil) sonra ne yapti --
hangi filtre kac setup eledi, elenenler TP'ye mi SL'ye mi gitti, filtre koruyor mu kazanc mi kesiyor. Veri
Setup Journal (`setup_journal`); Journal sayfasi ayni veriyi gosteriyor ama okunmuyordu (kullanici: "journal'dan
bu verileri okumak cok zor").

Kapsam: `outcome != 'signal'` (sinyale donusen satirin izlemesi durur; waiting'e ulasip dolmadan silinen setup
izlemeye geri doner -> burada "Signal deleted before fill"). Sonuc DUZ TP/SL: Journal seviyeleri ilk goruldugu
haliyle donar, kismi kar + BE yok. Motor kararina dokunmaz; yalniz okur.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SetupJournal, Signal
from app.setup_journal import PRE_SETUP_STAGES, STAGES

PAGE_SIZE = 50
DAYS_CHOICES = (7, 30, 90, 0)          # 0 = tum zamanlar
MIN_RESOLVED = 10                      # hukum icin en az cozulmus (TP+SL) setup
VERDICT_BAND = 0.15                    # R/setup bandi -- IZLEME karar kurallariyla ayni olcek

# Sayfada gosterilen filtre adi + NOT sutunu (22.10, kullanici: "tablo Turkce olsun, not alani olsun").
# (kod, ad, not) -- not: filtrenin ne yaptigi, tek cumle.
_GATES = [
    ("sweep_small", "Süpürme çok küçük", "C2, C1 ucunu eşiğin altında deldi (28.09'dan beri eşik yok, eski kayıtlar)"),
    ("range_atr", "C1 boyu ATR bandı dışında", "C1 mumu ortalama mum boyuna göre çok küçük ya da çok büyük"),
    ("c1_stale", "C1 ucu önceden alınmış", "C1'in ucu C2'den önce başka bir mumla zaten süpürülmüştü"),
    ("c2_breakout", "C2, C1 dışında kapandı", "C2 süpürdüğü ucun ötesinde kapandı — dönüş olmadı"),
    ("c2_wrong_color", "C2 yanlış renk", "C2 ters renkte kapandı (LONG'da kırmızı, SHORT'ta yeşil) — 4H'te kesin filtre"),
    ("c1_weak", "C1 gövdesi küçük", "C1 gövdesi aralığının %30'undan küçük (yalnız 4H)"),
    ("c2_second_sweep", "C2 önce öbür ucu süpürdü", "C2 iki ucu da süpürdü; yön ilk süpürülen uçtan, bu yön sonra süpürülen (24.10)"),
    ("not_selected", "Başka CRT adayı seçildi", "Aynı anda birden çok CRT vardı, motor diğerini seçti"),
    ("bias_mismatch", "1D bias ters", "Günlük bias işlem yönünün tersine"),
    ("target_taken", "Hedef önceden alınmış", "TP seviyesi setup oluşmadan fiyatça alınmıştı"),
    ("low_quality", "Skor eşiğin altında", "Kalite skoru eşiğin altında (4H/1D 7, 1W 6)"),
    ("no_cisd", "CISD / MSS yok", "Giriş seviyesi (CISD ya da MSS) bulunamadı"),
    ("low_rr", "RR 2'nin altında", "Risk/ödül oranı 2'den küçük"),
    ("score7", "Skor-7 kapısı", "Skor tam eşikte; ek şartlar (yapısal giriş, key level, kapalı C2) tutmadı"),
    ("tight_stop", "Stop çok dar", "Stop mesafesi küçük zaman diliminin mum boyundan kısa"),
    ("past_tp", "Fiyat TP'yi geçti", "Emir dolmadan fiyat TP'ye gitti"),
    ("near_tp", "Fiyat TP'ye %80 yaklaştı", "Emir dolmadan fiyat giriş → TP yolunun %80'ini gördü (28.10; fiyat akışı farkı)"),
    ("past_sl", "Fiyat SL'yi geçti", "Emir dolmadan fiyat SL'nin ötesine geçti"),
    ("invalidated", "CRT %60 geçildi", "Fiyat CRT'nin %60'ını geçti (kural 10.09'da kapandı, eski kayıtlar)"),
    ("missed_quality", "Retestte skor < 7", "Fiyat girişe döndüğünde skor eşiğin altına düşmüştü"),
    ("stale", "Retest çok eski", "Giriş, emir açılmadan çok önce test edilmişti (bayat emir)"),
    ("week_gap", "CISD hafta kapanışından önce", "Seanslı sembolde CISD hafta sonundan önceydi, Pazartesi diriltilmedi"),
    ("cluster_limit", "Kripto küme limiti", "Aynı yönde çok sayıda kripto işlemi vardı"),
    ("liq_unswept", "Yakın asıl likidite alınmadı", "C2 ucunun hemen ötesindeki (≤ 0.1 ATR) alınmamış swing / eşit dip-tepe süpürülmeden giriş yok (28.10); süpürülmediyse elendi"),
    ("same_setup", "Aynı setup başka coinde", "Aynı C1/C2 mumundan aynı yönde başka kripto sembolde sinyal vardı (28.10)"),
    ("corr_open", "Korele paritede açık işlem", "Bağlantılı paritede zaten açık işlem vardı"),
    ("has_open", "Sembolde açık sinyal", "Aynı sembol ve stratejide açık sinyal vardı"),
    ("duplicate", "Zaten kayıtlı", "Aynı setup daha önce kaydedilmişti"),
    ("deleted_before", "Bir kez silinmiş", "Aynı CRT'nin sinyali bir kez silinmişti, yeniden açılmadı"),
    ("c2_open", "C2 kapanmadan bozuldu", "C2 kapanışı beklenirken setup geçerliliğini yitirdi"),
    ("waiting", "Sinyal dolmadan silindi", "Sinyal açıldı ama emir dolmadan silindi (nedeni listede)"),
    ("week_close", "Hafta kapanışında iptal", "Cuma kapanışında dolmamış setup iptal edildi"),
]
GATE_LABELS = {c: lbl for c, lbl, _ in _GATES}
GATE_NOTES = {c: note for c, _, note in _GATES}
# Silinme nedeni (`deleted_reason`) -- kapi kodu degilse buradan; yoksa GATE_LABELS.
DELETE_LABELS = {"crt_gone": "CRT geçersizleşti", "crt_replaced": "başka CRT geldi",
                 "score7_gate": "Skor-7 kapısı", "manual": "elle"}
# Motorun kapilari hangi asamada: CRT tespiti (PRE_SETUP) / setup kapilari / sinyal sonrasi.
GROUP_PRE = "CRT tespiti"
GROUP_GATE = "Setup kapıları"
GROUP_POST = "Sinyal sonrası"
_POST_STAGES = {"waiting", "week_close"}

RESULT_META = {   # outcome -> (etiket, renk sinifi)
    "win": ("TP", "bg-accent-green/15 text-accent-green"),
    "loss": ("SL", "bg-accent-red/15 text-accent-red"),
    "tp_before_entry": ("TP (girişsiz)", "bg-accent-yellow/15 text-accent-yellow"),
    "ambiguous": ("Aynı mum", "bg-dark-600 text-gray-300"),
    "no_touch": ("Dolmadı", "bg-dark-600 text-gray-400"),
    "open": ("Süre doldu", "bg-dark-600 text-gray-400"),
    "pending": ("İzleniyor", "bg-accent-blue/15 text-accent-blue"),
    "filled": ("İzleniyor (doldu)", "bg-accent-blue/15 text-accent-blue"),
    None: ("Seviye yok", "bg-dark-600 text-gray-500"),
}
RESULT_FILTERS = {  # ?result= -> outcome kumesi
    "tp": ("win",),
    "sl": ("loss",),
    "tbe": ("tp_before_entry",),
    "tracking": ("pending", "filled"),
    "other": ("ambiguous", "no_touch", "open"),
}

VERDICT_META = {
    "protect": ("Koruyor", "text-accent-green", "elenenler zararda — filtre kaybı önledi"),
    "cut": ("Kazanç kesiyor", "text-accent-red", "elenenler kârda — filtre kazancı engelledi"),
    "neutral": ("Etkisiz", "text-gray-400", "elenenler başa baş"),
    "few": ("Az veri", "text-gray-500", f"TP ya da SL'ye giden setup {MIN_RESOLVED}'dan az"),
}


def _gate_group(code: str) -> str:
    if code in PRE_SETUP_STAGES:
        return GROUP_PRE
    return GROUP_POST if code in _POST_STAGES else GROUP_GATE


def verdict(tp: int, sl: int, r: float) -> str:
    """Filtrenin hukmu: elenenler cozuldugunde R/setup. Negatif = filtre kaybi onlemis."""
    n = tp + sl
    if n < MIN_RESOLVED:
        return "few"
    per = r / n
    if per <= -VERDICT_BAND:
        return "protect"
    if per >= VERDICT_BAND:
        return "cut"
    return "neutral"


def _js(v) -> dict:
    if not v:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return {}


def _pct(x) -> Optional[str]:
    return None if x is None else f"%{float(x) * 100:.0f}"


def why_text(row: SetupJournal) -> str:
    """Elemenin nedeni, filtreye ozgu olcumle (ör. C2 rengi elemesinde C2 govde orani)."""
    f = _js(row.features_at_levels) or _js(row.features)
    code = row.best_stage
    parts: list[str] = []
    if code == "c2_wrong_color":
        b = _pct(f.get("c2_body_frac"))
        parts.append(f"C2 {'kırmızı' if row.direction == 'LONG' else 'yeşil'} kapandı" + (f", gövde oranı {b}" if b else ""))
    elif code == "c1_weak":
        b = _pct(f.get("c1_body_frac"))
        parts.append(f"C1 gövde oranı {b} (en az %30)" if b else "C1 gövdesi küçük")
    elif code == "c2_second_sweep":
        parts.append("C2 önce C1 " + ("tepesini" if row.direction == "LONG" else "dibini")
                     + " süpürdü — setup ters yönde, bu uç onun hedefi")
    elif code == "c2_breakout":
        parts.append("C2, süpürdüğü C1 ucunun ötesinde kapandı")
    elif code == "range_atr" and f.get("range_atr") is not None:
        parts.append(f"C1 boyu {float(f['range_atr']):.2f}× ATR")
    elif code == "sweep_small" and f.get("sweep_pct") is not None:
        parts.append(f"süpürme %{float(f['sweep_pct']):.3f}")
    elif code == "bias_mismatch":
        parts.append(f"1D bias {row.htf_bias or '?'}, işlem {row.direction}")
    elif code == "low_rr" and row.rr is not None:
        parts.append(f"RR {row.rr:.2f} < 2")
    elif code in ("low_quality", "score7", "missed_quality") and row.score is not None:
        parts.append(f"skor {row.score:.0f}")
    elif code == "tight_stop" and f.get("stop_range_mult") is not None:
        parts.append(f"stop {float(f['stop_range_mult']):.2f}× LTF mum boyu")
    elif row.deleted_reason:
        reason = DELETE_LABELS.get(row.deleted_reason) or GATE_LABELS.get(row.deleted_reason, row.deleted_reason)
        parts.append(f"silindi: {reason}")
    rec = f.get("reclaim_pct")
    if rec is not None and code not in ("c2_breakout",) and float(rec) < 25:
        parts.append(f"zayıf C2 geri dönüşü %{float(rec):.0f}")
    return " · ".join(parts)


def row_r(row: SetupJournal) -> Optional[float]:
    if row.outcome == "win" and row.rr is not None:
        return round(float(row.rr), 2)
    if row.outcome == "loss":
        return -1.0
    return None


def row_hours(row: SetupJournal) -> Optional[float]:
    """Sonuca kac saatte gidildi: TP/SL'de girisin dolumundan, girissiz TP'de seviyelerin donmasindan."""
    if row.outcome_at is None:
        return None
    start = row.entry_touched_at if row.outcome in ("win", "loss") else row.levels_at
    if start is None:
        return None
    return round((row.outcome_at - start).total_seconds() / 3600, 1)


def _base_filters(tf: str, market: str, days: int, symbol: str) -> list:
    t = SetupJournal
    flt = [or_(t.outcome.is_(None), t.outcome != "signal")]
    if tf != "all":
        flt.append(t.strategy == tf)
    if market == "crypto":
        flt.append(t.market_type == "crypto")
    elif market == "fx":
        flt.append(t.market_type != "crypto")
    if days:
        flt.append(t.purge_time >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days))
    if symbol:
        flt.append(t.symbol.ilike(f"%{symbol}%"))
    return flt


async def _signal_reference(db: AsyncSession, tf: str, market: str, days: int) -> dict:
    """Ayni donemde gercekten acilan sinyaller (kapanis zamanina gore): W/L/BE + gercek R -- kiyas icin."""
    flt = [Signal.closed_at.is_not(None), Signal.result.is_not(None)]
    if tf != "all":
        flt.append(Signal.timeframe == tf)
    if market == "crypto":
        flt.append(Signal.market_type == "crypto")
    elif market == "fx":
        flt.append(Signal.market_type != "crypto")
    if days:
        flt.append(Signal.closed_at >= datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days))
    rows = (await db.execute(select(Signal.result, Signal.exit_reason, Signal.rr_value).where(*flt))).all()
    w = l = be = 0
    r = 0.0
    for res, ex, rv in rows:
        if ex == "be":
            be += 1          # kismi karli BE DB'de result='win' -- main._trade_outcome kurali
        elif res == "win":
            w += 1
        elif res == "loss":
            l += 1
        else:
            be += 1
        r += float(rv or 0.0)
    n = w + l + be
    return {"n": n, "w": w, "l": l, "be": be, "r": round(r, 2),
            "wr": round(w / (w + l) * 100, 1) if w + l else None}


async def build_eliminated(db: AsyncSession, *, tf: str = "all", market: str = "all", days: int = 30,
                           gate: str = "all", result: str = "all", symbol: str = "", page: int = 1) -> dict:
    tf = tf if tf in ("all", "4h", "1d", "1w", "1h") else "all"
    market = market if market in ("all", "crypto", "fx") else "all"
    days = days if days in DAYS_CHOICES else 30
    gate = gate if gate in GATE_LABELS else "all"
    result = result if result in RESULT_FILTERS else "all"
    symbol = (symbol or "").strip()
    t = SetupJournal
    base = _base_filters(tf, market, days, symbol)

    # 1) Filtre tablosu: SQL'de gruplanir (tum zamanlar ~11k satir).
    win_r = func.sum(case((t.outcome == "win", t.rr), else_=0.0))

    def _n(cond):
        return func.sum(case((cond, 1), else_=0))

    grouped = (await db.execute(select(
        t.best_stage, func.count(t.id), _n(t.outcome == "win"), _n(t.outcome == "loss"),
        _n(t.outcome == "tp_before_entry"), _n(t.outcome.in_(("pending", "filled"))),
        _n(or_(t.outcome.is_(None), t.outcome.in_(("no_touch", "open", "ambiguous")))), win_r,
    ).where(*base).group_by(t.best_stage))).all()
    per = {g[0]: g for g in grouped}
    gates = []
    for code, _label in STAGES:
        g = per.get(code)
        if not g:
            continue
        _, n, tp, sl, tbe, trk, other, wr_sum = g
        tp, sl = int(tp or 0), int(sl or 0)
        r = float(wr_sum or 0.0) - sl
        v = verdict(tp, sl, r)
        gates.append({
            "code": code, "label": GATE_LABELS.get(code, code), "note": GATE_NOTES.get(code, ""), "group": _gate_group(code),
            "n": int(n), "tp": tp, "sl": sl, "tbe": int(tbe or 0), "tracking": int(trk or 0), "other": int(other or 0),
            "wr": round(tp / (tp + sl) * 100, 1) if tp + sl else None,
            "r": round(r, 1), "r_per": round(r / (tp + sl), 2) if tp + sl else None,
            "verdict": v, "verdict_label": VERDICT_META[v][0], "verdict_class": VERDICT_META[v][1],
        })
    # 22.10 (kullanici): "Cutting winners" en ustte kendi bolumunde; kalan gruplarda hukum sirasi
    # (koruyan, etkisiz, az veri), esitlikte motor sirasi.
    order = {"protect": 0, "neutral": 1, "few": 2}
    groups = []
    cut = [g for g in gates if g["verdict"] == "cut"]
    if cut:
        groups.append({"name": "Kazanç kesenler", "cut": True, "gates": cut})
    for name in (GROUP_PRE, GROUP_GATE, GROUP_POST):
        rows_ = sorted((g for g in gates if g["group"] == name and g["verdict"] != "cut"),
                       key=lambda g: order[g["verdict"]])
        if rows_:
            groups.append({"name": name, "cut": False, "gates": rows_})

    tp = sum(g["tp"] for g in gates)
    sl = sum(g["sl"] for g in gates)
    r = sum(g["r"] for g in gates)
    totals = {
        "n": sum(g["n"] for g in gates), "tp": tp, "sl": sl,
        "tbe": sum(g["tbe"] for g in gates), "tracking": sum(g["tracking"] for g in gates),
        "other": sum(g["other"] for g in gates),
        "wr": round(tp / (tp + sl) * 100, 1) if tp + sl else None,
        "r": round(r, 1), "r_per": round(r / (tp + sl), 2) if tp + sl else None,
    }
    # CRT sayilan / sayilmayan ayri satir: "en bastaki" filtreler toplamin cogunu olusturuyor.
    split = []
    for name, cond in (("Motorun CRT saydığı setuplar", lambda g: g["group"] != GROUP_PRE),
                       ("CRT tespitinde elenen adaylar", lambda g: g["group"] == GROUP_PRE)):
        rows_ = [g for g in gates if cond(g)]
        s_tp, s_sl = sum(g["tp"] for g in rows_), sum(g["sl"] for g in rows_)
        s_r = sum(g["r"] for g in rows_)
        split.append({"name": name, "n": sum(g["n"] for g in rows_), "tp": s_tp, "sl": s_sl,
                      "wr": round(s_tp / (s_tp + s_sl) * 100, 1) if s_tp + s_sl else None,
                      "r_per": round(s_r / (s_tp + s_sl), 2) if s_tp + s_sl else None})

    # 2) Setup listesi
    flt = list(base)
    if gate != "all":
        flt.append(t.best_stage == gate)
    if result != "all":
        flt.append(t.outcome.in_(RESULT_FILTERS[result]))
    total = await db.scalar(select(func.count(t.id)).where(*flt)) or 0
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), total_pages)
    recs = (await db.execute(select(t).where(*flt).order_by(t.purge_time.desc(), t.id.desc())
                             .offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE))).scalars().all()
    rows = []
    for x in recs:
        label, cls = RESULT_META.get(x.outcome, RESULT_META[None])
        rows.append({
            "id": x.id, "symbol": x.symbol, "market_type": x.market_type, "tf": (x.strategy or "").upper(),
            "direction": x.direction, "c1": x.crt_bar_time, "c2": x.purge_time,
            "gate": GATE_LABELS.get(x.best_stage, x.best_stage), "gate_code": x.best_stage,
            "pre": x.best_stage in PRE_SETUP_STAGES, "why": why_text(x),
            "score": x.score, "rr": x.rr, "bias": x.htf_bias, "model": x.model,
            "entry": x.entry, "sl": x.sl, "tp": x.tp,
            "result": label, "result_class": cls, "r": row_r(x), "hours": row_hours(x),
            "outcome_at": x.outcome_at,
        })

    return {
        "tf": tf, "market": market, "days": days, "gate": gate, "result": result, "symbol": symbol,
        "rows": rows, "rows_open": gate != "all" or result != "all" or page > 1, "page": page, "total": total, "total_pages": total_pages,
        "groups": groups, "totals": totals, "split": split,
        "reference": await _signal_reference(db, tf, market, days),
        "gate_labels": GATE_LABELS, "days_choices": DAYS_CHOICES,
        "verdict_meta": VERDICT_META, "min_resolved": MIN_RESOLVED, "band": VERDICT_BAND,
    }
