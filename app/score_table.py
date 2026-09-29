"""Skor tablosu + hard filtreler sayfasi (/skor-tablosu, 28.09 kullanici istegi).

Uc stratejinin (4H / 1D / 1W) kalite skoru farkli oldugu icin unutuluyordu; sayfa her strateji
icin bir sekmede (1) skor kalemlerini, (2) hard filtreleri ve (3) her satirin CANLI saglik
kontrolunu gosterir. Saglik kontrolu Setup Journal'dan (tablo `setup_journal`) okunur, motora
dokunmaz:

- Skor kalemi: son pencerede motorun setup'larinda kalem kac kez puan verdi; beklenmeyen deger
  (izinli kume disinda) ya da bu stratejide olmamasi gereken puan -> hata. Hic tetiklenmeyen
  ya da neredeyse her setupta puan veren kalem -> uyari (ayirt etmiyor).
- Toplam: kalemlerin toplami `raw` ile, tavan formulu `score` ile tutuyor mu.
- Hard filtre: pencerede kac setup'i eledi (`best_stage`); bu stratejide KAPALI olmasi gereken
  filtre eleme yaptiysa -> hata.

Puanlar ve esikler motor sabitlerinden okunur (crt_engine / scanner), elle yazilmaz.
Arayuz metinleri bu sayfada TURKCE (kullanici istegi 28.09; diger sayfalar Ingilizce).
⚠️ SKOR TABLOSU DEGISINCE BU DOSYAYI DA GUNCELLE: kalem ekle/cikar (`_score_rows`), izinli
degerler, `RULES_SINCE` (kuralin degistigi gun -- oncesi saglik penceresine girmez) ve olcum
notlari (`_NOTES`). `crt_engine.SCORE_PART_KEYS`'te burada olmayan kalem varsa sayfa kirmizi
uyari basar. Dogrulama: tmp/tmp_score_table_check.py.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import crt_engine as ce
from app import scanner as sc
from app.models import SetupJournal
from app.setup_journal import PRE_SETUP_STAGES

log = logging.getLogger(__name__)

TABS = (("4h", "4H-15M"), ("1d", "1D-1H"), ("1w", "1W-4H"))

# Skor kuralinin bu stratejide son degistigi gun: saglik penceresi bundan once baslamaz
# (eski kuralla yazilmis satirlar yanlis alarm verir). Skor degisince guncelle.
#   4H/1D 25.09: IFVG mid'in C1'de olma sarti kalkti; 1D'de acik C2 +1 almiyor, yanlis renk 0.
#   1W 28.09: strateji acildi.
#   29.09 key level duzeltmesi (pd_major / pd_monthly / pd_struct tanimi) ayni gerekceyle ilerletilmedi.
#   29.09 IFVG'nin FVG'si C2 icinde olusmali (ifvg kalemi): tarih ILERLETILMEDI -- eski satirlar da
#   izinli degerleri (0/1) tasiyor, yanlis alarm uretmez; pencereyi sifirlamak tum kalemleri korlestirirdi.
RULES_SINCE = {
    "4h": datetime(2026, 9, 25, tzinfo=timezone.utc),
    "1d": datetime(2026, 9, 25, tzinfo=timezone.utc),
    "1w": datetime(2026, 9, 28, tzinfo=timezone.utc),
}
WINDOW = timedelta(days=7)
MIN_N = 10            # bunun altinda "veri az" -- oran yorumlanmaz
ALWAYS_FRAC = 0.95    # kalem setuplarin >= %95'inde puan veriyorsa ayirt etmiyor

# Olcum notu (IZLEME.md'deki kararin tek satirlik hali). Kalem/strateji karara baglaninca guncelle.
_NOTES: dict[tuple[str, str], str] = {
    ("4h", "htf"): "+2 hak ediliyor mu? Karar 03.10",
    ("4h", "pd_major"): "28.09 ilk okuma: katkısı görünmüyor, karar 03.10 · 29.09 tanım değişti (ilk süpüren C2)",
    ("4h", "pd_struct"): "29.09 tanım değişti (OB yapı kırılımı, bölgeye ilk giren C2) — öncesi setupların %65'inde yanıyordu",
    ("4h", "pd_monthly"): "28.09 ilk okuma: katkısı görünmüyor, karar 03.10",
    ("4h", "wick"): "28.09: eşiği 0.30'a çıkarmak aday, karar 03.10",
    ("1d", "htf"): "Filtre koruyor; +2 ödülü karar 03.10",
    ("1d", "c2_closed"): "25.09'dan beri açık C2 +1 almıyor, karar 03.10",
    ("1d", "base"): "25.09'dan beri yanlış renk de setup olur (0 puan)",
    ("1w", "htf"): "Yeni (28.09), henüz ölçüm yok",
}
_RECLAIM_NOTE = "19.09 kanıtlandı: zayıf dönüş belirgin kötü"


def _score_rows(strategy: str) -> list[dict]:
    """Stratejinin skor kalemleri. `allowed` = motorun setup'inda gorulebilecek degerler;
    `absent` = bu stratejide kalem yok, her zaman 0 olmali."""
    w1 = strategy == ce.W1_TIMEFRAME
    d1 = strategy == "1d"
    pen = ce.C2_RECLAIM_PENALTY
    doji = int(ce.DOJI_BODY_RATIO_MAX * 100)
    color_hard = strategy in ce.C2_COLOR_HARD_FILTER_TFS
    rows = [
        {
            "key": "base", "label": "C2 rengi",
            "points": "+2 / +1 / 0",
            "rule": f"Doğru renk +2 · doji (gövde < %{doji}) +1 · yanlış renk 0"
                    + (" — yanlış renk zaten hard filtrede elenir" if color_hard else " (yine de setup olur)"),
            "allowed": {0, 1, 2},
        },
        {
            "key": "htf", "label": "Aylık bias" if w1 else "1D bias",
            "points": "+2 / 0 / −2",
            "rule": ("Aylık bias yönde +2 · NEUTRAL iken PMH/PML'de dönüş +2 · ters −2 · NEUTRAL 0"
                     if w1 else
                     "1D bias yönde +2 · NEUTRAL iken major PD'de dönüş +2 · ters −2 · NEUTRAL 0"),
            "allowed": {-2, 0, 2},
        },
        {
            "key": "weekly", "label": "1W bias uyumu",
            "points": "—" if w1 else "+1",
            "rule": ("Bu stratejide yok (HTF zaten haftalık)" if w1
                     else "Haftalık bias işlem yönündeyse +1"),
            "allowed": {0} if w1 else {0, 1}, "absent": w1,
        },
        {
            "key": "c2_closed", "label": "C2 kapalı",
            "points": "+1",
            "rule": "C2 mumu kapandıysa +1"
                    + (" (C2 açıkken de setup olur, +1'i almaz)" if (d1 or w1) else ""),
            "allowed": {0, 1},
        },
        {
            "key": "pd_major", "label": "Likidite (key level)",
            "points": "+1",
            "rule": ("C2 PMH/PML'ye dokundu (PDH/PDL/PWH/PWL sayılmaz)" if w1
                     else ("C2 PWH/PWL'ye dokundu (PDH/PDL 1D'de totolojik, sayılmaz)" if d1
                           else "C2 PDH/PDL/PWH/PWL'ye dokundu"))
                    + ". Seviyeyi ilk süpüren C2 olmalı, seviyeler C2 açılışına göre. "
                      "EQH/EQL ve SSL/BSL etiket olarak yazılır, puan almaz",
            "allowed": {0, 1},
        },
        {
            "key": "pd_monthly", "label": "PD aylık",
            "points": "—" if w1 else "+1",
            "rule": ("Bu stratejide yok (aylık seviye PD major sayılıyor)" if w1
                     else "C2 PMH/PML'ye dokundu (major ile toplanır)"),
            "allowed": {0} if w1 else {0, 1}, "absent": w1,
        },
        {
            "key": "pd_struct", "label": "HTF FVG / OB",
            "points": "+1",
            "rule": ("Haftalık" if w1 else "HTF") + " FVG ya da OB'ye dokundu (ikisi birden yine +1). "
                    "Bölgeye ilk giren C2 olmalı; OB = yapıyı kıran hamleden önceki son ters mum; "
                    f"FVG boşluğu ≥ {ce.MIN_HTF_FVG_ATR_FRAC:g} × ATR",
            "allowed": {0, 1},
        },
        {
            "key": "wick", "label": "Purge fitili",
            "points": "+1",
            "rule": f"C2 fitili ≥ C1 aralığının %{ce.PURGE_WICK_SCORE_PCT * 100:g}'u",
            "allowed": {0, 1},
        },
        {
            "key": "ifvg", "label": "LTF IFVG",
            "points": "+1",
            "rule": (f"C2 mumu içinde oluşmuş{'' if ce.IFVG_FVG_FROM_C2 else ' (C1 dahil)'}, purge sonrası invert "
                     f"olmuş açık FVG (boşluk ≥ {ce.MIN_IFVG_GAP_RANGE_FRAC:g} × ort. LTF mumu)"),
            "allowed": {0, 1},
        },
        {
            "key": "reclaim", "label": "Zayıf C2 dönüşü",
            "points": f"−{pen}",
            "rule": f"C2, C1 aralığına %{ce.C2_RECLAIM_WEAK_PCT:g}'ten az geri döndüyse −{pen} (ceza, filtre değil)",
            "allowed": {0, -pen},
        },
        {
            "key": "smt", "label": "SMT divergence",
            "points": f"+{sc.SMT_QUALITY_BONUS}",
            "rule": f"Korele paritede SMT +{sc.SMT_QUALITY_BONUS} — tavandan SONRA eklenir",
            "allowed": {0, sc.SMT_QUALITY_BONUS},
        },
    ]
    for r in rows:
        r.setdefault("absent", False)
        r["note"] = _RECLAIM_NOTE if r["key"] == "reclaim" else _NOTES.get((strategy, r["key"]), "")
    return rows


def _limits(strategy: str) -> dict:
    cfg = sc.STRATEGY_CFG.get(strategy) or {}
    max_score = int(cfg.get("max_score", sc.MAX_QUALITY_SCORE))
    cap = ce.W1_SCORE_CAP if strategy == ce.W1_TIMEFRAME else max_score - sc.SMT_QUALITY_BONUS
    return {"cap": cap, "max": max_score, "min": sc._min_score(cfg)}


def _filter_rows(strategy: str) -> list[dict]:
    """Hard filtreler, motordaki sirayla. `on` = bu stratejide uygulanir mi (sabitten okunur).
    `stage` = Journal'daki eleme kodu (None: Journal saymiyor, canli kontrol yok)."""
    cfg = sc.STRATEGY_CFG.get(strategy) or {}
    lim = _limits(strategy)
    c2_wait = bool(cfg.get("require_c2_closed"))
    bias_on = bool(sc.REQUIRE_HTF_BIAS_ALIGN and cfg.get("require_bias_align", True))
    color_on = strategy in ce.C2_COLOR_HARD_FILTER_TFS
    crypto_alts = strategy == "4h"  # 1D/1W evreninde kripto yalniz BTC/ETH (kume limitinden muaf)
    rr = sc._min_rr_for_market("crypto")
    stop_mult = cfg.get("min_stop_range_mult", sc.MIN_STOP_RANGE_MULT)
    backfill = cfg.get("max_backfill_fill_bars", sc.MAX_BACKFILL_FILL_BARS)
    return [
        # --- CRT tanimi (setup hic dogmaz) ---
        {"group": "CRT yapısı", "label": "C2, C1 aralığının içinde kapanmalı", "stage": "c2_breakout", "on": True},
        {"group": "CRT yapısı",
         "label": f"C1 aralığı ATR bandında ({ce.MIN_RANGE_ATR_RATIO:g}–{ce.MAX_RANGE_ATR_RATIO:g} × ATR)",
         "stage": "range_atr", "on": True},
        {"group": "CRT yapısı", "label": "C1 ucu purge'den önce alınmamış olmalı", "stage": "c1_stale", "on": True},
        {"group": "CRT yapısı", "label": "C2 rengi (belirgin yanlış renk C1'i eler)", "stage": "c2_wrong_color",
         "on": color_on, "off_note": "Yalnız skorda (yanlış renk 0 puan)"},
        # --- Kapilar (scanner.detect_and_create_waiting sirasi) ---
        {"group": "Kapı", "label": "1D bias ters olmamalı (NEUTRAL geçer)", "stage": "bias_mismatch",
         "on": bias_on, "off_note": "Yok — aylık bias yalnız skorda"},
        {"group": "Kapı", "label": "Hedef tarafı C2'den sonra alınmamış olmalı", "stage": "target_taken", "on": True},
        {"group": "Kapı", "label": f"Skor ≥ {lim['min']}", "stage": "low_quality", "on": True},
        {"group": "Kapı", "label": "Aynı sembolde açık sinyal olmamalı", "stage": "has_open", "on": True},
        {"group": "Kapı", "label": "Korele paritede açık sinyal olmamalı", "stage": "corr_open", "on": True},
        {"group": "Kapı", "label": "Aynı setup daha önce kaydedilmemiş olmalı", "stage": "duplicate", "on": True},
        {"group": "Kapı",
         "label": f"Kripto aynı yön küme limiti (muaf: BTC/ETH ve skor ≥ {sc.CLUSTER_EXEMPT_MIN_SCORE})",
         "stage": "cluster_limit", "on": crypto_alts,
         "off_note": "Fiilen yok — evrende yalnız BTC/ETH var (muaf)"},
        {"group": "Kapı", "label": "CISD / MSS onayı ve giriş seviyesi olmalı", "stage": "no_cisd", "on": True},
        {"group": "Kapı", "label": f"RR ≥ {rr:g}", "stage": "low_rr", "on": True},
        {"group": "Kapı",
         "label": f"Skor tam {lim['min']} ise: CISD/MSS girişi + key level" + (" + kapalı C2" if c2_wait else ""),
         "stage": "score7", "on": bool(cfg.get("score7_requires_cisd_pd"))},
        {"group": "Kapı", "label": f"Stop ≥ {stop_mult:g} × ort. LTF mumu", "stage": "tight_stop", "on": True},
        {"group": "Kapı", "label": "CISD hafta kapanışından önce olmamalı (yalnız FX/metal/endeks/petrol)",
         "stage": "week_gap", "on": True},
        {"group": "Kapı", "label": "Girişten önce TP görülmemiş olmalı", "stage": "past_tp", "on": True},
        {"group": "Kapı", "label": "CISD'den sonra SL görülmemiş olmalı", "stage": "past_sl", "on": True},
        {"group": "Kapı", "label": "CRT %60 geçilmemiş olmalı", "stage": "invalidated",
         "on": bool(sc.REQUIRE_CRT_MID_INVALIDATION), "off_note": "Kapalı (10.09)"},
        {"group": "Kapı", "label": f"Retest anında skor ≥ {lim['min']}", "stage": "missed_quality", "on": True},
        {"group": "Kapı", "label": f"Retest {backfill} LTF mumundan eski olmamalı", "stage": "stale", "on": True},
        # --- Dolum ---
        {"group": "Dolum", "label": "C2 kapanmadan sinyal / dolum yok", "stage": None,
         "on": c2_wait, "off_note": "Yok — C2 açıkken dolabilir"},
        {"group": "Dolum", "label": "C2 kapanmadan IFVG/BPR girişi yok", "stage": None,
         "on": bool(cfg.get("ifvg_requires_c2_closed")), "off_note": "Yok — bölge girişi C2 açıkken serbest"},
    ]


def _expected_score(parts: dict, lim: dict) -> int:
    """Motorun formulu: tavan SMT'den ONCE, SMT sonra (scanner._apply_smt_bonus)."""
    smt = int(parts.get("smt") or 0)
    base = max(0, min(lim["cap"], int(parts.get("raw") or 0) - smt))
    return min(lim["max"], base + smt)


def _score_health(rows: list[dict], parts_list: list[dict], lim: dict) -> dict:
    """Her kaleme `health` = {level, text, detail} yazar; toplam kontrolunu doner.

    Kullanici istegi 28.09: sutun "puan veriyor mu"yu degil **dogru calisiyor mu**yu soyler.
    `text` = hukum (DURUM sutunu), `detail` = tek cumlelik ozet (OZET sutunu). Dort hukum:
    ok "Doğru çalışıyor" · warn "Etkisi yok" (hep ya da hic puan -> iyiyi kotuden ayirmiyor) ·
    bad "Hatalı" (kural disi puan / toplam-tavan tutmuyor) · na "Yetersiz veri"."""
    n = len(parts_list)
    item_keys = [r["key"] for r in rows]
    for r in rows:
        vals = [int(p.get(r["key"]) or 0) for p in parts_list]
        bad = sum(1 for v in vals if v not in r["allowed"])
        # "Etkisi yok" = setupların >= %95'i AYNI puanı alıyor (hep 0 ya da hep +1; C2 renginde
        # hep +2 gibi). Sifir-olmayan sayimi degil: cok degerli kalemde (+2/+1) yanlis alarm verirdi.
        top_v, top_n = max(Counter(vals).items(), key=lambda kv: kv[1]) if vals else (0, 0)
        if bad:
            wrong = ", ".join(f"{v:+d}" for v in sorted({v for v in vals if v not in r["allowed"]}))
            r["health"] = _h("bad", "Hatalı", f"{bad} setupa kurala aykırı puan verdi ({wrong}).")
        elif n < MIN_N:
            r["health"] = _h("na", "Yetersiz veri", f"Son 7 günde {n} setup var, karar için en az {MIN_N} gerekli.")
        elif r["absent"]:
            r["health"] = _h("ok", "Doğru çalışıyor", "Bu stratejide kullanılmıyor; beklendiği gibi hiç puan vermedi.")
        elif top_n / n >= ALWAYS_FRAC and top_v == 0:
            r["health"] = _h("warn", "Etkisi yok",
                             f"Neredeyse hiçbir setupa puan vermiyor ({n} setupta {n - top_n}); skoru değiştirmiyor.")
        elif top_n / n >= ALWAYS_FRAC:
            r["health"] = _h("warn", "Etkisi yok",
                             f"Neredeyse her setup aynı puanı alıyor ({n} setupta {top_n} kez {top_v:+d}); "
                             "iyiyi kötüden ayırmıyor.")
        else:
            r["health"] = _h("ok", "Doğru çalışıyor", "Kurala uygun puan veriyor, setupları ayırıyor.")
    raw_bad = sum(
        1 for p in parts_list
        if sum(int(p.get(k) or 0) for k in item_keys) != int(p.get("raw") or 0)
    )
    score_bad = sum(1 for p in parts_list if int(p.get("score") or 0) != _expected_score(p, lim))
    if n < MIN_N:
        return _h("na", "Yetersiz veri", f"Son 7 günde {n} setup var, karar için en az {MIN_N} gerekli.")
    if raw_bad or score_bad:
        return _h("bad", "Hatalı", f"{raw_bad + score_bad} setupta toplam skor kalemlerden yanlış hesaplanmış.")
    return _h("ok", "Doğru çalışıyor", "Toplam skor her setupta kalemlerden ve tavandan doğru hesaplanıyor.")


def _h(level: str, text: str, detail: str = "") -> dict:
    return {"level": level, "text": text, "detail": detail}


async def build_score_table(db: AsyncSession, strategy: str, now: datetime | None = None) -> dict:
    strategy = strategy if strategy in dict(TABS) else TABS[0][0]
    now = now or datetime.now(timezone.utc)
    since = max(now - WINDOW, RULES_SINCE.get(strategy, now - WINDOW))
    rows = _score_rows(strategy)
    filters = _filter_rows(strategy)
    lim = _limits(strategy)
    known = {r["key"] for r in rows}
    missing = [k for k in ce.SCORE_PART_KEYS if k not in ("raw", "score") and k not in known]

    stage_counts: dict[str, int] = {}
    parts_list: list[dict] = []
    error = None
    try:
        res = await db.execute(
            select(SetupJournal.best_stage, SetupJournal.score_parts).where(
                SetupJournal.strategy == strategy,
                SetupJournal.first_seen >= since.replace(tzinfo=None),
            )
        )
        for stage, parts in res.all():
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            # Skor sagligi yalniz motorun setup'larinda (CRT saymadigi adaylar haric).
            if parts and stage not in PRE_SETUP_STAGES:
                try:
                    parts_list.append(json.loads(parts))
                except (TypeError, ValueError):
                    pass
    except Exception as exc:  # sayfa akisi bozulmasin
        log.warning("score_table: journal okunamadi: %s", exc)
        error = str(exc)

    total = _score_health(rows, parts_list, lim)
    for f in filters:
        cnt = stage_counts.get(f["stage"], 0) if f["stage"] else None
        f["count"] = cnt
        if f["stage"] is None:
            f["health"] = _h("na", "Ölçülemiyor", "Journal bu kuralı ayrıca kaydetmiyor.")
        elif not f["on"]:
            f["health"] = (_h("bad", "Hatalı", f"Bu stratejide kapalı olmalıydı ama {cnt} setup eledi.") if cnt
                           else _h("ok", "Doğru çalışıyor", "Bu stratejide kapalı; beklendiği gibi hiç setup elemedi."))
        elif cnt:
            f["health"] = _h("ok", "Doğru çalışıyor", f"Son 7 günde {cnt} setup eledi.")
        else:
            f["health"] = _h("na", "Yetersiz veri", "Son 7 günde bu filtreye takılan setup olmadı.")

    issues = (
        sum(1 for r in rows if r["health"]["level"] == "bad")
        + sum(1 for f in filters if f["health"]["level"] == "bad")
        + (1 if total["level"] == "bad" else 0) + len(missing)
    )
    warns = sum(1 for r in rows if r["health"]["level"] == "warn")
    return {
        "tabs": TABS,
        "strategy": strategy,
        "rows": rows,
        "filters": filters,
        "limits": lim,
        "total": total,
        "missing_keys": missing,
        "since": since,
        "n_setups": len(parts_list),
        "issues": issues,
        "warns": warns,
        "error": error,
    }
