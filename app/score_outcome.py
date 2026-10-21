"""Skor kalemi basina SONUC: bu puani alan setuplar TP'ye mi SL'ye mi gitti? (04.10, kullanici: Score Table "efektif
calismali ve bize bilgi vermeli").

`scripts/score_item_outcome_stat.py` (terminal) ve `app/score_table.py` (/skor-tablosu) AYNI fonksiyonlari kullanir --
iki yerde ayri mantik olmasin. Saf hesap; DB okumaz, cagiran Journal satirlarini (dict / sqlite3.Row) verir.

Kaynak Setup Journal: motorun CRT saydigi her setup, kapidan gecsin gecmesin (ceza alan setup skor kapisinda elenir
ama Journal sonrasini izler -- soru tam olarak "elenmeseydi ne olurdu"). Sonuc `outcome` (secilen giris, DUZ TP/SL:
kismi kar + BE yok); sinyale donustuyse `shadow["1"]`. 4H'te yalniz C2 kapaliyken donmus satirlar, 1D/1W'de yalniz
seviyesi kesin satirlar (`levels_final`); skor `analysis_parts`.

HUKUM yalniz KARAR BANDINDAN (01.10): kalemin sinyali degistirdigi setuplar -- bu kalem HARIC skor oyle ki odul onu
esigin ustune tasir, ceza altina iter; ayni bantta puan alan vs almayan, taraf basina >= MIN_FLIP cozulmus. Ham fark
yaniltir (kalemler birbirine bagli: ceza alan setup zaten dusuk skorlu).
"""
from __future__ import annotations

import json

from app.setup_journal import PRE_SETUP_STAGES, analysis_parts, levels_final

BAND = 0.15            # karar bandi farki (R/setup)
MIN_FLIP = 10          # karar bandinda taraf basina cozulmus
MIN_TOTAL = 20         # toplam skor sirasi: taraf basina cozulmus
THR = {"4h": 7, "1d": 7, "1w": 6}
BANDS = ((0, 4), (5, 6), (7, 99))


def _get(row, k):
    try:
        return row[k]
    except (KeyError, IndexError):
        return None


def outcome_of(row) -> tuple[str | None, float]:
    """(sonuc, RR): sinyale donusen setupta golge k=1 (duz TP/SL), yoksa ana izleme."""
    o, rr = _get(row, "outcome"), None
    sh_raw = _get(row, "shadow")
    if o == "signal" and sh_raw:
        sh = (json.loads(sh_raw) or {}).get("1") or {}
        o, rr = sh.get("o"), sh.get("rr")
    ent = _get(row, "entries")
    if rr is None and ent:
        rr = ((json.loads(ent) or {}).get("chosen") or {}).get("rr")
    return o, (rr if rr is not None else (_get(row, "rr") or 0.0))


def collect(rows, strategy: str | None = None, all_levels: bool = False) -> list[dict]:
    """Journal satirlarindan analiz kumesi: CRT-oncesi aday yok, seviyesi kesin, skor kirilimi olan."""
    out = []
    for r in rows:
        st = _get(r, "strategy")
        if _get(r, "best_stage") in PRE_SETUP_STAGES or st not in THR:
            continue
        if strategy and st != strategy:
            continue
        if _get(r, "parts_at_levels") is None or _get(r, "entry") is None:
            continue
        if st == "4h":
            if (json.loads(_get(r, "parts_at_levels")) or {}).get("c2_closed") != 1:
                continue
        elif not all_levels and not levels_final(r):
            continue
        pal, fe = analysis_parts(r)
        if not pal:
            continue
        o, rr = outcome_of(r)
        sym, mk = _get(r, "symbol") or "", (_get(r, "market_type") or "").lower()
        seg = "fx" if mk != "crypto" else ("major" if sym in ("BTCUSDT.P", "ETHUSDT.P") else "alt")
        out.append(dict(s=st, pal=pal, o=o, rr=rr or 0.0, seg=seg, smt_ok=fe.get("smt_possible"),
                        first_seen=_get(r, "first_seen"), cand=candidate_values(r, pal, fe)))
    return out


def stats(sub) -> dict:
    w = [x for x in sub if x["o"] == "win"]
    l_ = [x for x in sub if x["o"] == "loss"]
    res = len(w) + len(l_)
    return dict(n=len(sub), res=res, w=len(w), l=len(l_),
                tpb=sum(x["o"] == "tp_before_entry" for x in sub),
                op=sum(x["o"] in ("open", "pending", "filled", "no_touch", "ambiguous", None) for x in sub),
                wr=(len(w) / res * 100) if res else None,
                rps=((sum(x["rr"] for x in w) - len(l_)) / res) if res else None)


def flip_diff(sub, k, v, thr):
    """KARAR BANDI: odul (v > 0) bu kalem haric skor [thr - v, thr - 1]; ceza (v < 0) [thr, thr - v - 1].
    Ayni bantta deger = v olanlar ile 0 olanlar. (got, base, fark, (lo, hi))."""
    lo, hi = (thr - v, thr - 1) if v > 0 else (thr, thr - v - 1)
    band = [x for x in sub if lo <= (x["pal"].get("raw", 0) - (x["pal"].get(k) or 0)) <= hi]
    g = stats([x for x in band if x["pal"].get(k) == v])
    b = stats([x for x in band if x["pal"].get(k) == 0])
    d = (g["rps"] - b["rps"]) if g["res"] and b["res"] else None
    return g, b, d, (lo, hi)


def controlled_diff(sub, k, v):
    """Bu kalem HARIC skor bandinda (puan alan - almayan) R/setup farki, puan alanin cozulmusuyle agirlikli."""
    num = w = 0.0
    for lo, hi in BANDS:
        band = [x for x in sub if lo <= (x["pal"].get("raw", 0) - (x["pal"].get(k) or 0)) <= hi]
        g = stats([x for x in band if x["pal"].get(k) == v])
        b = stats([x for x in band if x["pal"].get(k) == 0])
        if g["res"] and b["res"]:
            num += g["res"] * (g["rps"] - b["rps"])
            w += g["res"]
    return (num / w) if w else None


def judge(fg, fb, fd, penalty: bool) -> tuple[str, str]:
    """(seviye, hukum) -- seviye: ok / warn / bad / na. Odulde puan alan iyi olmali, cezada kotu."""
    if fg["res"] < MIN_FLIP or fb["res"] < MIN_FLIP or fd is None:
        return "na", "veri az"
    good = -fd if penalty else fd
    if good >= BAND:
        return "ok", "kazananı ayırıyor"
    if good <= -BAND:
        return "bad", "TERS"
    return "warn", "fark yok"


def verdict(fg, fb, fd, penalty: bool) -> str:
    """Terminal raporu icin metin (eski bicim)."""
    lvl, txt = judge(fg, fb, fd, penalty)
    if lvl == "na":
        return f"veri az (karar bandi {fg['res']}/{fb['res']}, taraf basina {MIN_FLIP})"
    word = {"ok": "DOGRU CALISIYOR", "bad": "TERS", "warn": "fark yok"}[lvl]
    return f"{word} (karar bandi fark {fd:+.2f})"


def item_outcomes(sub, key: str, thr: int) -> dict:
    """Bir kalemin her degeri icin sonuc + karar bandi hukmu (0'a karsi). Sayfa satiri icin."""
    vals = sorted({x["pal"].get(key) for x in sub if x["pal"].get(key) is not None}, reverse=True)
    by = {v: stats([x for x in sub if x["pal"].get(key) == v]) for v in vals}
    verdicts = []
    if len(vals) > 1 and 0 in by:
        for v in vals:
            if v == 0:
                continue
            fg, fb, fd, band = flip_diff(sub, key, v, thr)
            lvl, txt = judge(fg, fb, fd, v < 0)
            verdicts.append(dict(v=v, level=lvl, text=txt, d=fd, g=fg, b=fb, band=band))
    return dict(vals=vals, by=by, verdicts=verdicts, single=len(vals) == 1)


def total_ranking(sub, thr: int) -> dict:
    """Toplam skor kazanani siraliyor mu: skor >= esik vs < esik (duz TP/SL R/setup)."""
    hi = stats([x for x in sub if (x["pal"].get("score") or 0) >= thr])
    lo = stats([x for x in sub if (x["pal"].get("score") or 0) < thr])
    d = (hi["rps"] - lo["rps"]) if hi["res"] and lo["res"] else None
    if hi["res"] < MIN_TOTAL or lo["res"] < MIN_TOTAL or d is None:
        lvl, txt = "na", "veri az"
    elif d >= BAND:
        lvl, txt = "ok", "sıralıyor"
    elif d <= -BAND:
        lvl, txt = "bad", "TERS sıralıyor"
    else:
        lvl, txt = "warn", "sıralamıyor"
    return dict(hi=hi, lo=lo, d=d, level=lvl, text=txt)


# ---------------------------------------------------------------------------------------------------------------
# YENI ADAY KALEMLER (21.10, kullanici: "Skor tablosu yeni adaylar" -- tek yerden takip). Bugun skorda OLMAYAN
# kalemler; ayni karar bandi yontemiyle "bu puan verilseydi kazanani ayirir miydi?" sorulur. Motor karari degismez.
# Her aday: (anahtar, etiket, onerilen puan, kaynak, filtre karari hangi maddede -- yoksa None). Puan ve esikler
# ILK OKUMADAN ONCE yazildi (21.10); okumada ayar yapilmaz. Filtre sorusu (>= 0.3 R) eski maddesinde kalir, ceza
# kademesi buraya devredilir. Karar kurali: IZLEME.md "Skor tablosu yeni adaylar — 4H / 1D / 1W".
CANDIDATES = (
    ("session_kz", "Dolum London / NY AM seansında", +1, "entries.chosen.t → session.trading_session", None),
    ("cisd_disp", "CISD onay mumu güçlü (gövde ≥ 1.0 × ort. LTF mum)", +1, "features.cisd_disp (21.10+)", None),
    ("liq_near", "Purge ötesinde ≤ 0.5 ATR alınmamış likidite", +1, "features.liq_beyond_atr", None),
    ("eq_ssl", "Puansız key level (yalnız EQ / SSL-BSL)", +1, "features.key_level_n > 0 + pd_* = 0",
     "Key level teması yoksa işlem açılmasın mı?"),
    ("mss_broken", "MSS de kırılmış (dolumdan önce)", +1, "retrace.brk.mss < entries.chosen.t (29.09+)",
     "CISD onayı mı, CISD + MSS onayı mı?"),
    ("c1_big", "C1 aralığı büyük (≥ 1.3 × HTF ATR)", -1, "features.range_atr", None),
    ("no_key_level", "Key level teması yok", -2, "features.key_level_n = 0",
     "Key level teması yoksa işlem açılmasın mı?"),
    ("c1_tp_wick", "TP, C1'in uzun fitilinde (≥ %40)", -2, "features.c1_tp_wick_frac", "C1'in şekli"),
    # 21.10: "Bolgesiz setup" maddesi kapandi, filtre sorusu da burada -- karar bandinda >= FILTER_GAP kotu ise
    # puan cezasinin yaninda "bolge sarti" secenegi de sunulur.
    ("no_zone", "IFVG / BPR bölgesi oluşmamış", -2, "entries.bpr_near / ifvg_near", None),
    # 21.10 (kullanici, "Limit giris seviyesi — 4H" kapanisi): kapilari gecmis 4H BPR girisli setup 18 dolumda 1 TP / 17 SL,
    # hicbir giris varyanti kurtarmiyor -> setup'in kendisi mi kotu? Donmus secilen giris = BPR'nin yakin kenari.
    ("bpr_entry", "Giriş BPR'den (IFVG + FVG kesişimi)", -2, "entries.chosen.e = entries.bpr_near.e", None),
)
CAND_POINTS = {k: v for k, _l, v, _s, _f in CANDIDATES}
CAND_MIN_RR = 2.0      # karar kumesi: islem acilabilir dilim (C1 sekli dersi, 21.10)
CAND_MIN_SIDE = 30     # tetik: karar bandinda taraf basina cozulmus
CISD_DISP_MIN = 1.0
FILTER_GAP = 0.30     # eski maddesi kapanan adayda (no_zone) filtre secenegi esigi
FILTER_HERE = ("no_zone",)
KILLZONES = ("london", "ny_am")


def _jl(v):
    if isinstance(v, dict):
        return v
    try:
        d = json.loads(v) if v else None
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def candidate_values(row, pal: dict, fe: dict) -> dict:
    """Adaylarin bu setuptaki degeri: 1 = var, 0 = yok, None = olculemiyor (karsilastirmaya girmez)."""
    out: dict = {}
    ents = _jl(_get(row, "entries")) or {}
    ch = ents.get("chosen") or {}
    fill = ch.get("t")
    try:
        if fill:
            from datetime import datetime
            from app.session import trading_session
            ses = trading_session(datetime.fromisoformat(str(fill)[:19]))
            out["session_kz"] = 1 if (ses and ses.get("key") in KILLZONES) else 0
    except Exception:
        pass
    v = fe.get("cisd_disp")
    if isinstance(v, (int, float)):
        out["cisd_disp"] = 1 if v >= CISD_DISP_MIN else 0
    kln = fe.get("key_level_n")
    if kln is not None:
        la = fe.get("liq_beyond_atr")
        out["liq_near"] = 1 if (isinstance(la, (int, float)) and la <= 0.5) else 0
        scored = sum((pal.get(k) or 0) for k in ("pd_major", "pd_monthly", "pd_struct")) > 0
        out["eq_ssl"] = 1 if (kln > 0 and not scored) else 0
        out["no_key_level"] = 1 if kln == 0 else 0
    ra = fe.get("range_atr")
    if isinstance(ra, (int, float)):
        out["c1_big"] = 1 if ra >= 1.3 else 0
    tw = fe.get("c1_tp_wick_frac")
    if isinstance(tw, (int, float)):
        out["c1_tp_wick"] = 1 if tw >= 0.40 else 0
    brk = (_jl(_get(row, "retrace")) or {}).get("brk")
    if isinstance(brk, dict) and fill:
        m = brk.get("mss")
        out["mss_broken"] = 1 if (m and str(m)[:19] < str(fill)[:19]) else 0
    if ents:
        out["no_zone"] = 0 if ("bpr_near" in ents or "ifvg_near" in ents) else 1
    ce, be = ch.get("e"), (ents.get("bpr_near") or {}).get("e")
    if isinstance(ce, (int, float)):
        out["bpr_entry"] = 1 if (isinstance(be, (int, float)) and abs(ce - be) <= 1e-9 * max(1.0, abs(ce))) else 0
    return out


def cand_flip(sub, key: str, thr: int):
    """KARAR BANDI (aday): adayin onerilen puani sinyali degistirecek setuplar -- odulde bugunku raw [thr - v, thr - 1],
    cezada [thr, thr - v - 1]; ayni bantta aday var (1) / yok (0). Yalniz RR >= CAND_MIN_RR. (var, yok, fark, bant)."""
    v = CAND_POINTS[key]
    lo, hi = (thr - v, thr - 1) if v > 0 else (thr, thr - v - 1)
    band = [x for x in sub if x["rr"] >= CAND_MIN_RR and x["cand"].get(key) is not None
            and lo <= (x["pal"].get("raw") or 0) <= hi]
    g = stats([x for x in band if x["cand"][key] == 1])
    b = stats([x for x in band if x["cand"][key] == 0])
    d = (g["rps"] - b["rps"]) if g["res"] and b["res"] else None
    return g, b, d, (lo, hi), band


def cand_progress(sub, thr: int) -> list[tuple[str, int]]:
    """Aday basina karar bandinda AZ olan tarafin cozulmus sayisi (tetik CAND_MIN_SIDE), en ileri once."""
    out = []
    for key, *_ in CANDIDATES:
        g, b, _d, _band, _x = cand_flip(sub, key, thr)
        out.append((key, min(g["res"], b["res"])))
    return sorted(out, key=lambda kv: -kv[1])
