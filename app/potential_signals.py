"""Potential Signals sayfasi (24.10, kullanici istegi): kapida elenip Telegram'a "POTANSIYEL" olarak bildirilen 1D/1W
setuplar (`scanner._notify_gated_potential`, tablo `potential_notices`). Varsayilan bu hafta (Pazartesi 00:00 TSI);
iptal olan satir silinmez, DURUM sutununda "Cancelled" yazar.

Kaynaklar (ayni anahtar: strateji + sembol + yon + purge_time):
- `potential_notices`: bildirim zinciri (durum, kapi, iptal nedeni, bildirim anindaki seviyeler -- 24.10'dan beri).
- `setup_journal`: sonuc -- kullanici kurali, TP once mi SL once mi (giris dolmadan TP de TP). Seviyesi olmayan eski
  bildirimde seviyeler de buradan.
- `signals`: zincir sinyale dondu ise o sinyalin bugunku durumu (bekliyor / aktif / kapandi).
Motor kararina dokunmaz; yalniz okur.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PotentialNotice, SetupJournal, Signal

log = logging.getLogger(__name__)

TSI = timezone(timedelta(hours=3))
PAGE_SIZE = 50

# Motorun setup'i neden almadigi (kapi). scanner._GATED_NOTIFY_GATES ile ayni anahtarlar.
GATE_LABELS = {
    "bias_mismatch": "1D bias opposite",
    "low_quality": "Score below threshold",
    "score7_gate": "Score-7 gate (structural entry / key level)",
    "cluster_limit": "Same-direction cluster limit",
    "has_open": "Open signal on this symbol",
    "corr_open": "Correlated pair has an open signal",
    "duplicate": "Setup already saved",
    "same_setup": "Same setup already traded on another coin",
    "deleted_before": "Deleted before (sticky)",
    "no_cisd": "Waiting for CISD",
    "low_rr": "R:R below 2",
    "target_taken": "Target already taken",
}
CANCEL_LABELS = {
    "crt_gone": "CRT structure broke",
    "crt_replaced": "Replaced by a newer CRT",
    "expired": "No CISD in time",
    "target_taken": "Target already taken", "past_tp": "Target already taken", "near_tp": "Target almost taken",
    "past_sl": "Price passed SL", "stale": "Stale retest", "week_gap": "Weekend gap", "invalidated": "CRT 60% crossed",
    "missed_quality": "Score low at retest", "liq_unswept": "Liquidity not swept", "tight_stop": "Stop too tight",
}
# Durum anahtari -> (etiket, renk). Renk siniflari sablonun basinda acik yazili (Tailwind tarar).
STATUS_META = {
    "watch_open": ("Watching · C2 open", "yellow"),
    "watch_closed": ("Watching · C2 closed", "yellow"),
    "signal_waiting": ("Signal · waiting entry", "blue"),
    "signal_active": ("Signal · active", "blue"),
    "signal_closed": ("Signal · closed", "green"),
    "signal_cancelled": ("Cancelled (signal deleted)", "red"),
    "cancelled": ("Cancelled", "red"),
    # 11.11 yeni zincir (scanner._potential_chain_step)
    "pot_crt": ("CRT · waiting CISD", "yellow"),
    "pot_wait": ("Waiting entry", "yellow"),
    "pot_fill": ("Entry touched", "blue"),
    "pot_done": ("Finished", "green"),
    "pot_lowq": ("Low quality", "gray"),
}
STATUS_FILTERS = (("all", "All"), ("watch", "Watching"), ("signal", "Became signal"), ("cancelled", "Cancelled"))
# 11.11: motorun %50 kismi kar + BE kurali -- kismi kar alinip kalan girise dondu.
RESULT_META = {
    "tp": ("TP", "green"),
    "tp_nofill": ("TP (no fill)", "green"),
    "be50": ("50% + BE", "yellow"),
    "sl": ("SL", "red"),
    "expired": ("Expired", "gray"),
    "filled": ("Filled · open", "blue"),
    "open": ("Open", "gray"),
}
# Filtre secenekleri (01.11, kullanici istegi). Kripto disi hepsi FX (Radar ile ayni).
# 11.11 (kullanici): market filtresi coklu secim -- bos secim = hepsi. Eski tek deger (all / crypto / fx) da okunur.
MARKET_FILTERS = (("crypto", "Crypto"), ("fx", "FX"), ("metal", "Metal"), ("index", "Index"), ("oil", "Oil"))


def _market_key(symbol: str) -> str:
    if _is_crypto(symbol):
        return "crypto"
    try:
        from app.exchange import from_display_symbol, market_of
        m = market_of(from_display_symbol(symbol)[0])
        return m if m in dict(MARKET_FILTERS) else "fx"
    except Exception:
        return "fx"


def norm_markets(market) -> list[str]:
    """Sorgu degeri (tek str ya da liste) -> gecerli market listesi; 'all' / bos = []. Sira MARKET_FILTERS'a gore."""
    vals = [market] if isinstance(market, str) else list(market or [])
    keep = {v for v in vals if v in dict(MARKET_FILTERS)}
    return [k for k, _ in MARKET_FILTERS if k in keep]
DIRECTION_FILTERS = (("all", "All"), ("long", "LONG"), ("short", "SHORT"))
RESULT_FILTERS = (("all", "All"), ("tp", "TP"), ("tp_nofill", "TP (no fill)"), ("be50", "50% + BE"), ("sl", "SL"),
                  ("filled", "Filled · open"), ("open", "Open"), ("expired", "Expired"))


def _is_crypto(symbol: str) -> bool:
    return (symbol or "").upper().endswith("USDT.P")


def week_start_tsi(now: datetime | None = None) -> str:
    """Bu haftanin Pazartesi'si (TSI) -- Signals sayfasiyla ayni varsayilan."""
    n = (now or datetime.now(timezone.utc)).astimezone(TSI)
    return (n - timedelta(days=n.weekday())).strftime("%Y-%m-%d")


def _tsi_day(value: str, end: bool = False) -> datetime | None:
    try:
        d = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=TSI)
    except (TypeError, ValueError):
        return None
    if end:
        d += timedelta(days=1)
    return d.astimezone(timezone.utc).replace(tzinfo=None)


def _key(strategy, symbol, direction, purge_time) -> tuple:
    return (strategy, symbol, direction, str(purge_time)[:19] if purge_time is not None else None)


def _fmt_tsi(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v[:19])
        except ValueError:
            return v
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.astimezone(TSI).strftime("%d.%m %H:%M")


def _result(row, notice=None) -> str:
    """Setup Journal sonucu, TP once mi SL once mi (sinyale donmus setupta golge k=1 -- score_outcome ile ayni).

    11.11 (kullanici): motorun %50 kismi kar + BE kurali -- golge k=1 kismi kari tetikleyip (`pa`) kalan yariyi girişte
    kapattiysa (`cur` = be) sonuc "50% + BE"; eskiden duz TP/SL oldugu icin SL (ya da TP) yaziliyordu."""
    if row is None:
        return "open"
    import json
    from app.score_outcome import outcome_of
    o, _rr = outcome_of(row)
    # 11.11 (XAGUSD 1D 10.11): Journal 1D/1W seviyeleri C2 kapanisinda / sinyalde yeniden donuyor, eski seviyelerin
    # sonucu `pre_c2`'de kalir. Bildirim o eski seviyelerle gittiyse sonuc oradan (yoksa kazanan satir "Open" gorunuyordu).
    try:
        pre = json.loads(row["pre_c2"]) if row["pre_c2"] else None
    except Exception:
        pre = None
    if (pre and notice is not None and notice.entry is not None and pre.get("entry") is not None
            and abs(float(pre["entry"]) - float(notice.entry)) < 1e-9
            and (notice.sl is None or abs(float(pre.get("sl") or 0) - float(notice.sl)) < 1e-9)):
        po = pre.get("o") or pre.get("outcome")
        if po in ("win", "loss", "tp_before_entry", "filled"):
            o = po
            return {"win": "tp", "loss": "sl", "tp_before_entry": "tp_nofill", "filled": "filled"}[o]
    try:
        sh = (json.loads(row["shadow"]) if row["shadow"] else {}).get("1") or {}
    except Exception:
        sh = {}
    if sh.get("pa") and sh.get("cur") == "be" and o in ("win", "loss"):
        return "be50"
    if o == "win":
        return "tp"
    if o == "tp_before_entry":
        return "tp_nofill"
    if o == "loss":
        return "sl"
    if o == "filled":
        return "filled"
    return "open"


def _replay_notice(n, store) -> tuple[str, float | None] | None:
    """11.11 (GBPNZD 1D 11.11: sayfa SL diyordu, fiyat bildirimdeki SL'ye hic gitmemisti): ESKI zincir satirinin sonucu
    BILDIRIMDEKI seviyelerle, store'daki kapanmis LTF mumlarindan -- yeni zincirle ayni kural (`scanner.pot_step_bar`:
    %50'de yarisi + kalan BE, dolum mumunda yalniz SL). Journal'in seviyeleri bildirimdekinden farkli olabiliyor
    (1D C2 acikken donan kosan SL). Store yoksa / bildirim anini kapsamiyorsa None -> Journal'a dusulur."""
    if store is None or n.entry is None or n.sl is None or n.tp is None or n.created_at is None:
        return None
    try:
        from types import SimpleNamespace
        from app.exchange import from_display_symbol
        from app.scanner import POT_HORIZON, POT_LTF, pot_bars, pot_step_bar
        raw, _ = from_display_symbol(n.symbol)
        ltf_name = "4h" if n.strategy == "1w" else "1h"
        df = store.get_df(raw, ltf_name)
        if df is None or len(df) < 2:
            return None
        ltf, horizon = POT_LTF.get(n.strategy, timedelta(hours=1)), POT_HORIZON.get(n.strategy, timedelta(hours=120))
        start = n.created_at.replace(tzinfo=None) if n.created_at.tzinfo is None else             n.created_at.astimezone(timezone.utc).replace(tzinfo=None)
        first = df.index[0].to_pydatetime().replace(tzinfo=None)
        if first > start:
            return None   # store bildirim anini kapsamiyor
        t = SimpleNamespace(state="wait", direction=n.direction, entry=n.entry, sl=n.sl, tp=n.tp, rr=n.rr,
                            filled_at=None, partial_at=None, result=None, result_r=None, result_at=None)
        last_close = None
        for bts, hi_, lo_ in pot_bars(df, start, ltf, until=start + horizon):   # 11.11: yalniz pencere (hiz)
            close_at = (bts + ltf).replace(tzinfo=None)
            pot_step_bar(t, bts, hi_, lo_, ltf)
            last_close = close_at
            if t.state == "done":
                return t.result, t.result_r
        if last_close is not None and last_close >= start + horizon - ltf:
            return "expired", None
        return ("filled", None) if t.state == "fill" else ("open", None)
    except Exception as exc:
        log.debug("potential_signals: replay failed for %s: %s", getattr(n, "symbol", "?"), exc)
        return None


def _chain_result(n) -> str | None:
    """Yeni zincirin kendi sonucu (mesajdaki seviyeler, scanner._track_potential_bar)."""
    if n.result:
        return n.result
    if n.state == "fill":
        return "filled"
    if n.state in ("wait", "crt"):
        return "open"
    return None


def _signal_status(sig) -> str:
    st = (getattr(sig, "status", None) or "").lower()
    if st == "waiting_entry" or st == "pending_cisd":
        return "signal_waiting"
    if st == "active":
        return "signal_active"
    return "signal_closed"


async def build_potential_signals(
    db: AsyncSession, *, tf: str = "all", date_from: str = "", date_to: str = "", status: str = "all",
    market="all", direction: str = "all", result: str = "all", page: int = 1, store=None,
) -> dict:
    tf = tf if tf in ("all", "1d", "1w") else "all"
    status = status if status in dict(STATUS_FILTERS) else "all"
    markets = norm_markets(market)
    direction = direction if direction in dict(DIRECTION_FILTERS) else "all"
    result = result if result in dict(RESULT_FILTERS) else "all"
    q = select(PotentialNotice)
    if tf != "all":
        q = q.where(PotentialNotice.strategy == tf)
    dfrom, dto = _tsi_day(date_from), _tsi_day(date_to, end=True)
    if dfrom is not None:
        q = q.where(PotentialNotice.created_at >= dfrom)
    if dto is not None:
        q = q.where(PotentialNotice.created_at < dto)
    q = q.order_by(PotentialNotice.created_at.desc())
    error = None
    try:
        notices = (await db.execute(q)).scalars().all()
    except Exception as exc:   # sayfa akisi bozulmasin
        log.warning("potential_signals: okunamadi: %s", exc)
        notices, error = [], str(exc)

    # Journal + Signal satirlari: sembol bazinda tek sorgu, eslestirme Python'da (purge_time bicimi tablolar arasi farkli).
    syms = sorted({n.symbol for n in notices})
    jmap: dict = {}
    smap: dict = {}
    if syms:
        try:
            t = SetupJournal.__table__
            for r in (await db.execute(select(t).where(t.c.symbol.in_(syms)))).mappings().all():
                jmap[_key(r["strategy"], r["symbol"], r["direction"], r["purge_time"])] = r
            for s in (await db.execute(select(Signal).where(Signal.symbol.in_(syms)))).scalars().all():
                smap[_key(s.timeframe or "4h", s.symbol, s.direction, s.purge_time)] = s
        except Exception as exc:
            log.warning("potential_signals: journal/signal okunamadi: %s", exc)

    rows = []
    for n in notices:
        k = _key(n.strategy, n.symbol, n.direction, n.purge_time)
        j, sig = jmap.get(k), smap.get(k)
        new_chain = n.state in ("crt", "wait", "fill", "done", "lowq")
        if n.cancelled:
            st = "cancelled"
            why = CANCEL_LABELS.get(n.cancel_reason or "", n.cancel_reason or "")
            # 11.11: ayni C2'nin TERS yonlu satiri varsa "yeni CRT" degil -- C2 C1'in obur ucunu (bu setup'in TP'si) aldi,
            # motor ayni mumu ters yonde okudu (XAGUSD 1D 10.11). Eski zincir bunu crt_replaced diye iptal ediyordu.
            opp = "LONG" if n.direction == "SHORT" else "SHORT"
            if n.cancel_reason == "crt_replaced" and _key(n.strategy, n.symbol, opp, n.purge_time) in jmap:
                why = "Target reached (C2 swept the other C1 end)"
        elif n.state == "signal" or sig is not None:
            if sig is not None:
                st, why = _signal_status(sig), ""
            else:
                st = "signal_cancelled"
                why = (j["deleted_reason"] if j is not None and j["deleted_reason"] else "")
        elif new_chain:
            st, why = "pot_" + n.state, ""
        else:
            st = "watch_closed" if (n.state or "").startswith("closed") else "watch_open"
            why = ""
        label, color = STATUS_META[st]
        res, res_r = (_chain_result(n) if new_chain and n.state != "lowq" else None), n.result_r
        if res is None:
            rp = _replay_notice(n, store)
            res, res_r = rp if rp is not None else (_result(j, n), None)
        # CISD: zincirin kaydi (11.11+), sinyalinki, yoksa Journal'in ilk CISD kirilimi (retrace.brk); CRT asamasinda bos
        cisd = n.cisd_time or (getattr(sig, "cisd_time", None) if sig is not None else None)
        if cisd is None and j is not None and j["retrace"] and n.state != "crt":
            try:
                import json
                b = (json.loads(j["retrace"]) or {}).get("brk") or {}
                cisd = datetime.fromisoformat(b["cisd"]) if b.get("cisd") else None
            except Exception:
                cisd = None
        rlabel, rcolor = RESULT_META[res]
        entry = n.entry if n.entry is not None else (j["entry"] if j is not None else None)
        sl = n.sl if n.sl is not None else (j["sl"] if j is not None else None)
        tp = n.tp if n.tp is not None else (j["tp"] if j is not None else None)
        rr = n.rr if n.rr is not None else (j["rr"] if j is not None else None)
        score = n.score if n.score is not None else (j["score"] if j is not None else None)
        rows.append(dict(
            id=n.id, strategy=n.strategy, tf_label={"1d": "1D-1H", "1w": "1W-4H"}.get(n.strategy, n.strategy.upper()),
            symbol=n.symbol, direction=n.direction, score=int(score) if score is not None else None,
            entry=entry, sl=sl, tp=tp, rr=rr, model=(n.model or (j["model"] if j is not None else None) or "").upper(),
            gate=GATE_LABELS.get(n.gate or "", n.gate or "—"),
            status=st, status_label=label, status_color=color, status_why=why,
            result=res, result_label=rlabel, result_color=rcolor,
            notified=_fmt_tsi(n.created_at), updated=_fmt_tsi(n.updated_at),
            cisd=_fmt_tsi(cisd) if cisd is not None else "",
            result_r=res_r,
            c1=_fmt_tsi(n.crt_bar_time), c2=_fmt_tsi(n.purge_time),
            telegram=bool(n.tg_message_id),
        ))

    def _group(st: str) -> str:
        return "cancelled" if st in ("cancelled", "signal_cancelled") else ("signal" if st.startswith("signal") else "watch")

    # Market / yon her seye uyar; durum sekmesi sayaclari sonuc filtresine, sonuc ozeti durum sekmesine uyar.
    if markets:
        rows = [r for r in rows if _market_key(r["symbol"]) in markets]
    if direction != "all":
        rows = [r for r in rows if (r["direction"] or "").lower() == direction]
    by_result = rows if result == "all" else [r for r in rows if r["result"] == result]
    by_status = rows if status == "all" else [r for r in rows if _group(r["status"]) == status]
    counts = {"all": len(by_result)}
    for key, _ in STATUS_FILTERS[1:]:
        counts[key] = sum(1 for r in by_result if _group(r["status"]) == key)
    results = {k: sum(1 for r in by_status if r["result"] == k) for k in RESULT_META}
    shown = [r for r in by_status if result == "all" or r["result"] == result]
    # En yuksek skor ustte (01.11, kullanici); skorsuz en altta, esitlikte en yeni bildirim (sorgu sirasi korunur).
    shown.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1))
    pages = max(1, (len(shown) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), pages)
    return dict(
        rows=shown[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], total=len(shown), page=page, pages=pages,
        counts=counts, results=results, tf=tf, status=status, date_from=date_from, date_to=date_to,
        markets=markets, direction=direction, result=result, status_filters=STATUS_FILTERS,
        market_filters=MARKET_FILTERS, direction_filters=DIRECTION_FILTERS, result_filters=RESULT_FILTERS, error=error,
    )
