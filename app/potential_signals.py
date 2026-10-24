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
}
CANCEL_LABELS = {
    "crt_gone": "CRT structure broke",
    "crt_replaced": "Replaced by a newer CRT",
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
}
STATUS_FILTERS = (("all", "All"), ("watch", "Watching"), ("signal", "Became signal"), ("cancelled", "Cancelled"))
RESULT_META = {
    "tp": ("TP", "green"),
    "sl": ("SL", "red"),
    "filled": ("Filled · open", "blue"),
    "open": ("Open", "gray"),
}


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


def _result(row) -> str:
    """Setup Journal sonucu, TP once mi SL once mi (sinyale donmus setupta golge k=1 -- score_outcome ile ayni)."""
    if row is None:
        return "open"
    from app.score_outcome import outcome_of
    o, _rr = outcome_of(row)
    if o in ("win", "tp_before_entry"):
        return "tp"
    if o == "loss":
        return "sl"
    if o == "filled":
        return "filled"
    return "open"


def _signal_status(sig) -> str:
    st = (getattr(sig, "status", None) or "").lower()
    if st == "waiting_entry" or st == "pending_cisd":
        return "signal_waiting"
    if st == "active":
        return "signal_active"
    return "signal_closed"


async def build_potential_signals(
    db: AsyncSession, *, tf: str = "all", date_from: str = "", date_to: str = "", status: str = "all",
    page: int = 1,
) -> dict:
    tf = tf if tf in ("all", "1d", "1w") else "all"
    status = status if status in dict(STATUS_FILTERS) else "all"
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
        if n.cancelled:
            st = "cancelled"
            why = CANCEL_LABELS.get(n.cancel_reason or "", n.cancel_reason or "")
        elif n.state == "signal" or sig is not None:
            if sig is not None:
                st, why = _signal_status(sig), ""
            else:
                st = "signal_cancelled"
                why = (j["deleted_reason"] if j is not None and j["deleted_reason"] else "")
        else:
            st = "watch_closed" if (n.state or "").startswith("closed") else "watch_open"
            why = ""
        label, color = STATUS_META[st]
        res = _result(j)
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
            c1=_fmt_tsi(n.crt_bar_time), c2=_fmt_tsi(n.purge_time),
            telegram=bool(n.tg_message_id),
        ))

    def _group(st: str) -> str:
        return "cancelled" if st in ("cancelled", "signal_cancelled") else ("signal" if st.startswith("signal") else "watch")

    counts = {"all": len(rows)}
    for key, _ in STATUS_FILTERS[1:]:
        counts[key] = sum(1 for r in rows if _group(r["status"]) == key)
    results = {k: sum(1 for r in rows if r["result"] == k) for k in RESULT_META}
    shown = rows if status == "all" else [r for r in rows if _group(r["status"]) == status]
    pages = max(1, (len(shown) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), pages)
    return dict(
        rows=shown[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], total=len(shown), page=page, pages=pages,
        counts=counts, results=results, tf=tf, status=status, date_from=date_from, date_to=date_to,
        status_filters=STATUS_FILTERS, error=error,
    )
