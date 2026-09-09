"""Telegram bildirim modülü – CRT sinyallerini Telegram'a gönderir."""

from __future__ import annotations

import logging
import math
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Optional

import httpx

from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

if TYPE_CHECKING:
    from app.models import Signal

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Giris modeli etiketleri. Eski "IFVG 50%" yanlisti: giris artik bolgenin
# ortasindan degil, girise en yakin kenarindan yapiliyor. MSS de artik CISD'den
# ayri raporlaniyor.
_ENTRY_MODEL_LABELS = {
    "ifvg": "IFVG",
    "cisd": "CISD",
    "mss": "MSS",
}


def _strategy_label(sig: "Signal") -> str:
    tf = getattr(sig, "timeframe", None) or "4h"
    if tf == "1d":
        return "1D-1H"
    if tf == "1h":
        return "1H-5M"
    return "4H-15M"


def _format_active_signal(sig: Signal) -> str:
    direction_emoji = "\U0001f7e2" if sig.direction == "LONG" else "\U0001f534"
    # Bias satirlari bilgi amacli; 1W filtre olarak KULLANILMAZ.
    htf_bias = sig.htf_bias or "NEUTRAL"
    weekly_bias = sig.weekly_bias or "NEUTRAL"
    bias_emoji = {
        "BULLISH": "\U0001f4c8",
        "BEARISH": "\U0001f4c9",
        "NEUTRAL": "\u2796",
    }.get(htf_bias, "")
    weekly_emoji = {
        "BULLISH": "\U0001f4c8",
        "BEARISH": "\U0001f4c9",
        "NEUTRAL": "\u2796",
    }.get(weekly_bias, "")

    # Potansiyel RR: planned_rr veya orijinal SL ile hesap (trail/BE sonrasi SL bozulmasin).
    if sig.planned_rr is not None:
        rr_ratio = f"{math.floor(float(sig.planned_rr) * 100 + 1e-9) / 100:.2f}"
    else:
        sl_ref = sig.initial_stop_loss if sig.initial_stop_loss is not None else sig.stop_loss
        risk = abs(sig.entry_price - sl_ref) if sig.entry_price and sl_ref else 0
        reward = abs(sig.take_profit - sig.entry_price) if sig.take_profit and sig.entry_price else 0
        # RR'yi ASLA yukari yuvarlama (2.45 -> 2.4).
        rr_ratio = f"{math.floor(reward / risk * 100 + 1e-9) / 100:.2f}" if risk > 0 else "?"

    smt_line = (
        f"\U0001f517 <b>SMT:</b> {sig.smt_pair} \u2705\n"
        if sig.smt_pair else "\U0001f517 <b>SMT:</b> Yok\n"
    )

    # CISD saati UTC+3 (TSI) olarak gosterilir. cisd_time naive/aware UTC kabul edilir.
    if sig.cisd_time:
        ct = sig.cisd_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        cisd_txt = ct.astimezone(timezone(timedelta(hours=3))).strftime("%d.%m.%Y %H:%M") + " UTC+3"
    else:
        cisd_txt = "-"

    pd_line = f"\n\U0001f9f1 <b>PD Array:</b> {sig.pd_array}" if sig.pd_array else "\n\U0001f9f1 <b>PD Array:</b> Yok"

    # C2 (purge) mumunun kapanmis olmasi setup'i daha guvenilir kilar.
    c2_line = (
        "\n\U0001f56f\ufe0f <b>C2:</b> Kapali \u2705"
        if sig.c2_closed else "\n\U0001f56f\ufe0f <b>C2:</b> Kapanmadi \u26a0\ufe0f"
    )

    strat = _strategy_label(sig)
    return (
        f"{direction_emoji} <b>{sig.symbol} - {sig.direction} {strat} Signal ACTIVE</b>\n"
        f"\n"
        f"\U0001f3af <b>Entry:</b> <code>{sig.entry_price}</code>"
        f"  ·  {_ENTRY_MODEL_LABELS.get(getattr(sig, 'entry_model', None) or 'cisd', 'CISD')}\n"
        f"\U0001f6d1 <b>Stop Loss:</b> <code>{sig.stop_loss}</code>\n"
        f"\U00002705 <b>Take Profit:</b> <code>{sig.take_profit}</code>\n"
        f"\U0001f4ca <b>Plan R:R:</b> 1:{rr_ratio}\n"
        f"\n"
        f"{bias_emoji} <b>1D Bias:</b> {htf_bias}\n"
        f"{weekly_emoji} <b>1W Bias:</b> {weekly_bias}\n"
        f"\U00002b50 <b>Kalite:</b> {int(sig.bias_score or 0)}/10"
        f"{'  ·  <b>Premium</b>' if int(sig.bias_score or 0) >= 11 else ''}\n"
        f"{smt_line}"
        f"\U0001f552 <b>CISD:</b> {cisd_txt}"
        f"{c2_line}"
        f"{pd_line}"
    )


def _format_signal_result(sig: Signal) -> str:
    """Kapanan islem icin sonuc mesaji (aktif sinyale reply olarak gonderilir)."""
    rr = float(sig.rr_value) if sig.rr_value is not None else 0.0
    if sig.planned_rr is not None:
        plan_txt = f"1:{math.floor(float(sig.planned_rr) * 100 + 1e-9) / 100:.2f}"
    else:
        plan_txt = "-"
    dur = f"{sig.duration_hours}s" if sig.duration_hours is not None else "-"

    exit_kind = getattr(sig, "_exit_kind", None)

    strat = _strategy_label(sig)
    if sig.status == "breakeven" or sig.result == "breakeven" or exit_kind == "be":
        head = f"\u2796 <b>BREAKEVEN – {sig.symbol} ({strat})</b>"
        rr_txt = "0R"
        exit_line = f"\U0001f6d1 <b>Cikis:</b> <code>{sig.entry_price}</code> (giris)\n"
    elif sig.result == "win" and exit_kind == "trail":
        head = f"\u2705 <b>TRAIL EXIT – {sig.symbol} ({strat})</b>"
        rr_txt = f"+{rr:g}R"
        exit_line = f"\U0001f6d1 <b>Cikis (trail):</b> <code>{sig.stop_loss}</code>\n"
    elif sig.result == "win":
        head = f"\u2705 <b>TP HIT – {sig.symbol} ({strat})</b>"
        rr_txt = f"+{rr:g}R"
        exit_line = f"\U00002705 <b>Cikis (TP):</b> <code>{sig.take_profit}</code>\n"
    else:  # loss
        head = f"\U0001f6d1 <b>SL HIT – {sig.symbol} ({strat})</b>"
        rr_txt = f"{rr:g}R"
        exit_line = f"\U0001f6d1 <b>Cikis (SL):</b> <code>{sig.stop_loss}</code>\n"

    return (
        f"{head}\n"
        f"\n"
        f"\U0001f4cd <b>{sig.direction}</b> | <b>Gerceklesen:</b> {rr_txt}\n"
        f"\U0001f4ca <b>Plan R:R:</b> {plan_txt}\n"
        f"\U0001f3af <b>Giris:</b> <code>{sig.entry_price}</code>\n"
        f"{exit_line}"
        f"\U000023f1 <b>Sure:</b> {dur}"
    )


def is_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


async def _send_message(text: str, reply_to_message_id: Optional[int] = None) -> Optional[int]:
    """Mesaji gonderir; basarili olursa gonderilen mesajin message_id'sini dondurur."""
    if not is_configured():
        return None

    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
        # Orijinal mesaj silinmis/bulunamazsa yine de gonder.
        payload["allow_sending_without_reply"] = True

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", {}).get("message_id")
            log.warning("Telegram API error %d: %s", resp.status_code, resp.text)
            return None
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return None


async def send_signal_active(sig: Signal) -> Optional[int]:
    """Aktif sinyal mesajini gonderir; message_id dondurur (sonuc reply'i icin)."""
    if not is_configured():
        return None
    text = _format_active_signal(sig)
    mid = await _send_message(text)
    if mid:
        log.info("Telegram: active signal sent for %s %s (mid=%s)", sig.symbol, sig.direction, mid)
    return mid


async def send_signal_result(sig: Signal) -> bool:
    """Islem sonucunu, aktif sinyal mesajina reply olarak gonderir."""
    if not is_configured():
        return False
    text = _format_signal_result(sig)
    mid = await _send_message(text, reply_to_message_id=sig.tg_message_id)
    if mid:
        log.info("Telegram: result sent for %s %s (reply_to=%s)", sig.symbol, sig.direction, sig.tg_message_id)
    return mid is not None


async def send_test_message() -> bool:
    mid = await _send_message("\u2705 <b>TradeRadar</b> – Telegram bağlantısı başarılı!")
    return mid is not None
