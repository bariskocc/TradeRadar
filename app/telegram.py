"""Telegram bildirim modülü – CRT sinyallerini Telegram'a gönderir."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

if TYPE_CHECKING:
    from app.models import Signal

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _format_active_signal(sig: Signal) -> str:
    direction_emoji = "\U0001f7e2" if sig.direction == "LONG" else "\U0001f534"
    bias_emoji = {
        "BULLISH": "\U0001f4c8",
        "BEARISH": "\U0001f4c9",
        "NEUTRAL": "\u2796",
    }.get(sig.bias or "NEUTRAL", "")

    risk = abs(sig.entry_price - sig.stop_loss) if sig.entry_price and sig.stop_loss else 0
    reward = abs(sig.take_profit - sig.entry_price) if sig.take_profit and sig.entry_price else 0
    rr_ratio = f"{reward / risk:.1f}" if risk > 0 else "?"

    return (
        f"{direction_emoji} <b>CRT Signal ACTIVE – {sig.symbol}</b>\n"
        f"\n"
        f"\U0001f3af <b>Entry:</b> <code>{sig.entry_price}</code>\n"
        f"\U0001f6d1 <b>Stop Loss:</b> <code>{sig.stop_loss}</code>\n"
        f"\U00002705 <b>Take Profit:</b> <code>{sig.take_profit}</code>\n"
        f"\U0001f4ca <b>R:R Ratio:</b> 1:{rr_ratio}\n"
        f"\n"
        f"\U0001f4cd <b>Direction:</b> {sig.direction}\n"
        f"\U0001f50d <b>Purge:</b> {sig.purge_type}\n"
        f"{bias_emoji} <b>Bias:</b> {sig.bias} ({int(sig.bias_score or 0):+d})\n"
        f"\n"
        f"\U0001f6a7 <b>Invalidation (%50):</b> <code>{sig.invalidation_level}</code>\n"
        f"\U0001f4c8 <b>Key High:</b> <code>{sig.key_level_high}</code>\n"
        f"\U0001f4c9 <b>Key Low:</b> <code>{sig.key_level_low}</code>\n"
        f"\n"
        f"\U0001f552 <b>CISD:</b> {sig.cisd_time.strftime('%d.%m.%Y %H:%M') if sig.cisd_time else '-'} UTC\n"
        f"\U0001f30d <b>Market:</b> {sig.market_type.upper()}\n"
        f"\n"
        f"#TradeRadar #CRT #{sig.symbol} #{sig.direction}"
    )


def is_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


async def _send_message(text: str) -> bool:
    if not is_configured():
        return False

    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return True
            log.warning("Telegram API error %d: %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False


async def send_signal_active(sig: Signal) -> bool:
    if not is_configured():
        return False
    text = _format_active_signal(sig)
    ok = await _send_message(text)
    if ok:
        log.info("Telegram: active signal sent for %s %s", sig.symbol, sig.direction)
    return ok


async def send_test_message() -> bool:
    return await _send_message("\u2705 <b>TradeRadar</b> – Telegram bağlantısı başarılı!")
