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
    "bpr": "BPR",
}


# 11.11 (kullanici: "mesajlari anlamaya calisirken kayboluyorum"): butun mesajlar kisa -- baslik
# "<yon> <SEMBOL> <TF> <YON> · <TUR>" + en fazla 2 satir. Aciklama cumlesi, bias, CISD saati, key level yok.
_TF_SHORT = {"1d": "1D", "1h": "1H", "1w": "1W", "4h": "4H"}


def _tf_short(tf: str | None) -> str:
    return _TF_SHORT.get(tf or "4h", "4H")


def _strategy_label(sig: "Signal") -> str:
    return _tf_short(getattr(sig, "timeframe", None))


def _dir_emoji(direction: str) -> str:
    return "🟢" if direction == "LONG" else "🔴"


def _rr_txt(rr) -> str:
    # RR'yi ASLA yukari yuvarlama (2.45 -> 2.4).
    return f"{math.floor(float(rr) * 100 + 1e-9) / 100:.2f}" if rr is not None else "-"


def _plan_rr(sig: "Signal"):
    """Plan RR: planned_rr, yoksa ORIJINAL SL ile (trail/BE sonrasi SL bozulmasin)."""
    if sig.planned_rr is not None:
        return sig.planned_rr
    sl_ref = sig.initial_stop_loss if sig.initial_stop_loss is not None else sig.stop_loss
    risk = abs(sig.entry_price - sl_ref) if sig.entry_price and sl_ref else 0
    reward = abs(sig.take_profit - sig.entry_price) if sig.take_profit and sig.entry_price else 0
    return reward / risk if risk > 0 else None


def _levels_line(entry, sl, tp, model: str | None = None) -> str:
    m = f" ({_ENTRY_MODEL_LABELS.get(model or 'cisd', 'CISD')})" if model else ""
    return f"Giriş <code>{entry}</code>{m} · SL <code>{sl}</code> · TP <code>{tp}</code>"


def _r(v: float) -> str:
    return f"{v:+.2f}R".replace("+-", "−").replace("-", "−")


def _format_active_signal(sig: Signal) -> str:
    return (
        f"{_dir_emoji(sig.direction)} <b>{sig.symbol} {_strategy_label(sig)} {sig.direction} · AKTİF</b>\n"
        f"{_levels_line(sig.entry_price, sig.stop_loss, sig.take_profit, getattr(sig, 'entry_model', None) or 'cisd')}\n"
        f"RR {_rr_txt(_plan_rr(sig))} · skor {int(sig.bias_score or 0)}"
    )


def _partial_banked_r(sig: Signal) -> float | None:
    """Kismi karla kasaya giren R, TUM pozisyonun riski cinsinden (kesir x kismi R).

    `partial_rr` fiyatin kac R ilerledigidir; pozisyonun yalnizca `partial_size`
    kadari kapandigi icin kazanc bunun o kadaridir. Mesajlarda ham partial_rr
    yazilinca (+1.82R) sonuc mesajindaki toplam (+0.91R) dusus gibi gorunuyordu.
    """
    size = getattr(sig, "partial_size", None)
    prr = getattr(sig, "partial_rr", None)
    if not size or prr is None:
        return None
    return float(size) * float(prr)


def _format_signal_result(sig: Signal) -> str:
    """Kapanan islem icin sonuc mesaji (aktif sinyale reply olarak gonderilir)."""
    rr = float(sig.rr_value) if sig.rr_value is not None else 0.0
    exit_kind = getattr(sig, "_exit_kind", None) or getattr(sig, "exit_reason", None)
    head = f"{sig.symbol} {_strategy_label(sig)}"
    if exit_kind == "be" and rr > 0:
        # Kismi kar alinip kalan yari giriste kapandi: toplamda kar + BE.
        return f"🟡 <b>{head} · %50 + BE {_r(rr)}</b>"
    if sig.status == "breakeven" or sig.result == "breakeven" or exit_kind == "be":
        return f"➖ <b>{head} · BE 0R</b>"
    if exit_kind == "week_close":
        icon = "✅" if rr > 0 else ("❌" if rr < 0 else "➖")
        return f"{icon} <b>{head} · HAFTA KAPANIŞI {_r(rr)}</b>"
    if sig.result == "win" and exit_kind == "trail":
        return f"✅ <b>{head} · TRAIL {_r(rr)}</b>"
    if sig.result == "win":
        return f"✅ <b>{head} · TP {_r(rr)}</b>"
    return f"❌ <b>{head} · SL {_r(rr)}</b>"


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
    # 1D potansiyel / 4H-1H waiting bildirimi gittiyse ACTIVE onun devami olarak gelir.
    mid = await _send_message(text, reply_to_message_id=getattr(sig, "tg_potential_id", None))
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


def _format_signal_partial(sig: Signal) -> str:
    """Kismi kar bildirimi: pozisyonun bir kismi kapatildi, kalan SL girise cekildi."""
    pct = int(float(sig.partial_size or 0) * 100)
    banked = _partial_banked_r(sig) or 0.0
    return (
        f"💰 <b>{sig.symbol} {_strategy_label(sig)} · %{pct} KAPANDI {_r(banked)}</b>\n"
        f"Kalan SL girişte"
    )


async def send_signal_partial(sig: Signal) -> bool:
    """Kismi kar bildirimini aktif sinyal mesajina reply olarak gonderir."""
    if not is_configured():
        return False
    mid = await _send_message(_format_signal_partial(sig), reply_to_message_id=sig.tg_message_id)
    if mid:
        log.info("Telegram: partial sent for %s %s (reply_to=%s)", sig.symbol, sig.direction, sig.tg_message_id)
    return mid is not None


_TSI = timezone(timedelta(hours=3))

# Sinyal / zincir dolmadan silinme nedenleri (scanner reason kodlari) -- 11.11 kisa metin.
_POTENTIAL_CANCEL_REASONS = {
    "crt_gone": "CRT bozuldu",
    "crt_replaced": "yeni CRT geldi",
    "bias_mismatch": "bias ters döndü",
    "past_tp": "girişsiz TP'ye gitti",
    "near_tp": "girişsiz TP'ye yaklaştı",
    "low_quality": "skor düştü",
    "score7_gate": "skor kapısı",
    "no_levels": "seviyeler kayboldu",
    "low_rr": "RR düştü",
    "tight_stop": "stop çok dar",
    "past_sl": "girişsiz SL'i geçti",
    "crt_50": "CRT %60 geçildi",
    "invalidated": "CRT %60 geçildi",
    "missed_quality": "retestte skor düşük",
    "stale": "bayat retest",
    "week_close": "hafta kapanışı",
    "week_gap": "hafta sonu boşluğu",
    "liq_unswept": "likidite alınmadı",
    "same_setup": "aynı setup başka coinde",
}

# Motorun almadigi (kapida elenen) setup icin kisa kapi adi.
_POTENTIAL_GATE_LABELS = {
    "bias_mismatch": "bias ters",
    "low_quality": "skor eşik altı",
    "score7_gate": "skor kapısı",
    "low_rr": "RR 2 altı",
    "has_open": "sembolde açık işlem",
    "corr_open": "korele açık işlem",
    "duplicate": "aynı setup kayıtlı",
    "cluster_limit": "küme limiti",
    "same_setup": "aynı setup başka coinde",
    "deleted_before": "daha önce silindi",
}


def _fmt_tsi(dt) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_TSI).strftime("%d.%m %H:%M")


def _format_signal_potential(sig: Signal, c2_close_at=None, gate: str | None = None) -> str:
    """ESKI potansiyel mesaji (1D POTANSIYEL zinciri, 29.09'dan beri kapali; kod duruyor)."""
    tag = f" · motor almıyor: {_POTENTIAL_GATE_LABELS.get(gate, gate)}" if gate else ""
    return (
        f"\U0001f440 <b>{sig.symbol} {_strategy_label(sig)} {sig.direction} · POTANSİYEL</b>\n"
        f"{_levels_line(sig.entry_price, sig.stop_loss, sig.take_profit)}\n"
        f"RR {_rr_txt(sig.planned_rr)} · skor {int(sig.bias_score or 0)}{tag}"
    )


async def send_signal_potential(
    sig: Signal,
    *,
    c2_close_at=None,
    reply_to: Optional[int] = None,
    gate: str | None = None,
) -> Optional[int]:
    """ESKI potansiyel bildirimi (kapali zincir); 11.11'den beri kapida elenen 1D/1W zinciri
    `format_potential_*` + `send_text` ile gider."""
    if not is_configured():
        return None
    mid = await _send_message(
        _format_signal_potential(sig, c2_close_at, gate=gate), reply_to_message_id=reply_to,
    )
    if mid:
        log.info("Telegram: potential sent for %s %s (mid=%s reply_to=%s gate=%s)",
                 sig.symbol, sig.direction, mid, reply_to, gate or "-")
    return mid


def _format_signal_waiting(sig: Signal) -> str:
    """Sinyal waiting_entry oldu (limit emri entry'de bekliyor)."""
    return (
        f"⏳ <b>{sig.symbol} {_strategy_label(sig)} {sig.direction} · WAITING</b>\n"
        f"{_levels_line(sig.entry_price, sig.stop_loss, sig.take_profit, getattr(sig, 'entry_model', None) or 'cisd')}\n"
        f"RR {_rr_txt(sig.planned_rr)} · skor {int(sig.bias_score or 0)}"
    )


async def send_signal_waiting(sig: Signal, reply_to: Optional[int] = None) -> Optional[int]:
    """waiting_entry bildirimi; message_id doner (ACTIVE / IPTAL buna reply olur).

    `reply_to`: setup once potansiyel zincirinde bildirildiyse (1D/1W) o mesaj.
    """
    if not is_configured():
        return None
    mid = await _send_message(_format_signal_waiting(sig), reply_to_message_id=reply_to)
    if mid:
        log.info("Telegram: waiting sent for %s %s %s (mid=%s)",
                 _strategy_label(sig), sig.symbol, sig.direction, mid)
    return mid


def _format_potential_cancel(
    symbol: str, direction: str, reason: str, strategy: str | None = None, gated: bool = False,
) -> str:
    return (f"❌ <b>{symbol} {_tf_short(strategy or '1d')} · İPTAL</b> — "
            f"{_POTENTIAL_CANCEL_REASONS.get(reason, reason)}")


async def send_potential_cancel(
    symbol: str, direction: str, reason: str, reply_to: int, strategy: str | None = None,
    *, gated: bool = False,
) -> bool:
    """Waiting setup sinyale donusmeden silindi: zincire reply."""
    if not is_configured():
        return False
    mid = await _send_message(
        _format_potential_cancel(symbol, direction, reason, strategy, gated), reply_to_message_id=reply_to,
    )
    if mid:
        log.info("Telegram: potential cancel sent for %s %s (%s, reply_to=%s)", symbol, direction, reason, reply_to)
    return mid is not None


def _format_signal_void(symbol: str, direction: str, strategy: str | None, score_asof, min_score) -> str:
    return f"❌ <b>{symbol} {_tf_short(strategy)} · İPTAL</b> — dolum skoru {score_asof} (eşik {min_score})"


async def send_signal_void(
    symbol: str, direction: str, strategy: str | None, reply_to: int, score_asof, min_score,
) -> bool:
    """Dolmus sinyal VOID FILL ile silindi (04.10): aktif mesajina IPTAL reply'i."""
    if not is_configured():
        return False
    mid = await _send_message(
        _format_signal_void(symbol, direction, strategy, score_asof, min_score),
        reply_to_message_id=reply_to,
    )
    if mid:
        log.info("Telegram: void sent for %s %s (reply_to=%s)", symbol, direction, reply_to)
    return mid is not None


# ──────────────────── 1D / 1W potansiyel zinciri (11.11) ────────────────────
# Motorun ALMADIGI ama kaliteli 1D/1W setup: CRT -> WAITING -> girise degdi -> sonuc. Sonuc motorun kuraliyla
# (%50'de yarisi + kalan BE), mesajdaki seviyelerden, islem acilmasa da. Durum `potential_notices`ta.
_POT_RESULT = {
    "tp": ("✅", "TP"),
    "tp_nofill": ("✅", "TP (girişe dönmeden)"),
    "be50": ("\U0001f7e1", "%50 + BE"),
    "sl": ("❌", "SL"),
    "expired": ("⌛", "süre doldu"),
}


def format_potential_crt(symbol: str, strategy: str, direction: str, target, score: int) -> str:
    return (f"\U0001f514 <b>{symbol} {_tf_short(strategy)} {direction} · CRT</b>\n"
            f"Hedef <code>{target}</code> · skor {score} · CISD bekleniyor")


def format_potential_waiting(symbol: str, strategy: str, direction: str, entry, sl, tp, rr, score: int,
                             gate: str | None, model: str | None = None) -> str:
    g = _POTENTIAL_GATE_LABELS.get(gate, gate) if gate else None
    return (f"⏳ <b>{symbol} {_tf_short(strategy)} {direction} · POTANSİYEL WAITING</b>\n"
            f"{_levels_line(entry, sl, tp, model)}\n"
            f"RR {_rr_txt(rr)} · skor {score}" + (f" · motor almıyor: {g}" if g else ""))


def format_potential_lowq(symbol: str, strategy: str, score: int, rr) -> str:
    return (f"⚪️ <b>{symbol} {_tf_short(strategy)} · kalite yetersiz</b> — skor {score}"
            f" · RR {_rr_txt(rr)}, takip bitti")


def format_potential_filled(symbol: str, strategy: str, entry) -> str:
    return f"\U0001f4e5 <b>{symbol} {_tf_short(strategy)} · girişe değdi</b> <code>{entry}</code>"


def format_potential_result(symbol: str, strategy: str, result: str, r: float | None) -> str:
    icon, label = _POT_RESULT.get(result, ("ℹ️", result))
    tail = f" {_r(r)}" if r is not None and result in ("tp", "be50", "sl") else ""
    return f"{icon} <b>{symbol} {_tf_short(strategy)} · {label}{tail}</b>"


async def send_text(text: str, reply_to: Optional[int] = None) -> Optional[int]:
    """Hazir metni gonder (potansiyel zinciri); message_id doner."""
    if not is_configured():
        return None
    return await _send_message(text, reply_to_message_id=reply_to)


async def send_test_message() -> bool:
    mid = await _send_message("\u2705 <b>TradeRadar</b> – Telegram bağlantısı başarılı!")
    return mid is not None
