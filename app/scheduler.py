"""WebSocket durum koprusu.

Eski APScheduler cron kaldirildi; tarama artik BingX WebSocket olaylariyla
gercek zamanli calisiyor. Bu modul, scanner sayfasinin bekledigi durum
sozlugunu (running/jobs) BingX WS durumundan uretir.
"""

from __future__ import annotations

from app.market_data import market_data


def get_scheduler_status() -> dict:
    st = market_data.status()
    jobs = []
    if st.get("running"):
        conn = "Canli (baglandi)" if st.get("connected") else "Yeniden baglaniyor..."
        last = st.get("last_message_at") or "-"
        jobs.append({
            "id": "bingx-ws",
            "name": f"BingX WebSocket - {st.get('subscription_count', 0)} abonelik ({st.get('symbol_count', 0)} sembol)",
            "next_run": f"{conn} | son veri: {last}",
        })
    return {
        "running": bool(st.get("connected")),
        "jobs": jobs,
        "detail": st,
    }
