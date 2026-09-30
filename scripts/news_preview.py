"""Haber botu mesajlarinin onizlemesi (app/news.py). Sunucu gerekmez, DB'ye yazmaz.

Kullanim:
    python scripts/news_preview.py                  # canli feed'lerden bugunun mesajlarini basar
    python scripts/news_preview.py --ff ff.json --fj fj.xml   # kayitli dosyalardan (istek atmaz)
    python scripts/news_preview.py --send           # ayni mesajlari haber botuna da gonderir

Basilanlar: 09:00 ozeti, bugunun ilk haber grubunun on-bildirimi + sonuc reply'i, ilk konusma
grubu (varsa) ve ornek bir takvim disi haber (uydurma baslik, yalniz bicim icin).
FinancialJuice ard arda isteklere 429 verir; iki calistirma arasinda ~1 dk bekle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import news  # noqa: E402


def _plain(text: str) -> str:
    return re.sub(r"</?(b|i)>", "", text).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", help="kayitli ForexFactory JSON")
    ap.add_argument("--fj", help="kayitli FinancialJuice RSS")
    ap.add_argument("--send", action="store_true", help="haber botuna da gonder")
    args = ap.parse_args()

    async with httpx.AsyncClient(timeout=20, headers=news._HEADERS, follow_redirects=True) as client:
        ff = json.loads(Path(args.ff).read_text(encoding="utf-8")) if args.ff else (await client.get(news.FF_URL)).json()
        if args.fj:
            fj_text = Path(args.fj).read_text(encoding="utf-8")
        else:
            r = await client.get(news.FJ_URL)
            fj_text = r.text if r.status_code == 200 else ""
            if r.status_code != 200:
                print(f"! FinancialJuice HTTP {r.status_code} — sonuc eslestirmesi bos kalacak")
    events, holidays = news.parse_calendar(ff)
    headlines = news.parse_rss(fj_text) if fj_text else []
    now = datetime.now(timezone.utc)
    n = now.astimezone(news.TSI)

    msgs: list[tuple[str, str]] = []
    morning_now = n.replace(hour=news.NEWS_MORNING_HOUR, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    msgs.append(("09:00 ozeti", news.format_morning(morning_now, events, holidays)))

    today = {at: g for at, g in news.group_events(events).items() if at.astimezone(news.TSI).date() == n.date()}
    data_groups = [(at, g) for at, g in today.items() if any(e.info.kind in ("data", "oil") for e in g)]
    if data_groups:
        # En cok kirmizi haber iceren grup (esitlikte ilki)
        at, g = max(data_groups, key=lambda x: sum(e.impact == "High" for e in x[1]))
        msgs.append(("on-bildirim", news.format_alert(at, g, at - news._ALERT_BEFORE)))
        data = [e for e in g if e.info.kind in ("data", "oil")]
        results = {e.key: r for e in data if (r := news.match_result(e, headlines)) is not None}
        msgs.append(("sonuc (reply)", news.format_result(at, data, results)))
    talk_groups = [(at, g) for at, g in today.items() if any(e.info.kind in ("talk", "text") for e in g)]
    if talk_groups:
        at, g = talk_groups[0]
        talk = [e for e in g if e.info.kind in ("talk", "text")]
        msgs.append(("konusma on-bildirimi", news.format_alert(at, talk, at - news._ALERT_BEFORE)))
        found = {e.key: news.talk_headlines(e, headlines, news._TALK_WINDOW) for e in talk}
        msgs.append(("konusma sonrasi (reply)", news.format_talk(at, talk, found)))
    fake = news.Headline("demo", "White House: Trump to deliver remarks on tariffs at 16:00 ET", now)
    cat, label, meaning = news.sudden_category(fake.title)
    msgs.append((f"takvim disi ornek ({cat})", news.format_sudden(label, meaning, fake)))

    for name, text in msgs:
        print(f"\n----- {name} -----\n{_plain(text)}")

    if args.send:
        if not news.is_configured():
            print("\n! NEWS_TELEGRAM_BOT_TOKEN / NEWS_TELEGRAM_CHAT_ID .env'de yok — gonderilmedi")
            return
        last = None
        for name, text in msgs:
            reply = last if "reply" in name else None
            mid = await news.send(text, reply)
            last = mid if "reply" not in name else last
            print(f"sent {name}: {mid}")


if __name__ == "__main__":
    asyncio.run(main())
