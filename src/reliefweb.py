"""
reliefweb.py
------------
Live data connector for the ReliefWeb API (UN OCHA) — the only genuinely
real-time data source in this pipeline.

https://reliefweb.int/help/api

The API is free and public: no API key is required, just an `appname`
identifier that ReliefWeb uses for usage attribution/rate-limiting
(1,000 calls/day per appname, which is more than enough for a demo).

This module fetches recent, live humanitarian situation reports filtered
by location (and optionally a disaster-category-ish keyword), so the
pipeline has real, current, external context to reason over instead of
only the static needs.csv / disaster_tweets.csv files.

Design follows the same pattern as ai_message.py: every network call is
wrapped so a failure (offline demo, rate limit, API outage) degrades to
an empty result instead of crashing the app.
"""

import os
import requests

RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"
APP_NAME = os.environ.get("RELIEFWEB_APPNAME", "disaster_relief_capstone_agent")
REQUEST_TIMEOUT = 6  # seconds — keep the UI responsive even if ReliefWeb is slow


def fetch_live_reports(location: str, category: str = "", limit: int = 5) -> list:
    """
    Fetch recent, live humanitarian reports relevant to a location (and
    optionally a category keyword like 'food' or 'shelter').

    Returns a list of dicts:
        {"title": str, "summary": str, "source": str, "date": str, "url": str}

    Returns an empty list (never raises) if the API is unreachable, rate
    limited, or returns no results — callers should treat this as
    "no live context available right now", not as an error.
    """
    if not location or not str(location).strip():
        return []

    query_value = f"{location} {category}".strip()

    params = {
        "appname": APP_NAME,
        "query[value]": query_value,
        "query[operator]": "AND",
        "limit": limit,
        "sort[]": "date:desc",
        "fields[include][]": ["title", "body-html", "date.created", "source.name", "url_alias"],
    }

    try:
        response = requests.get(RELIEFWEB_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception:
        # Network error, timeout, non-200, malformed JSON, etc. — fail soft.
        return []

    reports = []
    for item in data.get("data", []):
        fields = item.get("fields", {})
        title = fields.get("title", "")
        body_html = fields.get("body-html", "") or ""
        # cheap HTML strip — good enough for a short context snippet
        summary = _strip_html(body_html)[:400]
        source_list = fields.get("source", [])
        source = source_list[0].get("name") if source_list else "ReliefWeb"
        date = fields.get("date", {}).get("created", "")
        url = fields.get("url_alias", "")

        reports.append({
            "title": title,
            "summary": summary,
            "source": source,
            "date": date,
            "url": url,
        })

    return reports


def _strip_html(html_text: str) -> str:
    """Minimal HTML tag stripper — avoids pulling in a heavy HTML parser dependency."""
    import re
    text = re.sub(r"<[^>]+>", " ", html_text)
    return re.sub(r"\s+", " ", text).strip()


def is_reliefweb_reachable() -> bool:
    """
    Lightweight connectivity check, used by the UI to tell the user whether
    live data is actually available right now (vs. silently falling back).
    """
    try:
        response = requests.get(
            RELIEFWEB_URL, params={"appname": APP_NAME, "limit": 1}, timeout=4
        )
        return response.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    results = fetch_live_reports("India", "flood", limit=3)
    print(f"Fetched {len(results)} live reports")
    for r in results:
        print(f"- [{r['date']}] {r['title']} ({r['source']})")
