"""Web tools: search the live web and read pages.

Two tools, deliberately split:

  - ``web_search`` queries a self-hosted SearXNG instance and returns a list of
    result snippets (title, url, short summary). Cheap — gives the model
    candidates to choose from.
  - ``web_fetch`` downloads one URL and extracts the readable article text with
    trafilatura (navigation, ads, and boilerplate stripped). This is where the
    real content comes from, once the model picks a result worth reading.

SearXNG runs as a sibling container (see docker-compose.yml); its base URL comes
from ``TORA_SEARXNG_URL``, defaulting to the compose service name. Like the rest
of the tools, anything that goes wrong is *returned* as a dict rather than
raised, so the model can read the error and adapt.
"""

import os

import requests
import trafilatura

from .base import Tool

# The SearXNG JSON API. The compose service name resolves on the shared network;
# override via env for local dev (e.g. http://localhost:8088).
_SEARXNG_URL = os.environ.get("TORA_SEARXNG_URL", "http://searxng:8080").rstrip("/")

# Keep results and pages from swamping the model's context window.
_MAX_SNIPPET_CHARS = 500
_MAX_PAGE_CHARS = 10_000
_VALID_TIME_RANGES = {"day", "week", "month", "year"}


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def web_search(
    query: str, max_results: int = 10, time_range: str | None = None
) -> dict:
    """Search the web via SearXNG and return ranked result snippets."""

    params = {"q": query, "format": "json", "language": "en"}
    if time_range:
        if time_range not in _VALID_TIME_RANGES:
            return {"error": f"time_range must be one of {sorted(_VALID_TIME_RANGES)}"}
        params["time_range"] = time_range

    try:
        resp = requests.get(f"{_SEARXNG_URL}/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Search request failed: {e}"}
    except ValueError:
        # Body wasn't JSON — almost always SearXNG without the `json` output
        # format enabled (it returns 403/HTML in that case).
        return {
            "error": "SearXNG did not return JSON. Enable the 'json' format under "
            "search.formats in its settings.yml."
        }

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": _clip(r.get("content") or "", _MAX_SNIPPET_CHARS),
        }
        for r in data.get("results", [])[: max(1, max_results)]
    ]
    if not results:
        return {"results": [], "note": "No results found."}
    return {"results": results}


def web_fetch(url: str) -> dict:
    """Download a page and return its main readable text (no nav/ads/boilerplate)."""

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return {"error": f"Failed to fetch {url}: {e}", "url": url}
    if not downloaded:
        return {"error": f"Could not download {url}", "url": url}

    text = trafilatura.extract(downloaded, include_comments=False, include_links=False)
    if not text:
        return {"error": f"Could not extract readable text from {url}", "url": url}

    return {"url": url, "text": _clip(text, _MAX_PAGE_CHARS)}


web_search_tool = Tool(
    name="web_search",
    description=(
        "Search the live web and return a list of results (title, URL, and a "
        "short snippet). Use this for current events, recent information, or "
        "anything beyond your training data instead of guessing. For freshness, "
        "pass `time_range` (day/week/month/year). Snippets are short — call "
        "`web_fetch` on a result's URL to read the full page."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default 10).",
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "Restrict results by recency. Omit for no limit.",
            },
        },
        "required": ["query"],
    },
    fn=web_search,
)


web_fetch_tool = Tool(
    name="web_fetch",
    description=(
        "Download a web page by URL and return its main readable text, with "
        "navigation, ads, and boilerplate stripped out. Use this after "
        "`web_search` to read a promising result in full."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full http(s) URL of the page to read.",
            },
        },
        "required": ["url"],
    },
    fn=web_fetch,
)
