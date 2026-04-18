"""Public web search helper for current troubleshooting information."""

from __future__ import annotations

from typing import Any

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Run DuckDuckGo search and return normalized result objects."""
    clean_query = query.strip()
    if not clean_query:
        return {
            "query": clean_query,
            "count": 0,
            "results": [],
            "error": "Query must be a non-empty string.",
        }

    limit = max(1, min(max_results, 10))
    rows: list[dict[str, str]] = []

    try:
        with DDGS() as ddgs:
            for item in ddgs.text(clean_query, max_results=limit):
                if not isinstance(item, dict):
                    continue

                url = str(item.get("href") or "").strip()
                if not url:
                    continue

                rows.append(
                    {
                        "title": str(item.get("title") or "").strip(),
                        "url": url,
                        "snippet": str(item.get("body") or "").strip(),
                    }
                )

        return {
            "query": clean_query,
            "count": len(rows),
            "results": rows,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "query": clean_query,
            "count": 0,
            "results": [],
            "error": f"Web search failed: {exc}",
        }
