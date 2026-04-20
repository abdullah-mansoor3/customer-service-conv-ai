"""Public web search helper for current troubleshooting information."""

from __future__ import annotations

import re
from typing import Any

from duckduckgo_search import DDGS


def _normalize_query(query: str) -> str:
    """Convert conversational requests into cleaner search-engine queries."""
    cleaned = " ".join(query.strip().split())
    lower = cleaned.lower()

    replacements = [
        r"^can you\s+",
        r"^could you\s+",
        r"^please\s+",
        r"\bsearch the web\b",
        r"\bsearch web\b",
        r"\bfind out\b",
        r"\blatest information about\b",
        r"\bfor me\b",
    ]
    for pattern in replacements:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[\?\!\"']", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    # Query-specific enrichment for recurring ISP vendor lookup asks.
    if (
        "nayatel" in lower
        and "router" in lower
        and ("company" in lower or "brand" in lower or "use" in lower)
    ):
        return "Nayatel router model brand"

    if len(cleaned) < 8:
        return query.strip()

    return cleaned


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Run DuckDuckGo search and return normalized result objects."""
    original_query = query.strip()
    if not original_query:
        return {
            "query": original_query,
            "count": 0,
            "results": [],
            "error": "Query must be a non-empty string.",
        }

    clean_query = _normalize_query(original_query)
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
            "original_query": original_query,
            "query": clean_query,
            "count": len(rows),
            "results": rows,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "original_query": original_query,
            "query": clean_query,
            "count": 0,
            "results": [],
            "error": f"Web search failed: {exc}",
        }
