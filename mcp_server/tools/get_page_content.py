"""Fetch web page content from URLs for detailed information."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 8000


def _strip_scripts_and_styles(html: str) -> str:
    script_pattern = r"<script[^>]*>[\s\S]*?</script>"
    style_pattern = r"<style[^>]*>[\s\S]*?</style>"
    nav_pattern = r"<nav[^>]*>[\s\S]*?</nav>"
    footer_pattern = r"<footer[^>]*>[\s\S]*?</footer>"
    header_pattern = r"<header[^>]*>[\s\S]*?</header>"
    aside_pattern = r"<aside[^>]*>[\s\S]*?</aside>"
    for pattern in [script_pattern, style_pattern, nav_pattern, footer_pattern, header_pattern, aside_pattern]:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE)
    return html


def _extract_main_content(html: str) -> str:
    html = _strip_scripts_and_styles(html)

    common_article_tags = [
        r"<article[^>]*>([\s\S]*?)</article>",
        r"<main[^>]*>([\s\S]*?)</main>",
        r"<div[^>]*class=\"[^\"]*content[^\"]*\"[^>]*>([\s\S]*?)</div>",
        r"<div[^>]*id=\"[^\"]*content[^\"]*\"[^>]*>([\s\S]*?)</div>",
    ]

    for pattern in common_article_tags:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            html = match.group(1)

    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    content = html.strip()

    return content


def get_page_content(url: str, max_length: int = MAX_CONTENT_LENGTH) -> dict[str, Any]:
    """Fetch and extract main content from a web page URL."""
    if not url or not url.strip():
        return {
            "url": url,
            "success": False,
            "error": "URL must be a non-empty string.",
            "content": "",
        }

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        return {
            "url": url,
            "success": False,
            "error": "URL must start with http:// or https://",
            "content": "",
        }

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            html_bytes = response.read()
            content_encoding = response.headers.get("Content-Encoding", "").lower()

            if content_encoding in ("gzip", "deflate"):
                import gzip

                if content_encoding == "gzip":
                    html_bytes = gzip.decompress(html_bytes)
                else:
                    import zlib

                    html_bytes = zlib.decompress(html_bytes)

            html = html_bytes.decode("utf-8", errors="ignore")

    except Exception as exc:
        logger.warning("get_page_content failed for %s: %s", url, exc)
        return {
            "url": url,
            "success": False,
            "error": f"Failed to fetch page: {exc}",
            "content": "",
        }

    main_content = _extract_main_content(html)

    if len(main_content) > max_length:
        main_content = main_content[:max_length].rsplit(" ", 1)[0] + "..."

    return {
        "url": url,
        "success": True,
        "content": main_content,
    }


GET_PAGE_CONTENT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_page_content",
        "description": (
            "Fetch the full content of a web page URL to get detailed information not available in search snippets. "
            "Use after web_search returns relevant URLs - call this tool to get the full page content, "
            "then use that content as context to answer the user's question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The complete URL of the web page to fetch content from.",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 8000).",
                    "default": 8000,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}