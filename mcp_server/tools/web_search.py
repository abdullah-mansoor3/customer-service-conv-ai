"""Public web search helper for current troubleshooting information."""

from __future__ import annotations

import re
import warnings
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

try:
    from ddgs import DDGS
except Exception:  # noqa: BLE001
    from duckduckgo_search import DDGS

warnings.filterwarnings(
    "ignore",
    message=r"This package \(`duckduckgo_search`\) has been renamed to `ddgs`!.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*renamed to `ddgs`.*",
    category=RuntimeWarning,
)


def _pick_first_non_empty(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _extract_result_fields(item: dict[str, Any]) -> tuple[str, str, str]:
    """Normalize result field names across different DDGS payload shapes."""
    url = _pick_first_non_empty(item.get("href"))
    if not url:
        url = _pick_first_non_empty(item.get("url"))
    if not url:
        url = _pick_first_non_empty(item.get("link"))

    title = _pick_first_non_empty(item.get("title"))
    if not title:
        title = _pick_first_non_empty(item.get("name"))

    snippet = _pick_first_non_empty(item.get("body"))
    if not snippet:
        snippet = _pick_first_non_empty(item.get("snippet"))
    if not snippet:
        snippet = _pick_first_non_empty(item.get("description"))

    return url, title, snippet


def _run_ddgs_text(ddgs: DDGS, query: str, max_results: int) -> tuple[list[Any], str]:
    """Try multiple DDGS backends because provider/parser behavior varies by version."""
    attempted_modes: list[str] = []
    for backend in ["lite", "html", "api", None]:
        mode_label = backend or "default"
        attempted_modes.append(mode_label)

        try:
            if backend is None:
                items = list(ddgs.text(query, max_results=max_results))
            else:
                items = list(ddgs.text(query, max_results=max_results, backend=backend))
        except TypeError:
            # Older versions may not accept backend kwarg.
            if backend is not None:
                continue
            try:
                items = list(ddgs.text(query, max_results=max_results))
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue

        if items:
            return items, mode_label

    return [], f"none({','.join(attempted_modes)})"


def _strip_html_tags(text: str) -> str:
    plain = re.sub(r"<[^>]+>", " ", text)
    return " ".join(unescape(plain).split())


def _bing_html_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Fallback search by scraping Bing HTML when DDGS yields no parsed rows."""
    url = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=en"
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            html_doc = resp.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return []

    result_blocks = re.findall(r"<li[^>]*class=\"b_algo\"[\s\S]*?</li>", html_doc, flags=re.IGNORECASE)
    parsed_rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for block in result_blocks:
        anchor = re.search(r"<a[^>]+href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>", block, flags=re.IGNORECASE)
        if not anchor:
            continue
        row_url = unescape(anchor.group(1)).strip()
        if not row_url or row_url in seen_urls:
            continue
        seen_urls.add(row_url)

        row_title = _strip_html_tags(anchor.group(2))
        snippet_match = re.search(r"<p[^>]*>([\s\S]*?)</p>", block, flags=re.IGNORECASE)
        row_snippet = _strip_html_tags(snippet_match.group(1)) if snippet_match else ""

        parsed_rows.append(
            {
                "title": row_title,
                "url": row_url,
                "snippet": row_snippet,
                "source_query": query,
            }
        )
        if len(parsed_rows) >= max_results:
            break

    return parsed_rows


def _duckduckgo_html_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Secondary fallback using DuckDuckGo HTML endpoint."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            html_doc = resp.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for match in re.finditer(
        r"<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>",
        html_doc,
        flags=re.IGNORECASE,
    ):
        raw_href = unescape(match.group(1)).strip()
        parsed = urlparse(raw_href)
        final_url = raw_href
        if parsed.path.startswith("/l/"):
            params = parse_qs(parsed.query)
            uddg = params.get("uddg", [""])[0]
            if uddg:
                final_url = unquote(uddg)

        if not final_url or final_url in seen_urls:
            continue
        seen_urls.add(final_url)

        title = _strip_html_tags(match.group(2))
        rows.append(
            {
                "title": title,
                "url": final_url,
                "snippet": "",
                "source_query": query,
            }
        )
        if len(rows) >= max_results:
            break

    return rows


def _normalize_query(query: str) -> str:
    """Convert conversational requests into cleaner search-engine queries."""
    cleaned = " ".join(query.strip().split())
    lower = cleaned.lower()

    replacements = [
        r"^can you\s+",
        r"^could you\s+",
        r"^please\s+",
        r"^i want to learn about\s+",
        r"^i want to know\s+",
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
    has_ptcl = "ptcl" in lower
    has_nayatel = "nayatel" in lower
    is_price_query = any(token in lower for token in ["price", "pricing", "package", "plan", "cost"])

    # Preserve multi-provider pricing requests as multi-provider queries.
    if has_ptcl and has_nayatel and is_price_query:
        return "PTCL and Nayatel internet package prices PKR"

    if (
        "nayatel" in lower
        and "router" in lower
        and ("company" in lower or "brand" in lower or "use" in lower)
    ):
        return "Nayatel router model brand"

    if "ptcl" in lower and ("helpline" in lower or "contact" in lower or "complaint" in lower):
        return "PTCL helpline number complaint contact"

    if "ptcl" in lower and ("price" in lower or "pricing" in lower or "package" in lower or "plan" in lower):
        return "PTCL internet package prices PKR"

    if "nayatel" in lower and ("price" in lower or "pricing" in lower or "package" in lower or "plan" in lower):
        return "Nayatel internet package prices PKR"

    if len(cleaned) < 8:
        return query.strip()

    return cleaned


def _build_query_variants(original_query: str, normalized_query: str) -> list[str]:
    """Expand selected intents into provider-specific queries for better coverage."""
    lower = original_query.strip().lower()
    has_ptcl = "ptcl" in lower
    has_nayatel = "nayatel" in lower
    is_price_query = any(token in lower for token in ["price", "pricing", "package", "plan", "cost"])

    if has_ptcl and has_nayatel and is_price_query:
        return [
            "PTCL internet package prices PKR",
            "Nayatel internet package prices PKR",
        ]

    return [normalized_query]


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
    query_variants = _build_query_variants(original_query, clean_query)
    limit = max(1, min(max_results, 10))
    rows: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []

    try:
        with DDGS() as ddgs:
            dedupe_urls: set[str] = set()
            per_variant_results: list[list[dict[str, str]]] = []

            per_variant_limit = max(2, limit) if len(query_variants) > 1 else limit

            for variant_query in query_variants:
                variant_rows: list[dict[str, str]] = []
                ddgs_items, ddgs_mode = _run_ddgs_text(ddgs, variant_query, per_variant_limit)
                raw_count = 0
                for item in ddgs_items:
                    raw_count += 1
                    if not isinstance(item, dict):
                        continue

                    url, title, snippet = _extract_result_fields(item)
                    if not url or url in dedupe_urls:
                        continue

                    dedupe_urls.add(url)
                    variant_rows.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source_query": variant_query,
                        }
                    )
                per_variant_results.append(variant_rows)
                diagnostics.append(
                    {
                        "query": variant_query,
                        "ddgs_mode": ddgs_mode,
                        "raw_items": raw_count,
                        "parsed_items": len(variant_rows),
                    }
                )

            # Round-robin merge so multi-provider queries keep provider coverage.
            cursor = 0
            while len(rows) < limit:
                progressed = False
                for bucket in per_variant_results:
                    if cursor < len(bucket):
                        rows.append(bucket[cursor])
                        progressed = True
                        if len(rows) >= limit:
                            break
                if not progressed:
                    break
                cursor += 1

            # Safety fallback: if variant parsing produced no rows, try one broad pass.
            if not rows:
                fallback_rows: list[dict[str, str]] = []
                fallback_raw_count = 0
                ddgs_items, ddgs_mode = _run_ddgs_text(ddgs, clean_query, limit)
                for item in ddgs_items:
                    fallback_raw_count += 1
                    if not isinstance(item, dict):
                        continue
                    url, title, snippet = _extract_result_fields(item)
                    if not url or url in dedupe_urls:
                        continue
                    dedupe_urls.add(url)
                    fallback_rows.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source_query": clean_query,
                        }
                    )
                    if len(fallback_rows) >= limit:
                        break

                if not fallback_rows:
                    fallback_rows = _bing_html_search(clean_query, limit)
                if not fallback_rows:
                    fallback_rows = _duckduckgo_html_search(clean_query, limit)
                    for row in fallback_rows:
                        row_url = row.get("url", "")
                        if row_url:
                            dedupe_urls.add(row_url)

                rows.extend(fallback_rows)
                diagnostics.append(
                    {
                        "query": clean_query,
                        "ddgs_mode": ddgs_mode,
                        "raw_items": fallback_raw_count,
                        "parsed_items": len(fallback_rows),
                        "mode": "fallback_single_query",
                        "fallback_engine": (
                            "ddgs"
                            if fallback_raw_count > 0
                            else ("bing_html_or_ddg_html" if fallback_rows else "none")
                        ),
                    }
                )

        return {
            "original_query": original_query,
            "query": clean_query,
            "query_variants": query_variants,
            "count": len(rows),
            "results": rows,
            "diagnostics": diagnostics,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "original_query": original_query,
            "query": clean_query,
            "count": 0,
            "results": [],
            "error": f"Web search failed: {exc}",
        }
