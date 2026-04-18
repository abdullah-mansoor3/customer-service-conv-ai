#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

ALLOWED_DOMAINS = {
    "ptcl.com.pk",
    "www.ptcl.com.pk",
    "nayatel.com",
    "www.nayatel.com",
    "stormfiber.com",
    "www.stormfiber.com",
    "transworld.com.pk",
    "www.transworld.com.pk",
    "transworld-home.com",
    "www.transworld-home.com",
    "tplink.com",
    "www.tp-link.com",
    "tp-link.com",
}

KEYWORDS = {
    "faq",
    "support",
    "help",
    "troubleshoot",
    "troubleshooting",
    "manual",
    "guide",
    "setup",
    "installation",
    "internet",
    "broadband",
    "wifi",
    "fiber",
    "router",
    "modem",
    "ont",
    "ftth",
}

BLOCK_PATH_PATTERNS = [
    re.compile(r"\.(?:jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|rar|mp4|mp3)$", re.I),
    re.compile(r"/(?:cart|checkout|login|signin|register|account)/", re.I),
]

SEED_URLS = [
    "https://ptcl.com.pk/",
    "https://ptcl.com.pk/Home/PageDetail?ItemId=421&linkId=905",
    "https://ptcl.com.pk/Home/PageDetail?ItemId=498",
    "https://nayatel.com/support",
    "https://nayatel.com/faqs",
    "https://stormfiber.com/support/",
    "https://transworld.com.pk/",
    "https://www.tp-link.com/pk/support/faq/",
    "https://www.tp-link.com/pk/support/download/",
]

SITEMAPS = [
    "https://ptcl.com.pk/sitemap.xml",
    "https://nayatel.com/sitemap_index.xml",
    "https://nayatel.com/sitemap.xml",
    "https://stormfiber.com/sitemap_index.xml",
    "https://stormfiber.com/sitemap.xml",
    "https://transworld.com.pk/sitemap_index.xml",
    "https://transworld.com.pk/sitemap.xml",
    "https://www.tp-link.com/pk/sitemap.xml",
]


@dataclass
class SavedDoc:
    url: str
    file_name: str
    title: str
    domain: str


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    clean = f"{scheme}://{netloc}{path}"
    if parsed.query:
        clean = f"{clean}?{parsed.query}"
    return clean.rstrip("#")


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    domain = parsed.netloc.lower()
    if domain not in ALLOWED_DOMAINS:
        return False
    lower = url.lower()
    if not any(keyword in lower for keyword in KEYWORDS):
        return False
    if any(p.search(lower) for p in BLOCK_PATH_PATTERNS):
        return False
    return True


def parse_sitemap_xml(content: str) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    child_sitemaps: list[str] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return urls, child_sitemaps

    tag = root.tag.lower()
    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    if tag.endswith("urlset"):
        for url_node in root.findall(f"{ns}url"):
            loc = url_node.find(f"{ns}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    elif tag.endswith("sitemapindex"):
        for sm in root.findall(f"{ns}sitemap"):
            loc = sm.find(f"{ns}loc")
            if loc is not None and loc.text:
                child_sitemaps.append(loc.text.strip())

    return urls, child_sitemaps


def fetch_text(url: str, timeout: int = 20) -> str | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code >= 400:
        return None
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def collect_from_sitemaps(max_urls: int = 800) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    queue = deque(SITEMAPS)

    while queue and len(found) < max_urls:
        sm = queue.popleft()
        if sm in seen:
            continue
        seen.add(sm)
        content = fetch_text(sm)
        if not content:
            continue
        urls, child = parse_sitemap_xml(content)
        for next_sm in child:
            if next_sm not in seen:
                queue.append(next_sm)
        for url in urls:
            url = normalize_url(url)
            if is_allowed(url):
                found.append(url)
                if len(found) >= max_urls:
                    break

    return list(dict.fromkeys(found))


def extract_links_from_html(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = normalize_url(urljoin(base_url, href))
        if is_allowed(abs_url):
            links.append(abs_url)
    return list(dict.fromkeys(links))


def crawl_from_seeds(max_pages: int = 500, max_depth: int = 2, sleep_s: float = 0.2) -> list[str]:
    found: list[str] = []
    visited: set[str] = set()
    queue = deque((normalize_url(url), 0) for url in SEED_URLS)

    while queue and len(found) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        if not is_allowed(url):
            continue

        html = fetch_text(url)
        if not html:
            continue
        found.append(url)
        if depth == max_depth:
            continue

        for link in extract_links_from_html(url, html):
            if link not in visited:
                queue.append((link, depth + 1))

        time.sleep(sleep_s)

    return list(dict.fromkeys(found))


def html_to_text(url: str, html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe", "header", "footer", "nav"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"

    blocks = []
    for selector in ["main", "article", "section", "div", "body"]:
        candidates = soup.select(selector)
        if candidates:
            text = "\n".join(c.get_text("\n", strip=True) for c in candidates[:5])
            if len(text) > 400:
                blocks.append(text)
    if not blocks:
        blocks.append(soup.get_text("\n", strip=True))

    text = "\n\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    content = f"# {title}\n\nSource: {url}\n\n{text.strip()}\n"
    return title, content


def existing_doc_count(rag_dir: Path) -> int:
    valid_suffixes = {".txt", ".md", ".pdf", ".html"}
    return sum(1 for p in rag_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes)


def save_page_as_doc(rag_dir: Path, url: str, html: str) -> SavedDoc | None:
    title, content = html_to_text(url, html)
    body = content.strip()
    if len(body) < 1200:
        return None

    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    slug_base = re.sub(r"[^a-z0-9]+", "-", f"{domain}-{parsed.path}-{parsed.query}".lower()).strip("-")
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    file_name = f"{slug_base[:80]}-{url_hash}.md"
    target = rag_dir / file_name
    if target.exists():
        return None

    target.write_text(content, encoding="utf-8")
    return SavedDoc(url=url, file_name=file_name, title=title, domain=domain)


def dedupe_preserve_order(urls: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(normalize_url(u) for u in urls))


def run(target_total: int, rag_dir: Path, max_fetch: int, sleep_s: float) -> None:
    rag_dir.mkdir(parents=True, exist_ok=True)
    start_count = existing_doc_count(rag_dir)
    needed = max(0, target_total - start_count)

    print(f"Existing docs: {start_count}")
    print(f"Target total: {target_total}")
    print(f"Need to add: {needed}")

    if needed == 0:
        print("Target already reached. No scraping needed.")
        return

    sitemap_urls = collect_from_sitemaps(max_urls=max_fetch)
    seed_crawl_urls = crawl_from_seeds(max_pages=max_fetch, max_depth=2, sleep_s=sleep_s)
    candidate_urls = dedupe_preserve_order([*sitemap_urls, *seed_crawl_urls])

    print(f"Candidate URLs discovered: {len(candidate_urls)}")

    saved: list[SavedDoc] = []
    for idx, url in enumerate(candidate_urls, start=1):
        if len(saved) >= needed:
            break

        html = fetch_text(url)
        if not html:
            continue
        result = save_page_as_doc(rag_dir, url, html)
        if result:
            saved.append(result)
            print(f"[{len(saved):03d}/{needed:03d}] saved {result.file_name}")

        if idx % 20 == 0:
            time.sleep(sleep_s)

    manifest_path = rag_dir / "scraped_manifest.json"
    manifest = {
        "generated_at_unix": int(time.time()),
        "target_total": target_total,
        "existing_before": start_count,
        "saved_now": len(saved),
        "total_after": existing_doc_count(rag_dir),
        "sources": [
            {"url": d.url, "file": d.file_name, "title": d.title, "domain": d.domain}
            for d in saved
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved this run: {len(saved)}")
    print(f"Total now: {manifest['total_after']}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ISP support/FAQ content for RAG dataset")
    parser.add_argument(
        "--target-total",
        type=int,
        default=100,
        help="Desired total number of documents in rag_data (default: 100)",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rag_data",
        help="Path to rag_data directory",
    )
    parser.add_argument(
        "--max-fetch",
        type=int,
        default=1200,
        help="Maximum URLs to collect from sitemap/crawl before filtering",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep (seconds) between batches to reduce request pressure",
    )

    args = parser.parse_args()
    run(
        target_total=args.target_total,
        rag_dir=args.rag_dir,
        max_fetch=args.max_fetch,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()
