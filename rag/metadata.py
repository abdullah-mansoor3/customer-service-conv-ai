from __future__ import annotations

import re
from pathlib import Path


PROVIDER_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("ptcl", ("ptcl", "ptcl.com.pk", "flashfiber")),
    ("nayatel", ("nayatel", "nayatel.com")),
    ("tplink", ("tp-link", "tplink", "tp-link.com", "tp-link-com")),
    ("xfinity", ("xfinity", "comcast", "how5220", "how9949", "xb6", "xb7", "xb8")),
    ("verizon", ("verizon", "fios")),
    ("spectrum", ("spectrum", "national_ca", "charter")),
    ("netgear", ("netgear", "c7000")),
    ("fcc", ("fcc", "doc-401799")),
]


def _normalize(value: str) -> str:
    value = value.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()


def infer_provider(file_name: str, preview_text: str = "") -> str:
    hay = _normalize(f"{file_name} {preview_text[:400]}")
    for provider, markers in PROVIDER_PATTERNS:
        if any(marker in hay for marker in markers):
            return provider
    return "unknown"


def provider_to_source(provider: str, file_name: str = "") -> str:
    file_lower = file_name.lower()
    if provider == "ptcl":
        return "ptcl.com.pk"
    if provider == "nayatel":
        return "nayatel.com"
    if provider == "tplink":
        return "tp-link.com"
    if provider == "xfinity":
        return "xfinity.com"
    if provider == "verizon":
        return "verizon.com"
    if provider == "spectrum":
        return "spectrum.com"
    if provider == "netgear":
        return "netgear.com"
    if provider == "fcc":
        return "fcc.gov"

    if "-com-" in file_lower:
        domain_like = file_lower.split("-")[:4]
        return ".".join([p for p in domain_like if p and p not in {"md", "pdf"}])
    return "unknown"


def infer_doc_type(file_name: str, preview_text: str = "") -> str:
    hay = _normalize(f"{file_name} {preview_text[:500]}")

    if any(k in hay for k in ("faq", "frequently asked", "q:", "q.")):
        return "faq"
    if any(k in hay for k in ("troubleshoot", "troubleshooting", "error code", "diagnostic")):
        return "troubleshooting"
    if any(
        k in hay
        for k in (
            "manual",
            "user guide",
            "quickstart",
            "quick start",
            "getting started",
            "setup guide",
            "installation",
            "gsg",
            "qsg",
        )
    ):
        return "manual"
    if any(k in hay for k in ("policy", "terms", "agreement", "safety", "acceptable use")):
        return "policy"
    return "guide"


def detect_provider_in_query(query: str) -> str | None:
    q = _normalize(query)
    for provider, markers in PROVIDER_PATTERNS:
        if any(marker in q for marker in markers):
            return provider
    return None


def safe_filename(path: Path) -> str:
    return path.name
