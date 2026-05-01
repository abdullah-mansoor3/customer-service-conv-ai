"""LangChain tool wrappers for retrieval-augmented support answers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_engine: Any | None = None
_engine_error: str | None = None

try:
    from mcp_server.tools.integrations import call_rag_bridge
except Exception:  # noqa: BLE001
    call_rag_bridge = None


def _import_rag_runtime():
    """Import RAG runtime with a repo-root fallback for local backend launches."""
    try:
        from rag.config import (  # type: ignore
            CHROMA_DIR,
            COLLECTION_NAME,
            DEFAULT_LLAMACPP_ENDPOINT,
            DEFAULT_RETRIEVE_TOP_K,
            DEFAULT_RERANK_TOP_K,
            DEFAULT_SENTENCES_PER_CHUNK,
        )
        from rag.inference import RAGEngine  # type: ignore
    except Exception:
        repo_root = Path(__file__).resolve().parents[4]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from rag.config import (  # type: ignore
            CHROMA_DIR,
            COLLECTION_NAME,
            DEFAULT_LLAMACPP_ENDPOINT,
            DEFAULT_RETRIEVE_TOP_K,
            DEFAULT_RERANK_TOP_K,
            DEFAULT_SENTENCES_PER_CHUNK,
        )
        from rag.inference import RAGEngine  # type: ignore

    return (
        CHROMA_DIR,
        COLLECTION_NAME,
        DEFAULT_LLAMACPP_ENDPOINT,
        DEFAULT_RETRIEVE_TOP_K,
        DEFAULT_RERANK_TOP_K,
        DEFAULT_SENTENCES_PER_CHUNK,
        RAGEngine,
    )


def retrieve_isp_knowledge(query: str) -> str:
    """Retrieve ISP troubleshooting and device context for factual support queries."""
    global _engine, _engine_error

    bridge_url = os.getenv("RAG_TOOL_BRIDGE_URL", "").strip()
    if bridge_url and call_rag_bridge is not None:
        bridge_result = call_rag_bridge(query=query, top_k=3)
        if bridge_result.get("ok"):
            payload = bridge_result.get("data", {})
            if isinstance(payload, dict):
                return json.dumps(payload, ensure_ascii=False)
            return str(payload)

    if _engine_error:
        return f"RAG retrieval unavailable: {_engine_error}"

    if _engine is None:
        try:
            (
                CHROMA_DIR,
                COLLECTION_NAME,
                DEFAULT_LLAMACPP_ENDPOINT,
                DEFAULT_RETRIEVE_TOP_K,
                DEFAULT_RERANK_TOP_K,
                DEFAULT_SENTENCES_PER_CHUNK,
                RAGEngine,
            ) = _import_rag_runtime()
        except Exception as exc:  # noqa: BLE001
            _engine_error = str(exc)
            return f"RAG retrieval unavailable: {_engine_error}"

        _engine = RAGEngine(
            chroma_dir=Path(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
            llm_endpoint=DEFAULT_LLAMACPP_ENDPOINT,
            llm_model="tinyllama",
            llm_api_key=None,
        )
        _engine._default_retrieve_top_k = DEFAULT_RETRIEVE_TOP_K
        _engine._default_rerank_top_k = DEFAULT_RERANK_TOP_K
        _engine._default_sentences_per_chunk = DEFAULT_SENTENCES_PER_CHUNK

    normalized_query = query.strip().lower()
    is_factual_lookup = any(
        key in normalized_query
        for key in [
            "helpline",
            "contact",
            "complaint",
            "number",
            "price",
            "pricing",
            "package",
            "plan",
            "pkr",
            "rupees",
        ]
    )

    top_k = getattr(_engine, "_default_retrieve_top_k", 10)
    rerank_k = getattr(_engine, "_default_rerank_top_k", 3)
    per_chunk_sentences = getattr(_engine, "_default_sentences_per_chunk", 3)

    if is_factual_lookup:
        # Keep retrieval fast by default; allow overrides for quality-focused runs.
        factual_top_k = int(os.getenv("RAG_FACTUAL_TOP_K", "8"))
        factual_rerank_k = int(os.getenv("RAG_FACTUAL_RERANK_K", "3"))
        factual_sentences_per_chunk = int(os.getenv("RAG_FACTUAL_SENTENCES_PER_CHUNK", "3"))

        top_k = max(4, min(factual_top_k, 12))
        rerank_k = max(2, min(factual_rerank_k, top_k))
        per_chunk_sentences = max(2, min(factual_sentences_per_chunk, 5))

    result = _engine.answer(
        query=query,
        top_k=top_k,
        rerank_k=rerank_k,
        per_chunk_sentences=per_chunk_sentences,
        use_llm=False,
    )

    context = str(result.get("context", "")).strip()
    best_similarity = float(result.get("best_similarity", 0.0))
    if not context:
        return "No relevant knowledge base context found for this query."

    payload = {
        "best_similarity": round(best_similarity, 4),
        "context": context,
    }
    return json.dumps(payload, ensure_ascii=False)


RETRIEVE_ISP_KNOWLEDGE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_isp_knowledge",
        "description": (
            "Primary ISP knowledge-base lookup tool. Use for provider-specific factual questions (PTCL/Nayatel "
            "router brands, package details, helplines, setup, troubleshooting) before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Focused factual lookup query with provider name and requested fact. "
                        "Example: 'PTCL helpline number' or 'Nayatel package prices PKR'."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
