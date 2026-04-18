"""LangChain tool wrappers for retrieval-augmented support answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_engine: Any | None = None
_engine_error: str | None = None


def retrieve_isp_knowledge(query: str) -> str:
    """Retrieve ISP troubleshooting and device context for factual support queries."""
    global _engine, _engine_error

    if _engine_error:
        return f"RAG retrieval unavailable: {_engine_error}"

    if _engine is None:
        try:
            from rag.config import (
                CHROMA_DIR,
                COLLECTION_NAME,
                DEFAULT_LLAMACPP_ENDPOINT,
                DEFAULT_RETRIEVE_TOP_K,
                DEFAULT_RERANK_TOP_K,
                DEFAULT_SENTENCES_PER_CHUNK,
            )
            from rag.inference import RAGEngine
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

    result = _engine.answer(
        query=query,
        top_k=getattr(_engine, "_default_retrieve_top_k", 10),
        rerank_k=getattr(_engine, "_default_rerank_top_k", 3),
        per_chunk_sentences=getattr(_engine, "_default_sentences_per_chunk", 3),
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
            "Retrieve ISP troubleshooting, router/device details, setup guides, and provider-specific technical "
            "context for factual customer support questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or troubleshooting issue to retrieve knowledge for.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
