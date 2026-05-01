from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import httpx
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.cache import SemanticLRUCache
from rag.chunking import sentence_split, token_count
from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_LLAMACPP_ENDPOINT,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_SENTENCES_PER_CHUNK,
    EMBEDDING_MODEL,
    RELEVANCE_THRESHOLD,
    RERANK_MODEL,
    SEMANTIC_CACHE_CAPACITY,
    SEMANTIC_CACHE_THRESHOLD,
)
from rag.metadata import detect_provider_in_query


def preload_models() -> None:
    """Preload SentenceTransformer models from local cache to avoid hitting HuggingFace API."""
    import os
    
    # First, try to load from local cache only (no API calls)
    # This ensures we don't hit HuggingFace on every startup
    try:
        _ = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        # If not in cache, download silently without hitting the API repeatedly
        # Use HF_HUB_OFFLINE to prevent version checks
        original_offline = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _ = SentenceTransformer(EMBEDDING_MODEL)
        finally:
            os.environ["HF_HUB_OFFLINE"] = original_offline
    
    try:
        _ = CrossEncoder(RERANK_MODEL, local_files_only=True)
    except Exception:
        original_offline = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _ = CrossEncoder(RERANK_MODEL)
        finally:
            os.environ["HF_HUB_OFFLINE"] = original_offline


GROUNDING_PROMPT_TEMPLATE = """You are a customer support assistant for ISP providers.
Answer the user's question ONLY using the context provided below.
If the context does not contain enough information to answer, say:
\"I don't have enough information to answer that.\"
Do not make up any technical details, IP addresses, settings, or procedures.

Context:
{retrieved_context}

Question: {user_query}
Answer:"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    rerank_score: float = 0.0
    selected_sentences: list[str] | None = None


class RAGEngine:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        llm_endpoint: str,
        llm_model: str,
        llm_api_key: str | None,
        cache_capacity: int = SEMANTIC_CACHE_CAPACITY,
        cache_threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ):
        try:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        try:
            self.reranker = CrossEncoder(RERANK_MODEL, local_files_only=True)
        except Exception:
            self.reranker = CrossEncoder(RERANK_MODEL)
        self.cache = SemanticLRUCache(capacity=cache_capacity, similarity_threshold=cache_threshold)

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(name=collection_name)

        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key

    @staticmethod
    def _safe_similarity_from_distance(distance: float | int | None) -> float:
        if distance is None:
            return 0.0
        dist = float(distance)
        sim = 1.0 - dist
        return max(0.0, min(1.0, sim))

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(emb, dtype=np.float32)

    def _query_chroma(self, query_embedding: np.ndarray, provider: str | None, top_k: int) -> list[RetrievedChunk]:
        where = {"provider": provider} if provider else None

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            ids = [f"retrieved-{index}" for index, _ in enumerate(docs)]

        chunks: list[RetrievedChunk] = []
        for chunk_id, doc, metadata, distance in zip(ids, docs, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=doc,
                    metadata=metadata or {},
                    similarity=self._safe_similarity_from_distance(distance),
                )
            )
        return chunks

    def _rerank(self, query: str, chunks: list[RetrievedChunk], keep_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(chunks, scores, strict=False):
            chunk.rerank_score = float(score)
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        positive = [chunk for chunk in chunks if chunk.rerank_score > 0.0]
        if positive:
            return positive[:keep_k]
        return chunks[:keep_k]

    def _extract_by_sentence(
        self,
        text: str,
        query_embedding: np.ndarray,
        per_chunk_sentences: int,
    ) -> list[str]:
        sentences = sentence_split(text)
        if not sentences:
            return []

        sent_embeddings = self.embedder.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sent_embeddings = np.asarray(sent_embeddings, dtype=np.float32)
        sims = np.dot(sent_embeddings, query_embedding)
        top_idx = np.argsort(sims)[::-1][:per_chunk_sentences]
        return [sentences[i] for i in sorted(top_idx)]

    def _extract_by_window(
        self,
        text: str,
        query_embedding: np.ndarray,
        window_size: int = 4,
        stride: int = 1,
    ) -> list[str]:
        sentences = sentence_split(text)
        if not sentences:
            return []
        if len(sentences) <= window_size:
            return [" ".join(sentences)]

        windows: list[str] = []
        for i in range(0, max(1, len(sentences) - window_size + 1), stride):
            windows.append(" ".join(sentences[i : i + window_size]))
        if not windows:
            windows = [" ".join(sentences)]

        window_embeddings = self.embedder.encode(
            windows,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        window_embeddings = np.asarray(window_embeddings, dtype=np.float32)
        sims = np.dot(window_embeddings, query_embedding)
        best_idx = int(np.argmax(sims))
        return [windows[best_idx]]

    def _compress_with_sentences(
        self,
        query_embedding: np.ndarray,
        chunks: list[RetrievedChunk],
        per_chunk_sentences: int,
    ) -> list[RetrievedChunk]:
        for chunk in chunks:
            doc_type = str(chunk.metadata.get("doc_type", "")).lower()
            has_steps = bool(chunk.metadata.get("has_numbered_steps", False))

            if doc_type in {"manual", "troubleshooting"}:
                if has_steps and token_count(chunk.text) <= 320:
                    chunk.selected_sentences = [chunk.text]
                else:
                    chunk.selected_sentences = self._extract_by_window(
                        text=chunk.text,
                        query_embedding=query_embedding,
                        window_size=4,
                        stride=1,
                    )
            else:
                chunk.selected_sentences = self._extract_by_sentence(
                    text=chunk.text,
                    query_embedding=query_embedding,
                    per_chunk_sentences=per_chunk_sentences,
                )
        return chunks

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            provider = chunk.metadata.get("provider", "unknown")
            doc_type = chunk.metadata.get("doc_type", "unknown")
            filename = chunk.metadata.get("filename", "unknown")
            excerpt = " ".join(chunk.selected_sentences or []) if chunk.selected_sentences else chunk.text
            parts.append(
                f"[Excerpt {i}] source={source} provider={provider} doc_type={doc_type} file={filename}\n{excerpt}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        with httpx.Client(timeout=40.0) as client:
            response = client.post(self.llm_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return "I don't have enough information to answer that."

    def answer(
        self,
        query: str,
        top_k: int,
        rerank_k: int,
        per_chunk_sentences: int,
        use_llm: bool,
    ) -> dict[str, Any]:
        query_embedding = self._embed_query(query)

        cache_result = self.cache.get(query_embedding)
        if cache_result is not None:
            return cache_result

        provider = detect_provider_in_query(query)
        retrieved = self._query_chroma(query_embedding, provider=provider, top_k=top_k)

        if provider and not retrieved:
            retrieved = self._query_chroma(query_embedding, provider=None, top_k=top_k)

        best_similarity = max((chunk.similarity for chunk in retrieved), default=0.0)
        if best_similarity < RELEVANCE_THRESHOLD:
            fallback = {
                "query": query,
                "provider_filter": provider,
                "best_similarity": best_similarity,
                "answer": "I don't have enough information to answer that.",
                "context": "",
                "retrieved": [],
                "cache_hit": False,
            }
            self.cache.put(query, query_embedding, fallback)
            return fallback

        reranked = self._rerank(query, retrieved, keep_k=rerank_k)
        compressed = self._compress_with_sentences(
            query_embedding=query_embedding,
            chunks=reranked,
            per_chunk_sentences=per_chunk_sentences,
        )
        context = self._format_context(compressed)

        prompt = GROUNDING_PROMPT_TEMPLATE.format(
            retrieved_context=context,
            user_query=query,
        )

        if use_llm:
            answer_text = self._call_llm(prompt)
        else:
            answer_text = "[LLM disabled] Retrieved supporting context is returned for inspection."

        result = {
            "query": query,
            "provider_filter": provider,
            "best_similarity": best_similarity,
            "answer": answer_text,
            "context": context,
            "retrieved": [
                {
                    "chunk_id": c.chunk_id,
                    "similarity": c.similarity,
                    "rerank_score": c.rerank_score,
                    "metadata": c.metadata,
                    "selected_sentences": c.selected_sentences or [],
                }
                for c in compressed
            ],
            "cache_hit": False,
        }

        self.cache.put(query, query_embedding, result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone RAG inference runner")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVE_TOP_K)
    parser.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--sentences-per-chunk", type=int, default=DEFAULT_SENTENCES_PER_CHUNK)
    parser.add_argument("--llm-endpoint", type=str, default=DEFAULT_LLAMACPP_ENDPOINT)
    parser.add_argument("--llm-model", type=str, default="tinyllama")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM call, only return retrieval data")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    return parser.parse_args()


def render_human_output(payload: dict[str, Any]) -> str:
    lines = [
        f"Answer: {payload['answer']}",
        f"Best similarity: {payload['best_similarity']:.4f}",
        f"Provider filter: {payload.get('provider_filter')}",
        f"Cache hit: {payload.get('cache_hit', False)}",
        "",
        "Top retrieved chunks:",
    ]

    retrieved = payload.get("retrieved", [])
    for idx, item in enumerate(retrieved, start=1):
        meta = item.get("metadata", {})
        lines.append(
            f"  {idx}. provider={meta.get('provider')} source={meta.get('source')} "
            f"doc_type={meta.get('doc_type')} file={meta.get('filename')} "
            f"sim={item.get('similarity', 0):.4f} rerank={item.get('rerank_score', 0):.4f}"
        )

    context = payload.get("context", "")
    if context:
        preview = re.sub(r"\s+", " ", context).strip()[:600]
        lines.extend(["", f"Context preview: {preview}..."])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    engine = RAGEngine(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
    payload = engine.answer(
        query=args.query,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        per_chunk_sentences=args.sentences_per_chunk,
        use_llm=not args.no_llm,
    )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_human_output(payload))


if __name__ == "__main__":
    main()
