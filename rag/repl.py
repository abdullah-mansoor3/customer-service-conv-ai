from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_LLAMACPP_ENDPOINT,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
    DEFAULT_SENTENCES_PER_CHUNK,
)
from rag.inference import RAGEngine, render_human_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive standalone RAG shell")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVE_TOP_K)
    parser.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--sentences-per-chunk", type=int, default=DEFAULT_SENTENCES_PER_CHUNK)
    parser.add_argument("--llm-endpoint", type=str, default=DEFAULT_LLAMACPP_ENDPOINT)
    parser.add_argument("--llm-model", type=str, default="tinyllama")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = RAGEngine(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )

    print("Standalone RAG REPL started. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        payload = engine.answer(
            query=query,
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
