from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk, markdown_to_chunks, pdf_page_to_chunks
from rag.config import (
    CHROMA_DIR,
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAG_DATA_DIR,
)
from rag.metadata import infer_doc_type, infer_provider, provider_to_source, safe_filename


def iter_documents(rag_data_dir: Path) -> list[Path]:
    files = [
        p
        for p in rag_data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".pdf"}
    ]
    return sorted(files)


def read_md_chunks(
    path: Path,
    max_tokens: int,
    overlap_tokens: int,
    preserve_numbered_steps: bool,
) -> list[Chunk]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return markdown_to_chunks(
        content,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        preserve_numbered_steps=preserve_numbered_steps,
    )


def read_pdf_chunks(
    path: Path,
    max_tokens: int,
    overlap_tokens: int,
    preserve_numbered_steps: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    with fitz.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or path.stem
        for page_index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            chunks.extend(
                pdf_page_to_chunks(
                    page_text,
                    doc_title=title,
                    page_number=page_index,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    preserve_numbered_steps=preserve_numbered_steps,
                )
            )
    return chunks


def build_metadata(path: Path, chunk: Chunk, chunk_index: int, file_doc_type: str) -> dict[str, Any]:
    preview = chunk.text[:500]
    provider = infer_provider(path.name, preview)
    return {
        "source": provider_to_source(provider, path.name),
        "provider": provider,
        "doc_type": file_doc_type or infer_doc_type(path.name, preview),
        "filename": safe_filename(path),
        "extension": path.suffix.lower().lstrip("."),
        "chunk_index": chunk_index,
        "section": chunk.section or "",
        "page": chunk.page or 0,
        "has_numbered_steps": bool(chunk.has_numbered_steps),
    }


def chunk_id(path: Path, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{path.name}:{chunk_index}:{text[:80]}".encode("utf-8")).hexdigest()[:16]
    stem = path.stem.lower().replace(" ", "-")
    return f"{stem}-{chunk_index:05d}-{digest}"


def get_collection(client: chromadb.PersistentClient, reset: bool = False):
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "ISP support RAG collection"},
    )


def flush_batch(
    collection,
    embedder: SentenceTransformer,
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
) -> None:
    if not ids:
        return

    vectors = embedder.encode(
        docs,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=vectors,
    )


def ingest(
    rag_data_dir: Path,
    chroma_dir: Path,
    max_tokens: int,
    overlap_tokens: int,
    reset: bool,
    batch_size: int,
) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    files = iter_documents(rag_data_dir)
    print(f"Found {len(files)} supported docs in {rag_data_dir}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = get_collection(client, reset=reset)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    id_batch: list[str] = []
    doc_batch: list[str] = []
    meta_batch: list[dict[str, Any]] = []

    total_chunks = 0
    for file_path in files:
        file_doc_type = infer_doc_type(file_path.name, "")
        preserve_numbered_steps = file_doc_type in {"manual", "troubleshooting"}

        try:
            if file_path.suffix.lower() == ".md":
                chunks = read_md_chunks(
                    file_path,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    preserve_numbered_steps=preserve_numbered_steps,
                )
            else:
                chunks = read_pdf_chunks(
                    file_path,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    preserve_numbered_steps=preserve_numbered_steps,
                )
        except Exception as exc:
            print(f"[WARN] Skipped {file_path.name}: {exc}")
            continue

        if not chunks:
            continue

        for index, chunk in enumerate(chunks):
            doc_text = chunk.text.strip()
            if not doc_text:
                continue
            id_batch.append(chunk_id(file_path, index, doc_text))
            doc_batch.append(doc_text)
            meta_batch.append(build_metadata(file_path, chunk, index, file_doc_type=file_doc_type))
            total_chunks += 1

            if len(id_batch) >= batch_size:
                flush_batch(collection, embedder, id_batch, doc_batch, meta_batch)
                id_batch.clear()
                doc_batch.clear()
                meta_batch.clear()

        print(f"Indexed {file_path.name}: {len(chunks)} chunks")

    flush_batch(collection, embedder, id_batch, doc_batch, meta_batch)

    print(f"Total chunks indexed: {total_chunks}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma path: {chroma_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local ISP docs into ChromaDB")
    parser.add_argument("--rag-data-dir", type=Path, default=RAG_DATA_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--max-tokens", type=int, default=CHUNK_TOKEN_SIZE)
    parser.add_argument("--overlap-tokens", type=int, default=CHUNK_TOKEN_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(
        rag_data_dir=args.rag_data_dir,
        chroma_dir=args.chroma_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        reset=args.reset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
