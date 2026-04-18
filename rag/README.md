# Standalone RAG Module

This folder contains a **standalone** RAG pipeline that reads documents from `../rag_data`, builds a Chroma index, and runs retrieval + grounded answer generation.

It is intentionally separate from the chatbot runtime so you can validate quality first.

---

## 1) What This Module Does

Two phases:

1. **Ingestion phase** (`python -m rag.ingest`)
   - Reads `.md` + `.pdf` docs
   - Splits into chunks with metadata
   - Embeds chunks using MiniLM
   - Stores vectors + metadata in ChromaDB on disk

2. **Inference phase** (`python -m rag.inference` / `python -m rag.repl`)
   - Detects provider from user query (PTCL, Nayatel, TP-Link, Xfinity, etc.)
   - Retrieves relevant chunks from Chroma (provider-filtered when possible)
   - Reranks results using cross-encoder
   - Compresses context (doc-type aware)
   - Applies similarity guardrail to prevent hallucinated answers
   - Optionally calls llama.cpp/OpenAI-compatible endpoint for final answer

---

## 2) Models and Storage

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Vector DB: ChromaDB (persistent local store)
- Chroma path (default): `./rag/chroma_store`
- Chroma collection (default): `isp_support_knowledge`

---

## 3) Ingestion Logic (Exactly What Happens)

### 3.1 Document loading

- `.md` files: parsed as markdown text
- `.pdf` files: parsed with PyMuPDF (`fitz`) page-by-page

### 3.2 Chunking strategy

- Base chunk size: ~400 tokens
- Overlap: 50 tokens

For markdown:
- Split by markdown headers first (`#`, `##`, `###`, ...)
- Detect FAQ Q/A blocks and keep Q/A together where possible

For PDF:
- Prefix each chunk with document + page context
- Keep page traceability through metadata

For procedural content:
- If doc type is `manual` or `troubleshooting`, ingestion tries to preserve numbered step blocks
- Numbered step chunks are marked with `has_numbered_steps=true` in metadata

### 3.3 Metadata per chunk

Each chunk carries metadata like:

```json
{
  "source": "nayatel.com",
  "provider": "nayatel",
  "doc_type": "faq",
  "filename": "nayatel-com-faqs-187476fa3c.md",
  "extension": "md",
  "chunk_index": 5,
  "section": "Find Answers to Your Questions",
  "page": 0,
  "has_numbered_steps": false
}
```

### 3.4 Embedding + upsert

- Chunks are embedded in batches
- Vectors + docs + metadata are upserted into Chroma

---

## 4) Inference Logic (Exactly What Happens)

Given a query, inference does:

1. **Embed query** with MiniLM
2. **Hot semantic cache lookup** (LRU, cosine threshold)
   - If cache hit, return cached result instantly
3. **Provider detection** from query text
   - If detected, query Chroma with metadata filter: `where={"provider": ...}`
4. **Vector retrieval** (`top-k`, default 10)
5. **Rerank** with cross-encoder, keep `rerank-k` (default 3)
6. **Doc-type-aware compression**
   - `faq/guide/policy` → sentence-level top-N extraction
   - `manual/troubleshooting` → sliding sentence-window extraction
   - `manual/troubleshooting` + `has_numbered_steps=true` → preserve step block when reasonably sized
7. **Guardrail check**
   - If best similarity < 0.30 → return: `I don't have enough information to answer that.`
8. **Grounded prompt** build + optional LLM call
9. Save response in hot semantic cache

---

## 5) Anti-Hallucination Controls

- Retrieval threshold gate (`best_similarity < 0.30` => no-answer fallback)
- Strict grounded prompt (“use only provided context”)
- Source/provider metadata included in context excerpts
- Reranking + compression reduces unrelated context noise

---

## 6) Installation

From project root:

```bash
/opt/miniconda3/bin/python -m pip install -r rag/requirements.txt
```

---

## 7) Commands

### 7.1 Rebuild index from scratch

```bash
/opt/miniconda3/bin/python -m rag.ingest --reset
```

### 7.2 Retrieval-only test (no LLM call)

```bash
/opt/miniconda3/bin/python -m rag.inference "My Nayatel connection keeps dropping" --no-llm --json
```

### 7.3 Inference with llama.cpp

Ensure llama.cpp server is running at `/v1/chat/completions`, then:

```bash
/opt/miniconda3/bin/python -m rag.inference "How do I reset my router?" --llm-endpoint http://localhost:8080/v1/chat/completions --llm-model tinyllama
```

### 7.4 Interactive shell (best for cache testing)

```bash
/opt/miniconda3/bin/python -m rag.repl --no-llm
```

Try semantically similar queries back-to-back to confirm cache hits.

---

## 8) Flag Reference

### Ingest flags

- `--reset`:
  - Deletes existing collection and recreates it
  - Use this when chunking/metadata logic changes
- `--rag-data-dir`:
  - Input docs folder (default `./rag_data`)
- `--chroma-dir`:
  - Output Chroma persistence directory
- `--max-tokens`:
  - Target chunk size
- `--overlap-tokens`:
  - Overlap between adjacent split chunks
- `--batch-size`:
  - Number of chunks embedded/upserted per batch

### Inference flags

- `query` (positional): user question text
- `--chroma-dir`: where Chroma index is stored
- `--collection`: Chroma collection name
- `--top-k`: initial retrieval count from vector search
- `--rerank-k`: final kept chunks after cross-encoder rerank
- `--sentences-per-chunk`: sentence budget for non-manual compression
- `--no-llm`: retrieval-only mode
- `--json`: prints full JSON output
- `--llm-endpoint`: OpenAI-compatible chat-completions endpoint
- `--llm-model`: model field sent to LLM endpoint
- `--llm-api-key`: optional bearer token (defaults to `OPENAI_API_KEY` if set)

---

## 9) Reading Output

Inference returns:

- `answer`: final text (or retrieval-only placeholder)
- `best_similarity`: best retrieved similarity score
- `provider_filter`: detected provider used for metadata filtering
- `cache_hit`: whether semantic hot cache answered directly
- `retrieved`: each kept chunk with metadata + similarity + rerank score
- `context`: compressed context actually passed into prompt

---

## 10) Quick Validation Checklist

1. Run ingest with `--reset`
2. Run inference in `--no-llm --json` mode
3. Verify:
   - provider filter is correct
   - retrieved docs are source-relevant
   - best similarity is sensible (not near zero)
4. Run same/near-same query in `rag.repl` and verify cache behavior
5. Enable LLM only after retrieval quality looks good
