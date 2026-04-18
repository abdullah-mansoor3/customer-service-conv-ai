from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Chunk:
    text: str
    section: str | None = None
    page: int | None = None
    has_numbered_steps: bool = False


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def sentence_split(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if s.strip()]


def contains_numbered_steps_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    step_lines = [
        line
        for line in lines
        if re.match(r"^(?:\d+[.)]|step\s+\d+[:.)-])\s+", line, flags=re.I)
    ]
    if len(step_lines) < 2:
        return False
    return (len(step_lines) / max(1, len(lines))) >= 0.12


def token_window_split(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def split_text_preserving_numbered_lists(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    preserve_numbered_steps: bool = False,
) -> list[tuple[str, bool]]:
    clean = normalize_whitespace(text)
    if not clean:
        return []

    if not preserve_numbered_steps:
        return [(part, contains_numbered_steps_text(part)) for part in token_window_split(clean, max_tokens, overlap_tokens)]

    step_start_re = re.compile(r"^\s*(?:\d+[.)]|step\s+\d+[:.)-])\s+", re.I)
    lines = clean.splitlines()
    raw_segments: list[tuple[str, bool]] = []
    plain_buffer: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if not step_start_re.match(line):
            plain_buffer.append(line)
            index += 1
            continue

        if plain_buffer:
            plain_text = normalize_whitespace("\n".join(plain_buffer))
            if plain_text:
                raw_segments.append((plain_text, False))
            plain_buffer = []

        step_lines: list[str] = []
        step_count = 0
        while index < len(lines):
            current = lines[index]
            if step_start_re.match(current):
                step_count += 1
                step_lines.append(current)
                index += 1
                continue

            if not current.strip():
                step_lines.append(current)
                index += 1
                if index < len(lines) and step_start_re.match(lines[index]):
                    continue
                break

            if step_lines:
                step_lines.append(current)
                index += 1
                continue

            break

        step_text = normalize_whitespace("\n".join(step_lines))
        if step_text and step_count >= 2:
            raw_segments.append((step_text, True))
        elif step_text:
            raw_segments.append((step_text, False))

    if plain_buffer:
        plain_text = normalize_whitespace("\n".join(plain_buffer))
        if plain_text:
            raw_segments.append((plain_text, False))

    final_segments: list[tuple[str, bool]] = []
    for segment_text, is_step in raw_segments:
        if is_step:
            final_segments.append((segment_text, True))
            continue

        if token_count(segment_text) <= max_tokens:
            final_segments.append((segment_text, False))
            continue

        for part in token_window_split(segment_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
            final_segments.append((part, False))

    return final_segments


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    lines = markdown_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_header = "Document"
    current_lines: list[str] = []

    header_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for line in lines:
        match = header_re.match(line)
        if match:
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_lines))

    normalized: list[tuple[str, str]] = []
    for header, content_lines in sections:
        body = normalize_whitespace("\n".join(content_lines))
        if body:
            normalized.append((header, body))
    return normalized


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    qa_pairs: list[tuple[str, str]] = []

    q_re = re.compile(r"^(?:Q\s*[:\-]|Question\s*[:\-])\s*(.+)", re.I)
    a_re = re.compile(r"^(?:A\s*[:\-]|Answer\s*[:\-])\s*(.+)", re.I)

    current_q: str | None = None
    current_a_parts: list[str] = []

    for line in lines:
        if not line:
            continue
        q_match = q_re.match(line)
        a_match = a_re.match(line)

        if q_match:
            if current_q and current_a_parts:
                qa_pairs.append((current_q, normalize_whitespace(" ".join(current_a_parts))))
            current_q = q_match.group(1).strip()
            current_a_parts = []
            continue

        if a_match and current_q:
            current_a_parts.append(a_match.group(1).strip())
            continue

        if current_q:
            current_a_parts.append(line)

    if current_q and current_a_parts:
        qa_pairs.append((current_q, normalize_whitespace(" ".join(current_a_parts))))

    return [(q, a) for q, a in qa_pairs if q and a]


def markdown_to_chunks(
    markdown_text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    preserve_numbered_steps: bool = False,
) -> list[Chunk]:
    markdown_text = normalize_whitespace(markdown_text)
    if not markdown_text:
        return []

    qa_pairs = extract_qa_pairs(markdown_text)
    chunks: list[Chunk] = []

    if qa_pairs:
        for q, a in qa_pairs:
            qa_text = f"Q: {q}\nA: {a}"
            if token_count(qa_text) <= max_tokens:
                chunks.append(Chunk(text=qa_text, section=f"FAQ: {q}", has_numbered_steps=contains_numbered_steps_text(qa_text)))
                continue
            subchunks = token_window_split(a, max_tokens=max_tokens - 20, overlap_tokens=overlap_tokens)
            for sub in subchunks:
                chunk_text = f"Q: {q}\nA: {sub}"
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        section=f"FAQ: {q}",
                        has_numbered_steps=contains_numbered_steps_text(chunk_text),
                    )
                )
        return chunks

    for header, section_body in split_markdown_sections(markdown_text):
        header_prefix = f"[Section: {header}]\n"
        if token_count(section_body) <= max_tokens:
            body_text = header_prefix + section_body
            chunks.append(
                Chunk(
                    text=body_text,
                    section=header,
                    has_numbered_steps=contains_numbered_steps_text(body_text),
                )
            )
            continue

        segments = split_text_preserving_numbered_lists(
            section_body,
            max_tokens=max_tokens - 10,
            overlap_tokens=overlap_tokens,
            preserve_numbered_steps=preserve_numbered_steps,
        )
        for sub, has_steps in segments:
            chunks.append(Chunk(text=header_prefix + sub, section=header, has_numbered_steps=has_steps))

    return chunks


def pdf_page_to_chunks(
    page_text: str,
    doc_title: str,
    page_number: int,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    preserve_numbered_steps: bool = False,
) -> list[Chunk]:
    clean = normalize_whitespace(page_text)
    if not clean:
        return []

    prefix = f"[Document: {doc_title} | Page: {page_number}]\n"
    if token_count(clean) <= max_tokens:
        full_text = prefix + clean
        return [Chunk(text=full_text, page=page_number, has_numbered_steps=contains_numbered_steps_text(full_text))]

    out: list[Chunk] = []
    segments = split_text_preserving_numbered_lists(
        clean,
        max_tokens=max_tokens - 12,
        overlap_tokens=overlap_tokens,
        preserve_numbered_steps=preserve_numbered_steps,
    )
    for sub, has_steps in segments:
        out.append(Chunk(text=prefix + sub, page=page_number, has_numbered_steps=has_steps))
    return out


def flatten[T](items: Iterable[list[T]]) -> list[T]:
    result: list[T] = []
    for group in items:
        result.extend(group)
    return result
