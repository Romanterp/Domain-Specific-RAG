"""
Page-boundary chunking for RAG retrieval.

Reads data/texts/*.pages.json (produced by extraction/text_extract.py)
and writes data/chunks.jsonl with one JSON object per chunk.

Chunk policy
------------
- One chunk per page if the page fits within CHUNK_MAX_TOKENS.
- Longer pages are split on paragraph boundaries (\\n\\n), greedy-packed
  up to CHUNK_MAX_TOKENS.
- A paragraph over the cap falls back to sentence splitting; a sentence
  over the cap falls back to token-window splitting (last resort).
- Sub-chunks from the same page keep the page number — citations stay
  exact ("Title, p. 14"), even when a page produces multiple chunks.
- No overlap. Overlap is a separate ablation condition, not a default.

Tokeniser matches the embedder (BGE-M3) so token counts are comparable
to the retrieval context window.

Output record schema
--------------------
    {
        "chunk_id": "<slug>:<page>:<sub_index>",
        "slug": "...",
        "page": 14,
        "sub_index": 0,
        "text": "...",
        "token_count": 487,
    }

Only English documents with text_extracted=True are processed.
"""

import json
import logging
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEXT_DIR = DATA_DIR / "texts"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

TOKENIZER_NAME = "BAAI/bge-m3"
CHUNK_MAX_TOKENS = 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def force_split_by_tokens(tokenizer, text: str, cap: int) -> list[str]:
    """Last-resort split for text with no paragraph or sentence structure
    (URL dumps, tables extracted as one blob). Slices by token window."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= cap:
        return [text]
    out = []
    for i in range(0, len(ids), cap):
        out.append(tokenizer.decode(ids[i:i + cap], skip_special_tokens=True))
    return out


def split_paragraph_by_sentences(tokenizer, text: str, cap: int) -> list[str]:
    """Fallback for paragraphs larger than cap. Splits on sentence punctuation
    and packs sentences up to cap; any single sentence larger than cap is
    force-split by token windows."""
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks, current, current_tokens = [], [], 0
    for s in sents:
        t = _token_len(tokenizer, s)
        if t > cap:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(force_split_by_tokens(tokenizer, s, cap))
            continue
        if current and current_tokens + t > cap:
            chunks.append(" ".join(current))
            current, current_tokens = [s], t
        else:
            current.append(s)
            current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_page(tokenizer, text: str) -> list[str]:
    """Split one page into one or more chunks."""
    if _token_len(tokenizer, text) <= CHUNK_MAX_TOKENS:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks, current, current_tokens = [], [], 0

    for p in paragraphs:
        p_tokens = _token_len(tokenizer, p)

        if p_tokens > CHUNK_MAX_TOKENS:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(split_paragraph_by_sentences(tokenizer, p, CHUNK_MAX_TOKENS))
            continue

        if current and current_tokens + p_tokens > CHUNK_MAX_TOKENS:
            chunks.append("\n\n".join(current))
            current, current_tokens = [p], p_tokens
        else:
            current.append(p)
            current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_all():
    log.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        documents = json.load(f)

    eligible = [
        d for d in documents
        if d.get("language", "en") == "en" and d.get("text_extracted")
    ]
    log.info(f"{len(eligible)} English documents with extracted text")

    total_chunks, skipped, long_pages = 0, 0, 0
    per_doc_chunks = []

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for i, doc in enumerate(eligible):
            slug = doc["slug"]
            pages_file = TEXT_DIR / f"{slug}.pages.json"
            if not pages_file.exists():
                skipped += 1
                continue

            try:
                with open(pages_file, encoding="utf-8") as f:
                    pages = json.load(f)
            except Exception as e:
                log.warning(f"{slug}: failed to load pages.json — {e}")
                skipped += 1
                continue

            doc_chunks = 0
            for entry in pages:
                page_num = entry.get("page")
                page_text = (entry.get("text") or "").strip()
                if not page_text:
                    continue

                sub_chunks = chunk_page(tokenizer, page_text)
                if len(sub_chunks) > 1:
                    long_pages += 1

                for sub_i, chunk_text in enumerate(sub_chunks):
                    record = {
                        "chunk_id": f"{slug}:{page_num}:{sub_i}",
                        "slug": slug,
                        "page": page_num,
                        "sub_index": sub_i,
                        "text": chunk_text,
                        "token_count": _token_len(tokenizer, chunk_text),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1
                    doc_chunks += 1

            per_doc_chunks.append(doc_chunks)

            if (i + 1) % 200 == 0:
                log.info(f"  {i + 1}/{len(eligible)} docs, {total_chunks} chunks")

    log.info(
        f"Done: {total_chunks} chunks from {len(eligible) - skipped} docs "
        f"({skipped} skipped)"
    )
    if per_doc_chunks:
        per_doc_chunks.sort()
        n = len(per_doc_chunks)
        log.info(
            f"Chunks per doc — p50: {per_doc_chunks[n // 2]}, "
            f"p90: {per_doc_chunks[int(n * 0.9)]}, max: {per_doc_chunks[-1]}"
        )
    log.info(f"Pages that produced >1 chunk: {long_pages}")


if __name__ == "__main__":
    sys.exit(chunk_all())
