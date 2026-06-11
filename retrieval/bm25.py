"""
BM25 sparse retrieval index over chunks + per-doc metadata.

Indexable text per chunk = chunk text + title + web_description +
retrieval_keywords + frameworks_referenced + regions + countries.

Tokenisation is lowercase + \\w+ regex split — BM25 cares about term
overlap, not morphology. Build once, save to disk, load fast for
querying.

Usage
-----
Build:
    .venv311/Scripts/python.exe -m retrieval.bm25
        --chunks data/chunks.jsonl
        --documents data/documents.json
        --out data/bm25_index.pkl

Query (smoke test):
    .venv311/Scripts/python.exe -m retrieval.bm25
        --query "Sendai Framework" --top-k 5
"""

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
DEFAULT_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

METADATA_LIST_FIELDS = (
    "retrieval_keywords",
    "frameworks_referenced",
    "regions",
    "countries",
)
METADATA_TEXT_FIELDS = ("title", "web_description")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_indexable_text(
    chunk: dict, doc: dict, expansion_questions: list[str] | None = None,
    include_metadata: bool = True,
) -> str:
    parts = [chunk["text"]]
    # Per-doc metadata (title/desc/keywords/frameworks/regions/countries). Set
    # include_metadata=False for the metadata ablation (text-only sparse index).
    if include_metadata:
        for field in METADATA_TEXT_FIELDS:
            v = (doc.get(field) or "").strip()
            if v:
                parts.append(v)
        for field in METADATA_LIST_FIELDS:
            v = doc.get(field) or []
            if isinstance(v, list):
                parts.extend(str(x) for x in v if x)
            elif v:
                parts.append(str(v))
    # doc2query augmentation: append the LLM-generated questions this chunk
    # answers, so the sparse channel can match queries by their vocabulary.
    if expansion_questions:
        parts.extend(expansion_questions)
    return " ".join(parts)


class BM25Index:
    def __init__(self, bm25: BM25Okapi, chunk_meta: list[dict]):
        self.bm25 = bm25
        self.chunk_meta = chunk_meta

    @classmethod
    def build(
        cls,
        chunks_path: Path,
        documents_path: Path,
        min_tokens: int = 30,
        expansions: dict[str, list[str]] | None = None,
        include_metadata: bool = True,
    ) -> "BM25Index":
        log.info(f"Loading docs index from {documents_path}")
        with open(documents_path, encoding="utf-8") as f:
            docs = {d["slug"]: d for d in json.load(f)}
        log.info(f"  {len(docs):,} doc entries")
        if not include_metadata:
            log.info("TEXT-ONLY index — per-doc metadata EXCLUDED (metadata ablation)")
        if expansions:
            log.info(f"doc2query expansions for {len(expansions):,} chunks")

        log.info(f"Loading chunks from {chunks_path}")
        chunk_meta: list[dict] = []
        corpus_tokens: list[list[str]] = []
        skipped_min = skipped_missing = skipped_empty = 0
        augmented = 0

        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["token_count"] < min_tokens:
                    skipped_min += 1
                    continue
                doc = docs.get(r["slug"])
                if doc is None:
                    skipped_missing += 1
                    continue
                exp = expansions.get(r["chunk_id"]) if expansions else None
                if exp:
                    augmented += 1
                tokens = tokenize(build_indexable_text(r, doc, exp, include_metadata))
                if not tokens:
                    skipped_empty += 1
                    continue
                chunk_meta.append({
                    "chunk_id": r["chunk_id"],
                    "slug": r["slug"],
                    "page": r["page"],
                    "title": doc.get("title"),
                    "text": r["text"],
                })
                corpus_tokens.append(tokens)

        log.info(f"  indexed   : {len(corpus_tokens):,}")
        log.info(f"  skipped <{min_tokens}t : {skipped_min:,}")
        log.info(f"  skipped no-doc      : {skipped_missing:,}")
        log.info(f"  skipped empty-tok   : {skipped_empty:,}")
        if expansions:
            log.info(f"  doc2query-augmented : {augmented:,}")
        log.info("Building BM25Okapi…")
        bm25 = BM25Okapi(corpus_tokens)
        return cls(bm25, chunk_meta)

    def save(self, path: Path) -> None:
        log.info(f"Saving BM25 index → {path}")
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_meta": self.chunk_meta}, f, protocol=4)
        log.info(f"  size: {path.stat().st_size / 1e6:.1f} MB")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        log.info(f"Loading BM25 index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        log.info(f"  {len(data['chunk_meta']):,} chunks loaded")
        return cls(data["bm25"], data["chunk_meta"])

    def query(self, query: str, top_k: int = 50) -> list[dict]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        n = len(scores)
        if top_k >= n:
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            top_idx = part[np.argsort(-scores[part])]
        out = []
        for rank, i in enumerate(top_idx, 1):
            entry = dict(self.chunk_meta[i])
            entry["rank"] = rank
            entry["score"] = float(scores[i])
            out.append(entry)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    ap.add_argument("--documents", default=str(DOCUMENTS_PATH))
    ap.add_argument("--out", default=str(DEFAULT_INDEX_PATH))
    ap.add_argument("--min-tokens", type=int, default=30)
    ap.add_argument("--expansions", default=None,
                    help="doc2query expansions JSON ({chunk_id: [questions]}); "
                         "appends generated questions to each chunk's indexable text. "
                         "Use a distinct --out (e.g. data/bm25_doc2query.pkl).")
    ap.add_argument("--no-metadata", action="store_true",
                    help="index chunk TEXT ONLY (exclude per-doc metadata) for the "
                         "metadata ablation; use a distinct --out (e.g. data/bm25_textonly.pkl)")
    ap.add_argument("--query", help="smoke-test query against an existing index (skips build)")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    if args.query:
        idx = BM25Index.load(Path(args.out))
        for h in idx.query(args.query, top_k=args.top_k):
            print(f"  {h['rank']:>2}. [{h['score']:.3f}] {h['title']}, p.{h['page']}  ({h['chunk_id']})")
        return 0

    expansions = None
    if args.expansions:
        with open(args.expansions, encoding="utf-8") as f:
            expansions = json.load(f)

    idx = BM25Index.build(
        Path(args.chunks), Path(args.documents),
        min_tokens=args.min_tokens, expansions=expansions,
        include_metadata=not args.no_metadata,
    )
    idx.save(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
