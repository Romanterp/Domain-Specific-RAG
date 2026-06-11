"""
Top-k dense retrieval against a Qdrant collection.

Usage
-----
Single query:
    python -m retrieval.search "your query here"

Batch from file (one query per line, blank lines and # comments ignored):
    python -m retrieval.search --queries example_q.txt

Options
-------
--collection NAME      Qdrant collection (default: theisus_none)
--top-k N              Results per query (default: 5)
--qdrant-path PATH     Qdrant store path (default: data/qdrant)
--jsonl                Emit JSONL records instead of formatted text
--snippet-chars N      Snippet length in characters (default: 240)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_QDRANT_PATH = DATA_DIR / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "refs/pr/130"  # see project_torch_bgem3_gotcha memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_queries(arg_query: str | None, queries_file: str | None) -> list[str]:
    if arg_query:
        return [arg_query]
    if queries_file:
        path = Path(queries_file)
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    log.error("Provide a query as positional arg or --queries FILE")
    sys.exit(2)


def truncate_snippet(text: str, max_chars: int) -> str:
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def format_hit(rank: int, hit, snippet_chars: int) -> str:
    p = hit.payload or {}
    title = p.get("title", "(no title)")
    page = p.get("page", "?")
    snippet = truncate_snippet(p.get("text", ""), snippet_chars)
    return (
        f"  {rank}. [{hit.score:.3f}] {title}, p.{page}\n"
        f"     chunk_id: {p.get('chunk_id')}\n"
        f"     {snippet}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("query", nargs="?", help="single query string")
    ap.add_argument("--queries", help="path to file of queries (one per line)")
    ap.add_argument("--collection", default="theisus_none")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--snippet-chars", type=int, default=240)
    args = ap.parse_args()

    queries = load_queries(args.query, args.queries)
    log.info(f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Loading {MODEL_NAME} (revision={MODEL_REVISION})…")
    model = SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION)

    log.info(f"Opening Qdrant at {args.qdrant_path}")
    client = QdrantClient(path=args.qdrant_path)

    try:
        total = client.count(collection_name=args.collection, exact=True).count
        log.info(f"Collection {args.collection}: {total:,} points")

        query_vecs = model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for q, vec in zip(queries, query_vecs):
            response = client.query_points(
                collection_name=args.collection,
                query=vec.tolist(),
                limit=args.top_k,
                with_payload=True,
                with_vectors=False,
            )
            hits = response.points

            if args.jsonl:
                for rank, h in enumerate(hits, 1):
                    pl = h.payload or {}
                    print(json.dumps({
                        "query": q,
                        "rank": rank,
                        "score": h.score,
                        "chunk_id": pl.get("chunk_id"),
                        "slug": pl.get("slug"),
                        "page": pl.get("page"),
                        "title": pl.get("title"),
                        "snippet": truncate_snippet(pl.get("text", ""), args.snippet_chars),
                    }, ensure_ascii=False))
            else:
                print(f"\nQuery: {q}\n")
                for rank, h in enumerate(hits, 1):
                    print(format_hit(rank, h, args.snippet_chars))
                print()
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
