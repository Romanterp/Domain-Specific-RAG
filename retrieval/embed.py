"""
BGE-M3 dense embedding + Qdrant indexing for Theisus RAG.

Reads data/chunks.jsonl and joins per-doc metadata from data/documents.json.
Embeds chunk text with BGE-M3 (dense only for baseline — sparse/multi-vector
deferred to the hybrid ablation) and upserts into a Qdrant collection with
full metadata payload.

The same chunks.jsonl can be re-embedded under multiple context modes into
separate collections; chunks.jsonl itself is never rewritten.

Ablation surface (CLI)
----------------------
--context-mode  none | title | title+desc | title+desc+keywords | full
    Controls the text passed to the embedder.
      none                   -> chunk text as-is
      title                  -> "{title}\\n\\n{chunk_text}"
      title+desc             -> "{title}\\n{web_description}\\n\\n{chunk_text}"
      title+desc+keywords    -> adds "Keywords: k1, k2, …" line
      full                   -> adds Keywords + Frameworks + Regions + Countries
    Missing fields are skipped; falls back gracefully. Default: none.

--min-tokens N
    Drop chunks below N tokens. These are cover pages, TOC fragments,
    page-number-only pages — retrieval noise. Default 30.

--collection NAME
    Qdrant collection name. Defaults to theisus_<mode>.

--batch-size N
    Embedding batch size. Default 16 (RTX 4080 @ 16 GB, up-to-1024-token
    chunks).

--qdrant-path PATH
    Local Qdrant storage path. Default data/qdrant.

--resume
    Skip chunks whose point ID is already in the collection.

--dry-run
    Report counts, skip model loading and writes.

Idempotence
-----------
Point IDs are uuid5(NAMESPACE, chunk_id) — deterministic, so repeated runs
upsert in place rather than duplicating.
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
DEFAULT_QDRANT_PATH = DATA_DIR / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
# BGE-M3 main branch only ships pytorch_model.bin. Transformers 5.x refuses
# to load .bin files with torch < 2.6 (CVE-2025-32434). Community PR #130
# adds safetensors; pinning to it sidesteps the torch upgrade.
MODEL_REVISION = "refs/pr/130"
DENSE_DIM = 1024

NAMESPACE = uuid.UUID("2b4e5f9a-1c3d-4e6f-8a9b-0c1d2e3f4a5b")

PAYLOAD_DOC_FIELDS = [
    "title", "document_type", "publication_date", "source_url",
    "filename", "pdf_download_url", "page_count", "word_count",
    "web_description", "organizations", "countries", "regions",
    "frameworks_referenced", "temporal_coverage", "retrieval_keywords",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_docs_index() -> dict[str, dict]:
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        documents = json.load(f)
    return {d["slug"]: d for d in documents}


def _fmt_list(label: str, values) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        joined = ", ".join(str(v) for v in values if v)
        return f"{label}: {joined}" if joined else ""
    return f"{label}: {values}"


def build_embed_input(chunk: dict, doc: dict, mode: str) -> str:
    body = chunk["text"]
    if mode == "none":
        return body

    title = (doc.get("title") or "").strip()
    desc = (doc.get("web_description") or "").strip()

    header: list[str] = []
    if title:
        header.append(title)
    if mode == "title":
        return f"{title}\n\n{body}" if title else body

    if desc and mode in ("title+desc", "title+desc+keywords", "full"):
        header.append(desc)

    if mode in ("title+desc+keywords", "full"):
        kw = _fmt_list("Keywords", doc.get("retrieval_keywords"))
        if kw:
            header.append(kw)

    if mode == "full":
        for label, field in [
            ("Frameworks", "frameworks_referenced"),
            ("Regions", "regions"),
            ("Countries", "countries"),
        ]:
            line = _fmt_list(label, doc.get(field))
            if line:
                header.append(line)

    if mode not in ("title+desc", "title+desc+keywords", "full"):
        raise ValueError(f"unknown context mode: {mode}")

    if not header:
        return body
    return "\n".join(header) + "\n\n" + body


def build_payload(chunk: dict, doc: dict) -> dict:
    payload = {
        "chunk_id": chunk["chunk_id"],
        "slug": chunk["slug"],
        "page": chunk["page"],
        "sub_index": chunk["sub_index"],
        "token_count": chunk["token_count"],
        "text": chunk["text"],
    }
    for field in PAYLOAD_DOC_FIELDS:
        value = doc.get(field)
        if value not in (None, "", []):
            payload[field] = value
    return payload


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        count = client.count(collection_name=name, exact=True).count
        log.info(f"Collection {name} exists — {count} points")
        return
    log.info(f"Creating Qdrant collection: {name}  (dim={DENSE_DIM}, distance=COSINE)")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    )


def existing_point_ids(client: QdrantClient, collection: str) -> set[str]:
    log.info(f"Scanning {collection} for resume…")
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=10_000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    log.info(f"  {len(ids):,} already indexed")
    return ids


def iter_eligible(docs: dict, min_tokens: int):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["token_count"] < min_tokens:
                continue
            if r["slug"] not in docs:
                continue
            yield r


def count_eligible(docs: dict, min_tokens: int) -> tuple[int, int, int]:
    eligible, skipped_min, skipped_missing = 0, 0, 0
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["token_count"] < min_tokens:
                skipped_min += 1
                continue
            if r["slug"] not in docs:
                skipped_missing += 1
                continue
            eligible += 1
    return eligible, skipped_min, skipped_missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--context-mode",
        choices=["none", "title", "title+desc", "title+desc+keywords", "full"],
        default="none",
    )
    p.add_argument("--min-tokens", type=int, default=30)
    p.add_argument("--collection", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    collection = args.collection or f"theisus_{args.context_mode.replace('+', '_')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cpu":
        log.warning("CUDA not available — this will be very slow.")

    docs = load_docs_index()
    log.info(f"Loaded {len(docs):,} doc entries")

    log.info(f"Context mode : {args.context_mode}")
    log.info(f"Min tokens   : {args.min_tokens}")
    log.info(f"Collection   : {collection}")
    log.info(f"Batch size   : {args.batch_size}")

    total, skipped_min, skipped_missing = count_eligible(docs, args.min_tokens)
    log.info(f"Eligible chunks : {total:,}")
    log.info(f"  filtered <{args.min_tokens} tokens : {skipped_min:,}")
    log.info(f"  no doc entry                : {skipped_missing:,}")

    if args.dry_run:
        log.info("Dry run — exiting.")
        return 0

    log.info(f"Loading {MODEL_NAME} (revision={MODEL_REVISION}) on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION)

    log.info(f"Opening Qdrant at {args.qdrant_path}")
    client = QdrantClient(path=args.qdrant_path)
    ensure_collection(client, collection)

    already: set[str] = existing_point_ids(client, collection) if args.resume else set()

    batch_texts: list[str] = []
    batch_pids: list[str] = []
    batch_payloads: list[dict] = []
    indexed = 0
    skipped_resume = 0
    t0 = time.time()
    last_log = t0

    def flush():
        nonlocal indexed
        if not batch_texts:
            return
        vectors = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=pl)
            for pid, pl, vec in zip(batch_pids, batch_payloads, vectors)
        ]
        client.upsert(collection_name=collection, points=points)
        indexed += len(points)
        batch_texts.clear()
        batch_pids.clear()
        batch_payloads.clear()

    for chunk in iter_eligible(docs, args.min_tokens):
        pid = point_id_for(chunk["chunk_id"])
        if pid in already:
            skipped_resume += 1
            continue
        doc = docs[chunk["slug"]]
        batch_texts.append(build_embed_input(chunk, doc, args.context_mode))
        batch_pids.append(pid)
        batch_payloads.append(build_payload(chunk, doc))

        if len(batch_texts) >= args.batch_size:
            flush()
            now = time.time()
            if now - last_log > 10:
                elapsed = now - t0
                rate = indexed / elapsed if elapsed else 0
                remaining = total - indexed - skipped_resume
                eta_min = (remaining / rate / 60) if rate else 0
                log.info(
                    f"  {indexed:,}/{total:,}  "
                    f"rate={rate:.1f}/s  eta={eta_min:.1f}min"
                )
                last_log = now

    flush()

    final = client.count(collection_name=collection, exact=True).count
    log.info(
        f"Done — indexed {indexed:,}, skipped-resume {skipped_resume:,}, "
        f"collection total {final:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
