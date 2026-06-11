"""
Compare retrieval ablation conditions across all collections.

Reads queries, encodes once, queries every collection, then computes
cross-condition agreement metrics (no gold labels — relative comparison
of how condition X reorders results vs baseline).

Outputs
-------
data/eval_results.jsonl   per-query × per-condition top-k records
data/eval_summary.md      markdown summary tables

Usage
-----
.venv311/Scripts/python.exe -m retrieval.evaluate \\
    --queries example_q.txt --top-k 10 \\
    --conditions none title title+desc title+desc+keywords full
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_QDRANT_PATH = DATA_DIR / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "refs/pr/130"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_queries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collection_for(mode: str) -> str:
    return f"theisus_{mode.replace('+', '_')}"


def kendall_tau(a: list, b: list) -> float:
    """Kendall's tau between two ranked lists. Items in a and b are compared
    by index position; lists must contain the same items.
    """
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 0.0
    rank_a = {x: i for i, x in enumerate(a)}
    rank_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            xi, xj = common[i], common[j]
            da = rank_a[xi] - rank_a[xj]
            db = rank_b[xi] - rank_b[xj]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--queries", default="example_q.txt")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["none", "title", "title+desc", "title+desc+keywords", "full"],
    )
    ap.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    ap.add_argument("--baseline", default="none", help="condition to compare others against")
    ap.add_argument("--out-jsonl", default=str(DATA_DIR / "eval_results.jsonl"))
    ap.add_argument("--out-summary", default=str(DATA_DIR / "eval_summary.md"))
    ap.add_argument("--rerank", action="store_true", help="apply BGE-reranker-v2-m3 to candidate pool")
    ap.add_argument("--candidate-pool", type=int, default=50, help="top-N from each retriever before fusion / rerank")
    ap.add_argument("--rerank-batch", type=int, default=32)
    ap.add_argument("--hybrid", action="store_true", help="fuse dense and BM25 channels via RRF")
    ap.add_argument("--bm25-path", default=str(DATA_DIR / "bm25_index.pkl"))
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    queries = load_queries(Path(args.queries))
    log.info(f"{len(queries)} queries from {args.queries}")
    log.info(f"Conditions: {args.conditions}")
    log.info(f"Baseline  : {args.baseline}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {MODEL_NAME} on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION)

    log.info("Encoding queries…")
    qvecs = model.encode(
        queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )

    reranker = None
    if args.rerank:
        from retrieval.rerank import Reranker
        reranker = Reranker(device=device)

    bm25_index = None
    if args.hybrid:
        from retrieval.bm25 import BM25Index
        bm25_index = BM25Index.load(Path(args.bm25_path))

    # Each retriever returns this many candidates before fusion / rerank.
    # Without rerank or hybrid, we go straight to top_k.
    pool_size = args.candidate_pool if (args.rerank or args.hybrid) else args.top_k

    client = QdrantClient(path=args.qdrant_path)

    try:
        # results[query_idx][condition] = list of hit dicts
        results: dict[int, dict[str, list[dict]]] = defaultdict(dict)

        # Pre-compute BM25 per query (independent of context-mode condition).
        bm25_hits_by_q: list[list[dict]] = []
        if bm25_index is not None:
            log.info(f"Running BM25 over {len(queries)} queries…")
            for q in queries:
                bm25_hits_by_q.append(bm25_index.query(q, top_k=pool_size))

        for cond in args.conditions:
            coll = collection_for(cond)
            log.info(f"Querying {coll}…")
            for qi, vec in enumerate(qvecs):
                resp = client.query_points(
                    collection_name=coll,
                    query=vec.tolist(),
                    limit=pool_size,
                    with_payload=True,
                    with_vectors=False,
                )
                dense_hits = []
                for rank, h in enumerate(resp.points, 1):
                    p = h.payload or {}
                    dense_hits.append({
                        "rank": rank,
                        "score": float(h.score),  # dense cosine
                        "chunk_id": p.get("chunk_id"),
                        "slug": p.get("slug"),
                        "page": p.get("page"),
                        "title": p.get("title"),
                        "text": p.get("text", ""),  # for reranker
                    })

                if bm25_index is not None:
                    from retrieval.hybrid import reciprocal_rank_fusion
                    fused_top = pool_size if reranker is not None else args.top_k
                    candidates = reciprocal_rank_fusion(
                        dense_hits, bm25_hits_by_q[qi],
                        k=args.rrf_k, top_k=fused_top,
                    )
                else:
                    candidates = dense_hits

                if reranker is not None:
                    candidates = reranker.rerank(
                        queries[qi], candidates,
                        top_k=args.top_k, batch_size=args.rerank_batch,
                    )
                else:
                    candidates = candidates[: args.top_k]

                # drop bulky text from in-memory record once we don't need it
                for h in candidates:
                    h.pop("text", None)
                results[qi][cond] = candidates

        # ---------- write raw jsonl ----------
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for qi, q in enumerate(queries):
                for cond in args.conditions:
                    for h in results[qi][cond]:
                        f.write(json.dumps({"query": q, "condition": cond, **h}, ensure_ascii=False) + "\n")
        log.info(f"Wrote {args.out_jsonl}")

        # ---------- aggregate metrics ----------
        # Final ranking score: rerank > rrf > dense, depending on pipeline.
        if args.rerank:
            score_key = "rerank_score"
        elif args.hybrid:
            score_key = "rrf_score"
        else:
            score_key = "score"

        # Per-condition: mean top-1 score, mean top-k score, distinct slugs in top-k
        per_cond_top1 = defaultdict(list)
        per_cond_topk = defaultdict(list)
        per_cond_unique_slugs = defaultdict(list)
        for qi in range(len(queries)):
            for cond in args.conditions:
                hits = results[qi][cond]
                if hits:
                    per_cond_top1[cond].append(hits[0][score_key])
                    per_cond_topk[cond].append(mean(h[score_key] for h in hits))
                    per_cond_unique_slugs[cond].append(len({h["slug"] for h in hits}))

        # Per-condition vs baseline: Jaccard@k of chunk_ids, Kendall tau on shared, top-1 change rate
        baseline = args.baseline
        per_cond_jaccard = defaultdict(list)
        per_cond_tau = defaultdict(list)
        per_cond_top1_changed = defaultdict(int)
        for qi in range(len(queries)):
            base_hits = results[qi].get(baseline, [])
            base_chunks = [h["chunk_id"] for h in base_hits]
            base_set = set(base_chunks)
            for cond in args.conditions:
                if cond == baseline:
                    continue
                cond_hits = results[qi][cond]
                cond_chunks = [h["chunk_id"] for h in cond_hits]
                per_cond_jaccard[cond].append(jaccard(base_set, set(cond_chunks)))
                per_cond_tau[cond].append(kendall_tau(base_chunks, cond_chunks))
                if base_hits and cond_hits and base_hits[0]["chunk_id"] != cond_hits[0]["chunk_id"]:
                    per_cond_top1_changed[cond] += 1

        # ---------- write markdown summary ----------
        out = []
        suffix_bits = []
        if args.hybrid:
            suffix_bits.append("hybrid (dense+BM25 RRF)")
        if args.rerank:
            suffix_bits.append("rerank")
        title_suffix = f" — {' + '.join(suffix_bits)}" if suffix_bits else ""
        out.append(f"# Retrieval ablation summary{title_suffix}\n")
        out.append(f"- Queries: **{len(queries)}** from `{args.queries}`")
        out.append(f"- Top-k:   **{args.top_k}**")
        if args.hybrid:
            out.append(f"- Hybrid:  **dense top-{args.candidate_pool} ⊕ BM25 top-{args.candidate_pool} → RRF (k={args.rrf_k})**")
        if args.rerank:
            out.append(f"- Rerank:  **BAAI/bge-reranker-v2-m3** (pool top-{args.candidate_pool} → rerank → top-{args.top_k})")
        out.append(f"- Baseline: **{baseline}**\n")

        out.append(f"## Per-condition score & diversity (mean over {len(queries)} queries)\n")
        out.append("| Condition | mean top-1 | mean top-k | mean distinct slugs in top-k |")
        out.append("|---|---|---|---|")
        for cond in args.conditions:
            t1 = mean(per_cond_top1[cond]) if per_cond_top1[cond] else 0
            tk = mean(per_cond_topk[cond]) if per_cond_topk[cond] else 0
            us = mean(per_cond_unique_slugs[cond]) if per_cond_unique_slugs[cond] else 0
            out.append(f"| {cond} | {t1:.4f} | {tk:.4f} | {us:.2f} |")
        out.append("")

        out.append(f"## Agreement vs baseline ({baseline})\n")
        out.append("| Condition | mean Jaccard@k | mean Kendall τ | top-1 changed (% queries) |")
        out.append("|---|---|---|---|")
        for cond in args.conditions:
            if cond == baseline:
                continue
            j = mean(per_cond_jaccard[cond]) if per_cond_jaccard[cond] else 0
            t = mean(per_cond_tau[cond]) if per_cond_tau[cond] else 0
            ch = 100 * per_cond_top1_changed[cond] / len(queries)
            out.append(f"| {cond} | {j:.3f} | {t:.3f} | {ch:.1f}% |")
        out.append("")

        # Per-query top-1 across conditions for skim
        out.append("## Per-query top-1 across conditions\n")
        header = "| # | Query | " + " | ".join(args.conditions) + " |"
        sep = "|---|---|" + "---|" * len(args.conditions)
        out.append(header)
        out.append(sep)
        for qi, q in enumerate(queries):
            row = [f"| {qi+1} | {q[:60]}{'…' if len(q) > 60 else ''}"]
            for cond in args.conditions:
                hits = results[qi][cond]
                if hits:
                    h = hits[0]
                    row.append(f"{h['slug']} p.{h['page']} ({h[score_key]:.3f})")
                else:
                    row.append("-")
            out.append(" | ".join(row) + " |")
        out.append("")

        Path(args.out_summary).write_text("\n".join(out), encoding="utf-8")
        log.info(f"Wrote {args.out_summary}")

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
