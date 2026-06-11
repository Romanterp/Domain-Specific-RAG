"""
Evaluate retrieval pipelines against the synthetic gold-passage question set.

Reads `data/synthetic_questions_eval.jsonl` (one record per question, with the
chunk_id of the gold passage that the question was generated from). Runs each
question through the requested pipeline configuration and computes standard
retrieval metrics with KNOWN labels:
  - Hit@1, Hit@5, Hit@10  (binary: was the gold chunk in top-k?)
  - MRR                   (mean reciprocal rank; 1/rank, 0 if not found)
  - NDCG@10               (single-relevant-doc DCG normalized by IDCG=1)
  - Rank-of-gold buckets  (1, 2-5, 6-20, 21-100, not retrieved)
                          → free difficulty stratification

One run = one (hybrid, rerank) combination across all dense conditions. Run
multiple times for the full ablation matrix; compare the four summary files
side-by-side. This mirrors `retrieval/evaluate.py` so the metric tables sit
next to each other for the thesis writeup.

Usage
-----
Dense baseline (no rerank, no hybrid):
    .venv311/Scripts/python.exe -m retrieval.eval_synthetic \\
        --questions data/synthetic_questions_eval.jsonl

Dense + rerank:
    .venv311/Scripts/python.exe -m retrieval.eval_synthetic \\
        --questions data/synthetic_questions_eval.jsonl \\
        --rerank --candidate-pool 100

Hybrid + rerank (the recommended production pipeline):
    .venv311/Scripts/python.exe -m retrieval.eval_synthetic \\
        --questions data/synthetic_questions_eval.jsonl \\
        --hybrid --rerank --candidate-pool 100

Outputs (auto-named based on flags, override with --out-jsonl / --out-summary):
    data/eval_synthetic_results_<config>.jsonl   per-query rank-of-gold
    data/eval_synthetic_summary_<config>.md      markdown metrics tables
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_QDRANT_PATH = DATA_DIR / "qdrant"

MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "refs/pr/130"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def collection_for(mode: str) -> str:
    return f"theisus_{mode.replace('+', '_')}"


def load_questions(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def find_rank(hits: list[dict], gold_chunk_id: str) -> int | None:
    """Return 1-based rank of gold chunk in hits, or None if absent."""
    for i, h in enumerate(hits, 1):
        if h.get("chunk_id") == gold_chunk_id:
            return i
    return None


def ndcg_at_k(rank: int | None, k: int) -> float:
    """Single-relevant-doc NDCG@k.

    Only one gold doc per query, so DCG = 1/log2(rank+1) when rank ≤ k else 0.
    Ideal DCG = 1/log2(2) = 1, so NDCG = DCG.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def metrics_for(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "hit@10": sum(1 for r in ranks if r is not None and r <= 10) / n,
        "mrr": mean(1.0 / r if r is not None else 0.0 for r in ranks),
        "ndcg@10": mean(ndcg_at_k(r, 10) for r in ranks),
    }


def rank_buckets(ranks: list[int | None]) -> list[tuple[str, int]]:
    """Returns a list of (label, count) for each rank bucket."""
    buckets = [
        ("rank=1", lambda r: r is not None and r == 1),
        ("rank=2-5", lambda r: r is not None and 2 <= r <= 5),
        ("rank=6-20", lambda r: r is not None and 6 <= r <= 20),
        ("rank=21-100", lambda r: r is not None and 21 <= r <= 100),
        ("not retrieved", lambda r: r is None),
    ]
    return [(label, sum(1 for r in ranks if pred(r))) for label, pred in buckets]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--questions", default=str(DATA_DIR / "synthetic_questions_eval.jsonl"))
    ap.add_argument(
        "--conditions", nargs="+",
        default=["none", "title", "title+desc", "title+desc+keywords", "full"],
        help="dense ablation collections to evaluate",
    )
    ap.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    ap.add_argument("--top-k", type=int, default=100,
                    help="rank depth at which gold lookup stops (use ≥100 for "
                         "meaningful 'not retrieved' bucket)")
    ap.add_argument("--candidate-pool", type=int, default=100,
                    help="top-N from each retriever before fusion / rerank")
    ap.add_argument("--hybrid", action="store_true",
                    help="fuse dense and BM25 channels via RRF")
    ap.add_argument("--rerank", action="store_true",
                    help="apply BGE-reranker-v2-m3 to candidate pool")
    ap.add_argument("--bm25-path", default=str(DATA_DIR / "bm25_index.pkl"))
    ap.add_argument("--rerank-batch", type=int, default=32)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    # Auto-name outputs by config so multiple runs don't clobber each other.
    pieces = []
    if args.hybrid:
        pieces.append("hybrid")
    if args.rerank:
        pieces.append("rerank")
    config_tag = "_".join(pieces) if pieces else "dense"
    if args.out_jsonl is None:
        args.out_jsonl = str(DATA_DIR / f"eval_synthetic_results_{config_tag}.jsonl")
    if args.out_summary is None:
        args.out_summary = str(DATA_DIR / f"eval_synthetic_summary_{config_tag}.md")

    questions = load_questions(Path(args.questions))
    log.info(f"Loaded {len(questions)} questions from {args.questions}")
    log.info(f"Conditions: {args.conditions}")
    log.info(f"Pipeline  : {config_tag}  (top-k={args.top_k}, pool={args.candidate_pool})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Loading {MODEL_NAME} on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION)

    log.info("Encoding questions…")
    q_texts = [q["question"] for q in questions]
    qvecs = model.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)

    reranker = None
    if args.rerank:
        from retrieval.rerank import Reranker
        reranker = Reranker(device=device)

    bm25 = None
    bm25_hits_per_q: list[list[dict]] = []
    if args.hybrid:
        from retrieval.bm25 import BM25Index
        bm25 = BM25Index.load(Path(args.bm25_path))
        log.info(f"Pre-computing BM25 hits for {len(questions)} questions…")
        for q in q_texts:
            bm25_hits_per_q.append(bm25.query(q, top_k=args.candidate_pool))

    pool_size = args.candidate_pool if (args.rerank or args.hybrid) else args.top_k
    client = QdrantClient(path=args.qdrant_path)

    try:
        # ranks_by_cond[cond][qi] = rank-of-gold (int) or None (not in top-k)
        ranks_by_cond: dict[str, list[int | None]] = defaultdict(list)

        for cond in args.conditions:
            coll = collection_for(cond)
            log.info(f"Querying {coll}…")
            for qi, q in enumerate(questions):
                resp = client.query_points(
                    collection_name=coll,
                    query=qvecs[qi].tolist(),
                    limit=pool_size,
                    with_payload=True,
                    with_vectors=False,
                )
                dense_hits = []
                for h in resp.points:
                    p = h.payload or {}
                    dense_hits.append({
                        "chunk_id": p.get("chunk_id"),
                        "slug": p.get("slug"),
                        "page": p.get("page"),
                        "score": float(h.score),
                        "text": p.get("text", ""),
                    })

                if bm25 is not None:
                    from retrieval.hybrid import reciprocal_rank_fusion
                    fused_top = pool_size if reranker is not None else args.top_k
                    candidates = reciprocal_rank_fusion(
                        dense_hits, bm25_hits_per_q[qi],
                        k=args.rrf_k, top_k=fused_top,
                    )
                else:
                    candidates = dense_hits

                if reranker is not None:
                    candidates = reranker.rerank(
                        q["question"], candidates,
                        top_k=args.top_k, batch_size=args.rerank_batch,
                    )
                else:
                    candidates = candidates[: args.top_k]

                ranks_by_cond[cond].append(find_rank(candidates, q["chunk_id"]))

        # ---------- write per-query jsonl ----------
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for qi, q in enumerate(questions):
                for cond in args.conditions:
                    f.write(json.dumps({
                        "q_idx": qi,
                        "question": q["question"],
                        "gold_chunk_id": q["chunk_id"],
                        "gold_slug": q["slug"],
                        "gold_page": q.get("page"),
                        "condition": cond,
                        "pipeline": config_tag,
                        "rank_of_gold": ranks_by_cond[cond][qi],
                    }, ensure_ascii=False) + "\n")
        log.info(f"Wrote {args.out_jsonl}")

        # ---------- write markdown summary ----------
        out = []
        suffix_bits = []
        if args.hybrid:
            suffix_bits.append("hybrid (dense+BM25 RRF)")
        if args.rerank:
            suffix_bits.append("rerank")
        suffix = f" — {' + '.join(suffix_bits)}" if suffix_bits else " — dense only"
        out.append(f"# Synthetic gold-passage eval{suffix}\n")
        out.append(f"- Questions: **{len(questions)}** from `{args.questions}`")
        out.append(f"- Top-k:    **{args.top_k}**  (rank-of-gold lookup depth)")
        if args.hybrid:
            out.append(f"- Hybrid:   dense top-{args.candidate_pool} ⊕ BM25 top-{args.candidate_pool} → RRF (k={args.rrf_k})")
        if args.rerank:
            out.append(f"- Rerank:   BAAI/bge-reranker-v2-m3 (pool top-{args.candidate_pool} → rerank → top-{args.top_k})")
        out.append("")

        # Main metrics table — these are the headline numbers for the thesis.
        out.append(f"## Headline metrics (mean over {len(questions)} questions)\n")
        out.append("| Condition | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 |")
        out.append("|---|---|---|---|---|---|")
        for cond in args.conditions:
            m = metrics_for(ranks_by_cond[cond])
            out.append(
                f"| {cond} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | "
                f"{m['hit@10']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} |"
            )
        out.append("")

        # Rank-of-gold distribution per condition (free difficulty stratification).
        out.append("## Rank-of-gold distribution (% of queries)\n")
        out.append("| Condition | rank=1 | rank=2-5 | rank=6-20 | rank=21-100 | not retrieved |")
        out.append("|---|---|---|---|---|---|")
        for cond in args.conditions:
            ranks = ranks_by_cond[cond]
            total = len(ranks) or 1
            counts = rank_buckets(ranks)
            row = [cond] + [f"{100*c/total:.1f}%" for _, c in counts]
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # Worst-case spotlight: 5 queries where every condition fails. These
        # are likely either (a) bad questions (manual annotation drops them)
        # or (b) chunks that genuinely the retrieval pipeline can't find —
        # both are worth eyeballing.
        out.append(f"## Hardest queries (gold not retrieved by ANY condition)\n")
        all_miss = []
        for qi, q in enumerate(questions):
            if all(ranks_by_cond[c][qi] is None for c in args.conditions):
                all_miss.append((qi, q))
        if all_miss:
            out.append(f"{len(all_miss)} of {len(questions)} queries missed across every condition.")
            out.append("Likely poor questions or out-of-corpus content; flag for manual review.\n")
            out.append("| # | Gold slug | p. | Question |")
            out.append("|---|---|---|---|")
            for qi, q in all_miss[:10]:
                qtext = q["question"][:120].replace("|", "\\|")
                out.append(f"| {qi} | {q['slug']} | {q.get('page','?')} | {qtext} |")
        else:
            out.append("Every query had its gold chunk retrieved by at least one condition.")
        out.append("")

        Path(args.out_summary).write_text("\n".join(out), encoding="utf-8")
        log.info(f"Wrote {args.out_summary}")

    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
