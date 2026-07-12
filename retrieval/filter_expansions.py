"""
Doc2Query-- : relevance-filter the generated doc2query expansions.

Scores each generated question against its SOURCE chunk with the same
cross-encoder used for reranking (BAAI/bge-reranker-v2-m3), then drops the
least-relevant questions before indexing. Gospodinov, MacAvaney & Macdonald
(ECIR 2023, "Doc2Query--: When Less is More") showed that doc2query generators
emit many off-topic / hallucinated queries, and pruning the least-relevant ones
with a relevance model *improves* retrieval while shrinking the index.

Two phases:
  1. SCORE (default): score every (chunk, question) pair, cache to
     data/doc2query_expansion_scores.jsonl (resumable — re-run with --resume
     to continue after an interruption). This is the GPU-bound step.
  2. APPLY: from the cached scores, write filtered expansion JSONs at one or
     more global keep-fractions. Each output is a drop-in for
     `bm25.py --expansions`. A keep-fraction is applied as a GLOBAL score
     percentile over all questions (Doc2Query-- style), so chunks whose
     generated questions are all low-relevance can lose all of them — that is
     the intended index-shrinking behaviour.

Usage
-----
Score, then write filtered files at several keep-fractions (keep=1.0 reproduces
the unfiltered set as a sanity baseline):
    .venv311/Scripts/python.exe -m retrieval.filter_expansions \\
        --expansions data/doc2query_expansions_full.json \\
        --keep-fracs 1.0 0.75 0.5 0.25 --resume

Re-apply thresholds later WITHOUT rescoring (reads the cache only):
    .venv311/Scripts/python.exe -m retrieval.filter_expansions --apply-only \\
        --keep-fracs 0.6 0.4

Then, per filtered file, build an index and run the sparse A/B:
    .venv311/Scripts/python.exe -m retrieval.bm25 \\
        --expansions data/doc2query_expansions_full_keep50.json \\
        --out data/bm25_doc2query_keep50.pkl
    .venv311/Scripts/python.exe -m retrieval.eval_doc2query \\
        --questions data/doc2query_eval_full_paraphrased.jsonl \\
        --baseline data/bm25_index.pkl \\
        --doc2query data/bm25_doc2query_keep50.pkl \\
        --out-summary data/eval_doc2query_keep50_paraphrased.md
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from retrieval.bm25 import CHUNKS_PATH

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_EXPANSIONS = DATA_DIR / "doc2query_expansions_full.json"
DEFAULT_SCORES = DATA_DIR / "doc2query_expansion_scores.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_chunk_text(path: Path) -> dict[str, str]:
    """chunk_id -> chunk text, for every chunk in the corpus."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["chunk_id"]] = r["text"]
    return out


def load_scores(path: Path) -> dict[tuple[str, int], dict]:
    """Cached scores keyed by (chunk_id, q_idx)."""
    cache: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[(r["chunk_id"], r["q_idx"])] = r
    return cache


def score_phase(expansions: dict[str, list[str]], scores_path: Path,
                resume: bool, batch_size: int, limit: int | None) -> None:
    from retrieval.rerank import Reranker
    import torch

    cache = load_scores(scores_path) if resume else {}
    if cache:
        log.info(f"Resuming — {len(cache):,} pairs already scored, skipping")

    # Flatten to (chunk_id, q_idx, question) work items, skipping cached ones.
    chunk_ids = list(expansions.keys())
    if limit:
        chunk_ids = chunk_ids[:limit]
        log.info(f"Limit: first {len(chunk_ids):,} chunks")

    chunk_text = load_chunk_text(CHUNKS_PATH)
    log.info(f"Loaded text for {len(chunk_text):,} chunks")

    todo: list[tuple[str, int, str]] = []
    missing_text = 0
    for cid in chunk_ids:
        if cid not in chunk_text:
            missing_text += 1
            continue
        for qi, q in enumerate(expansions[cid]):
            if (cid, qi) in cache:
                continue
            todo.append((cid, qi, q))
    if missing_text:
        log.warning(f"{missing_text:,} chunks had no text in chunks.jsonl (skipped)")
    total_pairs = sum(len(v) for v in expansions.values())
    log.info(f"Pairs: {total_pairs:,} total, {len(cache):,} cached, {len(todo):,} to score")
    if not todo:
        log.info("Nothing to score.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available — scoring will be slow.")
    reranker = Reranker(device=device)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (resume and scores_path.exists()) else "w"
    done = len(cache)
    t0 = time.time()
    last_log = t0
    with open(scores_path, mode, encoding="utf-8") as fout:
        for start in range(0, len(todo), batch_size):
            block = todo[start:start + batch_size]
            pairs = [[chunk_text[cid], q] for cid, _, q in block]
            with torch.inference_mode():
                preds = reranker.model.predict(
                    pairs, batch_size=batch_size, show_progress_bar=False)
            for (cid, qi, q), s in zip(block, preds):
                fout.write(json.dumps({
                    "chunk_id": cid, "q_idx": qi,
                    "question": q, "score": float(s),
                }, ensure_ascii=False) + "\n")
            fout.flush()
            done += len(block)
            now = time.time()
            if now - last_log > 15 or start + batch_size >= len(todo):
                rate = (done - len(cache)) / (now - t0) if now > t0 else 0
                remaining = len(todo) - (done - len(cache))
                eta = remaining / rate / 60 if rate else 0
                log.info(f"  scored {done:,}/{total_pairs:,}  rate={rate:.0f} pair/s  "
                         f"eta={eta:.1f}min")
                last_log = now
    log.info(f"Scoring complete → {scores_path}")


def apply_phase(expansions: dict[str, list[str]], scores_path: Path,
                keep_fracs: list[float]) -> None:
    cache = load_scores(scores_path)
    if not cache:
        log.error(f"No scores in {scores_path}. Run the score phase first.")
        return
    all_scores = np.array([r["score"] for r in cache.values()], dtype=float)
    pct = np.percentile(all_scores, [0, 10, 25, 50, 75, 90, 100])
    log.info(f"Score distribution over {len(all_scores):,} questions:")
    log.info(f"  min={pct[0]:.2f}  p10={pct[1]:.2f}  p25={pct[2]:.2f}  "
             f"median={pct[3]:.2f}  p75={pct[4]:.2f}  p90={pct[5]:.2f}  max={pct[6]:.2f}")

    total_q = sum(len(v) for v in expansions.values())
    n_chunks = len([c for c in expansions if expansions[c]])
    for frac in keep_fracs:
        if not 0 < frac <= 1:
            log.warning(f"keep-frac {frac} out of (0,1]; skipping")
            continue
        thr = float(np.quantile(all_scores, 1 - frac)) if frac < 1 else float("-inf")
        filtered: dict[str, list[str]] = {}
        kept_q = 0
        for cid, qs in expansions.items():
            keep = [q for qi, q in enumerate(qs)
                    if (cid, qi) in cache and cache[(cid, qi)]["score"] >= thr]
            if keep:
                filtered[cid] = keep
                kept_q += len(keep)
        tag = f"keep{int(round(frac * 100))}"
        out = DATA_DIR / f"doc2query_expansions_full_{tag}.json"
        out.write_text(json.dumps(filtered, ensure_ascii=False), encoding="utf-8")
        log.info(
            f"[{tag}] thr={thr if frac < 1 else 'none':<6}  "
            f"questions {kept_q:,}/{total_q:,} ({100*kept_q/total_q:.0f}%)  "
            f"chunks {len(filtered):,}/{n_chunks:,} "
            f"({100*len(filtered)/n_chunks:.0f}% retain, "
            f"{n_chunks - len(filtered):,} emptied)  → {out.name}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--expansions", default=str(DEFAULT_EXPANSIONS))
    ap.add_argument("--scores", default=str(DEFAULT_SCORES),
                    help="per-question relevance score cache (JSONL)")
    ap.add_argument("--keep-fracs", type=float, nargs="+",
                    default=[1.0, 0.75, 0.5, 0.25],
                    help="global keep fractions to emit filtered expansion files for")
    ap.add_argument("--apply-only", action="store_true",
                    help="skip scoring; only write filtered files from the cache")
    ap.add_argument("--resume", action="store_true",
                    help="continue scoring from the existing cache")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="score only the first N chunks (smoke test)")
    args = ap.parse_args()

    exp_path = Path(args.expansions)
    if not exp_path.exists():
        log.error(f"Expansions not found: {exp_path}")
        return 2
    expansions = json.loads(exp_path.read_text(encoding="utf-8"))
    log.info(f"Loaded expansions for {len(expansions):,} chunks "
             f"({sum(len(v) for v in expansions.values()):,} questions)")

    scores_path = Path(args.scores)
    if not args.apply_only:
        score_phase(expansions, scores_path, args.resume, args.batch_size, args.limit)

    apply_phase(expansions, scores_path, sorted(set(args.keep_fracs), reverse=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
