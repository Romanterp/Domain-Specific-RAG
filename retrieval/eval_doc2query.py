"""
doc2query A/B evaluation — BM25 with vs without doc2query augmentation.

Isolates doc2query's effect on the SPARSE channel (its canonical use). Same
held-out eval queries, same corpus, same tokeniser — the ONLY difference is the
indexable text: baseline = chunk + metadata; doc2query = chunk + metadata +
LLM-generated questions. The metric delta is the doc2query lift.

Why BM25-only here: doc2query (Nogueira & Lin) was designed to enrich the
lexical surface of documents for sparse retrieval. The dense channel is left
unchanged (no re-embedding needed), so this prototype runs in minutes on CPU.
For the hybrid view (dense ⊕ doc2query-BM25 → rerank), run eval_synthetic.py
on the same held-out file, pointing --bm25-path at each index — see the README
note printed at the end.

Reads the held-out eval queries from retrieval/doc2query.py (gold chunk_id per
query) and reuses the metric helpers from eval_synthetic.py so the numbers are
defined identically to the main ablation tables.

Usage
-----
    .venv311/Scripts/python.exe -m retrieval.eval_doc2query \\
        --questions data/doc2query_eval.jsonl \\
        --baseline data/bm25_index.pkl \\
        --doc2query data/bm25_doc2query.pkl
"""

import argparse
import sys
from pathlib import Path

from retrieval.bm25 import BM25Index
from retrieval.eval_synthetic import (
    load_questions,
    find_rank,
    metrics_for,
    rank_buckets,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def eval_index(idx: BM25Index, questions: list[dict], top_k: int) -> list[int | None]:
    """Rank-of-gold for each held-out query against one BM25 index."""
    ranks: list[int | None] = []
    for q in questions:
        hits = idx.query(q["question"], top_k=top_k)
        ranks.append(find_rank(hits, q["chunk_id"]))
    return ranks


def _fmt_delta(base: float, aug: float, pct: bool = False) -> str:
    d = aug - base
    scale = 100 if pct else 1
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d * scale:.1f} pp"
    return f"{sign}{d:.3f}"


def main() -> int:
    try:  # Windows consoles default to cp1252 and choke on Δ/→ etc.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--questions", default=str(DATA_DIR / "doc2query_eval.jsonl"))
    ap.add_argument("--baseline", default=str(DATA_DIR / "bm25_index.pkl"),
                    help="BM25 index WITHOUT doc2query (the production baseline)")
    ap.add_argument("--doc2query", default=str(DATA_DIR / "bm25_doc2query.pkl"),
                    help="BM25 index WITH doc2query augmentation")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out-summary", default=str(DATA_DIR / "eval_doc2query_summary.md"))
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    if not questions:
        print(f"No questions in {args.questions}", file=sys.stderr)
        return 2
    print(f"Held-out eval queries: {len(questions)}")

    base_idx = BM25Index.load(Path(args.baseline))
    aug_idx = BM25Index.load(Path(args.doc2query))

    print("Scoring baseline BM25…")
    base_ranks = eval_index(base_idx, questions, args.top_k)
    print("Scoring doc2query BM25…")
    aug_ranks = eval_index(aug_idx, questions, args.top_k)

    mb = metrics_for(base_ranks)
    ma = metrics_for(aug_ranks)

    not_retrieved_b = sum(1 for r in base_ranks if r is None) / len(base_ranks)
    not_retrieved_a = sum(1 for r in aug_ranks if r is None) / len(aug_ranks)

    # ---- console table (so you see numbers immediately) ----
    rows = [
        ("Hit@1", mb["hit@1"], ma["hit@1"], True),
        ("Hit@5", mb["hit@5"], ma["hit@5"], True),
        ("Hit@10", mb["hit@10"], ma["hit@10"], True),
        ("MRR", mb["mrr"], ma["mrr"], False),
        ("NDCG@10", mb["ndcg@10"], ma["ndcg@10"], False),
        ("Not retrieved", not_retrieved_b, not_retrieved_a, True),
    ]
    print()
    print(f"{'Metric':<16}{'BM25 base':>12}{'+doc2query':>12}{'Δ':>12}")
    print("-" * 52)
    for name, b, a, pct in rows:
        bs = f"{b*100:.1f}%" if pct else f"{b:.3f}"
        as_ = f"{a*100:.1f}%" if pct else f"{a:.3f}"
        print(f"{name:<16}{bs:>12}{as_:>12}{_fmt_delta(b, a, pct):>12}")
    print()

    # ---- markdown summary (drop-in for the thesis / EXPERIMENTS.md) ----
    out = []
    out.append("# doc2query A/B — BM25 with vs without question augmentation\n")
    out.append(f"- Held-out eval queries: **{len(questions)}** "
               f"(from `{Path(args.questions).name}`; never indexed — see doc2query.py)")
    out.append(f"- Baseline index : `{Path(args.baseline).name}` (chunk + metadata)")
    out.append(f"- doc2query index: `{Path(args.doc2query).name}` (chunk + metadata + generated questions)")
    out.append(f"- Lookup depth   : top-{args.top_k}\n")
    out.append("## Sparse-channel retrieval (BM25)\n")
    out.append("| Metric | BM25 baseline | BM25 + doc2query | Δ |")
    out.append("|---|---|---|---|")
    for name, b, a, pct in rows:
        bs = f"{b*100:.1f}%" if pct else f"{b:.3f}"
        as_ = f"{a*100:.1f}%" if pct else f"{a:.3f}"
        out.append(f"| {name} | {bs} | {as_} | {_fmt_delta(b, a, pct)} |")
    out.append("")
    out.append("## Rank-of-gold distribution (% of queries)\n")
    out.append("| Index | rank=1 | rank=2-5 | rank=6-20 | rank=21-100 | not retrieved |")
    out.append("|---|---|---|---|---|---|")
    for label, ranks in [("BM25 baseline", base_ranks), ("BM25 + doc2query", aug_ranks)]:
        total = len(ranks) or 1
        counts = rank_buckets(ranks)
        row = [label] + [f"{100*c/total:.1f}%" for _, c in counts]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("## Caveats\n")
    out.append("- Prototype: 7B generator, single seed, small held-out set — directional, "
               "not final. The production run (32B on Habrok) tightens these.")
    out.append("- Held-out queries share some vocabulary with their indexed sibling "
               "questions (same passage); this residual leakage is identical in both "
               "columns, so the **Δ** is the clean signal. Cross-check on the 33 "
               "hand-written naturalistic queries before claiming the lift.")
    out.append("- Hybrid view (dense ⊕ doc2query-BM25 → rerank): run "
               "`eval_synthetic --questions data/doc2query_eval.jsonl --hybrid [--rerank] "
               "--conditions none` once with `--bm25-path data/bm25_index.pkl` and once "
               "with `--bm25-path data/bm25_doc2query.pkl`; compare the two summaries.")
    Path(args.out_summary).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
