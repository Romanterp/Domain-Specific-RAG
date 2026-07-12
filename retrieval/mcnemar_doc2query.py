"""
Paired McNemar significance for the doc2query BM25 A/B.

eval_doc2query.py reports point estimates only. This re-runs the same two BM25
indices over the same held-out queries, captures rank-of-gold per query, and
runs a paired McNemar test on each binary outcome (Hit@1/5/10 and
retrieved-at-all). McNemar is the right test here because the two systems are
scored on the *same* queries — only the discordant pairs carry information.

For each metric: b = baseline hit & doc2query miss (doc2query LOST these),
c = baseline miss & doc2query hit (doc2query GAINED these). net = c - b. A
two-sided exact binomial on (b, c) gives the p-value.

Usage
-----
    .venv311/Scripts/python.exe -m retrieval.mcnemar_doc2query \\
        --questions data/doc2query_eval_full_paraphrased.jsonl \\
        --baseline data/bm25_index.pkl \\
        --doc2query data/bm25_doc2query_full.pkl
"""

import argparse
import math
import sys
from pathlib import Path

from retrieval.bm25 import BM25Index
from retrieval.eval_synthetic import load_questions, find_rank

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (label, predicate on rank-of-gold). "Retrieved@100" is the complement of the
# not-retrieved bucket — the recall outcome doc2query is supposed to move.
METRICS = [
    ("Hit@1", lambda r: r is not None and r <= 1),
    ("Hit@5", lambda r: r is not None and r <= 5),
    ("Hit@10", lambda r: r is not None and r <= 10),
    ("Retrieved@100", lambda r: r is not None),
]


def ranks_for(idx: BM25Index, questions: list[dict], top_k: int) -> list:
    return [find_rank(idx.query(q["question"], top_k=top_k), q["chunk_id"])
            for q in questions]


def mcnemar(ranks_base: list, ranks_d2q: list, pred) -> tuple:
    """Returns (base_hits, d2q_hits, b, c, p).

    b = baseline hit & doc2query miss (lost); c = baseline miss & doc2query hit
    (gained). Two-sided exact binomial on the discordant pairs.
    """
    base_hits = d2q_hits = b = c = 0
    for rb, rd in zip(ranks_base, ranks_d2q):
        hb, hd = pred(rb), pred(rd)
        base_hits += hb
        d2q_hits += hd
        if hb and not hd:
            b += 1
        elif hd and not hb:
            c += 1
    if b + c == 0:
        p = 1.0
    else:
        try:
            from scipy.stats import binomtest
            p = binomtest(b, b + c, 0.5).pvalue
        except Exception:  # chi-square w/ continuity correction
            chi = (abs(b - c) - 1) ** 2 / (b + c)
            p = math.erfc(math.sqrt(chi / 2))
    return base_hits, d2q_hits, b, c, p


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--questions", default=str(DATA_DIR / "doc2query_eval_full_paraphrased.jsonl"))
    ap.add_argument("--baseline", default=str(DATA_DIR / "bm25_index.pkl"))
    ap.add_argument("--doc2query", default=str(DATA_DIR / "bm25_doc2query_full.pkl"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out-summary", default=str(DATA_DIR / "mcnemar_doc2query_paraphrased.md"))
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    print(f"Paired queries: {len(questions)}")
    base_idx = BM25Index.load(Path(args.baseline))
    d2q_idx = BM25Index.load(Path(args.doc2query))

    print("Scoring baseline…")
    rb = ranks_for(base_idx, questions, args.top_k)
    print("Scoring doc2query…")
    rd = ranks_for(d2q_idx, questions, args.top_k)

    n = len(questions)
    rows = []
    for label, pred in METRICS:
        bh, dh, b, c, p = mcnemar(rb, rd, pred)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        rows.append((label, bh, dh, b, c, c - b, p, sig))

    hdr = f"{'Metric':<15}{'base':>7}{'+d2q':>7}{'lost':>7}{'gain':>7}{'net':>7}{'p-value':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, bh, dh, b, c, net, p, sig in rows:
        print(f"{label:<15}{bh:>7}{dh:>7}{b:>7}{c:>7}{net:>+7}{p:>10.2e} {sig}")
    print("\nlost = baseline hit & doc2query miss; gain = baseline miss & doc2query hit")

    out = ["# doc2query BM25 A/B — paired McNemar significance\n",
           f"- Queries: **{n}** (`{Path(args.questions).name}`)",
           f"- Baseline : `{Path(args.baseline).name}`",
           f"- doc2query: `{Path(args.doc2query).name}`",
           f"- Lookup depth: top-{args.top_k}\n",
           "| Metric | base hits | +d2q hits | lost | gained | net | p-value |",
           "|---|---|---|---|---|---|---|"]
    for label, bh, dh, b, c, net, p, sig in rows:
        out.append(f"| {label} | {bh} ({100*bh/n:.1f}%) | {dh} ({100*dh/n:.1f}%) "
                   f"| {b} | {c} | {net:+d} | {p:.2e} {sig} |")
    out += ["",
            "*lost = baseline hit & doc2query miss; gained = baseline miss & doc2query hit. "
            "net = gained − lost. Two-sided exact binomial on discordant pairs. "
            "Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.*"]
    Path(args.out_summary).write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
