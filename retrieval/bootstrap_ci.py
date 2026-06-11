"""
Bootstrap 95% CIs + McNemar significance for the retrieval eval.

Reads the per-query rank-of-gold JSONL files written by eval_synthetic.py
(fields: q_idx, condition, rank_of_gold) and reports Hit@1/5/10, MRR, NDCG@10
and "not retrieved" with 95% bootstrap confidence intervals — plus pairwise
McNemar tests on Hit@1 between pipelines (paired on the same queries). This is
the rigor layer for the production numbers: point estimates become intervals,
and "is hybrid+rerank significantly better than dense+rerank?" gets a p-value.

Usage
-----
    .venv311/Scripts/python.exe -m retrieval.bootstrap_ci \\
        --files data/prod_dense.jsonl data/prod_rerank.jsonl data/prod_hybrid.jsonl data/prod_hybrid_rerank.jsonl \\
        --labels dense "dense+rerank" hybrid "hybrid+rerank" \\
        --condition none --out data/prod_ci.md
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

METRIC_KEYS = ["Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG@10", "NotRetr"]


def load_ranks(path: str, condition: str) -> dict:
    """q_idx -> rank_of_gold (int or None) for the requested condition."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if condition is not None and r.get("condition") != condition:
            continue
        out[r["q_idx"]] = r.get("rank_of_gold")
    return out


def metrics(ranks: list) -> dict:
    n = len(ranks) or 1
    def hit(k): return sum(1 for r in ranks if r is not None and r <= k) / n
    return {
        "Hit@1": hit(1), "Hit@5": hit(5), "Hit@10": hit(10),
        "MRR": sum((1.0 / r) if r else 0.0 for r in ranks) / n,
        "NDCG@10": sum((1.0 / math.log2(r + 1)) if (r and r <= 10) else 0.0 for r in ranks) / n,
        "NotRetr": sum(1 for r in ranks if r is None) / n,
    }


def bootstrap_ci(ranks: list, B: int = 2000, seed: int = 42) -> dict:
    """95% percentile bootstrap CI for every metric, resampling queries."""
    rng = np.random.default_rng(seed)
    n = len(ranks)
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m = metrics([ranks[i] for i in idx])
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    return {k: tuple(np.percentile(samples[k], [2.5, 97.5])) for k in METRIC_KEYS}


def mcnemar_hit1(ranks_a: list, ranks_b: list) -> tuple:
    """Paired McNemar on Hit@1. Returns (b, c, p) where b = A-hit/B-miss,
    c = A-miss/B-hit. Exact binomial via scipy if available, else chi-square."""
    b = c = 0
    for ra, rb in zip(ranks_a, ranks_b):
        ha = ra is not None and ra <= 1
        hb = rb is not None and rb <= 1
        if ha and not hb:
            b += 1
        elif hb and not ha:
            c += 1
    if b + c == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(b, b + c, 0.5).pvalue
    except Exception:  # noqa: BLE001 — fall back to chi-square w/ continuity correction
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        p = math.erfc(math.sqrt(chi / 2))  # chi-square df=1 survival
    return b, c, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display names (default: filename stems)")
    ap.add_argument("--condition", default="none")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/prod_ci.md")
    args = ap.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        print("--labels must match --files length", file=sys.stderr)
        return 2

    # Load, align to the common query set (paired comparisons need identical queries).
    per_file = [load_ranks(f, args.condition) for f in args.files]
    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)
    common = sorted(common)
    if not common:
        print("No overlapping q_idx across files (check --condition).", file=sys.stderr)
        return 2
    ranks = [[d[q] for q in common] for d in per_file]
    print(f"{len(common)} queries (condition={args.condition})")

    point = [metrics(r) for r in ranks]
    cis = [bootstrap_ci(r, B=args.bootstrap, seed=args.seed) for r in ranks]

    def fmt(v, lo, hi, pct=True):
        return f"{v*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    out = [f"# Production retrieval metrics — 95% bootstrap CIs (B={args.bootstrap})\n",
           f"- {len(common)} queries · condition `{args.condition}` · seed {args.seed}\n",
           "| Pipeline | Hit@1 | Hit@10 | MRR | NDCG@10 | Not retrieved |",
           "|---|---|---|---|---|---|"]
    for lab, pt, ci in zip(labels, point, cis):
        out.append(
            f"| {lab} "
            f"| {fmt(pt['Hit@1'], *ci['Hit@1'])} "
            f"| {fmt(pt['Hit@10'], *ci['Hit@10'])} "
            f"| {fmt(pt['MRR'], *ci['MRR'], pct=False)} "
            f"| {fmt(pt['NDCG@10'], *ci['NDCG@10'], pct=False)} "
            f"| {fmt(pt['NotRetr'], *ci['NotRetr'])} |"
        )
    out.append("")

    # Pairwise McNemar on Hit@1 (adjacent + first-vs-last).
    out.append("## McNemar on Hit@1 (paired significance)\n")
    out.append("| A vs B | A-only hits | B-only hits | p-value |")
    out.append("|---|---|---|---|")
    pairs = [(i, i + 1) for i in range(len(labels) - 1)]
    if len(labels) > 2:
        pairs.append((0, len(labels) - 1))
    for i, j in pairs:
        b, c, p = mcnemar_hit1(ranks[i], ranks[j])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        out.append(f"| {labels[i]} vs {labels[j]} | {b} | {c} | {p:.2e} {sig} |")
    out.append("")
    out.append("*b = A hit@1 & B miss; c = A miss & B hit@1. Significance: "
               "\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.*")

    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
