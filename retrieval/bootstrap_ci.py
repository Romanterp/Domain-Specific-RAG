"""
Bootstrap 95% CIs + McNemar significance for the retrieval eval.

Reads the per-query rank-of-gold JSONL files written by eval_synthetic.py
(fields: q_idx, condition, rank_of_gold) and reports Hit@1/5/10, MRR, NDCG@10
and "not retrieved" with 95% bootstrap confidence intervals — plus pairwise
McNemar tests on Hit@1 between pipelines (paired on the same queries). This is
the rigor layer for the production numbers: point estimates become intervals,
and "is hybrid+rerank significantly better than dense+rerank?" gets a p-value.

2026-07-12: CIs are now GOLD-CHUNK CLUSTER bootstraps — the
2,481 production questions come ~5 per gold chunk and siblings hit/miss
together, so query-level resampling understated CI width
~1.4×. With 1 query per chunk this reduces to the plain bootstrap. Also added:
paired metric-DELTA CIs (per-pipeline CI overlap is not a test of the delta)
and Holm correction across the McNemar family.

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


def load_ranks(path: str, condition: str) -> tuple[dict, dict]:
    """(q_idx -> rank_of_gold, q_idx -> gold_chunk_id) for the condition."""
    ranks, chunks = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if condition is not None and r.get("condition") != condition:
            continue
        ranks[r["q_idx"]] = r.get("rank_of_gold")
        chunks[r["q_idx"]] = r.get("gold_chunk_id")
    return ranks, chunks


def metrics(ranks: list) -> dict:
    n = len(ranks) or 1
    def hit(k): return sum(1 for r in ranks if r is not None and r <= k) / n
    return {
        "Hit@1": hit(1), "Hit@5": hit(5), "Hit@10": hit(10),
        "MRR": sum((1.0 / r) if r else 0.0 for r in ranks) / n,
        "NDCG@10": sum((1.0 / math.log2(r + 1)) if (r and r <= 10) else 0.0 for r in ranks) / n,
        "NotRetr": sum(1 for r in ranks if r is None) / n,
    }


def bootstrap_ci(ranks: list, clusters: list, B: int = 2000, seed: int = 42) -> dict:
    """95% percentile bootstrap CI for every metric, resampling gold-chunk
    CLUSTERS (sibling questions share a gold chunk and hit/miss together;
    query-level resampling understates CI width). One query per chunk reduces
    this to the plain query bootstrap."""
    rng = np.random.default_rng(seed)
    cl = [np.asarray(c, dtype=int) for c in clusters]
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(B):
        idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        m = metrics([ranks[i] for i in idx])
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    return {k: tuple(np.percentile(samples[k], [2.5, 97.5])) for k in METRIC_KEYS}


def mcnemar_hit1(ranks_a: list, ranks_b: list) -> tuple:
    """Paired McNemar on Hit@1. Returns (b, c, p, method) where b = A-hit/B-miss,
    c = A-miss/B-hit. Exact binomial via scipy if available, else chi-square —
    the method is returned so a silent downgrade can't masquerade as exact."""
    b = c = 0
    for ra, rb in zip(ranks_a, ranks_b):
        ha = ra is not None and ra <= 1
        hb = rb is not None and rb <= 1
        if ha and not hb:
            b += 1
        elif hb and not ha:
            c += 1
    if b + c == 0:
        return b, c, 1.0, "exact binomial"
    try:
        from scipy.stats import binomtest
        p = binomtest(b, b + c, 0.5).pvalue
        method = "exact binomial"
    except Exception:  # noqa: BLE001 — fall back to chi-square w/ continuity correction
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        p = math.erfc(math.sqrt(chi / 2))  # chi-square df=1 survival
        method = "chi-square approximation (scipy MISSING — install it)"
    return b, c, p, method


def holm(pvals: list) -> list:
    """Holm–Bonferroni step-down adjusted p-values (family = reported pairs)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [1.0] * m, 0.0
    for step, i in enumerate(order):
        running = max(running, (m - step) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_delta_ci(ranks_a: list, ranks_b: list, clusters: list,
                    B: int = 2000, seed: int = 42) -> dict:
    """Cluster-bootstrap 95% CI on metric DELTAS (A − B) over the same queries.

    Resamples clusters once per replicate and applies the SAME index set to
    both pipelines — proper paired inference. (Non-overlap of two marginal CIs
    is conservative evidence; overlap is NOT evidence of no difference — this
    replaces both with a direct CI on the delta.)
    """
    rng = np.random.default_rng(seed)
    cl = [np.asarray(c, dtype=int) for c in clusters]
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(B):
        idx = np.concatenate([cl[i] for i in rng.integers(0, len(cl), len(cl))])
        ma = metrics([ranks_a[i] for i in idx])
        mb = metrics([ranks_b[i] for i in idx])
        for k in METRIC_KEYS:
            samples[k].append(ma[k] - mb[k])
    pa, pb = metrics(ranks_a), metrics(ranks_b)
    return {k: (pa[k] - pb[k], *np.percentile(samples[k], [2.5, 97.5]))
            for k in METRIC_KEYS}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
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
    loaded = [load_ranks(f, args.condition) for f in args.files]
    per_file = [d for d, _ in loaded]
    chunk_of = loaded[0][1]  # pairing is identical across files, so first suffices
    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)
    common = sorted(common)
    if not common:
        print("No overlapping q_idx across files (check --condition).", file=sys.stderr)
        return 2
    ranks = [[d[q] for q in common] for d in per_file]

    # Gold-chunk clusters: sibling questions hit/miss together, so all CIs
    # resample chunks, not queries. Queries without a chunk id become singletons.
    groups: dict = {}
    for pos, q in enumerate(common):
        groups.setdefault(chunk_of.get(q) or f"_solo_{q}", []).append(pos)
    clusters = list(groups.values())
    print(f"{len(common)} queries (condition={args.condition}) in {len(clusters)} "
          f"gold-chunk clusters (mean {len(common)/len(clusters):.2f} queries/cluster)")

    point = [metrics(r) for r in ranks]
    cis = [bootstrap_ci(r, clusters, B=args.bootstrap, seed=args.seed) for r in ranks]

    def fmt(v, lo, hi, pct=True):
        return f"{v*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    out = [f"# Production retrieval metrics — 95% cluster-bootstrap CIs (B={args.bootstrap})\n",
           f"- {len(common)} queries in {len(clusters)} gold-chunk clusters "
           f"(mean {len(common)/len(clusters):.2f}/cluster) · condition `{args.condition}` "
           f"· seed {args.seed} · CIs resample chunks, not queries\n",
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

    # Pairwise McNemar on Hit@1 (adjacent + first-vs-last), Holm-corrected.
    out.append("## McNemar on Hit@1 (paired significance, Holm-corrected)\n")
    out.append("| A vs B | A-only hits | B-only hits | p (raw) | p (Holm) |")
    out.append("|---|---|---|---|---|")
    pairs = [(i, i + 1) for i in range(len(labels) - 1)]
    if len(labels) > 2:
        pairs.append((0, len(labels) - 1))
    mc_rows = [(i, j, *mcnemar_hit1(ranks[i], ranks[j])) for i, j in pairs]
    adj = holm([r[4] for r in mc_rows])
    methods = {r[5] for r in mc_rows}
    for (i, j, b, c, p, _), ph in zip(mc_rows, adj):
        sig = "***" if ph < 0.001 else "**" if ph < 0.01 else "*" if ph < 0.05 else "n.s."
        out.append(f"| {labels[i]} vs {labels[j]} | {b} | {c} | {p:.2e} | {ph:.2e} {sig} |")
    out.append("")
    out.append(f"*b = A hit@1 & B miss; c = A miss & B hit@1. Test: {', '.join(sorted(methods))}. "
               "Stars on the HOLM-adjusted p: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.*\n")
    if any("MISSING" in m for m in methods):
        print("WARNING: scipy missing — McNemar degraded to chi-square approximation",
              file=sys.stderr)

    # Paired metric deltas — the direct inference per-pipeline CIs can't give.
    out.append("## Paired metric deltas (A − B), cluster-bootstrap 95% CIs\n")
    out.append("| A vs B | ΔHit@1 | ΔHit@10 | ΔMRR | ΔNDCG@10 | ΔNotRetr |")
    out.append("|---|---|---|---|---|---|")

    def dfmt(d, lo, hi, pct=True):
        star = "**" if (lo > 0 or hi < 0) else ""
        if pct:
            return f"{star}{d*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]{star}"
        return f"{star}{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"

    for i, j in pairs:
        dd = paired_delta_ci(ranks[i], ranks[j], clusters, B=args.bootstrap, seed=args.seed)
        out.append(f"| {labels[i]} vs {labels[j]} "
                   f"| {dfmt(*dd['Hit@1'])} | {dfmt(*dd['Hit@10'])} "
                   f"| {dfmt(*dd['MRR'], pct=False)} | {dfmt(*dd['NDCG@10'], pct=False)} "
                   f"| {dfmt(*dd['NotRetr'])} |")
    out.append("")
    out.append("*Deltas in percentage points (MRR/NDCG absolute). "
               "Bold = 95% cluster CI excludes 0.*")

    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
