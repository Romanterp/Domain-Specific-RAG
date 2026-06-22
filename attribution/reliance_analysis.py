"""
Analyse the reliance spine output (reliance_records.jsonl) → the RQ3 result.

The question: does better retrieval change what the model RELIES ON? We answer
it as a condition × question-class interaction, not a single number:

  - CTI shift = per-question (CTI under hybrid+rerank − CTI under dense).
  - On RESCUED questions (dense lacked the gold passage, hybrid surfaced it) the
    shift should be large — better retrieval gives the model something to rely
    on. On CONTROL questions (both already had gold at rank 1) the shift should
    be small. The shift being large *only where retrieval differs* is the
    evidence that retrieval QUALITY drives reliance, not question type.
  - LOO confirms causal use: on rescued/hybrid, removing the (now-present) gold
    passage should drop the answer's log-prob sharply.

Refusals and empty generations are excluded from CTI means (a "cannot answer"
collapses CTI for a degenerate reason, not low reliance).

Usage
-----
    .venv311/Scripts/python.exe -m attribution.reliance_analysis \\
        --records data/reliance_records.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONDITIONS = ["dense", "hybrid_rerank"]


def load(path: Path) -> dict:
    """q_idx -> {condition: record}."""
    by_q: dict = defaultdict(dict)
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        by_q[r["q_idx"]][r["condition"]] = r
    return by_q


def usable(rec: dict) -> bool:
    """A record whose CTI is meaningful (coherent, non-refusal answer)."""
    return rec is not None and not rec.get("answer_empty") and not rec.get("refusal")


def boot_ci(vals, B=2000, seed=42):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def cohens_d(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean() / a.std(ddof=1)) if len(a) > 1 and a.std(ddof=1) > 0 else float("nan")


def boot_ci_diff(x, y, B=2000, seed=42):
    """95% CI on mean(x) − mean(y) for unpaired samples (resample each)."""
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    diffs = [x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean()
             for _ in range(B)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def gold_top_fraction(rec: dict) -> bool | None:
    """Is the gold passage the single most-important passage by LOO?"""
    loo = rec.get("loo_drops") or []
    gr = rec.get("gold_rank")
    if not loo or gr is None or gr > len(loo):
        return None
    gi = gr - 1
    if loo[gi] is None or any(d is None for d in loo):
        return None
    return max(range(len(loo)), key=lambda k: loo[k]) == gi


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--records", default=str(DATA_DIR / "reliance_records.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "reliance_analysis.md"))
    args = ap.parse_args()

    by_q = load(Path(args.records))
    if not by_q:
        print("No records loaded.", file=sys.stderr)
        return 2

    classes = sorted({r["contrast_class"] for q in by_q.values() for r in q.values()})
    out = ["# RQ3 reliance — CTI shift × question-class interaction\n",
           f"- Records: {sum(len(v) for v in by_q.values())} "
           f"across {len(by_q)} questions; classes: {', '.join(classes)}\n"]

    # ---- refusal / empty bookkeeping ----
    out.append("## Answer health (excluded from CTI means)\n")
    out.append("| class | condition | n | refusals | empty |")
    out.append("|---|---|---:|---:|---:|")
    for cls in classes:
        for cond in CONDITIONS:
            recs = [q[cond] for q in by_q.values()
                    if cond in q and q[cond]["contrast_class"] == cls]
            ref = sum(1 for r in recs if r.get("refusal"))
            emp = sum(1 for r in recs if r.get("answer_empty"))
            out.append(f"| {cls} | {cond} | {len(recs)} | {ref} | {emp} |")
    out.append("")

    # ---- CTI by class × condition + the within-question shift ----
    out.append("## CTI (context reliance) by class × condition\n")
    out.append("| class | n(paired) | CTI dense | CTI hybrid+rr | shift (hyb−dense) [95% CI] | Cohen's d |")
    out.append("|---|---:|---:|---:|---|---:|")
    shifts_by_class = {}
    for cls in classes:
        pairs = []
        for q in by_q.values():
            d, h = q.get("dense"), q.get("hybrid_rerank")
            if d is None or h is None or d["contrast_class"] != cls:
                continue
            if not (usable(d) and usable(h)):
                continue
            pairs.append((d["answer_cti_mean"], h["answer_cti_mean"]))
        if not pairs:
            out.append(f"| {cls} | 0 | — | — | — | — |")
            continue
        dv = [p[0] for p in pairs]
        hv = [p[1] for p in pairs]
        sh = [b - a for a, b in pairs]
        shifts_by_class[cls] = sh
        lo, hi = boot_ci(sh)
        out.append(f"| {cls} | {len(pairs)} | {np.mean(dv):.3f} | {np.mean(hv):.3f} | "
                   f"{np.mean(sh):+.3f} [{lo:+.3f}, {hi:+.3f}] | {cohens_d(sh):.2f} |")
    out.append("")

    # ---- the interaction: rescued shift vs control shift ----
    if "rescued" in shifts_by_class and "control" in shifts_by_class:
        lo, hi = boot_ci_diff(shifts_by_class["rescued"], shifts_by_class["control"])
        diff = np.mean(shifts_by_class["rescued"]) - np.mean(shifts_by_class["control"])
        out.append("## Interaction (the headline)\n")
        out.append(f"Rescued shift − control shift = **{diff:+.3f}** [95% CI {lo:+.3f}, {hi:+.3f}].")
        verdict = ("CI excludes 0 → retrieval QUALITY drives reliance beyond question type."
                   if (lo > 0 or hi < 0) else
                   "CI includes 0 → no significant interaction at this n.")
        out.append(verdict + "\n")

    # ---- gold-LOO: causal reliance on the gold passage ----
    out.append("## Gold-passage LOO drop (causal reliance; higher = relied on more)\n")
    out.append("| class | condition | n(gold present) | mean gold LOO drop [95% CI] | gold = top passage |")
    out.append("|---|---|---:|---|---:|")
    for cls in classes:
        for cond in CONDITIONS:
            recs = [q[cond] for q in by_q.values()
                    if cond in q and q[cond]["contrast_class"] == cls and usable(q[cond])]
            drops = [r["gold_loo_drop"] for r in recs if r.get("gold_loo_drop") is not None]
            tops = [t for r in recs if (t := gold_top_fraction(r)) is not None]
            if not drops:
                out.append(f"| {cls} | {cond} | 0 | — | — |")
                continue
            lo, hi = boot_ci(drops)
            topfrac = f"{100*sum(tops)/len(tops):.0f}% (n={len(tops)})" if tops else "n/a"
            out.append(f"| {cls} | {cond} | {len(drops)} | "
                       f"{np.mean(drops):.2f} [{lo:.2f}, {hi:.2f}] | {topfrac} |")
    out.append("")
    out.append("*CTI = KL(with-context ‖ without-context), higher = more context-driven. "
               "Gold LOO drop = fall in the fixed answer's log-prob when the gold passage is "
               "attention-masked. Refusals/empties excluded from all means.*")

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
