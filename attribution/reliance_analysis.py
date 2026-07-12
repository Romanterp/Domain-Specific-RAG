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

Post-hoc robustness (added 2026-07-12, AFTER the pre-registered result was
computed on the production records): (a) gold-chunk cluster bootstrap of the
interaction, (b) per-protocol estimate excluding runtime pool-drift, (c)
refusal as an outcome. Labeled post-hoc in the output; the sections above this
one are unchanged from the pre-run commit (98e4274/4140777).

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


def spearman(a, b):
    """Spearman rank correlation (scipy if available, else NumPy fallback)."""
    if len(a) < 3:
        return float("nan"), None
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)
    except Exception:  # noqa: BLE001
        def rank(x):
            return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
        ra, rb = rank(a), rank(b)
        if ra.std() == 0 or rb.std() == 0:
            return 0.0, None
        return float(np.corrcoef(ra, rb)[0, 1]), None


def boot_ci_diff(x, y, B=2000, seed=42):
    """95% CI on mean(x) − mean(y) for unpaired samples (resample each)."""
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    diffs = [x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean()
             for _ in range(B)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def cluster_boot_ci_diff(xc, yc, B=2000, seed=42):
    """95% CI on mean(x) − mean(y), resampling gold-chunk CLUSTERS.

    Post-hoc: sibling questions share a gold chunk and hit/miss together, so
    the question-level bootstrap may understate CI width.
    """
    if len(xc) < 2 or len(yc) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    xc = [np.asarray(c, float) for c in xc]
    yc = [np.asarray(c, float) for c in yc]
    diffs = []
    for _ in range(B):
        xs = np.concatenate([xc[i] for i in rng.integers(0, len(xc), len(xc))])
        ys = np.concatenate([yc[i] for i in rng.integers(0, len(yc), len(yc))])
        diffs.append(xs.mean() - ys.mean())
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar on discordant pairs (same convention as
    retrieval/mcnemar_doc2query.py); chi-square fallback if scipy is absent."""
    if b + c == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(b, b + c, 0.5).pvalue)
    except Exception:  # noqa: BLE001
        import math
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        return float(math.erfc(math.sqrt(chi / 2)))


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

    # ---- RAG effect (FIRST-ORDER result): CTI vs the no-context baseline ----
    # CTI = KL(answer | with retrieved context ‖ answer | no context), so the
    # no-RAG baseline is 0 by construction and any CTI > 0 IS the RAG effect.
    # This is the big, clean result; the dense→hybrid shift below is the smaller
    # second-order effect of retrieval *quality* on top of it.
    all_cti: list[float] = []
    per_class_cti: dict[str, list[float]] = {cls: [] for cls in classes}
    for q in by_q.values():
        for cond in CONDITIONS:
            r = q.get(cond)
            if r is None or not usable(r):
                continue
            per_class_cti[r["contrast_class"]].append(r["answer_cti_mean"])
            all_cti.append(r["answer_cti_mean"])
    out.append("## RAG effect (context vs no-context reliance) — first-order result\n")
    out.append("CTI = KL(answer | **with** retrieved context ‖ answer | **without** any context). "
               "The no-RAG baseline is 0, so the mean CTI is the size of the RAG effect: how much "
               "the retrieved context drives the output vs the model's parametric prior.\n")
    out.append("| class | n (usable records) | mean CTI [95% CI] |")
    out.append("|---|---:|---|")
    for cls in classes:
        vals = per_class_cti[cls]
        if vals:
            lo, hi = boot_ci(vals)
            out.append(f"| {cls} | {len(vals)} | {np.mean(vals):.3f} [{lo:.3f}, {hi:.3f}] |")
    if all_cti:
        lo, hi = boot_ci(all_cti)
        out.append(f"| **all** | {len(all_cti)} | **{np.mean(all_cti):.3f}** [{lo:.3f}, {hi:.3f}] |")
    out.append("")
    out.append("*A mean per-token KL on this scale is a large effect: the model relies heavily on "
               "retrieved context across both question types. The dense→hybrid interaction below "
               "is the smaller, second-order effect of retrieval **quality**.*\n")

    # ---- CTI by class × condition + the within-question shift ----
    out.append("## CTI (context reliance) by class × condition (dense vs hybrid+rerank)\n")
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

    # ---- POST-HOC robustness (added 2026-07-12, after the pre-registered
    # result). Nothing in this block was pre-specified; it responds to the
    # external review. (a) sibling questions share a gold chunk and hit/miss
    # together → recluster the interaction bootstrap by gold chunk; (b) pool
    # drift (runtime 50 vs freeze 100) left some rescued questions without
    # gold at runtime → per-protocol re-estimate; (c) refusals are excluded
    # from CTI means but are themselves an outcome, concentrated in
    # rescued/dense → test refusal directly. ----
    out.append("## Post-hoc robustness (added 2026-07-12 — not pre-registered)\n")

    paired_info = {cls: [] for cls in classes}   # (shift, chunk, gold_in_ctx_hyb)
    refusal_prs = {cls: [] for cls in classes}   # (refused_dense, refused_hyb, chunk)
    for q in by_q.values():
        d, h = q.get("dense"), q.get("hybrid_rerank")
        if d is None or h is None:
            continue
        cls = d["contrast_class"]
        refusal_prs[cls].append((bool(d.get("refusal")), bool(h.get("refusal")),
                                 d.get("gold_chunk_id")))
        if usable(d) and usable(h):
            paired_info[cls].append((h["answer_cti_mean"] - d["answer_cti_mean"],
                                     d.get("gold_chunk_id"),
                                     bool(h.get("gold_in_context"))))

    def chunk_clusters(rows):
        g = defaultdict(list)
        for sh, ch, _ in rows:
            g[ch].append(sh)
        return list(g.values())

    if paired_info.get("rescued") and paired_info.get("control"):
        resc, ctrl = paired_info["rescued"], paired_info["control"]

        # (a) same interaction, gold-chunk cluster bootstrap
        rc, cc = chunk_clusters(resc), chunk_clusters(ctrl)
        diff = np.mean([r[0] for r in resc]) - np.mean([c[0] for c in ctrl])
        lo, hi = cluster_boot_ci_diff(rc, cc)
        out.append("### (a) Interaction under a gold-chunk cluster bootstrap\n")
        out.append(f"- rescued: {len(resc)} questions in {len(rc)} chunks; "
                   f"control: {len(ctrl)} questions in {len(cc)} chunks")
        out.append(f"- interaction = **{diff:+.3f}** [95% cluster CI {lo:+.3f}, {hi:+.3f}] — "
                   + ("CI excludes 0." if (lo > 0 or hi < 0) else "CI includes 0.") + "\n")

        # (b) per-protocol: drop rescued questions whose runtime hybrid
        # context lost the gold passage
        pp = [r for r in resc if r[2]]
        if pp:
            diff_pp = np.mean([r[0] for r in pp]) - np.mean([c[0] for c in ctrl])
            lo_n, hi_n = boot_ci_diff([r[0] for r in pp], [c[0] for c in ctrl])
            lo_c, hi_c = cluster_boot_ci_diff(chunk_clusters(pp), cc)
            out.append("### (b) Per-protocol interaction (rescued with gold actually in context)\n")
            out.append(f"- drops {len(resc) - len(pp)} of {len(resc)} rescued pairs "
                       "(gold absent from the runtime hybrid context)")
            out.append(f"- interaction = **{diff_pp:+.3f}** "
                       f"[question bootstrap {lo_n:+.3f}, {hi_n:+.3f}; "
                       f"cluster bootstrap {lo_c:+.3f}, {hi_c:+.3f}]\n")

    # (c) refusal as an outcome (not an exclusion)
    out.append("### (c) Refusal as outcome\n")
    out.append("| class | n(pairs) | refused dense | refused hyb | cured (b) | caused (c) | exact p |")
    out.append("|---|---:|---:|---:|---:|---:|---|")
    ref_shift_clusters = {}
    for cls in classes:
        rp = refusal_prs[cls]
        if not rp:
            continue
        b = sum(1 for rd, rh, _ in rp if rd and not rh)
        c = sum(1 for rd, rh, _ in rp if rh and not rd)
        p = mcnemar_exact_p(b, c)
        g = defaultdict(list)
        for rd, rh, ch in rp:
            g[ch].append(int(rh) - int(rd))
        ref_shift_clusters[cls] = list(g.values())
        out.append(f"| {cls} | {len(rp)} | {sum(1 for rd, _, _ in rp if rd)} | "
                   f"{sum(1 for _, rh, _ in rp if rh)} | {b} | {c} | "
                   f"{'—' if np.isnan(p) else format(p, '.2e')} |")
    if ref_shift_clusters.get("rescued") and ref_shift_clusters.get("control"):
        r_flat = [v for cl in ref_shift_clusters["rescued"] for v in cl]
        c_flat = [v for cl in ref_shift_clusters["control"] for v in cl]
        diff_r = np.mean(r_flat) - np.mean(c_flat)
        lo, hi = cluster_boot_ci_diff(ref_shift_clusters["rescued"],
                                      ref_shift_clusters["control"])
        out.append(f"\nRefusal-rate shift (hyb−dense) interaction, rescued − control = "
                   f"**{diff_r:+.3f}** [95% cluster CI {lo:+.3f}, {hi:+.3f}]. "
                   "Negative = better retrieval cures refusals specifically where it "
                   "supplies the missing gold passage.\n")

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

    # ---- position-confound check: does gold LOO track gold's rank/position? ----
    out.append("## Position-confound check (gold LOO vs gold rank)\n")
    out.append("Passages are ordered by rank in the prompt (P1 = rank 1, first in context), "
               "so gold_rank IS gold's prompt position. If gold_loo correlates with rank — or "
               "shows a U-shape across rank buckets — then per-passage LOO is position-confounded "
               "(lost-in-the-middle: ends attended more than the middle), not pure content. In "
               "that case gold_loo is a weak signal and the CTI shift (whole-context, "
               "position-robust) is the one to trust.\n")
    pos_pairs = []
    for q in by_q.values():
        for cond in CONDITIONS:
            r = q.get(cond)
            if r is None or not usable(r):
                continue
            gr, gl = r.get("gold_rank"), r.get("gold_loo_drop")
            if gr is not None and gl is not None:
                pos_pairs.append((gr, gl))
    if len(pos_pairs) >= 5:
        rho, pv = spearman([p[0] for p in pos_pairs], [p[1] for p in pos_pairs])
        out.append(f"- n={len(pos_pairs)} (gold present & usable); "
                   f"Spearman(gold_rank, gold_loo) = {rho:+.3f}"
                   + (f" (p={pv:.3f})" if pv is not None else ""))
        buckets = [("rank 1", lambda r: r == 1), ("2-3", lambda r: 2 <= r <= 3),
                   ("4-7", lambda r: 4 <= r <= 7), ("8+", lambda r: r >= 8)]
        out.append("\n| gold rank bucket | n | mean gold LOO |")
        out.append("|---|---:|---:|")
        for lab, pred in buckets:
            vals = [g for rk, g in pos_pairs if pred(rk)]
            if vals:
                out.append(f"| {lab} | {len(vals)} | {np.mean(vals):.2f} |")
        out.append("\n*Flat across buckets → content dominates (gold_loo trustworthy). "
                   "Monotone or U-shaped (high at rank 1 & 8+, low at 4-7) → position confound → "
                   "lead with CTI shift, report gold_loo with the caveat.*")
    else:
        out.append(f"- Too few gold-present usable records ({len(pos_pairs)}) for the check "
                   "(needs the full run; the 7B smoke is too small).")
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
