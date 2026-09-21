"""
Answer-controlled CTI analysis — validate, then decompose.

Post-hoc sensitivity

Reads the cells written by attribution.answer_controlled_cti and reports, in
this order:

  1. Diagonal validation gate, the diagonal cells recompute a number that
     already exists (answer_cti_mean), so agreement validates passage
     rehydration, re-tokenisation, prompt format and aggregation in one shot.
     Nothing below is trustworthy until this passes, so it is printed first and
     its verdict is stamped on the report. Its outlier count also answers the
     pending "prompt-fallback contamination bound" item: production records
     predate the prompt_format field, so a record whose run silently fell back
     to a plain prompt can only be detected as a diagonal outlier.

  2. Identical-answer unit test. Where both conditions generated the same
     string, the answer effect and the home-field term are zero by construction
     and Delta must equal S exactly. Delta == S == the recorded shift is the
     assertion with teeth (|A| and |I| near zero also hold under a full context
     swap, so they alone would catch nothing).

  3. Decomposition. With X[a,c] = claim-mean CTI of answer a under context c:
         S     = X[h,h] - X[d,d]                              production shift
         Delta = 0.5*((X[d,h]-X[d,d]) + (X[h,h]-X[h,d]))      context effect
         A     = 0.5*((X[h,d]-X[d,d]) + (X[h,h]-X[d,h]))      answer effect
         I     = X[d,d] + X[h,h] - X[d,h] - X[h,d]            home-field
     S = Delta + A exactly, so DiD_Delta + DiD_A = DiD_S: the published
     interaction is split into a context share and an answer-text share.

  4. Delta(a_d) — the dense answer under both contexts is the non-circular row: 
     lexical copying alone predicts about -0.02 for
     it. Delta(a_h) is partly self-confirming (copy-only budget +0.04..+0.11
     against a published +0.117) because the hybrid answer quotes the hybrid
     context. The headline is Delta(a_d); Delta(a_h) is reported and labelled.

  5. Signed metric. CTI is an unsigned expectation KL, so a context that
     confidently contradicts the forced text scores as high as one that
     supports it. The signed companion (the same log-ratio at the realised
     token) separates them, and only it licenses the word "reliance" off the
     diagonal.

  6. Robustness. Copy-coverage adjustment, citation-swap stratification,
     residual soft-refusal exclusion, token-mean aggregation, and a
     Monte-Carlo stability footnote (bootstrap endpoints here sit within ~0.005
     of zero, the same size as MC noise at B=2000, so no verdict flip that lies
     inside that spread may be claimed).

Usage
-----
    .venv311/Scripts/python.exe -m attribution.answer_controlled_analysis \
        --cells data/answer_controlled_cells.jsonl \
        --out data/answer_controlled_analysis.md
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from attribution.answer_controlled_cti import read_jsonl
from attribution.reliance_analysis import (boot_ci, boot_ci_diff,
                                           cluster_boot_ci_diff, cohens_d, spearman)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Published targets the diagonal must reproduce (data/reliance_analysis.md).
TARGET_DIAG = {("control", "dense"): 1.6545, ("control", "hybrid_rerank"): 1.6642,
               ("rescued", "dense"): 1.2878, ("rescued", "hybrid_rerank"): 1.4147}
TARGET_INTERACTION = 0.1172

# Pre-committed gate. Loosened only by the 4dp storage floor (5e-5) and the
# bf16 reduction-order difference a sharded forward can introduce.
GATE = dict(max_abs_bias=0.01, max_median_abs=0.01, min_r=0.995,
            max_frac_gross=0.05, gross=0.10, max_did_gap=0.020)


def cell(rec, a, c, key="cti_mean"):
    return rec["cells"][f"{a}|{c}"][key]


def derive(rec, key="cti_mean"):
    """S, Delta, A, I and the two row simple effects for one question."""
    dd, dh = cell(rec, "d", "d", key), cell(rec, "d", "h", key)
    hd, hh = cell(rec, "h", "d", key), cell(rec, "h", "h", key)
    return {
        "S": hh - dd,
        "Delta": 0.5 * ((dh - dd) + (hh - hd)),
        "A": 0.5 * ((hd - dd) + (hh - dh)),
        "I": dd + hh - dh - hd,
        "Delta_ad": dh - dd,     # dense answer, hybrid minus dense context
        "Delta_ah": hh - hd,     # hybrid answer, same contrast
    }


def fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def by_class(recs, fn):
    out = {}
    for r in recs:
        out.setdefault(r["contrast_class"], []).append(fn(r))
    return out


def clusters(recs, fn):
    """Values grouped by gold chunk — the house cluster-bootstrap unit."""
    g = {}
    for r in recs:
        g.setdefault(r.get("gold_chunk_id"), []).append(fn(r))
    return list(g.values())


def triplet(recs, fn, label, out):
    """Report rescued-minus-control for a per-question quantity, house style."""
    vals = by_class(recs, fn)
    r_, c_ = vals.get("rescued", []), vals.get("control", [])
    if len(r_) < 2 or len(c_) < 2:
        out.append(f"| {label} | insufficient n | | | |")
        return float("nan")
    diff = float(np.mean(r_) - np.mean(c_))
    q_ci = boot_ci_diff(r_, c_)
    rc = clusters([r for r in recs if r["contrast_class"] == "rescued"], fn)
    cc = clusters([r for r in recs if r["contrast_class"] == "control"], fn)
    cl_ci = cluster_boot_ci_diff(rc, cc)
    pp = [r for r in recs if r["contrast_class"] != "rescued"
          or r["gold_in_context"].get("hybrid_rerank")]
    ppv = by_class(pp, fn)
    pp_d = float(np.mean(ppv["rescued"]) - np.mean(ppv["control"])) if len(ppv.get("rescued", [])) > 1 else float("nan")
    out.append(f"| {label} | {diff:+.4f} | {fmt_ci(q_ci)} | {fmt_ci(cl_ci)} | "
               f"{pp_d:+.4f} (n={len(ppv.get('rescued', []))}) |")
    return diff


def diagonal_gate(recs, out) -> bool:
    """Validate recomputed diagonal against the recorded production numbers."""
    out.append("\n## 1. Diagonal validation gate\n")
    out.append("Recomputed own-answer-under-own-context cells vs the recorded "
               "`answer_cti_mean`. This validates passage rehydration, "
               "re-tokenisation, prompt format and aggregation together.\n")
    out.append("| Cell | n | recomputed | recorded | bias | median abs | r | frac abs>0.10 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    passed, worst = True, []
    for a, cond in (("d", "dense"), ("h", "hybrid_rerank")):
        new = np.array([cell(r, a, a) for r in recs], float)
        old = np.array([r["recorded_cti"][cond] or 0.0 for r in recs], float)
        d = new - old
        bias, med = float(d.mean()), float(np.median(np.abs(d)))
        r_p = float(np.corrcoef(new, old)[0, 1]) if len(new) > 2 and new.std() > 0 else float("nan")
        frac = float((np.abs(d) > GATE["gross"]).mean())
        ok = (abs(bias) <= GATE["max_abs_bias"] and med <= GATE["max_median_abs"]
              and (np.isnan(r_p) or r_p >= GATE["min_r"]) and frac <= GATE["max_frac_gross"])
        passed &= ok
        out.append(f"| {cond} | {len(new)} | {new.mean():.4f} | {old.mean():.4f} | "
                   f"{bias:+.5f} | {med:.5f} | {r_p:.5f} | {frac:.3%} |")
        # Bland-Altman limits of agreement + the outliers that answer the
        # prompt-fallback contamination item.
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        worst.append((cond, sd, [r["q_idx"] for r in recs
                                 if abs(cell(r, a, a) - (r["recorded_cti"][cond] or 0.0)) > GATE["gross"]]))
    for cond, sd, ids in worst:
        out.append(f"\n- **{cond}** limits of agreement ±{1.96 * sd:.4f}; "
                   f"{len(ids)} record(s) beyond ±{GATE['gross']}"
                   + (f" (q_idx {ids[:20]}{'…' if len(ids) > 20 else ''})" if ids else "")
                   + ". Production records predate the `prompt_format` field, so "
                     "these are the only visible candidates for a silent plain-prompt "
                     "fallback; a count of 0 bounds that contamination at 0.")

    # Recomputed per-class shifts + interaction vs the published values.
    sh = by_class(recs, lambda r: derive(r)["S"])
    out.append("\n| Recomputed from diagonal | value | published |")
    out.append("|---|---:|---:|")
    for k in ("rescued", "control"):
        if sh.get(k):
            out.append(f"| {k} shift | {np.mean(sh[k]):+.4f} | "
                       f"{'+0.1269' if k == 'rescued' else '+0.0097'} |")
    did = float("nan")
    if sh.get("rescued") and sh.get("control"):
        did = float(np.mean(sh["rescued"]) - np.mean(sh["control"]))
        gap = abs(did - TARGET_INTERACTION)
        passed &= gap <= GATE["max_did_gap"]
        out.append(f"| interaction (DiD_S) | {did:+.4f} | {TARGET_INTERACTION:+.4f} "
                   f"(gap {gap:.4f}) |")
    out.append(f"\n**GATE: {'PASS' if passed else 'FAIL'}** "
               f"(criteria: |bias| ≤ {GATE['max_abs_bias']}, median |diff| ≤ "
               f"{GATE['max_median_abs']}, r ≥ {GATE['min_r']}, "
               f"≤{GATE['max_frac_gross']:.0%} beyond ±{GATE['gross']}, "
               f"DiD_S within ±{GATE['max_did_gap']} of published).")
    if not passed:
        out.append("\n> Gate failed — treat every number below as UNVALIDATED. "
                   "First suspects, in order: a plain-prompt fallback during the "
                   "production run (check the Habrok logs), bf16 reduction-order "
                   "differences from a different GPU placement, then re-tokenisation.")
    return passed


def identical_answer_test(recs, out):
    ident = [r for r in recs if r.get("identical_answers")]
    out.append("\n## 2. Identical-answer unit test\n")
    if not ident:
        out.append("No identical-answer pairs in this sample — test not run.")
        return
    A = [abs(derive(r)["A"]) for r in ident]
    I = [abs(derive(r)["I"]) for r in ident]
    gap = [abs(derive(r)["Delta"] - derive(r)["S"]) for r in ident]
    rec_gap = [abs(derive(r)["S"] - ((r["recorded_cti"]["hybrid_rerank"] or 0.0)
                                     - (r["recorded_cti"]["dense"] or 0.0))) for r in ident]
    out.append(f"n = {len(ident)} pairs whose two conditions generated the same string. "
               f"Then the answer effect and home-field term are 0 by construction and "
               f"Delta must equal S.\n")
    out.append(f"- max |A| = {max(A):.6f}, max |I| = {max(I):.6f} (expect ~0)")
    out.append(f"- max |Delta − S| = {max(gap):.6f} (expect ~0)")
    out.append(f"- max |S − recorded shift| = {max(rec_gap):.4f} "
               "— the assertion with teeth: |A| and |I| near zero also hold under a "
               "full context swap, so only agreement with the RECORDED shift proves "
               "the two contexts are wired to the right columns.")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--cells", default=str(DATA_DIR / "answer_controlled_cells.jsonl"))
    ap.add_argument("--refusal-cells", default=str(DATA_DIR / "answer_controlled_refusal.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "answer_controlled_analysis.md"))
    ap.add_argument("--seeds", type=int, default=25,
                    help="extra bootstrap seeds for the MC-stability footnote")
    args = ap.parse_args()

    path = Path(args.cells)
    if not path.exists():
        print(f"No cells file at {path} — run attribution.answer_controlled_cti first.",
              file=sys.stderr)
        return 2
    recs = [r for r in read_jsonl(path) if r.get("arm", "main") == "main"]
    if not recs:
        print("No main-arm records found.", file=sys.stderr)
        return 2

    out = ["# Answer-controlled CTI (teacher-forced 2x2) — post-hoc sensitivity",
           "",
           "**Label: post-hoc, added 2026-09-19, not pre-registered.** The "
           "pre-registered RQ3 interaction (+0.117) remains the headline; this "
           "decomposes it. Bootstrap conventions are the house ones (B=2000, "
           "seed=42, percentile CI, gold-chunk clusters), imported from "
           "`reliance_analysis` so they cannot drift.",
           "",
           f"- Cells: `{path}` — {len(recs)} questions "
           f"({', '.join(f'{k} {len(v)}' for k, v in sorted(by_class(recs, lambda r: r).items()))})",
           f"- Model: {recs[0].get('model')}",
           f"- Non-chat prompt format in any cell: "
           f"{sum(1 for r in recs if not r.get('all_chat_format'))} question(s)"]
    if any(r.get("top_k", 10) != 10 for r in recs):
        out.append("- **WARNING: truncated contexts (`--top-k`) — diagonal is "
                   "NOT comparable to production. Smoke run only.**")

    gate_ok = diagonal_gate(recs, out)
    identical_answer_test(recs, out)

    # ---- 3. decomposition -------------------------------------------------
    out.append("\n## 3. Decomposition of the published interaction\n")
    out.append("S = Delta + A exactly, so DiD_S splits into a context share and "
               "an answer-text share.\n")
    out.append("| Quantity | rescued − control | question boot 95% | cluster boot 95% | per-protocol |")
    out.append("|---|---:|---|---|---|")
    did_delta = triplet(recs, lambda r: derive(r)["Delta"], "**DiD_Delta (context) — PRIMARY**", out)
    did_a = triplet(recs, lambda r: derive(r)["A"], "DiD_A (answer text)", out)
    did_s = triplet(recs, lambda r: derive(r)["S"], "DiD_S (= published shift)", out)
    out.append(f"\nAdditivity check: DiD_Delta + DiD_A = {did_delta + did_a:+.4f} "
               f"vs DiD_S = {did_s:+.4f} (must agree to rounding).")
    if did_s:
        out.append(f"\nContext share of the published interaction: "
                   f"**{did_delta / did_s:.1%}**; answer-text share: {did_a / did_s:.1%}.")

    out.append("\n### Per-class context effect (Delta)\n")
    out.append("| Class | n | mean Delta | 95% CI | Cohen's d | mean S (production) |")
    out.append("|---|---:|---:|---|---:|---:|")
    for k, v in sorted(by_class(recs, lambda r: derive(r)["Delta"]).items()):
        s_v = by_class(recs, lambda r: derive(r)["S"])[k]
        out.append(f"| {k} | {len(v)} | {np.mean(v):+.4f} | {fmt_ci(boot_ci(v))} | "
                   f"{cohens_d(v):.3f} | {np.mean(s_v):+.4f} |")
    sd_d = st.pstdev([derive(r)["Delta"] for r in recs])
    sd_s = st.pstdev([derive(r)["S"] for r in recs])
    out.append(f"\nVariance check: sd(Delta) = {sd_d:.3f} vs sd(S) = {sd_s:.3f}. "
               "Holding the answer fixed was expected to shrink the noise "
               "substantially; if it did not, the design's precision premise failed "
               "and the run is underpowered rather than null.")

    # ---- 4. the two rows --------------------------------------------------
    out.append("\n## 4. The two rows separately (circularity)\n")
    out.append("| Quantity | rescued − control | question boot 95% | cluster boot 95% | per-protocol |")
    out.append("|---|---:|---|---|---|")
    triplet(recs, lambda r: derive(r)["Delta_ad"], "**Delta(a_d) — dense answer, NON-CIRCULAR**", out)
    triplet(recs, lambda r: derive(r)["Delta_ah"], "Delta(a_h) — hybrid answer, SELF-CONFIRMING", out)
    triplet(recs, lambda r: derive(r)["I"], "I (home-field; carries the citation artifact)", out)
    cov = {}
    for r in recs:
        own_minus_foreign_d = r["cells"]["d|d"]["coverage_3gram"] - r["cells"]["d|h"]["coverage_3gram"]
        own_minus_foreign_h = r["cells"]["h|h"]["coverage_3gram"] - r["cells"]["h|d"]["coverage_3gram"]
        cov.setdefault(r["contrast_class"], []).append((own_minus_foreign_d, own_minus_foreign_h))
    out.append("\n### Copy-overlap budget\n")
    out.append("| Class | own−foreign coverage, dense answer | hybrid answer |")
    out.append("|---|---:|---:|")
    for k, v in sorted(cov.items()):
        out.append(f"| {k} | {np.mean([x[0] for x in v]):+.4f} | {np.mean([x[1] for x in v]):+.4f} |")
    out.append("\nCTI rises roughly +0.65 nats per unit 3-gram coverage, so the "
               "hybrid-answer row carries a copy-only budget of about +0.04..+0.11 "
               "for rescued — comparable to the whole published interaction. The "
               "dense-answer row's budget is about −0.02, which is why it is the "
               "headline: a clearly positive Delta(a_d) cannot be manufactured by "
               "lexical self-confirmation. Slope caveat: that coefficient is "
               "cross-sectional across questions and is not identified for a "
               "within-question context swap — which is precisely what this "
               "experiment measures directly.")

    # ---- 5. signed metric -------------------------------------------------
    out.append("\n## 5. Signed metric (support vs contradiction)\n")
    out.append("CTI is an unsigned expectation KL: a context that confidently "
               "contradicts the forced text scores as high as one that supports it. "
               "The signed companion is the same log-ratio at the realised token.\n")
    out.append("| Quantity (signed) | rescued − control | question boot 95% | cluster boot 95% | per-protocol |")
    out.append("|---|---:|---|---|---|")
    triplet(recs, lambda r: derive(r, "signed_mean")["Delta"], "DiD_Delta (signed)", out)
    triplet(recs, lambda r: derive(r, "signed_mean")["Delta_ad"], "Delta(a_d) (signed)", out)
    out.append("\n| Class | mean signed Delta(a_d) | median | frac > 0 |")
    out.append("|---|---:|---:|---:|")
    for k, v in sorted(by_class(recs, lambda r: derive(r, "signed_mean")["Delta_ad"]).items()):
        out.append(f"| {k} | {np.mean(v):+.4f} | {np.median(v):+.4f} | {np.mean(np.array(v) > 0):.1%} |")
    out.append("\nMedians and the positive fraction are reported because these "
               "quantities are heavy-tailed (the gold-LOO analogue has mean 33.9 "
               "against median 9.2), so a mean alone tracks a ~10% tail.")

    # ---- 6. robustness ----------------------------------------------------
    out.append("\n## 6. Robustness\n")
    strata = {"neither cites": [], "both cite": [], "one only": []}
    for r in recs:
        cd = r["citations"].get("d", {}).get("n_tags", 0) > 0
        ch = r["citations"].get("h", {}).get("n_tags", 0) > 0
        k = "both cite" if cd and ch else ("neither cites" if not cd and not ch else "one only")
        strata[k].append(r)
    out.append("**Citation stratification** — ~78% of `[P#]` tags point at a "
               "different chunk after a context swap, and the artifact cancels from "
               "the symmetric main effects only when both rows carry an equal boost, "
               "so this is a required row, not an optional one.\n")
    out.append("| Stratum | n | DiD_Delta | rescued n |")
    out.append("|---|---:|---:|---:|")
    for k, v in strata.items():
        vals = by_class(v, lambda r: derive(r)["Delta"])
        d = (float(np.mean(vals["rescued"]) - np.mean(vals["control"]))
             if len(vals.get("rescued", [])) > 1 and len(vals.get("control", [])) > 1 else float("nan"))
        out.append(f"| {k} | {len(v)} | {d:+.4f} | {len(vals.get('rescued', []))} |")

    clean = [r for r in recs if not any(r.get("soft_refusal", {}).values())]
    v = by_class(clean, lambda r: derive(r)["Delta"])
    if len(v.get("rescued", [])) > 1 and len(v.get("control", [])) > 1:
        out.append(f"\n**Residual refusals** — the production flag misses wording like "
                   f"'cannot be determined'; these survive ~5x more often in "
                   f"rescued/dense than rescued/hybrid and depress exactly the arm that "
                   f"drives the interaction. Excluding them (n={len(clean)}): "
                   f"DiD_Delta = {np.mean(v['rescued']) - np.mean(v['control']):+.4f}.")

    v = by_class(recs, lambda r: derive(r, "token_cti_mean")["Delta"])
    if len(v.get("rescued", [])) > 1 and len(v.get("control", [])) > 1:
        out.append(f"\n**Aggregation** — claim-means weight a 4-token sentence like a "
                   f"40-token one. Plain token-weighted mean: DiD_Delta = "
                   f"{np.mean(v['rescued']) - np.mean(v['control']):+.4f}.")

    dv = by_class(recs, lambda r: derive(r)["Delta"])
    if len(dv.get("rescued", [])) > 1 and len(dv.get("control", [])) > 1:
        lows = [boot_ci_diff(dv["rescued"], dv["control"], seed=s)[0] for s in range(args.seeds)]
        out.append(f"\n**Monte-Carlo stability** — over {args.seeds} bootstrap seeds the "
                   f"lower endpoint of DiD_Delta ranges [{min(lows):+.4f}, {max(lows):+.4f}] "
                   f"(sd {np.std(lows):.4f}). The published interaction's own endpoint sits "
                   f"~0.005 from zero, the same size as this MC noise, so no "
                   f"excludes-zero / includes-zero verdict flip inside that spread may be "
                   f"claimed as a finding.")

    cv = [r["cells"]["d|h"]["coverage_3gram"] - r["cells"]["d|d"]["coverage_3gram"] for r in recs]
    dvv = [derive(r)["Delta_ad"] for r in recs]
    rho, p = spearman(cv, dvv)
    out.append(f"\n**Copy covariate** — Spearman(Δcoverage, Delta(a_d)) = {rho:+.3f}"
               + (f" (p={p:.3g})" if p is not None else "") +
               ". A near-zero correlation supports reading Delta(a_d) as context "
               "effect rather than copy artifact.")

    # ---- 7. refusal arm ---------------------------------------------------
    rpath = Path(args.refusal_cells)
    rrecs = [r for r in read_jsonl(rpath) if r.get("arm") == "refusal"] if rpath.exists() else []
    out.append("\n## 7. Refusal arm (the excluded pairs)\n")
    if not rrecs:
        out.append(f"Not run (no `{rpath.name}`). This arm forces the dense REFUSAL "
                   "text under both contexts on the 54 pairs the main arm must "
                   "exclude — the strongest effect in the dataset, and structurally "
                   "immune to copy circularity because a refusal contains no gold "
                   "content to quote.")
    else:
        out.append("| Class | n | signed Δ(hybrid − dense context) | 95% CI | frac < 0 |")
        out.append("|---|---:|---:|---|---:|")
        for k, v in sorted(by_class(
                rrecs, lambda r: r["cells"]["r|h"]["signed_mean"]
                - r["cells"]["r|d"]["signed_mean"]).items()):
            out.append(f"| {k} | {len(v)} | {np.mean(v):+.4f} | {fmt_ci(boot_ci(v))} | "
                       f"{np.mean(np.array(v) < 0):.1%} |")
        out.append("\nA NEGATIVE value is the predicted direction: the better context "
                   "should make 'I cannot answer' markedly less probable.")

    # ---- interpretation ---------------------------------------------------
    out.append("\n## Interpretation rules (pre-committed)\n")
    out.append("- **Claim language.** The answer is a deterministic function of the "
               "context, so the row factor is post-treatment. This is a DESCRIPTIVE "
               "DECOMPOSITION of an observed difference, never an unconfounded causal "
               "estimate of retrieval quality on reliance.")
    out.append("- **Asymmetric reading.** Refusal exclusion is differential (it drops "
               "all 54 refusal-cured pairs, 44 of them rescued). A positive result is "
               "therefore conservative and usable; a NULL is NOT decisive, because the "
               "retained rescued questions are exactly those where dense retrieval was "
               "already adequate. The estimand is 'the context effect among rescued "
               "questions on which dense retrieval nevertheless supported an answer'.")
    out.append("- **Which row counts.** Delta(a_d) is the result that answers the "
               "stated limitation. Delta(a_h) is reported for completeness and is "
               "partly self-confirming.")
    out.append("- **Length.** Within a row the sentence split and token offsets depend "
               "only on the answer text, so length and the mean-of-sentence-means "
               "normalisation cancel exactly — the context main effect is free of the "
               "length objection. They do NOT cancel across rows, so the answer main "
               "effect and home-field term remain length-biased.")

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote {args.out}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
