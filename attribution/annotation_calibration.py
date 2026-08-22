"""Calibrate CTI against the annotation-campaign labels (thesis SS4.6).

Fills the three PENDING rows in status-of-numbers.md that were blocked on the
annotation campaign:
  1. CTI threshold calibration + bucket precision (AUC, per-bucket label mix,
     threshold sweep for CTI_STRONG/CTI_WEAK currently 0.30/0.05)
  2. Hallucination-cell precision (parametric x no-pretraining-trace spans:
     how often is "no visible source" actually not supported by the passages?)
  3. Answer-correctness by retrieval condition (first better-retrieval ->
     better-answers evidence; stratified-sample caveat applies)

Label source: data/annotation_labels.jsonl, latest label per (kind, key) wins
across annotators — llm-consensus today, human adjudication rows take
precedence automatically once appended by the workbench. The report states
which annotators actually contributed, so it stays honest about provenance.

Run:  .venv311/Scripts/python.exe -m attribution.annotation_calibration
"""
import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SPAN_CATS = ("supported", "partially", "not_supported", "cant_tell")
ANS_CATS = ("correct", "partially", "incorrect", "cant_tell")
BUCKETS = ("parametric", "mixed", "context")
CTI_STRONG = 0.30  # keep in sync with attribution/mirage.py — under calibration here
CTI_WEAK = 0.05


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def latest_labels(rows: list[dict], kind: str) -> dict[str, dict]:
    """Latest row per key across all annotators (ts order, file order breaks ties)."""
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("kind") != kind:
            continue
        k = r["key"]
        if k not in out or (r.get("ts") or "") >= (out[k].get("ts") or ""):
            out[k] = r
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 * P(tie)."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    order = np.concatenate([scores_pos, scores_neg])
    ranks = np.argsort(np.argsort(order, kind="mergesort"), kind="mergesort") + 1.0
    # midranks for ties
    _, inv, counts = np.unique(order, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    mid = (cum - (counts - 1) / 2.0)
    ranks = mid[inv]
    r_pos = ranks[: len(scores_pos)].sum()
    n1, n2 = len(scores_pos), len(scores_neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "n=0"
    return f"{p:.1%} [{lo:.1%}, {hi:.1%}] ({k}/{n})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--sample", default=str(DATA_DIR / "annotation_sample.jsonl"))
    ap.add_argument("--labels", default=str(DATA_DIR / "annotation_labels.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "cti_calibration.md"))
    args = ap.parse_args()

    sample = {s["span_key"]: s for s in load_jsonl(Path(args.sample))}
    label_rows = load_jsonl(Path(args.labels))
    span_lab = latest_labels(label_rows, "span")
    ans_lab = latest_labels(label_rows, "answer")

    spans = []
    for sk, s in sample.items():
        lab = span_lab.get(sk)
        if lab is None:
            continue
        spans.append({
            "span_key": sk, "cti_mean": s["cti_mean"], "cti_bucket": s["cti_bucket"],
            "pretraining_hit": s.get("pretraining_hit"), "condition": s["condition"],
            "label": lab["label"], "annotator": lab["annotator"],
        })
    n_unlabeled = len(sample) - len(spans)

    who_spans = Counter(x["annotator"] for x in spans)
    who_ans = Counter(v["annotator"] for v in ans_lab.values())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rep = ["# CTI calibration against annotation labels (SS4.6)\n"]
    rep.append(f"Generated {now} by `python -m attribution.annotation_calibration`. "
               f"Labels: latest per key from `annotation_labels.jsonl` — span rows by "
               f"{dict(who_spans)}, answer rows by {dict(who_ans)}. "
               "Re-run after human adjudication; provenance above updates itself.\n")
    if n_unlabeled:
        rep.append(f"**{n_unlabeled} sample spans have no label yet.**\n")

    # ---- 1. bucket x label cross-tab + precisions -------------------------
    rep.append(f"## 1. CTI bucket x support label ({len(spans)} spans)\n")
    cross = {b: Counter() for b in BUCKETS}
    for x in spans:
        cross[x["cti_bucket"]][x["label"]] += 1
    rep.append("| bucket | " + " | ".join(SPAN_CATS) + " | n |")
    rep.append("|---|" + "---|" * (len(SPAN_CATS) + 1))
    for b in BUCKETS:
        n = sum(cross[b].values())
        rep.append(f"| {b} | " + " | ".join(str(cross[b].get(c, 0)) for c in SPAN_CATS)
                   + f" | {n} |")
    rep.append("")
    ctx = cross["context"]
    par = cross["parametric"]
    n_ctx, n_par = sum(ctx.values()), sum(par.values())
    rep.append(f"- P(supported | context bucket, CTI >= {CTI_STRONG}): "
               f"{fmt_ci(ctx.get('supported', 0), n_ctx)}")
    rep.append(f"- P(supported or partially | context bucket): "
               f"{fmt_ci(ctx.get('supported', 0) + ctx.get('partially', 0), n_ctx)}")
    rep.append(f"- P(supported | parametric bucket, CTI <= {CTI_WEAK}): "
               f"{fmt_ci(par.get('supported', 0), n_par)}")
    rep.append(f"- P(not_supported | parametric bucket): "
               f"{fmt_ci(par.get('not_supported', 0), n_par)}\n")
    rep.append("Note: CTI is causal reliance, support is textual consistency — a "
               "parametric span can still echo passage content ('both' quadrant), so "
               "high P(supported|parametric) is not a calibration failure per se.\n")

    # ---- 2. CTI as a score for support (AUC + threshold sweep) ------------
    rep.append("## 2. CTI_mean as a predictor of passage support\n")
    judged = [x for x in spans if x["label"] != "cant_tell"]
    for name, pos_set in (("strict (supported vs partially+not_supported)", {"supported"}),
                          ("lenient (supported+partially vs not_supported)",
                           {"supported", "partially"})):
        pos = np.array([x["cti_mean"] for x in judged if x["label"] in pos_set])
        neg = np.array([x["cti_mean"] for x in judged if x["label"] not in pos_set])
        rep.append(f"- AUC {name}: **{auc(pos, neg):.3f}** "
                   f"(n_pos={len(pos)}, n_neg={len(neg)}; cant_tell excluded)")
    rep.append("")
    rep.append("Threshold sweep — flag = CTI_mean >= t (candidate CTI_STRONG):\n")
    rep.append("| t | flagged | P(supported \\| flagged) | P(supp+part \\| flagged) |")
    rep.append("|---|---|---|---|")
    for t in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
        fl = [x for x in judged if x["cti_mean"] >= t]
        k_s = sum(1 for x in fl if x["label"] == "supported")
        k_sp = sum(1 for x in fl if x["label"] in ("supported", "partially"))
        rep.append(f"| {t:.2f} | {len(fl)} | {fmt_ci(k_s, len(fl))} | "
                   f"{fmt_ci(k_sp, len(fl))} |")
    rep.append("")

    # ---- 3. hallucination-cell precision ---------------------------------
    rep.append("## 3. Hallucination-cell precision (parametric x no pretraining trace)\n")
    cell = [x for x in spans if x["cti_bucket"] == "parametric"
            and x["pretraining_hit"] is False]
    untraced = sum(1 for x in spans if x["cti_bucket"] == "parametric"
                   and x["pretraining_hit"] is None)
    dist = Counter(x["label"] for x in cell)
    rep.append(f"Sampled spans in the 'no visible source' cell: **{len(cell)}** "
               f"(+{untraced} parametric spans untraced, excluded).\n")
    rep.append("| label | n |")
    rep.append("|---|---|")
    for c in SPAN_CATS:
        rep.append(f"| {c} | {dist.get(c, 0)} |")
    rep.append("")
    n_cell = len(cell)
    rep.append(f"- P(not_supported | cell): {fmt_ci(dist.get('not_supported', 0), n_cell)}")
    rep.append(f"- P(not fully supported, i.e. partially or not_supported | cell): "
               f"{fmt_ci(dist.get('partially', 0) + dist.get('not_supported', 0), n_cell)}")
    rep.append(f"- P(supported | cell) — 'both'-quadrant leakage: "
               f"{fmt_ci(dist.get('supported', 0), n_cell)}\n")

    # ---- 4. answer correctness by condition ------------------------------
    rep.append("## 4. Whole-answer correctness by retrieval condition\n")
    ans = []
    for key, lab in ans_lab.items():
        cond = key.split(":", 1)[1]
        ans.append({"condition": cond, "label": lab["label"]})
    conds = sorted({a["condition"] for a in ans})
    rep.append("| condition | " + " | ".join(ANS_CATS) + " | n | correct | correct+partial |")
    rep.append("|---|" + "---|" * (len(ANS_CATS) + 3))
    for c in conds:
        rows_c = [a for a in ans if a["condition"] == c]
        d = Counter(a["label"] for a in rows_c)
        n = len(rows_c)
        rep.append(f"| {c} | " + " | ".join(str(d.get(k, 0)) for k in ANS_CATS)
                   + f" | {n} | {fmt_ci(d.get('correct', 0), n)} | "
                   f"{fmt_ci(d.get('correct', 0) + d.get('partially', 0), n)} |")
    rep.append("")
    rep.append("Caveat: records entered the sample via CTI-stratified SPAN sampling "
               "(parametric/mixed oversampled by design), so these are not unbiased "
               "population rates — report the condition CONTRAST, not the absolutes, "
               "and state the stratification.\n")

    out = Path(args.out)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
