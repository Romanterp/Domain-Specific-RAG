"""Thesis figure pipeline

Generates the three thesis figures:

  fig_decomposition   RQ1: Hit@1 / Hit@10 / Not-retrieved bars with 95%
                      cluster-bootstrap CIs across the four pipelines.
                      
  fig_reliance        RQ3: (a) paired CTI by class x condition, recomputed
                      from data/reliance_records.jsonl with a seeded paired
                      question bootstrap and cross-checked against the frozen
                      means 
  fig_2x2             Cross-lens 2x2: reliance x pretraining-trace counts
                      

Run:  .venv311/Scripts/python.exe scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ---- palette (validated: scripts/validate_palette.js, light surface) --------
INK = "#0b0b0b"
SEC_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ORDINAL4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # pipeline progression
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # condition pair
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ_STEPS)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": SEC_INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def style_axis(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_jsonl(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# ============================================================================
# Figure 1 — RQ1 retrieval decomposition (frozen values, data/prod_ci_1024.md)
# ============================================================================
PIPELINES = ["dense", "dense\n+ rerank", "hybrid", "hybrid\n+ rerank"]
FROZEN = {  # metric -> (values, lo, hi) in the pipeline order above
    "Hit@1": ([23.7, 48.0, 34.3, 55.8], [21.4, 45.1, 31.8, 52.9], [25.9, 50.7, 36.9, 58.6]),
    "Hit@10": ([48.5, 67.1, 66.8, 81.6], [45.7, 64.4, 64.2, 79.4], [51.3, 69.6, 69.3, 83.6]),
    "Not retrieved": ([30.3, 30.3, 13.7, 13.7], [27.9, 27.9, 12.0, 12.0], [32.8, 32.8, 15.5, 15.5]),
}


def fig_decomposition():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    x = np.arange(4)
    for ax, (metric, (val, lo, hi)) in zip(axes, FROZEN.items()):
        val, lo, hi = map(np.asarray, (val, lo, hi))
        ax.bar(x, val, width=0.62, color=ORDINAL4, zorder=3)
        ax.errorbar(x, val, yerr=[val - lo, hi - val], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        for xi, v, h in zip(x, val, hi):
            ax.text(xi, h + 1.8, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
        title = metric + ("  (lower is better)" if metric == "Not retrieved" else "")
        ax.set_title(title, fontsize=9, color=INK, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PIPELINES, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        style_axis(ax)
    axes[0].set_ylabel("% of 2,481 queries", fontsize=8)
    fig.suptitle("Retrieval decomposition — production eval, 95% cluster-bootstrap CIs",
                 fontsize=10, color=INK, y=1.06)
    save(fig, "fig_decomposition")


# ============================================================================
# Figure 2 — RQ3 reliance: CTI by class x condition + interaction forest
# ============================================================================
FROZEN_CTI = {("control", "dense"): 1.655, ("control", "hybrid_rerank"): 1.664,
              ("rescued", "dense"): 1.288, ("rescued", "hybrid_rerank"): 1.415}
FOREST = [  # (label, estimate, lo, hi) — frozen, data/reliance_analysis.md
    ("Pre-registered\n(question bootstrap)", 0.117, 0.005, 0.239),
    ("Cluster bootstrap\n(post-hoc)", 0.117, -0.002, 0.227),
    ("Per-protocol\n(gold in context)", 0.155, 0.032, 0.274),
]


def fig_reliance():
    recs = load_jsonl(DATA / "reliance_records.jsonl")
    usable = {}
    for r in recs:
        if r.get("refusal") or r.get("answer_empty") or r.get("answer_cti_mean") is None:
            continue
        usable[(r["q_idx"], r["condition"])] = (r["contrast_class"], r["answer_cti_mean"])

    paired = {"control": [], "rescued": []}  # (cti_dense, cti_hyb) per question
    for (q, cond), (cls, cti) in usable.items():
        if cond == "dense" and (q, "hybrid_rerank") in usable:
            paired[cls].append((cti, usable[(q, "hybrid_rerank")][1]))

    rng = np.random.default_rng(42)
    stats = {}
    for cls, pairs in paired.items():
        arr = np.array(pairs)  # n x 2
        for j, cond in enumerate(("dense", "hybrid_rerank")):
            mean = arr[:, j].mean()
            boots = [arr[rng.integers(0, len(arr), len(arr)), j].mean() for _ in range(2000)]
            stats[(cls, cond)] = (mean, np.percentile(boots, 2.5), np.percentile(boots, 97.5))
            frozen = FROZEN_CTI[(cls, cond)]
            if abs(mean - frozen) > 0.01:
                print(f"WARN: {cls}/{cond} mean {mean:.3f} != frozen {frozen:.3f}",
                      file=sys.stderr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.7),
                                   gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.62})
    # (a) paired CTI means
    classes = ["control", "rescued"]
    for j, (cond, color, lbl) in enumerate(
            [("dense", BLUE, "dense"), ("hybrid_rerank", ORANGE, "hybrid + rerank")]):
        xs = np.arange(2) + (j - 0.5) * 0.18
        means = [stats[(c, cond)][0] for c in classes]
        los = [stats[(c, cond)][1] for c in classes]
        his = [stats[(c, cond)][2] for c in classes]
        ax1.errorbar(xs, means, yerr=[np.subtract(means, los), np.subtract(his, means)],
                     fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=3,
                     markersize=6, label=lbl, zorder=3)
        for xi, m in zip(xs, means):
            ax1.text(xi + 0.05, m, f"{m:.2f}", fontsize=7.5, color=SEC_INK,
                     va="center", ha="left")
    ax1.set_xticks(np.arange(2))
    ax1.set_xticklabels([f"control\n(n={len(paired['control'])} pairs)",
                         f"rescued\n(n={len(paired['rescued'])} pairs)"], fontsize=8)
    ax1.set_ylabel("mean CTI (context reliance)", fontsize=8)
    ax1.set_title("(a) Context reliance by class and condition", fontsize=9, pad=8)
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")
    style_axis(ax1)
    # (b) interaction forest (frozen estimates)
    ys = np.arange(len(FOREST))[::-1]
    for y, (lbl, est, lo, hi) in zip(ys, FOREST):
        ax2.plot([lo, hi], [y, y], color=BLUE, linewidth=1.6, zorder=3)
        ax2.plot(est, y, "o", color=BLUE, markersize=6, zorder=4)
        ax2.text(hi + 0.012, y, f"+{est:.3f} [{lo:+.3f}, {hi:+.3f}]",
                 fontsize=7, color=SEC_INK, va="center")
    ax2.axvline(0, color=AXIS, linewidth=0.9, linestyle="--", zorder=1)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f[0] for f in FOREST], fontsize=7.5)
    ax2.set_xlabel("interaction: rescued shift − control shift (ΔCTI)", fontsize=8)
    ax2.set_xlim(-0.08, 0.52)
    ax2.set_title("(b) Retrieval-quality interaction", fontsize=9, pad=8)
    style_axis(ax2, ygrid=False)
    ax2.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    fig.suptitle("RQ3 — retrieved context reliance (CTI), dense vs hybrid+rerank",
                 fontsize=10, color=INK, y=1.08)
    save(fig, "fig_reliance")


# ============================================================================
# Figure 3 — cross-lens 2x2 heatmap (computed from attribution_2x2.jsonl)
# ============================================================================
def fig_2x2():
    rows_ = load_jsonl(DATA / "attribution_2x2.jsonl")
    reliance_order = ["context", "mixed", "parametric"]
    trace_order = [True, False]
    counts = np.zeros((3, 2), int)
    untraced = 0
    for r in rows_:
        hit = r.get("pretraining_hit")
        if hit is None:
            untraced += 1
            continue
        i = reliance_order.index(r["reliance"])
        counts[i, trace_order.index(hit)] += 1
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    norm = counts / counts.max()
    ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(2):
            n = counts[i, j]
            dark = norm[i, j] > 0.55
            ax.text(j, i, f"{n:,}\n({n / total:.1%})", ha="center", va="center",
                    fontsize=9, color="white" if dark else INK)
    # outline the "no visible source" cell (parametric x no trace)
    ax.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, fill=False, edgecolor=INK,
                               linewidth=1.2, linestyle=(0, (3, 2))))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pretraining match", "no match"], fontsize=8.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["context-driven\n(CTI ≥ 0.30)", "mixed", "parametric\n(CTI ≤ 0.05)"],
                       fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sub = f"{total:,} spans" + (f" ({untraced} untraced excluded)" if untraced else "")
    ax.set_title(f"Cross-lens 2×2 — reliance × training-data overlap\n{sub}",
                 fontsize=9.5, color=INK, pad=10)
    fig.text(0.5, -0.04, "dashed cell: parametric and untraced-in-pretraining "
             "(“no visible source”)", ha="center", fontsize=7.5, color=SEC_INK)
    save(fig, "fig_2x2")


if __name__ == "__main__":
    print(f"-> {OUT}")
    fig_decomposition()
    fig_reliance()
    fig_2x2()
