"""
Doc2Query-- keep-fraction sweep on the paraphrased sparse A/B.

For each relevance-filtered expansion set (keep100/75/50/25 from
filter_expansions.py), build a BM25 index, evaluate it on the paraphrased
held-out queries, and run paired McNemar vs the un-augmented baseline. Answers:
does pruning low-relevance generated questions recover the Hit@1 the unfiltered
doc2query lost (cause #2), and where is the recall/precision knee?

Builds (and caches) data/bm25_doc2query_keep{75,50,25}.pkl. keep100 reuses the
existing data/bm25_doc2query_full.pkl (all questions kept = unfiltered).

Usage
-----
    .venv311/Scripts/python.exe -m retrieval.sweep_doc2query_filter
"""

import argparse
import json
import sys
from pathlib import Path

from retrieval.bm25 import BM25Index, CHUNKS_PATH, DOCUMENTS_PATH
from retrieval.eval_synthetic import load_questions, find_rank, metrics_for
from retrieval.mcnemar_doc2query import mcnemar, METRICS

DATA = Path(__file__).resolve().parent.parent / "data"
PARAPHRASED = DATA / "doc2query_eval_full_paraphrased.jsonl"
SYNTHETIC = DATA / "doc2query_eval_full.jsonl"
BASELINE = DATA / "bm25_index.pkl"
TOP_K = 100

# (label, expansions file or None if prebuilt, index pkl path)
RUNS = [
    ("baseline", None, BASELINE),
    ("keep100",  None, DATA / "bm25_doc2query_full.pkl"),
    ("keep75",   DATA / "doc2query_expansions_full_keep75.json", DATA / "bm25_doc2query_keep75.pkl"),
    ("keep50",   DATA / "doc2query_expansions_full_keep50.json", DATA / "bm25_doc2query_keep50.pkl"),
    ("keep25",   DATA / "doc2query_expansions_full_keep25.json", DATA / "bm25_doc2query_keep25.pkl"),
]


def get_index(exp_path: Path | None, pkl: Path) -> BM25Index:
    if pkl.exists():
        return BM25Index.load(pkl)
    print(f"  building {pkl.name} from {exp_path.name}…")
    expansions = json.loads(exp_path.read_text(encoding="utf-8"))
    idx = BM25Index.build(CHUNKS_PATH, DOCUMENTS_PATH, min_tokens=30,
                          expansions=expansions, include_metadata=True)
    idx.save(pkl)
    return idx


def ranks_for(idx: BM25Index, questions: list[dict]) -> list:
    return [find_rank(idx.query(q["question"], top_k=TOP_K), q["chunk_id"])
            for q in questions]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--questions", default=str(PARAPHRASED),
                    help="held-out eval queries (paraphrased or synthetic)")
    ap.add_argument("--tag", default=None,
                    help="output filename suffix; inferred from --questions if omitted")
    args = ap.parse_args()

    q_path = Path(args.questions)
    tag = args.tag or ("synthetic" if "paraphrased" not in q_path.name else "paraphrased")
    questions = load_questions(q_path)
    n = len(questions)
    print(f"Eval set [{tag}]: {n} queries ({q_path.name})\n")

    results = {}  # label -> ranks
    for label, exp_path, pkl in RUNS:
        print(f"[{label}] index…")
        idx = get_index(exp_path, pkl)
        results[label] = ranks_for(idx, questions)
        del idx

    base = results["baseline"]

    # ---- metrics table ----
    rows = []
    for label, _, _ in RUNS:
        m = metrics_for(results[label])
        nr = sum(1 for r in results[label] if r is None) / n
        rows.append((label, m["hit@1"], m["hit@5"], m["hit@10"], m["mrr"], nr))

    print(f"\n{'config':<10}{'Hit@1':>8}{'Hit@5':>8}{'Hit@10':>8}{'MRR':>8}{'NotRet':>8}")
    print("-" * 50)
    for label, h1, h5, h10, mrr, nr in rows:
        print(f"{label:<10}{h1*100:>7.1f}%{h5*100:>7.1f}%{h10*100:>7.1f}%{mrr:>8.3f}{nr*100:>7.1f}%")

    # ---- McNemar vs baseline (Hit@1 / Hit@10 / Retrieved@100) ----
    print(f"\nMcNemar vs baseline (lost = baseline hit & filtered miss; gain = reverse):")
    print(f"{'config':<10}{'metric':<15}{'lost':>6}{'gain':>6}{'net':>6}{'p':>10}")
    print("-" * 53)
    mc_rows = []
    for label, _, _ in RUNS:
        if label == "baseline":
            continue
        for mname, pred in METRICS:
            if mname == "Hit@5":
                continue
            bh, dh, b, c, p = mcnemar(base, results[label], pred)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"{label:<10}{mname:<15}{b:>6}{c:>6}{c-b:>+6}{p:>9.3f} {sig}")
            mc_rows.append((label, mname, b, c, c - b, p, sig))
        print()

    # ---- markdown ----
    out = ["# Doc2Query-- keep-fraction sweep — paraphrased sparse A/B\n",
           f"- Queries: **{n}** (`{q_path.name}`), BM25 top-{TOP_K}",
           f"- Baseline: `{BASELINE.name}` (no doc2query). keep100 = unfiltered "
           "doc2query; keep75/50/25 = relevance-pruned.\n",
           "## Retrieval metrics\n",
           "| config | Hit@1 | Hit@5 | Hit@10 | MRR | Not retrieved |",
           "|---|---|---|---|---|---|"]
    for label, h1, h5, h10, mrr, nr in rows:
        out.append(f"| {label} | {h1*100:.1f}% | {h5*100:.1f}% | {h10*100:.1f}% "
                   f"| {mrr:.3f} | {nr*100:.1f}% |")
    out += ["", "## McNemar vs baseline\n",
            "| config | metric | lost | gained | net | p-value |",
            "|---|---|---|---|---|---|"]
    for label, mname, b, c, net, p, sig in mc_rows:
        out.append(f"| {label} | {mname} | {b} | {c} | {net:+d} | {p:.3f} {sig} |")
    out += ["", "*lost = baseline hit & filtered-doc2query miss; gained = reverse. "
            "Two-sided exact binomial on discordant pairs.*"]
    out_path = DATA / f"doc2query_filter_sweep_{tag}.md"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
