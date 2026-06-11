"""
doc2query question-count ablation — how many generated questions to index?

Data-backed answer to "is 4 enough vs 5/6?" using the EXISTING prototype
expansions (no regeneration). For N = 0..max, build a BM25 index that uses the
first N expansion questions per chunk and evaluate on the held-out doc2query
eval queries. Where Hit@k stops improving is the count to use for the full run.

N=0 = no-augmentation baseline. Same questions, same eval set, only N varies —
so the curve is a clean measure of the marginal retrieval value per question.
(Indexed questions are in the order stored in expansions; reducing N drops the
*last* ones, so this is a conservative test — quality-ordered generation keeps
the strongest questions.)

Usage
-----
    .venv311/Scripts/python.exe -m retrieval.doc2query_count_ablation \\
        --expansions data/doc2query_expansions.json \\
        --questions data/doc2query_eval.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from retrieval.bm25 import BM25Index, CHUNKS_PATH, DOCUMENTS_PATH
from retrieval.eval_synthetic import load_questions, find_rank, metrics_for

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--expansions", default=str(DATA_DIR / "doc2query_expansions.json"))
    ap.add_argument("--questions", default=str(DATA_DIR / "doc2query_eval.jsonl"))
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    ap.add_argument("--documents", default=str(DOCUMENTS_PATH))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-n", type=int, default=None, help="default: max questions available")
    ap.add_argument("--out", default=str(DATA_DIR / "doc2query_count_ablation.md"))
    args = ap.parse_args()

    expansions = json.loads(Path(args.expansions).read_text(encoding="utf-8"))
    avail = max(len(v) for v in expansions.values())
    max_n = args.max_n if args.max_n is not None else avail
    questions = load_questions(Path(args.questions))
    print(f"{len(expansions)} augmented chunks (up to {avail} questions each) · "
          f"{len(questions)} held-out eval queries · sweeping N=0..{max_n}\n")

    results = []
    for n in range(0, max_n + 1):
        trunc = None if n == 0 else {cid: qs[:n] for cid, qs in expansions.items() if qs[:n]}
        idx = BM25Index.build(Path(args.chunks), Path(args.documents), expansions=trunc)
        ranks = [find_rank(idx.query(q["question"], top_k=args.top_k), q["chunk_id"])
                 for q in questions]
        m = metrics_for(ranks)
        notret = sum(1 for r in ranks if r is None) / (len(ranks) or 1)
        results.append((n, m, notret))
        print(f"N={n}:  Hit@1={m['hit@1']:.3f}  Hit@5={m['hit@5']:.3f}  "
              f"Hit@10={m['hit@10']:.3f}  MRR={m['mrr']:.3f}  not-retr={notret:.3f}")

    base = results[0][1]
    out = ["# doc2query indexing-count ablation\n",
           f"- {len(expansions)} augmented chunks · {len(questions)} held-out eval queries · top-{args.top_k}",
           "- N = number of generated questions indexed per chunk (N=0 = baseline)\n",
           "| N indexed | Hit@1 | Hit@5 | Hit@10 | MRR | Not retrieved | ΔHit@1 vs N=0 |",
           "|---|---|---|---|---|---|---|"]
    for n, m, notret in results:
        d = (m["hit@1"] - base["hit@1"]) * 100
        out.append(f"| {n} | {m['hit@1']*100:.1f}% | {m['hit@5']*100:.1f}% | "
                   f"{m['hit@10']*100:.1f}% | {m['mrr']:.3f} | {notret*100:.1f}% | {d:+.1f} pp |")
    out.append("")
    # marginal gains
    out.append("**Marginal Hit@1 gain per added question:**")
    for i in range(1, len(results)):
        dn = (results[i][1]["hit@1"] - results[i-1][1]["hit@1"]) * 100
        out.append(f"- N={i-1}→{i}: {dn:+.1f} pp")
    out.append("\nUse the smallest N past which the marginal gain is negligible.")
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
