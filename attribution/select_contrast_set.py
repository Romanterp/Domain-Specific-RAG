"""
Freeze the attribution eval set — the retrieval "natural experiment".

The RQ3 finding rides on a controlled contrast: the SAME question and the SAME
generator, but different retrieved context. We get that for free from the
production retrieval run — comparing where each pipeline placed the gold passage
lets us bucket every eval question by how much the retrieval intervention
changed the context the model would see:

  rescued  — dense NEVER retrieved gold (top-100), hybrid+rerank put it in top-k.
             The cleanest causal case: dense answers with wrong/no context,
             hybrid+rerank has the right passage. Does the answer change?
  gain     — dense had gold deep (rank k+1..100), hybrid+rerank promoted it to
             top-k. Context improved but wasn't absent.
  control  — both pipelines nailed gold at rank 1. Retrieval is identical, so
             context-use SHOULD be identical — the baseline that isolates the
             retrieval effect from everything else.

Reads the per-query rank-of-gold JSONL written by eval_synthetic.py (which
already carries the question text + gold metadata), so this needs NO model and
NO doc2query — it runs on the finalized Round 6 output in seconds.

Usage
-----
    .venv311/Scripts/python.exe -m attribution.select_contrast_set
    # custom inputs / top-k:
    .venv311/Scripts/python.exe -m attribution.select_contrast_set \\
        --dense data/prod_dense.jsonl --hybrid-rerank data/prod_hybrid_rerank.jsonl \\
        --top-k 10 --condition none --out data/attribution_contrast_set.jsonl

    # emit a small plain-text question sample for a local smoke test:
    .venv311/Scripts/python.exe -m attribution.select_contrast_set \\
        --sample-txt data/attribution_smoke.txt --sample-n 3 --sample-class rescued
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_rows(path: Path, condition: str) -> dict[int, dict]:
    """q_idx -> full record (question, gold_*, rank_of_gold) for one condition."""
    out: dict[int, dict] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("condition") != condition:
            continue
        out[r["q_idx"]] = r
    return out


def hit(rank, k: int) -> bool:
    return rank is not None and rank <= k


def classify(dense_rank, hyr_rank, k: int) -> str | None:
    """Mutually-exclusive contrast class, or None if the question isn't part of
    the experiment (e.g. both missed, or only dense found it)."""
    if not hit(hyr_rank, k):
        return None
    if dense_rank is None:
        return "rescued"                      # dense never retrieved gold at all
    if not hit(dense_rank, k):
        return "gain"                          # dense had it deep; hyr promoted it
    if hit(dense_rank, 1) and hit(hyr_rank, 1):
        return "control"                       # both at rank 1 — retrieval identical
    return None                                # both found within k but not the clean control


def main() -> int:
    try:  # Windows consoles default to cp1252 and choke on → / arrows
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--dense", default=str(DATA_DIR / "prod_dense.jsonl"))
    ap.add_argument("--hybrid-rerank", default=str(DATA_DIR / "prod_hybrid_rerank.jsonl"))
    ap.add_argument("--condition", default="none")
    ap.add_argument("--top-k", type=int, default=10,
                    help="threshold for 'context available' (gold in top-k)")
    ap.add_argument("--out", default=str(DATA_DIR / "attribution_contrast_set.jsonl"))
    # optional: dump a plain-text question sample for a local smoke test
    ap.add_argument("--sample-txt", default=None,
                    help="also write N questions of --sample-class to this .txt")
    ap.add_argument("--sample-n", type=int, default=3)
    ap.add_argument("--sample-class", default="rescued",
                    choices=["rescued", "gain", "control"])
    args = ap.parse_args()

    dense = load_rows(Path(args.dense), args.condition)
    hyr = load_rows(Path(args.hybrid_rerank), args.condition)
    common = sorted(set(dense) & set(hyr))
    if not common:
        print("No overlapping q_idx — check --condition / paths.", file=sys.stderr)
        return 2

    records = []
    counts = {"rescued": 0, "gain": 0, "control": 0}
    for q in common:
        d, h = dense[q]["rank_of_gold"], hyr[q]["rank_of_gold"]
        cls = classify(d, h, args.top_k)
        if cls is None:
            continue
        counts[cls] += 1
        src = hyr[q]
        records.append({
            "q_idx": q,
            "contrast_class": cls,
            "question": src["question"],
            "gold_chunk_id": src["gold_chunk_id"],
            "gold_slug": src["gold_slug"],
            "gold_page": src.get("gold_page"),
            "dense_rank": d,
            "hybrid_rerank_rank": h,
        })

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"paired questions     : {len(common)}")
    print(f"  rescued (dense missed top-100, hyr top-{args.top_k}) : {counts['rescued']}")
    print(f"  gain    (dense deep, hyr top-{args.top_k})            : {counts['gain']}")
    print(f"  control (both rank 1)                            : {counts['control']}")
    print(f"  → contrast (rescued+gain)                        : {counts['rescued'] + counts['gain']}")
    print(f"Wrote {len(records)} records → {out_path}")

    if args.sample_txt:
        sample = [r["question"] for r in records if r["contrast_class"] == args.sample_class][: args.sample_n]
        txt = Path(args.sample_txt)
        header = (f"# {len(sample)} '{args.sample_class}' questions for a local attribution smoke test\n"
                  f"# (from {out_path.name}; gold was missed by dense, found by hybrid+rerank)\n")
        txt.write_text(header + "\n".join(sample) + "\n", encoding="utf-8")
        print(f"Wrote {len(sample)} sample questions → {txt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
