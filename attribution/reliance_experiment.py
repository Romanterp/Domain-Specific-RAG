"""
RQ3 core experiment — does better retrieval change what the model RELIES ON?

The intrinsic spine (forward-only; no gradients, no inseq). For each contrast-set
question, under BOTH retrieval conditions {dense, hybrid+rerank}:
  1. retrieve that condition's passages,
  2. generate the answer (greedy, fixed),
  3. per-claim CTI = contrastive context-sensitivity (with vs without context) —
     "did the retrieved context drive this claim?" (mirage, cci OFF),
  4. LOO = leave-one-passage-out re-scoring of the fixed answer — the causal,
     faithful "which passage mattered" signal (replaces noisy gradient CCI).

The analysis (separate script) then asks the real question via the condition ×
question-class interaction: CTI rises hybrid-vs-dense on RESCUED questions (dense
lacked the gold passage) but not on CONTROL questions (both already had it) — the
gap being large only where retrieval differs is the evidence that retrieval
*quality* drives reliance, not question type. LOO confirms hybrid's reliance is
causally on the gold passage.

Staged for a 16 GB card (retriever freed before the generator). Resumable per
(q_idx, condition). Local 7B for a --limit smoke; 32B on Habrok for the real run.

Usage
-----
    # local smoke (7B, 4 questions, both conditions)
    .venv311/Scripts/python.exe -m attribution.reliance_experiment --limit 4

    # production (Habrok, 32B)
    python -m attribution.reliance_experiment --model allenai/Olmo-3.1-32B-Instruct \\
        --out data/reliance_records.jsonl --resume
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONDITIONS = {
    "dense": dict(use_hybrid=False, use_rerank=False),
    "hybrid_rerank": dict(use_hybrid=True, use_rerank=True),
}


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_contrast(path: Path, classes: set[str], limit: int | None) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("contrast_class") in classes:
            out.append(r)
    out.sort(key=lambda r: r["q_idx"])  # deterministic order for resume
    return out[:limit] if limit else out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--contrast-set", default=str(DATA_DIR / "attribution_contrast_set.jsonl"))
    ap.add_argument("--classes", nargs="+", default=["rescued", "control"],
                    help="contrast classes to include (rescued + control = the interaction)")
    ap.add_argument("--limit", type=int, help="first N questions (smoke test)")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--collection", default="theisus_none")
    ap.add_argument("--bm25-path", default=str(DATA_DIR / "bm25_index.pkl"))
    ap.add_argument("--qdrant-path", default=str(DATA_DIR / "qdrant"))
    ap.add_argument("--candidate-pool", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10,
                    help="passages of context per condition. MUST match the contrast "
                         "set's classification k (select_contrast_set --top-k, default 10) "
                         "or class labels diverge from runtime gold presence (esp. 'gain').")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default=str(DATA_DIR / "reliance_records.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (SLURM array)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (SLURM array)")
    args = ap.parse_args()

    questions = load_contrast(Path(args.contrast_set), set(args.classes), args.limit)
    if args.num_shards > 1:  # P4: SLURM array sharding (q_idx is globally unique)
        questions = [q for n, q in enumerate(questions) if n % args.num_shards == args.shard]
    if not questions:
        print("No questions loaded (check --classes / path / shard).", file=sys.stderr)
        return 2
    print(f"{len(questions)} questions × {len(CONDITIONS)} conditions; model={args.model}"
          + (f"  [shard {args.shard}/{args.num_shards}]" if args.num_shards > 1 else ""))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:  # B2: a torn final line from a crash must not abort the whole resume
                r = json.loads(ln)
            except json.JSONDecodeError:
                print(f"  skip corrupt resume line ({len(ln)} chars)", file=sys.stderr)
                continue
            done.add(f"{r['q_idx']}:{r['condition']}")
        print(f"Resuming — {len(done)} (q,condition) records already done")

    # ---- Stage 1: retrieve both conditions for every question, then free ----
    from attribution.pipeline import Retriever
    print("Stage 1/2 — retrieval (both conditions)")
    retriever = Retriever(qdrant_path=args.qdrant_path, collection=args.collection,
                          bm25_path=args.bm25_path, use_hybrid=True, use_rerank=True)
    passages = {}  # (q_idx, cond) -> [passage dicts]
    for q in questions:
        for cond, flags in CONDITIONS.items():
            if f"{q['q_idx']}:{cond}" in done:
                continue
            passages[(q["q_idx"], cond)] = retriever.retrieve(
                q["question"], candidate_pool=args.candidate_pool, top_k=args.top_k, **flags)
    retriever.close()
    del retriever
    free_gpu()

    # ---- Stage 2: generate + CTI + LOO per (question, condition) ----
    from attribution.mirage import MirageAttributor
    print("Stage 2/2 — generate + CTI + LOO")
    attr = MirageAttributor(model=args.model, dtype=args.dtype)

    n_written = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for q in questions:
            for cond in CONDITIONS:
                key = f"{q['q_idx']}:{cond}"
                if key in done:
                    continue
                ps = passages.get((q["q_idx"], cond), [])
                if not ps:  # P2: don't persist an empty-context record (poisons resume)
                    print(f"  WARN q{q['q_idx']}/{cond}: no passages retrieved, skipping")
                    continue
                # generate + per-claim CTI (cci OFF → forward-only)
                res = attr.attribute(q["question"], ps,
                                     max_new_tokens=args.max_new_tokens, cci_per_sentence=False)
                claims = [{"text": s.text, "cti_mean": round(s.cti_mean, 4),
                           "cti_max": round(s.cti_max, 4)} for s in res.spans]
                answer_cti = (sum(c["cti_mean"] for c in claims) / len(claims)) if claims else 0.0
                answer_empty = not res.answer.strip()
                # crude refusal flag so refusal-driven CTI collapses can be excluded
                low = res.answer.lower()
                refusal = any(p in low for p in (
                    "cannot answer", "can't answer", "cannot be answered",
                    "not in the", "no information", "unable to answer",
                    "does not provide", "doesn't provide", "not provided in"))

                # gold position in THIS condition's retrieved list (runtime, not the
                # offline contrast-set rank — they can drift; record both downstream).
                gold_idx = next((i for i, p in enumerate(ps)
                                 if p.get("chunk_id") == q["gold_chunk_id"]), None)

                if answer_empty:  # P3: degenerate generation — don't fake a 0 LOO effect
                    full_lp, loo, gold_drop = None, [], None
                else:
                    # LOO via ATTENTION-MASKING: hide each passage's tokens from
                    # attention with positions + length held fixed, so the drop
                    # isolates content (not the prompt-length/position shift that
                    # text-blanking leaked). >0 = passage supported the answer.
                    full_lp, raw_drops = attr.loo_drops(q["question"], ps, res.answer)
                    loo = [round(d, 4) if d is not None else None for d in raw_drops]
                    full_lp = round(full_lp, 4)
                    gold_drop = (loo[gold_idx] if (gold_idx is not None
                                                   and loo[gold_idx] is not None) else None)

                rec = {
                    "q_idx": q["q_idx"], "contrast_class": q["contrast_class"],
                    "condition": cond, "question": q["question"],
                    "gold_chunk_id": q["gold_chunk_id"],
                    "gold_in_context": gold_idx is not None,
                    "gold_rank": (gold_idx + 1) if gold_idx is not None else None,
                    "gold_loo_drop": gold_drop,
                    "n_passages": len(ps),
                    "answer": res.answer,
                    "answer_empty": answer_empty,
                    "refusal": refusal,
                    "answer_cti_mean": round(answer_cti, 4),
                    "claims": claims,
                    "full_logprob": full_lp,
                    "loo_drops": loo,
                    "passage_chunk_ids": [p.get("chunk_id") for p in ps],
                    "model": args.model,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_written += 1
                rate = n_written / (time.time() - t0 + 1e-9)
                print(f"  [{n_written}] q{q['q_idx']} {q['contrast_class']}/{cond}: "
                      f"CTI={answer_cti:.2f} gold_in_ctx={gold_idx is not None} "
                      f"gold_loo={rec['gold_loo_drop']} ({rate*60:.1f}/min)")

    print(f"\nDone — wrote {n_written} records → {out_path}")
    print("Next: analyse with `attribution.reliance_analysis` (CTI shift × class "
          "interaction, effect sizes, gold-LOO comparison).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
