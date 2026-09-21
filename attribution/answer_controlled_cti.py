"""
Answer-controlled CTI — the teacher-forced 2x2 sensitivity check for RQ3.

Usage
-----
    # free pre-flight, no GPU, no model
    .venv311/Scripts/python.exe -m attribution.answer_controlled_cti --preflight-only

    # local 7B smoke (slice logits + fewer passages so it fits a 16 GB card)
    .venv311/Scripts/python.exe -m attribution.answer_controlled_cti \
        --limit 4 --slice-logits --top-k 4 --out data/ac_smoke.jsonl

    # production (Habrok, 32B) — see scripts/habrok/answer_controlled.slurm
    python -m attribution.answer_controlled_cti --model allenai/Olmo-3.1-32B-Instruct \
        --device-map auto --shard $I --num-shards 8 \
        --out data/answer_controlled_cells.shard$I.jsonl --resume
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Answer -> context column keys. "d" = dense, "h" = hybrid_rerank.
COND = {"d": "dense", "h": "hybrid_rerank"}

# Refusal wording the production `refusal` flag misses. Recorded per record (NOT
# used to exclude — changing the rule would break comparability with the
# published n=696) so a sensitivity re-run needs no re-scoring. These survive
# asymmetrically: ~35 in rescued/dense vs ~7 in rescued/hybrid, i.e. they depress
# exactly the arm that drives the interaction.
SOFT_REFUSAL = ("cannot be determined", "not specified", "i cannot",
                "not mentioned", "insufficient information", "not addressed")

_TAG = re.compile(r"\[P(\d+)\]")
_WORD = re.compile(r"\w+", re.UNICODE)


def read_jsonl(path: Path):
    """Line-iterating JSONL reader.

    Deliberately NOT Path.read_text().splitlines(): data/chunks.jsonl contains
    707 U+2028, 116 U+2029 and 204 U+0085 characters inside JSON strings, and
    str.splitlines() breaks on all three — 1,027 spurious splits and a
    JSONDecodeError. Iterating the file object splits on \\n only, matching
    retrieval/bm25.py and retrieval/embed.py.
    """
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def usable(rec: dict) -> bool:
    """The published eligibility rule, applied pairwise by the caller.

    Mirrors attribution.reliance_analysis.usable exactly; kept as a local copy
    only so this module does not import numpy (and therefore stays importable
    for --preflight-only on a machine with no ML stack).
    """
    return rec is not None and not rec.get("answer_empty") and not rec.get("refusal")


def ngrams(text: str, n: int) -> set:
    w = [t.lower() for t in _WORD.findall(text or "")]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()


def coverage(answer: str, context_text: str, n: int = 3) -> float:
    """Fraction of the answer's word n-grams that also occur in the context.

    The copy-circularity covariate: an answer that quotes its own context scores
    high CTI partly because the context makes its own wording predictable.
    """
    a = ngrams(answer, n)
    return round(len(a & ngrams(context_text, n)) / len(a), 4) if a else 0.0


def citation_stats(answer: str, ids_own: list, ids_other: list) -> dict:
    """[P#] tags in the answer and how many survive a context swap intact."""
    tags = [int(t) for t in _TAG.findall(answer or "")]
    safe = sum(1 for t in tags
               if 1 <= t <= len(ids_own) and 1 <= t <= len(ids_other)
               and ids_own[t - 1] == ids_other[t - 1])
    return {"n_tags": len(tags), "n_swap_safe": safe,
            "fully_swap_safe": bool(tags) and safe == len(tags)}


def load_pairs(records: Path) -> dict:
    """q_idx -> {condition: record}."""
    by_q: dict = {}
    for r in read_jsonl(records):
        by_q.setdefault(r["q_idx"], {})[r["condition"]] = r
    return by_q


def eligible_questions(by_q: dict, arm: str) -> list:
    """Sorted q_idx list for the requested arm (deterministic for sharding)."""
    out = []
    for q, pair in by_q.items():
        d, h = pair.get("dense"), pair.get("hybrid_rerank")
        if d is None or h is None:
            continue
        if arm == "main":
            if usable(d) and usable(h):
                out.append(q)
        else:  # refusal arm: dense refused, hybrid answered
            if d.get("refusal") and usable(h) and (d.get("answer") or "").strip():
                out.append(q)
    return sorted(out)


def preflight(by_q: dict, chunks: dict, qs: list, arm: str) -> int:
    """Free checks — no GPU, no model. Returns the number of problems found."""
    from attribution.mirage import split_sentences
    problems = 0
    n_claim_mismatch = n_missing_chunk = n_bad_len = 0
    for q in qs:
        for cond in COND.values():
            rec = by_q[q][cond]
            # The stored claims must be exactly what split_sentences produces,
            # or the recomputed aggregation cannot reproduce answer_cti_mean.
            stored = [c["text"] for c in rec.get("claims", [])]
            if stored and stored != split_sentences(rec.get("answer", "")):
                n_claim_mismatch += 1
            ids = rec.get("passage_chunk_ids") or []
            if len(ids) != rec.get("n_passages") or any(i is None for i in ids):
                n_bad_len += 1
            n_missing_chunk += sum(1 for i in ids if i not in chunks)
    for label, n in (("claim/split mismatches", n_claim_mismatch),
                     ("records with bad passage list", n_bad_len),
                     ("passage chunk_ids missing from corpus", n_missing_chunk)):
        print(f"  {label}: {n}")
        problems += n
    return problems


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--records", default=str(DATA_DIR / "reliance_records.jsonl"))
    ap.add_argument("--chunks", default=str(DATA_DIR / "chunks.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "answer_controlled_cells.jsonl"))
    ap.add_argument("--arm", choices=["main", "refusal"], default="main",
                    help="'main' = the 2x2 over both generated answers (N=696); "
                         "'refusal' = force the DENSE REFUSAL text under both "
                         "contexts on the pairs the main arm must exclude")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device-map", default=None,
                    help="'auto' shards the 32B across visible GPUs (Habrok)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="truncate each context to the first K passages. 0 = all "
                         "10 = the production context. ONLY for a local smoke: "
                         "any value >0 makes the diagonal incomparable.")
    ap.add_argument("--slice-logits", action="store_true",
                    help="ask the model for only the logit rows actually read "
                         "(identical numbers, ~60x less logit memory). Needed to "
                         "fit the 7B smoke on a 16 GB card; leave OFF for the "
                         "production run so it matches the recorded path exactly.")
    ap.add_argument("--limit", type=int, help="first N questions (smoke)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run the free consistency checks and exit (no model load)")
    args = ap.parse_args()

    records, chunks_path, out_path = Path(args.records), Path(args.chunks), Path(args.out)
    print(f"Loading records from {records}")
    by_q = load_pairs(records)
    qs_all = eligible_questions(by_q, args.arm)
    n_cls: dict = {}
    for q in qs_all:
        c = by_q[q]["dense"].get("contrast_class")
        n_cls[c] = n_cls.get(c, 0) + 1
    print(f"Arm '{args.arm}': {len(qs_all)} eligible questions "
          f"({', '.join(f'{k} {v}' for k, v in sorted(n_cls.items()))}) "
          f"out of {len(by_q)} paired")

    print(f"Loading chunk corpus from {chunks_path}")
    chunks = {c["chunk_id"]: c["text"] for c in read_jsonl(chunks_path)}
    print(f"  {len(chunks):,} chunks")

    print("Pre-flight (no GPU):")
    problems = preflight(by_q, chunks, qs_all, args.arm)
    if problems:
        print(f"  !! {problems} problem(s) found — fix before scoring", file=sys.stderr)
        if args.preflight_only:
            return 1
    else:
        print("  all clear")
    if args.preflight_only:
        return 0

    qs = qs_all[:args.limit] if args.limit else qs_all
    if args.num_shards > 1:
        qs = [q for n, q in enumerate(qs) if n % args.num_shards == args.shard]
        print(f"  shard {args.shard}/{args.num_shards}: {len(qs)} questions")

    done: set = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        for r in read_jsonl(out_path):
            done.add(r["q_idx"])
        print(f"Resuming — {len(done)} questions already scored")

    from attribution.mirage import MirageAttributor
    attr = MirageAttributor(model=args.model, dtype=args.dtype, device_map=args.device_map)

    n_written, t0 = 0, time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for q in qs:
            if q in done:
                continue
            pair = by_q[q]
            question = pair["dense"]["question"]
            ids = {k: list(pair[c]["passage_chunk_ids"]) for k, c in COND.items()}
            if args.top_k:
                ids = {k: v[:args.top_k] for k, v in ids.items()}
            # Rebuild each context from the corpus. _build_prompt_ids reads only
            # p["text"]; chunk_id is carried for provenance, never for the prompt.
            ctx = {k: [{"text": chunks[i], "chunk_id": i} for i in ids[k]] for k in COND}
            ctx_text = {k: "\n".join(p["text"] for p in ctx[k]) for k in COND}

            if args.arm == "main":
                answers = {k: pair[c]["answer"] for k, c in COND.items()}
            else:
                # Force the dense refusal under both contexts. One "answer",
                # two cells; keyed "r" so the schema stays uniform.
                answers = {"r": pair["dense"]["answer"]}

            # Which context each answer was actually generated under — the
            # "own" side of the own/foreign coverage and citation contrasts.
            # The refusal arm's single answer came from the dense condition.
            own_ctx = {"d": "d", "h": "h", "r": "d"}

            cells, ok = {}, True
            for a_key, answer_text in answers.items():
                lp_without = lp_fmt = None
                for c_key in COND:
                    res = attr.cti_fixed_answer(
                        question, ctx[c_key], answer_text,
                        lp_without=lp_without, lp_without_fmt=lp_fmt,
                        slice_logits=args.slice_logits)
                    # lp_without depends only on (question, answer) — reuse it
                    # for the second context column. 6 forwards/question, not 8.
                    lp_without, lp_fmt = res.pop("lp_without"), res["fmt"]
                    if res["fmt"] != "chat":
                        print(f"  WARN q{q} {a_key}|{c_key}: prompt_format="
                              f"{res['fmt']} (not chat) — cell not comparable")
                        ok = False
                    res["coverage_3gram"] = coverage(answer_text, ctx_text[c_key])
                    res["answer_of"] = a_key
                    res["context_of"] = c_key
                    cells[f"{a_key}|{c_key}"] = res
                del lp_without

            rec = {
                "q_idx": q,
                "contrast_class": pair["dense"].get("contrast_class"),
                "arm": args.arm,
                "gold_chunk_id": pair["dense"].get("gold_chunk_id"),
                "model": args.model,
                "all_chat_format": ok,
                "top_k": args.top_k or 10,
                "slice_logits": bool(args.slice_logits),
                "cells": cells,
                # --- validation + covariate payload -------------------------
                "recorded_cti": {c: pair[c].get("answer_cti_mean") for c in COND.values()},
                "recorded_full_logprob": {c: pair[c].get("full_logprob") for c in COND.values()},
                "gold_in_context": {c: pair[c].get("gold_in_context") for c in COND.values()},
                "gold_rank": {c: pair[c].get("gold_rank") for c in COND.values()},
                "identical_answers": (args.arm == "main"
                                      and pair["dense"]["answer"] == pair["hybrid_rerank"]["answer"]),
                "ctx_overlap": round(len(set(ids["d"]) & set(ids["h"])) / len(set(ids["d"]) | set(ids["h"])), 4),
                "citations": {
                    k: citation_stats(ans, ids[own_ctx[k]],
                                      ids["h" if own_ctx[k] == "d" else "d"])
                    for k, ans in answers.items()},
                "soft_refusal": {c: any(p in (pair[c].get("answer") or "").lower()
                                        for p in SOFT_REFUSAL) for c in COND.values()},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_written += 1
            rate = n_written / (time.time() - t0 + 1e-9)
            if args.arm == "main":
                diag = (f"dd={cells['d|d']['cti_mean']:.3f}(rec "
                        f"{rec['recorded_cti']['dense'] or 0.0:.3f}) "
                        f"hh={cells['h|h']['cti_mean']:.3f}(rec "
                        f"{rec['recorded_cti']['hybrid_rerank'] or 0.0:.3f})")
            else:
                diag = (f"r|d={cells['r|d']['signed_mean']:+.3f} "
                        f"r|h={cells['r|h']['signed_mean']:+.3f} (signed)")
            print(f"  [{n_written}] q{q} {rec['contrast_class']}: {diag} "
                  f"({rate * 60:.1f}/min)")

    print(f"\nDone — wrote {n_written} questions → {out_path}")
    print("Next: analyse with `attribution.answer_controlled_analysis` "
          "(diagonal validation gate first, then the decomposition).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
