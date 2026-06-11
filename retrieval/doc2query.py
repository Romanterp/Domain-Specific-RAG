"""
doc2query document expansion + held-out evaluation split.

Two modes:

PROTOTYPE (default) — 1-per-doc sample, every sampled chunk is an eval gold
chunk (1 question held out, the rest indexed). Quick, local. This is what
produced the 300-chunk Round 6b results.

FULL-CORPUS (--full-corpus) — augment *every* eligible chunk (removes the
gold-only-augmentation bias of the prototype), sharded for a SLURM array, with
the eval queries drawn from a separate 1-per-doc held-out sample that is never
indexed. This is the rigorous RQ2 run.

Circularity control (both modes): a chunk that supplies an eval query has that
one question HELD OUT (never indexed); the chunk is still augmented with its
other questions. So the eval query is never seen at index time.

Outputs
-------
Prototype:    data/doc2query_expansions.json , data/doc2query_eval.jsonl
Full-corpus:  data/doc2query_expansions_shard{i}.json , data/doc2query_eval_shard{i}.jsonl
              (then `--merge` → doc2query_expansions_full.json / doc2query_eval_full.jsonl)

Usage
-----
Prototype (local 7B, 300 chunks):
    .venv311/Scripts/python.exe -m retrieval.doc2query --n-chunks 300 --n-questions 6

Full-corpus (32B on Habrok):
    # ONCE, before submitting the array (CPU-only, ~3 min — applies the chunk
    # filters and persists the held-out eval chunk ids all shards will share):
    python -m retrieval.doc2query --make-holdout

    # then each SLURM array task runs one shard (2x A100):
    python -m retrieval.doc2query --full-corpus --shard $SLURM_ARRAY_TASK_ID \\
        --num-shards 24 --model allenai/Olmo-3.1-32B-Instruct --n-questions 6 --resume

Merge shards after the array finishes:
    python -m retrieval.doc2query --merge
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval.generate_questions import (
    DEFAULT_MODEL,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    load_chunks,
    load_docs,
    stratified_sample,
    build_prompt,
    parse_questions,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_EXPANSIONS = DATA_DIR / "doc2query_expansions.json"
DEFAULT_EVAL = DATA_DIR / "doc2query_eval.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def merge_shards() -> int:
    """Combine per-shard outputs into the final full-corpus files."""
    exp_files = sorted(DATA_DIR.glob("doc2query_expansions_shard*.json"))
    eval_files = sorted(DATA_DIR.glob("doc2query_eval_shard*.jsonl"))
    if not exp_files:
        log.error("No doc2query_expansions_shard*.json found to merge.")
        return 2
    expansions: dict[str, list[str]] = {}
    for f in exp_files:
        expansions.update(json.loads(f.read_text(encoding="utf-8")))
    out_exp = DATA_DIR / "doc2query_expansions_full.json"
    out_exp.write_text(json.dumps(expansions, ensure_ascii=False), encoding="utf-8")

    out_eval = DATA_DIR / "doc2query_eval_full.jsonl"
    n_eval = 0
    with open(out_eval, "w", encoding="utf-8") as fout:
        for f in eval_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fout.write(line + "\n")
                    n_eval += 1
    log.info(f"Merged {len(exp_files)} shards → {out_exp.name} "
             f"({len(expansions):,} chunks augmented), {out_eval.name} ({n_eval:,} eval queries)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--merge", action="store_true", help="merge shard outputs and exit")
    # mode
    ap.add_argument("--full-corpus", action="store_true",
                    help="augment ALL eligible chunks (rigorous); else 1-per-doc prototype")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (full-corpus)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (full-corpus)")
    ap.add_argument("--eval-chunks", type=int, default=1000,
                    help="held-out eval gold chunks (1/doc) for full-corpus mode")
    ap.add_argument("--make-holdout", action="store_true",
                    help="compute the full-corpus held-out eval sample, write it to "
                         "--holdout-file, and exit. Run ONCE before the array job; "
                         "every shard then reads the same file.")
    ap.add_argument("--holdout-file", default=str(DATA_DIR / "doc2query_holdout_ids.txt"),
                    help="persisted held-out eval chunk-id list shared by all "
                         "full-corpus shards (one chunk_id per line)")
    # prototype
    ap.add_argument("--n-chunks", type=int, default=300)
    ap.add_argument("--n-questions", type=int, default=6)
    ap.add_argument("--holdout", type=int, default=1,
                    help="questions per EVAL chunk reserved for eval (never indexed)")
    # chunk filters (identical to generate_questions)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--min-sentences", type=int, default=3)
    ap.add_argument("--no-prose-filter", action="store_true")
    ap.add_argument("--no-language-filter", action="store_true")
    ap.add_argument("--no-content-filter", action="store_true")
    # generation
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--limit", type=int, help="stop after N chunks (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1)
    ap.add_argument("--out-expansions", default=str(DEFAULT_EXPANSIONS))
    ap.add_argument("--out-eval", default=str(DEFAULT_EVAL))
    args = ap.parse_args()

    if args.merge:
        return merge_shards()

    if args.n_questions <= args.holdout:
        log.error(f"--n-questions ({args.n_questions}) must exceed --holdout ({args.holdout}).")
        return 2

    docs = load_docs(DOCUMENTS_PATH)
    log.info(f"Loaded {len(docs):,} doc entries")
    chunks_all, _ = load_chunks(
        CHUNKS_PATH, args.min_tokens,
        require_prose=not args.no_prose_filter, min_sentences=args.min_sentences,
        require_english=not args.no_language_filter,
        require_content_page=not args.no_content_filter,
    )
    log.info(f"Eligible chunks after filters: {len(chunks_all):,}")

    holdout_path = Path(args.holdout_file)

    if args.make_holdout:
        sample = stratified_sample(chunks_all, docs, args.eval_chunks, args.seed)
        ids = sorted(c["chunk_id"] for c in sample)
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        log.info(f"Wrote {len(ids):,} held-out eval chunk ids → {holdout_path}")
        log.info("Now submit the array job; every shard reads this file.")
        return 0

    # --- select working set + eval gold chunks per mode ---
    if args.full_corpus:
        # augment EVERY eligible chunk; shard by deterministic chunk_id order.
        chunks_sorted = sorted(chunks_all, key=lambda c: c["chunk_id"])
        per = math.ceil(len(chunks_sorted) / args.num_shards)
        working = chunks_sorted[args.shard * per:(args.shard + 1) * per]
        # All shards read ONE persisted holdout list — never recomputed per
        # shard, so identical eval splits are guaranteed even if a shard is
        # re-run later with different filter flags.
        if not holdout_path.exists():
            log.error(f"Holdout list not found: {holdout_path}")
            log.error("Run once first:  python -m retrieval.doc2query --make-holdout "
                      f"--eval-chunks {args.eval_chunks} --seed {args.seed}")
            return 2
        eval_ids = {ln.strip() for ln in
                    holdout_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
        exp_path = DATA_DIR / f"doc2query_expansions_shard{args.shard}.json"
        eval_path = DATA_DIR / f"doc2query_eval_shard{args.shard}.jsonl"
        log.info(f"FULL-CORPUS shard {args.shard}/{args.num_shards}: "
                 f"{len(working):,} chunks this shard, {len(eval_ids):,} held-out "
                 f"eval chunks (from {holdout_path.name})")
    else:
        # prototype: 1/doc sample, every sampled chunk is an eval gold chunk.
        working = stratified_sample(chunks_all, docs, args.n_chunks, args.seed)
        eval_ids = {c["chunk_id"] for c in working}
        exp_path = Path(args.out_expansions)
        eval_path = Path(args.out_eval)
        log.info(f"PROTOTYPE: {len(working):,} chunks (1/doc, all eval gold)")

    if args.limit:
        working = working[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cpu":
        log.warning("CUDA not available — generation will be very slow.")
    dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    log.info(f"Loading {args.model} (dtype={args.dtype}, device_map=auto)…")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded.")

    split_rng = random.Random(args.seed + 1)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: a chunk is done only if its expansion is persisted
    expansions: dict[str, list[str]] = {}
    done: set[str] = set()
    eval_mode = "w"
    if args.resume and exp_path.exists():
        try:
            expansions = json.loads(exp_path.read_text(encoding="utf-8"))
        except Exception:
            expansions = {}
        done = set(expansions.keys())
        eval_mode = "a" if eval_path.exists() else "w"
        log.info(f"Resuming — {len(done):,} chunks already done in this shard, skipping")

    def flush_expansions():
        exp_path.write_text(json.dumps(expansions, ensure_ascii=False), encoding="utf-8")

    n_eval = 0
    n_index_q = sum(len(v) for v in expansions.values())
    too_few = 0
    kept_since_flush = 0
    t0 = time.time()
    last_log = t0

    with open(eval_path, eval_mode, encoding="utf-8") as feval:
        for ci, chunk in enumerate(working, 1):
            cid = chunk["chunk_id"]
            if cid in done:
                continue
            doc = docs[chunk["slug"]]
            user_prompt = build_prompt(chunk, doc, args.n_questions)
            messages = [{"role": "user", "content": user_prompt}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = user_prompt
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            questions = parse_questions(generated, args.n_questions)

            is_eval = cid in eval_ids
            need = args.holdout + 1 if is_eval else 1
            if len(questions) < need:
                too_few += 1
                continue

            if is_eval:
                split_rng.shuffle(questions)
                held = questions[: args.holdout]
                keep = questions[args.holdout:]
                for q in held:
                    feval.write(json.dumps({
                        "question": q, "chunk_id": cid, "slug": chunk["slug"],
                        "page": chunk["page"], "model": args.model,
                    }, ensure_ascii=False) + "\n")
                    n_eval += 1
                feval.flush()
            else:
                # Index the same number of questions as an eval chunk gets
                # (n_questions − holdout), so gold and distractor chunks have
                # identical augmentation depth — no asymmetry in the A/B.
                keep = questions[: args.n_questions - args.holdout]

            expansions[cid] = keep
            n_index_q += len(keep)
            kept_since_flush += 1
            if kept_since_flush >= args.flush_every:
                flush_expansions()
                kept_since_flush = 0

            now = time.time()
            if now - last_log > 15 or ci == len(working):
                rate = ci / (now - t0) if now > t0 else 0
                eta = (len(working) - ci) / rate / 60 if rate else 0
                log.info(f"  {ci:,}/{len(working):,}  rate={rate:.2f} chunk/s  "
                         f"eta={eta:.1f}min  (eval={n_eval}, index_q={n_index_q})")
                last_log = now

    flush_expansions()
    log.info("=" * 60)
    log.info(f"Chunks augmented      : {len(expansions):,}")
    log.info(f"Index questions       : {n_index_q:,}")
    log.info(f"Held-out eval queries : {n_eval:,}")
    log.info(f"Skipped (too few qs)  : {too_few:,}")
    log.info(f"Expansions → {exp_path}")
    log.info(f"Eval queries → {eval_path}")
    if args.full_corpus:
        log.info("When all shards finish: `python -m retrieval.doc2query --merge`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
