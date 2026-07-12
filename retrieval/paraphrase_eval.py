"""
Paraphrase challenge — reword the held-out eval queries to break the
synthetic lexical-overlap bias, then re-run the doc2query BM25 A/B on them.

Motivation
----------
The synthetic eval queries (data/doc2query_eval_full.jsonl) are LLM-generated
*from the gold chunk*, so they reuse the chunk's surface vocabulary. BM25
already matches them on that overlap, which leaves doc2query almost no
vocabulary gap to bridge — and is why the full-corpus doc2query lift nets ~0
on this eval (EXPERIMENTS.md Round 6b / full-corpus). This script paraphrases
each query (same meaning + same answer, different words) so the gold chunk's
literal terms no longer leak into the query. If doc2query's recall lift
reappears on the paraphrased set, that confirms the mechanism: doc2query helps
exactly when the query and the document use different words.

The gold labels transfer for free — paraphrasing changes only the question
text, not which chunk answers it — so the output is a drop-in for
eval_doc2query.py (it reads `question` + `chunk_id` per line).

Output schema (JSONL, one record per query; mirrors the input plus the original)
    {question, chunk_id, slug, page, model, original_question}

Usage
-----
Smoke test (10 queries, local 7B):
    .venv311/Scripts/python.exe -m retrieval.paraphrase_eval --limit 10

Full run (1000 queries):
    .venv311/Scripts/python.exe -m retrieval.paraphrase_eval

Then re-run the existing sparse A/B on the paraphrased queries:
    .venv311/Scripts/python.exe -m retrieval.eval_doc2query \\
        --questions data/doc2query_eval_full_paraphrased.jsonl \\
        --baseline data/bm25_index.pkl \\
        --doc2query data/bm25_doc2query_full.pkl \\
        --out-summary data/eval_doc2query_paraphrased_summary.md
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval.generate_questions import DEFAULT_MODEL, clean_question

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_IN = DATA_DIR / "doc2query_eval_full.jsonl"
DEFAULT_OUT = DATA_DIR / "doc2query_eval_full_paraphrased.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Reword for the SAME answer using DIFFERENT surface words. Proper nouns,
# framework names, places, organisations and numbers are kept verbatim — they
# are part of the answer, not stylistic vocabulary, and a real user asking this
# question would use them too. The point is to strip the incidental term overlap
# (verbs, connectors, generic nouns) that inflates BM25 on the synthetic set.
PROMPT_TEMPLATE = """Rewrite the following question so it asks for exactly the same answer but uses different wording and sentence structure. Replace ordinary words with synonyms and change the phrasing. Keep all proper names, framework names, place names, organisations, dates and numbers unchanged — they are essential to the answer. Do not add or remove any information, and do not answer the question.

Question: {question}

Output only the rewritten question on a single line, with no preamble, quotes, or commentary."""

# Strip leading labels the model sometimes prepends despite the instruction.
_LABEL_RE = re.compile(
    r"^\s*(?:rewritten|rephrased|paraphrased|reworded|new)?\s*question\s*[:\-]\s*",
    re.IGNORECASE,
)


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def parse_paraphrase(raw: str) -> str:
    """Extract a single rewritten question from freeform model output.

    Takes the first line that contains a '?', stripping bullets, surrounding
    quotes, and "Rewritten question:"-style labels. Falls back to the first
    non-empty line. Returns "" if nothing usable was produced.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    pick = next((ln for ln in lines if "?" in ln), lines[0])
    # strip bullets / numbering
    for prefix in ("* ", "- ", "• ", "→ "):
        if pick.startswith(prefix):
            pick = pick[len(prefix):].strip()
    pick = re.sub(r"^\d+[.)]\s*", "", pick)
    pick = _LABEL_RE.sub("", pick)
    pick = pick.strip().strip('"').strip("'").strip()
    # cut at the first '?' so trailing model chatter is dropped
    if "?" in pick:
        pick = pick.split("?", 1)[0].strip() + "?"
    return clean_question(pick)


def load_done(path: Path) -> set[str]:
    """Resume key set: chunk_id + original question already written."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(f"{r.get('chunk_id')}||{r.get('original_question')}")
    return done


def main() -> int:
    try:  # Windows consoles default to cp1252 and choke on logged questions
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN),
                    help="held-out eval queries to paraphrase (JSONL)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="slightly high to encourage lexical variation")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--limit", type=int, help="stop after N queries (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip queries already present in --out and append")
    args = ap.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out)
    if not in_path.exists():
        log.error(f"Input not found: {in_path}")
        return 2

    queries = [json.loads(ln) for ln in
               in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    log.info(f"Loaded {len(queries):,} eval queries from {in_path.name}")
    if args.limit:
        queries = queries[: args.limit]
        log.info(f"Limit: first {len(queries):,} queries")

    done = load_done(out_path) if args.resume else set()
    if done:
        log.info(f"Resuming — {len(done):,} already paraphrased, will skip")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cpu":
        log.warning("CUDA not available — paraphrasing will be very slow.")
    dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    log.info(f"Loading {args.model} (dtype={args.dtype})…")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device, trust_remote_code=True)
    model.eval()
    log.info("Model loaded.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and out_path.exists()) else "w"

    written = empty = skipped = 0
    t0 = time.time()
    last_log = t0
    with open(out_path, mode, encoding="utf-8") as fout:
        for qi, q in enumerate(queries, 1):
            original = q["question"]
            if f"{q['chunk_id']}||{original}" in done:
                skipped += 1
                continue

            messages = [{"role": "user", "content": build_prompt(original)}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = build_prompt(original)
            inputs = tokenizer(input_text, return_tensors="pt",
                               truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id)
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            paraphrased = parse_paraphrase(generated)

            # Fall back to the original if the model produced nothing usable, so
            # the eval set stays aligned 1:1 with the gold labels (a dropped row
            # would silently shrink the A/B set).
            if not paraphrased:
                empty += 1
                paraphrased = original

            fout.write(json.dumps({
                "question": paraphrased,
                "chunk_id": q["chunk_id"],
                "slug": q.get("slug"),
                "page": q.get("page"),
                "model": args.model,
                "original_question": original,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            now = time.time()
            if now - last_log > 10 or qi == len(queries):
                rate = qi / (now - t0) if now > t0 else 0
                eta = (len(queries) - qi) / rate / 60 if rate else 0
                log.info(f"  {qi:,}/{len(queries):,}  rate={rate:.2f} q/s  "
                         f"eta={eta:.1f}min  (written={written}, "
                         f"fellback={empty}, skipped={skipped})")
                last_log = now

    log.info("=" * 60)
    log.info(f"Paraphrased written : {written:,}")
    log.info(f"Fell back to original (unparseable): {empty:,}")
    if skipped:
        log.info(f"Skipped (resume)    : {skipped:,}")
    log.info(f"Output → {out_path}")
    log.info("Next: re-run eval_doc2query.py with --questions "
             f"{out_path.name} (see this file's docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
