"""
Generate doc2query-style evaluation questions per chunk using a local OLMo.

Reads a stratified chunk sample from data/chunks.jsonl, prompts the model
for N specific factual questions per chunk, writes results to
data/synthetic_questions.jsonl.

Usage
-----
Smoke test (10 chunks):
    .venv311/Scripts/python.exe -m retrieval.generate_questions
        --n-chunks 10 --model allenai/OLMo-2-1124-7B-Instruct

Full prototype (500 chunks held out from doc2query indexing):
    .venv311/Scripts/python.exe -m retrieval.generate_questions
        --n-chunks 500 --n-questions 5 --temperature 0.7
        --out data/synthetic_questions_eval.jsonl

Output schema (JSONL, one record per question):
    {chunk_id, slug, page, q_idx, question, model, temperature, prompt_seed}

Model selection
---------------
- **Default (4080 prototype):** `allenai/Olmo-3-7B-Instruct` — fits in
  16 GB VRAM, fast iteration, throwaway runs to validate the pipeline.
- **Production (Habrok, final eval set + doc2query at full scale):**
  `allenai/Olmo-3.1-32B-Instruct` — pass via `--model`. Same model must
  be used for the end-to-end RAG generator side, so doc2query teacher
  capability does not confound retrieval ablations.

The prototype questions are engineering validation only — they should be
discarded and regenerated with the production model before any number
goes into the thesis.

Other notes
-----------
- Stratified sampling: max 1 chunk per document, so eval coverage is
  spread across the corpus rather than clustered in a few large docs.
- Stores the random-sampling seed in each record so the held-out eval
  set can be reconstructed exactly later.
- Difficulty scoring is intentionally NOT done here — run the questions
  against the retrieval index post-hoc and use rank-of-gold as measured
  difficulty.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
DEFAULT_OUT = DATA_DIR / "synthetic_questions.jsonl"

# Default is the 4080 prototype model. For thesis-grade runs override
# with --model allenai/Olmo-3.1-32B-Instruct (Habrok-only).
DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct"

PROMPT_TEMPLATE = """You are generating evaluation questions for a retrieval system over UN disaster risk reduction documents.

Read the passage below and write exactly {n} factual, SPECIFIC questions that can be answered by reading this passage. Each question must reference distinctive concepts, named entities, frameworks, places, dates, or numbers from the passage. Do NOT ask generic questions that any disaster-related document could answer.

Passage from "{title}", page {page}:
---
{text}
---

Output exactly {n} questions, one per line. No numbering, no commentary, no preamble. Just the questions."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Sentence-boundary heuristic: a real prose chunk has many `[lowercase]. [Capital]`
# transitions. TOC pages, section-header lists, and tabular fragments have very
# few — they're titles, codes, and numbers, not running text.
_SENTENCE_BOUNDARY = re.compile(r"[a-z]\.\s+[A-Z]")


def looks_like_prose(text: str, min_sentences: int = 3) -> bool:
    """Return True if the text reads like real prose, not a TOC / heading list.

    Counts inferred sentence boundaries (`[lower]. [Upper]`). Structural pages
    have <3; running text comfortably exceeds that.
    """
    return len(_SENTENCE_BOUNDARY.findall(text)) >= min_sentences


# documents.json tags language at the doc level, but English UN documents
# commonly embed non-English appendices (translated resolutions, country lists
# in Spanish/French, multi-language conference proceedings). Detect at chunk
# time so we don't ask OLMo to write English questions for a Spanish passage.
def is_english(text: str) -> bool:
    """Return True iff langdetect labels this chunk English.

    `langdetect` is imported lazily so callers that pass --no-language-filter
    do not need the dependency installed.
    """
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    try:
        # 1500-char window keeps detection stable on long chunks.
        return detect(text[:1500]) == "en"
    except LangDetectException:
        return False


# Acknowledgements / references / annex / ToC / worksheet pages contain prose
# and survive both prose and language filters, but the questions OLMo generates
# from them are about *who wrote the report* or *what should you fill in*, not
# about disaster-risk-reduction content — useless as eval signal even when
# factually correct.
_NON_CONTENT_HEADING = re.compile(
    r"\b(acknowledge?ments?|references|bibliography|appendix|annex(es)?"
    r"|table of contents|list of (figures|tables|abbreviations|acronyms)"
    r"|contributors|about the authors?|abbreviations and acronyms"
    r"|worksheet|checklist|self.?assessment|application form|template)\b",
    re.IGNORECASE,
)
_CREDIT_LABEL = re.compile(
    r"\b(coordinators?|authors?|reviewers?|editors?|contributors?"
    r"|produced (with|by)|published (in|by)|copyright|all rights reserved"
    r"|under the supervision of|supported by|funded by)\s*[:\-]",
    re.IGNORECASE,
)
# Bibliography / references list signature: many "(YYYY)" date-in-parens.
# Real prose has 0–2 such patterns; references lists routinely have 5–20.
_BIB_YEAR_PAREN = re.compile(r"\b(?:19|20)\d{2}\)")


def looks_like_content_page(
    text: str,
    max_credit_density: float = 0.012,
    max_bib_year_count: int = 5,
) -> bool:
    """Reject acknowledgements / references / credits / annex / worksheet pages.

    Three cheap checks:
    1. The first ~300 chars must not begin with a non-content section heading
       (Acknowledgements, References, Annex, Worksheet, Checklist, etc.).
    2. The density of "Coordinator:", "Author:", "Published by:" credit
       labels stays low — content pages have ~0; credits pages have several.
    3. The chunk has fewer than `max_bib_year_count` "(YYYY)" patterns —
       references lists mid-page (where the heading sits on a previous page
       and is missed by check 1) are caught by their bibliographic year-in-
       parens density.
    """
    head = text[:300]
    if _NON_CONTENT_HEADING.search(head):
        return False
    word_count = len(text.split())
    if word_count == 0:
        return False
    if len(_CREDIT_LABEL.findall(text)) / word_count > max_credit_density:
        return False
    if len(_BIB_YEAR_PAREN.findall(text)) > max_bib_year_count:
        return False
    return True


def load_chunks(
    path: Path,
    min_tokens: int,
    require_prose: bool = True,
    min_sentences: int = 3,
    require_english: bool = True,
    require_content_page: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Load chunks, dropping under-token, non-prose, non-English, and
    acknowledgements / references / annex pages.

    Filters are applied cheapest-first so the slow English check runs only
    on chunks that already passed the structural / content gates.

    Returns (kept_chunks, drop_counts) where drop_counts has keys
    {"short", "non_content", "non_prose", "non_english"}.
    """
    kept: list[dict] = []
    drops = {"short": 0, "non_content": 0, "non_prose": 0, "non_english": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["token_count"] < min_tokens:
                drops["short"] += 1
                continue
            if require_content_page and not looks_like_content_page(r["text"]):
                drops["non_content"] += 1
                continue
            if require_prose and not looks_like_prose(r["text"], min_sentences):
                drops["non_prose"] += 1
                continue
            if require_english and not is_english(r["text"]):
                drops["non_english"] += 1
                continue
            kept.append(r)
    return kept, drops


def load_docs(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {d["slug"]: d for d in json.load(f)}


def stratified_sample(
    chunks: list[dict], docs: dict, n: int, seed: int
) -> list[dict]:
    """Max 1 chunk per document. Sample chunks until we hit n; if we run out
    of unique-doc slots, skip — we don't double-pick from the same doc."""
    rng = random.Random(seed)
    by_slug: dict[str, list[dict]] = {}
    for c in chunks:
        if c["slug"] in docs:
            by_slug.setdefault(c["slug"], []).append(c)
    slugs = list(by_slug.keys())
    rng.shuffle(slugs)
    out = []
    for slug in slugs:
        out.append(rng.choice(by_slug[slug]))
        if len(out) >= n:
            break
    return out


# Boilerplate phrases that announce "I'm reading a document" — real user
# queries don't talk about "the passage" / "the document". Strip these so the
# eval distribution is closer to actual retrieval traffic.
_DOC_REF = r"(?:the|this)(?:\s+\w+){0,4}?\s+(?:passage|document|text|chunk|excerpt|report)"
_NOISE_PATTERNS = [
    rf"\baccording to {_DOC_REF}[,\s]*",
    rf"\bin {_DOC_REF}[,\s]*",
    rf"\bas (?:mentioned|described|stated|noted|indicated) in {_DOC_REF}[,\s]*",
    rf"\b{_DOC_REF}\s+(?:says|states|mentions|notes|describes|reports|indicates)\s+(?:that\s+)?",
]
_NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


def clean_question(q: str) -> str:
    """Strip "according to the passage" / "in the document" boilerplate.

    Repairs leading capitalization and double spaces after stripping. Returns
    the cleaned question with a single trailing '?'.
    """
    cleaned = q
    for rx in _NOISE_REGEXES:
        cleaned = rx.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([?.!,;:])", r"\1", cleaned).strip(" ,;:")
    if not cleaned:
        return ""
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned


def parse_questions(raw: str, max_n: int) -> list[str]:
    """Robustly extract questions from freeform model output."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip common bullet prefixes
        for prefix in ("* ", "- ", "• ", "→ "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        # strip "1." / "1)" numbering
        if len(line) > 2 and line[0].isdigit():
            for sep in (". ", ") ", " - ", ": "):
                if sep in line[:5]:
                    line = line.split(sep, 1)[1].strip()
                    break
        if "?" not in line:
            continue
        q = line.split("?", 1)[0].strip() + "?"
        q = clean_question(q)
        if q and 10 <= len(q) <= 300 and q not in out:
            out.append(q)
        if len(out) >= max_n:
            break
    return out


def build_prompt(chunk: dict, doc: dict, n_questions: int) -> str:
    return PROMPT_TEMPLATE.format(
        n=n_questions,
        title=(doc.get("title") or "(untitled)").strip(),
        page=chunk.get("page", "?"),
        text=chunk["text"].strip(),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--n-chunks", type=int, default=500)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--min-tokens", type=int, default=100,
                    help="skip chunks with fewer tokens (cover/TOC fragments)")
    ap.add_argument("--min-sentences", type=int, default=3,
                    help="prose filter: require N sentence-boundary transitions"
                         " (`[lower]. [Upper]`); structural pages have <3")
    ap.add_argument("--no-prose-filter", action="store_true",
                    help="disable the prose / TOC-page filter")
    ap.add_argument("--no-language-filter", action="store_true",
                    help="disable English-only filter (langdetect)")
    ap.add_argument("--no-content-filter", action="store_true",
                    help="disable acknowledgements / references / annex filter")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, help="stop after N chunks (smoke test)")
    args = ap.parse_args()

    docs = load_docs(DOCUMENTS_PATH)
    log.info(f"Loaded {len(docs):,} doc entries")

    chunks_all, drops = load_chunks(
        CHUNKS_PATH,
        args.min_tokens,
        require_prose=not args.no_prose_filter,
        min_sentences=args.min_sentences,
        require_english=not args.no_language_filter,
        require_content_page=not args.no_content_filter,
    )
    enabled = []
    if not args.no_content_filter:
        enabled.append("content-page")
    if not args.no_prose_filter:
        enabled.append("prose")
    if not args.no_language_filter:
        enabled.append("English")
    filt_label = ", ".join(enabled) if enabled else "no filters"
    log.info(f"Loaded {len(chunks_all):,} chunks "
             f"(≥{args.min_tokens} tokens; filters: {filt_label})")
    log.info(f"  dropped <{args.min_tokens} tokens         : {drops['short']:,}")
    if not args.no_content_filter:
        log.info(f"  dropped non-content (credits/refs/etc) : {drops['non_content']:,}")
    if not args.no_prose_filter:
        log.info(f"  dropped non-prose                      : {drops['non_prose']:,}")
    if not args.no_language_filter:
        log.info(f"  dropped non-English                    : {drops['non_english']:,}")

    sample = stratified_sample(chunks_all, docs, args.n_chunks, args.seed)
    if args.limit:
        sample = sample[: args.limit]
    log.info(f"Sampled {len(sample):,} chunks (1 per doc, seed={args.seed})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cpu":
        log.warning("CUDA not available — generation will be very slow.")

    dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    log.info(f"Loading {args.model} (dtype={args.dtype})…")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    parsed_short = 0
    t0 = time.time()
    last_log = t0

    with open(out_path, "w", encoding="utf-8") as fout:
        for ci, chunk in enumerate(sample, 1):
            doc = docs[chunk["slug"]]
            user_prompt = build_prompt(chunk, doc, args.n_questions)

            # Use chat template if the tokenizer supports it (instruct models).
            messages = [{"role": "user", "content": user_prompt}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = user_prompt

            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=4096
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            questions = parse_questions(generated, args.n_questions)
            if len(questions) < args.n_questions:
                parsed_short += 1

            for q_idx, q in enumerate(questions):
                fout.write(json.dumps({
                    "chunk_id": chunk["chunk_id"],
                    "slug": chunk["slug"],
                    "page": chunk["page"],
                    "q_idx": q_idx,
                    "question": q,
                    "model": args.model,
                    "temperature": args.temperature,
                    "prompt_seed": args.seed,
                }, ensure_ascii=False) + "\n")
                written += 1

            now = time.time()
            if now - last_log > 10 or ci == len(sample):
                elapsed = now - t0
                rate = ci / elapsed if elapsed else 0
                remaining = len(sample) - ci
                eta_min = (remaining / rate / 60) if rate else 0
                log.info(
                    f"  {ci}/{len(sample)} chunks  "
                    f"{written} questions  rate={rate:.2f} chunks/s  "
                    f"eta={eta_min:.1f}min  short_parses={parsed_short}"
                )
                last_log = now

    log.info(
        f"Done — {written} questions across {len(sample)} chunks. "
        f"Short-parse chunks: {parsed_short} (got fewer than {args.n_questions})."
    )
    log.info(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
