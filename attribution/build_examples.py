"""
Build cached provenance examples for the Streamlit viewer.

Runs the full pipeline over a list of questions and writes one JSON per question
to data/attribution/. Loads models in three stages (retriever → generator →
grounder), freeing the GPU between stages, so OLMo and the retrieval models are
never there at the same time

Usage
-----
Local prototype (7B, the hand-written naturalistic queries):
    .venv311/Scripts/python.exe -m attribution.build_examples \\
        --questions example_q.txt --limit 5

Habrok production (32B):
    python -m attribution.build_examples \\
        --questions example_q.txt --model allenai/Olmo-3.1-32B-Instruct --dtype bf16
"""

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean

from attribution.pipeline import (
    Retriever, Generator, Grounder,
    split_into_spans, DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = DATA_DIR / "attribution"


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_questions(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def slugify(text: str, idx: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return f"{idx:02d}-{s[:50]}" if s else f"{idx:02d}-q"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--questions", default=str(Path(__file__).resolve().parent.parent / "example_q.txt"))
    ap.add_argument("--collection", default="theisus_none")
    ap.add_argument("--bm25-path", default=str(DATA_DIR / "bm25_index.pkl"))
    ap.add_argument("--qdrant-path", default=str(DATA_DIR / "qdrant"))
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--candidate-pool", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        log.error("No questions loaded.")
        return 2
    log.info(f"{len(questions)} questions")

    pipeline_tag = "+".join(
        (["hybrid"] if not args.no_hybrid else [])
        + ["dense"] + (["rerank"] if not args.no_rerank else [])
    ) if (args.no_hybrid or args.no_rerank) else "hybrid+rerank"

    # ---- Stage 1: retrieve ----
    log.info("Stage 1/3 — retrieval")
    retriever = Retriever(
        qdrant_path=args.qdrant_path, collection=args.collection,
        bm25_path=args.bm25_path, use_hybrid=not args.no_hybrid,
        use_rerank=not args.no_rerank,
    )
    passages_by_q = []
    for q in questions:
        passages_by_q.append(retriever.retrieve(q, candidate_pool=args.candidate_pool, top_k=args.top_k))
        log.info(f"  retrieved {len(passages_by_q[-1])} passages for: {q[:60]}")
    retriever.close()
    del retriever
    free_gpu()

    # ---- Stage 2: generate ----
    log.info("Stage 2/3 — RAG generation")
    generator = Generator(model=args.model, dtype=args.dtype)
    answers = []
    for q, passages in zip(questions, passages_by_q):
        ans = generator.answer(q, passages, max_new_tokens=args.max_new_tokens)
        answers.append(ans)
        log.info(f"  answered ({len(ans)} chars): {q[:60]}")
    del generator
    free_gpu()

    # ---- Stage 3: ground ----
    log.info("Stage 3/3 — grounding")
    grounder = Grounder()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, (q, passages, answer) in enumerate(zip(questions, passages_by_q, answers), 1):
        spans = split_into_spans(answer)
        scored = grounder.ground(spans, passages)
        support = mean(s["support_score"] for s in scored) if scored else 0.0
        record = {
            "id": slugify(q, idx),
            "question": q,
            "pipeline": pipeline_tag,
            "model": args.model,
            "generated_at": __import__("datetime").date.today().isoformat(),
            "is_mock": False,
            "support_score": round(support, 3),
            "answer_plain": answer,
            "spans": scored,
            "passages": passages,
        }
        out_path = out_dir / f"{record['id']}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
        log.info(f"  wrote {out_path.name}  (support={support:.2f}, spans={len(scored)})")
    del grounder
    free_gpu()

    log.info(f"Done — {written} example(s) in {out_dir}")
    log.info("View with:  streamlit run attribution/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
