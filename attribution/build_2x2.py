"""
build_2x2.py — the extrinsic (OLMoTrace) pass that turns the intrinsic reliance
records into the RQ3 intrinsic × extrinsic 2×2.

Design option (a): keep the Habrok reliance run FORWARD-ONLY and pure
(reliance_experiment.py → reliance_records.jsonl: per-(question, condition)
answers with per-claim CTI + gold LOO), then attach the extrinsic lens HERE, as a
separate disk-cached pass over those SAME answers. Running OLMoTrace on the
reliance run's own greedy decode — not a re-generation — is what keeps both
lenses describing the identical text (the constraint stated in mirage.py).

Per claim span we cross two independent axes:

  intrinsic (MIRAGE CTI) : context | mixed | parametric   — did the model USE the
                            retrieved context to produce this span?
  extrinsic (OLMoTrace)  : pretraining-hit | none | untraced — is the span verbatim
                            in OLMo-3's actual training data (usage=="Pre-training")?

→ label_2x2:
    context/mixed reliance  + pretraining-hit → "both"        (grounded AND in training data)
    context/mixed reliance  + none            → "rag"         (RAG-grounded, not traced)
    parametric reliance     + pretraining-hit → "parametric"  (from memory, traceable to Dolma)
    parametric reliance     + none            → "unverified"  (not context, not traced → hallucination candidate)
    (pretraining untraced → label deferred; provenance "uncorroborated")

OLMoTrace is the throttled, UNDOCUMENTED Playground endpoint, so it runs LAST and
is disk-cached (attribution/olmotrace_playground). Re-runs are free; --no-live uses
the cache only (never hits the API). The rigorous parametric signal is
usage=="Pre-training" (default); --include-web also counts weaker full_CC matches.

Inputs : data/reliance_records.jsonl  (override with --records)
Outputs:
  data/attribution_2x2.jsonl  — one row per (q_idx, condition, span): all signals + label
  data/attribution_2x2.md     — reliance × pretraining cross-tabs, by condition and class

Usage
-----
    # smoke first (cheap), against the reliance smoke output:
    .venv311/Scripts/python.exe -m attribution.build_2x2 --records data/reliance_smoke.jsonl --limit 16
    # full pass over the Habrok run:
    .venv311/Scripts/python.exe -m attribution.build_2x2
    # offline (only spans already cached, no API):
    .venv311/Scripts/python.exe -m attribution.build_2x2 --no-live
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from attribution.mirage import reliance_bucket
from attribution.olmotrace_playground import (
    CACHE_DIR, DEFAULT_MODEL_ID, _cache_key, per_span, trace_response,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RECORDS = DATA_DIR / "reliance_records.jsonl"
CONDITIONS = ["dense", "hybrid_rerank"]

# A pretraining span must be at least this long (normalised chars / tokens) to
# count as a match — short stopword fragments ("the", "in the") are noise.
MIN_MATCH_CHARS = 16
MIN_MATCH_TOKENS = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace, for substring matching."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_records(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            log.warning("skipping malformed record line")
    return rows


def is_cached(question: str, answer: str, model_id: str, max_documents: int) -> bool:
    """Mirror trace_response's payload + key so --no-live can check the cache."""
    payload = {"prompt": question, "modelResponse": answer,
               "modelId": model_id, "max_documents": max_documents}
    return (CACHE_DIR / f"{_cache_key(payload)}.json").exists()


def pretraining_spans(data: dict, pretraining_only: bool) -> list[dict]:
    """[{text_norm, raw, docs}] for each OLMoTrace span long enough to count."""
    out = []
    for span_text, docs in per_span(data, pretraining_only=pretraining_only).items():
        nt = _norm(span_text)
        if len(nt) < MIN_MATCH_CHARS or len(nt.split()) < MIN_MATCH_TOKENS:
            continue
        out.append({"text_norm": nt, "raw": span_text, "docs": docs})
    return out


def span_hits(claim_norm: str, ptspans: list[dict]) -> list[dict]:
    """OLMoTrace pretraining spans that overlap this claim (substring either way)."""
    return [s for s in ptspans
            if s["text_norm"] in claim_norm or claim_norm in s["text_norm"]]


def label_2x2(reliance: str, pretraining: bool | None) -> tuple[str, str]:
    """(label_2x2, provenance_enum). Mirrors pipeline.provenance_for, but the
    grounding axis is intrinsic CTI reliance, not the cosine support proxy."""
    grounded = reliance in ("context", "mixed")
    if pretraining is None:
        return ("grounded?" if grounded else "memory?"), "uncorroborated"
    if grounded:
        return ("both" if pretraining else "rag"), "corpus"
    return ("parametric" if pretraining else "unverified"), \
           ("parametric" if pretraining else "unverified")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--records", default=str(DEFAULT_RECORDS),
                    help="reliance_experiment.py output (canonical answers + CTI/LOO)")
    ap.add_argument("--out-jsonl", default=str(DATA_DIR / "attribution_2x2.jsonl"))
    ap.add_argument("--out-md", default=str(DATA_DIR / "attribution_2x2.md"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="OLMoTrace index model id (the 32B's own training index)")
    ap.add_argument("--max-documents", type=int, default=10)
    ap.add_argument("--include-web", action="store_true",
                    help="also count weaker full_CC web matches (default: Pre-training only)")
    ap.add_argument("--include-refusals", action="store_true",
                    help="keep refusal/empty answers (excluded by default, as in reliance_analysis)")
    ap.add_argument("--no-live", action="store_true",
                    help="use the OLMoTrace disk cache only; never call the API")
    ap.add_argument("--limit", type=int, help="first N records (smoke)")
    args = ap.parse_args()

    rec_path = Path(args.records)
    if not rec_path.exists():
        log.error(f"records not found: {rec_path}  (run the reliance experiment first)")
        return 2
    records = load_records(rec_path)
    if args.limit:
        records = records[: args.limit]
    if not records:
        log.error("no records loaded")
        return 2
    pretraining_only = not args.include_web
    log.info(f"{len(records)} reliance records; OLMoTrace model_id={args.model_id} "
             f"({'Pre-training only' if pretraining_only else 'incl. web'}, "
             f"{'cache-only' if args.no_live else 'live+cache'})")

    rows: list[dict] = []
    traced = skipped_untraced = skipped_answer = 0
    for ri, rec in enumerate(records, 1):
        if rec.get("answer_empty") or (rec.get("refusal") and not args.include_refusals):
            skipped_answer += 1
            continue
        q, answer = rec["question"], rec.get("answer", "")
        claims = rec.get("claims") or []
        if not answer.strip() or not claims:
            skipped_answer += 1
            continue

        # ---- extrinsic lens over THIS answer (cached) ----
        ptspans: list[dict] = []
        pretraining_known = False
        if args.no_live and not is_cached(q, answer, args.model_id, args.max_documents):
            skipped_untraced += 1
        else:
            try:
                data = trace_response(q, answer, model_id=args.model_id,
                                      max_documents=args.max_documents, use_cache=True)
                ptspans = pretraining_spans(data, pretraining_only)
                pretraining_known = True
                traced += 1
            except Exception as e:  # API down / changed — leave spans untraced
                log.warning(f"  q{rec.get('q_idx')}/{rec.get('condition')}: OLMoTrace failed ({e})")
                skipped_untraced += 1

        # ---- per-span join: intrinsic CTI × extrinsic pretraining ----
        for si, claim in enumerate(claims):
            cnorm = _norm(claim.get("text", ""))
            reliance = reliance_bucket(claim.get("cti_mean", 0.0))
            hits = span_hits(cnorm, ptspans) if pretraining_known else []
            pretraining = (len(hits) > 0) if pretraining_known else None
            lab, prov = label_2x2(reliance, pretraining)
            corpora = sorted({d.get("corpus") or d.get("source")
                              for h in hits for d in h["docs"] if (d.get("corpus") or d.get("source"))})
            rows.append({
                "q_idx": rec.get("q_idx"), "condition": rec.get("condition"),
                "contrast_class": rec.get("contrast_class"), "span_idx": si,
                "text": claim.get("text", ""),
                "cti_mean": claim.get("cti_mean"), "cti_max": claim.get("cti_max"),
                "reliance": reliance,                         # context | mixed | parametric
                "pretraining_hit": pretraining,               # True | False | None(untraced)
                "n_pretraining_spans": len(hits),
                "olmotrace_corpora": corpora,
                "label_2x2": lab,                             # rag | both | parametric | unverified | *?*
                "provenance": prov,                           # viewer enum (pipeline.py)
                "gold_loo_drop": rec.get("gold_loo_drop"),    # record-level causal "which passage"
                "gold_in_context": rec.get("gold_in_context"),
                "model": rec.get("model"),
            })

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"{len(rows)} spans → {out_jsonl}  "
             f"(traced={traced}, untraced={skipped_untraced}, skipped answers={skipped_answer})")

    write_report(rows, Path(args.out_md), pretraining_only, traced, skipped_untraced)
    return 0


# ---- reporting: the cross-tabs that ARE the extrinsic half of RQ3 -----------
def _crosstab(rows: list[dict]) -> list[str]:
    """reliance (rows) × pretraining (cols) counts."""
    rel_order = ["context", "mixed", "parametric"]
    cells = Counter((r["reliance"],
                     "hit" if r["pretraining_hit"] is True else
                     "none" if r["pretraining_hit"] is False else "untraced")
                    for r in rows)
    lines = ["| reliance \\ training-data | pretraining-hit | none | untraced | row total |",
             "|---|---:|---:|---:|---:|"]
    for rel in rel_order:
        h, n, u = cells[(rel, "hit")], cells[(rel, "none")], cells[(rel, "untraced")]
        if h + n + u == 0:
            continue
        lines.append(f"| {rel} | {h} | {n} | {u} | {h + n + u} |")
    return lines


def _label_dist(rows: list[dict]) -> str:
    n = len(rows) or 1
    c = Counter(r["label_2x2"] for r in rows)
    parts = [f"{k} {c[k]} ({100*c[k]/n:.0f}%)" for k in
             ("rag", "both", "parametric", "unverified") if c.get(k)]
    other = sum(v for k, v in c.items() if k not in ("rag", "both", "parametric", "unverified"))
    if other:
        parts.append(f"untraced {other} ({100*other/n:.0f}%)")
    return ", ".join(parts) if parts else "—"


def write_report(rows, out_md: Path, pretraining_only: bool,
                 traced: int, untraced: int) -> None:
    grounded = lambda r: r["reliance"] in ("context", "mixed")
    out = ["# RQ3 attribution — intrinsic (CTI) × extrinsic (OLMoTrace) 2×2\n",
           f"- Spans: **{len(rows)}** over {len({(r['q_idx'], r['condition']) for r in rows})} "
           f"(question, condition) answers; OLMoTrace traced {traced}, untraced {untraced}.",
           f"- Extrinsic signal: {'usage==Pre-training only (rigorous)' if pretraining_only else 'incl. full_CC web matches'}.",
           "- `rag` = context-driven, not in training data · `both` = context-driven AND in "
           "training · `parametric` = from training, not context · `unverified` = neither "
           "(hallucination candidate).\n",
           "## Overall reliance × training-data\n"]
    out += _crosstab(rows)

    out += ["", "## Span label distribution by retrieval condition\n",
            "*The extrinsic complement to the CTI-shift result: does better retrieval move "
            "spans out of `unverified`/`parametric` into `rag`/`both`?*\n",
            "| condition | n spans | distribution | grounded% (rag+both) | unverified% |",
            "|---|---:|---|---:|---:|"]
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        n = len(sub)
        g = 100 * sum(1 for r in sub if r["label_2x2"] in ("rag", "both")) / n
        u = 100 * sum(1 for r in sub if r["label_2x2"] == "unverified") / n
        out.append(f"| {cond} | {n} | {_label_dist(sub)} | {g:.0f}% | {u:.0f}% |")

    out += ["", "## Span label distribution by contrast class\n",
            "| class | condition | n spans | grounded% | unverified% |",
            "|---|---|---:|---:|---:|"]
    for cls in sorted({r["contrast_class"] for r in rows if r.get("contrast_class")}):
        for cond in CONDITIONS:
            sub = [r for r in rows if r["contrast_class"] == cls and r["condition"] == cond]
            if not sub:
                continue
            n = len(sub)
            g = 100 * sum(1 for r in sub if r["label_2x2"] in ("rag", "both")) / n
            u = 100 * sum(1 for r in sub if r["label_2x2"] == "unverified") / n
            out.append(f"| {cls} | {cond} | {n} | {g:.0f}% | {u:.0f}% |")

    out += ["", "*Intrinsic reliance from MIRAGE CTI (context ≥0.30, parametric ≤0.05, else "
            "mixed). Extrinsic from OLMoTrace over the reliance run's own greedy answer; a span "
            "counts as in-training only on a usage==Pre-training match ≥"
            f"{MIN_MATCH_CHARS} chars. Refusals/empties excluded.*"]
    out_md.write_text("\n".join(out), encoding="utf-8")
    log.info(f"cross-tabs → {out_md}")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
