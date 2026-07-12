"""
Freeze a stratified annotation sample from the RQ3 production records.

The human-annotation campaign (review blocker: zero human validation) needs a
fixed, reproducible set of claim spans to label. This builds it ONCE and embeds
everything the annotator must see — question, answer, claim, all retrieved
passage texts — so the annotation app (annotate_app.py) is self-contained and
needs neither chunks.jsonl (211 MB) nor any model at runtime.

Design choices that matter for the calibration:
  - Unit = claim span (one row of a record's `claims`). Refusal/empty records
    are excluded (their spans are not calibratable claims).
  - Stratified by CTI bucket (parametric <=0.05 < mixed < 0.30 <= context, the
    thresholds under calibration) so the decision region is covered even though
    production CTI mass sits far above 0.30 — buckets are OVERSAMPLED relative
    to their natural frequency, which is the point.
  - Balanced across condition (dense / hybrid_rerank) within each bucket.
  - File order is RANDOMIZED (seeded) — the annotator works the file top to
    bottom and must stay blind to strata; CTI values are stored for the later
    ROC analysis but hidden by the app in annotate mode.
  - If the 2x2 join (attribution_2x2.jsonl) exists, pretraining_hit is attached
    as analysis metadata (never shown while annotating).

Usage
-----
    .venv311/Scripts/python.exe -m attribution.build_annotation_sample \\
        [--per-bucket 50] [--seed 42]
"""

import argparse
import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Keep in sync with attribution/mirage.py CTI_STRONG / CTI_WEAK — duplicated
# here (not imported) so this file documents the thresholds under calibration.
CTI_STRONG = 0.30
CTI_WEAK = 0.05
BUCKETS = ("parametric", "mixed", "context")


def cti_bucket(cti_mean: float) -> str:
    if cti_mean >= CTI_STRONG:
        return "context"
    if cti_mean <= CTI_WEAK:
        return "parametric"
    return "mixed"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--records", default=str(DATA_DIR / "reliance_records.jsonl"))
    ap.add_argument("--two-by-two", default=str(DATA_DIR / "attribution_2x2.jsonl"),
                    help="optional; joins pretraining_hit as hidden metadata")
    ap.add_argument("--chunks", default=str(DATA_DIR / "chunks.jsonl"))
    ap.add_argument("--documents", default=str(DATA_DIR / "documents.json"))
    ap.add_argument("--out", default=str(DATA_DIR / "annotation_sample.jsonl"))
    ap.add_argument("--per-bucket", type=int, default=50,
                    help="target spans per CTI bucket (3 buckets => ~150 total)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    records = load_jsonl(Path(args.records))
    print(f"{len(records)} records")

    # optional 2x2 join: (q_idx, condition, span_idx) -> pretraining_hit
    # (span_idx in build_2x2 output == index into the record's claims list)
    hit_by_key: dict = {}
    p2 = Path(args.two_by_two)
    if p2.exists():
        for r in load_jsonl(p2):
            hit_by_key[(r.get("q_idx"), r.get("condition"),
                        r.get("span_idx"))] = r.get("pretraining_hit")
        print(f"2x2 join loaded: {len(hit_by_key)} spans")

    # enumerate candidate spans
    spans = []
    for rec in records:
        if rec.get("answer_empty") or rec.get("refusal"):
            continue
        for ci, claim in enumerate(rec.get("claims") or []):
            txt = (claim.get("text") or "").strip()
            if not txt or claim.get("cti_mean") is None:
                continue
            spans.append({
                "span_key": f"{rec['q_idx']}:{rec['condition']}:{ci}",
                "q_idx": rec["q_idx"],
                "condition": rec["condition"],
                "claim_idx": ci,
                "contrast_class": rec.get("contrast_class"),
                "question": rec["question"],
                "answer": rec["answer"],
                "claim_text": txt,
                "cti_mean": claim.get("cti_mean"),
                "cti_max": claim.get("cti_max"),
                "cti_bucket": cti_bucket(claim.get("cti_mean") or 0.0),
                "pretraining_hit": hit_by_key.get(
                    (rec["q_idx"], rec["condition"], ci)),
                "gold_chunk_id": rec.get("gold_chunk_id"),
                "gold_rank": rec.get("gold_rank"),
                "gold_in_context": rec.get("gold_in_context"),
                "passage_chunk_ids": rec.get("passage_chunk_ids") or [],
                "model": rec.get("model"),
            })
    n_by_bucket = {b: sum(1 for s in spans if s["cti_bucket"] == b) for b in BUCKETS}
    print(f"{len(spans)} candidate spans; natural bucket counts: {n_by_bucket}")

    # stratified sample: per bucket, balanced across condition where possible
    sampled = []
    for b in BUCKETS:
        pool = [s for s in spans if s["cti_bucket"] == b]
        take = min(args.per_bucket, len(pool))
        by_cond: dict[str, list] = {}
        for s in pool:
            by_cond.setdefault(s["condition"], []).append(s)
        for lst in by_cond.values():
            rng.shuffle(lst)
        picked: list = []
        # round-robin across conditions until the bucket quota is met
        conds = sorted(by_cond)
        i = 0
        while len(picked) < take:
            lst = by_cond[conds[i % len(conds)]]
            if lst:
                picked.append(lst.pop())
            elif all(not by_cond[c] for c in conds):
                break
            i += 1
        sampled.extend(picked)
        print(f"  bucket {b:<10}: sampled {len(picked)} of {len(pool)}")

    # resolve passage texts: union of needed chunk ids, one stream over chunks.jsonl
    needed = set()
    for s in sampled:
        needed.update(s["passage_chunk_ids"])
        if s["gold_chunk_id"]:
            needed.add(s["gold_chunk_id"])
    texts: dict[str, dict] = {}
    with open(args.chunks, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            c = json.loads(ln)
            if c["chunk_id"] in needed:
                texts[c["chunk_id"]] = c
                if len(texts) == len(needed):
                    break
    missing = needed - set(texts)
    if missing:
        print(f"WARN: {len(missing)} chunk ids not found in {args.chunks}",
              file=sys.stderr)

    title_by_slug = {d.get("slug"): d.get("title") or d.get("slug")
                     for d in json.loads(Path(args.documents).read_text(encoding="utf-8"))}

    for s in sampled:
        passages = []
        for cid in s.pop("passage_chunk_ids"):
            c = texts.get(cid)
            passages.append({
                "chunk_id": cid,
                "title": title_by_slug.get(c["slug"], cid.split(":")[0]) if c else cid,
                "page": c.get("page") if c else None,
                "text": c.get("text", "(chunk text not found)") if c else "(chunk text not found)",
                "is_gold": cid == s["gold_chunk_id"],
            })
        s["passages"] = passages
        s["sample_seed"] = args.seed

    # randomized annotation order — annotator stays blind to strata
    rng.shuffle(sampled)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in sampled:
            # ensure_ascii: corpus text contains U+2028/U+2029, which Python's
            # splitlines() treats as line breaks — escaped, the file stays
            # one-JSON-object-per-physical-line for any reader.
            f.write(json.dumps(s, ensure_ascii=True) + "\n")
    print(f"Wrote {len(sampled)} spans -> {args.out}")
    print("Next: streamlit run attribution/workbench.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
