"""
OLMoTrace-style extrinsic attribution via the public infini-gram API.

Operations used:
- count            → does this exact span appear in the corpus, and how often?
- find             → locate documents containing it (ranges per shard)
- get_doc_by_rank  → fetch a matching training document (with the span span-marked)

match the index to your generator's training data
--------------------------------------------------------------
OLMoTrace's claim is "this is in *this model's* training data." So the index
must be the generator's corpus, or the attribution is only approximate:
  - Generate with OLMo-2-32B-Instruct  → use `v4_olmo-2-0325-32b-instruct_llama` (exact)
  - Generate with an OLMo on the mix    → `v4_olmo-mix-1124_llama`
  - Foundational / approximate          → `v4_dolma-v1_7_llama`

Usage
-----
    # CLI smoke: is a phrase in Dolma, and where?
    .venv311/Scripts/python.exe -m attribution.olmotrace --query "Sendai Framework for Disaster Risk Reduction"
    .venv311/Scripts/python.exe -m attribution.olmotrace --query "..." --index v4_olmo-2-0325-32b-instruct_llama --docs 3
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

_COUNT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "attribution" / "_infinigram_cache"

API_BASE = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"  # OLMo foundational training data (2.6T)

# Indexes whose corpus is (close to) an OLMo model's training data.
OLMO_INDEXES = {
    "dolma-1.7": "v4_dolma-v1_7_llama",
    "olmo-mix-1124": "v4_olmo-mix-1124_llama",
    "olmo2-32b-instruct": "v4_olmo-2-0325-32b-instruct_llama",
    "olmo2-13b-instruct": "v4_olmo-2-1124-13b-instruct_llama",
    "olmoe-1b-7b-instruct": "v4_olmoe-0125-1b-7b-instruct_llama",
}


class InfiniGramError(RuntimeError):
    pass


def _post(payload: dict, retries: int = 4, timeout: int = 25) -> dict:
    """POST with retries — the API explicitly does not guarantee 100% uptime."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(API_BASE, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise InfiniGramError(data["error"])
            return data
        except Exception as e:  # noqa: BLE001 — retry transient failures
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise InfiniGramError(f"infini-gram API failed after {retries} tries: {last}")


def count(query: str, index: str = DEFAULT_INDEX) -> dict:
    """How many times the exact n-gram appears in the corpus."""
    return _post({"index": index, "query_type": "count", "query": query})


def count_cached(query: str, index: str = DEFAULT_INDEX) -> dict:
    """count() with on-disk caching keyed by (index, query). A phrase's corpus
    frequency is fixed, so caching keeps a production run (hundreds of repeated
    spans) cheap and reproducible from disk."""
    key = hashlib.sha256(f"{index}␟{query}".encode("utf-8")).hexdigest()[:24]
    cf = _COUNT_CACHE_DIR / f"{key}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = count(query, index)
    try:
        _COUNT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


def find(query: str, index: str = DEFAULT_INDEX) -> dict:
    """Locate documents containing the exact string (rank ranges per shard)."""
    return _post({"index": index, "query_type": "find", "query": query})


def get_doc_by_rank(s: int, rank: int, query: str, index: str = DEFAULT_INDEX,
                    max_disp_len: int = 500) -> dict:
    """Fetch one matching document by (shard, rank)."""
    return _post({
        "index": index, "query_type": "get_doc_by_rank",
        "s": s, "rank": rank, "query": query, "max_disp_len": max_disp_len,
    })


def trace_span(text: str, index: str = DEFAULT_INDEX, max_docs: int = 2) -> dict:
    """Extrinsic attribution for one answer span.

    Returns a dict matching attribution.pipeline.olmotrace_lookup's shape:
      {parametric: bool|None, count: int, dolma_matches: [{snippet, doc}], index}
    `parametric` is True if the exact span occurs in the corpus at least once.
    (v1 = exact full-span match; the full OLMoTrace also finds maximal partial
    spans — a documented next step, not needed for a first pass.)
    """
    text = " ".join(text.split())
    out = {"parametric": None, "count": 0, "dolma_matches": [], "index": index}
    try:
        c = count(text, index)
    except InfiniGramError as e:
        out["error"] = str(e)
        return out

    out["count"] = int(c.get("count", 0))
    out["parametric"] = out["count"] > 0
    if out["count"] == 0:
        return out

    # Pull a couple of example training documents containing the span.
    try:
        f = find(text, index)
        shards = f.get("segment_by_shard", [])
        for s, seg in enumerate(shards):
            if not seg or len(seg) < 2:
                continue
            start_rank = seg[0]
            doc = get_doc_by_rank(s=s, rank=start_rank, query=text, index=index)
            snippet = (doc.get("text") or "")[:300]
            out["dolma_matches"].append({
                "snippet": snippet,
                "doc": doc.get("doc_ix") or doc.get("metadata") or f"shard {s} rank {start_rank}",
            })
            if len(out["dolma_matches"]) >= max_docs:
                break
    except InfiniGramError:
        pass  # count is the load-bearing signal; example docs are a bonus
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--query", required=True, help="span / phrase to trace")
    ap.add_argument("--index", default=DEFAULT_INDEX,
                    help=f"infini-gram index (default {DEFAULT_INDEX}); "
                         f"shortcuts: {', '.join(OLMO_INDEXES)}")
    ap.add_argument("--docs", type=int, default=2, help="example documents to fetch")
    args = ap.parse_args()

    index = OLMO_INDEXES.get(args.index, args.index)
    print(f"[olmotrace] index={index}")
    print(f"[olmotrace] query={args.query!r}\n")

    res = trace_span(args.query, index=index, max_docs=args.docs)
    if res.get("error"):
        print(f"API error: {res['error']}", file=sys.stderr)
        return 1
    print(f"in training data : {res['parametric']}")
    print(f"occurrences      : {res['count']:,}")
    for i, m in enumerate(res["dolma_matches"], 1):
        print(f"\n  --- match {i} ({m['doc']}) ---")
        print("  " + m["snippet"].replace("\n", " ")[:280])
    if res["parametric"]:
        print("\n-> PARAMETRIC: this span is traceable to the training corpus.")
    else:
        print("\n-> NOT FOUND: not a verbatim match in this corpus "
              "(could be paraphrase, or genuinely novel/hallucinated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
