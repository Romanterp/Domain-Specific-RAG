"""
OLMoTrace via the Ai2 Playground backend — programmatic OLMo-3 training-data tracing.

This calls the same endpoint the Ai2 Playground UI uses for OLMoTrace, which
indexes OLMo-3's actual training data (the public api.infini-gram.io does not
yet expose an OLMo-3 index — see attribution/olmotrace.py for that OLMo-2/Dolma
path). Given a prompt + the model's response, it returns the training documents
whose text matches spans of the response.

Unofficial endpoint (no auth, but not a supported public API).

`usage` field distinguishes the claim strength:
  - usage == "Pre-training"  → the span is in OLMo-3's ACTUAL training data
    (e.g. displayName "olmo-mix-1124"). This is the rigorous parametric signal.
  - usage is null (e.g. source "full_CC") → match in a general web corpus, NOT
    necessarily the model's training set. Useful context, weaker claim.

Usage
-----
    .venv311/Scripts/python.exe -m attribution.olmotrace_playground \\
        --prompt "What is the Sendai Framework?" \\
        --response "The Sendai Framework for Disaster Risk Reduction 2015-2030 sets out four priorities for action."
"""

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import requests

PLAYGROUND_URL = "https://prod-api.playground.pandajungle.org/v5/attribution/"
DEFAULT_MODEL_ID = "Olmo-3.1-32B-Instruct"  # server maps → index olmo-3-0625-32b-instruct

# --- courtesy controls for an undocumented endpoint -----------------------
# 1. cache every response to disk so re-running tests never re-hits the API
# 2. throttle live calls so a loop can't burst
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "attribution" / "_olmotrace_cache"
MIN_INTERVAL_S = 2.0
_last_call = [0.0]


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL_S:
        time.sleep(MIN_INTERVAL_S - dt)
    _last_call[0] = time.time()


def trace_response(prompt: str, model_response: str,
                   model_id: str = DEFAULT_MODEL_ID,
                   max_documents: int = 10, timeout: int = 90,
                   use_cache: bool = True) -> dict:
    """POST to the Playground attribution endpoint; return the raw JSON.

    Cached to disk by request hash (use_cache=False to force a live call).
    Live calls are throttled to <= 1 per MIN_INTERVAL_S to stay courteous.
    """
    payload = {
        "prompt": prompt,
        "modelResponse": model_response,
        "modelId": model_id,
        "max_documents": max_documents,
    }
    cache_file = CACHE_DIR / f"{_cache_key(payload)}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    headers = {
        "Content-Type": "application/json",
        "Origin": "https://playground.allenai.org",
        "x-anonymous-user-id": str(uuid.uuid4()),
    }
    _throttle()
    r = requests.post(PLAYGROUND_URL, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def per_span(data: dict, pretraining_only: bool = False) -> dict:
    """Reshape the response into {span_text: [matching docs]}.

    Each doc match carries which corpus it's from and whether it's actually in
    pre-training. Set pretraining_only=True to keep only rigorous
    (usage == "Pre-training") matches — the defensible parametric signal.
    """
    spans: dict[str, list[dict]] = {}
    for doc in data.get("documents", []):
        if pretraining_only and doc.get("usage") != "Pre-training":
            continue
        match = {
            "url": doc.get("url") or doc.get("sourceUrl"),
            "source": doc.get("source"),
            "usage": doc.get("usage"),
            "corpus": doc.get("displayName") or doc.get("secondaryName"),
            "relevanceScore": doc.get("relevanceScore"),
            "snippet": (doc.get("snippets") or [{}])[0].get("text", "")[:200],
        }
        for span_text in doc.get("correspondingSpanTexts", []):
            spans.setdefault(span_text.strip(), []).append(match)
    return spans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--max-documents", type=int, default=10)
    ap.add_argument("--pretraining-only", action="store_true",
                    help="keep only usage=='Pre-training' matches (rigorous)")
    ap.add_argument("--raw", action="store_true", help="dump raw JSON")
    ap.add_argument("--refresh", action="store_true",
                    help="bypass the disk cache and force a live call")
    args = ap.parse_args()

    print(f"[olmotrace-playground] modelId={args.model_id}")
    data = trace_response(args.prompt, args.response,
                          model_id=args.model_id, max_documents=args.max_documents,
                          use_cache=not args.refresh)
    if args.raw:
        print(json.dumps(data, indent=2)[:4000])
        return 0

    print(f"[olmotrace-playground] traced index: {data.get('index')}")
    print(f"[olmotrace-playground] {len(data.get('documents', []))} matching documents\n")

    spans = per_span(data, pretraining_only=args.pretraining_only)
    if not spans:
        print("No matching training-data spans "
              + ("(with usage=Pre-training)." if args.pretraining_only else "."))
        return 0
    for span_text, docs in spans.items():
        pre = sum(1 for d in docs if d["usage"] == "Pre-training")
        print(f'SPAN: "{span_text}"')
        print(f"  {len(docs)} match(es), {pre} in pre-training")
        for d in docs[:3]:
            tag = "PRE-TRAIN" if d["usage"] == "Pre-training" else "web"
            print(f"    [{tag}] {d['corpus'] or d['source']}  score={d['relevanceScore']:.0f}  {d['url'] or ''}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
