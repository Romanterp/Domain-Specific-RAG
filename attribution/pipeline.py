"""
Provenance pipeline: retrieve → RAG-generate → per-span grounding (+ OLMoTrace stub).

Produces the cache records consumed by attribution/app.py. Designed for STAGED
execution (see build_examples.py) so it fits a 16 GB GPU: the retriever
(BGE-M3 + reranker), the generator (OLMo), and the grounder (BGE-M3) are loaded
one stage at a time, never simultaneously.

Heavy ML imports (torch / transformers / sentence-transformers / qdrant) are
LAZY — imported inside the classes — so the Streamlit app can import the
constants and `derive_attribution` from this module without pulling in torch.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_QDRANT_PATH = DATA_DIR / "qdrant"
DEFAULT_BM25_PATH = DATA_DIR / "bm25_index.pkl"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_REVISION = "refs/pr/130"  # see project_torch_bgem3_gotcha memory

# ---- support thresholds (tunable) ---------------------------------------
# A span's similarity to a passage is cosine (BGE-M3, normalized) in [0, 1].
SUPPORT_STRONG = 0.65   # top source >= this        → "strong"
SUPPORT_PARTIAL = 0.45  # top source >= this        → "partial"; below → "none"
SOURCE_FLOOR = 0.40     # passages >= this count as a corroborating source

SUPPORT_ORDER = ["strong", "partial", "none"]
# Background = SUPPORT strength only (how well the CORPUS backs the claim).
# "none" is neutral grey, NOT red — a claim the corpus doesn't cover isn't
# automatically wrong; provenance (below) says whether it's general knowledge
# or unverified.
SUPPORT_COLORS = {
    "strong": "#c8e6c9",   # green
    "partial": "#fff3cd",  # amber
    "none": "#eef0f2",     # neutral grey
}
SUPPORT_HELP = {
    "strong": "Well supported — at least one high-confidence corpus passage backs this claim.",
    "partial": "Partially supported — a corpus passage is related but not a strong match.",
    "none": "Not found in the corpus. See provenance for whether it's general knowledge or unverified.",
}

# Provenance is a SEPARATE axis from strength: where the claim comes from.
PROVENANCE_ORDER = ["corpus", "parametric", "unverified", "uncorroborated"]
PROVENANCE_HELP = {
    "corpus": "Backed by your UNDRR corpus (RAG). The specialized retrieval is doing the work.",
    "parametric": "Not in your corpus, but in the model's training data — general knowledge, not a corpus-grounded answer.",
    "unverified": "Not in the corpus and not in the model's training data — a hallucination candidate.",
    "uncorroborated": "Not found in the corpus; training-data check (OLMoTrace) not yet run.",
}
PROVENANCE_COLORS = {
    "corpus": "#2e7d32",
    "parametric": "#1565c0",
    "unverified": "#c62828",
    "uncorroborated": "#777777",
}


def support_bucket(top_score: float) -> str:
    if top_score >= SUPPORT_STRONG:
        return "strong"
    if top_score >= SUPPORT_PARTIAL:
        return "partial"
    return "none"


def provenance_for(support: str, parametric: bool | None) -> str:
    """Where a claim comes from, given corpus support and (optional) OLMoTrace.

    Strength and provenance are independent: a strongly-supported claim is
    `corpus` regardless of whether the model also knew it parametrically; only
    claims the corpus does NOT cover fall back to the parametric / unverified
    distinction.
    """
    if support != "none":
        return "corpus"
    if parametric is True:
        return "parametric"
    if parametric is False:
        return "unverified"
    return "uncorroborated"  # Phase 1: OLMoTrace not run


def split_into_spans(answer: str) -> list[str]:
    """Split an answer into sentence-ish spans for per-claim attribution.

    Deliberately simple: split on sentence-final punctuation followed by
    whitespace + capital/quote. Keeps the punctuation. Good enough for prose
    answers; a future version could use a real sentence splitter.
    """
    answer = answer.strip()
    if not answer:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", answer)
    return [p.strip() for p in pieces if p.strip()]


def build_rag_prompt(question: str, passages: list[dict], max_passages: int = 6) -> str:
    ctx = []
    for i, p in enumerate(passages[:max_passages], 1):
        title = p.get("title", "(untitled)")
        page = p.get("page", "?")
        ctx.append(f"[{i}] {title} (p.{page}): {p.get('text','').strip()}")
    context = "\n\n".join(ctx)
    return (
        "You are answering questions about UN disaster risk reduction using ONLY "
        "the context passages below. Do not use outside knowledge. If the answer "
        "is not in the passages, say you cannot answer from the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================================================================
# Stage 1 — retrieval (BGE-M3 dense + BM25 + RRF + cross-encoder rerank)
# =========================================================================
class Retriever:
    def __init__(
        self,
        qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
        collection: str = "theisus_none",
        bm25_path: str | Path = DEFAULT_BM25_PATH,
        device: str | None = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ):
        import torch
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.collection = collection
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank

        log.info(f"[retriever] loading {EMBED_MODEL} on {self.device}")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device, revision=EMBED_REVISION)
        self.client = QdrantClient(path=str(qdrant_path))

        self.bm25 = None
        if use_hybrid:
            from retrieval.bm25 import BM25Index
            self.bm25 = BM25Index.load(Path(bm25_path))

        self.reranker = None
        if use_rerank:
            from retrieval.rerank import Reranker
            self.reranker = Reranker(device=self.device)

    def retrieve(self, query: str, candidate_pool: int = 50, top_k: int = 10) -> list[dict]:
        from retrieval.hybrid import reciprocal_rank_fusion

        qvec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        resp = self.client.query_points(
            collection_name=self.collection, query=qvec.tolist(),
            limit=candidate_pool, with_payload=True, with_vectors=False,
        )
        dense_hits = []
        for h in resp.points:
            p = h.payload or {}
            dense_hits.append({
                "chunk_id": p.get("chunk_id"), "slug": p.get("slug"),
                "page": p.get("page"), "title": p.get("title"),
                "source_url": p.get("source_url"), "score": float(h.score),
                "text": p.get("text", ""),
            })

        if self.bm25 is not None:
            bm25_hits = self.bm25.query(query, top_k=candidate_pool)
            fused_top = candidate_pool if self.reranker else top_k
            candidates = reciprocal_rank_fusion(dense_hits, bm25_hits, k=60, top_k=fused_top)
        else:
            candidates = dense_hits

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        out = []
        for rank, c in enumerate(candidates, 1):
            out.append({
                "rank": rank,
                "chunk_id": c.get("chunk_id"), "slug": c.get("slug"),
                "title": c.get("title"), "page": c.get("page"),
                "source_url": c.get("source_url"),
                "rerank_score": float(c.get("rerank_score", 0.0)),
                "rrf_score": float(c.get("rrf_score", 0.0)),
                "text": c.get("text", ""),
            })
        return out

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# =========================================================================
# Stage 2 — RAG generation (OLMo)
# =========================================================================
class Generator:
    def __init__(self, model: str = "allenai/Olmo-3-7B-Instruct",
                 device: str | None = None, dtype: str = "bf16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        dt = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        log.info(f"[generator] loading {model} ({dtype}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=dt, device_map=self.device, trust_remote_code=True,
        )
        self.model.eval()

    def answer(self, question: str, passages: list[dict], max_new_tokens: int = 256,
               temperature: float = 0.3) -> str:
        import torch

        prompt = build_rag_prompt(question, passages)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=0.9, pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# =========================================================================
# Stage 3 — per-span grounding against the retrieved passages
# =========================================================================
class Grounder:
    def __init__(self, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[grounder] loading {EMBED_MODEL} on {self.device}")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device, revision=EMBED_REVISION)

    def ground(self, spans: list[str], passages: list[dict], max_sources: int = 5,
               question: str | None = None, answer: str | None = None,
               pretraining_only: bool = True) -> list[dict]:
        """For each span, cosine-match to every passage and record ALL
        corroborating sources (>= SOURCE_FLOOR) and the support strength.

        Provenance (separate axis): if `question` and `answer` are given, the
        WHOLE answer is traced once through OLMoTrace and each span's parametric
        flag is set by whether a training-data match overlaps it. If they are
        omitted (or OLMoTrace is unreachable), parametric stays None — the
        previous behaviour — so the demo/app keeps working offline.
        """
        import numpy as np

        # One whole-answer OLMoTrace call (not per span — OLMoTrace returns the
        # maximal matching sub-spans of the full response).
        ot_ok, traces = (False, {})
        if question is not None and answer is not None:
            ot_ok, traces = olmotrace_answer(question, answer, pretraining_only=pretraining_only)

        def provenance_fields(span: str) -> dict:
            if not ot_ok:
                return {"parametric": None, "dolma_matches": []}
            matches = span_training_matches(span, traces)
            return {"parametric": bool(matches), "dolma_matches": matches}

        def empty(span: str) -> dict:
            pf = provenance_fields(span)
            return {
                "text": span, "support_score": 0.0, "support": "none",
                "n_sources": 0, "sources": [],
                "parametric": pf["parametric"], "dolma_matches": pf["dolma_matches"],
                "provenance": provenance_for("none", pf["parametric"]),
            }

        if not passages:
            return [empty(s) for s in spans]

        passage_texts = [p.get("text", "") for p in passages]
        pvecs = self.embedder.encode(passage_texts, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        svecs = self.embedder.encode(spans, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        sims = svecs @ pvecs.T  # cosine (both normalized)

        results = []
        for i, span in enumerate(spans):
            order = np.argsort(-sims[i])
            sources = []
            for j in order:
                score = float(sims[i][j])
                if score < SOURCE_FLOOR or len(sources) >= max_sources:
                    break
                p = passages[int(j)]
                sources.append({
                    "chunk_id": p.get("chunk_id"), "rank": p.get("rank"),
                    "title": p.get("title"), "page": p.get("page"),
                    "score": round(score, 3),
                })
            top_score = float(sims[i][int(order[0])])
            support = support_bucket(top_score)
            pf = provenance_fields(span)
            results.append({
                "text": span,
                "support_score": round(top_score, 3),
                "support": support,
                "n_sources": len(sources),
                "sources": sources,
                "parametric": pf["parametric"],
                "dolma_matches": pf["dolma_matches"],
                "provenance": provenance_for(support, pf["parametric"]),
            })
        return results


# =========================================================================
# OLMoTrace — extrinsic provenance (Ai2 Playground backend, OLMo-3 index)
# =========================================================================
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def olmotrace_answer(question: str, answer: str, pretraining_only: bool = True,
                     max_documents: int = 10) -> tuple[bool, dict]:
    """Trace the WHOLE answer once against OLMo's training index.

    Returns (ok, traces) where traces maps each matched answer substring to its
    list of training-document matches (see olmotrace_playground.per_span). On any
    failure (offline, endpoint down) returns (False, {}) so callers fall back to
    parametric=None rather than crashing.
    """
    try:
        from attribution.olmotrace_playground import trace_response, per_span
        data = trace_response(question, answer, max_documents=max_documents)
        return True, per_span(data, pretraining_only=pretraining_only)
    except Exception as e:  # noqa: BLE001 — offline / endpoint change must not crash grounding
        log.warning(f"[olmotrace] unavailable ({type(e).__name__}: {e}); parametric=None")
        return False, {}


def span_training_matches(span_text: str, traces: dict) -> list[dict]:
    """Training-data matches that overlap this span. OLMoTrace's matched
    substrings are substrings of the full answer, so a span is 'in training
    data' if any matched substring is contained in it."""
    sn = _norm_ws(span_text)
    if not sn:
        return []
    out = []
    for matched, docs in traces.items():
        m = _norm_ws(matched)
        if m and m in sn:
            for d in docs:
                out.append({
                    "matched": matched, "corpus": d.get("corpus"),
                    "usage": d.get("usage"), "url": d.get("url"),
                    "snippet": d.get("snippet"),
                })
    return out


def olmotrace_lookup(span_text: str) -> dict:
    """Deprecated per-span shim. OLMoTrace operates on the whole answer (see
    olmotrace_answer); kept only so any old caller still imports. Always
    parametric=None — use Grounder.ground(question=, answer=) for real traces."""
    return {"parametric": None, "dolma_matches": []}
