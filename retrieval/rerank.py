"""
Cross-encoder reranking for dense retrieval candidates.

Pairs with BGE-M3 dense retrieval: retrieve a candidate pool with dense
top-N, then reorder with the cross-encoder for final top-k.

Default model: BAAI/bge-reranker-v2-m3 (canonical pairing with BGE-M3,
multilingual, ~568M params, 8192-token context).
"""

import logging

import torch
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Pin the revision so reported rerank scores are reproducible. This SHA is the
# snapshot actually in use (resolve a fresh one with
# huggingface_hub.HfApi().model_info(RERANKER_MODEL).sha).
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class Reranker:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = RERANKER_MODEL,
        max_length: int = 1024,
        revision: str = RERANKER_REVISION,
    ):
        # 1024 matches the chunk token cap (retrieval/chunk.py CHUNK_MAX_TOKENS).
        # The old 512 default silently truncated ~60% of chunks (median 552 tok),
        # so the cross-encoder only saw the front half of a typical page. The
        # model (bge-reranker-v2-m3) supports up to 8192, so 1024 is safe; note
        # the reranker's XLM-R tokenizer may differ slightly from BGE-M3's, so a
        # handful of the longest chunks can still lose a few trailing tokens.
        self.device = device
        # fp16 weights on GPU: ~2x faster on tensor cores and ~half the memory, so
        # the longer 1024-token pairs fit comfortably; the tiny fp16 score noise
        # does not change ranking order. CPU stays fp32 (fp16 CPU ops are partial).
        model_kwargs = {"torch_dtype": torch.float16} if "cuda" in str(device) else {}
        rev = (revision[:8] + "…") if revision else "main"
        log.info(f"Loading reranker {model_name}@{rev} on {device} "
                 f"(max_length={max_length}, dtype={'fp16' if model_kwargs else 'fp32'})…")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length,
                                  revision=revision, model_kwargs=model_kwargs)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[dict]:
        """Rerank candidates by (query, candidate.text) cross-encoder score.

        Each candidate dict must contain a 'text' field. Returns a new list
        of length min(top_k, len(candidates)) with `rerank_score` populated
        and `rank` reset to the post-rerank order.
        """
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
        with torch.inference_mode():  # no autograd tracking on the query path
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        ranked = sorted(
            zip(candidates, scores),
            key=lambda cs: float(cs[1]),
            reverse=True,
        )
        out = []
        for new_rank, (c, s) in enumerate(ranked[:top_k], 1):
            entry = dict(c)
            entry["rerank_score"] = float(s)
            entry["rank"] = new_rank
            out.append(entry)
        return out
