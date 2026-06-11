"""
Cross-encoder reranking for dense retrieval candidates.

Pairs with BGE-M3 dense retrieval: retrieve a candidate pool with dense
top-N, then reorder with the cross-encoder for final top-k.

Default model: BAAI/bge-reranker-v2-m3 (canonical pairing with BGE-M3,
multilingual, ~568M params, 8192-token context).
"""

import logging

from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = RERANKER_MODEL,
        max_length: int = 512,
    ):
        log.info(f"Loading reranker {model_name} on {device}…")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[dict]:
        """Rerank candidates by (query, candidate.text) cross-encoder score.

        Each candidate dict must contain a 'text' field. Returns a new list
        of length min(top_k, len(candidates)) with `rerank_score` populated
        and `rank` reset to the post-rerank order.
        """
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
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
