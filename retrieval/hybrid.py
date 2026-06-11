"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines a dense ranked list (Qdrant cosine over BGE-M3 chunk embeddings)
with a sparse ranked list (BM25 over chunk text + per-doc metadata).
Final order is by RRF score:

    RRF(d) = Σ_i 1 / (k + rank_i(d))

where rank_i(d) is the rank of document d in the i-th retriever (1-indexed,
omitted if not present). Constant k=60 is the convention from
Cormack et al. 2009; results are not very sensitive to it.

RRF deliberately ignores raw scores — that's its strength here, since dense
cosine and BM25 scores live on incomparable scales.
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Each input list is ordered by rank (rank 1 = best); each entry must
    contain a 'chunk_id'. Returns a fused list of length min(top_k, …),
    each entry annotated with `rrf_score` and a freshly assigned `rank`.
    The first ranked_list a chunk_id appears in supplies the entry payload.
    """
    scores: dict[str, float] = defaultdict(float)
    repr_entry: dict[str, dict] = {}
    contributing_ranks: dict[str, dict[int, int]] = defaultdict(dict)

    for source_idx, ranked in enumerate(ranked_lists):
        for rank, hit in enumerate(ranked, 1):
            cid = hit.get("chunk_id")
            if cid is None:
                continue
            scores[cid] += 1.0 / (k + rank)
            contributing_ranks[cid][source_idx] = rank
            if cid not in repr_entry:
                repr_entry[cid] = dict(hit)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for new_rank, (cid, s) in enumerate(fused[:top_k], 1):
        entry = dict(repr_entry[cid])
        entry["rrf_score"] = float(s)
        entry["rank"] = new_rank
        entry["source_ranks"] = dict(contributing_ranks[cid])
        out.append(entry)
    return out
