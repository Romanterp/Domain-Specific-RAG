"""
Install-gate: does the inseq Integrated-Gradients path work on transformers
5.x stack? Replaces the dead LXT/AttnLRP gate. captum IG is architecture-agnostic
(embedding hooks), so a tiny model validates the API for OLMo-3 too. 
"""
import sys

def main():
    import inseq
    model_id = "HuggingFaceTB/SmolLM-135M"
    print(f"[gate] loading {model_id} with integrated_gradients via inseq…")
    model = inseq.load_model(model_id, "integrated_gradients")
    print("[gate] loaded. attributing a short generation…")
    out = model.attribute(
        "Context: Paris is the capital of France.\nQuestion: What is the capital of France?\nAnswer:",
        generation_args={"max_new_tokens": 4, "do_sample": False},
        n_steps=16, internal_batch_size=4, show_progress=False,
    )
    import torch
    # Aggregate over the hidden dim the canonical inseq way (L2), collapsing
    # [source, target, hidden] -> [source, target]. Causally-masked (future)
    # positions are NaN by design — drop them, then check real per-token signal.
    agg = out.aggregate()  # default aggregator over the last (hidden) dim
    ta = agg.sequence_attributions[0].target_attributions
    t = ta if isinstance(ta, torch.Tensor) else torch.tensor(ta)
    print("[gate] aggregated target_attributions shape:", tuple(t.shape))
    valid = torch.isfinite(t)
    n_valid = int(valid.sum())
    vals = t[valid]
    signal = float(vals.abs().sum())
    spread = float(vals.std()) if n_valid > 1 else 0.0
    print(f"[gate] valid (causal) entries: {n_valid} | sum|attr|: {round(signal,4)} | std: {round(spread,4)}")
    # Per target token: at least one finite, non-degenerate source attribution
    per_tok_ok = bool((valid.any(dim=0)).all()) if t.ndim == 2 else n_valid > 0
    ok = n_valid > 0 and signal > 0 and spread > 0 and per_tok_ok
    print("\n[gate] RESULT:", "PASS — inseq Integrated Gradients works on transformers 5.x, "
          "non-degenerate per-token attribution" if ok else "FAIL — degenerate after aggregation")
    return 0 if ok else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n[gate] FAIL — {type(e).__name__}: {e}")
        sys.exit(2)
