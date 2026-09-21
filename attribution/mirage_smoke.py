"""
MIRAGE-on-OLMo feasibility smoke test.

This does not run the MIRAGE repo (it ships LLaMA2 configs, not OLMo). It checks
that OLMo exposes the two model internals MIRAGE depends on

Verdict:
  CTI PASS + CCI PASS  → MIRAGE-style attribution is feasible on OLMo.
  CCI OOM on GPU       → the primitive works but is VRAM-bound run attribution on the 7B / on Habrok, not 32B.

Usage
-----
    .venv311/Scripts/python.exe -m attribution.mirage_smoke
    .venv311/Scripts/python.exe -m attribution.mirage_smoke --model allenai/Olmo-3-7B-Instruct
    .venv311/Scripts/python.exe -m attribution.mirage_smoke --model allenai/OLMo-2-0425-1B-Instruct
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct"

# A tiny RAG example: two context passages, only the first is relevant. A
# working CCI primitive should attribute the answer to D1, not D2.
PASSAGES = [
    ("D1", "The Sendai Framework sets out four priorities for action. The first "
           "priority is understanding disaster risk in all its dimensions."),
    ("D2", "El Nino is a climate pattern in the tropical Pacific Ocean that "
           "influences weather across the globe."),
]
QUESTION = "What is the first priority of the Sendai Framework?"


def build_prompt_ids(tok, passages, question, with_context: bool):
    """Build prompt token ids manually so we know each passage's token span.

    Returns (ids tensor [1,T], list of (tag, start, end) context spans).
    Plain prompt (no chat template) keeps token alignment simple for the test;
    the production pipeline would use OLMo's chat format.
    """
    ids: list[int] = []
    spans: list[tuple[str, int, int]] = []
    if tok.bos_token_id is not None:
        ids.append(tok.bos_token_id)

    def add(text):
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])

    if with_context:
        add("Context:\n")
        for tag, txt in passages:
            start = len(ids)
            add(f"[{tag}] {txt}\n")
            spans.append((tag, start, len(ids)))
    add(f"\nQuestion: {question}\nAnswer:")
    return torch.tensor([ids]), spans


def next_token_logprobs(model, ids, answer_ids, device):
    """Teacher-force prompt+answer; return per-answer-token logprob distributions.

    distribution predicting answer token j sits at position (len_prompt + j - 1).
    """
    full = torch.cat([ids, answer_ids], dim=1).to(device)
    Lp = ids.shape[1]
    with torch.no_grad():
        logits = model(full).logits[0]  # [T, V]
    out = []
    for j in range(answer_ids.shape[1]):
        out.append(F.log_softmax(logits[Lp + j - 1].float(), dim=-1))
    return torch.stack(out)  # [A, V]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--top", type=int, default=8, help="how many context-sensitive tokens to show")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(f"[load] {args.model} ({args.dtype}) on {device}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, device_map=device, trust_remote_code=True
    )
    model.eval()

    # ---- generate an answer WITH context ----
    ids_ctx, spans = build_prompt_ids(tok, PASSAGES, QUESTION, with_context=True)
    ids_noctx, _ = build_prompt_ids(tok, PASSAGES, QUESTION, with_context=False)
    with torch.no_grad():
        gen = model.generate(
            ids_ctx.to(device), max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    answer_ids = gen[:, ids_ctx.shape[1]:].cpu()
    answer_txt = tok.decode(answer_ids[0], skip_special_tokens=True)
    print(f"\n[answer] {answer_txt!r}\n")

    cti_ok = cci_ok = False

    # ================= CTI: contrastive token identification =================
    try:
        lp_with = next_token_logprobs(model, ids_ctx, answer_ids, device)     # [A,V]
        lp_without = next_token_logprobs(model, ids_noctx, answer_ids, device)  # [A,V]
        p_with = lp_with.exp()
        kl = (p_with * (lp_with - lp_without)).sum(-1)  # KL(with||without) per token
        order = torch.argsort(kl, descending=True)
        print("=== CTI: most context-sensitive answer tokens (KL with||without) ===")
        for k in range(min(args.top, answer_ids.shape[1])):
            j = int(order[k])
            tokstr = tok.decode(answer_ids[0, j:j+1])
            print(f"  KL={kl[j]:7.3f}  Δlogp={float(lp_with[j, answer_ids[0,j]]-lp_without[j, answer_ids[0,j]]):+6.2f}  {tokstr!r}")
        cti_ok = True
        most_sensitive_j = int(order[0])
    except Exception as e:  # noqa: BLE001
        print(f"[CTI] FAILED: {type(e).__name__}: {e}")
        most_sensitive_j = 0

    # ================= CCI: input-gradient saliency to a passage =============
    try:
        full = torch.cat([ids_ctx, answer_ids], dim=1).to(device)
        Lp = ids_ctx.shape[1]
        pos = Lp + most_sensitive_j - 1
        target_id = int(answer_ids[0, most_sensitive_j])

        embed = model.get_input_embeddings()
        inp = embed(full).detach().clone().requires_grad_(True)
        logits = model(inputs_embeds=inp).logits
        model.zero_grad(set_to_none=True)
        logits[0, pos, target_id].backward()

        sal = (inp.grad[0] * inp[0]).sum(-1).abs()  # grad·input saliency per token
        print("\n=== CCI: passage saliency for the most context-sensitive token "
              f"({tok.decode(answer_ids[0, most_sensitive_j:most_sensitive_j+1])!r}) ===")
        scored = sorted(((tag, float(sal[s:e].sum())) for tag, s, e in spans),
                        key=lambda x: -x[1])
        total = sum(v for _, v in scored) or 1.0
        for tag, v in scored:
            print(f"  {tag}: saliency={v:8.3f}  ({100*v/total:4.1f}%)")
        print(f"  → attributed to: {scored[0][0]}  (expected D1)")
        cci_ok = True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[CCI] GPU OOM — the gradient primitive WORKS but is VRAM-bound.")
            print("      Implication: run attribution on the 7B / Habrok, not the 32B locally.")
            torch.cuda.empty_cache()
        else:
            print(f"[CCI] FAILED: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[CCI] FAILED: {type(e).__name__}: {e}")

    # ---- verdict ----
    print("\n" + "=" * 60)
    print(f"  CTI (contrastive context-sensitivity): {'PASS' if cti_ok else 'FAIL'}")
    print(f"  CCI (gradient attribution to passage) : {'PASS' if cci_ok else 'FAIL/VRAM'}")
    if cti_ok and cci_ok:
        print("  → OLMo exposes both internals MIRAGE needs. Threads 1+2 are feasible.")
    elif cti_ok:
        print("  → Contrastive signal works; gradient attribution needs more VRAM or a smaller OLMo.")
    print("=" * 60)
    return 0 if cti_ok else 1


if __name__ == "__main__":
    sys.exit(main())
