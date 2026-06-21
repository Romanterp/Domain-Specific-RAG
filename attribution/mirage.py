"""
MIRAGE-style intrinsic attribution on OLMo — the reusable primitive.

Generalises attribution/mirage_smoke.py (which used one hard-coded toy example)
into a class the pipeline can call on a real (question, retrieved passages)
pair. Two model-internal signals, both validated in the smoke test:

  CTI  Context-sensitive token identification. For each generated answer token,
       compare the model's next-token distribution WITH the retrieved context vs
       WITHOUT it; the KL divergence is how much that token was *caused by* the
       context rather than parametric memory. Aggregated to sentence level →
       "did retrieval drive this claim?" (the intrinsic axis of the 2×2).

  CCI  For a context-sensitive token, input-gradient saliency of its logit w.r.t.
       the context token embeddings, summed per passage → WHICH retrieved passage
       drove it.

Methodological note (state in the thesis): MIRAGE attributes against a *plain,
span-tracked* prompt (we need exact passage token spans), not the generator's
chat-rendered prompt. So this module GENERATES the answer it attributes, with the
same instruction as `pipeline.build_rag_prompt` but without chat formatting. The
attributed answer is therefore MIRAGE's own greedy decode, which is what the
extrinsic (OLMoTrace) lens should also be run on, so both lenses describe the
same text.

VRAM: forward+backward on the 7B fits a 16 GB 4080 (smoke test confirmed). On the
32B, gradients are heavy → run attribution on the 7B (generate-32B / attribute-7B
split) or on Habrok.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Sentence-ish splitter (kept local so this module imports without pipeline).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“])")

CTI_STRONG = 0.30   # mean per-token KL ≥ this → sentence is "context-driven"
CTI_WEAK = 0.05     # below this → "parametric" (context barely moved the tokens)


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]


@dataclass
class SpanAttribution:
    text: str
    cti_mean: float                       # mean context-sensitivity over the span's tokens
    cti_max: float
    reliance: str                         # "context" | "mixed" | "parametric"
    top_passage: str | None = None        # passage tag with most CCI saliency
    passage_saliency: dict = field(default_factory=dict)  # tag -> fraction


@dataclass
class MirageResult:
    question: str
    answer: str
    token_cti: list                       # [(token_str, kl_float)]
    spans: list                           # [SpanAttribution]
    passage_tags: dict                    # tag -> {chunk_id, title}


def reliance_bucket(cti_mean: float) -> str:
    if cti_mean >= CTI_STRONG:
        return "context"
    if cti_mean <= CTI_WEAK:
        return "parametric"
    return "mixed"


class MirageAttributor:
    def __init__(self, model: str = "allenai/Olmo-3-7B-Instruct",
                 device: str | None = None, dtype: str = "bf16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        dt = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        log.info(f"[mirage] loading {model} ({dtype}) on {self.device}")
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=dt, device_map=self.device, trust_remote_code=True,
        )
        self.model.eval()
        # CCI needs gradients only w.r.t. the INPUT embeddings, never the weights.
        # Freezing params means backward() doesn't allocate per-parameter grads
        # (~tens of GB on the 7B) — without this the per-sentence CCI loop OOMs.
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---- prompt construction with tracked passage token spans ----
    def _build_prompt_ids(self, question: str, passages: list[dict], with_context: bool):
        """Return (ids[1,T], spans[(tag,start,end)]).

        Uses the model's CHAT TEMPLATE so an Instruct model answers in-
        distribution (a plain prompt makes OLMo-Instruct degenerate). Passage
        token spans for CCI are recovered by locating each passage's verbatim
        text in the rendered string via offset mapping. Falls back to a plain
        tracked prompt only if the chat template / fast-tokenizer offsets are
        unavailable.
        """
        import torch

        passages = passages if with_context else []
        # Build the user message exactly like the generator's RAG prompt so the
        # attributed answer matches what the pipeline would produce.
        instr = ("You are answering questions about UN disaster risk reduction using "
                 "ONLY the context passages below. Do not use outside knowledge. If the "
                 "answer is not in the passages, say you cannot answer from the provided "
                 "context.\n\n") if with_context else ""
        ctx = ""
        if passages:
            ctx = "Context:\n" + "\n\n".join(
                f"[P{i}] {p.get('text','').strip()}" for i, p in enumerate(passages, 1)
            ) + "\n\n"
        user = f"{instr}{ctx}Question: {question}"

        sys_msg = ("You are a helpful assistant answering questions about UN disaster "
                   "risk reduction. Answer concisely using only the provided context.")
        try:
            rendered = self.tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = self.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            spans: list[tuple[str, int, int]] = []
            for i, p in enumerate(passages, 1):
                ptext = p.get("text", "").strip()
                ci = rendered.find(ptext)
                if ci < 0 or not ptext:
                    continue
                cj = ci + len(ptext)
                t0 = next((k for k, (a, b) in enumerate(offs) if b > ci), None)
                t1 = next((k for k, (a, b) in enumerate(offs) if a >= cj), len(ids))
                if t0 is not None:
                    spans.append((f"P{i}", t0, t1))
            if with_context and passages and not spans:
                raise RuntimeError("no passage spans located in rendered prompt")
            return torch.tensor([ids]), spans
        except Exception as e:  # noqa: BLE001 — fall back to plain tracked prompt
            log.warning(f"[mirage] chat-template span tracking unavailable ({e}); "
                        "using plain prompt (answer may be lower quality)")

        ids2: list[int] = []
        spans2: list[tuple[str, int, int]] = []
        if self.tok.bos_token_id is not None:
            ids2.append(self.tok.bos_token_id)

        def add(text: str):
            ids2.extend(self.tok(text, add_special_tokens=False)["input_ids"])

        if with_context:
            add(instr + "Context:\n")
            for i, p in enumerate(passages, 1):
                start = len(ids2)
                add(f"[P{i}] {p.get('text','').strip()}\n")
                spans2.append((f"P{i}", start, len(ids2)))
        add(f"\nQuestion: {question}\nAnswer:")
        return torch.tensor([ids2]), spans2

    def _generate(self, ids, max_new_tokens: int,
                  repetition_penalty: float = 1.15, no_repeat_ngram_size: int = 0):
        """Greedy decode with a MILD repetition penalty + explicit EOS.

        no_repeat_ngram_size is OFF by default: in this domain the answer must
        legitimately repeat key phrases ("Disaster Risk Reduction"), and hard
        n-gram blocking mangles them into garbage ("DRRisk reduction"). A soft
        repetition_penalty discourages the '[1] the [1] the' loop without banning
        necessary repeats. Decoding-only — CTI/CCI attribute whatever is produced,
        so the measurement stays unbiased."""
        import torch
        kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.no_grad():
            gen = self.model.generate(ids.to(self.device), **kwargs)
        return gen[:, ids.shape[1]:].cpu()  # answer token ids only

    def _answer_logprobs(self, prompt_ids, answer_ids):
        """Teacher-force prompt+answer; per-answer-token logprob distribution.
        Distribution predicting answer token j sits at position (Lp + j - 1)."""
        import torch
        import torch.nn.functional as F
        full = torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        Lp = prompt_ids.shape[1]
        with torch.no_grad():
            logits = self.model(full).logits[0]
        out = [F.log_softmax(logits[Lp + j - 1].float(), dim=-1)
               for j in range(answer_ids.shape[1])]
        return torch.stack(out)  # [A, V]

    def _token_char_offsets(self, answer_ids) -> list[tuple[int, int]]:
        """Char span of each answer token via cumulative decode (BPE-additive)."""
        offsets = []
        prev = 0
        ids = answer_ids[0].tolist()
        for j in range(len(ids)):
            s = self.tok.decode(ids[: j + 1], skip_special_tokens=True)
            offsets.append((prev, len(s)))
            prev = len(s)
        return offsets

    def _cci_for_token(self, full_ids, prompt_len: int, answer_pos: int,
                       target_id: int, spans):
        """grad·input saliency of the answer token's logit, summed per passage.

        Memory-frugal for a 16 GB card: (1) causal truncation — the logit at the
        target position depends only on tokens up to it, so the tail is dropped;
        (2) gradient checkpointing — activations are recomputed in backward
        rather than stored. Passage spans live in the prompt (before the target),
        so truncation never drops them.
        """
        import torch
        pos = prompt_len + answer_pos - 1
        trunc = full_ids[:, : pos + 1]                 # causal: tail irrelevant
        tpos = trunc.shape[1] - 1
        embed = self.model.get_input_embeddings()
        inp = embed(trunc.to(self.device)).detach().clone().requires_grad_(True)
        self.model.gradient_checkpointing_enable()
        try:
            logits = self.model(inputs_embeds=inp, use_cache=False).logits
            self.model.zero_grad(set_to_none=True)
            logits[0, tpos, target_id].backward()
            sal = (inp.grad[0] * inp[0]).sum(-1).abs()  # [T]
            scored = {tag: float(sal[s:e].sum()) for tag, s, e in spans if s < trunc.shape[1]}
        finally:
            self.model.gradient_checkpointing_disable()
        total = sum(scored.values()) or 1.0
        return {tag: v / total for tag, v in scored.items()}

    def attribute(self, question: str, passages: list[dict],
                  max_new_tokens: int = 200, cci_per_sentence: bool = True) -> MirageResult:
        """Full intrinsic attribution for one (question, passages)."""
        import torch

        ids_ctx, spans = self._build_prompt_ids(question, passages, with_context=True)
        ids_noctx, _ = self._build_prompt_ids(question, passages, with_context=False)

        answer_ids = self._generate(ids_ctx, max_new_tokens)
        answer = self.tok.decode(answer_ids[0], skip_special_tokens=True).strip()

        # CTI: KL(with || without) per answer token.
        lp_with = self._answer_logprobs(ids_ctx, answer_ids)
        lp_without = self._answer_logprobs(ids_noctx, answer_ids)
        p_with = lp_with.exp()
        kl = (p_with * (lp_with - lp_without)).sum(-1)  # [A]
        token_cti = [(self.tok.decode(answer_ids[0, j:j + 1]).strip(), float(kl[j]))
                     for j in range(answer_ids.shape[1])]

        # Map tokens → sentences via char offsets.
        offs = self._token_char_offsets(answer_ids)
        sentences = split_sentences(answer)
        sent_ranges = []
        cur = 0
        for s in sentences:
            i = answer.find(s, cur)
            if i < 0:
                i = cur
            sent_ranges.append((i, i + len(s)))
            cur = i + len(s)

        full_ctx = torch.cat([ids_ctx, answer_ids], dim=1)
        Lp = ids_ctx.shape[1]
        passage_tags = {f"P{i}": {"chunk_id": p.get("chunk_id"), "title": p.get("title")}
                        for i, p in enumerate(passages, 1)}

        span_attrs = []
        for (s_start, s_end), sent in zip(sent_ranges, sentences):
            tok_idx = [j for j, (a, b) in enumerate(offs)
                       if a < s_end and b > s_start]  # tokens overlapping this sentence
            if not tok_idx:
                continue
            kls = [float(kl[j]) for j in tok_idx]
            cti_mean = sum(kls) / len(kls)
            cti_max = max(kls)
            top_passage, sal = None, {}
            if cci_per_sentence and spans:
                peak = max(tok_idx, key=lambda j: float(kl[j]))  # most context-sensitive token
                try:
                    sal = self._cci_for_token(full_ctx, Lp, peak,
                                              int(answer_ids[0, peak]), spans)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.warning("[mirage] CCI OOM on this span — skipping passage "
                                "attribution (CTI unaffected). Try fewer passages "
                                "(--top-k) or --no-cci.")
                    sal = {}
                if sal:
                    top_passage = max(sal, key=sal.get)
            span_attrs.append(SpanAttribution(
                text=sent, cti_mean=cti_mean, cti_max=cti_max,
                reliance=reliance_bucket(cti_mean),
                top_passage=top_passage, passage_saliency={k: round(v, 3) for k, v in sal.items()},
            ))

        return MirageResult(question=question, answer=answer, token_cti=token_cti,
                            spans=span_attrs, passage_tags=passage_tags)
