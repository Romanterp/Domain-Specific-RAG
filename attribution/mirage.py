"""
MIRAGE-style intrinsic attribution on OLMo

Generalises attribution/mirage_smoke.py (which used one hard-coded toy example)
into a class the pipeline can call on a real (question, retrieved passages)
pair. Two model-internal signals, both validated in the smoke test:

  CTI  Context-sensitive token identification. For each generated answer token,
       compare the model's next-token distribution with the retrieved context vs
       without it; the KL divergence is how much that token was caused by the
       context rather than parametric memory. Aggregated to sentence level →
       "did retrieval drive this claim?" (the intrinsic axis of the 2×2).

  CCI  For a context-sensitive token, input-gradient saliency of its logit w.r.t.
       the context token embeddings, summed per passage → which retrieved passage
       drove it.

Methodological note: MIRAGE attributes against a plain,
span-tracked prompt (we need exact passage token spans), not the generator's
chat-rendered prompt. So this module generates the answer it attributes, with the
same instruction as `pipeline.build_rag_prompt` but without chat formatting. The
attributed answer is therefore MIRAGE's own greedy decode, which is what the
extrinsic (OLMoTrace) measure should also be run on, so both measures describe the
same text.

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
    prompt_format: str = "chat"           # "chat" | "plain" — "plain" marks the
                                          # fallback path; CTI is only comparable
                                          # within one format


def reliance_bucket(cti_mean: float) -> str:
    if cti_mean >= CTI_STRONG:
        return "context"
    if cti_mean <= CTI_WEAK:
        return "parametric"
    return "mixed"


class MirageAttributor:
    def __init__(self, model: str = "allenai/Olmo-3-7B-Instruct",
                 device: str | None = None, dtype: str = "bf16",
                 device_map: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        dt = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        n_gpu = torch.cuda.device_count()
        # Auto-shard whenever asked OR when >1 GPU is visible and no single device
        # was pinned (the 32B on 2x A100 40GB — the config doc2query proved).
        # Reserve ~6 GiB/GPU so accelerate doesn't fill GPU0 to the brim and OOM
        # (inputs + forward activations live on the embedding shard).
        _shard = (device_map == "auto") or (device is None and n_gpu > 1)
        _maxmem = None
        if _shard and n_gpu > 0:
            try:
                _gib = [torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                        for i in range(n_gpu)]
                _maxmem = {i: f"{max(1, g - 6)}GiB" for i, g in enumerate(_gib)}
            except Exception:
                _maxmem = None
        if _shard:
            log.info(f"[mirage] loading {model} ({dtype}) sharded across {n_gpu} GPU(s), "
                     f"max_memory={_maxmem}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=dt, device_map="auto", max_memory=_maxmem,
                trust_remote_code=True,
            )
            self.device = self.model.get_input_embeddings().weight.device
            log.info(f"[mirage] embedding shard={self.device}  "
                     f"map={getattr(self.model, 'hf_device_map', '?')}")
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"[mirage] loading {model} ({dtype}) on {self.device}")
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
    def _build_prompt_ids(self, question: str, passages: list[dict], with_context: bool,
                          force_plain: bool = False):
        """Return (ids[1,T], spans[(tag,start,end)], fmt "chat"|"plain").

        Uses the model's CHAT TEMPLATE so an Instruct model answers in-
        distribution (a plain prompt makes OLMo-Instruct degenerate). Passage
        token spans for CCI are recovered by locating each passage's verbatim
        text in the rendered string via offset mapping. Falls back to a plain
        tracked prompt only if the chat template / fast-tokenizer offsets are
        unavailable. force_plain skips the chat template outright — attribute()
        uses it to keep the with/without-context prompt pair in ONE format, so
        CTI never measures the template difference.
        """
        import torch

        passages = passages if with_context else []
        # Build the user message exactly like the generator's RAG prompt so the
        # attributed answer matches what the pipeline would produce.
        # Instruction held CONSTANT across with/without-context: CTI =
        # KL(with||without) must isolate the PASSAGES, not the instruction
        # framing. Only the Context block differs between the two prompts.
        instr = ("You are answering questions about UN disaster risk reduction using "
                 "ONLY the context passages below. Do not use outside knowledge. If the "
                 "answer is not in the passages, say you cannot answer from the provided "
                 "context.\n\n")
        ctx = ""
        if passages:
            ctx = "Context:\n" + "\n\n".join(
                f"[P{i}] {p.get('text','').strip()}" for i, p in enumerate(passages, 1)
            ) + "\n\n"
        user = f"{instr}{ctx}Question: {question}"

        sys_msg = ("You are a helpful assistant answering questions about UN disaster "
                   "risk reduction. Answer concisely using only the provided context.")
        if not force_plain:
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
                return torch.tensor([ids]), spans, "chat"
            except Exception as e:  # noqa: BLE001 — fall back to plain tracked prompt
                log.warning(f"[mirage] chat-template span tracking unavailable ({e}); "
                            "using plain prompt (answer may be lower quality)")

        ids2: list[int] = []
        spans2: list[tuple[str, int, int]] = []
        if self.tok.bos_token_id is not None:
            ids2.append(self.tok.bos_token_id)

        def add(text: str):
            ids2.extend(self.tok(text, add_special_tokens=False)["input_ids"])

        add(instr)  # constant across conditions (B3)
        if with_context:
            add("Context:\n")
            for i, p in enumerate(passages, 1):
                start = len(ids2)
                add(f"[P{i}] {p.get('text','').strip()}\n")
                spans2.append((f"P{i}", start, len(ids2)))
        add(f"\nQuestion: {question}\nAnswer:")
        return torch.tensor([ids2]), spans2, "plain"

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

    def _answer_logprobs(self, prompt_ids, answer_ids, slice_logits: bool = False):
        """Teacher-force prompt+answer; per-answer-token logprob distribution.
        Distribution predicting answer token j sits at position (Lp + j - 1).

        slice_logits requests only the final A+1 logit rows — exactly positions
        Lp-1 … Lp+A-1, which covers every position read below. Each row is an
        independent LM-head matvec over an unchanged hidden state, so the numbers
        are identical; it only avoids materialising the rest. That matters on a
        16 GB card: at T≈6.3k the full [1, T, 100278] tensor is ~1.2 GiB versus
        ~20 MiB sliced. OFF by default so the run that must reproduce recorded
        numbers issues the same call as the run that produced them.
        """
        import torch
        import torch.nn.functional as F
        full = torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        Lp, A = prompt_ids.shape[1], answer_ids.shape[1]
        base = 0  # index of the first logit row returned
        with torch.no_grad():
            if slice_logits:
                try:
                    logits = self.model(full, logits_to_keep=A + 1).logits[0]
                    base = full.shape[1] - (A + 1)
                except TypeError:  # transformers too old for logits_to_keep
                    logits = self.model(full).logits[0]
            else:
                logits = self.model(full).logits[0]
        out = [F.log_softmax(logits[Lp + j - 1 - base].float(), dim=-1)
               for j in range(A)]
        return torch.stack(out)  # [A, V]

    def answer_logprob(self, question: str, passages: list[dict], answer_text: str) -> float:
        """Total teacher-forced log-prob of `answer_text` given `passages`.

        The forward-only primitive behind LOO: re-score the SAME fixed answer
        under a passage subset (drop one passage → measure how far the answer's
        probability falls = that passage's causal importance). Reuses the exact
        prompt builder and per-token log-softmax used for CTI, so conditioning
        matches. NB re-tokenises answer_text (R7 re-encoding caveat); fine for the
        relative LOO drop.
        """
        ids, _, _ = self._build_prompt_ids(question, passages, with_context=True)
        tgt = self.tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids
        if tgt.shape[1] == 0:
            return 0.0
        lp = self._answer_logprobs(ids, tgt)  # [A, V] log-softmax per target position
        return float(sum(lp[j, int(tgt[0, j])] for j in range(tgt.shape[1])))

    def answer_logprob_masked(self, prompt_ids, answer_ids, mask_span=None) -> float:
        """Summed teacher-forced log-prob of answer_ids given prompt_ids, with an
        optional context span ATTENTION-MASKED.

        Masking hides a passage's tokens from attention while keeping their
        positions and the total sequence length byte-identical — explicit
        position_ids stop the masked-middle from renumbering downstream
        positions. So a leave-one-out drop isolates the passage's *content*, not
        prompt-length / position shifts (the residual flaw in text-blanking).
        Forward-only.
        """
        import torch
        import torch.nn.functional as F
        full = torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        T, Lp = full.shape[1], prompt_ids.shape[1]
        attn = torch.ones((1, T), dtype=torch.long, device=self.device)
        if mask_span is not None:
            s, e = mask_span
            attn[0, s:e] = 0  # hide these context-key tokens from every query
        pos = torch.arange(T, device=self.device).unsqueeze(0)  # positions FIXED
        with torch.inference_mode():
            logits = self.model(input_ids=full, attention_mask=attn,
                                position_ids=pos).logits[0]
        total = 0.0
        for j in range(answer_ids.shape[1]):
            lp = F.log_softmax(logits[Lp + j - 1].float(), dim=-1)
            total += float(lp[int(answer_ids[0, j])])
        return total

    def loo_drops(self, question: str, passages: list[dict], answer_text: str,
                  only_idx: int | None = None):
        """Per-passage leave-one-out importance via attention-masking.

        drop_i = logp(answer | all passages) − logp(answer | all but passage i),
        where "but passage i" masks that passage's tokens from attention with the
        rest of the prompt held identical. Returns (full_logprob, drops) with
        drops aligned to `passages` order (None where a passage's tokens could
        not be located in the prompt). Builds prompt + answer once and only flips
        the attention mask per passage — so it's also cheaper than rebuilding.

        only_idx: if set, compute the drop ONLY for that passage (others None) —
        for the gold-only mode that keeps the 32B production run tractable
        (2 forwards instead of n+1).
        """
        ids, spans, _ = self._build_prompt_ids(question, passages, with_context=True)
        tgt = self.tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids
        if tgt.shape[1] == 0:
            return 0.0, [None] * len(passages)
        full_lp = self.answer_logprob_masked(ids, tgt, mask_span=None)
        span_by_tag = {tag: (s, e) for tag, s, e in spans}
        drops = []
        for i in range(len(passages)):
            if only_idx is not None and i != only_idx:
                drops.append(None)
                continue
            span = span_by_tag.get(f"P{i + 1}")
            if span is None:
                drops.append(None)
                continue
            lp_i = self.answer_logprob_masked(ids, tgt, mask_span=span)
            drops.append(full_lp - lp_i)
        return full_lp, drops

    def _sentence_token_index(self, raw: str, answer_ids) -> list[tuple[str, list]]:
        """[(sentence_text, [token indices])] for per-claim aggregation.

        Extracted so attribute() and cti_fixed_answer share ONE implementation.
        The lead-whitespace correction, the sequential find and the half-open
        overlap rule each shift token→sentence assignment; two copies would
        drift silently while both still ran clean, and the drift would surface
        only as an unexplained gap in the diagonal validation.
        """
        lead = len(raw) - len(raw.lstrip())
        answer = raw.strip()
        offs = [(a - lead, b - lead) for (a, b) in self._token_char_offsets(answer_ids)]
        out, cur = [], 0
        for s in split_sentences(answer):
            i = answer.find(s, cur)
            if i < 0:
                i = cur
            s_start, s_end = i, i + len(s)
            cur = s_end
            out.append((s, [j for j, (a, b) in enumerate(offs)
                            if a < s_end and b > s_start]))
        return out

    def noctx_logprobs(self, question: str, answer_ids, force_plain: bool = False,
                       slice_logits: bool = False):
        """(lp_without [A,V], fmt) — the no-context reference for a FIXED answer.

        _build_prompt_ids drops the passages outright when with_context=False, so
        this term is a function of (question, answer) ALONE — independent of which
        retrieval condition supplied the context. That is what makes the
        answer-controlled 2x2 cheap: one no-context forward per (question, answer)
        serves both context columns, i.e. 6 forwards per question rather than 8.
        fmt is returned so a caller can refuse to reuse a reference that was built
        in a different prompt format.
        """
        ids_noctx, _, fmt = self._build_prompt_ids(question, [], with_context=False,
                                                   force_plain=force_plain)
        return self._answer_logprobs(ids_noctx, answer_ids,
                                     slice_logits=slice_logits), fmt

    def cti_fixed_answer(self, question: str, passages: list[dict], answer_text: str,
                         lp_without=None, lp_without_fmt: str | None = None,
                         slice_logits: bool = False) -> dict:
        """Score a FIXED answer under a supplied context — one cell of the 2x2.

        attribute() attributes its own greedy decode, so comparing CTI across
        retrieval conditions varies the answer text and the context together
        (the limitation this routine exists to quantify). Here the answer is an
        input, so a within-row contrast varies the context alone.

        Two metrics, deliberately:
          cti_mean     mean-of-sentence-means of per-token KL(with||without),
                       aggregated exactly as reliance_experiment.py did (each
                       claim rounded to 4dp BEFORE averaging) so the diagonal
                       cells reproduce the recorded answer_cti_mean.
          signed_mean  the SAME log-ratio evaluated at the realised token,
                       log p_with(a_j) - log p_without(a_j). KL is that ratio's
                       expectation under p_with and is therefore UNSIGNED: a
                       context that confidently contradicts the forced text
                       scores as high as one that supports it. On the diagonal
                       the two nearly coincide; on a foreign answer they can
                       diverge, and that divergence is the diagnostic. Without
                       it, an off-diagonal "high CTI" cannot be read as reliance.

        NB the answer is re-encoded from stored text (the R7 re-encoding caveat
        that answer_logprob already carries) — the generation's own ids were
        never persisted, which is precisely why the diagonal is the real proof.
        """
        import torch

        ids_ctx, _, fmt = self._build_prompt_ids(question, passages, with_context=True)
        # Both prompts must share one format, or CTI measures the template
        # difference rather than the passages (same dance as attribute()).
        ids_noctx, _, fmt_noctx = self._build_prompt_ids(
            question, passages, with_context=False, force_plain=(fmt == "plain"))
        if fmt_noctx != fmt:
            ids_ctx, _, fmt = self._build_prompt_ids(question, passages,
                                                     with_context=True, force_plain=True)
            ids_noctx, _, fmt_noctx = self._build_prompt_ids(
                question, passages, with_context=False, force_plain=True)

        answer_ids = self.tok(answer_text, add_special_tokens=False,
                              return_tensors="pt").input_ids
        if answer_ids.shape[1] == 0:
            return {"cti_mean": 0.0, "signed_mean": 0.0, "token_cti_mean": 0.0,
                    "claims": [], "fmt": fmt, "n_answer_tokens": 0,
                    "logp_with": None, "logp_without": None, "lp_without": None}

        lp_with = self._answer_logprobs(ids_ctx, answer_ids, slice_logits=slice_logits)
        if lp_without is None or (lp_without_fmt is not None and lp_without_fmt != fmt):
            lp_without = self._answer_logprobs(ids_noctx, answer_ids,
                                               slice_logits=slice_logits)
        kl = (lp_with.exp() * (lp_with - lp_without)).sum(-1)          # [A]
        tgt = answer_ids[0]
        idx = torch.arange(tgt.shape[0])
        signed = (lp_with[idx, tgt] - lp_without[idx, tgt])            # [A]

        claims = []
        for sent, tok_idx in self._sentence_token_index(answer_text, answer_ids):
            if not tok_idx:
                continue
            kls = [float(kl[j]) for j in tok_idx]
            sgn = [float(signed[j]) for j in tok_idx]
            claims.append({"text": sent,
                           "cti_mean": round(sum(kls) / len(kls), 4),
                           "cti_max": round(max(kls), 4),
                           "signed_mean": round(sum(sgn) / len(sgn), 4)})
        # Mean of the already-rounded claim means, then rounded — the exact
        # chain reliance_experiment.py used; averaging full-precision means
        # instead would miss the recorded value by ~1e-5 and muddy the gate.
        cti_mean = round(sum(c["cti_mean"] for c in claims) / len(claims), 4) if claims else 0.0
        signed_mean = round(sum(c["signed_mean"] for c in claims) / len(claims), 4) if claims else 0.0
        return {
            "cti_mean": cti_mean,
            "signed_mean": signed_mean,
            # Plain token-weighted means, reported as an aggregation sensitivity:
            # claim-means weight a 4-token sentence like a 40-token one.
            "token_cti_mean": round(float(kl.mean()), 4),
            "token_signed_mean": round(float(signed.mean()), 4),
            "logp_with": round(float(lp_with[idx, tgt].sum()), 4),
            "logp_without": round(float(lp_without[idx, tgt].sum()), 4),
            "claims": claims,
            "fmt": fmt,
            "n_answer_tokens": int(answer_ids.shape[1]),
            "lp_without": lp_without,   # caller reuses across context columns
        }

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
                  max_new_tokens: int = 256, cci_per_sentence: bool = True) -> MirageResult:
        """Full intrinsic attribution for one (question, passages)."""
        import torch

        ids_ctx, spans, fmt = self._build_prompt_ids(question, passages, with_context=True)
        # Both prompts must share one format: if the with-context build fell back
        # to plain, a chat-templated no-context prompt would make CTI measure the
        # template difference, not the passages.
        ids_noctx, _, fmt_noctx = self._build_prompt_ids(
            question, passages, with_context=False, force_plain=(fmt == "plain"))
        if fmt_noctx != fmt:  # template failed only on the no-context side — rare
            ids_ctx, spans, fmt = self._build_prompt_ids(
                question, passages, with_context=True, force_plain=True)

        answer_ids = self._generate(ids_ctx, max_new_tokens)
        # P1: drop a trailing EOS so it doesn't leak into CTI / sentence mapping.
        if (answer_ids.shape[1] > 0 and self.tok.eos_token_id is not None
                and int(answer_ids[0, -1]) == self.tok.eos_token_id):
            answer_ids = answer_ids[:, :-1]
        raw = self.tok.decode(answer_ids[0], skip_special_tokens=True)
        answer = raw.strip()

        passage_tags = {f"P{i}": {"chunk_id": p.get("chunk_id"), "title": p.get("title")}
                        for i, p in enumerate(passages, 1)}
        if answer_ids.shape[1] == 0:  # degenerate empty generation — caller flags it
            return MirageResult(question=question, answer="", token_cti=[],
                                spans=[], passage_tags=passage_tags, prompt_format=fmt)

        # CTI: KL(with || without) per answer token.
        lp_with = self._answer_logprobs(ids_ctx, answer_ids)
        lp_without = self._answer_logprobs(ids_noctx, answer_ids)
        p_with = lp_with.exp()
        kl = (p_with * (lp_with - lp_without)).sum(-1)  # [A]
        token_cti = [(self.tok.decode(answer_ids[0, j:j + 1]).strip(), float(kl[j]))
                     for j in range(answer_ids.shape[1])]

        # Map tokens → sentences via char offsets. B1: _token_char_offsets builds
        # offsets from the unstripped cumulative decode, but sentences are located
        # in the stripped answer — _sentence_token_index reconciles the two
        # coordinate systems, else boundary tokens get mis-assigned. That mapping
        # is shared with cti_fixed_answer so the generated and teacher-forced
        # paths aggregate identically.
        full_ctx = torch.cat([ids_ctx, answer_ids], dim=1)
        Lp = ids_ctx.shape[1]

        span_attrs = []
        for sent, tok_idx in self._sentence_token_index(raw, answer_ids):
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
                            spans=span_attrs, passage_tags=passage_tags, prompt_format=fmt)
