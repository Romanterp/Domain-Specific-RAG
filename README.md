# Retrieval Quality and Answer Attribution in Domain-Specific RAG

Code, corpus and analysis for an MSc Information Science thesis (University of Groningen, 2026)
on whether better retrieval changes what a generator actually relies on.

Everything runs on a fully open stack — **OLMo 3.1 32B Instruct** for generation, **BGE-M3** for
embedding, **bge-reranker-v2-m3** for reranking — because the attribution measures need the
model's token distributions and a searchable copy of its training data. A closed model supports
neither.

## What is here

A RAG system over **3,065 UNDRR publications** scraped for this thesis (84,595 chunks, 80,765
indexed), a frozen evaluation set of **2,481 questions** each generated from a known gold passage,
and two attribution measures applied to the generated answers.

| RQ | Question | Code | Results |
|---|---|---|---|
| RQ1 | What does each retrieval component contribute? | `retrieval/{search,hybrid,rerank,eval_synthetic}.py` | dense 23.7% → full pipeline 55.8% Hit@1 |
| RQ2 | Does doc2query help on this corpus? | `retrieval/{doc2query,eval_doc2query,sweep_doc2query_filter}.py` | null end-to-end; filtering hurts |
| RQ3 | Does better retrieval change context reliance? | `attribution/{select_contrast_set,reliance_experiment,reliance_analysis}.py` | refusals 45→4; CTI +0.117 [−0.002, +0.227] |
| RQ3 | Does reliance track faithfulness? | `attribution/{build_2x2,olmotrace,annotation_calibration}.py` | no — AUC 0.342 on the annotated sample |

## Layout

```
scraper/      catalogue discovery and file download
extraction/   text extraction, documents.json, heuristic metadata enrichment
retrieval/    chunking, embedding, BM25, fusion, reranking, evaluation, doc2query
attribution/  contrast set, CTI + leave-one-out, OLMoTrace, the 2x2, annotation workbench
scripts/      figure generation and Habrok (SLURM) job scripts
thesis/       chapter drafts and the results-number register
data/         corpus artefacts; large binaries are gitignored (see below)
```

## Install

```bash
python -m venv .venv311          # Python 3.11 — 3.14 has no CUDA wheels
.venv311/Scripts/pip install -r requirements.txt
```

`requirements-habrok.txt` is the cluster environment for the 32B jobs;
`requirements-311-local.txt` is a pinned freeze of the local environment.

**Gotcha:** BGE-M3 must be loaded with `revision="refs/pr/130"` on torch 2.5 / transformers 5.x —
CVE-2025-32434 blocks `.bin` loading and the default revision has no safetensors.

## Pipeline order

```
scraper.discover → scraper.download → extraction.text_extract → extraction.build_documents
→ extraction.heuristic_enrich → retrieval.chunk → retrieval.embed → retrieval.bm25
→ retrieval.generate_questions → retrieval.eval_synthetic        # RQ1
→ retrieval.doc2query → retrieval.eval_doc2query                 # RQ2
→ attribution.select_contrast_set → attribution.reliance_experiment
→ attribution.build_2x2 → attribution.annotation_calibration     # RQ3
```

Each stage is resumable and writes to `data/`. The 32B stages (question generation, the
full-corpus expansions, the reliance run) were executed on Habrok, the university cluster; see
`scripts/habrok/`.

## Reproducing the numbers

- Sampling steps run under recorded seeds and generation decodes greedily, so the reliance run
  reproduces exactly. Question generation sampled at temperature 0.7 and cannot be regenerated
  bit-for-bit, so every intermediate artefact is committed instead: question sets, holdout lists,
  the contrast-set assignment, generated answers with their scores, and the cached trace responses.
- Confidence intervals throughout use a **cluster bootstrap over the 500 gold passages**
  (B = 2000, seed 42), not over questions — questions come in groups of ~5 per passage with an
  intraclass correlation around 0.2 (`retrieval/bootstrap_ci.py`).

## Data

Committed: `chunks.jsonl`, the evaluation question sets, `documents.json`, `catalog.json`, the
contrast set, the reliance records, the 2x2 spans, the annotation sample, and the OLMoTrace
response cache (so the tracing analysis replays offline without contacting the service).

Not committed: the PDF corpus (~8.6 GB), extracted texts, the Qdrant collections and the BM25
pickles. Rebuild them by running the pipeline from `scraper.discover`.

## External services

OLMoTrace is reached through the Ai2 Playground attribution backend, which is **undocumented and
may disappear**. Every response is cached under `data/attribution/_olmotrace_cache/`, so the
analysis replays from cache. The public infini-gram API is the documented fallback, with the
caveat that it indexes OLMo 2 / Dolma rather than OLMo 3.

## Licence and attribution

The UNDRR publications are the property of their publishers and are redistributed here only as
derived artefacts (chunk text and embeddings) for research reproducibility. The code is released
for academic use.
