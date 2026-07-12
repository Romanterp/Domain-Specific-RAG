"""
Theisus thesis workbench (Streamlit) — annotation + dual-lens provenance explorer.

Three modes in one app (same JSONL contracts everywhere, so a later
FastAPI+React rewrite only replaces this file):

  ANNOTATE   Blind labeling of the frozen annotation sample
             (data/annotation_sample.jsonl -> data/annotation_labels.jsonl).
             Hides CTI, buckets, gold, condition and class on purpose: the
             labels calibrate CTI, so the annotator judges support only
             against the passages. Append-only labels; latest wins.

  EXPLORE    Record-centric dual-lens view over the 32B production run
             (data/reliance_records.jsonl + data/attribution_2x2.jsonl).
             The two retrieval conditions sit side by side — a rescued
             question SHOWS the thesis result. Visual grammar (one channel
             per axis, because the lenses differ in KIND):
               fill colour   = intrinsic reliance, CAUSAL, measured on this
                               generation (CTI: context/mixed/parametric)
               dotted under  = extrinsic verbatim overlap with training data
                               (OLMoTrace LOOKUP — overlap, not causation)
               dashed red    = parametric AND no trace -> "no visible source"
             The claim inspector separates the two provenance chains
             explicitly: RETRIEVED CONTEXT (causal) vs TRAINING DATA (lookup).

  DASHBOARD  Aggregate reliance x trace cross-tab with per-cell drill-down
             into Explore. Filterable by contrast class and condition.

Run
---
    .venv311/Scripts/python.exe -m streamlit run attribution/workbench.py
"""

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from attribution.build_annotation_sample import cti_bucket

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "annotation_sample.jsonl"
LABELS_PATH = DATA_DIR / "annotation_labels.jsonl"
RECORDS_PATH = DATA_DIR / "reliance_records.jsonl"
TWO_BY_TWO_PATH = DATA_DIR / "attribution_2x2.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCS_PATH = DATA_DIR / "documents.json"

CONDITIONS = ["dense", "hybrid_rerank"]
COND_LABEL = {"dense": "dense", "hybrid_rerank": "hybrid + rerank"}

SUPPORT_OPTIONS = {
    "supported": "Supported — the passages back the claim",
    "partially": "Partially — some of the claim is backed, some is not",
    "not_supported": "Not supported — the passages do not back the claim",
    "cant_tell": "Can't tell — claim is vague / passages unreadable",
}
ANSWER_OPTIONS = {
    "correct": "Correct — answers the question, consistent with the passages",
    "partially": "Partially correct",
    "incorrect": "Incorrect or contradicted by the passages",
    "cant_tell": "Can't tell",
}

CSS = """
<style>
  .block-container {padding-top: 2.0rem; max-width: 1250px;}
  .wb-question {font-size: 1.3rem; font-weight: 650; line-height: 1.35; margin: .2rem 0 .5rem;}
  .wb-answer {line-height: 2.0; font-size: .98rem; background:#fcfcfc;
              border:1px solid #eee; border-radius:8px; padding:12px 14px;}
  .wb-claim {border-radius:4px; padding:1px 3px;}
  .wb-context {background:#d6e6f7;}
  .wb-mixed {background:#e9e9ee;}
  .wb-parametric {background:#fde2b8;}
  .wb-hit {border-bottom:2px dotted #555;}
  .wb-unver {outline:2px dashed #d9534f;}
  .wb-sup {font-size:.65rem; vertical-align:super; color:#333; margin-right:1px;}
  .wb-meta {color:#666; font-size:.85rem;}
  .wb-gold {color:#b8860b; font-weight:600;}
  .wb-refusal {background:#f8d7da; border-radius:6px; padding:8px 12px; font-weight:600;}
  .wb-legend {font-size:.85rem; color:#333; line-height:1.9;}
  .wb-chip {padding:2px 9px; border-radius:11px; font-size:.8rem; background:#eef0f2; margin-right:6px;}
</style>
"""

_MD_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+.!$~<>|-])")


def md_escape(s: str) -> str:
    """Escape Markdown/KaTeX specials for widget labels rendered as Markdown."""
    return _MD_SPECIALS.sub(r"\\\1", s or "")


def read_jsonl(path: Path) -> list[dict]:
    """One JSON object per PHYSICAL line. split('\\n'), never splitlines():
    corpus text contains U+2028/U+2029, which splitlines() treats as breaks."""
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_sample(mtime: float) -> list[dict]:
    return read_jsonl(SAMPLE_PATH)


@st.cache_data
def load_records(mtime: float) -> dict:
    """q_idx -> {'question', 'contrast_class', conditions: {cond: record}}."""
    by_q: dict = {}
    for r in read_jsonl(RECORDS_PATH):
        e = by_q.setdefault(r["q_idx"], {"question": r["question"],
                                         "contrast_class": r.get("contrast_class"),
                                         "conditions": {}})
        e["conditions"][r["condition"]] = r
    return by_q


@st.cache_data
def load_trace(mtime: float) -> dict:
    """(q_idx, condition, span_idx) -> 2x2 row (pretraining_hit, matches, ...)."""
    return {(r.get("q_idx"), r.get("condition"), r.get("span_idx")): r
            for r in read_jsonl(TWO_BY_TWO_PATH)}


@st.cache_resource(show_spinner="Loading passage texts (one-time, ~15s)…")
def load_passage_texts() -> tuple[dict, dict]:
    """chunk_id -> {text,page,slug} for every chunk any record cites; slug -> title."""
    needed: set = set()
    for e in load_records(_mtime(RECORDS_PATH)).values():
        for rec in e["conditions"].values():
            needed.update(rec.get("passage_chunk_ids") or [])
            if rec.get("gold_chunk_id"):
                needed.add(rec["gold_chunk_id"])
    chunks: dict = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                c = json.loads(ln)
                if c["chunk_id"] in needed:
                    chunks[c["chunk_id"]] = c
                    if len(chunks) == len(needed):
                        break
    titles = {}
    if DOCS_PATH.exists():
        titles = {d.get("slug"): d.get("title") or d.get("slug")
                  for d in json.loads(DOCS_PATH.read_text(encoding="utf-8"))}
    return chunks, titles


def load_labels() -> list[dict]:
    return read_jsonl(LABELS_PATH)  # deliberately uncached — reread every rerun


def latest_labels(rows: list[dict], kind: str, annotator: str | None = None) -> dict:
    out: dict = {}
    for r in rows:  # chronological; later rows overwrite
        if r.get("kind") != kind:
            continue
        if annotator is not None and r.get("annotator") != annotator:
            continue
        out[r["key"]] = r
    return out


def append_label(row: dict) -> None:
    with open(LABELS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- rendering ----------

def simple_answer_html(answer: str, claim: str) -> str:
    """Annotate mode: answer with ONLY the current claim highlighted (blind)."""
    a, c = html.escape(answer), html.escape(claim)
    if c and c in a:
        a = a.replace(c, f'<span class="wb-claim wb-mixed">{c}</span>', 1)
    return f'<div class="wb-answer">{a.replace(chr(10), "<br>")}</div>'


def record_answer_html(rec: dict, trace: dict) -> str:
    """Explore mode: every claim encoded on both axes (fill=reliance, mark=trace)."""
    ans = rec.get("answer", "")
    segs, cur = [], 0
    for ci, c in enumerate(rec.get("claims") or []):
        txt = (c.get("text") or "").strip()
        i = ans.find(txt, cur) if txt else -1
        if i < 0:
            continue
        segs.append(html.escape(ans[cur:i]))
        bucket = cti_bucket(c.get("cti_mean") or 0.0)
        t = trace.get((rec["q_idx"], rec["condition"], ci))
        hit = t.get("pretraining_hit") if t else None
        cls = ["wb-claim", f"wb-{bucket}"]
        if hit:
            cls.append("wb-hit")
        if bucket == "parametric" and hit is False:
            cls.append("wb-unver")
        tip = (f"claim {ci + 1} · CTI {c.get('cti_mean'):.2f} · reliance: {bucket} · "
               f"trace: {'verbatim match' if hit else ('no match' if hit is False else 'not traced')}")
        segs.append(f'<span class="{" ".join(cls)}" title="{html.escape(tip)}">'
                    f'<span class="wb-sup">{ci + 1}</span>{html.escape(txt)}</span>')
        cur = i + len(txt)
    segs.append(html.escape(ans[cur:]))
    return '<div class="wb-answer">' + "".join(segs).replace("\n", "<br>") + "</div>"


def render_passages(passages: list[dict], show_gold: bool) -> None:
    for i, p in enumerate(passages, 1):
        page = f", p. {p['page']}" if p.get("page") is not None else ""
        gold = "  ·  GOLD" if (show_gold and p.get("is_gold")) else ""
        with st.expander(md_escape(f"[P{i}] {p.get('title', p.get('chunk_id', '?'))}{page}{gold}")):
            st.text(p.get("text", ""))  # st.text: verbatim, no Markdown/KaTeX mangling


LEGEND = """
<div class="wb-legend">
<b>fill = the model's reliance</b> (causal, measured on this answer: KL with vs without context) —
<span class="wb-claim wb-context">context-driven</span>
<span class="wb-claim wb-mixed">mixed</span>
<span class="wb-claim wb-parametric">parametric</span><br>
<b>marks = training-data overlap</b> (verbatim lookup in OLMo's pretraining index — overlap, <i>not</i> causation) —
<span class="wb-claim wb-hit">dotted&nbsp;=&nbsp;match&nbsp;found</span>,
no mark = none found, hover a claim for "not traced" ·
<span class="wb-claim wb-parametric wb-unver">dashed&nbsp;red</span> = parametric <i>and</i> no trace → no visible source (hallucination signal)
</div>
"""


# ---------- ANNOTATE ----------

def annotate_mode(annotator: str) -> None:
    sample = load_sample(_mtime(SAMPLE_PATH))
    if not sample:
        st.error("No sample at data/annotation_sample.jsonl — "
                 "run: python -m attribution.build_annotation_sample")
        return
    labels = load_labels()
    span_done = latest_labels(labels, "span", annotator)
    answer_done = latest_labels(labels, "answer", annotator)

    todo = [s for s in sample if s["span_key"] not in span_done]
    st.sidebar.progress(1 - len(todo) / max(1, len(sample)),
                        text=f"{len(sample) - len(todo)} / {len(sample)} spans labeled")
    if not todo:
        st.success("All spans labeled — thank you. Explore mode shows results in full.")
        return

    s = todo[0]
    pos = sample.index(s)  # positional key: no condition/q_idx leaks into the DOM
    record_key = f"{s['q_idx']}:{s['condition']}"
    st.markdown('<div class="wb-meta">Blind mode: scores and provenance are hidden on '
                "purpose. Judge ONLY whether the highlighted claim is backed by the "
                "passages below.</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="wb-question">{html.escape(s["question"])}</div>',
                unsafe_allow_html=True)
    st.markdown(simple_answer_html(s["answer"], s["claim_text"]), unsafe_allow_html=True)
    st.markdown("**Claim to judge:**")
    st.info(s["claim_text"])
    st.markdown("**Retrieved passages** (open to read):")
    render_passages(s["passages"], show_gold=False)

    with st.form(key=f"annot-{pos}"):
        support = st.radio("Is the claim supported by the retrieved passages?",
                           options=list(SUPPORT_OPTIONS),
                           format_func=SUPPORT_OPTIONS.get, index=None)
        note = st.text_input("Note (optional)")
        ans_label = None
        if record_key not in answer_done:
            ans_label = st.radio(
                "Whole-answer check (asked once per answer): does the full answer "
                "correctly answer the question?",
                options=["(skip)"] + list(ANSWER_OPTIONS),
                format_func=lambda k: ANSWER_OPTIONS.get(k, k), index=0)
        submitted = st.form_submit_button("Save & next")

    if submitted:
        if support is None:
            st.warning("Pick a support label first.")
            return
        ts = now_iso()
        append_label({"ts": ts, "annotator": annotator, "kind": "span",
                      "key": s["span_key"], "q_idx": s["q_idx"],
                      "condition": s["condition"], "claim_idx": s["claim_idx"],
                      "label": support, "note": note or None})
        if ans_label is not None:
            # persist "(skip)" as skipped so the question is truly asked once
            append_label({"ts": ts, "annotator": annotator, "kind": "answer",
                          "key": record_key, "q_idx": s["q_idx"],
                          "condition": s["condition"],
                          "label": "skipped" if ans_label == "(skip)" else ans_label,
                          "note": None})
        st.rerun()


# ---------- EXPLORE ----------

def condition_column(rec: dict | None, trace: dict) -> None:
    if rec is None:
        st.markdown('<div class="wb-meta">no record for this condition</div>',
                    unsafe_allow_html=True)
        return
    gold = (f"gold: rank {rec['gold_rank']}" if rec.get("gold_in_context")
            else "gold: NOT in context")
    loo = rec.get("gold_loo_drop")
    loo_s = f" · gold LOO drop {loo:.1f}" if loo is not None else ""
    st.markdown(f'<div class="wb-meta">{gold}{loo_s}</div>', unsafe_allow_html=True)
    if rec.get("refusal"):
        st.markdown('<div class="wb-refusal">REFUSAL — the model declined to answer</div>',
                    unsafe_allow_html=True)
    if rec.get("answer_empty"):
        st.markdown('<div class="wb-meta">(empty generation)</div>', unsafe_allow_html=True)
        return
    st.markdown(record_answer_html(rec, trace), unsafe_allow_html=True)


def claim_inspector(rec: dict, ci: int, trace: dict, chunks: dict, titles: dict,
                    annotator: str) -> None:
    claim = rec["claims"][ci]
    bucket = cti_bucket(claim.get("cti_mean") or 0.0)
    t = trace.get((rec["q_idx"], rec["condition"], ci))
    hit = t.get("pretraining_hit") if t else None

    labels = load_labels()
    span_key = f"{rec['q_idx']}:{rec['condition']}:{ci}"
    human = latest_labels(labels, "span").get(span_key)  # any annotator

    chips = [f'<span class="wb-chip">reliance: {bucket}</span>',
             f'<span class="wb-chip">trace: '
             f'{"verbatim match" if hit else ("no match" if hit is False else "not traced")}</span>']
    if t:
        chips.append(f'<span class="wb-chip">cell: {t.get("label_2x2")}</span>')
    if human:
        chips.append(f'<span class="wb-chip">human: {human["label"]} ({human["annotator"]})</span>')
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.info(claim.get("text", ""))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Retrieved context — causal (this generation)")
        st.markdown(f'<div class="wb-meta">CTI mean {claim.get("cti_mean"):.3f} · '
                    f'max {claim.get("cti_max"):.3f} → <b>{bucket}</b></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="wb-meta">Per-claim passage attribution was not '
                    "measured in this run (CCI off); the gold LOO drop above is "
                    "answer-level. Passages shown in prompt order.</div>",
                    unsafe_allow_html=True)
        passages = []
        for cid in rec.get("passage_chunk_ids") or []:
            c = chunks.get(cid)
            passages.append({
                "chunk_id": cid,
                "title": titles.get(c["slug"], cid.split(":")[0]) if c else cid,
                "page": c.get("page") if c else None,
                "text": c.get("text", "(chunk text not available)") if c else "(chunk text not available)",
                "is_gold": cid == rec.get("gold_chunk_id"),
            })
        render_passages(passages, show_gold=True)
    with right:
        st.markdown("#### Training data — verbatim overlap (lookup)")
        if t is None:
            st.markdown('<div class="wb-meta">Not traced yet — run '
                        "<code>python -m attribution.build_2x2</code> over the "
                        "production records.</div>", unsafe_allow_html=True)
        elif hit is None:
            st.markdown('<div class="wb-meta">Trace unavailable for this answer '
                        "(API failure at build time).</div>", unsafe_allow_html=True)
        else:
            matches = t.get("pretraining_matches")
            if matches:
                for m in matches:
                    corp = ", ".join(m.get("corpora") or []) or "corpus n/a"
                    st.markdown(f'- "{md_escape(m["raw"])}" '
                                f'<span class="wb-meta">({corp} · {m.get("n_docs", "?")} docs)</span>',
                                unsafe_allow_html=True)
            elif hit:
                corp = ", ".join(t.get("olmotrace_corpora") or []) or "corpus n/a"
                st.markdown(f'<div class="wb-meta">{t.get("n_pretraining_spans")} matched '
                            f"span(s) · {corp} — matched substrings not stored in this "
                            "2x2 build; re-run build_2x2 --no-live to embed them (cache-only)."
                            "</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="wb-meta">No verbatim match ≥16 chars / ≥3 tokens '
                            "found in the pretraining index.</div>", unsafe_allow_html=True)
        st.markdown('<div class="wb-meta">⚠ Overlap ≠ source: a match says this phrasing '
                    "exists in OLMo's training data, not that the model drew on it. "
                    "Unofficial playground endpoint; usage==Pre-training matches only."
                    "</div>", unsafe_allow_html=True)

    # relabel from Explore (any span; sample spans join the calibration set)
    with st.form(key=f"explore-label-{span_key}"):
        support = st.radio("Human label (optional, appended to annotation_labels.jsonl)",
                           options=list(SUPPORT_OPTIONS),
                           format_func=SUPPORT_OPTIONS.get, index=None)
        note = st.text_input("Note (optional)", key=f"note-{span_key}")
        if st.form_submit_button("Save label") and support:
            append_label({"ts": now_iso(), "annotator": annotator, "kind": "span",
                          "key": span_key, "q_idx": rec["q_idx"],
                          "condition": rec["condition"], "claim_idx": ci,
                          "label": support, "note": note or None})
            st.rerun()


def explore_mode(annotator: str) -> None:
    by_q = load_records(_mtime(RECORDS_PATH))
    if not by_q:
        st.error("No production records at data/reliance_records.jsonl.")
        return
    trace = load_trace(_mtime(TWO_BY_TWO_PATH))
    chunks, titles = load_passage_texts()

    classes = sorted({e["contrast_class"] for e in by_q.values() if e["contrast_class"]})
    f_class = st.sidebar.multiselect("Class", classes, default=classes)
    search = st.sidebar.text_input("Search question text")
    qs = [q for q, e in sorted(by_q.items())
          if e["contrast_class"] in f_class
          and (not search or search.lower() in e["question"].lower())]
    if not qs:
        st.info("No questions match the filters.")
        return
    default_q = st.session_state.pop("explore_q", None)
    idx = qs.index(default_q) if default_q in qs else 0
    q_idx = st.sidebar.selectbox(
        "Question", qs, index=idx,
        format_func=lambda q: f"q{q} · {by_q[q]['contrast_class']} · {by_q[q]['question'][:60]}")

    e = by_q[q_idx]
    st.markdown(f'<span class="wb-chip">q{q_idx}</span>'
                f'<span class="wb-chip">{e["contrast_class"]}</span>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="wb-question">{html.escape(e["question"])}</div>',
                unsafe_allow_html=True)
    with st.expander("Legend — how to read the two axes", expanded=False):
        st.markdown(LEGEND, unsafe_allow_html=True)

    cols = st.columns(2)
    for col, cond in zip(cols, CONDITIONS):
        with col:
            st.markdown(f"##### {COND_LABEL[cond]}")
            condition_column(e["conditions"].get(cond), trace)

    # claim inspector
    options = []
    for cond in CONDITIONS:
        rec = e["conditions"].get(cond)
        if rec and not rec.get("answer_empty") and not rec.get("refusal"):
            for ci, c in enumerate(rec.get("claims") or []):
                options.append((cond, ci, (c.get("text") or "")[:70]))
    if options:
        st.markdown("---")
        sel = st.selectbox("Inspect claim", options,
                           format_func=lambda o: f"{COND_LABEL[o[0]]} · #{o[1] + 1} · {o[2]}")
        claim_inspector(e["conditions"][sel[0]], sel[1], trace, chunks, titles, annotator)


# ---------- DASHBOARD ----------

def trace_status(row: dict) -> str:
    h = row.get("pretraining_hit")
    return "match" if h else ("no match" if h is False else "untraced")


def dashboard_mode() -> None:
    rows = list(load_trace(_mtime(TWO_BY_TWO_PATH)).values())
    if not rows:
        st.error("No 2x2 rows at data/attribution_2x2.jsonl — run "
                 "python -m attribution.build_2x2 first.")
        return
    models = sorted({r.get("model") or "?" for r in rows})
    st.markdown(f'<div class="wb-meta">{len(rows)} claim spans · model(s): '
                f'{", ".join(models)}</div>', unsafe_allow_html=True)

    classes = sorted({r.get("contrast_class") for r in rows if r.get("contrast_class")})
    f_class = st.sidebar.multiselect("Class", classes, default=classes)
    f_cond = st.sidebar.multiselect("Condition", CONDITIONS, default=CONDITIONS)
    rows = [r for r in rows
            if r.get("contrast_class") in f_class and r.get("condition") in f_cond]

    rel_order = ["context", "mixed", "parametric"]
    tr_order = ["match", "no match", "untraced"]
    counts = {(rl, tr): 0 for rl in rel_order for tr in tr_order}
    for r in rows:
        counts[(r.get("reliance"), trace_status(r))] = \
            counts.get((r.get("reliance"), trace_status(r)), 0) + 1
    total = max(1, len(rows))
    lines = ["| reliance \\ trace | " + " | ".join(tr_order) + " |",
             "|---|" + "---|" * len(tr_order)]
    for rl in rel_order:
        cells = [f"{counts[(rl, tr)]} ({100 * counts[(rl, tr)] / total:.0f}%)" for tr in tr_order]
        lines.append(f"| **{rl}** | " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))
    st.markdown('<div class="wb-meta">parametric × no-match = the "no visible source" '
                "cell (hallucination signal — pending human calibration).</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        rl = st.selectbox("Drill into reliance", rel_order, index=2)
    with c2:
        tr = st.selectbox("… × trace", tr_order, index=1)
    cell = [r for r in rows if r.get("reliance") == rl and trace_status(r) == tr]
    st.markdown(f"**{len(cell)} spans in {rl} × {tr}** (showing up to 50)")
    for r in cell[:50]:
        bcol, tcol = st.columns([1, 9])
        with bcol:
            if st.button("view", key=f"dd-{r['q_idx']}-{r['condition']}-{r['span_idx']}"):
                st.session_state["explore_q"] = r["q_idx"]
                st.session_state["pending_mode"] = "Explore"
                st.rerun()
        with tcol:
            st.markdown(f'<div class="wb-meta">q{r["q_idx"]} · {r["condition"]} · '
                        f'CTI {r.get("cti_mean"):.2f} · {html.escape((r.get("text") or "")[:110])}</div>',
                        unsafe_allow_html=True)


# ---------- main ----------

def main() -> None:
    st.set_page_config(page_title="Theisus workbench", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if "pending_mode" in st.session_state:  # set by dashboard drill-down buttons
        st.session_state["mode"] = st.session_state.pop("pending_mode")

    st.sidebar.title("Theisus workbench")
    annotator = st.sidebar.text_input("Annotator",
                                      value=st.session_state.get("annotator", ""))
    st.session_state["annotator"] = annotator
    mode = st.sidebar.radio("Mode", ["Annotate", "Explore", "Dashboard"], key="mode")
    st.sidebar.markdown("---")

    if mode == "Annotate":
        st.title("Claim-support annotation")
        if not annotator:
            st.info("Enter an annotator name in the sidebar to begin.")
            return
        annotate_mode(annotator)
    elif mode == "Explore":
        st.title("Dual-lens provenance explorer")
        explore_mode(annotator or "explorer")
    else:
        st.title("Reliance × training-data trace")
        dashboard_mode()


if __name__ == "__main__":
    main()
