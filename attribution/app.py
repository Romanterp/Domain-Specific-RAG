"""
Theisus answer-provenance viewer (Streamlit).

For a question → OLMo answer, shows TWO independent things per claim:
  1. Support strength — how well (and by how many sources) the UNDRR corpus
     backs the claim. Drives the highlight colour.
  2. Provenance — where the claim comes from: the corpus (RAG), the model's
     general/parametric knowledge, or neither (unverified). A separate marker.

Click a claim in the inspector to see its actual supporting passages.

Reads cached example JSONs from data/attribution/*.json (built by
attribution/build_examples.py, or the shipped mock). Loads NO models.

Run
---
    streamlit run attribution/app.py
"""

import json
import sys
from pathlib import Path

# Streamlit runs this file as a script, so the repo root isn't on sys.path by
# default — add it so `attribution.pipeline` imports cleanly from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from attribution.pipeline import (
    SUPPORT_COLORS,
    SUPPORT_ORDER,
    SUPPORT_HELP,
    PROVENANCE_HELP,
    PROVENANCE_COLORS,
    support_bucket,
    provenance_for,
)

ATTR_DIR = Path(__file__).resolve().parent.parent / "data" / "attribution"

CSS = """
<style>
  .block-container {padding-top: 2.5rem; max-width: 1100px;}
  .ap-question {font-size: 1.55rem; font-weight: 650; line-height: 1.35;
                color: #1a1a1a; margin: .2rem 0 .4rem;}
  .ap-answer {line-height: 2.25; font-size: 1.05rem;}
  .ap-chip {padding: 2px 9px; border-radius: 11px; font-size: .8rem; margin-right: 6px;}
  .ap-span {padding: 2px 3px; border-radius: 4px;}
  .ap-idx {font-size: .62rem; vertical-align: super; color: #555; margin-right: 1px;}
  .ap-legend {font-size: .82rem; color: #444; margin-bottom: .4rem;}
</style>
"""


def load_examples() -> dict[str, dict]:
    out = {}
    if not ATTR_DIR.exists():
        return out
    for p in sorted(ATTR_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out[data.get("id", p.stem)] = data
        except Exception as e:  # noqa: BLE001 — surface bad files in the UI
            st.sidebar.error(f"Could not load {p.name}: {e}")
    return out


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def span_support(sp: dict) -> str:
    return sp.get("support") or support_bucket(sp.get("support_score", 0.0))


def span_provenance(sp: dict) -> str:
    return sp.get("provenance") or provenance_for(span_support(sp), sp.get("parametric"))


def render_answer(spans: list[dict]) -> str:
    """Highlight = SUPPORT strength. Markers = provenance for unsupported spans:
    blue dotted = general knowledge (parametric); red dashed = unverified."""
    parts = []
    for i, sp in enumerate(spans, 1):
        support = span_support(sp)
        prov = span_provenance(sp)
        bg = SUPPORT_COLORS.get(support, "#eef0f2")
        extra = ""
        if support == "none" and prov == "parametric":
            extra = "border-bottom:2px dotted #1565c0;"
        elif prov == "unverified":
            bg = "#f8d7da"
            extra = "border-bottom:2px dashed #c62828;"
        tip = (f"support: {support} ({sp.get('support_score',0):.2f}) · "
               f"{sp.get('n_sources',0)} source(s) · provenance: {prov}")
        parts.append(
            f"<span class='ap-idx'>{i}</span>"
            f"<span class='ap-span' title='{esc(tip)}' style='background:{bg};{extra}'>"
            f"{esc(sp.get('text',''))}</span>"
        )
    return "<div class='ap-answer'>" + " ".join(parts) + "</div>"


def render_legend(has_olmotrace: bool) -> None:
    sup = " ".join(
        f"<span class='ap-chip' style='background:{SUPPORT_COLORS[s]}'>{s}</span>"
        for s in SUPPORT_ORDER
    )
    line1 = f"<b>Highlight</b> = how well the corpus supports the claim: &nbsp;{sup}"
    if has_olmotrace:
        marks = (
            "<span style='border-bottom:2px dotted #1565c0;'>general knowledge</span>"
            " &nbsp;&nbsp; "
            "<span style='border-bottom:2px dashed #c62828;background:#f8d7da;"
            "padding:0 3px;border-radius:3px;'>unverified</span>"
        )
        line2 = f"<b>Underline</b> = where it comes from: &nbsp;{marks}"
    else:
        line2 = ("<b>Underline</b> = provenance — appears once OLMoTrace is "
                 "wired <span style='color:#777'>(Phase 2)</span>")
    st.markdown(
        f"<div class='ap-legend'>{line1}<br>{line2}</div>", unsafe_allow_html=True
    )


def prov_badge(prov: str) -> str:
    color = PROVENANCE_COLORS.get(prov, "#555")
    return f"<span style='color:{color};font-weight:650'>{prov}</span>"


def main() -> None:
    st.set_page_config(page_title="Theisus · Answer Provenance", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    examples = load_examples()
    if not examples:
        st.warning(f"No example files in `{ATTR_DIR}`. Run "
                   "`attribution/build_examples.py`, or check the mock example exists.")
        return

    with st.sidebar:
        st.header("Example")
        key = st.selectbox("Question", list(examples.keys()),
                           format_func=lambda k: examples[k].get("question", k)[:80])
        ex = examples[key]
        st.markdown(f"**Pipeline:** `{ex.get('pipeline','?')}`")
        st.markdown(f"**Generator:** `{ex.get('model','?')}`")
        st.markdown(f"**Generated:** {ex.get('generated_at','?')}")
        spans = ex.get("spans", [])
        has_olmotrace = any(sp.get("parametric") is not None for sp in spans)
        st.markdown("**OLMoTrace:** " + ("✅ present" if has_olmotrace else "⏳ Phase 2"))

    st.title("Answer Provenance Viewer")

    # ---- 1. Question first ----
    st.markdown(f"<div class='ap-question'>{esc(ex.get('question',''))}</div>",
                unsafe_allow_html=True)
    if ex.get("is_mock"):
        st.caption("🧪 mock example — illustrative scores, not measured.")

    # ---- 2. Overall corpus support ----
    n = len(spans) or 1
    n_corpus = sum(1 for sp in spans if span_support(sp) != "none")
    n_param = sum(1 for sp in spans if span_provenance(sp) == "parametric")
    n_unver = sum(1 for sp in spans if span_provenance(sp) == "unverified")
    score = ex.get("support_score", 0.0)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Corpus support", f"{score*100:.0f}%",
                  help="Mean per-claim support confidence across the answer.")
        st.progress(min(max(score, 0.0), 1.0))
    with right:
        st.markdown(
            f"**{n_corpus}/{n}** claims backed by the corpus &nbsp;·&nbsp; "
            f"**{n_param}** general knowledge &nbsp;·&nbsp; "
            f"**{n_unver}** unverified"
        )
        if n_unver:
            st.warning(f"{n_unver} claim(s) supported by **neither** the corpus nor "
                       "the model's training data — hallucination candidates.")
        elif n_param and not has_olmotrace:
            st.caption("Run OLMoTrace (Phase 2) to split unsupported claims into "
                       "general-knowledge vs unverified.")

    st.divider()

    # ---- 3. Highlighted answer (left) + claim inspector (right) ----
    st.markdown("#### Answer")
    render_legend(has_olmotrace)
    col_ans, col_insp = st.columns([3, 2], gap="large")

    with col_ans:
        st.markdown(render_answer(spans), unsafe_allow_html=True)

    with col_insp:
        st.markdown("**Inspect a claim** — pick one to see its sources")
        idx = st.selectbox(
            "Claim", range(len(spans)),
            format_func=lambda i: f"Claim {i+1}",
            label_visibility="collapsed",
        )
        sp = spans[idx]
        support = span_support(sp)
        prov = span_provenance(sp)
        st.markdown(
            f"<div style='font-size:.95rem;font-style:italic;color:#333;"
            f"border-left:3px solid {SUPPORT_COLORS.get(support,'#ccc')};"
            f"padding-left:.6rem;margin:.2rem 0 .7rem'>{esc(sp.get('text',''))}</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Support", support.title(), help=SUPPORT_HELP.get(support, ""))
        m2.metric("Confidence", f"{sp.get('support_score',0):.2f}")
        m3.metric("Sources", sp.get("n_sources", 0))
        st.markdown(f"Provenance: {prov_badge(prov)}", unsafe_allow_html=True)
        st.caption(PROVENANCE_HELP.get(prov, ""))

        sources = sp.get("sources", [])
        if sources:
            passages_by_id = {p.get("chunk_id"): p for p in ex.get("passages", [])}
            for s in sources:
                full = passages_by_id.get(s.get("chunk_id"), {})
                head = (f"score {s.get('score',0):.2f} · "
                        f"{s.get('title','(untitled)')} p.{s.get('page','?')}")
                with st.expander(head):
                    url = full.get("source_url")
                    if url:
                        st.markdown(f"[source document]({url}) · `{s.get('chunk_id')}`")
                    st.write(full.get("text", "(passage text unavailable)"))
        else:
            st.info("No corpus passage supports this claim above threshold.")
            if sp.get("dolma_matches"):
                st.markdown("**In training data (OLMoTrace):**")
                for mm in sp["dolma_matches"]:
                    st.markdown(f"- “{esc(mm.get('snippet',''))}” — *{esc(mm.get('doc',''))}*")

    st.divider()

    # ---- 5. All retrieved passages ----
    with st.expander(f"All retrieved passages ({len(ex.get('passages', []))})"):
        for p in ex.get("passages", []):
            st.markdown(
                f"**#{p.get('rank','?')}** · {p.get('title','(untitled)')} "
                f"· p.{p.get('page','?')} · rerank {p.get('rerank_score', float('nan')):.2f}  \n"
                f"`{p.get('chunk_id')}`"
            )
            st.write(p.get("text", ""))
            st.markdown("---")


if __name__ == "__main__":
    main()
