"""
Export the provenance viewer to a SINGLE self-contained HTML file.

Reads the cached example JSONs from data/attribution/*.json and emits one
standalone .html with all data, CSS, and JS inlined — no Streamlit, no server

The highlighted answer spans are clickable: click a claim to see its sources.

Usage
-----
    .venv311/Scripts/python.exe -m attribution.export_html
    .venv311/Scripts/python.exe -m attribution.export_html --out provenance.html
"""

import argparse
import json
import sys
from pathlib import Path

ATTR_DIR = Path(__file__).resolve().parent.parent / "data" / "attribution"
DEFAULT_OUT = ATTR_DIR / "provenance_report.html"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theisus · Answer Provenance</title>
<style>
  :root { --green:#c8e6c9; --amber:#fff3cd; --grey:#eef0f2; --red-bg:#f8d7da; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color:#1a1a1a; max-width:1080px; margin:0 auto; padding:28px 22px 60px; line-height:1.5; }
  h1 { font-size:1.7rem; margin:0 0 .2rem; }
  .sub { color:#666; margin:0 0 1.2rem; }
  select { font-size:1rem; padding:5px 8px; border-radius:6px; border:1px solid #ccc; }
  .question { font-size:1.5rem; font-weight:650; line-height:1.35; margin:1rem 0 .3rem; }
  .mock { color:#777; font-size:.85rem; margin-bottom:1rem; }
  .summary { background:#f7f8fa; border:1px solid #eceef1; border-radius:10px;
             padding:14px 16px; margin:.6rem 0 1.2rem; }
  .score { font-size:2rem; font-weight:700; }
  .bar { height:9px; background:#e3e6ea; border-radius:5px; overflow:hidden; margin:.4rem 0; }
  .bar > div { height:100%; background:#3b82f6; }
  .legend { font-size:.85rem; color:#444; margin:.3rem 0 .8rem; line-height:1.9; }
  .chip { padding:2px 9px; border-radius:11px; font-size:.8rem; margin-right:5px; }
  .layout { display:flex; gap:26px; align-items:flex-start; }
  .answer { flex:3; line-height:2.2; font-size:1.05rem; }
  .panel { flex:2; position:sticky; top:14px; background:#fafbfc; border:1px solid #eceef1;
           border-radius:10px; padding:14px 16px; min-height:120px; }
  .span { padding:2px 3px; border-radius:4px; cursor:pointer; }
  .span:hover { outline:2px solid #94a3b8; }
  .span.sel { outline:2px solid #1f2937; }
  .idx { font-size:.62rem; vertical-align:super; color:#666; margin-right:1px; }
  .param { border-bottom:2px dotted #1565c0; }
  .unver { background:var(--red-bg); border-bottom:2px dashed #c62828; }
  .metrics { display:flex; gap:14px; margin:.6rem 0; }
  .metric .k { font-size:.72rem; color:#666; text-transform:uppercase; letter-spacing:.03em; }
  .metric .v { font-size:1.15rem; font-weight:650; }
  .prov { font-weight:650; }
  .src { border:1px solid #e6e8eb; border-radius:8px; padding:8px 10px; margin:.4rem 0; background:#fff; }
  .src .h { font-size:.85rem; color:#333; font-weight:600; }
  .src .t { font-size:.85rem; color:#444; margin-top:4px; }
  .warn { background:#fff4f4; border:1px solid #f3c2c2; color:#9b1c1c; border-radius:8px;
          padding:8px 12px; font-size:.9rem; margin-top:.5rem; }
  a { color:#1565c0; }
  .claimtext { font-style:italic; color:#333; border-left:3px solid #ccc; padding-left:.6rem; margin:.2rem 0 .6rem; }
</style>
</head>
<body>
  <h1>Answer Provenance Viewer</h1>
  <p class="sub">How well does the UNDRR corpus support each claim — and where does it come from? (RAG vs the model's general knowledge)</p>
  <div id="picker"></div>
  <div id="view"></div>

<script>
const SUPPORT_COLORS = {strong:"#c8e6c9", partial:"#fff3cd", none:"#eef0f2"};
const PROV_COLORS = {corpus:"#2e7d32", parametric:"#1565c0", unverified:"#c62828", uncorroborated:"#777"};
const PROV_HELP = {
  corpus:"Backed by your UNDRR corpus (RAG).",
  parametric:"Not in your corpus, but in the model's training data — general knowledge.",
  unverified:"Not in the corpus and not in training data — a hallucination candidate.",
  uncorroborated:"Not found in the corpus; OLMoTrace not yet run."
};
const EXAMPLES = /*__DATA__*/;
let curKey = Object.keys(EXAMPLES)[0];
let curClaim = 0;

function esc(s){ return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function spanSupport(sp){ return sp.support || "none"; }
function spanProv(sp){ return sp.provenance || "uncorroborated"; }

function renderPicker(){
  const keys = Object.keys(EXAMPLES);
  if (keys.length < 2){ document.getElementById("picker").innerHTML=""; return; }
  let h = '<select onchange="curKey=this.value;curClaim=0;render()">';
  for (const k of keys) h += `<option value="${k}" ${k===curKey?'selected':''}>${esc(EXAMPLES[k].question)}</option>`;
  h += '</select>';
  document.getElementById("picker").innerHTML = h;
}

function answerHTML(spans){
  return spans.map((sp,i)=>{
    const sup = spanSupport(sp), prov = spanProv(sp);
    let style = `background:${SUPPORT_COLORS[sup]||'#eef0f2'};`;
    let cls = "span";
    if (sup==="none" && prov==="parametric") cls += " param";
    if (prov==="unverified"){ cls += " unver"; style=""; }
    if (i===curClaim) cls += " sel";
    return `<span class="idx">${i+1}</span><span class="${cls}" style="${style}" onclick="curClaim=${i};render()">${esc(sp.text)}</span>`;
  }).join(" ");
}

function legendHTML(hasOT){
  let sup = Object.keys(SUPPORT_COLORS).map(s=>`<span class="chip" style="background:${SUPPORT_COLORS[s]}">${s}</span>`).join(" ");
  let l1 = `<b>Highlight</b> = corpus support: &nbsp;${sup}`;
  let l2 = hasOT
    ? `<b>Underline</b> = provenance: &nbsp;<span class="param">general knowledge</span> &nbsp;&nbsp; <span class="unver" style="padding:0 3px;border-radius:3px;">unverified</span>`
    : `<b>Underline</b> = provenance — appears once OLMoTrace is wired (Phase 2)`;
  return `<div class="legend">${l1}<br>${l2}</div>`;
}

function panelHTML(ex){
  const sp = ex.spans[curClaim]; if(!sp) return "";
  const sup = spanSupport(sp), prov = spanProv(sp);
  let h = `<div class="claimtext" style="border-left-color:${SUPPORT_COLORS[sup]||'#ccc'}">${esc(sp.text)}</div>`;
  h += `<div class="metrics">
    <div class="metric"><div class="k">Support</div><div class="v">${sup}</div></div>
    <div class="metric"><div class="k">Confidence</div><div class="v">${(sp.support_score||0).toFixed(2)}</div></div>
    <div class="metric"><div class="k">Sources</div><div class="v">${sp.n_sources||0}</div></div>
  </div>`;
  h += `<div>Provenance: <span class="prov" style="color:${PROV_COLORS[prov]}">${prov}</span></div>`;
  h += `<div style="font-size:.85rem;color:#666;margin:.2rem 0 .6rem">${PROV_HELP[prov]||""}</div>`;
  const byId = {}; (ex.passages||[]).forEach(p=>byId[p.chunk_id]=p);
  if ((sp.sources||[]).length){
    for (const s of sp.sources){
      const full = byId[s.chunk_id]||{};
      const url = full.source_url ? ` · <a href="${full.source_url}" target="_blank">source</a>` : "";
      h += `<div class="src"><div class="h">score ${(s.score||0).toFixed(2)} · ${esc(s.title)} p.${s.page}${url}</div>
            <div class="t">${esc(full.text||"")}</div></div>`;
    }
  } else {
    h += `<div class="warn">No corpus passage supports this claim above threshold.</div>`;
    if ((sp.dolma_matches||[]).length){
      h += `<div style="font-size:.85rem;margin-top:.5rem"><b>In training data (OLMoTrace):</b><ul>`;
      for (const m of sp.dolma_matches) h += `<li>“${esc(m.snippet)}” — <i>${esc(m.doc)}</i></li>`;
      h += `</ul></div>`;
    }
  }
  return h;
}

function render(){
  renderPicker();
  const ex = EXAMPLES[curKey];
  const spans = ex.spans||[];
  const hasOT = spans.some(s=>s.parametric!==null && s.parametric!==undefined);
  const n = spans.length||1;
  const nCorpus = spans.filter(s=>spanSupport(s)!=="none").length;
  const nParam = spans.filter(s=>spanProv(s)==="parametric").length;
  const nUnver = spans.filter(s=>spanProv(s)==="unverified").length;
  const score = ex.support_score||0;
  let h = `<div class="question">${esc(ex.question)}</div>`;
  if (ex.is_mock) h += `<div class="mock">🧪 mock example — illustrative scores, not measured.</div>`;
  h += `<div class="summary">
      <div class="score">${Math.round(score*100)}%<span style="font-size:.9rem;font-weight:400;color:#666"> corpus support</span></div>
      <div class="bar"><div style="width:${Math.round(score*100)}%"></div></div>
      <div style="font-size:.92rem"><b>${nCorpus}/${n}</b> claims backed by the corpus · <b>${nParam}</b> general knowledge · <b>${nUnver}</b> unverified</div>
      ${nUnver?`<div class="warn">${nUnver} claim(s) supported by neither the corpus nor training data — hallucination candidates.</div>`:""}
      <div style="font-size:.8rem;color:#888;margin-top:.5rem">pipeline: ${esc(ex.pipeline||"")} · generator: ${esc(ex.model||"")}</div>
    </div>`;
  h += legendHTML(hasOT);
  h += `<div class="layout"><div class="answer">${answerHTML(spans)}</div><div class="panel">${panelHTML(ex)}</div></div>`;
  document.getElementById("view").innerHTML = h;
}
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--dir", default=str(ATTR_DIR), help="folder of cached example JSONs")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = Path(args.dir)
    examples = {}
    for p in sorted(src.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            examples[data.get("id", p.stem)] = data
        except Exception as e:  # noqa: BLE001
            print(f"skip {p.name}: {e}", file=sys.stderr)

    if not examples:
        print(f"No example JSONs in {src}", file=sys.stderr)
        return 2

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(examples, ensure_ascii=False))
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(examples)} example(s), {out.stat().st_size/1024:.0f} KB)")
    print("Open it in a browser to check, then drop it on Drive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
