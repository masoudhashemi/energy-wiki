"""Build the article's Exp5 walkthrough from the recorded conditional run (no API calls)."""
from pathlib import Path
import hashlib
import html
import json
import re

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/exp5_measured_live_loop/runs/conditional_prepared"
REPO = "https://github.com/masoudhashemi/energy-wiki/blob/main/"
RUN_URL = REPO + RUN.relative_to(ROOT).as_posix() + "/"
START, END = "<!-- EXP5-SYNTHESIS:START -->", "<!-- EXP5-SYNTHESIS:END -->"


def link(path, label):
    return f'<a href="{REPO}{path}">{html.escape(label)}</a>'


def render_text(text, title_paths):
    # Escape model-authored HTML before rendering our small Markdown subset.
    def inline(value):
        value = html.escape(value)
        def wiki(match):
            title = html.unescape(match[1])
            path = title_paths.get(title)
            return link(path, html.unescape(match[0])) if path else match[0]
        value = re.sub(r"\[\[([^\]]+)\]\]", wiki, value)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    blocks = []
    for block in text.split("\n\n"):
        if block.startswith("## "):
            blocks.append("<h4>" + inline(block[3:]) + "</h4>")
        elif block.strip():
            blocks.append("<p>" + inline(block).replace("\n", " ") + "</p>")
    return "\n".join(blocks)


def chart(records, title, maximum, ticks, suffix):
    left, top, width, height = 48, 24, 680, 220
    first, last = records[0]["iteration"], records[-1]["iteration"]
    x = lambda step: left + width * (step-first)/max(1, last-first)
    y = lambda value: top + height * (1-value/maximum)
    result = [f'<svg class="exp5-chart" viewBox="0 0 760 290" role="img" aria-labelledby="exp5-chart-{suffix}">',
              f'<title id="exp5-chart-{suffix}">{html.escape(title)}</title>']
    for tick in ticks:
        result.append(f'<line x1="{left}" x2="{left+width}" y1="{y(tick):.2f}" y2="{y(tick):.2f}" stroke="var(--border)"/>')
        result.append(f'<text x="{left-10}" y="{y(tick)+4:.2f}" text-anchor="end" fill="var(--muted)" font-size="12">{tick:g}</text>')
    series = [("total", "E1 + E2", "var(--accent)", ""),
              ("E1", "E1", "var(--green)", "6 3"),
              ("E2", "E2", "var(--amber)", "2 3")]
    for key, name, color, dash in series:
        points = " ".join(f'{x(r["iteration"]):.2f},{y(r["energy"][key]):.2f}' for r in records)
        result.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-dasharray="{dash}"/>')
        for r in records:
            val = r["energy"][key]
            result.append(f'<circle cx="{x(r["iteration"]):.2f}" cy="{y(val):.2f}" r="3.3" fill="{color}"><title>Revision {r["iteration"]}: {name} = {val:g}</title></circle>')
    for r in records:
        result.append(f'<text x="{x(r["iteration"]):.2f}" y="267" text-anchor="middle" fill="var(--muted)" font-size="12">{r["iteration"]}</text>')
    result.append('<text x="388" y="286" text-anchor="middle" fill="var(--muted)" font-size="12">Actual document revision (0 = seed)</text></svg>')
    return "\n".join(result)


def main():
    records = json.loads((RUN / "trajectory.json").read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == "conditional-synthesis-v1" and summary["complete"]
    assert len(records) == summary["revision_slots"] + 1
    assert [r["iteration"] for r in records] == list(range(len(records)))
    fingerprint = hashlib.sha256(json.dumps(manifest["fixed_conditions"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert all(r["fixed_conditions_sha256"] == fingerprint for r in records)
    for r in records:
        assert abs(r["energy"]["E1"] + r["energy"]["E2"] - r["energy"]["total"]) < 1e-9
    paths = {value["title"]: manifest["source_files"][key]["path"] for key, value in manifest["fixed_conditions"]["D"].items()}
    final = records[-1]
    words = len(final["document"]["content"].split())
    citations = len(set(re.findall(r"\[\[([^\]]+)\]\]", final["document"]["content"])))
    buttons, panels = [], []
    for r in records:
        step = r["iteration"]
        selected = step == final["iteration"]
        label = "Seed" if step == 0 else f"{step} · final" if selected else str(step)
        buttons.append(f'<button type="button" class="step-btn{" active" if selected else ""}" data-exp5-step="{step}" aria-controls="exp5-panel-{step}" aria-pressed="{str(selected).lower()}">{label}</button>')
        e = r["energy"]
        components = []
        for concept in ("w1", "w2"):
            ev = r["evaluation"][concept]
            components.append(f'<div><h4>{"Energy &amp; sampling" if concept == "w1" else "Critique-guided editing"}</h4><p>{html.escape(ev["critic_summary"])}</p><ul>' +
                "".join(f'<li><strong>{html.escape(k.replace("_", " "))}: {ev[k]:g}</strong> — {html.escape(v)}</li>' for k,v in ev["evidence"].items()) + '</ul></div>')
        feedback = ""
        if step:
            feedback = '<details><summary>Critiques used to produce this revision</summary>'
            for concept in ("w1", "w2"):
                instructions = r["feedback_used"][concept]["grad_x_directives"]
                feedback += f'<h4>{html.escape(concept)}: document-edit directives</h4>'
                feedback += ('<ul>' + ''.join('<li>'+render_text(v, paths)+'</li>' for v in instructions) + '</ul>') if instructions else '<p>No new edit directives from this critic.</p>'
            feedback += '</details>'
        panels.append(f'''<section id="exp5-panel-{step}" class="exp5-panel"{' hidden' if not selected else ''}>
          <p class="exp5-score">{'Seed' if step == 0 else f'Revision {step}'} · E₁ = {e['E1']:g} · E₂ = {e['E2']:g} · <strong>Total = {e['total']:g}</strong></p>
          <article class="exp5-document"><h3>{html.escape(r['document']['title'])}</h3>{render_text(r['document']['content'], paths)}</article>
          {feedback}
          <details><summary>Critic evaluations of this document</summary><div class="grid-2">{''.join(components)}</div></details>
        </section>''')
    table = '<table><caption>Saved critic scores for every actual revision</caption><thead><tr><th>Revision</th><th>E₁</th><th>E₂</th><th>E₁ + E₂</th></tr></thead><tbody>' + ''.join(f'<tr><th scope="row">{r["iteration"]}</th><td>{r["energy"]["E1"]:g}</td><td>{r["energy"]["E2"]:g}</td><td>{r["energy"]["total"]:g}</td></tr>' for r in records) + '</tbody></table>'
    section = r'''
    <h2 id="measured-loop">4. Generating a Bridge Document with Two Fixed Concepts</h2>
    <p>Can one new document explain energy-based sampling and critique-guided text editing together? <strong>Experiment 5 turns a two-sentence seed into a source-linked explanation through ten actual revisions.</strong> The final article distinguishes scoring from sampling, explains textual editing, and connects the selected papers while preserving their different assumptions.</p>
    <div class="callout"><strong>The starting point</strong><blockquote>SEED_TEXT</blockquote><strong>The generated result:</strong> WORD_COUNT words with CITE_COUNT source citations, organized into “Energy scores and sampling” and “Critique-guided text editing.” <a href="#exp5-walkthrough">Read the final document below</a> or <a href="FINAL_URL">open the saved Markdown</a>.</div>
    <p>The setup keeps $D$, $w_1$, $a_1$, $w_2$, and $a_2$ fixed. Only $x_{\mathrm{new}}$ changes. The first concept selects six energy/sampling sources; the second selects five text-optimization sources. The same critic evaluates the candidate separately under each fixed pair, and the editor receives both critiques:</p>
    $$x_{\mathrm{new}}^*=\arg\min_{x_{\mathrm{new}}}\left[E_\theta([D,x_{\mathrm{new}}],a_1,w_1)+E_\theta([D,x_{\mathrm{new}}],a_2,w_2)\right].$$
    <p>This makes the composition concrete: the editor must improve the document under both existing concepts. It cannot improve the score by rewriting either concept, changing memberships, or altering the source corpus.</p>

    <h3 id="exp5-walkthrough">The generated document, from seed to final revision</h3>
    <p>The final document is selected initially. Choose any revision to inspect its exact text, the critiques that produced it, and the subsequent evaluations. The text and feedback below come directly from the saved run.</p>
    <div class="figure-box">
      <div class="step-controls"><span>Actual revision</span><div class="btn-group exp5-buttons" role="group" aria-label="Select generated document revision">BUTTONS</div></div>
      <p id="exp5-selection" class="figure-caption" aria-live="polite">Showing final revision 10.</p>
      PANELS
      <noscript><p>Revision controls require JavaScript. The final document is shown above; all drafts are available in the linked trajectory JSON.</p></noscript>
    </div>

    <h3>What the critiques changed</h3>
    <ul>
      <li><strong>From intuition to mechanism:</strong> the seed's “low energy means good states” becomes an explanation of energy scoring, the Boltzmann distribution, and separate sampling procedures.</li>
      <li><strong>From a loose analogy to source-based relationships:</strong> the draft connects ProTeGi's failure-based critiques with TextGrad's rubric-guided edits, and distinguishes both from numerical updates.</li>
      <li><strong>From apparent conflict to explicit assumptions:</strong> the final text attributes the lab note's finite-step claim and distinguishes it from asymptotic sampling statements.</li>
    </ul>

    <div class="figure-box">
      <div class="figure-title">Figure 4: Measured Energy Scores Across Ten Document Revisions</div>
      <div class="legend-row"><span style="color:var(--accent)">● E₁ + E₂ (solid)</span><span style="color:var(--green)">● E₁: energy &amp; sampling (dashed)</span><span style="color:var(--amber)">● E₂: text editing (dotted)</span></div>
      FULL_CHART
      <div class="figure-caption">Every point is a saved critic evaluation. Both component scores first reach zero at revision 3, but the total rises to 0.25 at revision 4 and 0.50 at revision 9. Every valid edit was applied; the trace was not forced to decrease. Lines connect recorded observations without adding intermediate steps.</div>
      <details><summary>Inspect the small score increases (revisions 2–10)</summary>DETAIL_CHART</details>
      <details><summary>View all measured scores as a table</summary>TABLE</details>
    </div>
    <p><strong>Run record:</strong> TEN_EDITS valid edits, CALL_COUNT model calls, TOKEN_COUNT reported tokens. The hashes of the corpus, concepts and membership masks remained unchanged. The critic uses a fixed rubric for concept satisfaction, source support, relationships and calibration. A zero score means that critic rated its requirements satisfied; this single run does not establish convergence or an advantage over other methods.</p>
    <p class="figure-caption">RUNNER_LINK · <a href="REPORT_URL">Run report</a> · <a href="TRACE_URL">Every draft and score</a> · <a href="MANIFEST_URL">Fixed conditions and rubric</a> · <a href="CALLS_URL">Complete model calls</a></p>

    <h2>5. Other Generative Explorations: Experiment 4</h2>
    <p>Exp4 explores two further uses of the energy formulation. These remain separate from the conditional-synthesis walkthrough above; their task-specific ratings are not directly comparable with Exp5's per-concept scores.</p>
    <div class="grid-2">
      <div class="figure-box"><h3>Generating a hard-negative document</h3><p>The prompt asks for a biochemistry note using familiar technical vocabulary in a different context. Its keyword-rubric score is 0.20, while its composite deficit is 9.40 (including a separate pairwise deficit of 4.80). This is a concrete prompted distractor example, not a benchmark of retrieval systems.</p>TRAP_LINK</div>
      <div class="figure-box"><h3>Expressing a cross-domain analogy</h3><p>The prompt supplies a correspondence between energy-model mechanisms and distributed-consensus mechanisms. A separate isomorphism critic evaluates the resulting design. The example illustrates generation under a requested structural analogy; it does not show that the mapping was independently discovered.</p>ANALOGY_LINK</div>
    </div>
    <p>Exp4 also includes the earlier seed-plus-two-revisions bridge example. It remains available in its saved artifacts: EXP4_RESULTS · EXP4_RUNNER · EXP4_CALLS.</p>
    <h2>6. What the Current Experiments Establish</h2>
'''
    e4 = "experiments/exp4_generative_data_synthesis_live_llm/"
    replacements = {"SEED_TEXT":html.escape(records[0]["document"]["content"]), "WORD_COUNT":str(words), "CITE_COUNT":str(citations),
        "FINAL_URL":RUN_URL+"final_document.md", "BUTTONS":"\n".join(buttons), "PANELS":"\n".join(panels),
        "FULL_CHART":chart(records, "Measured E1, E2 and combined scores for seed and revisions 1 through 10", 18, (0,3,6,9,12,15,18), "full"),
        "DETAIL_CHART":chart(records[2:], "Near-zero measured scores for revisions 2 through 10", 0.6, (0,0.25,0.5), "detail"), "TABLE":table,
        "TEN_EDITS":str(summary["valid_edits"]), "CALL_COUNT":str(summary["total_calls"]), "TOKEN_COUNT":f'{summary["total_tokens"]:,}',
        "RUNNER_LINK":link("experiments/exp5_measured_live_loop/run_exp5.py", "Exp5 runner"), "REPORT_URL":RUN_URL+"REPORT.md",
        "TRACE_URL":RUN_URL+"trajectory.json", "MANIFEST_URL":RUN_URL+"manifest.json", "CALLS_URL":RUN_URL+"calls.jsonl",
        "TRAP_LINK":link(e4+"wiki_output/generated_docs/gen_trap_biochem_allostery.md", "Read the generated hard negative"),
        "ANALOGY_LINK":link(e4+"wiki_output/generated_docs/gen_transfer_distributed_consensus.md", "Read the generated analogy"),
        "EXP4_RESULTS":link(e4+"wiki_output/generative_results.json", "Exp4 results"), "EXP4_RUNNER":link(e4+"run_exp4_generative_luna.py", "runner"),
        "EXP4_CALLS":link(e4+"wiki_output/llm_calls_log.json", "call log")}
    for key,value in replacements.items():
        section = section.replace(key,value)
    page_path = ROOT / "docs/index.html"
    page = page_path.read_text(encoding="utf-8")
    block = START + section + END
    if START in page:
        page = page[:page.index(START)] + block + page[page.index(END)+len(END):]
    else:
        start = page.index('    <h2>4. Generating New Data')
        end = page.index('    <h3>What the current experiments establish</h3>') + len('    <h3>What the current experiments establish</h3>')
        page = page[:start] + block + page[end:]
    page = re.sub(r'    function drawGenerativeChart\(\) \{.*?(?=    window.addEventListener)', '', page, flags=re.S)
    page = page.replace('      drawGenerativeChart();\n', '')
    if 'src="exp5-walkthrough.js"' not in page:
        page = page.replace('</head>', '  <script defer src="exp5-walkthrough.js"></script>\n</head>')
    if '/* Exp5 walkthrough */' not in page:
        page = page.replace('  </style>', '''    /* Exp5 walkthrough */
    .exp5-buttons { flex-wrap: wrap; }
    .exp5-document { padding: 0.25rem 1rem 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; overflow-wrap: anywhere; }
    .exp5-document h3 { margin-top: 1rem; }
    .exp5-document h4 { margin-bottom: 0.3rem; }
    .exp5-document p, .exp5-panel li p { font-size: 0.94rem; }
    .exp5-chart { display: block; width: 100%; height: auto; margin-top: 0.7rem; }
    .exp5-panel[hidden] { display: none; }
    details { margin-top: 1rem; }
    summary { cursor: pointer; font-weight: 600; }
    blockquote { margin: 0.7rem 0 1rem; }
    button:focus-visible, summary:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
  </style>''')
    page_path.write_text("\n".join(line.rstrip() for line in page.splitlines()) + "\n", encoding="utf-8")
    print(f"Built Exp5 walkthrough: {len(records)} saved states; {words} final words; {citations} citations.")


if __name__ == "__main__":
    main()
