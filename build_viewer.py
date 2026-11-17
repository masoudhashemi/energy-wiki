"""
Builds the Master Multi-Experiment Interactive HTML Viewer (`wiki_viewer.html`)
combining all three experiment folders:
  1. experiments/exp1_clean_10papers_symbolic/
  2. experiments/exp2_messy_18docs_symbolic/
  3. experiments/exp3_messy_18docs_live_llm/
"""

from __future__ import annotations

import json
import os


def build_compact_payload(bundle: dict, llm_calls: list | None = None) -> dict:
    """Extracts a compact, self-contained payload for the interactive HTML viewer."""
    compact_concepts = []
    for c in bundle["concepts"]:
        rel_filtered = [
            {
                "i": r["entity_i"],
                "j": r["entity_j"],
                "ti": r["title_i"],
                "tj": r["title_j"],
                "gate": r["gate_weight"],
                "coh": r["coherence_deficit"],
                "link": r["link_deficit"],
                "contra": r.get("has_contradiction", False),
                "shared": r["shared_mechanisms"],
                "critique": r["critique"],
            }
            for r in c["pairwise_relations"]
        ]
        compact_concepts.append({
            "concept": c["concept"],
            "w_energy_trace": c["w_energy_trace"],
            "w_grad_history": c["w_grad_history"][:6],
            "a_energy_trace": c["a_energy_trace"],
            "a_step_history": [
                {
                    "step": s["step"],
                    "energy": s["energy"],
                    "coherence_term": s["coherence_term"],
                    "coverage_term": s["coverage_term"],
                    "gates": s["gates"],
                    "top_critiques": s["top_critiques"],
                }
                for s in c["a_step_history"]
            ],
            "x_energy_trace": c["x_energy_trace"],
            "x_edit_history": c["x_edit_history"],
            "final_gates": c["final_gates"],
            "concept_page": c["concept_page"],
            "pairwise_relations": rel_filtered,
        })

    return {
        "experiment_suite": bundle.get("experiment_suite", "default"),
        "title": bundle.get("title", "Energy-Based LLM Wiki"),
        "llm_model_info": bundle.get("llm_model_info", None),
        "llm_calls": llm_calls or [],
        "raw_entities": bundle["raw_entities"],
        "compiled_entities": bundle["compiled_entities"],
        "index_md": bundle["index_md"],
        "meta_losses": bundle["meta_losses"],
        "concepts": compact_concepts,
        "experiments": bundle.get("experiments", {}),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Energy-Based LLM Wiki Explorer (Multi-Experiment Suite)</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {
      color-scheme: light dark;
    }
    .wiki-link {
      color: var(--primary, #3b82f6);
      font-weight: 600;
      text-decoration: underline;
      text-underline-offset: 2px;
      cursor: pointer;
    }
    .wiki-link:hover {
      opacity: 0.85;
    }
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-thumb {
      background: rgba(128, 128, 128, 0.35);
      border-radius: 3px;
    }
  </style>
</head>
<body class="bg-[var(--background,#0f172a)] text-[var(--foreground,#f8fafc)] antialiased p-4 m-0 min-h-screen">
  <div class="max-w-7xl mx-auto space-y-4">

    <!-- Header & Experiment Suite Switcher -->
    <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-4 space-y-3">
      <div class="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 text-xs font-mono rounded bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#ffffff)]">arXiv:1811.02486v1 → LLM Wiki</span>
            <h1 class="text-lg font-bold tracking-tight">Energy-Based LLM Wiki &amp; Textual SGLD Explorer</h1>
          </div>
          <p id="suite-subtitle" class="text-xs text-[var(--muted-foreground,#94a3b8)] mt-1"></p>
        </div>
        <!-- Experiment Folder Selector -->
        <div class="flex items-center gap-2 text-xs">
          <label class="font-semibold text-amber-400">Experiment Folder:</label>
          <select id="suite-select" onchange="onSuiteChange()" class="bg-[var(--background,#0f172a)] border border-amber-500/50 rounded px-3 py-1.5 text-xs font-semibold text-[var(--foreground,#f8fafc)]"></select>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <div class="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[var(--border,#334155)] text-xs">
        <div class="flex flex-wrap gap-1.5" id="nav-tabs">
          <button onclick="switchTab('graph')" id="tab-btn-graph" class="px-3 py-1.5 rounded font-medium bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#fff)]">1. Knowledge Graph &amp; SGLD</button>
          <button onclick="switchTab('wiki')" id="tab-btn-wiki" class="px-3 py-1.5 rounded font-medium bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)]">2. Wiki Browser (x⁰ → x¹)</button>
          <button onclick="switchTab('matrix')" id="tab-btn-matrix" class="px-3 py-1.5 rounded font-medium bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)]">3. Pairwise Relation Matrix g_θ</button>
          <button onclick="switchTab('experiments')" id="tab-btn-experiments" class="px-3 py-1.5 rounded font-medium bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)]">4. Experiments (Sec. 5)</button>
          <button onclick="switchTab('llm')" id="tab-btn-llm" class="px-3 py-1.5 rounded font-medium bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)]">5. Live LLM Call Log</button>
        </div>
        <span id="llm-telemetry-pill" class="font-mono text-xs px-2.5 py-1 rounded bg-emerald-500/15 text-emerald-400"></span>
      </div>
    </div>

    <!-- Concept & SGLD Controls Bar -->
    <div id="concept-control-bar" class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 flex flex-wrap items-center justify-between gap-4 text-xs">
      <div class="flex flex-wrap items-center gap-3">
        <label class="text-[var(--muted-foreground,#94a3b8)] font-medium">Discovered / Active Concept $\mathbf{w}^* = [\mathbf{w}_a^*, \mathbf{w}_x^*]$:</label>
        <select id="concept-select" onchange="onConceptChange()" class="bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] rounded px-2.5 py-1.5 text-xs font-medium"></select>
      </div>
      <div class="flex items-center gap-3">
        <label class="text-[var(--muted-foreground,#94a3b8)] font-medium">SGLD Step $k$:</label>
        <input id="sgld-slider" type="range" min="0" max="10" value="10" oninput="onSliderChange()" class="w-36 accent-[var(--primary,#3b82f6)]" />
        <span id="sgld-step-readout" class="font-mono font-semibold tabular-nums px-2 py-0.5 rounded bg-[var(--background,#0f172a)] border border-[var(--border,#334155)]">k = 10 / 10</span>
        <button onclick="playSGLD()" id="play-btn" class="px-2.5 py-1 rounded bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#fff)] font-medium">Replay SGLD</button>
      </div>
    </div>

    <!-- ==================== TAB 1: KNOWLEDGE GRAPH & SGLD ==================== -->
    <div id="tab-graph" class="space-y-4">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="lg:col-span-2 bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 flex flex-col">
          <div class="flex items-center justify-between pb-2 mb-2 border-b border-[var(--border,#334155)] text-xs">
            <span class="font-semibold">Attention Subgraph $\sigma(\mathbf{a}^k)$ &amp; Compiled `[[wiki-links]]` ($\mathbf{x}^1$)</span>
            <span class="text-[var(--muted-foreground,#94a3b8)]">Click any node to open in Wiki Browser</span>
          </div>
          <canvas id="kg-canvas" width="760" height="440" class="w-full rounded bg-[var(--background,#0f172a)] cursor-pointer"></canvas>
          <div class="flex flex-wrap items-center justify-between gap-2 pt-2 text-xs text-[var(--muted-foreground,#94a3b8)]">
            <div class="flex flex-wrap items-center gap-3">
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block bg-emerald-500"></span> Attended ($\sigma(a_i) \ge 0.5$)</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block bg-rose-500"></span> Polysemous Trap (Rejected by $g_\theta$)</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full inline-block bg-slate-500"></span> Unattended / Ops Clutter</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded inline-block bg-blue-500"></span> Concept Hub</span>
            </div>
            <span id="graph-status-readout" class="font-mono"></span>
          </div>
        </div>

        <div class="space-y-3 flex flex-col">
          <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 space-y-2 text-xs">
            <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-1.5">
              <span class="font-semibold">Discovered Concept Code $\mathbf{w}^* = [\mathbf{w}_a, \mathbf{w}_x]$</span>
              <span id="concept-id-badge" class="font-mono text-[var(--muted-foreground,#94a3b8)]"></span>
            </div>
            <div>
              <span class="font-semibold text-emerald-400">Argument Code ($\mathbf{w}_a$ — Identification):</span>
              <p id="w-a-text" class="text-[var(--muted-foreground,#94a3b8)] mt-0.5 leading-relaxed"></p>
            </div>
            <div>
              <span class="font-semibold text-blue-400">Action Code ($\mathbf{w}_x$ — Wiki Synthesis):</span>
              <p id="w-x-text" class="text-[var(--muted-foreground,#94a3b8)] mt-0.5 leading-relaxed"></p>
            </div>
            <div id="w-keywords" class="flex flex-wrap gap-1 pt-1"></div>
          </div>

          <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 space-y-2 text-xs">
            <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-1.5">
              <span class="font-semibold">Energy Minimization $E_\theta(\mathbf{x}, \mathbf{a}^k, \mathbf{w}^k)$</span>
              <span id="energy-readout" class="font-mono font-bold text-emerald-400"></span>
            </div>
            <canvas id="energy-canvas" width="340" height="125" class="w-full rounded bg-[var(--background,#0f172a)]"></canvas>
          </div>

          <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 space-y-2 text-xs flex-1">
            <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-1.5">
              <span class="font-semibold">Textual Gradients $\nabla_{\mathbf{w}} E, \nabla_{\mathbf{a}} E, \nabla_{\mathbf{x}^1} E$</span>
              <span id="step-badge" class="font-mono text-[var(--muted-foreground,#94a3b8)]">Step k=10</span>
            </div>
            <div id="textual-grad-list" class="space-y-1.5 max-h-44 overflow-y-auto font-mono text-xs"></div>
          </div>
        </div>
      </div>

      <!-- Entity Attention Table -->
      <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 text-xs">
        <div class="font-semibold pb-2 mb-2 border-b border-[var(--border,#334155)]">
          Knowledge Base Documents $\mathbf{x}_i$, Source Type &amp; Attention Gates $\sigma(\mathbf{a}_i^k)$ at Step $k$
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse">
            <thead>
              <tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
                <th class="py-1.5 pr-3">Document Title</th>
                <th class="py-1.5 px-3">Source Type</th>
                <th class="py-1.5 px-3 w-44">Attention Gate $\sigma(a_i^k)$</th>
                <th class="py-1.5 px-3">Textual Gradient Rationale ($\nabla_{a_i} E_\theta$)</th>
                <th class="py-1.5 pl-3">Action</th>
              </tr>
            </thead>
            <tbody id="entity-attention-tbody" class="divide-y divide-[var(--border,#334155)]"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 2: WIKI BROWSER ==================== -->
    <div id="tab-wiki" class="hidden grid grid-cols-1 lg:grid-cols-4 gap-4">
      <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 space-y-3 text-xs max-h-[720px] overflow-y-auto">
        <div>
          <div class="font-bold text-[var(--muted-foreground,#94a3b8)] uppercase tracking-wider mb-1.5">Catalog</div>
          <button onclick="openWikiPage('index', 'index')" class="w-full text-left px-2.5 py-1.5 rounded font-medium hover:bg-[var(--secondary,#334155)]">
            📄 index.md (Master Catalog)
          </button>
        </div>
        <div>
          <div class="font-bold text-[var(--muted-foreground,#94a3b8)] uppercase tracking-wider mb-1.5">Discovered Concept Pages (w*)</div>
          <div id="sidebar-concepts" class="space-y-1"></div>
        </div>
        <div>
          <div class="font-bold text-[var(--muted-foreground,#94a3b8)] uppercase tracking-wider mb-1.5">KB Entity Pages (x¹)</div>
          <div id="sidebar-entities" class="space-y-1"></div>
        </div>
      </div>

      <div class="lg:col-span-3 bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-4 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[var(--border,#334155)]">
          <div>
            <span id="wiki-page-type-badge" class="px-2 py-0.5 text-xs font-mono rounded bg-blue-500/20 text-blue-400">CONCEPT SYNTHESIS</span>
            <h2 id="wiki-page-title" class="text-base font-bold mt-1"></h2>
          </div>
          <div id="wiki-diff-toggle-wrap" class="hidden flex items-center gap-1.5 text-xs">
            <button onclick="setEntityViewMode('compiled')" id="btn-mode-compiled" class="px-2.5 py-1 rounded bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#fff)] font-medium">Compiled Wiki (x¹)</button>
            <button onclick="setEntityViewMode('raw')" id="btn-mode-raw" class="px-2.5 py-1 rounded bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)] font-medium">Raw Source (x⁰)</button>
            <button onclick="setEntityViewMode('diff')" id="btn-mode-diff" class="px-2.5 py-1 rounded bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)] font-medium">Diff (x⁰ → x¹)</button>
          </div>
        </div>

        <div id="wiki-page-body" class="text-sm leading-relaxed space-y-3"></div>

        <div id="wiki-links-footer" class="pt-3 border-t border-[var(--border,#334155)] grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div class="bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] rounded p-2.5">
            <div class="font-semibold text-[var(--muted-foreground,#94a3b8)] mb-1.5">Outgoing `[[wiki-links]]`</div>
            <div id="wiki-outgoing-links" class="flex flex-wrap gap-1.5"></div>
          </div>
          <div class="bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] rounded p-2.5">
            <div class="font-semibold text-[var(--muted-foreground,#94a3b8)] mb-1.5">Incoming Backlinks &amp; Participating Concepts</div>
            <div id="wiki-incoming-links" class="flex flex-wrap gap-1.5"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 3: PAIRWISE RELATION MATRIX ==================== -->
    <div id="tab-matrix" class="hidden grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div class="lg:col-span-2 bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 text-xs space-y-2">
        <div class="flex items-center justify-between pb-2 border-b border-[var(--border,#334155)]">
          <span class="font-semibold">Pairwise Relational Operator $\sigma(a_i)\sigma(a_j) \cdot g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$ (Eq. 4)</span>
          <span class="text-[var(--muted-foreground,#94a3b8)]">Click any cell $(i, j)$ to inspect relational critique</span>
        </div>
        <div id="matrix-container" class="overflow-x-auto"></div>
      </div>

      <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3 text-xs space-y-3">
        <div class="font-semibold pb-2 border-b border-[var(--border,#334155)]">Selected Pair $(i, j)$ Relational Inspector</div>
        <div id="pair-inspector" class="space-y-2.5 text-[var(--muted-foreground,#94a3b8)]">
          Click any cell in the matrix to inspect how the Relation Network evaluates the pair under the current concept $\mathbf{w}$.
        </div>
      </div>
    </div>

    <!-- ==================== TAB 4: EXPERIMENTAL BENCHMARK SUITE ==================== -->
    <div id="tab-experiments" class="hidden space-y-4 text-xs">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3.5 space-y-2.5">
          <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
            <span class="font-bold">Exp 1: Multi-Step Textual SGLD ($K=1 \dots 20$) vs. One-Shot ($K=1$)</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Sec. 5.3 / Fig. 7</span>
          </div>
          <div id="exp1-table-wrap"></div>
        </div>

        <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3.5 space-y-2.5">
          <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
            <span class="font-bold">Exp 2: Pairwise Relation Network vs. Unary RAG ("No RN")</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Sec. 5.3 / Eq. 4</span>
          </div>
          <div id="exp2-table-wrap"></div>
        </div>

        <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3.5 space-y-2.5">
          <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
            <span class="font-bold">Exp 3: Configured Weights (Identification $\leftrightarrow$ Generation)</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Sec. 5.2 / Table p. 11</span>
          </div>
          <div id="exp3-table-wrap"></div>
        </div>

        <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3.5 space-y-2.5">
          <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
            <span class="font-bold">Exp 4 &amp; 5: Code Interchangeability ($\mathbf{w}_a \leftrightarrow \mathbf{w}_x$) &amp; Optimizer Settings</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Fig. 6 &amp; Fig. 8</span>
          </div>
          <div id="exp5-table-wrap"></div>
        </div>

        <div class="md:col-span-2 bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-3.5 space-y-2.5">
          <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
            <span class="font-bold text-amber-400">Exp 6 (Messy KB Stress Test): Non-Monotonic Polysemous Trap Rejection ($k=0 \to 2 \to 10$)</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Sec. 5.3 / Fig. 7 Top-Right</span>
          </div>
          <div id="exp6-table-wrap"></div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 5: LIVE LLM CALL INSPECTOR ==================== -->
    <div id="tab-llm" class="hidden space-y-3 text-xs">
      <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-lg p-4 space-y-3">
        <div class="flex items-center justify-between border-b border-[var(--border,#334155)] pb-2">
          <span class="font-bold text-sm">Live LiteLLM / OpenAI API Call Trace (Unsupervised Concept Discovery, Relational Critic &amp; Wiki Synthesis)</span>
          <span id="llm-summary-badge" class="font-mono text-emerald-400"></span>
        </div>
        <div id="llm-calls-container" class="space-y-3"></div>
      </div>
    </div>

  </div>

  <script>
    const SUITES = __SUITES_JSON_PAYLOAD__;
    let currentSuiteIdx = SUITES.length - 1;
    let WIKI_DATA = SUITES[currentSuiteIdx];

    let currentTab = 'graph';
    let currentConceptIdx = 0;
    let currentStep = 10;
    let playInterval = null;
    let currentWikiPage = { type: 'concept', id: '' };
    let entityViewMode = 'compiled';
    let nodePositions = {};

    function initViewer() {
      const suiteSel = document.getElementById('suite-select');
      suiteSel.innerHTML = '';
      SUITES.forEach((s, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = s.title;
        suiteSel.appendChild(opt);
      });
      suiteSel.value = currentSuiteIdx;

      const canvas = document.getElementById('kg-canvas');
      canvas.addEventListener('click', onCanvasClick);
      window.addEventListener('themechange', () => {
        drawKnowledgeGraph();
        drawEnergyChart();
      });

      onSuiteChange();
    }

    function onSuiteChange() {
      currentSuiteIdx = parseInt(document.getElementById('suite-select').value, 10);
      WIKI_DATA = SUITES[currentSuiteIdx];

      document.getElementById('suite-subtitle').textContent =
        `Folder: experiments/${WIKI_DATA.experiment_suite}/ • ${WIKI_DATA.compiled_entities.length} KB Docs • ${WIKI_DATA.concepts.length} Discovered Concepts`;

      const pill = document.getElementById('llm-telemetry-pill');
      if (WIKI_DATA.llm_model_info) {
        pill.textContent = `Live LLM: ${WIKI_DATA.llm_model_info.active_model} (${WIKI_DATA.llm_model_info.total_llm_calls} calls, ${WIKI_DATA.llm_model_info.total_tokens} tokens)`;
      } else {
        pill.textContent = `Symbolic / Unsupervised Contextual Critic (0 API calls)`;
      }

      const sel = document.getElementById('concept-select');
      sel.innerHTML = '';
      WIKI_DATA.concepts.forEach((c, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `${idx + 1}. ${c.concept.name}`;
        sel.appendChild(opt);
      });

      currentConceptIdx = 0;
      populateWikiSidebar();
      renderExperimentsTab();
      renderLLMCallsTab();
      onConceptChange();
      openWikiPage('concept', WIKI_DATA.concepts[0].concept.concept_id);
    }

    function switchTab(tab) {
      currentTab = tab;
      ['graph', 'wiki', 'matrix', 'experiments', 'llm'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById(`tab-btn-${t}`);
        if (t === tab) {
          btn.className = 'px-3 py-1.5 rounded font-medium bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#fff)]';
        } else {
          btn.className = 'px-3 py-1.5 rounded font-medium bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)]';
        }
      });
      document.getElementById('concept-control-bar').classList.toggle('hidden', tab === 'wiki' || tab === 'experiments' || tab === 'llm');
      if (tab === 'graph') {
        drawKnowledgeGraph();
        drawEnergyChart();
      } else if (tab === 'matrix') {
        renderPairwiseMatrix();
      }
    }

    function onConceptChange() {
      currentConceptIdx = parseInt(document.getElementById('concept-select').value, 10);
      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const maxSteps = cObj.a_step_history.length - 1;
      const slider = document.getElementById('sgld-slider');
      slider.max = maxSteps;
      slider.value = maxSteps;
      currentStep = maxSteps;

      document.getElementById('concept-id-badge').textContent = cObj.concept.concept_id;
      document.getElementById('w-a-text').textContent = cObj.concept.w_a;
      document.getElementById('w-x-text').textContent = cObj.concept.w_x;

      const kwWrap = document.getElementById('w-keywords');
      kwWrap.innerHTML = cObj.concept.keywords.map(k =>
        `<span class="px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 font-mono text-xs">${escapeHtml(k)}</span>`
      ).join('');

      updateStepView();
      if (currentTab === 'matrix') renderPairwiseMatrix();
    }

    function onSliderChange() {
      currentStep = parseInt(document.getElementById('sgld-slider').value, 10);
      updateStepView();
    }

    function playSGLD() {
      if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
      }
      const slider = document.getElementById('sgld-slider');
      currentStep = 0;
      slider.value = 0;
      updateStepView();

      playInterval = setInterval(() => {
        const maxSteps = parseInt(slider.max, 10);
        if (currentStep >= maxSteps) {
          clearInterval(playInterval);
          playInterval = null;
          return;
        }
        currentStep += 1;
        slider.value = currentStep;
        updateStepView();
      }, 420);
    }

    function updateStepView() {
      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const maxSteps = cObj.a_step_history.length - 1;
      const aStep = cObj.a_step_history[Math.min(currentStep, maxSteps)];
      const xStep = cObj.x_edit_history[Math.min(currentStep, cObj.x_edit_history.length - 1)];

      document.getElementById('sgld-step-readout').textContent = `k = ${currentStep} / ${maxSteps}`;
      document.getElementById('step-badge').textContent = `Step k=${currentStep}`;
      document.getElementById('energy-readout').textContent =
        `E_a = ${aStep.energy.toFixed(4)} | E_x = ${xStep.energy.toFixed(4)}`;

      const gradList = document.getElementById('textual-grad-list');
      const items = [];
      if (cObj.w_grad_history && cObj.w_grad_history.length) {
        const wIdx = Math.min(currentStep, cObj.w_grad_history.length - 1);
        items.push(`<div class="p-1.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300"><strong>∇_w E:</strong> ${escapeHtml(cObj.w_grad_history[wIdx])}</div>`);
      }
      (xStep.applied_edits || []).forEach(ed => {
        items.push(`<div class="p-1.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-300"><strong>∇_x¹ E:</strong> ${escapeHtml(ed)}</div>`);
      });
      (aStep.top_critiques || []).forEach(cr => {
        items.push(`<div class="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300"><strong>∇_a E:</strong> ${escapeHtml(cr)}</div>`);
      });
      gradList.innerHTML = items.join('');

      const tbody = document.getElementById('entity-attention-tbody');
      tbody.innerHTML = WIKI_DATA.compiled_entities.map(ent => {
        const g = aStep.gates[ent.entity_id] ?? 0;
        const active = g >= 0.5;
        const isTrap = ent.source_type === 'polysemous_trap';
        const isOps = ent.source_type === 'ops_clutter';
        const pct = Math.round(g * 100);
        const typeBadge = isTrap ? 'bg-rose-500/20 text-rose-400'
                        : isOps ? 'bg-slate-600/30 text-slate-400'
                        : ent.source_type === 'contradiction_note' ? 'bg-amber-500/20 text-amber-400'
                        : ent.source_type === 'dev_log' ? 'bg-purple-500/20 text-purple-400'
                        : 'bg-emerald-500/15 text-emerald-400';
        const rat = (aStep.rationales && aStep.rationales[ent.entity_id]) || '';
        return `
          <tr class="hover:bg-[var(--secondary,#334155)]/40">
            <td class="py-1.5 pr-3 font-medium">
              <span class="inline-block w-2 h-2 rounded-full mr-1.5 ${active ? 'bg-emerald-400' : (isTrap ? 'bg-rose-400' : 'bg-slate-500')}"></span>
              ${escapeHtml(ent.title)}
            </td>
            <td class="py-1.5 px-3"><span class="px-1.5 py-0.5 rounded font-mono text-[10px] ${typeBadge}">${escapeHtml(ent.source_type || 'research')}</span></td>
            <td class="py-1.5 px-3">
              <div class="flex items-center gap-2">
                <div class="w-24 h-2 bg-slate-700 rounded overflow-hidden">
                  <div class="h-full ${active ? 'bg-emerald-500' : (isTrap ? 'bg-rose-500' : 'bg-slate-500')}" style="width:${pct}%"></div>
                </div>
                <span class="font-mono tabular-nums ${active ? 'text-emerald-400 font-semibold' : 'text-[var(--muted-foreground,#94a3b8)]'}">${g.toFixed(3)}</span>
              </div>
            </td>
            <td class="py-1.5 px-3 text-[var(--muted-foreground,#94a3b8)] truncate max-w-xs" title="${escapeHtml(rat)}">${escapeHtml(rat)}</td>
            <td class="py-1.5 pl-3">
              <button onclick="jumpToEntity('${ent.entity_id}')" class="px-2 py-0.5 rounded bg-[var(--secondary,#334155)] hover:opacity-90 text-xs">Open</button>
            </td>
          </tr>
        `;
      }).join('');

      drawKnowledgeGraph();
      drawEnergyChart();
    }

    function drawKnowledgeGraph() {
      const canvas = document.getElementById('kg-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const aStep = cObj.a_step_history[Math.min(currentStep, cObj.a_step_history.length - 1)];
      const gates = aStep.gates;
      const entities = WIKI_DATA.compiled_entities;
      const cx = W / 2;
      const cy = H / 2;
      const rx = W * 0.38;
      const ry = H * 0.38;

      nodePositions = {};
      entities.forEach((ent, idx) => {
        const angle = (2 * Math.PI * idx) / entities.length - Math.PI / 2;
        nodePositions[ent.entity_id] = {
          x: cx + rx * Math.cos(angle),
          y: cy + ry * Math.sin(angle),
          r: 15,
          type: 'entity',
          id: ent.entity_id,
          title: ent.title,
          source_type: ent.source_type
        };
      });

      nodePositions['__concept_hub__'] = {
        x: cx,
        y: cy,
        r: 23,
        type: 'concept',
        id: cObj.concept.concept_id,
        title: cObj.concept.name
      };

      const stepProgress = currentStep / Math.max(1, cObj.a_step_history.length - 1);
      cObj.pairwise_relations.forEach(rel => {
        const gi = gates[rel.i] ?? 0;
        const gj = gates[rel.j] ?? 0;
        const gateProd = gi * gj;
        if (gateProd >= 0.22 && rel.coh < 0.55) {
          const p1 = nodePositions[rel.i];
          const p2 = nodePositions[rel.j];
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = rel.contra
              ? `rgba(245, 158, 11, ${Math.min(0.85, gateProd * stepProgress + 0.25)})`
              : `rgba(16, 185, 129, ${Math.min(0.75, gateProd * stepProgress * 0.9 + 0.15)})`;
            ctx.lineWidth = 1.2 + 2.2 * gateProd * stepProgress;
            ctx.stroke();
          }
        }
      });

      entities.forEach(ent => {
        const g = gates[ent.entity_id] ?? 0;
        if (g >= 0.5 && currentStep > 0) {
          const p = nodePositions[ent.entity_id];
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(p.x, p.y);
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = `rgba(59, 130, 246, ${0.45 * stepProgress + 0.15})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });

      ctx.beginPath();
      ctx.arc(cx, cy, 23, 0, 2 * Math.PI);
      ctx.fillStyle = '#2563eb';
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#93c5fd';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('CONCEPT HUB', cx, cy - 3);
      ctx.font = '10px monospace';
      ctx.fillText(`k=${currentStep}`, cx, cy + 10);

      let activeCount = 0;
      entities.forEach(ent => {
        const pos = nodePositions[ent.entity_id];
        const g = gates[ent.entity_id] ?? 0;
        const active = g >= 0.5;
        const isTrap = ent.source_type === 'polysemous_trap';
        if (active) activeCount++;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, pos.r, 0, 2 * Math.PI);
        ctx.fillStyle = active ? '#10b981' : (isTrap ? '#e11d48' : '#475569');
        ctx.fill();
        ctx.lineWidth = active ? 2.5 : 1.5;
        ctx.strokeStyle = active ? '#a7f3d0' : (isTrap ? '#fda4af' : '#94a3b8');
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(g.toFixed(2), pos.x, pos.y + 3);

        const shortTitle = ent.title.length > 22 ? ent.title.slice(0, 20) + '…' : ent.title;
        const labelY = pos.y < cy ? pos.y - 20 : pos.y + 27;
        ctx.fillStyle = active ? '#ecfdf5' : (isTrap ? '#fda4af' : '#94a3b8');
        ctx.font = active ? 'bold 10px sans-serif' : '10px sans-serif';
        ctx.fillText(shortTitle, pos.x, labelY);
      });

      document.getElementById('graph-status-readout').textContent =
        `Attended Nodes: ${activeCount} / ${entities.length}`;
    }

    function onCanvasClick(e) {
      const canvas = document.getElementById('kg-canvas');
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      for (const key of Object.keys(nodePositions)) {
        const n = nodePositions[key];
        const dist = Math.hypot(mx - n.x, my - n.y);
        if (dist <= n.r + 6) {
          switchTab('wiki');
          openWikiPage(n.type, n.id);
          return;
        }
      }
    }

    function drawEnergyChart() {
      const canvas = document.getElementById('energy-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const aTrace = cObj.a_energy_trace;
      const xTrace = cObj.x_energy_trace;
      const maxVal = Math.max(...aTrace, ...xTrace, 0.5);

      const padL = 34, padR = 14, padT = 14, padB = 22;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, H - padB);
      ctx.lineTo(W - padR, H - padB);
      ctx.stroke();

      function plotLine(trace, color) {
        if (!trace.length) return;
        ctx.beginPath();
        trace.forEach((v, idx) => {
          const x = padL + (idx / Math.max(1, aTrace.length - 1)) * plotW;
          const y = padT + plotH * (1 - v / maxVal);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      plotLine(aTrace, '#10b981');
      plotLine(xTrace, '#3b82f6');

      const markerX = padL + (Math.min(currentStep, aTrace.length - 1) / Math.max(1, aTrace.length - 1)) * plotW;
      ctx.beginPath();
      ctx.moveTo(markerX, padT);
      ctx.lineTo(markerX, H - padB);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Green: E(x⁰,a,w_a)  Blue: E(x¹,a,w_x)', padL + 4, padT + 10);
    }

    function populateWikiSidebar() {
      const cWrap = document.getElementById('sidebar-concepts');
      cWrap.innerHTML = WIKI_DATA.concepts.map(c => `
        <button onclick="openWikiPage('concept', '${c.concept.concept_id}')"
          class="w-full text-left px-2.5 py-1.5 rounded hover:bg-[var(--secondary,#334155)] truncate">
          ⚡ ${escapeHtml(c.concept.name)}
        </button>
      `).join('');

      const eWrap = document.getElementById('sidebar-entities');
      eWrap.innerHTML = WIKI_DATA.compiled_entities.map(e => {
        const icon = e.source_type === 'polysemous_trap' ? '🚫'
                   : e.source_type === 'ops_clutter' ? '🗑️'
                   : e.source_type === 'contradiction_note' ? '⚠️'
                   : e.source_type === 'dev_log' ? '🛠️' : '📘';
        return `
          <button onclick="openWikiPage('entity', '${e.entity_id}')"
            class="w-full text-left px-2.5 py-1.5 rounded hover:bg-[var(--secondary,#334155)] truncate">
            ${icon} ${escapeHtml(e.title)}
          </button>
        `;
      }).join('');
    }

    function jumpToEntity(entityId) {
      switchTab('wiki');
      openWikiPage('entity', entityId);
    }

    function setEntityViewMode(mode) {
      entityViewMode = mode;
      ['compiled', 'raw', 'diff'].forEach(m => {
        const b = document.getElementById(`btn-mode-${m}`);
        if (m === mode) {
          b.className = 'px-2.5 py-1 rounded bg-[var(--primary,#3b82f6)] text-[var(--primary-foreground,#fff)] font-medium';
        } else {
          b.className = 'px-2.5 py-1 rounded bg-[var(--secondary,#334155)] text-[var(--secondary-foreground,#e2e8f0)] font-medium';
        }
      });
      openWikiPage(currentWikiPage.type, currentWikiPage.id);
    }

    function renderWikiLinksInline(text) {
      const escaped = escapeHtml(text);
      return escaped.replace(/\[\[([^\]]+)\]\]/g, (match, targetTitle) => {
        return `<span class="wiki-link" onclick="navigateByTitle('${targetTitle.replace(/'/g, "\\'")}')">[[${targetTitle}]]</span>`;
      });
    }

    function formatMarkdownSimple(md) {
      return md.split('\n').map(line => {
        const trimmed = line.trim();
        if (!trimmed) return '';
        if (trimmed.startsWith('# ')) {
          return `<h3 class="text-base font-bold mt-2">${renderWikiLinksInline(trimmed.slice(2))}</h3>`;
        }
        if (trimmed.startsWith('## ')) {
          return `<h4 class="text-sm font-bold text-blue-400 mt-3 border-b border-[var(--border,#334155)] pb-1">${renderWikiLinksInline(trimmed.slice(3))}</h4>`;
        }
        if (trimmed.startsWith('### ')) {
          return `<h5 class="text-xs font-bold text-emerald-400 mt-2">${renderWikiLinksInline(trimmed.slice(4))}</h5>`;
        }
        if (trimmed.startsWith('- ')) {
          return `<div class="pl-3 border-l-2 border-[var(--border,#334155)] my-1 text-xs">${renderWikiLinksInline(trimmed.slice(2))}</div>`;
        }
        return `<p class="text-xs leading-relaxed">${renderWikiLinksInline(trimmed)}</p>`;
      }).join('');
    }

    function navigateByTitle(title) {
      const cMatch = WIKI_DATA.concepts.find(c => c.concept.name.toLowerCase() === title.toLowerCase());
      if (cMatch) {
        openWikiPage('concept', cMatch.concept.concept_id);
        return;
      }
      const eMatch = WIKI_DATA.compiled_entities.find(e =>
        e.title.toLowerCase() === title.toLowerCase() || e.entity_id.toLowerCase() === title.toLowerCase()
      );
      if (eMatch) {
        openWikiPage('entity', eMatch.entity_id);
      }
    }

    function openWikiPage(type, id) {
      currentWikiPage = { type, id };
      const diffWrap = document.getElementById('wiki-diff-toggle-wrap');
      const badge = document.getElementById('wiki-page-type-badge');
      const titleEl = document.getElementById('wiki-page-title');
      const bodyEl = document.getElementById('wiki-page-body');
      const outEl = document.getElementById('wiki-outgoing-links');
      const inEl = document.getElementById('wiki-incoming-links');

      if (type === 'index') {
        diffWrap.classList.add('hidden');
        badge.textContent = 'MASTER CATALOG (index.md)';
        titleEl.textContent = 'Energy-Based LLM Wiki Catalog (index.md)';
        bodyEl.innerHTML = formatMarkdownSimple(WIKI_DATA.index_md);
        outEl.innerHTML = `<span class="text-[var(--muted-foreground,#94a3b8)]">Links to all Discovered Concepts &amp; Entities</span>`;
        inEl.innerHTML = `<span class="text-[var(--muted-foreground,#94a3b8)]">Root entry point</span>`;
        return;
      }

      if (type === 'concept') {
        diffWrap.classList.add('hidden');
        const cObj = WIKI_DATA.concepts.find(c => c.concept.concept_id === id) || WIKI_DATA.concepts[0];
        badge.textContent = `DISCOVERED CONCEPT SYNTHESIS (${cObj.concept.concept_id})`;
        titleEl.textContent = cObj.concept.name;
        const pageMd = cObj.concept_page ? cObj.concept_page.content : 'No concept page generated.';
        bodyEl.innerHTML = formatMarkdownSimple(pageMd);

        const links = cObj.concept_page ? cObj.concept_page.wiki_links : [];
        outEl.innerHTML = links.map(lk =>
          `<button onclick="navigateByTitle('${lk.replace(/'/g, "\\'")}')" class="px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 font-medium">[[${escapeHtml(lk)}]]</button>`
        ).join('');
        inEl.innerHTML = `<span class="text-[var(--muted-foreground,#94a3b8)]">Referenced in index.md and ${links.length} entity backlinks</span>`;
        return;
      }

      if (type === 'entity') {
        diffWrap.classList.remove('hidden');
        const compEnt = WIKI_DATA.compiled_entities.find(e => e.entity_id === id);
        const rawEnt = WIKI_DATA.raw_entities.find(e => e.entity_id === id);
        if (!compEnt) return;

        badge.textContent = `ENTITY PAGE (${compEnt.source_type || 'research'} • ${compEnt.arxiv_id})`;
        titleEl.textContent = `${compEnt.title} — ${compEnt.authors}`;

        const metaBox = `
          <div class="p-2.5 rounded bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] text-xs space-y-1">
            <div><strong>Formula / Snippet:</strong> <code class="font-mono text-emerald-400">${escapeHtml(compEnt.key_equation)}</code></div>
            <div><strong>Source Type:</strong> <code class="font-mono">${escapeHtml(compEnt.source_type || 'research_note')}</code></div>
          </div>
        `;

        if (entityViewMode === 'raw') {
          bodyEl.innerHTML = metaBox + `<div class="p-3 rounded bg-amber-500/10 border border-amber-500/30 text-xs"><strong>Raw Uncompiled Source (x⁰ in raw/${compEnt.entity_id}.md):</strong><p class="mt-1.5 leading-relaxed">${escapeHtml(rawEnt.content)}</p></div>`;
        } else if (entityViewMode === 'diff') {
          const addedLines = compEnt.content.split('\n').filter(l => l.includes('Bridge') || l.includes('Reconciled') || l.includes('Synthesis'));
          bodyEl.innerHTML = metaBox + `
            <div class="space-y-2 text-xs">
              <div class="p-2.5 rounded bg-slate-800 border border-slate-700">
                <strong>Original Raw Content (x⁰):</strong>
                <p class="mt-1 text-[var(--muted-foreground,#94a3b8)]">${escapeHtml(rawEnt.content)}</p>
              </div>
              <div class="p-2.5 rounded bg-emerald-500/10 border border-emerald-500/30">
                <strong class="text-emerald-400">+ Added by Textual SGLD State Optimization (x⁰ → x¹):</strong>
                <div class="mt-1 space-y-1">${addedLines.length ? addedLines.map(l => formatMarkdownSimple(l)).join('') : '<em>Filtered out as Distractor / Ops Clutter (0 links added)</em>'}</div>
              </div>
            </div>
          `;
        } else {
          bodyEl.innerHTML = metaBox + formatMarkdownSimple(compEnt.content);
        }

        outEl.innerHTML = compEnt.wiki_links.map(lk =>
          `<button onclick="navigateByTitle('${lk.replace(/'/g, "\\'")}')" class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-medium">[[${escapeHtml(lk)}]]</button>`
        ).join('') || `<span class="text-[var(--muted-foreground,#94a3b8)]">None (Filtered Distractor / Clutter)</span>`;

        const participatingConcepts = WIKI_DATA.concepts.filter(c => (c.final_gates[id] ?? 0) >= 0.5);
        inEl.innerHTML = participatingConcepts.map(c =>
          `<button onclick="openWikiPage('concept', '${c.concept.concept_id}')" class="px-2 py-0.5 rounded bg-blue-500/15 text-blue-400 font-medium">[[${escapeHtml(c.concept.name)}]] (σ=${c.final_gates[id].toFixed(2)})</button>`
        ).join('') || `<span class="text-rose-400 font-mono">Rejected across all concepts (σ &lt; 0.5)</span>`;
      }
    }

    function renderPairwiseMatrix() {
      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const entities = WIKI_DATA.compiled_entities;
      const relMap = {};
      cObj.pairwise_relations.forEach(r => {
        relMap[`${r.i}__${r.j}`] = r;
        relMap[`${r.j}__${r.i}`] = r;
      });

      let html = `<table class="w-full border-collapse text-center text-[11px]"><thead><tr><th class="p-1 border border-[var(--border,#334155)]">Doc</th>`;
      entities.forEach((e, idx) => {
        html += `<th class="p-1 border border-[var(--border,#334155)] font-mono" title="${escapeHtml(e.title)}">D${idx + 1}</th>`;
      });
      html += `</tr></thead><tbody>`;

      entities.forEach((ei, i) => {
        html += `<tr><td class="p-1 border border-[var(--border,#334155)] text-left font-medium truncate max-w-[155px]" title="${escapeHtml(ei.title)}"><span class="font-mono text-[var(--muted-foreground,#94a3b8)] mr-1">D${i + 1}</span>${escapeHtml(ei.title.slice(0, 18))}…</td>`;
        entities.forEach((ej, j) => {
          if (i === j) {
            html += `<td class="p-1 border border-[var(--border,#334155)] bg-slate-800/50 font-mono text-[var(--muted-foreground,#94a3b8)]">—</td>`;
          } else {
            const rel = relMap[`${ei.entity_id}__${ej.entity_id}`];
            const synergy = rel ? Math.max(0, 1 - rel.coh) * rel.gate : 0;
            const bg = synergy > 0.45 ? 'bg-emerald-500/30 text-emerald-300 font-semibold'
                     : synergy > 0.15 ? 'bg-blue-500/20 text-blue-300'
                     : 'text-[var(--muted-foreground,#94a3b8)]';
            html += `<td onclick="inspectPair('${ei.entity_id}', '${ej.entity_id}')" class="p-1 border border-[var(--border,#334155)] cursor-pointer hover:opacity-80 font-mono ${bg}">${synergy.toFixed(2)}</td>`;
          }
        });
        html += `</tr>`;
      });
      html += `</tbody></table>`;
      document.getElementById('matrix-container').innerHTML = html;
    }

    function inspectPair(id1, id2) {
      const cObj = WIKI_DATA.concepts[currentConceptIdx];
      const rel = cObj.pairwise_relations.find(r =>
        (r.i === id1 && r.j === id2) || (r.i === id2 && r.j === id1)
      );
      if (!rel) return;
      document.getElementById('pair-inspector').innerHTML = `
        <div class="p-2.5 rounded bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] space-y-1.5 text-[var(--foreground,#f8fafc)]">
          <div class="font-bold text-emerald-400">${escapeHtml(rel.ti)}</div>
          <div class="text-center font-mono text-xs text-[var(--muted-foreground,#94a3b8)]">↕ Pairwise g_θ(x_i, x_j, w) ↕</div>
          <div class="font-bold text-blue-400">${escapeHtml(rel.tj)}</div>
        </div>
        <div class="space-y-1 font-mono">
          <div>Gate Product σ(a_i)σ(a_j): <strong>${rel.gate.toFixed(3)}</strong></div>
          <div>Coherence Deficit: <strong>${rel.coh.toFixed(3)}</strong></div>
          <div>Shared Mechanisms: <strong>${rel.shared.join(', ') || 'None'}</strong></div>
        </div>
        <div class="p-2.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-200">
          <strong>Relational Critic Diagnosis:</strong><br/>${escapeHtml(rel.critique)}
        </div>
      `;
    }

    function renderExperimentsTab() {
      const exps = WIKI_DATA.experiments || {};
      if (exps.exp1_iterative_sgld) {
        const rows = Object.entries(exps.exp1_iterative_sgld).map(([kLabel, m]) => `
          <tr class="border-b border-[var(--border,#334155)]">
            <td class="py-1 font-mono font-semibold">${kLabel}</td>
            <td class="py-1 font-mono">${m.attn_bce.toFixed(4)}</td>
            <td class="py-1 font-mono ${m.gen_error === 0 ? 'text-emerald-400 font-bold' : ''}">${m.gen_error.toFixed(3)}</td>
            <td class="py-1 font-mono">${m.final_energy_a.toFixed(4)}</td>
            <td class="py-1 font-mono">${m.final_energy_x.toFixed(4)}</td>
          </tr>
        `).join('');
        document.getElementById('exp1-table-wrap').innerHTML = `
          <table class="w-full text-left border-collapse">
            <thead><tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
              <th class="py-1">Steps</th><th class="py-1">Attn BCE ↓</th><th class="py-1">Gen Error ↓</th><th class="py-1">E(a) ↓</th><th class="py-1">E(x¹) ↓</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }

      if (exps.exp2_relational_vs_unary) {
        const rows = Object.entries(exps.exp2_relational_vs_unary).map(([arch, m]) => `
          <tr class="border-b border-[var(--border,#334155)]">
            <td class="py-1.5 font-semibold">${arch}</td>
            <td class="py-1.5 font-mono">${(m.attn_bce ?? 0).toFixed(4)}</td>
            <td class="py-1.5 font-mono ${(m.false_positive_traps ?? 0) === 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}">${m.false_positive_traps ?? 0} traps</td>
            <td class="py-1.5 font-mono ${m.gen_error === 0 ? 'text-emerald-400 font-bold' : 'text-rose-400'}">${m.gen_error.toFixed(3)}</td>
            <td class="py-1.5 font-mono">${m.links_created} links</td>
          </tr>
        `).join('');
        document.getElementById('exp2-table-wrap').innerHTML = `
          <table class="w-full text-left border-collapse">
            <thead><tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
              <th class="py-1">Architecture</th><th class="py-1">Attn BCE ↓</th><th class="py-1">False Traps ↓</th><th class="py-1">Gen Error ↓</th><th class="py-1">Links Wired</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }

      if (exps.exp3_context_transfer) {
        const rows = Object.entries(exps.exp3_context_transfer).map(([cond, m]) => `
          <tr class="border-b border-[var(--border,#334155)]">
            <td class="py-1 font-medium">${configurationLabel(cond)}</td>
            <td class="py-1 font-mono">${m.Generation_Error.toFixed(3)}</td>
            <td class="py-1 font-mono">${m.Attention_Error.toFixed(3)}</td>
          </tr>
        `).join('');
        document.getElementById('exp3-table-wrap').innerHTML = `
          <table class="w-full text-left border-collapse">
            <thead><tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
              <th class="py-1">Hand-configured weights</th><th class="py-1">Generation Error ↓</th><th class="py-1">Attention BCE ↓</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }

      if (exps.exp5_kl_objective_ablation) {
        const rows = Object.entries(exps.exp5_kl_objective_ablation).map(([mode, m]) => `
          <tr class="border-b border-[var(--border,#334155)]">
            <td class="py-1.5 font-medium">${configurationLabel(mode)}</td>
            <td class="py-1.5 font-mono text-emerald-400">${m.gold_energy.toFixed(4)}</td>
            <td class="py-1.5 font-mono ${m.sampled_energy < 0.15 ? 'text-emerald-400 font-bold' : 'text-amber-400'}">${m.sampled_energy.toFixed(4)}</td>
            <td class="py-1.5 font-mono text-rose-400">${m.random_energy.toFixed(4)}</td>
          </tr>
        `).join('');
        document.getElementById('exp5-table-wrap').innerHTML = `
          <table class="w-full text-left border-collapse">
            <thead><tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
              <th class="py-1">Optimizer configuration</th><th class="py-1">Gold Energy</th><th class="py-1">K=10 Sampled Energy</th><th class="py-1">Random Energy</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }

      if (exps.exp6_messy_trap_trajectories) {
        const rows = Object.entries(exps.exp6_messy_trap_trajectories).map(([label, trace]) => `
          <tr class="border-b border-[var(--border,#334155)]">
            <td class="py-1.5 font-mono font-medium">${label}</td>
            <td class="py-1.5 font-mono">${trace[0].toFixed(3)}</td>
            <td class="py-1.5 font-mono">${trace[2].toFixed(3)}</td>
            <td class="py-1.5 font-mono">${trace[5].toFixed(3)}</td>
            <td class="py-1.5 font-mono ${trace[10] < 0.2 ? 'text-emerald-400 font-bold' : (label.startsWith('Unary') ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold')}">${trace[10].toFixed(3)}</td>
          </tr>
        `).join('');
        document.getElementById('exp6-table-wrap').innerHTML = `
          <table class="w-full text-left border-collapse">
            <thead><tr class="border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]">
              <th class="py-1">Document &amp; Model Trajectory</th><th class="py-1">k=0 (Init Keyword Prior)</th><th class="py-1">k=2</th><th class="py-1">k=5</th><th class="py-1">k=10 (Converged)</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      } else {
        document.getElementById('exp6-table-wrap').innerHTML = `<div class="text-[var(--muted-foreground,#94a3b8)]">Switch to Suite 2 or Suite 3 (Messy 18-Doc KB) to view polysemous trap trajectories.</div>`;
      }
    }

    function renderLLMCallsTab() {
      const calls = WIKI_DATA.llm_calls || [];
      const badge = document.getElementById('llm-summary-badge');
      const wrap = document.getElementById('llm-calls-container');
      if (!calls.length) {
        badge.textContent = '0 Live LLM API Calls (Deterministic Symbolic Suite)';
        wrap.innerHTML = `<div class="p-3 rounded bg-[var(--background,#0f172a)] text-[var(--muted-foreground,#94a3b8)]">This experiment folder uses the deterministic symbolic/contextual critic. Switch the <strong>Experiment Folder</strong> dropdown at the top-right to <strong>Experiment Suite 3 (Live LLM)</strong> to inspect live LiteLLM / OpenAI API calls.</div>`;
        return;
      }
      badge.textContent = `${calls.length} Live LiteLLM Calls • ${WIKI_DATA.llm_model_info ? WIKI_DATA.llm_model_info.total_tokens : ''} Total Tokens`;
      wrap.innerHTML = calls.map(c => `
        <div class="p-3 rounded bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] space-y-2">
          <div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border,#334155)] pb-1.5">
            <span class="font-bold text-emerald-400">Call #${c.call_index}: ${escapeHtml(c.purpose)}</span>
            <span class="font-mono text-[var(--muted-foreground,#94a3b8)]">Model: ${escapeHtml(c.model)} • ${c.elapsed_sec}s • ${c.total_tokens} tokens</span>
          </div>
          <pre class="text-[11px] font-mono overflow-x-auto max-h-56 bg-slate-950 p-2.5 rounded text-slate-300">${escapeHtml(JSON.stringify(c.response, null, 2))}</pre>
        </div>
      `).join('');
    }

    function configurationLabel(key) {
      const labels = {
        Untrained_Critic: 'Small weights / small step',
        Trained_on_Generation: 'Generation-emphasized weights',
        Trained_on_Identification: 'Identification-emphasized weights',
        Trained_on_Both: 'Combined weights',
        With_KL_Objective: 'Step 1.25 / noise 0.02',
        No_KL_Objective: 'Step 0.08 / noise 0.25'
      };
      return labels[key] || escapeHtml(key.replace(/_/g, ' '));
    }

    function escapeHtml(str) {
      return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    window.addEventListener('DOMContentLoaded', initViewer);
  </script>
</body>
</html>
"""


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    suite_dirs = [
        ("exp1_clean_10papers_symbolic", "Suite 1: Clean 10-Paper KB (Symbolic Critic)"),
        ("exp2_messy_18docs_symbolic", "Suite 2: Messy 18-Doc KB (Unsupervised Contextual Critic)"),
        ("exp3_messy_18docs_live_llm", "Suite 3: Messy 18-Doc KB (Live OpenAI LLM — Unknown Concepts Discovered)"),
    ]

    suites_payload = []
    for folder_name, fallback_title in suite_dirs:
        b_path = os.path.join(base_dir, "experiments", folder_name, "wiki_output", "wiki_bundle.json")
        if os.path.exists(b_path):
            with open(b_path, "r", encoding="utf-8") as f:
                b = json.load(f)
            b.setdefault("experiment_suite", folder_name)
            b.setdefault("title", fallback_title)
            llm_log_path = os.path.join(base_dir, "experiments", folder_name, "wiki_output", "llm_calls_log.json")
            llm_calls = []
            if os.path.exists(llm_log_path):
                with open(llm_log_path, "r", encoding="utf-8") as lf:
                    llm_calls = json.load(lf)
            compact = build_compact_payload(b, llm_calls=llm_calls)
            suites_payload.append(compact)

            # Also write per-folder single-suite viewer
            single_html = HTML_TEMPLATE.replace("__SUITES_JSON_PAYLOAD__", json.dumps([compact]))
            with open(os.path.join(base_dir, "experiments", folder_name, "wiki_viewer.html"), "w", encoding="utf-8") as sf:
                sf.write(single_html)

    if not suites_payload:
        raise RuntimeError("No experiment bundles found!")

    master_html = HTML_TEMPLATE.replace("__SUITES_JSON_PAYLOAD__", json.dumps(suites_payload))
    viewer_path = os.path.join(base_dir, "wiki_viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(master_html)

    print(f"[OK] Built Master Multi-Experiment Viewer ({len(suites_payload)} suites, {len(master_html)} bytes) at: {viewer_path}")


if __name__ == "__main__":
    main()
