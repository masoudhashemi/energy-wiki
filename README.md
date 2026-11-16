# Energy-Based LLM Wiki (`energy-wiki`)

This repository adapts **[Concept Learning with Energy-Based Models](1811.02486v1.pdf)** (Igor Mordatch, OpenAI, 2018) to autonomously discover latent concepts `w`, identify connected document subgraphs `a`, compile interlinked **LLM Wikis** (`x¹`) over messy Knowledge Bases (`x⁰`), and synthesize new documents via **$K$-Step Textual Stochastic Gradient Langevin Dynamics (Textual SGLD)**.

---

## Documentation Map

Each document covers a distinct aspect of the project without duplication:

1. **[PROBLEM_AND_FORMULATION.md](PROBLEM_AND_FORMULATION.md)** — Problem statement, theoretical background (EBMs, LLM Wikis, Textual Gradients), unified ontology & entity-relationship tables, and complete mathematical formulation ($E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$, $K$-step Textual SGLD, and meta-losses $\mathcal{L}_p^{\text{ML}} + \mathcal{L}_\pi^{\text{KL}}$).
2. **[EXPERIMENTS.md](EXPERIMENTS.md)** — Design of the 6 benchmark experiments (mirroring Section 5 of Mordatch, 2018), quantitative metric definitions (`attn_bce`, `gen_error`), and cross-suite benchmark comparison.
3. **[GENERATIVE_ENERGY_SYNTHESIS.md](GENERATIVE_ENERGY_SYNTHESIS.md)** — Theory and formulation of the 5 generative and evaluative ways to use learned energy functions $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$.
4. **[wiki_viewer.html](wiki_viewer.html)** — **Master Multi-Experiment Interactive HTML Explorer** with a top-right dropdown to switch between Experiment Suites and inspect the knowledge graph, $K=10$ SGLD step scrubber, compiled wiki pages (`x⁰ → x¹` diff), $g_\theta$ pairwise matrix, and live LLM call logs.

---

## Experiment Suites (`experiments/`)

Each experiment folder is self-contained with its own `raw/` inputs, `wiki_output/` artifacts, runner script, and suite-specific `README.md`:

| Suite Folder | Corpus & Setting | Engine | Suite Report & Viewer |
| :--- | :--- | :--- | :--- |
| **[exp1_clean_10papers_symbolic/](experiments/exp1_clean_10papers_symbolic/)** | 10 clean AI/ML papers (`raw/*.md`), few-shot concept inference | Deterministic Symbolic Relation Network Critic | [README.md](experiments/exp1_clean_10papers_symbolic/README.md) • [Viewer](experiments/exp1_clean_10papers_symbolic/wiki_viewer.html) |
| **[exp2_messy_18docs_symbolic/](experiments/exp2_messy_18docs_symbolic/)** | 18 messy multi-topic docs (contradictions, dev logs, 4 polysemous keyword traps, 2 ops clutter notes), no oracle tags | Unsupervised Contextual BOW + Pairwise Frustration Critic | [README.md](experiments/exp2_messy_18docs_symbolic/README.md) • [Viewer](experiments/exp2_messy_18docs_symbolic/wiki_viewer.html) |
| **[exp3_messy_18docs_live_llm/](experiments/exp3_messy_18docs_live_llm/)** | 18 messy multi-topic docs, **concepts `w` unknown in advance** | Domain-Agnostic Live `gpt-6-luna` via LiteLLM (9 calls) | [README.md](experiments/exp3_messy_18docs_live_llm/README.md) • [Viewer](experiments/exp3_messy_18docs_live_llm/wiki_viewer.html) |
| **[exp4_generative_data_synthesis_live_llm/](experiments/exp4_generative_data_synthesis_live_llm/)** | Generative synthesis of new bridge papers, adversarial hard negatives, and cross-domain re-enactment | Generative Textual SGLD with Live `gpt-6-luna` (8 calls) | [README.md](experiments/exp4_generative_data_synthesis_live_llm/README.md) • [Results JSON](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/generative_results.json) |

---

## Quick Start (`uv`)

```powershell
# 1. Run Suite 1 (Clean 10-Paper Symbolic Benchmark)
uv run --no-sync python experiments/exp1_clean_10papers_symbolic/run_exp1.py

# 2. Run Suite 2 (Messy 18-Doc Unsupervised Contextual Benchmark)
uv run --no-sync python experiments/exp2_messy_18docs_symbolic/run_exp2.py

# 3. Run Suite 3 (Domain-Agnostic Unsupervised Wiki Compilation with Live gpt-6-luna)
uv run --no-sync python experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py

# 4. Run Suite 4 (Generative Data Synthesis with Live gpt-6-luna)
uv run --no-sync python experiments/exp4_generative_data_synthesis_live_llm/run_exp4_generative_luna.py

# 5. Rebuild Interactive HTML Viewers
uv run --no-sync python build_viewer.py
```
