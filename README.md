# Energy-Based LLM Wiki with Textual Gradients

This project adapts **[Concept Learning with Energy-Based Models](1811.02486v1.pdf)** (Igor Mordatch, OpenAI, 2018) to autonomously discover concepts $\mathbf{w}$, identify connected document subgraphs $\mathbf{a}$, and compile interlinked **LLM Wikis** ($\mathbf{x}^1$) over messy, multi-topic Knowledge Bases ($\mathbf{x}^0$) using **Textual Gradient Descent** (ProTeGi / TextGrad).

---

## Project & Experiment Folder Structure

* **[1811.02486v1.pdf](1811.02486v1.pdf)** — Original paper: *Concept Learning with Energy-Based Models* (Mordatch, 2018).
* **[PROBLEM_AND_FORMULATION.md](PROBLEM_AND_FORMULATION.md)** — Complete problem definition, paper breakdown, conceptual mapping table, and mathematical formulation for $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$, $K$-step Textual SGLD, unsupervised joint $(\mathbf{w}, \mathbf{a})$ discovery on messy KBs, and meta-optimization ($\mathcal{L}_p^{\text{ML}} + \mathcal{L}_\pi^{\text{KL}}$).
* **[EXPERIMENTS.md](EXPERIMENTS.md)** — Detailed design and empirical results across all 3 experiment folders and 6 benchmark experiments.
* **[wiki_viewer.html](wiki_viewer.html)** — **Master Multi-Experiment Interactive Explorer** with a top-right dropdown to switch between all 3 Experiment Suites and 5 tabs:
  1. **Knowledge Graph & SGLD Step Scrubber ($k = 0 \dots 10$)**
  2. **Wiki Browser ($\mathbf{x}^0 \to \mathbf{x}^1$ Diff & clickable `[[wiki-links]]`)**
  3. **Pairwise Relation Matrix $g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$**
  4. **Experiments Benchmark Dashboard (Exps 1–6)**
  5. **Live LLM Call Log (`gpt-6-luna` JSON prompts & responses)**

### Self-Contained Experiment Folders (`experiments/`)

1. **[experiments/exp1_clean_10papers_symbolic/](experiments/exp1_clean_10papers_symbolic/)**
   * **Corpus:** 10 clean foundational research papers (`raw/*.md`).
   * **Engine:** Deterministic Symbolic Relation Network Critic.
   * **Outputs:** [wiki_viewer.html](experiments/exp1_clean_10papers_symbolic/wiki_viewer.html), [index.md](experiments/exp1_clean_10papers_symbolic/wiki_output/index.md), [wiki_bundle.json](experiments/exp1_clean_10papers_symbolic/wiki_output/wiki_bundle.json).

2. **[experiments/exp2_messy_18docs_symbolic/](experiments/exp2_messy_18docs_symbolic/)**
   * **Corpus:** 18 messy, untagged multi-topic documents (`raw/*.md`) mixing research notes, a contradictory lab note (`contradiction_note_mcmc_mixing`), an entangled Redis/Wiki dev log (`entangled_devlog_wiki_compiler`), 4 polysemous keyword traps (`trap_hvac_building_energy`, `trap_agronomy_soil_sampling`, `trap_css_gradient_ui_bug`, `trap_hr_org_adaptation`), and 2 ops clutter notes (`clutter_postgres_autovacuum`, `clutter_k8s_ingress_tls`).
   * **Engine:** Unsupervised Contextual BOW + Pairwise Frustration Critic (operates purely on raw text without reading `tags`).
   * **Outputs:** [wiki_viewer.html](experiments/exp2_messy_18docs_symbolic/wiki_viewer.html), [index.md](experiments/exp2_messy_18docs_symbolic/wiki_output/index.md), [wiki_bundle.json](experiments/exp2_messy_18docs_symbolic/wiki_output/wiki_bundle.json).

3. **[experiments/exp3_messy_18docs_live_llm/](experiments/exp3_messy_18docs_live_llm/)**
   * **Corpus:** Same 18 messy, untagged multi-topic documents (`raw/*.md`), assuming **concepts $\mathbf{w}$ are UNKNOWN in advance**.
   * **Engine:** **Live `gpt-6-luna` via LiteLLM** ([run_exp3_live_llm.py](experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py) — 9 live LLM calls, 39,372 tokens) performing unsupervised concept discovery ($\nabla_{\mathbf{w}} E_\theta$), pairwise relational trap rejection ($g_\theta, \nabla_{\mathbf{a}} E_\theta$), and abstractive Markdown wiki compilation ($\nabla_{\mathbf{x}^1} E_\theta$).
   * **Outputs:**
     * Viewer: [wiki_viewer.html](experiments/exp3_messy_18docs_live_llm/wiki_viewer.html)
     * Live LLM Call Log: [llm_calls_log.json](experiments/exp3_messy_18docs_live_llm/wiki_output/llm_calls_log.json)
     * Master Catalog: [index.md](experiments/exp3_messy_18docs_live_llm/wiki_output/index.md)
     * `gpt-6-luna` Compiled Concept Pages:
       * [concept_1_energy_based_trajectory_learning.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_1_energy_based_trajectory_learning.md)
       * [concept_2_demonstration_conditioned_inner_loop_adaptation.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_2_demonstration_conditioned_inner_loop_adaptation.md)
       * [concept_3_pairwise_relational_concept_representation.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_3_pairwise_relational_concept_representation.md)
       * [concept_4_llm_composite_bridge.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_4_llm_composite_bridge.md)

---

## Running the Suites via `uv`

```powershell
# 1. Run symbolic/contextual messy KB experiments & regenerate Suite 1 + Suite 2
uv run --no-sync python run_experiments.py
uv run --no-sync python setup_exp1_and_exp2.py

# 2. Run Suite 3 (Live LLM with gpt-6-luna via .env and LiteLLM)
uv run --no-sync python experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py

# 3. Rebuild the Master Multi-Experiment Viewer (wiki_viewer.html)
uv run --no-sync python build_viewer.py
```
