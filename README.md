# Energy-Based LLM Wiki with Textual Gradients

> **Blog Post & Interactive Walkthrough:** **[https://masoudhashemi.github.io/energy-wiki/](https://masoudhashemi.github.io/energy-wiki/)** ([`docs/index.html`](docs/index.html))

This project adapts **[Concept Learning with Energy-Based Models](1811.02486v1.pdf)** (Igor Mordatch, OpenAI, 2018) to autonomously discover concepts `w`, identify connected document subgraphs `a`, and compile interlinked **LLM Wikis** (`x¹`) over messy, multi-topic Knowledge Bases (`x⁰`) using **Textual Gradient Descent** (ProTeGi / TextGrad).

---

## Core Mathematical Formulation

1. **Relational Energy Function (Eq. 4):**



$$
E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w}) = f_\theta\left( \sum_{t, i, j} \sigma(\mathbf{a}_i)\sigma(\mathbf{a}_j) \cdot g_\theta(\mathbf{x}_i^t, \mathbf{x}_j^t, \mathbf{w}), \, \mathbf{w} \right)^2
$$



2. **Three Unified Execution-Time Operations via **$K$**-Step Textual SGLD:**
   * **Concept Inference (Eq. 8):**



$$
\mathbf{w}^*_\theta(X^{\text{demo}}) = \arg\min_{\mathbf{w}} \mathbb{E}_{(\mathbf{x}, \mathbf{a}) \sim X^{\text{demo}}} \big[ E_\theta(\mathbf{x}^1, \mathbf{a}, \mathbf{w}_x) + E_\theta(\mathbf{x}^0, \mathbf{a}, \mathbf{w}_a) \big]
$$



   * **Entity Identification / Attention Discovery (Eq. 2b):**



$$
\mathbf{a}^*(\mathbf{x}^0) = \arg\min_{\mathbf{a}} E_\theta(\mathbf{x}^0, \mathbf{a}, \mathbf{w}_a)
$$



   * **Wiki Page & Link Generation (Eq. 2a):**



$$
\mathbf{x}^{1*}(\mathbf{a}) = \arg\min_{\mathbf{x}^1} E_\theta(\mathbf{x}^1, \mathbf{a}, \mathbf{w}_x)
$$



3. **Meta-Level Schema Optimization (Eq. 9 & Algorithm 1):**



$$
\min_\theta \; \mathcal{L}_{p(\theta)}^{\text{ML}}\big(X^{\text{train}}, \mathbf{w}^*_\theta(X^{\text{demo}})\big) + \mathcal{L}_{\pi(\theta)}^{\text{KL}}\big(X^{\text{train}}, \mathbf{w}^*_\theta(X^{\text{demo}})\big)
$$



---

## Project & Experiment Folder Structure

* **[1811.02486v1.pdf](1811.02486v1.pdf)** — Original paper: *Concept Learning with Energy-Based Models* (Mordatch, 2018).
* **[PROBLEM_AND_FORMULATION.md](PROBLEM_AND_FORMULATION.md)** — Complete problem definition, paper breakdown, conceptual mapping table, and mathematical formulation for $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$, $K$-step Textual SGLD, unsupervised joint `(w, a)` discovery on messy KBs, and meta-optimization.
* **[EXPERIMENTS.md](EXPERIMENTS.md)** — Detailed design and empirical results across the experiment folders and benchmark experiments.
* **[GENERATIVE_ENERGY_SYNTHESIS.md](GENERATIVE_ENERGY_SYNTHESIS.md)** — Mathematical formulation and experimental evaluation of **5 generative & evaluative ways to use learned energy functions** $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ (synthesizing missing bridge documents, compositional data generation, adversarial hard-negative synthesis, and cross-domain concept re-enactment).
* **[wiki_viewer.html](wiki_viewer.html)** — **Master Multi-Experiment Interactive Explorer for Suites 1–3**, with a top-right dropdown and 5 tabs. Exp5's measured synthesis walkthrough is in the [article](https://masoudhashemi.github.io/energy-wiki/#measured-loop).
  1. **Knowledge Graph & SGLD Step Scrubber** (`k = 0 ... 10`)
  2. **Wiki Browser** (`x⁰ → x¹` Diff & clickable `[[wiki-links]]`)
  3. **Pairwise Relation Matrix** $g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$
  4. **Experiments Benchmark Dashboard** (Exps 1–6)
  5. **Live LLM Call Log** (`gpt-6-luna` JSON prompts & responses)

### Self-Contained Experiment Folders (`experiments/`)

Suites 1–3 contain `raw/` inputs, a compiled wiki, runner, README, and standalone viewer. Suite 4 stores generated-document examples; Suite 5 reuses Suite 4's sources read-only and stores each measured run under `runs/`.

1. **[experiments/exp1_clean_10papers_symbolic/](experiments/exp1_clean_10papers_symbolic/)**
   * **Corpus:** 10 clean foundational research papers (`raw/*.md`).
   * **Engine:** Deterministic Symbolic Relation Network Critic ([run_exp1.py](experiments/exp1_clean_10papers_symbolic/run_exp1.py)).
   * **Outputs:** [wiki_viewer.html](experiments/exp1_clean_10papers_symbolic/wiki_viewer.html), [index.md](experiments/exp1_clean_10papers_symbolic/wiki_output/index.md), [wiki_bundle.json](experiments/exp1_clean_10papers_symbolic/wiki_output/wiki_bundle.json).

2. **[experiments/exp2_messy_18docs_symbolic/](experiments/exp2_messy_18docs_symbolic/)**
   * **Corpus:** 18 messy, untagged multi-topic documents (`raw/*.md`) mixing research notes, a contradictory lab note (`contradiction_note_mcmc_mixing`), an entangled Redis/Wiki dev log (`entangled_devlog_wiki_compiler`), 4 polysemous keyword traps (`trap_hvac_building_energy`, `trap_agronomy_soil_sampling`, `trap_css_gradient_ui_bug`, `trap_hr_org_adaptation`), and 2 ops clutter notes (`clutter_postgres_autovacuum`, `clutter_k8s_ingress_tls`).
   * **Engine:** Unsupervised Contextual BOW + Pairwise Frustration Critic ([run_exp2.py](experiments/exp2_messy_18docs_symbolic/run_exp2.py)).
   * **Outputs:** [wiki_viewer.html](experiments/exp2_messy_18docs_symbolic/wiki_viewer.html), [index.md](experiments/exp2_messy_18docs_symbolic/wiki_output/index.md), [wiki_bundle.json](experiments/exp2_messy_18docs_symbolic/wiki_output/wiki_bundle.json).

3. **[experiments/exp3_messy_18docs_live_llm/](experiments/exp3_messy_18docs_live_llm/)**
   * **Corpus:** Same 18 messy, untagged multi-topic documents (`raw/*.md`), assuming **concepts `w` are UNKNOWN in advance**.
   * **Engine:** **Live `gpt-6-luna` via LiteLLM** ([run_exp3_live_llm.py](experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py) — 9 live LLM calls, 39,727 tokens) performing unsupervised concept discovery ($\nabla_{\mathbf{w}} E_\theta$), pairwise relational trap rejection ($g_\theta, \nabla_{\mathbf{a}} E_\theta$), and abstractive Markdown wiki compilation ($\nabla_{\mathbf{x}^1} E_\theta$).
   * **Outputs:**
     * Viewer: [wiki_viewer.html](experiments/exp3_messy_18docs_live_llm/wiki_viewer.html)
     * Live LLM Call Log: [llm_calls_log.json](experiments/exp3_messy_18docs_live_llm/wiki_output/llm_calls_log.json)
     * Master Catalog: [index.md](experiments/exp3_messy_18docs_live_llm/wiki_output/index.md)
     * `gpt-6-luna` Compiled Concept Pages:
       * [concept_1_relational_energy_concept_learning.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_1_relational_energy_concept_learning.md)
       * [concept_2_energy_based_learning_and_sampling.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_2_energy_based_learning_and_sampling.md)
       * [concept_3_textual_gradient_optimization.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_3_textual_gradient_optimization.md)
       * [concept_composite_sum.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_composite_sum.md)

4. **[experiments/exp4_generative_data_synthesis_live_llm/](experiments/exp4_generative_data_synthesis_live_llm/)**
   * **Task:** Live generative explorations: a two-revision bridge document, a prompted hard negative, and a prompted cross-domain analogy ([run_exp4_generative_luna.py](experiments/exp4_generative_data_synthesis_live_llm/run_exp4_generative_luna.py) — current saved run: 8 calls, 16,965 tokens). Exp5 below is the main iterative-synthesis example in the article.
   * **Outputs & Documentation:**
     * Report: [experiments/exp4_generative_data_synthesis_live_llm/README.md](experiments/exp4_generative_data_synthesis_live_llm/README.md)
     * Telemetry & Call Log: [generative_results.json](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/generative_results.json) • [llm_calls_log.json](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/llm_calls_log.json)
     * Generated Synthetic Documents (`wiki_output/generated_docs/`):
       * [gen_bridge_boltzmann_textual_sgld.md](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/generated_docs/gen_bridge_boltzmann_textual_sgld.md) *(Task 1: Two-revision bridge; critic totals $9.80 \to 4.40 \to 1.90$)*
       * [gen_trap_biochem_allostery.md](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/generated_docs/gen_trap_biochem_allostery.md) *(Task 2: Biochemistry hard negative; keyword score 0.20, pairwise deficit 4.80, composite deficit 9.40)*
       * [gen_transfer_distributed_consensus.md](experiments/exp4_generative_data_synthesis_live_llm/wiki_output/generated_docs/gen_transfer_distributed_consensus.md) *(Task 3: Prompted distributed-consensus analogy; isomorphism deficit 0.55)*

5. **[experiments/exp5_measured_live_loop/](experiments/exp5_measured_live_loop/)**
   * **Task:** Conditional bridge synthesis: fix `D`, `w1`, `a1`, `w2`, `a2` and optimize only `x_new`. The same critic separately evaluates each concept–mask pair; the composite is `E1 + E2`.
   * **Runner & protocol:** [run_exp5.py](experiments/exp5_measured_live_loop/run_exp5.py) · [README](experiments/exp5_measured_live_loop/README.md).
   * **Measured run:** Ten actual document revisions, 32 model calls, 144,168 tokens. Composite critic scores: `17 → 3.35 → 0.50 → 0 → 0.25 → 0 → 0 → 0 → 0 → 0.50 → 0`. All fixed-condition hashes remained unchanged. These ratings do not establish independent correctness or convergence.
   * **Outputs:** [Final generated document](experiments/exp5_measured_live_loop/runs/conditional_prepared/final_document.md) · [Report](experiments/exp5_measured_live_loop/runs/conditional_prepared/REPORT.md) · [All 11 saved states and critiques](experiments/exp5_measured_live_loop/runs/conditional_prepared/trajectory.json) · [Manifest](experiments/exp5_measured_live_loop/runs/conditional_prepared/manifest.json) · [Full calls](experiments/exp5_measured_live_loop/runs/conditional_prepared/calls.jsonl).
   * **Presentation:** Main [synthesis walkthrough](https://masoudhashemi.github.io/energy-wiki/#measured-loop), with every revision selectable and measured component-energy curves. Rebuild the article excerpt from the saved run with `uv run --no-sync python build_exp5_walkthrough.py` (no API calls).

---

## Running the Suites via `uv`

```powershell
# 1. Run Experiment Suite 1 (Clean 10-Paper KB)
uv run --no-sync python experiments/exp1_clean_10papers_symbolic/run_exp1.py

# 2. Run Experiment Suite 2 (Messy 18-Doc Unsupervised Contextual Critic)
uv run --no-sync python experiments/exp2_messy_18docs_symbolic/run_exp2.py

# 3. Run Experiment Suite 3 (Domain-Agnostic Live LLM with gpt-6-luna via .env and LiteLLM)
uv run --no-sync python experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py

# 4. Run Experiment Suite 4 (live generative explorations)
uv run --no-sync python experiments/exp4_generative_data_synthesis_live_llm/run_exp4_generative_luna.py

# 5. Prepare a NEW conditional-synthesis run (no API calls; choose an unused path)
uv run --no-sync python experiments/exp5_measured_live_loop/run_exp5.py --out experiments/exp5_measured_live_loop/runs/conditional_next

# Execute that prepared run when ready (paid API calls)
uv run --no-sync python experiments/exp5_measured_live_loop/run_exp5.py --out experiments/exp5_measured_live_loop/runs/conditional_next --resume --live
```
