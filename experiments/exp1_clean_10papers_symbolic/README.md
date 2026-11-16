# Experiment Suite 1: Clean 10-Paper Knowledge Base (Symbolic Relational Critic)

This folder (`experiments/exp1_clean_10papers_symbolic/`) contains **Experiment Suite 1**, which evaluates the Energy-Based LLM Wiki framework ([Mordatch, 2018](../../1811.02486v1.pdf)) on a clean, curated 10-paper research Knowledge Base using a deterministic Symbolic/Contextual Relation Network Critic.

---

## 1. Inputs (`raw/*.md` and Demonstration Setup)

### 1.1 Raw Knowledge Base Documents ($\mathbf{x}^0$)
Located in [raw/](./raw/), consisting of **10 foundational AI/ML research papers** with zero distractors or operational clutter:
1. **[mordatch_2018_ebm.md](./raw/mordatch_2018_ebm.md)** — *Concept Learning with Energy-Based Models* (Mordatch, 2018)
2. **[welling_teh_2011_sgld.md](./raw/welling_teh_2011_sgld.md)** — *Bayesian Learning via Stochastic Gradient Langevin Dynamics* (Welling & Teh, 2011)
3. **[finn_2016_gcl.md](./raw/finn_2016_gcl.md)** — *Guided Cost Learning: Deep Inverse Optimal Control* (Finn et al., 2016)
4. **[hinton_2002_cd.md](./raw/hinton_2002_cd.md)** — *Training Products of Experts by Minimizing Contrastive Divergence* (Hinton, 2002)
5. **[ziebart_2008_maxent_irl.md](./raw/ziebart_2008_maxent_irl.md)** — *Maximum Entropy Inverse Reinforcement Learning* (Ziebart et al., 2008)
6. **[finn_2017_maml.md](./raw/finn_2017_maml.md)** — *Model-Agnostic Meta-Learning for Fast Adaptation (MAML)* (Finn et al., 2017)
7. **[grant_2017_meta_concepts.md](./raw/grant_2017_meta_concepts.md)** — *Concept Acquisition via Meta-Learning from Positive Examples* (Grant et al., 2017)
8. **[yuksekgonul_2024_textgrad.md](./raw/yuksekgonul_2024_textgrad.md)** — *TextGrad: Automatic Differentiation via Text* (Yuksekgonul et al., 2024)
9. **[pryzant_2023_protegi.md](./raw/pryzant_2023_protegi.md)** — *ProTeGi: Prompt Optimization with Textual Gradients* (Pryzant et al., 2023)
10. **[santoro_2017_relation_networks.md](./raw/santoro_2017_relation_networks.md)** — *A Simple Neural Network Module for Relational Reasoning* (Santoro et al., 2017)

### 1.2 Concept Supervision Assumption
Following Section 4 of Mordatch (2018), positive demonstration masks $X^{\text{demo}} = \{(\mathbf{x}^0, \mathbf{a}^{\text{demo}})\}$ are provided for Concept 1 and Concept 2, from which execution-time Concept Inference (Equation 8) infers $\mathbf{w}^* = [\mathbf{w}_a^*, \mathbf{w}_x^*]$.

---

## 2. Scripts Used & How to Run

### Scripts Involved
* **[run_exp1.py](./run_exp1.py)** — Self-contained runner script in this folder that executes Suite 1 end-to-end.
* **[setup_exp1_and_exp2.py](../../setup_exp1_and_exp2.py)** (`setup_exp1_clean()`) — Populates `raw/` with the 10 clean foundational papers, runs $K=10$ Textual SGLD loops, runs Experiments 1–5, and compiles `wiki_output/` and `wiki_viewer.html`.
* **[energy_wiki.py](../../energy_wiki.py)** — Core `RelationalWikiEnergyModel`, `TextualSGLDOptimizer`, `compute_meta_losses`, and `export_markdown_wiki`.
* **[run_experiments.py](../../run_experiments.py)** — Quantitative benchmark functions (`run_exp1_iterative_sgld` through `run_exp5_kl_objective_ablation`).
* **[build_viewer.py](../../build_viewer.py)** — Builds the standalone interactive HTML viewer.

### Execution Command
```powershell
uv run --no-sync python experiments/exp1_clean_10papers_symbolic/run_exp1.py
```

---

## 3. Method Used

Implemented in [energy_wiki.py](../../energy_wiki.py) and invoked by [run_exp1.py](./run_exp1.py):

1. **Relational Energy Function (Eq. 4):**


$$
E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w}) = f_\theta\left( \sum_{i, j} \sigma(\mathbf{a}_i)\sigma(\mathbf{a}_j) \cdot g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w}), \, \mathbf{w} \right)^2
$$


   where $g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$ evaluates normalized mechanism stem overlap and contextual bag-of-words cosine similarity between papers $i$ and $j$, plus whether bidirectional `[[wiki-links]]` exist in $\mathbf{x}^1$.
2. **Three **$K=10$** Textual SGLD Loops:**
   * **Concept Inference (`infer_concept_code`, Eq. 8):** Extracts shared mechanisms across attended demonstration papers via $\nabla_{\mathbf{w}}^{\text{text}} E_\theta$ to populate $\mathbf{w}_a$ and $\mathbf{w}_x$.
   * **Entity Identification (`optimize_attention_with_history`, Eq. 3b):** Updates continuous attention logits $\mathbf{a}_i^k = \mathbf{a}_i^{k-1} - \frac{\alpha}{2}\nabla_{a_i} E_\theta + \omega^k$ over $K=10$ steps.
   * **Wiki State Compilation (`optimize_wiki_state_with_history`, Eq. 3a):** Executes discrete textual gradient directives $\nabla_{\mathbf{x}^1}^{\text{text}} E_\theta$ (`CREATE_CONCEPT_PAGE`, `UPDATE_CONCEPT_LINKS`, `LINK_DEFICIT: Connect [[A]] <-> [[B]]`) up to step budget $\lfloor 3\alpha \rfloor$ per iteration until $E_\theta \to 0$.
3. **Additive Concept Composition (`compose_concepts`):**
   Minimizes $E(\mathbf{x}, \mathbf{a}, \mathbf{w}_1) + E(\mathbf{x}, \mathbf{a}, \mathbf{w}_2)$ to synthesize cross-concept bridge pages.

---

## 3. Outputs (`wiki_output/` and Viewer)

* **Standalone Interactive Viewer:** [wiki_viewer.html](./wiki_viewer.html)
* **Master Wiki Catalog:** [wiki_output/index.md](./wiki_output/index.md)
* **Full Telemetry & Experiment Bundle:** [wiki_output/wiki_bundle.json](./wiki_output/wiki_bundle.json)
* **Compiled Concept Pages (`wiki_output/concepts/`):**
  * [concept_ebm_sampling.md](./wiki_output/concepts/concept_ebm_sampling.md) (*Boltzmann Energy & Partition Sampling*)
  * [concept_inner_loop_textgrad.md](./wiki_output/concepts/concept_inner_loop_textgrad.md) (*Execution-Time Inner-Loop & Textual Gradients*)
  * [concept_relational_reasoning.md](./wiki_output/concepts/concept_relational_reasoning.md) (*Pairwise Relational & Compositional Architectures*)
  * [concept_composite_ebm_meta.md](./wiki_output/concepts/concept_composite_ebm_meta.md) (*Composite: Energy Sampling + Inner-Loop Meta-Adaptation*)
* **Compiled Entity Pages (`wiki_output/entities/`):** 10 interlinked Markdown files with bidirectional `[[wiki-links]]`.

---

## 4. Empirical Outcomes & Benchmark Results

1. **Multi-Step SGLD Convergence (Exp 1):**
   * At $K=1$ (one-shot synthesis), generation error is `0.870` (`E_x = 6.039`).
   * By $K=10$, generation error reaches **`0.000`** (`E_x = 0.0123`) and attention BCE drops from `0.5017` to **`0.075`**.
2. **Relation Network vs. Unary Ablation (Exp 2):**
   * Pairwise `Relation_Network` wires all **20 bidirectional `[[wiki-links]]`** (`0.000` generation error).
   * Unary `No_RN_Unary` fails to detect pairwise link deficits (`0.650` generation error, `0` pairwise links wired).
3. **Cross-Context Transfer & Code Interchangeability (Exp 3 & Exp 4):**
   * Calibrating $E_\theta$ on generation transfers zero-shot to identification (`0.051` attention BCE vs. `0.471` untrained).
   * Swapping $\mathbf{w}_a$ into $\mathbf{w}_x$ yields `0.000` generation error.
4. **Sampler KL Objective **$\mathcal{L}_\pi^{\text{KL}}$** Ablation (Exp 5):**
   * Training with $\mathcal{L}_p^{\text{ML}} + \mathcal{L}_\pi^{\text{KL}}$ reduces $K=10$ sampled energy to `0.0421` (matching gold `0.0126`), whereas $\mathcal{L}_p^{\text{ML}}$ alone leaves $K=10$ samples at `0.7409`.
