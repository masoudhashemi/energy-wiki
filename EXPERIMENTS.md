# Experimental Design, Three Experiment Suites & Empirical Results

This document defines the experimental protocol, explains the three self-contained experiment folders in `experiments/`, and reports the empirical results from both the **Unsupervised Contextual Symbolic Critic** and **Live LLM Calls (`gpt-6-luna` via LiteLLM)** over clean and messy Knowledge Bases, directly mirroring Section 5 of [Mordatch (2018), *Concept Learning with Energy-Based Models*](1811.02486v1.pdf).

---

## 1. The Three Experiment Suites (`experiments/`)

| Experiment Folder | Corpus (`x⁰`) | Are Concepts `w` Known in Advance? | Critic & Optimizer Engine | Viewer & Bundle |
| :--- | :--- | :--- | :--- | :--- |
| **[exp1_clean_10papers_symbolic/](experiments/exp1_clean_10papers_symbolic/)** | **10 Clean Research Papers** (Mordatch '18, Welling & Teh '11, Finn '16 GCL, Hinton '02 CD, Ziebart '08 MaxEnt, Finn '17 MAML, Grant '17, Yuksekgonul '24 TextGrad, Pryzant '23 ProTeGi, Santoro '17 RN) | Inferred from few-shot demonstration masks `X^demo` (Eq. 8) | Deterministic Symbolic Relation Network Critic (0 API calls) | [wiki_viewer.html](experiments/exp1_clean_10papers_symbolic/wiki_viewer.html) • [wiki_bundle.json](experiments/exp1_clean_10papers_symbolic/wiki_output/wiki_bundle.json) |
| **[exp2_messy_18docs_symbolic/](experiments/exp2_messy_18docs_symbolic/)** | **18 Messy Multi-Topic Docs** (10 research notes + 1 contradictory lab note + 1 entangled Redis/Wiki dev log + 4 polysemous keyword traps: HVAC energy, soil sampling, CSS gradients, HR adaptation + 2 Postgres/K8s ops clutter notes) | Inferred from raw text (NO oracle tags read by `E_θ`) | Unsupervised Contextual BOW + Pairwise Frustration Critic (0 API calls) | [wiki_viewer.html](experiments/exp2_messy_18docs_symbolic/wiki_viewer.html) • [wiki_bundle.json](experiments/exp2_messy_18docs_symbolic/wiki_output/wiki_bundle.json) |
| **[exp3_messy_18docs_live_llm/](experiments/exp3_messy_18docs_live_llm/)** | **Same 18 Messy Multi-Topic Docs** (untagged raw notes, contradictions, dev logs, polysemous traps, ops clutter) | **NO — Concepts `w` are UNKNOWN!** Jointly discovered with `a` via LLM | **Live `gpt-6-luna` via LiteLLM** (9 live LLM calls, 39,727 tokens) | [wiki_viewer.html](experiments/exp3_messy_18docs_live_llm/wiki_viewer.html) • [llm_calls_log.json](experiments/exp3_messy_18docs_live_llm/wiki_output/llm_calls_log.json) |

* **Master Multi-Experiment Explorer:** [wiki_viewer.html](wiki_viewer.html) at the project root lets you switch interactively between all 3 experiment folders.

---

## 2. Unsupervised Concept & Attention Discovery in Messy KBs (Suite 3: Live `gpt-6-luna`)

In a realistic messy Knowledge Base, **concepts `w` are not known ahead of time**, and surface keywords are unreliable because polysemous trap documents (e.g., `trap_hvac_building_energy`: *"Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling"*, `trap_agronomy_soil_sampling`: *"Agricultural Soil Core Sampling & Moisture Gradient Partitioning"*) share the exact words `"energy"`, `"sampling"`, `"partition"`, and `"gradient"`.

In [run_exp3_live_llm.py](experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py), `gpt-6-luna` executes **Unsupervised Alternating Textual SGLD** to solve $\min_{\mathbf{w}, \mathbf{a}, \mathbf{x}^1} E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$:

1. **Call #1 — Unsupervised Discovery of Unknown Concepts from Messy KB (`discover_unknown_concepts_via_llm`):**
   Given only the 18 untagged raw documents $\mathbf{x}^0$, `gpt-6-luna` proposes initial $k=0$ concept hypotheses $\mathbf{w}^{(m), 0}$ and refines them into 3 distinct latent scientific concepts while flagging candidate polysemous traps:
   * **Discovered Concept 1:** *Energy-based concept acquisition through pairwise relational reasoning* ([concept_1_relational_energy_concept_learning.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_1_relational_energy_concept_learning.md))
   * **Discovered Concept 2:** *Boltzmann energy learning with partition-function estimation and MCMC sampling* ([concept_2_energy_based_learning_and_sampling.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_2_energy_based_learning_and_sampling.md))
   * **Discovered Concept 3:** *Critic-guided optimization of natural-language artifacts via textual gradients* ([concept_3_textual_gradient_optimization.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_3_textual_gradient_optimization.md))
   * **Discovered Concept 4 (Additive Composition** $E(\mathbf{w}_1) + E(\mathbf{w}_2)$): *Composite Energy Bridge* ([concept_composite_sum.md](experiments/exp3_messy_18docs_live_llm/wiki_output/concepts/concept_composite_sum.md))
2. **Calls #2, #4, #6, #8 — Pairwise Relational Critic** $g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$ **& Textual Gradients** ($\nabla_{\mathbf{w}} E_\theta, \nabla_{\mathbf{a}} E_\theta$):
   `gpt-6-luna` evaluates pairwise interactions across candidate documents, emitting progressive concept refinements $\nabla_{\mathbf{w}} E_\theta$ and per-document attention critiques $\nabla_{\mathbf{a}} E_\theta$ that suppress polysemous traps from $k=0$ ($\sigma(a_i^0) \approx 0.58$) down to $k=10$ ($\sigma(a_i^{10}) \le 0.08$).
3. **Calls #3, #5, #7, #9 — Abstractive Wiki Page Compilation & Contradiction Reconciliation** ($\nabla_{\mathbf{x}^1} E_\theta$):
   `gpt-6-luna` executes state optimization $\arg\min_{\mathbf{x}^1} E_\theta(\mathbf{x}^1, \mathbf{a}^*, \mathbf{w}_x^*)$, writing 8–14 KB analytical Markdown concept pages that:
   * Unify the mathematical equations across attended documents with clickable `[[wiki-links]]`
   * Reconcile the contradiction between Welling & Teh (2011)'s asymptotic SGLD convergence guarantee and Lab Notebook #42's empirical $K=10$ short-chain failure without $\mathcal{L}^{\text{KL}}$ shaping
   * Extract the pairwise wiki-compiler textual gradient insight from `entangled_devlog_wiki_compiler` while discarding its unrelated Redis connection-pool noise
   * Document why each polysemous keyword trap was rejected by pairwise relational energy.

---

## 3. Quantitative Metrics

Following Section 5.2 (Figure 5 & Appendix Table, p. 11), we evaluate both contexts quantitatively:

1. **Identification Error — Attention Cross-Entropy (`attn_bce`) ↓:**



$$
\text{BCE}(\mathbf{a}^K, \mathbf{a}^{\text{gold}}) = -\frac{1}{N} \sum_{i=1}^N \Big[ a_i^{\text{gold}} \log \sigma(a_i^K) + (1 - a_i^{\text{gold}}) \log\big(1 - \sigma(a_i^K)\big) \Big]
$$



2. **Generation Error — Wiki Topology & Synthesis Error (`gen_error`) ↓:**



$$
\text{GenErr}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}}) = 0.65 \cdot \mathcal{E}_{\text{links}}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}}) + 0.35 \cdot \mathcal{E}_{\text{synthesis}}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}})
$$



3. **Residual Relational Energy (`final_energy`) ↓:**



$$
E_\theta(\mathbf{x}^{1,K}, \mathbf{a}^K, \mathbf{w}^*) \in \mathbb{R}^+
$$



---

## 4. Empirical Results Across All 6 Experiments (Messy 18-Doc KB)

### Experiment 1: Multi-Step Textual SGLD ($K \in \{1, 2, 5, 10, 20\}$) vs. One-Shot ($K = 1$)
* **Paper Reference:** Section 5.3 & Figure 7 (Left & Top-Right).

| SGLD Steps (`K`) | Attention BCE (`attn_bce`) ↓ | False-Positive Traps Attended ↓ | Avg Trap Gate `σ(a_trap^K)` ↓ | Generation Error (`gen_error`) ↓ | Generation Energy `E(x¹, a, wₓ)` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`K = 1` (One-Shot)** | `0.4220` | `0` | `0.3293` | `0.8700` | `6.8184` |
| **`K = 2`** | `0.3044` | `0` | `0.2447` | `0.7400` | `6.0493` |
| **`K = 5`** | `0.1529` | `0` | `0.1338` | `0.3500` | `3.5618` |
| **`K = 10` (Default)** | **`0.0785`** | **`0`** | **`0.0749`** | **`0.0000`** | **`0.1500`** |
| **`K = 20`** | **`0.0390`** | **`0`** | **`0.0394`** | **`0.0000`** | **`0.1500`** |

---

### Experiment 2: Pairwise Relation Network (`Relation_Network`) vs. Unary Pointwise RAG (`No_RN_Unary`) on Messy KB
* **Paper Reference:** Section 5.3 ("Effect of a Relational Architecture", `"no RN"` in Figure 7).

| Energy Model Architecture | Attention BCE (`attn_bce`) ↓ | False-Positive Polysemous Traps Attended ↓ | Generation Error (`gen_error`) ↓ | Bidirectional `[[wiki-links]]` Wired ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **`Relation_Network` (Pairwise `g_θ(xᵢ, xⱼ, w)`)** | **`0.0539`** | **`0 traps`** (All rejected!) | **`0.0000`** | **30 links** (Full 6 × 5 clique) |
| **`No_RN_Unary` (Pointwise `g_θ(xᵢ, w)`)** | `0.3507` | **`2 false-positive traps!`** (`HVAC` & `Soil`) | `0.6500` | **0 links** (No pairwise bridges) |

* **Key Finding on Messy KBs:** Standard Unary RAG (`No_RN_Unary`) is fooled by polysemous keyword traps (`trap_hvac_building_energy` and `trap_agronomy_soil_sampling` both match `"energy"`, `"sampling"`, `"partition"`, and `"gradient"`), resulting in **2 false-positive trap inclusions** (`attn_bce = 0.3507`) and **0 pairwise links**. Our Pairwise Relation Network checks contextual coherence across entity pairs $\sigma(a_i)\sigma(a_j)g_\theta(x_i, x_j, w)$, rejecting **all traps (`0` false positives, `attn_bce = 0.0539`)** and wiring all **30 bidirectional `[[wiki-links]]`**.

---

### Experiment 3: Transfer of Learning Between Identification and Generation Contexts
* **Paper Reference:** Section 5.2, Figure 5, and Appendix Table (p. 11).

| Condition (Mirroring Paper Table, p. 11) | Generation Error (`gen_error`) ↓ | Attention Error (`attn_bce`) ↓ | Paper Reference Values (Gen / Attn) |
| :--- | :---: | :---: | :---: |
| **`Untrained_Critic`** | `0.567` | `0.471` | `0.621` / `0.698` |
| **`Trained_on_Generation`** | **`0.000`** | `0.051` | `0.006` / `0.061` |
| **`Trained_on_Identification`** | **`0.000`** | `0.051` | `0.016` / `0.001` |
| **`Trained_on_Both`** | **`0.000`** | **`0.052`** | `0.007` / `0.001` |

---

### Experiment 4: Sharing & Interchangeability of Concept Codes (`wₐ ↔ wₓ`)
* **Paper Reference:** Section 5.2 & Figure 6.
* **Result:** Generation Error using $\mathbf{w}_x$ (`0.0000`) matches Generation Error when swapping in $\mathbf{w}_a$ (`0.0000`).

---

### Experiment 5: Effect of Meta-Training with vs. without Sampler Objective $\mathcal{L}_\pi^{\text{KL}}$
* **Paper Reference:** Section 5.3 & Figure 8 ("With KL Objective" vs. "No KL Objective").

| Training Objective | Gold Demonstration Energy (Green) ↓ | `K=10` SGLD Sampled Energy (Red) ↓ | Random Initial Energy (Black) |
| :--- | :---: | :---: | :---: |
| **`With_KL_Objective` (`L_p^ML + L_π^KL`)** | `0.0782` | **`0.0796`** (Matches Gold!) | `2.1962` |
| **`No_KL_Objective` (`L_p^ML` Only)** | `0.0782` | `1.3377` (Fails to converge in `K=10`) | `2.1962` |

---

### Experiment 6 (Messy KB Stress Test): Non-Monotonic Polysemous Trap Rejection Trajectory (`k = 0 → 2 → 10`)
* **Paper Reference:** Section 5.3 & Figure 7 (Top-Right: non-trivial multi-step attention feedback corrections).

| Document & Model Trajectory | `k=0` (Initial Surface Keyword Prior) | `k=2` | `k=5` | `k=10` (Converged Gate `σ(aᵢ¹⁰)`) | Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`RN_true_sgld_paper` (`welling_teh_2011_sgld`)** | `0.595` | `0.610` | `0.817` | **`0.938`** | Promoted (True Member) |
| **`RN_contradiction_lab_note` (`lab-note-42`)** | `0.599` | `0.771` | `0.892` | **`0.949`** | Promoted & Reconciled |
| **`RN_trap_hvac_energy` (Relation Network)** | `0.583` | `0.273` | `0.130` | **`0.079`** | **Trap Rejected by Pairwise `g_θ`!** |
| **`RN_trap_soil_sampling` (Relation Network)** | `0.592` | `0.245` | `0.105` | **`0.060`** | **Trap Rejected by Pairwise `g_θ`!** |
| **`Unary_NoRN_trap_hvac_energy` (Unary RAG)** | `0.596` | `0.787` | `0.888` | **`0.933`** | **False Positive (Fooled by Keywords!)** |
| **`Unary_NoRN_trap_soil_sampling` (Unary RAG)** | `0.601` | `0.789` | `0.889` | **`0.933`** | **False Positive (Fooled by Keywords!)** |
