# Experimental Design, Metrics & Cross-Suite Benchmark Summary

This document defines the quantitative evaluation metrics and the 6 benchmark experiments mirroring Section 5 of [Mordatch (2018)](1811.02486v1.pdf), implemented in [run_experiments.py](run_experiments.py).

For suite-specific inputs, generated files, and detailed logs, see the individual experiment reports:
* **Suite 1 (Clean 10-Paper Symbolic):** [experiments/exp1_clean_10papers_symbolic/README.md](experiments/exp1_clean_10papers_symbolic/README.md)
* **Suite 2 (Messy 18-Doc Unsupervised Contextual):** [experiments/exp2_messy_18docs_symbolic/README.md](experiments/exp2_messy_18docs_symbolic/README.md)
* **Suite 3 (Messy 18-Doc Live `gpt-6-luna` Unsupervised Discovery):** [experiments/exp3_messy_18docs_live_llm/README.md](experiments/exp3_messy_18docs_live_llm/README.md)
* **Suite 4 (Generative Data Synthesis with Live `gpt-6-luna`):** [experiments/exp4_generative_data_synthesis_live_llm/README.md](experiments/exp4_generative_data_synthesis_live_llm/README.md)

---

## 1. Quantitative Evaluation Metrics

Following Section 5.2 (Figure 5 & Appendix Table, p. 11 of Mordatch, 2018), we evaluate both **Identification** ($\mathbf{a}^*$) and **Generation** ($\mathbf{x}^{1*}$):

1. **Identification Error — Attention Binary Cross-Entropy (`attn_bce`) ↓:**

$$
\text{BCE}(\mathbf{a}^K, \mathbf{a}^{\text{gold}}) = -\frac{1}{N} \sum_{i=1}^N \Big[ a_i^{\text{gold}} \log \sigma(a_i^K) + (1 - a_i^{\text{gold}}) \log\big(1 - \sigma(a_i^K)\big) \Big]
$$

2. **False-Positive Distractor Trap Rate (`false_positive_traps`) ↓:**
   Count and mean attention gate $\sigma(a_{\text{trap}}^K)$ of polysemous trap documents and operational clutter falsely attended ($\sigma(a_i^K) \ge 0.5$).
3. **Generation Error — Wiki Topology & Synthesis Error (`gen_error`) ↓:**

$$
\text{GenErr}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}}) = 0.65 \cdot \mathcal{E}_{\text{links}}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}}) + 0.35 \cdot \mathcal{E}_{\text{synthesis}}(\mathbf{x}^{1,K}, \mathbf{a}^{\text{gold}})
$$

4. **Residual Relational Energy (`final_energy`) ↓:**
   $E_\theta(\mathbf{x}^{1,K}, \mathbf{a}^K, \mathbf{w}^*) \in \mathbb{R}^+$.

---

## 2. Benchmark Experiments & Key Results

### Experiment 1: Multi-Step Textual SGLD ($K \in \{1, 2, 5, 10, 20\}$) vs. One-Shot ($K = 1$)
* **Paper Reference:** Section 5.3 & Figure 7 (Left & Top-Right).
* **Hypothesis:** One-shot RAG/synthesis ($K=1$) cannot wire all pairwise links or recover from initial surface-keyword noise; multi-step Textual SGLD ($K=10, 20$) drives generation error to zero via iterative feedback edits.

| SGLD Steps (`K`) | Suite 1 (Clean 10-Doc) `attn_bce` / `gen_error` ↓ | Suite 2 (Messy 18-Doc) `attn_bce` / `gen_error` ↓ | Suite 2 Avg Trap Gate `σ(a_trap^K)` ↓ | Suite 2 Energy `E(x¹)` ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **`K = 1` (One-Shot)** | `0.5017` / `0.8700` | `0.4220` / `0.8700` | `0.3293` | `6.8184` |
| **`K = 2`** | `0.3956` / `0.7400` | `0.3044` / `0.7400` | `0.2447` | `6.0493` |
| **`K = 5`** | `0.2189` / `0.3500` | `0.1529` / `0.3500` | `0.1338` | `3.5618` |
| **`K = 10` (Default)** | **`0.1199` / `0.0000`** | **`0.0785` / `0.0000`** | **`0.0749`** | **`0.1500`** |
| **`K = 20`** | **`0.0633` / `0.0000`** | **`0.0390` / `0.0000`** | **`0.0394`** | **`0.1500`** |

---

### Experiment 2: Pairwise Relation Network (`Relation_Network`) vs. Unary Pointwise RAG (`No_RN_Unary`)
* **Paper Reference:** Section 5.3 ("Effect of a Relational Architecture", `"no RN"` in Figure 7).
* **Hypothesis:** Standard Unary RAG scores documents independently via $\sum_i \sigma(a_i)g_\theta(x_i, w)$ and gets fooled by polysemous keyword traps while failing to wire pairwise `[[wiki-links]]`.

| Architecture (Messy 18-Doc KB) | Attention BCE (`attn_bce`) ↓ | False-Positive Polysemous Traps Attended ↓ | Generation Error (`gen_error`) ↓ | Bidirectional `[[wiki-links]]` Wired ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **`Relation_Network` (Pairwise `g_θ(xᵢ, xⱼ, w)`)** | **`0.0539`** | **`0 traps`** (All rejected!) | **`0.0000`** | **30 links** (Full 6 × 5 clique) |
| **`No_RN_Unary` (Pointwise `g_θ(xᵢ, w)`)** | `0.3507` | **`2 false-positive traps!`** (`HVAC` & `Soil`) | `0.6500` | **0 links** (No pairwise bridges) |

---

### Experiment 3: Zero-Shot Context Transfer (Identification ↔ Generation)
* **Paper Reference:** Section 5.2, Figure 5, and Appendix Table (p. 11).

| Condition (Mirroring Paper Table, p. 11) | Suite 1 `gen_error` / `attn_bce` ↓ | Suite 2 `gen_error` / `attn_bce` ↓ | Paper Reference Values (Gen / Attn) |
| :--- | :---: | :---: | :---: |
| **`Untrained_Critic`** | `0.350` / `0.573` | `0.567` / `0.471` | `0.621` / `0.698` |
| **`Trained_on_Generation`** | **`0.000`** / `0.086` | **`0.000`** / `0.051` | `0.006` / `0.061` |
| **`Trained_on_Identification`** | **`0.000`** / `0.086` | **`0.000`** / `0.051` | `0.016` / `0.001` |
| **`Trained_on_Both`** | **`0.000`** / **`0.085`** | **`0.000`** / **`0.052`** | `0.007` / `0.001` |

---

### Experiment 4: Concept Code Interchangeability (`wₐ ↔ wₓ`)
* **Paper Reference:** Section 5.2 & Figure 6.
* **Result:** Swapping identification code $\mathbf{w}_a$ into generation code $\mathbf{w}_x$ preserves **`0.0000`** generation error across all suites.

---

### Experiment 5: Sampler Objective Ablation (With vs. Without $\mathcal{L}_\pi^{\text{KL}}$)
* **Paper Reference:** Section 5.3 & Figure 8 ("With KL Objective" vs. "No KL Objective").

| Training Objective (Messy 18-Doc KB) | Gold Demonstration Energy (Green) ↓ | `K=10` SGLD Sampled Energy (Red) ↓ | Random Initial Energy (Black) |
| :--- | :---: | :---: | :---: |
| **`With_KL_Objective` (`L_p^ML + L_π^KL`)** | `0.0782` | **`0.0796`** (Matches Gold!) | `2.1962` |
| **`No_KL_Objective` (`L_p^ML` Only)** | `0.0782` | `1.3377` (Fails to converge in `K=10`) | `2.1962` |

---

### Experiment 6: Non-Monotonic Polysemous Trap Rejection Trajectory (`k = 0 → 2 → 10`)
* **Paper Reference:** Section 5.3 & Figure 7 (Top-Right: non-trivial multi-step attention feedback corrections).
* See [experiments/exp2_messy_18docs_symbolic/README.md](experiments/exp2_messy_18docs_symbolic/README.md) and [experiments/exp3_messy_18docs_live_llm/README.md](experiments/exp3_messy_18docs_live_llm/README.md) for the full per-document trajectories showing how `trap_hvac_building_energy` and `trap_agronomy_soil_sampling` start at $\sigma(a_i^0) \approx 0.59$ and are suppressed to $\le 0.08$ by pairwise relational feedback while Unary RAG amplifies them to $0.933$.
