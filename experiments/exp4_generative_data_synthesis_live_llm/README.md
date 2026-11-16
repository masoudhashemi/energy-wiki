# Experiment Suite 4: Generative Data Synthesis with Learned Energy Functions (`gpt-6-luna` via LiteLLM)

This folder (`experiments/exp4_generative_data_synthesis_live_llm/`) implements and evaluates **Generative Data Synthesis** using the Relational Energy Function $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ and live **`gpt-6-luna`** calls via `litellm`.

See **[../../GENERATIVE_ENERGY_SYNTHESIS.md](../../GENERATIVE_ENERGY_SYNTHESIS.md)** for the full theoretical overview of the 5 ways to use learned energy functions.

---

## 1. Inputs & Generative Objectives

### 1.1 Input Knowledge Base Subgraphs (`raw/*.md`)
From the 18-document KB in [./raw/](./raw/), we take two distinct conceptual clusters that have not yet been unified into a single mathematical formulation of **discrete Textual SGLD with calibrated partition estimation**:
* **Cluster 1 (**$\mathbf{w}_1$** — Boltzmann Energy, Partition Function **$Z$** & Short-Run Langevin MCMC):**
  `mordatch_2018_ebm`, `finn_2016_gcl`, `hinton_2002_cd`, `welling_teh_2011_sgld`, `ziebart_2008_maxent_irl`, `contradiction_note_mcmc_mixing`.
* **Cluster 2 (**$\mathbf{w}_2$** — Critic-Guided Textual Gradients & Pairwise Wiki Optimization):**
  `yuksekgonul_2024_textgrad`, `pryzant_2023_protegi`, `entangled_devlog_wiki_compiler`, `finn_2017_maml`, `santoro_2017_relation_networks`.

### 1.2 Three Generative Tasks Evaluated
1. **Task 1 — Multi-Step Compositional Bridge Document Generation ($x_{N+1}^*$):**
   Starting from an uncalibrated 2-sentence stub $x_{N+1}^{(0)}$, run iterative Textual SGLD ($k = 0 \to 3 \to 10$) to synthesize a brand-new research document minimizing the additive composite energy:



$$
E_{\text{comp}}(x_{N+1}) = E_\theta([\mathbf{x}, x_{N+1}], \mathbf{a}_1, \mathbf{w}_1) + E_\theta([\mathbf{x}, x_{N+1}], \mathbf{a}_2, \mathbf{w}_2)
$$



2. **Task 2 — Adversarial Hard-Negative / Polysemous Trap Generation (**$\tilde{x}_{\text{trap}}$**) & Discrimination:**
   Synthesize a deceptive Molecular Biology / Biochemistry document (`gen_trap_biochem_allostery.md`) that packs in all the surface vocabulary (`Boltzmann`, `partition function Z`, `Langevin dynamics`, `textual gradient`, `inner-loop`, `attention`, `critique`) and evaluate it with the **independent Relational Energy Critic** vs. **Unary Keyword Energy**.
3. **Task 3 — Cross-Domain Concept Re-Enactment ($x_{\text{transfer}}^*$, Mirroring Section 5.4 / Figure 9):**
   Transfer the abstract relational concept structure $\mathbf{w}^*$ from the AI/ML KB to generate a synthetic **Distributed Systems & Consensus Protocols** architecture document (`gen_transfer_distributed_consensus.md`) and score its structural isomorphism energy with an independent Critic.

---

## 2. Scripts Used & How to Run

* **[./run_exp4_generative_luna.py](./run_exp4_generative_luna.py)** — Self-contained live `gpt-6-luna` runner executing all 3 generative tasks and independent Critic evaluations (8 live LLM calls, 25,911 tokens).
* **[../../GENERATIVE_ENERGY_SYNTHESIS.md](../../GENERATIVE_ENERGY_SYNTHESIS.md)** — Root documentation of the 5 generative and evaluative uses of $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$.

```powershell
uv run --no-sync python experiments/exp4_generative_data_synthesis_live_llm/run_exp4_generative_luna.py
```

---

## 3. Generated Outputs (`wiki_output/`)

* **Complete Telemetry & Evaluations JSON:** [./wiki_output/generative_results.json](./wiki_output/generative_results.json)
* **Full 8-Call `gpt-6-luna` Trace Log (25,911 tokens):** [./wiki_output/llm_calls_log.json](./wiki_output/llm_calls_log.json)
* **Newly Generated Synthetic Documents (`wiki_output/generated_docs/`):**
  1. **[gen_bridge_boltzmann_textual_sgld.md](./wiki_output/generated_docs/gen_bridge_boltzmann_textual_sgld.md)** — *Textual SGLD: Contrastive Energy Estimation and KL-Shaped Rollouts* (Converged Bridge Document $x_{N+1}^{(10)}$).
  2. **[gen_trap_biochem_allostery.md](./wiki_output/generated_docs/gen_trap_biochem_allostery.md)** — *Allosteric Folding, Ribosomal Proofreading, and Sequence-Resolved Conformational Ensembles* (Adversarial Hard-Negative Trap $\tilde{x}_{\text{trap}}$).
  3. **[gen_transfer_distributed_consensus.md](./wiki_output/generated_docs/gen_transfer_distributed_consensus.md)** — *Bounded-Round Gossip with Cluster-State Normalization, Replay-Weighted Proposals, and Pairwise Quorum Verification* (Cross-Domain Re-Enacted Document $x_{\text{transfer}}^*$).

---

## 4. Empirical Evaluation Results (Independent `gpt-6-luna` Critic)

### 4.1 Task 1: Multi-Step Textual SGLD Bridge Document Generation ($k = 0 \to 3 \to 10$)

| SGLD Iteration | Draft Title | Cluster 1 Deficit (`e_w1`) ↓ | Cluster 2 Deficit (`e_w2`) ↓ | Link Topology Deficit ↓ | Short-Chain Calibration Deficit ↓ | Unary Keyword Energy | Pairwise Relational Energy ↓ | **Total Composite Energy `E_comp` ↓** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`k = 0` (Seed Stub)** | *Draft Note: Combining Energy Models and Textual Gradients* | `2.80` | `2.70` | `2.50` (0 links) | `2.00` | `0.40` | `4.70` | **`10.0000`** |
| **`k = 3` (1st SGLD Edit)** | *Draft Note: Boltzmann Energy Models with Pairwise-Gated Textual Gradients* | `0.70` | `1.00` | `1.80` (3 links) | `1.20` | `0.10` | `2.00` | **`4.7000`** |
| **`k = 10` (Converged)** | *Textual SGLD: Contrastive Energy Estimation and KL-Shaped Rollouts* | **`0.25`** | **`0.20`** | **`0.00` (8 links)** | **`0.20`** | **`0.05`** | **`0.85`** | **`0.6500` (-93.5%)** |

* **How Textual SGLD Evolved the Document:**
  * **At `k=0` (`E_comp = 10.00`):** The seed stub had low unary keyword energy (`0.40`) because it mentioned *"energy-based models"* and *"textual gradients"*, but had high pairwise relational energy (`4.70`) because it lacked equations, partition estimators, and `[[wiki-links]]`.
  * **At `k=3` (`E_comp = 4.70`):** Applying Call #1's textual gradients introduced $p_\theta(x \mid w) = \exp[-E_\theta(x,w)]/Z_\theta(w)$, pairwise attention gating $m_{ij} = \sigma(a_i)\sigma(a_j)g_\theta(h_i, h_j)$, and 3 `[[wiki-links]]`, cutting energy by `53%`. However, Call #3's Critic diagnosed that mixture importance sampling $q(x)$ and an operational $\mathcal{L}^{\text{KL}}$ rollout objective for Lab Notebook #42's $K=10$ short-chain failure were still underspecified.
  * **At `k=10` (`E_comp = 0.65`):** Applying Call #3's textual gradients derived the full mixture-proposal softplus contrastive estimator $\mathcal{L}^{\text{ML}}$, formulated the discrete edit-policy forward KL + cluster-coverage objective $\mathcal{L}^{\text{KL}} = D_{\text{KL}}(r_\phi^K \parallel p_\theta) + \beta[\max(0, C_{\min} - C(r_\phi^K))]^2$, and wired **8 exact `[[wiki-links]]`** across both clusters (`e_link_topology_deficit = 0.00`).

---

### 4.2 Task 2: Independent Critic Evaluation on Adversarial Hard-Negative Trap (`gen_trap_biochem_allostery.md`)

In Call #7, the **exact same independent Critic function** evaluated the newly generated adversarial biochemistry document (*Allosteric Folding, Ribosomal Proofreading, and Sequence-Resolved Conformational Ensembles*):

| Candidate Document | Unary Keyword Energy ↓ | Pairwise Relational Energy (`g_θ`) ↓ | Total Composite Energy (`E_comp`) ↓ | Contrastive Softplus Loss `L_p^ML` ↓ | Critic Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Converged Bridge Doc **$x_{N+1}^{(10)}$ | `0.05` | **`0.85`** | **`0.65`** | **`0.00019`** | **Accepted (Coherent Bridge)** |
| **Adversarial Biochem Trap **$\tilde{x}_{\text{trap}}$ | `0.25` *(Fools Unary!)* | **`4.70`** *(Caught by `g_θ`!)* | **`9.20` (14.2× higher)** | — | **Rejected (Polysemous Trap)** |

* **Independent Critic Diagnosis (Call #7):**
  > *"The draft contains Boltzmann and partition-function notation, but applies it mainly to biochemical conformational ensembles and omits sampling-based Z estimation and the short-run MCMC failure. Its textual-gradient and attention terminology is explicitly recast as biology, leaving the pairwise links to both clusters weak and the required wiki-link and contradiction calibration absent."*

---

### 4.3 Task 3: Cross-Domain Concept Re-Enactment (`gen_transfer_distributed_consensus.md`, Mirroring Section 5.4)

In Call #6, `gpt-6-luna` re-enacted the learned concept code $\mathbf{w}^*$ in a **Distributed Systems & Consensus Protocols** domain (*Bounded-Round Gossip with Cluster-State Normalization, Replay-Weighted Proposals, and Pairwise Quorum Verification*). In Call #8, an independent Structural Isomorphism Critic scored the 4 transferred mechanisms:

| Source AI/ML Concept Mechanism ($\mathbf{w}^*$) | Re-Enacted Distributed Systems Mechanism ($x_{\text{transfer}}^*$) | Independent Deficit Score `[0, 1]` ↓ |
| :--- | :--- | :---: |
| **1. Boltzmann Partition Normalizer **$Z_\theta(\mathbf{w})$ | Global cluster-state normalizer $Z_\theta(A, W) = \sum_S \exp(-E_\theta(S, A, W))$ over admissible protocol states | **`0.00`** |
| **2. Mixture Proposal **$q(x)$** & Contrastive Estimation** | Gossip + incident-replay + fault-injection proposal mixture $q(S) = \sum_m \rho_m q_m(S)$ with importance estimator $\hat{Z}$ | `0.55` *(omits explicit softplus loss)* |
| **3. **$K$**-Step KL-Shaped Rollout (**$\mathcal{L}^{\text{KL}}$**)** | Bounded-$K$-round gossip trajectory $\tau = (S^1, \dots, S^K)$ minimizing $\text{KL}(Q_\phi(\tau \mid S^0) \parallel P^*)$ | `0.30` |
| **4. Attention-Gated Pairwise Critic **$\sigma(a_i)\sigma(a_j)g_\theta$ | Pairwise quorum-certificate compatibility check $g_\theta(S_i, S_j, W)$ gated by node trust weights $h_{ij} = s_i s_j$ | **`0.10`** |
| **Total Structural Isomorphism Energy** | — | **`0.95` / `4.00`** |
