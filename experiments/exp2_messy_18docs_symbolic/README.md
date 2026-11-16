# Experiment Suite 2: Messy 18-Doc Multi-Topic KB (Unsupervised Contextual Critic)

This folder (`experiments/exp2_messy_18docs_symbolic/`) contains **Experiment Suite 2**, which stress-tests the Energy-Based LLM Wiki framework on a **messy, unstructured, multi-topic Knowledge Base of 18 documents** containing contradictory lab notes, entangled engineering dev logs, **polysemous surface-keyword traps**, and operational infrastructure clutter—using an **unsupervised contextual Relation Network Critic that never reads oracle tags**.

---

## 1. Inputs (`raw/*.md` — 18 Messy Multi-Topic Documents)

Located in [raw/](./raw/), the 18 input documents mix 5 distinct source categories:

### A. Genuine AI/ML Research Notes (10 Documents)
* `mordatch_2018_ebm.md`, `welling_teh_2011_sgld.md`, `finn_2016_gcl.md`, `hinton_2002_cd.md`, `ziebart_2008_maxent_irl.md`, `finn_2017_maml.md`, `grant_2017_meta_concepts.md`, `yuksekgonul_2024_textgrad.md`, `pryzant_2023_protegi.md`, `santoro_2017_relation_networks.md`.

### B. Contradictory Lab Note (1 Document)
* **[contradiction_note_mcmc_mixing.md](./raw/contradiction_note_mcmc_mixing.md)** — *Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping*. Explicitly challenges the assumption in classical contrastive divergence (`hinton_2002_cd`) that short-run $K=10$ chains mix adequately without differentiating through the sampler via $\mathcal{L}^{\text{KL}}$.

### C. Entangled Engineering Dev Log (1 Document)
* **[entangled_devlog_wiki_compiler.md](./raw/entangled_devlog_wiki_compiler.md)** — *DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients*. Mixes irrelevant staging Redis connection-pool debugging notes with a genuine insight about gating pairwise `LINK_DEFICIT` textual gradients by $\sigma(a_i)\sigma(a_j)$.

### D. Polysemous Surface-Keyword Traps (4 Documents — Designed to Fool Unary Keyword RAG!)
* **[trap_hvac_building_energy.md](./raw/trap_hvac_building_energy.md)** — *Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling* (uses the exact words `"energy"`, `"partition"`, `"gradient"`, `"sampling"` in a building HVAC context).
* **[trap_agronomy_soil_sampling.md](./raw/trap_agronomy_soil_sampling.md)** — *Agricultural Soil Core Sampling & Moisture Gradient Partitioning* (uses `"sampling"`, `"partition"`, `"gradient"`, `"energy"`, `"adaptation"` in an agronomy context).
* **[trap_css_gradient_ui_bug.md](./raw/trap_css_gradient_ui_bug.md)** — *Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus* (uses `"gradient"`, `"attention"`, `"pairwise"` in a frontend CSS context).
* **[trap_hr_org_adaptation.md](./raw/trap_hr_org_adaptation.md)** — *Q4 Organizational Adaptation & Peer Feedback Critique Guidelines* (uses `"adaptation"`, `"demonstration"`, `"critiques"` in an HR review context).

### E. Out-of-Domain Operational Clutter (2 Documents)
* **[clutter_postgres_autovacuum.md](./raw/clutter_postgres_autovacuum.md)** — *Postmortem: PostgreSQL Transaction ID Wraparound & Autovacuum Tuning*.
* **[clutter_k8s_ingress_tls.md](./raw/clutter_k8s_ingress_tls.md)** — *Runbook: Kubernetes NGINX Ingress TLS Certificate Rotation*.

---

## 2. Scripts Used & How to Run

### Scripts Involved
* **[run_exp2.py](./run_exp2.py)** — Self-contained runner script in this folder that executes Suite 2 end-to-end on the 18 messy documents.
* **[energy_wiki.py](../../energy_wiki.py)** — Core `RelationalWikiEnergyModel` (unsupervised contextual BOW + pairwise frustration critic), `TextualSGLDOptimizer`, `compute_meta_losses`, and `export_markdown_wiki`.
* **[run_experiments.py](../../run_experiments.py)** — Quantitative benchmark functions (`run_exp1_iterative_sgld` through `run_exp6_messy_trap_trajectories`).
* **[build_viewer.py](../../build_viewer.py)** — Compiles `wiki_viewer.html` from `wiki_output/wiki_bundle.json`.

### Execution Command
```powershell
uv run --no-sync python experiments/exp2_messy_18docs_symbolic/run_exp2.py
```

---

## 3. Method Used (Unsupervised Contextual Relation Network + Pairwise Frustration)

Implemented in [energy_wiki.py](../../energy_wiki.py) and benchmarked in [run_exp2.py](./run_exp2.py):

1. **Zero Oracle Tags Used:** `RelationalWikiEnergyModel` ignores `entity.tags` completely and operates strictly on raw text (`title`, `content`, `key_equation`).
2. **Surface-Keyword Initialization at **$k=0$**:**
   At $k=0$, initial attention logits $\mathbf{a}^0$ are seeded by surface-keyword overlap with $\mathbf{w}$. Consequently, polysemous trap documents (`trap_hvac_building_energy` and `trap_agronomy_soil_sampling`) start with **positive false-positive attention (**$\sigma(a_i^0) \approx 0.59$**)**!
3. **Pairwise Contextual Frustration in **$g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$**:**
   For every co-attended pair $(i, j)$, $g_\theta$ computes both shared mechanism stems and full contextual bag-of-words cosine similarity $\cos(\text{BOW}_i, \text{BOW}_j)$.
   * When a polysemous trap (e.g., HVAC ductwork) is paired with genuine mathematical papers (e.g., Welling & Teh SGLD or Mordatch EBM), $\cos(\text{BOW}_i, \text{BOW}_j) < 0.11$, triggering high **pairwise frustration** (`coherence_deficit > 0.85`).
   * In the SGLD attention gradient $\nabla_{a_i} E_\theta$, accumulated pairwise frustration $\sum_{j \ne i} \sigma(a_j)\text{frust}(i, j)$ overrules the surface keyword match and emits `REJECT_POLYSEMOUS_TRAP`, driving $\sigma(a_{\text{trap}}^k)$ down from `0.59` ($k=0$) to `< 0.08` ($k=10$).
4. **Contradiction & Entangled Note Handling:**
   * When $g_\theta$ detects conflicting claims between two contextually coherent documents (`contradiction_note_mcmc_mixing` vs. `hinton_2002_cd`), it emits `RESOLVE_CONTRADICTION` gradients in $\nabla_{\mathbf{x}^1} E_\theta$.

---

## 3. Outputs (`wiki_output/` and Viewer)

* **Standalone Interactive Viewer:** [wiki_viewer.html](./wiki_viewer.html)
* **Master Wiki Catalog:** [wiki_output/index.md](./wiki_output/index.md)
* **Full Telemetry & Experiment Bundle:** [wiki_output/wiki_bundle.json](./wiki_output/wiki_bundle.json)
* **Compiled Concept Pages (`wiki_output/concepts/`):**
  * [concept_ebm_sampling.md](./wiki_output/concepts/concept_ebm_sampling.md)
  * [concept_inner_loop_textgrad.md](./wiki_output/concepts/concept_inner_loop_textgrad.md)
  * [concept_relational_reasoning.md](./wiki_output/concepts/concept_relational_reasoning.md)
  * [concept_composite_ebm_meta.md](./wiki_output/concepts/concept_composite_ebm_meta.md)
* **Compiled Entity Pages (`wiki_output/entities/`):** 18 Markdown files (12 connected research/lab/devlog pages with bidirectional `[[wiki-links]]`, and 6 distractor/clutter pages explicitly marked as filtered out with `0` links).

---

## 4. Empirical Outcomes (Highlighting Exp 2 & Exp 6 on Messy KB)

### A. Relation Network vs. Unary RAG (`No_RN_Unary`) on Polysemous Traps
| Architecture | Attention BCE $\downarrow$ | False-Positive Traps Attended $\downarrow$ | Generation Error $\downarrow$ | Bidirectional `[[wiki-links]]` Wired $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **`Relation_Network` (Pairwise **$g_\theta$**)** | **`0.0539`** | **`0 traps`** | **`0.0000`** | **30 links** |
| **`No_RN_Unary` (Pointwise RAG)** | `0.3507` | **`2 traps` (`HVAC` & `Soil`)** | `0.6500` | **0 links** |

### B. Non-Monotonic Polysemous Trap Rejection Trajectory (Exp 6, Mirroring Fig. 7 Top-Right)
| Document & Model Trajectory | $k=0$ (Surface Keyword Prior) | $k=2$ | $k=5$ | $k=10$ (Converged Gate $\sigma(a_i^{10})$) | Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`RN_true_sgld_paper` (`welling_teh_2011_sgld`)** | `0.595` | `0.610` | `0.817` | **`0.938`** | Promoted (True Member) |
| **`RN_contradiction_lab_note` (`lab-note-42`)** | `0.599` | `0.771` | `0.892` | **`0.949`** | Promoted & Reconciled |
| **`RN_trap_hvac_energy` (Relation Network)** | `0.583` | `0.273` | `0.130` | **`0.079`** | **Trap Rejected by Pairwise **$g_\theta$**!** |
| **`RN_trap_soil_sampling` (Relation Network)** | `0.592` | `0.245` | `0.105` | **`0.060`** | **Trap Rejected by Pairwise **$g_\theta$**!** |
| **`Unary_NoRN_trap_hvac_energy` (Unary RAG)** | `0.596` | `0.787` | `0.888` | **`0.933`** | **False Positive (Fooled by Keywords!)** |
| **`Unary_NoRN_trap_soil_sampling` (Unary RAG)** | `0.601` | `0.789` | `0.889` | **`0.933`** | **False Positive (Fooled by Keywords!)** |
