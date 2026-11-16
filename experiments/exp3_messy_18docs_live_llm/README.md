# Experiment Suite 3: Messy 18-Doc KB with Domain-Agnostic Live LLM (`gpt-6-luna` via LiteLLM)

This folder (`experiments/exp3_messy_18docs_live_llm/`) contains **Experiment Suite 3**, which runs the **domain-agnostic Live LLM Energy-Based Wiki Compiler** ([run_exp3_live_llm.py](./run_exp3_live_llm.py)) using **`gpt-6-luna`** via `litellm` and `.env`.

Crucially, **neither the concepts **$\mathbf{w}$** nor the attention masks **$\mathbf{a}$** are known in advance**, and **all system prompts (**$\theta$**) are strictly domain-agnostic** (containing zero references to AI/ML, document counts, or specific distractors).

---

## 1. Inputs (`raw/*.md` — 18 Untagged Multi-Topic Documents, Unknown Concepts)

Located in [raw/](./raw/) and ingested via `load_arbitrary_raw_folder(--raw-dir)`:
* **18 untagged raw documents** ($\mathbf{x}^0$) mixing foundational papers, a contradictory lab notebook (`contradiction_note_mcmc_mixing.md`), an entangled engineering dev log (`entangled_devlog_wiki_compiler.md`), 4 polysemous surface-keyword traps (`trap_hvac_building_energy.md`, `trap_agronomy_soil_sampling.md`, `trap_css_gradient_ui_bug.md`, `trap_hr_org_adaptation.md`), and 2 infrastructure ops runbooks (`clutter_postgres_autovacuum.md`, `clutter_k8s_ingress_tls.md`).
* **Zero Prior Concept Labels:** No demonstration masks, topic tags, or concept names are passed to the model.

---

## 2. Scripts Used & How to Run

### Scripts Involved
* **[run_exp3_live_llm.py](./run_exp3_live_llm.py)** — Primary domain-agnostic Live LLM runner script (`load_arbitrary_raw_folder`, `discover_unknown_concepts_via_llm`, `run_llm_relational_sgld_for_concept`) using `litellm` and `.env` (`OPENAI_API_KEY`).
* **[energy_wiki.py](../../energy_wiki.py)** — Data structures (`KBEntity`, `EventState`, `AttentionMask`, `ConceptCode`) and Markdown wiki exporter (`export_markdown_wiki`).
* **[run_experiments.py](../../run_experiments.py)** — Benchmark evaluation suite (`exp1` through `exp6`).
* **[build_viewer.py](../../build_viewer.py)** — Compiles this folder's [wiki_viewer.html](./wiki_viewer.html).

### Execution Commands
```powershell
# Run on the default 18-doc messy corpus in this folder
uv run --no-sync python experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py

# Or run on ANY arbitrary folder of .md / .txt documents
uv run --no-sync python experiments/exp3_messy_18docs_live_llm/run_exp3_live_llm.py --raw-dir path/to/any_docs --out-dir path/to/output_wiki --model gpt-6-luna --max-concepts 3
```

---

## 3. Method Used (Domain-Agnostic Unsupervised Alternating Textual SGLD with `gpt-6-luna`)

Implemented in [run_exp3_live_llm.py](./run_exp3_live_llm.py):

1. **Phase 1 — Unsupervised Joint Discovery of Unknown Concepts (`discover_unknown_concepts_via_llm`, Call #1):**
   * Solving $\arg\min_{\{\mathbf{w}^{(m)}, \mathbf{a}^{(m)}\}} \sum_m E_\theta(\mathbf{x}^0, \mathbf{a}^{(m)}, \mathbf{w}^{(m)})$, `gpt-6-luna` scans the raw corpus and proposes initial $k=0$ surface hypotheses $\mathbf{w}^{(m), 0}$, refines them into sharp $k=K$ concept specifications $\mathbf{w}^{(m), *} = [\mathbf{w}_a^*, \mathbf{w}_x^*]$, identifies core member documents (`core_entity_ids`), flags polysemous false-friends (`trap_entity_ids`), and isolates singleton ops clutter (`unassigned_clutter_ids`).
2. **Phase 2 — Live LLM Pairwise Relational Critic **$g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$** & Textual Gradients **$\nabla_{\mathbf{w}} E_\theta, \nabla_{\mathbf{a}} E_\theta$** (Calls #2, #4, #6, #8):**
   * For each discovered concept (plus the additive composite concept $E(\mathbf{w}_1) + E(\mathbf{w}_2)$), `gpt-6-luna` evaluates pairwise interactions $g_\theta(\mathbf{x}_i, \mathbf{x}_j, \mathbf{w})$ across candidate documents:
     * Produces 3 progressive concept-code refinement gradients ($\nabla_{\mathbf{w}}^{\text{text}} E_\theta$).
     * Produces per-document attention critiques ($\nabla_{\mathbf{a}}^{\text{text}} E_\theta$) showing how polysemous traps start with positive surface attraction at $k=0$ ($\sigma(a_i^0) \in [0.55, 0.65]$) and are suppressed by pairwise domain frustration to $\sigma(a_i^{10}) \le 0.08$ at $k=10$.
     * Outputs structured pairwise diagnoses (`coherence_deficit`, `has_contradiction`, `shared_mechanisms`, `critique`).
3. **Phase 3 — Live LLM Abstractive Wiki Page Compilation & Contradiction Reconciliation **$\nabla_{\mathbf{x}^1} E_\theta$** (Calls #3, #5, #7, #9):**
   * Executing $\mathbf{x}^{1*} = \arg\min_{\mathbf{x}^1} E_\theta(\mathbf{x}^1, \mathbf{a}^*, \mathbf{w}_x^*)$, `gpt-6-luna` emits 5 concrete textual gradient edit directives (`grad_x`) and writes comprehensive Markdown concept synthesis articles (`wiki_output/concepts/*.md`) and entity relational bridge notes (`wiki_output/entities/*.md`).

---

## 3. Outputs (`wiki_output/`, Live LLM Call Log & Viewer)

* **Standalone Interactive Viewer:** [wiki_viewer.html](./wiki_viewer.html)
* **Complete 9-Call `gpt-6-luna` JSON Trace Log (39,727 tokens):** [wiki_output/llm_calls_log.json](./wiki_output/llm_calls_log.json)
* **Master Wiki Catalog:** [wiki_output/index.md](./wiki_output/index.md)
* **Full Telemetry & Experiment Bundle:** [wiki_output/wiki_bundle.json](./wiki_output/wiki_bundle.json)
* **Compiled Concept Articles Written by `gpt-6-luna` (`wiki_output/concepts/`):**
  * [concept_1_relational_energy_concept_learning.md](./wiki_output/concepts/concept_1_relational_energy_concept_learning.md) (*Energy-based concept acquisition through pairwise relational interaction and inner-loop adaptation*)
  * [concept_2_energy_based_learning_and_sampling.md](./wiki_output/concepts/concept_2_energy_based_learning_and_sampling.md) (*Boltzmann energy learning with partition-function estimation and Langevin MCMC sampling*)
  * [concept_3_textual_gradient_optimization.md](./wiki_output/concepts/concept_3_textual_gradient_optimization.md) (*Critic-guided optimization of natural-language artifacts via textual gradients*)
  * [concept_composite_sum.md](./wiki_output/concepts/concept_composite_sum.md) (*Composite $E(\mathbf{w}_1)+E(\mathbf{w}_2)$ Bridge Article*)
* **Compiled Entity Pages (`wiki_output/entities/*.md`):** 18 entity pages with LLM-authored relational synthesis bullets and bidirectional `[[wiki-links]]`.

---

## 5. Verification & Quality Audit of Live `gpt-6-luna` Outputs

A detailed audit of [wiki_output/llm_calls_log.json](./wiki_output/llm_calls_log.json), [wiki_output/index.md](./wiki_output/index.md), and `wiki_output/concepts/*.md` confirms the following properties of the live `gpt-6-luna` run:

1. **Zero-Hint Unsupervised Concept Discovery (Call #1):**
   Without any domain hints or document counts in the system prompt, `gpt-6-luna` partitioned the 18 untagged documents into 3 coherent latent concepts and isolated the operational clutter (`clutter_k8s_ingress_tls`, `clutter_postgres_autovacuum`) and the 4 polysemous traps (`trap_hvac_building_energy`, `trap_agronomy_soil_sampling`, `trap_css_gradient_ui_bug`, `trap_hr_org_adaptation`) into `unassigned_clutter_ids` / `trap_entity_ids`:
   * **Concept 1 (`concept_1_relational_energy_concept_learning`):** *Energy-based concept acquisition through pairwise relational reasoning* (`mordatch_2018_ebm`, `santoro_2017_relation_networks`, `grant_2017_meta_concepts`, `finn_2017_maml`).
   * **Concept 2 (`concept_2_energy_based_learning_and_sampling`):** *Boltzmann energy learning with partition-function estimation and MCMC sampling* (`finn_2016_gcl`, `hinton_2002_cd`, `welling_teh_2011_sgld`, `ziebart_2008_maxent_irl`, `contradiction_note_mcmc_mixing`).
   * **Concept 3 (`concept_3_textual_gradient_optimization`):** *Critic-guided optimization of natural-language artifacts via textual gradients* (`pryzant_2023_protegi`, `yuksekgonul_2024_textgrad`, `entangled_devlog_wiki_compiler`, bridged with `finn_2017_maml` and `santoro_2017_relation_networks`).

2. **Pairwise Relational Critique & Non-Monotonic Trap Suppression (Calls #2, #4, #6, #8):**
   * In Call #1, `gpt-6-luna` initially placed `finn_2017_maml` and `santoro_2017_relation_networks` into `trap_entity_ids` of Concept 3 as candidate "false friends" (numerical vs. textual gradients) to test. Then in Call #6, when evaluating pairwise relations (`finn_2017_maml <-> yuksekgonul_2024_textgrad` and `santoro_2017_relation_networks <-> entangled_devlog_wiki_compiler`), `gpt-6-luna` recognized that they form genuine methodological bridges (`coherence_deficit = 0.08` and `0.06`) and promoted them (`k10_gate = 0.91` and `0.90`), while suppressing the actual traps (`trap_css_gradient_ui_bug: 0.64 -> 0.05` and `trap_hr_org_adaptation: 0.61 -> 0.04`).
   * In Call #4 (Concept 2), `gpt-6-luna` promoted `welling_teh_2011_sgld` (`0.67 -> 0.93`) and `contradiction_note_mcmc_mixing` (`0.64 -> 0.91`) while suppressing `trap_hvac_building_energy` (`0.59 -> 0.04`) and `trap_agronomy_soil_sampling` (`0.61 -> 0.05`).
   * Across all 4 concepts, **0 out of 6** distractor/clutter documents received $\sigma(a_i^{10}) \ge 0.5$ ([index.md lines 31–32 & 42–45](./wiki_output/index.md#L31-L45)).

3. **Contradiction & Entangled DevLog Resolution (Calls #3, #5, #7, #9):**
   * In [concept_2_energy_based_learning_and_sampling.md](./wiki_output/concepts/concept_2_energy_based_learning_and_sampling.md), `gpt-6-luna` accurately reconciled `hinton_2002_cd` and `welling_teh_2011_sgld` with `contradiction_note_mcmc_mixing` (distinguishing asymptotic SGLD convergence as $\alpha \to 0, K \to \infty$ from finite $K=10$ uncalibrated chains that require $\mathcal{L}^{\text{KL}}$ gradient shaping).
   * In [concept_3_textual_gradient_optimization.md](./wiki_output/concepts/concept_3_textual_gradient_optimization.md), it extracted the afternoon wiki-compiler insight from `entangled_devlog_wiki_compiler` while explicitly separating the morning Redis timeout incident as operational noise.
