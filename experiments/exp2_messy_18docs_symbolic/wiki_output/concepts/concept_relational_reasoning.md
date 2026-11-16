# Pairwise Relational & Attention-Gated Architectures

> **Compiled via **$K$**-Step Textual SGLD over Messy KB** | Concept ID: `concept_relational_reasoning`

- **Argument Selection Code (**$\mathbf{w}_a$**):** Identify notes modeling pairwise entity interactions g_theta(x_i, x_j, w) gated by attention sigma(a_i)sigma(a_j).
- **Relational Synthesis Code (**$\mathbf{w}_x$**):** Synthesize pairwise relation networks, entangled dev logs, and entity-gated energy models via [[wiki-links]].
- **Discovered Mechanisms:** `relational`, `pairwise`, `attention`, `compositional`, `gradient`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 18 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **3 coherent documents** while rejecting polysemous traps (`trap_hvac_building_energy`, `trap_agronomy_soil_sampling`, `trap_css_gradient_ui_bug`):

### [[Concept Learning with Energy-Based Models]] `[RESEARCH_NOTE]` (Igor Mordatch (OpenAI), 2018)
- **Formula / Mechanism:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.966` — *Promote 'Concept Learning with Energy-B': unary=['a_i', 'a_j', 'attention'], pairwise_synergy=0.45.*
- **Extracted Note:** Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[A Simple Neural Network Module for Relational Reasoning]]: Co-instantiates relational, pairwise, attention, compositional.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates relational, pairwise, attention, compositional.

### [[A Simple Neural Network Module for Relational Reasoning]] `[RESEARCH_NOTE]` (Adam Santoro, Peter Battaglia, Timothy Lillicrap et al., 2017)
- **Formula / Mechanism:** `RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.954` — *Promote 'A Simple Neural Network Module': unary=['attention', 'compositional', 'entity'], pairwise_synergy=0.33.*
- **Extracted Note:** Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, spatial, and graph topology relations where independent unary entity encoders fail.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates relational, pairwise, attention, compositional.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates relational, pairwise, attention, compositional.

### [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]] `[DEV_LOG]` (Platform ML Eng Team, 2026)
- **Formula / Mechanism:** `grad_{x^1} E_theta = LINK_DEFICIT([[x_i]] <-> [[x_j]]) when sigma(a_i)sigma(a_j) > 0.25`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.959` — *Promote 'DevLog 2026-11-12: Redis Cache': unary=['a_i', 'a_j', 'attention'], pairwise_synergy=0.36.*
- **Extracted Note:** Morning standup: fixed Redis connection pool timeout on staging node-04 by bumping max_clients. Afternoon research note on LLM Wiki compilation: when running inner-loop textual gradient descent (TextGrad / ProTeGi adaptation) to compile wiki pages, unary critiques miss cross-document links. By gating pairwise relation critiques g_theta(x_i, x_j, w) with attention product sigma(a_i)sigma(a_j), the critic emits explicit LINK_DEFICIT textual gradients that wire bidirectional [[wiki-links]] between co-attended entities during K-step execution-time adaptation.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates relational, pairwise, attention, compositional.

- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[A Simple Neural Network Module for Relational Reasoning]]: Co-instantiates relational, pairwise, attention, compositional.

