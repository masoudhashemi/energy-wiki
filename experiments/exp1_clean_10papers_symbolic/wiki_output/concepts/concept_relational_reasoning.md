# Pairwise Relational & Compositional Architectures

> **Compiled via **$K$**-Step Textual SGLD over Messy KB** | Concept ID: `concept_relational_reasoning`

- **Argument Selection Code (**$\mathbf{w}_a$**):** Identify papers modeling pairwise entity interactions g_theta(x_i, x_j, w) gated by attention.
- **Relational Synthesis Code (**$\mathbf{w}_x$**):** Synthesize pairwise relation networks and entity-gated energy models via [[wiki-links]].
- **Discovered Mechanisms:** `relational`, `pairwise`, `attention`, `compositional`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 10 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **2 coherent documents** while rejecting polysemous traps ():

### [[Concept Learning with Energy-Based Models]] `[RESEARCH_NOTE]` (Igor Mordatch (OpenAI), 2018)
- **Formula / Mechanism:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.959` — *Promote 'Concept Learning with Energy-B': unary=['attention', 'entity', 'g_theta'], pairwise_synergy=0.41.*
- **Extracted Note:** Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Pairwise Relational & Compositional Architectures]])** $\leftrightarrow$ [[A Simple Neural Network Module for Relational Reasoning]]: Co-instantiates relational, pairwise, attention, compositional.

### [[A Simple Neural Network Module for Relational Reasoning]] `[RESEARCH_NOTE]` (Adam Santoro, Peter Battaglia, Timothy Lillicrap et al., 2017)
- **Formula / Mechanism:** `RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.957` — *Promote 'A Simple Neural Network Module': unary=['attention', 'compositional', 'entity'], pairwise_synergy=0.38.*
- **Extracted Note:** Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, spatial, and graph topology relations where independent unary entity encoders fail.

- **Relational Bridge ([[Pairwise Relational & Compositional Architectures]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates relational, pairwise, attention, compositional.

