# A Simple Neural Network Module for Relational Reasoning

**Source Type:** `research_note` | **Authors:** Adam Santoro, Peter Battaglia, Timothy Lillicrap et al. (2017) | **ID:** `1706.01427`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Energy-based concept acquisition through pairwise relational reasoning]] (`0.92`), [[Critic-guided optimization of natural-language artifacts via textual gradients]] (`0.90`)

## Compiled Content & Relational Bridges
Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, spatial, and graph topology relations where independent unary entity encoders fail.

- **LLM Relational Synthesis ([[Energy-based concept acquisition through pairwise relational reasoning]]:)** Its attention-gated pairwise interactions provide relational evidence for the compatibility objective in [[Concept Learning with Energy-Based Models]] and can be understood alongside the latent representation inferred in [[Concept Acquisition via Meta-Learning from Positive Examples]].
- **LLM Relational Synthesis ([[Critic-guided optimization of natural-language artifacts via textual gradients]]:)** Pairwise interactions in [[A Simple Neural Network Module for Relational Reasoning]] motivate checking relations that unary processing can miss, a structural idea used for link-deficit feedback in [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]], but unlike [[TextGrad: Automatic Differentiation via Text]] they are not textual edits.

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[TextGrad: Automatic Differentiation via Text]]
