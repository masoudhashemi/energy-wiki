# ProTeGi: Prompt Optimization with Textual Gradients

**Source Type:** `research_note` | **Authors:** Reid Pryzant et al. (2023) | **ID:** `2305.03495`

**Formula / Snippet:** `g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Execution-Time Inner-Loop Adaptation & Textual Gradients]] (`0.66`)

## Compiled Content & Relational Bridges
Performs execution-time inner-loop adaptation of natural language instructions using nonparametric textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits natural-language edit critiques that guide iterative semantic gradient descent.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
