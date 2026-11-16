# TextGrad: Automatic Differentiation via Text

**Source Type:** `research_note` | **Authors:** Mert Yuksekgonul, Carlos Guestrin, James Zou et al. (2024) | **ID:** `2406.07496`

**Formula / Snippet:** `v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Execution-Time Inner-Loop Adaptation & Textual Gradients]] (`0.64`)

## Compiled Content & Relational Bridges
Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
