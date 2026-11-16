# Model-Agnostic Meta-Learning for Fast Adaptation (MAML)

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Pieter Abbeel, Sergey Levine (2017) | **ID:** `1703.03400`

**Formula / Snippet:** `min_theta L(theta - alpha grad_theta L_demo(theta))`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Execution-Time Inner-Loop Adaptation & Textual Gradients]] (`0.64`)

## Compiled Content & Relational Bridges
Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

## Outgoing Wiki Links
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
