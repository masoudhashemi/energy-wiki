# ProTeGi: Prompt Optimization with Textual Gradients

**Source Type:** `research_note` | **Authors:** Reid Pryzant et al. (2023) | **ID:** `2305.03495`

**Formula / Snippet:** `g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Execution-Time Inner-Loop & Textual Gradients]] (`0.94`)

## Compiled Content & Relational Bridges
Performs execution-time inner-loop adaptation of natural language instructions using nonparametric textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits natural-language edit critiques that guide iterative semantic gradient descent.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
