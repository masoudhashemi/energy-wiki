# TextGrad: Automatic Differentiation via Text

**Source Type:** `research_note` | **Authors:** Mert Yuksekgonul, Carlos Guestrin, James Zou et al. (2024) | **ID:** `2406.07496`

**Formula / Snippet:** `v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Execution-Time Inner-Loop & Textual Gradients]] (`0.95`)

## Compiled Content & Relational Bridges
Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
