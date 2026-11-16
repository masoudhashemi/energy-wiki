# Bayesian Learning via Stochastic Gradient Langevin Dynamics

**Source Type:** `research_note` | **Authors:** Max Welling, Yee Whye Teh (2011) | **ID:** `ICML-2011`

**Formula / Snippet:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann Energy & Partition Sampling]] (`0.96`), [[Execution-Time Inner-Loop & Textual Gradients]] (`0.92`), [[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]] (`0.94`)

## Compiled Content & Relational Bridges
Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
