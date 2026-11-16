# Training Products of Experts by Minimizing Contrastive Divergence

**Source Type:** `research_note` | **Authors:** Geoffrey E. Hinton (2002) | **ID:** `Neural-Comp-2002`

**Formula / Snippet:** `Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann Energy & Partition Sampling]] (`0.96`), [[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]] (`0.93`)

## Compiled Content & Relational Bridges
Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
