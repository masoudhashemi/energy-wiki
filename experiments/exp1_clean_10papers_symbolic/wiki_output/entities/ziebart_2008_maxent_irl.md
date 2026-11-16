# Maximum Entropy Inverse Reinforcement Learning

**Source Type:** `research_note` | **Authors:** Brian D. Ziebart et al. (2008) | **ID:** `AAAI-2008`

**Formula / Snippet:** `p(x | w) = (1 / Z(w)) exp(-E(x, w))`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Boltzmann Energy & Partition Sampling]] (`0.95`), [[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]] (`0.51`)

## Compiled Content & Relational Bridges
Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts compose additively by summing their constituent energy functions.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
