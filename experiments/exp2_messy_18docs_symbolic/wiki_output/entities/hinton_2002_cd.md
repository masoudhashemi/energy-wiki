# Training Products of Experts by Minimizing Contrastive Divergence

**Source Type:** `research_note` | **Authors:** Geoffrey E. Hinton (2002) | **ID:** `Neural-Comp-2002`

**Formula / Snippet:** `Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Boltzmann Energy, Partition Function & Langevin Sampling]] (`0.96`), [[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]] (`0.58`)

## Compiled Content & Relational Bridges
Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]
