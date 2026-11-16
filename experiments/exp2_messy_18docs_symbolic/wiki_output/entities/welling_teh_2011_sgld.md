# Bayesian Learning via Stochastic Gradient Langevin Dynamics

**Source Type:** `research_note` | **Authors:** Max Welling, Yee Whye Teh (2011) | **ID:** `ICML-2011`

**Formula / Snippet:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann Energy, Partition Function & Langevin Sampling]] (`0.96`), [[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]] (`0.65`)

## Compiled Content & Relational Bridges
Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]
