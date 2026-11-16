# Guided Cost Learning: Deep Inverse Optimal Control

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Sergey Levine, Pieter Abbeel (2016) | **ID:** `1603.00448`

**Formula / Snippet:** `L_IOC(theta) = E_{x~demo}[E_theta(x)] + log E_{x~q}[exp(-E_theta(x)) / q(x)]`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann Energy, Partition Function & Langevin Sampling]] (`0.95`), [[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]] (`0.54`)

## Compiled Content & Relational Bridges
Learns neural network cost (energy) functions from expert demonstration trajectories under the Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. Approximates the intractable partition function Z using importance sampling with mixture distribution q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the sampling distribution pi generates low-energy negative samples.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]
