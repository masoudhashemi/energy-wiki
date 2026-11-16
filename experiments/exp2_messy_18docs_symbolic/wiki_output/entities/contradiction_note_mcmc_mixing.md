# Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping

**Source Type:** `contradiction_note` | **Authors:** Research Lab Internal Note (2025) | **ID:** `lab-note-42`

**Formula / Snippet:** `KL(pi_K || p_Boltzmann) >> 0 when trained with L^ML only`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann Energy, Partition Function & Langevin Sampling]] (`0.95`), [[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]] (`0.55`)

## Compiled Content & Relational Bridges
CONTROVERSY / CONTRADICTION NOTE: Classical contrastive divergence (Hinton 2002) assumes K-step truncated Langevin MCMC chains mix sufficiently without differentiating through the sampler. However, our experiments contradict uncalibrated short-run MCMC: when training the energy function E_theta with L^ML alone, K=10 Langevin steps fail to reach low-energy modes of the Boltzmann distribution exp(-E_theta)/Z (sampled energy stays > 0.74). Differentiating through the K-step SGLD rollout via the sampler KL objective L^KL is mandatory to shape the gradient field grad_x E_theta.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]
