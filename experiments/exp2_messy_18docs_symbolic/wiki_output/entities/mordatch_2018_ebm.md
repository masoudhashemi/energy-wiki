# Concept Learning with Energy-Based Models

**Source Type:** `research_note` | **Authors:** Igor Mordatch (OpenAI) (2018) | **ID:** `1811.02486v1`

**Formula / Snippet:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Boltzmann Energy, Partition Function & Langevin Sampling]] (`0.96`), [[Pairwise Relational & Attention-Gated Architectures]] (`0.97`), [[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]] (`0.66`)

## Compiled Content & Relational Bridges
Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Contradiction Reconciled ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

- **Relational Bridge ([[Boltzmann Energy, Partition Function & Langevin Sampling]])** $\leftrightarrow$ [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[A Simple Neural Network Module for Relational Reasoning]]: Co-instantiates relational, pairwise, attention, compositional.
- **Relational Bridge ([[Pairwise Relational & Attention-Gated Architectures]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates relational, pairwise, attention, compositional.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Relational Bridge ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, langevin, partition_function.
- **Contradiction Reconciled ([[Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]: Co-instantiates energy, boltzmann, langevin, partition_function.

## Outgoing Wiki Links
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling]]
- [[A Simple Neural Network Module for Relational Reasoning]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
