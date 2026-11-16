# Model-Agnostic Meta-Learning for Fast Adaptation (MAML)

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Pieter Abbeel, Sergey Levine (2017) | **ID:** `1703.03400`

**Formula / Snippet:** `min_theta L(theta - alpha grad_theta L_demo(theta))`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Execution-Time Inner-Loop & Textual Gradients]] (`0.96`), [[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]] (`0.56`)

## Compiled Content & Relational Bridges
Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.
- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[TextGrad: Automatic Differentiation via Text]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
