# Training Products of Experts by Minimizing Contrastive Divergence

**Source Type:** `research_note` | **Authors:** Geoffrey E. Hinton (2002) | **ID:** `Neural-Comp-2002`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann energy learning with partition-function estimation and MCMC sampling]] (`0.96`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.92`)

## Compiled Content & Relational Bridges
Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).
- **LLM Relational Synthesis ([[Boltzmann energy learning with partition-function estimation and MCMC sampling]]:)** This method approximates the partition-function contribution with short-chain negative samples, a practical assumption challenged for uncalibrated K=10 runs by [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] and distinct from the asymptotic target-sampling claim in [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its contrastive positive-versus-negative energy learning approximates partition-function effects and connects to the finite-rollout qualification in [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] and the sampler-KL remedy in [[Concept Learning with Energy-Based Models]].

## Outgoing Wiki Links
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
