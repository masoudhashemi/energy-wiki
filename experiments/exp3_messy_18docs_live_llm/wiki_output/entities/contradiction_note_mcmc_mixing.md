# Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping

**Source Type:** `contradiction_note` | **Authors:** Research Lab Internal Note (2025) | **ID:** `lab-note-42`

**Formula / Snippet:** `contradiction_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann energy learning with partition-function estimation and MCMC sampling]] (`0.91`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.89`)

## Compiled Content & Relational Bridges
CONTROVERSY / CONTRADICTION NOTE: Classical contrastive divergence (Hinton 2002) assumes K-step truncated Langevin MCMC chains mix sufficiently without differentiating through the sampler. However, our experiments contradict uncalibrated short-run MCMC: when training the energy function E_theta with L^ML alone, K=10 Langevin steps fail to reach low-energy modes of the Boltzmann distribution exp(-E_theta)/Z (sampled energy stays > 0.74). Differentiating through the K-step SGLD rollout via the sampler KL objective L^KL is mandatory to shape the gradient field grad_x E_theta.
- **LLM Relational Synthesis ([[Boltzmann energy learning with partition-function estimation and MCMC sampling]]:)** This note qualifies the short-chain negative-sampling approximation in [[Training Products of Experts by Minimizing Contrastive Divergence]] and contrasts its finite-run KL-shaping proposal with the asymptotic calibrated-noise guarantee in [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its finite-run warning complements [[Concept Learning with Energy-Based Models]]: the latter's differentiable sampler-KL objective addresses the short-chain failure reported here without contradicting the asymptotic conditions in [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]].

## Outgoing Wiki Links
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
