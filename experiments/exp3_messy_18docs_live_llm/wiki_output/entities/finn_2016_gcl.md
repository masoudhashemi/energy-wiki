# Guided Cost Learning: Deep Inverse Optimal Control

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Sergey Levine, Pieter Abbeel (2016) | **ID:** `1603.00448`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann energy learning with partition-function estimation and MCMC sampling]] (`0.96`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.96`)

## Compiled Content & Relational Bridges
Learns neural network cost (energy) functions from expert demonstration trajectories under the Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. Approximates the intractable partition function Z using importance sampling with mixture distribution q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the sampling distribution pi generates low-energy negative samples.
- **LLM Relational Synthesis ([[Boltzmann energy learning with partition-function estimation and MCMC sampling]]:)** This method extends the trajectory Boltzmann model in [[Maximum Entropy Inverse Reinforcement Learning]] by using a mixture-proposal importance-sampling estimate of its intractable partition function, distinct from the short-chain estimator in [[Training Products of Experts by Minimizing Contrastive Divergence]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its importance-sampling estimate of the Boltzmann partition function complements the MCMC negative-sample approximation in [[Training Products of Experts by Minimizing Contrastive Divergence]] and provides a learned-energy setting relevant to [[Maximum Entropy Inverse Reinforcement Learning]].

## Outgoing Wiki Links
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
