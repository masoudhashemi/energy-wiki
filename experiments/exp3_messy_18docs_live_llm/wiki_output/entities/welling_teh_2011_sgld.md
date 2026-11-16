# Bayesian Learning via Stochastic Gradient Langevin Dynamics

**Source Type:** `research_note` | **Authors:** Max Welling, Yee Whye Teh (2011) | **ID:** `ICML-2011`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann energy learning with partition-function estimation and MCMC sampling]] (`0.93`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.93`)

## Compiled Content & Relational Bridges
Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.
- **LLM Relational Synthesis ([[Boltzmann energy learning with partition-function estimation and MCMC sampling]]:)** Its calibrated-noise, asymptotic Langevin target-sampling result distinguishes the limiting regime from the finite uncalibrated chains criticized in [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] and provides a sampling counterpart to the energy-defined targets in [[Maximum Entropy Inverse Reinforcement Learning]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its calibrated-noise, limiting SGLD account supplies the sampling basis for [[Concept Learning with Energy-Based Models]] while clarifying why the finite K=10 failures in [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] do not refute asymptotic sampling claims.

## Outgoing Wiki Links
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
