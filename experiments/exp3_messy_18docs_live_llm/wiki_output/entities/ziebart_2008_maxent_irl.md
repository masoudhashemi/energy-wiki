# Maximum Entropy Inverse Reinforcement Learning

**Source Type:** `research_note` | **Authors:** Brian D. Ziebart et al. (2008) | **ID:** `AAAI-2008`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Boltzmann energy learning with partition-function estimation and MCMC sampling]] (`0.95`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.91`)

## Compiled Content & Relational Bridges
Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts compose additively by summing their constituent energy functions.
- **LLM Relational Synthesis ([[Boltzmann energy learning with partition-function estimation and MCMC sampling]]:)** This trajectory Boltzmann formulation supplies the energy-and-partition-function target extended with importance sampling by [[Guided Cost Learning: Deep Inverse Optimal Control]] and shares the normalized-energy sampling problem addressed more generally by [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its additive path-energy composition directly supports the joint-energy formulation and links Boltzmann trajectory probabilities to the partition-function estimation in [[Guided Cost Learning: Deep Inverse Optimal Control]].

## Outgoing Wiki Links
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
