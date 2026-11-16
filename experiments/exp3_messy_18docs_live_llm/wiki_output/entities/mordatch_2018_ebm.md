# Concept Learning with Energy-Based Models

**Source Type:** `research_note` | **Authors:** Igor Mordatch (OpenAI) (2018) | **ID:** `1811.02486v1`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Energy-based concept acquisition through pairwise relational reasoning]] (`0.97`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.97`)

## Compiled Content & Relational Bridges
Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **LLM Relational Synthesis ([[Energy-based concept acquisition through pairwise relational reasoning]]:)** Its joint energy over entity trajectories, attention, and concept code connects pairwise interaction aggregation in [[A Simple Neural Network Module for Relational Reasoning]] with demonstration-conditioned code inference in [[Concept Acquisition via Meta-Learning from Positive Examples]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its shared relation-network energy supports execution-time state and attention inference, while [[Maximum Entropy Inverse Reinforcement Learning]] explains additive concept energies and [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] qualifies short-run sampling.

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[A Simple Neural Network Module for Relational Reasoning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
