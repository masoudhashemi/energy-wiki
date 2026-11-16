# Concept Acquisition via Meta-Learning from Positive Examples

**Source Type:** `research_note` | **Authors:** Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al. (2017) | **ID:** `NIPS-2017`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention $\sigma(a_i^K)$):** [[Energy-based concept acquisition through pairwise relational reasoning]] (`0.94`), [[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]] (`0.95`)

## Compiled Content & Relational Bridges
Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy over positive demonstrations while meta-learned prior structure in theta prevents collapse.

- **LLM Relational Synthesis ([[Energy-based concept acquisition through pairwise relational reasoning]]:)** Its positive demonstrations infer a latent concept code that can condition energy-based candidate evaluation in [[Concept Learning with Energy-Based Models]], and its adaptation framing is a specific bridge to [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]].
- **LLM Relational Synthesis ([[Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p]]:)** Its positive-example adaptation supplies concept codes that can condition the execution-time state-energy inference in [[Concept Learning with Energy-Based Models]], while additive joint constraints are formalized in [[Maximum Entropy Inverse Reinforcement Learning]].

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[Concept Learning with Energy-Based Models]]
- [[A Simple Neural Network Module for Relational Reasoning]]
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]
- [[Guided Cost Learning: Deep Inverse Optimal Control]]
- [[Training Products of Experts by Minimizing Contrastive Divergence]]
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]
- [[Maximum Entropy Inverse Reinforcement Learning]]
