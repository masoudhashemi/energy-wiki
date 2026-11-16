---
entity_id: hinton_2002_cd
title: "Training Products of Experts by Minimizing Contrastive Divergence"
source_type: "research_note"
authors: "Geoffrey E. Hinton"
year: 2002
arxiv_id: "Neural-Comp-2002"
---

# Training Products of Experts by Minimizing Contrastive Divergence

**Source Type:** `research_note` | **Authors:** Geoffrey E. Hinton (2002) — `Neural-Comp-2002`

**Formula / Snippet:** `Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)`

## Raw Unstructured Note
Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).
