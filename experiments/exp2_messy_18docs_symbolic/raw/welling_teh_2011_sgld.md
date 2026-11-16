---
entity_id: welling_teh_2011_sgld
title: "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
source_type: "research_note"
authors: "Max Welling, Yee Whye Teh"
year: 2011
arxiv_id: "ICML-2011"
---

# Bayesian Learning via Stochastic Gradient Langevin Dynamics

**Source Type:** `research_note` | **Authors:** Max Welling, Yee Whye Teh (2011) — `ICML-2011`

**Formula / Snippet:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`

## Raw Unstructured Note
Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.
