---
entity_id: contradiction_note_mcmc_mixing
title: "Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping"
source_type: "contradiction_note"
authors: "Research Lab Internal Note"
year: 2025
arxiv_id: "lab-note-42"
---

# Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping

**Source Type:** `contradiction_note` | **Authors:** Research Lab Internal Note (2025) — `lab-note-42`

**Formula / Snippet:** `KL(pi_K || p_Boltzmann) >> 0 when trained with L^ML only`

## Raw Unstructured Note
CONTROVERSY / CONTRADICTION NOTE: Classical contrastive divergence (Hinton 2002) assumes K-step truncated Langevin MCMC chains mix sufficiently without differentiating through the sampler. However, our experiments contradict uncalibrated short-run MCMC: when training the energy function E_theta with L^ML alone, K=10 Langevin steps fail to reach low-energy modes of the Boltzmann distribution exp(-E_theta)/Z (sampled energy stays > 0.74). Differentiating through the K-step SGLD rollout via the sampler KL objective L^KL is mandatory to shape the gradient field grad_x E_theta.
