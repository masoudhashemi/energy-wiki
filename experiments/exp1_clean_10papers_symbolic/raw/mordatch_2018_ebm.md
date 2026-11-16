---
entity_id: mordatch_2018_ebm
title: "Concept Learning with Energy-Based Models"
source_type: "research_note"
authors: "Igor Mordatch (OpenAI)"
year: 2018
arxiv_id: "1811.02486v1"
---

# Concept Learning with Energy-Based Models

**Source Type:** `research_note` | **Authors:** Igor Mordatch (OpenAI) (2018) — `1811.02486v1`

**Formula / Snippet:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`

## Raw Unstructured Note
Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.
