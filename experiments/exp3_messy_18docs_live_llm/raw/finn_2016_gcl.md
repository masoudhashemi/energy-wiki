---
entity_id: finn_2016_gcl
title: "Guided Cost Learning: Deep Inverse Optimal Control"
source_type: "research_note"
authors: "Chelsea Finn, Sergey Levine, Pieter Abbeel"
year: 2016
arxiv_id: "1603.00448"
---

# Guided Cost Learning: Deep Inverse Optimal Control

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Sergey Levine, Pieter Abbeel (2016) — `1603.00448`

**Formula / Snippet:** `L_IOC(theta) = E_{x~demo}[E_theta(x)] + log E_{x~q}[exp(-E_theta(x)) / q(x)]`

## Raw Unstructured Note
Learns neural network cost (energy) functions from expert demonstration trajectories under the Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. Approximates the intractable partition function Z using importance sampling with mixture distribution q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the sampling distribution pi generates low-energy negative samples.
