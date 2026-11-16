---
entity_id: ziebart_2008_maxent_irl
title: "Maximum Entropy Inverse Reinforcement Learning"
source_type: "research_note"
authors: "Brian D. Ziebart et al."
year: 2008
arxiv_id: "AAAI-2008"
---

# Maximum Entropy Inverse Reinforcement Learning

**Source Type:** `research_note` | **Authors:** Brian D. Ziebart et al. (2008) — `AAAI-2008`

**Formula / Snippet:** `p(x | w) = (1 / Z(w)) exp(-E(x, w))`

## Raw Unstructured Note
Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts compose additively by summing their constituent energy functions.
