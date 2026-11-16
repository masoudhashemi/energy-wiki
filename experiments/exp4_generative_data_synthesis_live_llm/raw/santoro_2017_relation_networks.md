---
entity_id: santoro_2017_relation_networks
title: "A Simple Neural Network Module for Relational Reasoning"
source_type: "research_note"
authors: "Adam Santoro, Peter Battaglia, Timothy Lillicrap et al."
year: 2017
arxiv_id: "1706.01427"
---

# A Simple Neural Network Module for Relational Reasoning

**Source Type:** `research_note` | **Authors:** Adam Santoro, Peter Battaglia, Timothy Lillicrap et al. (2017) — `1706.01427`

**Formula / Snippet:** `RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )`

## Raw Unstructured Note
Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, spatial, and graph topology relations where independent unary entity encoders fail.
