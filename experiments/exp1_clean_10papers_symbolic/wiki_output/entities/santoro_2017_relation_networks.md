# A Simple Neural Network Module for Relational Reasoning

**Source Type:** `research_note` | **Authors:** Adam Santoro, Peter Battaglia, Timothy Lillicrap et al. (2017) | **ID:** `1706.01427`

**Formula / Snippet:** `RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Pairwise Relational & Compositional Architectures]] (`0.96`)

## Compiled Content & Relational Bridges
Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, spatial, and graph topology relations where independent unary entity encoders fail.
- **Relational Bridge ([[Pairwise Relational & Compositional Architectures]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates relational, pairwise, attention, compositional.

## Outgoing Wiki Links
- [[Concept Learning with Energy-Based Models]]
