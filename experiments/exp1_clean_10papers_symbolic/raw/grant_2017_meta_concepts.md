---
entity_id: grant_2017_meta_concepts
title: "Concept Acquisition via Meta-Learning from Positive Examples"
source_type: "research_note"
authors: "Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al."
year: 2017
arxiv_id: "NIPS-2017"
---

# Concept Acquisition via Meta-Learning from Positive Examples

**Source Type:** `research_note` | **Authors:** Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al. (2017) — `NIPS-2017`

**Formula / Snippet:** `w*_theta(X^demo) = argmin_w E_{x~X^demo}[E_theta(x, w)]`

## Raw Unstructured Note
Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy over positive demonstrations while meta-learned prior structure in theta prevents collapse.
