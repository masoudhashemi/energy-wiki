---
entity_id: pryzant_2023_protegi
title: "ProTeGi: Prompt Optimization with Textual Gradients"
source_type: "research_note"
authors: "Reid Pryzant et al."
year: 2023
arxiv_id: "2305.03495"
---

# ProTeGi: Prompt Optimization with Textual Gradients

**Source Type:** `research_note` | **Authors:** Reid Pryzant et al. (2023) — `2305.03495`

**Formula / Snippet:** `g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)`

## Raw Unstructured Note
Performs execution-time inner-loop adaptation of natural language instructions using nonparametric textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits natural-language edit critiques that guide iterative semantic gradient descent.
