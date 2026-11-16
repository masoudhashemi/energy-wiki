---
entity_id: yuksekgonul_2024_textgrad
title: "TextGrad: Automatic Differentiation via Text"
source_type: "research_note"
authors: "Mert Yuksekgonul, Carlos Guestrin, James Zou et al."
year: 2024
arxiv_id: "2406.07496"
---

# TextGrad: Automatic Differentiation via Text

**Source Type:** `research_note` | **Authors:** Mert Yuksekgonul, Carlos Guestrin, James Zou et al. (2024) — `2406.07496`

**Formula / Snippet:** `v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))`

## Raw Unstructured Note
Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.
