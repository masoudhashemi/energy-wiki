# TextGrad: Automatic Differentiation via Text

**Source Type:** `research_note` | **Authors:** Mert Yuksekgonul, Carlos Guestrin, James Zou et al. (2024) | **ID:** `2406.07496`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Critic-guided optimization of natural-language artifacts via textual gradients]] (`0.97`)

## Compiled Content & Relational Bridges
Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.
- **LLM Relational Synthesis ([[Critic-guided optimization of natural-language artifacts via textual gradients]]:)** The general textual-gradient loop in [[TextGrad: Automatic Differentiation via Text]] includes the prompt-specific instance [[ProTeGi: Prompt Optimization with Textual Gradients]] and supports the wiki-page application in [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]], whose pairwise critiques add relation-level guidance.

## Outgoing Wiki Links
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[A Simple Neural Network Module for Relational Reasoning]]
