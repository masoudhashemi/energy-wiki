# Model-Agnostic Meta-Learning for Fast Adaptation (MAML)

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Pieter Abbeel, Sergey Levine (2017) | **ID:** `1703.03400`

**Formula / Snippet:** `research_note`

**Participating Concepts (Converged Attention **$\sigma(a_i^K)$**):** [[Energy-based concept acquisition through pairwise relational reasoning]] (`0.89`), [[Critic-guided optimization of natural-language artifacts via textual gradients]] (`0.91`)

## Compiled Content & Relational Bridges
Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.

- **LLM Relational Synthesis ([[Energy-based concept acquisition through pairwise relational reasoning]]:)** Its inner-loop adaptation provides the meta-learning context for [[Concept Acquisition via Meta-Learning from Positive Examples]], while its general algorithm remains distinct from the SGLD-based relational energy inference in [[Concept Learning with Energy-Based Models]].
- **LLM Relational Synthesis ([[Critic-guided optimization of natural-language artifacts via textual gradients]]:)** The execution-time adaptation loop in [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] parallels the iterative structure of [[ProTeGi: Prompt Optimization with Textual Gradients]] and [[TextGrad: Automatic Differentiation via Text]], but adapts numerical parameters rather than text and differs from the pairwise wiki edits in [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]].

## Outgoing Wiki Links
- [[Concept Acquisition via Meta-Learning from Positive Examples]]
- [[Concept Learning with Energy-Based Models]]
- [[A Simple Neural Network Module for Relational Reasoning]]
- [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
- [[TextGrad: Automatic Differentiation via Text]]
