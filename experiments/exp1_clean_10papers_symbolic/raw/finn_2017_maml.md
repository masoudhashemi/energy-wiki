---
entity_id: finn_2017_maml
title: "Model-Agnostic Meta-Learning for Fast Adaptation (MAML)"
source_type: "research_note"
authors: "Chelsea Finn, Pieter Abbeel, Sergey Levine"
year: 2017
arxiv_id: "1703.03400"
---

# Model-Agnostic Meta-Learning for Fast Adaptation (MAML)

**Source Type:** `research_note` | **Authors:** Chelsea Finn, Pieter Abbeel, Sergey Levine (2017) — `1703.03400`

**Formula / Snippet:** `min_theta L(theta - alpha grad_theta L_demo(theta))`

## Raw Unstructured Note
Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.
