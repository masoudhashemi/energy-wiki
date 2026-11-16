# Execution-Time Inner-Loop Adaptation & Textual Gradients

> **Compiled via $K$-Step Textual SGLD over Messy KB** | Concept ID: `concept_inner_loop_textgrad`

- **Argument Selection Code ($\mathbf{w}_a$):** Identify KB notes grounded in [gradient, inner_loop, adaptation] whose contextual formulations co-depend on shared probabilistic or optimization mechanisms (rejecting polysemous noise).
- **Relational Synthesis Code ($\mathbf{w}_x$):** Compile a canonical wiki synthesis around [gradient, inner_loop, adaptation], wiring bidirectional [[wiki-links]] and reconciling contradictions across attended notes.
- **Discovered Mechanisms:** `gradient`, `inner_loop`, `adaptation`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 18 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **3 coherent documents** while rejecting polysemous traps (`trap_hvac_building_energy`, `trap_agronomy_soil_sampling`, `trap_css_gradient_ui_bug`):

### [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] `[RESEARCH_NOTE]` (Chelsea Finn, Pieter Abbeel, Sergey Levine, 2017)
- **Formula / Mechanism:** `min_theta L(theta - alpha grad_theta L_demo(theta))`
- **Converged Gate $\sigma(a_i^K)$:** `0.645` — *Promote 'Model-Agnostic Meta-Learning f': unary=['adaptation', 'gradient', 'inner_loop'], pairwise_synergy=0.28.*
- **Extracted Note:** Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

### [[TextGrad: Automatic Differentiation via Text]] `[RESEARCH_NOTE]` (Mert Yuksekgonul, Carlos Guestrin, James Zou et al., 2024)
- **Formula / Mechanism:** `v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))`
- **Converged Gate $\sigma(a_i^K)$:** `0.644` — *Promote 'TextGrad: Automatic Differenti': unary=['adaptation', 'gradient', 'inner_loop'], pairwise_synergy=0.29.*
- **Extracted Note:** Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

### [[ProTeGi: Prompt Optimization with Textual Gradients]] `[RESEARCH_NOTE]` (Reid Pryzant et al., 2023)
- **Formula / Mechanism:** `g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)`
- **Converged Gate $\sigma(a_i^K)$:** `0.659` — *Promote 'ProTeGi: Prompt Optimization w': unary=['adaptation', 'gradient', 'inner_loop'], pairwise_synergy=0.36.*
- **Extracted Note:** Performs execution-time inner-loop adaptation of natural language instructions using nonparametric textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits natural-language edit critiques that guide iterative semantic gradient descent.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation.

- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

