# Execution-Time Inner-Loop & Textual Gradients

> **Compiled via $K$-Step Textual SGLD over Messy KB** | Concept ID: `concept_inner_loop_textgrad`

- **Argument Selection Code ($\mathbf{w}_a$):** Identify KB notes grounded in [gradient, inner_loop, adaptation, energy, metalearning, demonstration] whose contextual formulations co-depend on shared probabilistic or optimization mechanisms (rejecting polysemous noise).
- **Relational Synthesis Code ($\mathbf{w}_x$):** Compile a canonical wiki synthesis around [gradient, inner_loop, adaptation, energy, metalearning, demonstration], wiring bidirectional [[wiki-links]] and reconciling contradictions across attended notes.
- **Discovered Mechanisms:** `gradient`, `inner_loop`, `adaptation`, `energy`, `metalearning`, `demonstration`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 10 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **6 coherent documents** while rejecting polysemous traps ():

### [[Concept Learning with Energy-Based Models]] `[RESEARCH_NOTE]` (Igor Mordatch (OpenAI), 2018)
- **Formula / Mechanism:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`
- **Converged Gate $\sigma(a_i^K)$:** `0.952` — *Promote 'Concept Learning with Energy-B': unary=['energy', 'gradient', 'inner_loop'], pairwise_synergy=0.37.*
- **Extracted Note:** Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.

### [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] `[RESEARCH_NOTE]` (Max Welling, Yee Whye Teh, 2011)
- **Formula / Mechanism:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`
- **Converged Gate $\sigma(a_i^K)$:** `0.917` — *Promote 'Bayesian Learning via Stochast': unary=['energy', 'gradient', 'noise'], pairwise_synergy=0.16.*
- **Extracted Note:** Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.

### [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] `[RESEARCH_NOTE]` (Chelsea Finn, Pieter Abbeel, Sergey Levine, 2017)
- **Formula / Mechanism:** `min_theta L(theta - alpha grad_theta L_demo(theta))`
- **Converged Gate $\sigma(a_i^K)$:** `0.956` — *Promote 'Model-Agnostic Meta-Learning f': unary=['adaptation', 'demonstration', 'gradient'], pairwise_synergy=0.41.*
- **Extracted Note:** Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.

### [[Concept Acquisition via Meta-Learning from Positive Examples]] `[RESEARCH_NOTE]` (Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al., 2017)
- **Formula / Mechanism:** `w*_theta(X^demo) = argmin_w E_{x~X^demo}[E_theta(x, w)]`
- **Converged Gate $\sigma(a_i^K)$:** `0.957` — *Promote 'Concept Acquisition via Meta-L': unary=['adaptation', 'demonstration', 'energy'], pairwise_synergy=0.39.*
- **Extracted Note:** Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy over positive demonstrations while meta-learned prior structure in theta prevents collapse.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.

### [[TextGrad: Automatic Differentiation via Text]] `[RESEARCH_NOTE]` (Mert Yuksekgonul, Carlos Guestrin, James Zou et al., 2024)
- **Formula / Mechanism:** `v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))`
- **Converged Gate $\sigma(a_i^K)$:** `0.946` — *Promote 'TextGrad: Automatic Differenti': unary=['adaptation', 'energy', 'gradient'], pairwise_synergy=0.37.*
- **Extracted Note:** Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} at execution time across K iterative refinement steps.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation, energy.

### [[ProTeGi: Prompt Optimization with Textual Gradients]] `[RESEARCH_NOTE]` (Reid Pryzant et al., 2023)
- **Formula / Mechanism:** `g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)`
- **Converged Gate $\sigma(a_i^K)$:** `0.940` — *Promote 'ProTeGi: Prompt Optimization w': unary=['adaptation', 'demonstration', 'gradient'], pairwise_synergy=0.31.*
- **Extracted Note:** Performs execution-time inner-loop adaptation of natural language instructions using nonparametric textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits natural-language edit critiques that guide iterative semantic gradient descent.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates gradient, inner_loop, adaptation, energy.

- **Relational Bridge ([[Execution-Time Inner-Loop & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation, energy.

