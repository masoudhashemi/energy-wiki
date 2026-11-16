# Composite: Energy Sampling + Inner-Loop Meta-Adaptation

> **Compiled via **$K$**-Step Textual SGLD over Messy KB** | Concept ID: `concept_composite_ebm_meta`

- **Argument Selection Code (**$\mathbf{w}_a$**):** Composite Sum E(w_1) + E(w_2): (Boltzmann Energy & Partition Sampling) + (Execution-Time Inner-Loop & Textual Gradients)
- **Relational Synthesis Code (**$\mathbf{w}_x$**):** Bridge Boltzmann partition sampling with inner-loop meta-learning and textual gradient updates.
- **Discovered Mechanisms:** `energy`, `boltzmann`, `trajectory`, `partition_function`, `gradient`, `sampling`, `langevin`, `inner_loop`, `adaptation`, `metalearning`, `demonstration`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 10 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **7 coherent documents** while rejecting polysemous traps ():

### [[Concept Learning with Energy-Based Models]] `[RESEARCH_NOTE]` (Igor Mordatch (OpenAI), 2018)
- **Formula / Mechanism:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.958` — *[Boltzmann Energy &] Promote 'Concept Learning with Energy-B': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.47. | [Execution-Time Inn] Promote 'Concept Learning with Energy-B': unary=['energy', 'gradient', 'inner_loop'], pairwise_synergy=0.36.*
- **Extracted Note:** Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] `[RESEARCH_NOTE]` (Max Welling, Yee Whye Teh, 2011)
- **Formula / Mechanism:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.944` — *[Boltzmann Energy &] Promote 'Bayesian Learning via Stochast': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.35. | [Execution-Time Inn] Promote 'Bayesian Learning via Stochast': unary=['energy', 'gradient', 'noise'], pairwise_synergy=0.21.*
- **Extracted Note:** Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Guided Cost Learning: Deep Inverse Optimal Control]] `[RESEARCH_NOTE]` (Chelsea Finn, Sergey Levine, Pieter Abbeel, 2016)
- **Formula / Mechanism:** `L_IOC(theta) = E_{x~demo}[E_theta(x)] + log E_{x~q}[exp(-E_theta(x)) / q(x)]`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.892` — *[Boltzmann Energy &] Promote 'Guided Cost Learning: Deep Inv': unary=['boltzmann', 'energy', 'partition_function'], pairwise_synergy=0.35. | [Execution-Time Inn] Promote 'Guided Cost Learning: Deep Inv': unary=['demonstration', 'energy'], pairwise_synergy=0.11.*
- **Extracted Note:** Learns neural network cost (energy) functions from expert demonstration trajectories under the Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. Approximates the intractable partition function Z using importance sampling with mixture distribution q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the sampling distribution pi generates low-energy negative samples.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Training Products of Experts by Minimizing Contrastive Divergence]] `[RESEARCH_NOTE]` (Geoffrey E. Hinton, 2002)
- **Formula / Mechanism:** `Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.925` — *[Boltzmann Energy &] Promote 'Training Products of Experts b': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.36. | [Execution-Time Inn] Promote 'Training Products of Experts b': unary=['energy', 'gradient'], pairwise_synergy=0.20.*
- **Extracted Note:** Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Maximum Entropy Inverse Reinforcement Learning]] `[RESEARCH_NOTE]` (Brian D. Ziebart et al., 2008)
- **Formula / Mechanism:** `p(x | w) = (1 / Z(w)) exp(-E(x, w))`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.510` — *[Boltzmann Energy &] Promote 'Maximum Entropy Inverse Reinfo': unary=['boltzmann', 'energy', 'partition_function'], pairwise_synergy=0.35. | [Execution-Time Inn] Suppress 'Maximum Entropy Inverse Rein': off-concept clutter (overlap=['energy']).*
- **Extracted Note:** Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts compose additively by summing their constituent energy functions.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] `[RESEARCH_NOTE]` (Chelsea Finn, Pieter Abbeel, Sergey Levine, 2017)
- **Formula / Mechanism:** `min_theta L(theta - alpha grad_theta L_demo(theta))`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.562` — *[Boltzmann Energy &] Suppress 'Model-Agnostic Meta-Learning': off-concept clutter (overlap=['gradient']). | [Execution-Time Inn] Promote 'Model-Agnostic Meta-Learning f': unary=['adaptation', 'demonstration', 'gradient'], pairwise_synergy=0.24.*
- **Extracted Note:** Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes theta so adapted parameters generalize on X^train.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Acquisition via Meta-Learning from Positive Examples]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Concept Acquisition via Meta-Learning from Positive Examples]] `[RESEARCH_NOTE]` (Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al., 2017)
- **Formula / Mechanism:** `w*_theta(X^demo) = argmin_w E_{x~X^demo}[E_theta(x, w)]`
- **Converged Gate **$\sigma(a_i^K)$**:** `0.643` — *[Boltzmann Energy &] Suppress 'Concept Acquisition via Meta': off-concept clutter (overlap=['energy', 'gradient']). | [Execution-Time Inn] Promote 'Concept Acquisition via Meta-L': unary=['adaptation', 'demonstration', 'energy'], pairwise_synergy=0.32.*
- **Extracted Note:** Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy over positive demonstrations while meta-learned prior structure in theta prevents collapse.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Composite: Energy Sampling + Inner-Loop Meta-Adaptation]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

