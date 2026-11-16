# Boltzmann Energy & Partition Sampling

> **Compiled via $K$-Step Textual SGLD over Messy KB** | Concept ID: `concept_ebm_sampling`

- **Argument Selection Code ($\mathbf{w}_a$):** Identify KB notes grounded in [energy, boltzmann, trajectory, partition_function, gradient, sampling, langevin] whose contextual formulations co-depend on shared probabilistic or optimization mechanisms (rejecting polysemous noise).
- **Relational Synthesis Code ($\mathbf{w}_x$):** Compile a canonical wiki synthesis around [energy, boltzmann, trajectory, partition_function, gradient, sampling, langevin], wiring bidirectional [[wiki-links]] and reconciling contradictions across attended notes.
- **Discovered Mechanisms:** `energy`, `boltzmann`, `trajectory`, `partition_function`, `gradient`, `sampling`, `langevin`

## 1. Filtered & Reconciled Knowledge Base Synthesis

From 10 messy raw documents (including operational logs and polysemous distractors), the Relational Energy Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ isolated **5 coherent documents** while rejecting polysemous traps ():

### [[Concept Learning with Energy-Based Models]] `[RESEARCH_NOTE]` (Igor Mordatch (OpenAI), 2018)
- **Formula / Mechanism:** `E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2`
- **Converged Gate $\sigma(a_i^K)$:** `0.967` — *Promote 'Concept Learning with Energy-B': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.53.*
- **Extracted Note:** Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] `[RESEARCH_NOTE]` (Max Welling, Yee Whye Teh, 2011)
- **Formula / Mechanism:** `x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)`
- **Converged Gate $\sigma(a_i^K)$:** `0.962` — *Promote 'Bayesian Learning via Stochast': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.51.*
- **Extracted Note:** Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without Metropolis-Hastings accept-reject steps.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Guided Cost Learning: Deep Inverse Optimal Control]] `[RESEARCH_NOTE]` (Chelsea Finn, Sergey Levine, Pieter Abbeel, 2016)
- **Formula / Mechanism:** `L_IOC(theta) = E_{x~demo}[E_theta(x)] + log E_{x~q}[exp(-E_theta(x)) / q(x)]`
- **Converged Gate $\sigma(a_i^K)$:** `0.956` — *Promote 'Guided Cost Learning: Deep Inv': unary=['boltzmann', 'energy', 'partition_function'], pairwise_synergy=0.51.*
- **Extracted Note:** Learns neural network cost (energy) functions from expert demonstration trajectories under the Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. Approximates the intractable partition function Z using importance sampling with mixture distribution q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the sampling distribution pi generates low-energy negative samples.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Training Products of Experts by Minimizing Contrastive Divergence]] `[RESEARCH_NOTE]` (Geoffrey E. Hinton, 2002)
- **Formula / Mechanism:** `Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)`
- **Converged Gate $\sigma(a_i^K)$:** `0.960` — *Promote 'Training Products of Experts b': unary=['boltzmann', 'energy', 'gradient'], pairwise_synergy=0.51.*
- **Extracted Note:** Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K).

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Maximum Entropy Inverse Reinforcement Learning]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

### [[Maximum Entropy Inverse Reinforcement Learning]] `[RESEARCH_NOTE]` (Brian D. Ziebart et al., 2008)
- **Formula / Mechanism:** `p(x | w) = (1 / Z(w)) exp(-E(x, w))`
- **Converged Gate $\sigma(a_i^K)$:** `0.953` — *Promote 'Maximum Entropy Inverse Reinfo': unary=['boltzmann', 'energy', 'partition_function'], pairwise_synergy=0.50.*
- **Extracted Note:** Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts compose additively by summing their constituent energy functions.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Concept Learning with Energy-Based Models]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Guided Cost Learning: Deep Inverse Optimal Control]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

- **Relational Bridge ([[Boltzmann Energy & Partition Sampling]])** $\leftrightarrow$ [[Training Products of Experts by Minimizing Contrastive Divergence]]: Co-instantiates energy, boltzmann, trajectory, partition_function.

