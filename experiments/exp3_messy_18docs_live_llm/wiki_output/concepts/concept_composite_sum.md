# Composite E(w₁)+E(w₂): Energy-based concept acquisition × Boltzmann energy learning with p

## Overview

A composite concept can be represented by adding the energies of its component concepts over the same candidate state and attention mask:

\[
E_{\mathrm{sum}}(x,a;w_1,w_2)=E_\theta(x,a,w_1)+E_\theta(x,a,w_2).
\]

The candidate state is favored when it is jointly compatible with both concepts: lowering either term alone is insufficient if the other remains large. At execution time, concept acquisition uses gradients of this summed objective with respect to the state being generated, while Boltzmann learning supplies the probabilistic interpretation, sampling machinery, and training signals for energy functions. The bridge is therefore both algebraic (additive energies) and procedural (gradient-based state generation and distribution learning).

## Unified Formulation

Let \(x\) denote an entity state or trajectory, \(a\) an attention mask selecting the relevant entities or relations, and \(w_i\) the code for concept \(i\). Define

\[
E_{\mathrm{sum}}(x,a;w_1,w_2)=E_\theta(x,a,w_1)+E_\theta(x,a,w_2),
\qquad
p_{\mathrm{sum}}(x\mid a,w_1,w_2)=\frac{\exp[-E_{\mathrm{sum}}(x,a;w_1,w_2)]}{Z_{\mathrm{sum}}(a,w_1,w_2)}.
\]

The corresponding state gradient is

\[
\nabla_x E_{\mathrm{sum}}=\nabla_x E_\theta(x,a,w_1)+\nabla_x E_\theta(x,a,w_2).
\]

Thus, with attention and concept codes held fixed, both concepts contribute to each state update. In the concept-acquisition setting, the entity state is optimized at execution time; in a Boltzmann learning setting, the same energy defines an unnormalized density whose normalization is supplied by the partition function. For \(p\propto\exp(-E)\), lower energy corresponds to greater unnormalized probability.

The partition function for the sum is the normalization of the combined energy. In general, it is not valid to replace it with the product of the two component partition functions: that simplification requires suitable independence or factorization assumptions. Likewise, an additive energy induces a product of unnormalized factors, \(\exp(-E_1)\exp(-E_2)\), but the normalized joint distribution still has its own partition function.

For state generation, a gradient descent step reduces the summed energy. A stochastic-gradient Langevin update can instead take the form

\[
x^{k+1}=x^k-\frac{\alpha_k}{2}\nabla_{x^k}E_{\mathrm{sum}}+\sqrt{\alpha_k}\,\xi_k,
\]

where \(\xi_k\) is calibrated Gaussian noise; equivalent conventions absorb the factor of \(1/2\) into the step size. Noise supports distributional exploration rather than merely finding a single minimum. The exact sampling interpretation depends on the step-size and long-run conditions, so a short finite rollout should not automatically be treated as an equilibrium sample.

Training and execution have related but distinct objectives. Maximum-likelihood or contrastive training estimates the effect of the partition function through negative samples. Execution-time acquisition optimizes or samples states under the learned energy. In models that explicitly train a finite inner-loop sampler, a rollout-dependent KL objective can additionally shape the energy gradients so that a limited number of sampling steps produce useful states. This auxiliary sampler objective supplements, rather than changes the definition of, the composite energy.

## Pairwise Bridges

- [[Concept Learning with Energy-Based Models]] supplies the execution-time representation: one relation-network energy conditions on state, attention, and concept code, and both state generation and entity identification can be performed through SGLD. The composite extension applies the same state gradient to the sum of two concept-conditioned energies.
- [[Maximum Entropy Inverse Reinforcement Learning]] makes the additive composition principle explicit: multiple constraints or concepts can contribute constituent path energies whose sum defines the joint preference over trajectories. This is the probabilistic counterpart to optimizing a state against both concept energies.
- [[Training Products of Experts by Minimizing Contrastive Divergence]] connects the sum to product-of-experts learning. Since exponentiating a negative sum multiplies the unnormalized factors, CD compares data energy with energy at MCMC-generated negative samples to approximate the otherwise intractable likelihood gradient.
- [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] provides the stochastic update mechanism for exploring the Boltzmann distribution. Its asymptotic sampling claim applies under calibrated noise and limiting conditions, not automatically to every short, finite execution-time chain.
- [[Guided Cost Learning: Deep Inverse Optimal Control]] shows how an energy can be learned from expert trajectories while approximating its partition function with importance sampling. Its adaptive sampling distribution seeks informative low-energy negatives, complementing the contrastive-sampling view.
- [[Concept Acquisition via Meta-Learning from Positive Examples]] connects concept codes to few-shot adaptation from positive demonstrations. Its meta-learned prior helps structure concept inference; the composite objective here concerns the additive energies used to evaluate or generate a state under multiple acquired concepts.
- [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] qualifies the use of SGLD in finite execution-time inference: the energy-learning signal alone did not make the notebook's K=10 chains reach low-energy modes, motivating a differentiable sampler-KL signal for that setting.

Taken together, these documents distinguish three operations that are easy to conflate: composing concepts by summing energies; learning an energy distribution while accounting for its partition function; and running a finite sampler whose behavior may need its own training signal. The same energy function can participate in all three, but each operation has a different objective and approximation burden.

## Contradictions & Entangled Notes Reconciled

The apparent conflict is between classical contrastive-divergence reasoning and the notebook's report that K=10 Langevin steps fail without sampler-KL shaping. [[Training Products of Experts by Minimizing Contrastive Divergence]] describes a short-chain approximation to the likelihood gradient; it does not establish that every short chain mixes sufficiently for every energy landscape. The notebook reports a particular failure regime: under its experiments, optimizing the maximum-likelihood-style loss alone left sampled energy above 0.74 and failed to reach low-energy modes. [[Concept Learning with Energy-Based Models]] addresses this concern by jointly training a softplus contrastive maximum-likelihood objective and a KL loss that differentiates through its K-step inner-loop rollout. These claims are compatible once the CD chain is understood as an approximation whose quality is contingent on the model, initialization, and number of steps—not as a guarantee of mixing.

There is a second scope distinction between [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] and finite-run results. The SGLD account concerns calibrated Gaussian noise and limiting behavior as the step size shrinks and the number of iterations grows. It does not imply exact Boltzmann samples after ten steps. Conversely, failure of one short rollout does not invalidate the asymptotic result. The practical lesson is to distinguish equilibrium sampling guarantees from finite-time sampler performance and, when needed, train the finite rollout directly.

The partition-function methods are also complementary rather than contradictory. [[Guided Cost Learning: Deep Inverse Optimal Control]] uses an importance-sampling mixture involving demonstrations and the current policy, whereas [[Training Products of Experts by Minimizing Contrastive Divergence]] uses MCMC-generated negatives to approximate likelihood gradients. Both address the intractability of normalization, but they choose different approximations and sampling distributions; neither makes the partition function disappear from the underlying normalized model.

Finally, positive-example concept acquisition and Boltzmann energy learning operate at related but distinct levels. [[Concept Acquisition via Meta-Learning from Positive Examples]] describes adaptation of a concept code from demonstrations under a learned prior. [[Concept Learning with Energy-Based Models]] describes energy-based state and attention inference at execution time. Adding two concept energies defines a joint state preference; it does not by itself specify how the concept codes were acquired or guarantee that the two concepts are statistically independent.

## Distractor Rejection Analysis

The rejected documents share isolated words such as *gradient*, *energy*, *partition*, *sampling*, or *attention*, but they do not share the relevant relational structure. The accepted subgraph connects concept-conditioned energies to Boltzmann densities, partition-function estimation, negative sampling, SGLD state updates, and additive composition. A candidate that merely mentions a gradient or a partition without relating it to \(\exp(-E)/Z\), energy-based learning, or the execution-time inference mechanism has high lexical overlap but low pairwise relational support.

In particular, agricultural soil partitioning concerns spatial sampling and moisture profiles, not a Boltzmann partition function or concept-energy normalization. HVAC energy consumption concerns physical power use and sensor sampling, not an energy function defining a probability density. CSS linear-gradient and attention-focus behavior are rendering and interface mechanisms, not gradients of an energy with respect to a candidate state or attention mask. The Redis/wiki-compiler devlog contains compilation vocabulary but does not supply a research mechanism that bridges concept acquisition and Boltzmann learning. The generic MAML paper is a closer distractor because it shares meta-learning and inner-loop adaptation vocabulary; it is not the specific positive-example concept-acquisition account or a source for partition-function-based energy composition. Pairwise relational energy therefore rejects these items based on missing mechanistic links, not merely on topic labels.

