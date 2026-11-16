# Boltzmann energy learning with partition-function estimation and MCMC sampling

## Unified Formulation

These documents share a probabilistic construction: an energy or cost assigns an unnormalized weight to a datum or trajectory, and normalization turns that weight into a Boltzmann distribution. For a state or trajectory $x$,



$$
p_\theta(x)=\frac{\exp[-E_\theta(x)]}{Z_\theta},\qquad
Z_\theta=\int \exp[-E_\theta(x)]\,dx.
$$



In inverse reinforcement learning, $x$ is a demonstrated trajectory and the energy may depend on a cost parameter $w$; in more general energy-based learning, $x$ is a model datum or sampler state. The shared difficulty is that $Z_\theta$ is typically intractable. It enters the normalized likelihood and its gradient, even when the energy itself is straightforward to evaluate.

For a data example, the log-likelihood gradient has the characteristic positive-phase/negative-phase structure:



$$
\nabla_\theta \log p_\theta(x_{\mathrm{data}})
= -\nabla_\theta E_\theta(x_{\mathrm{data}})
+ \mathbb{E}_{x\sim p_\theta}[\nabla_\theta E_\theta(x)].
$$



The first term scores observed data; the second depends on expectation under the normalized model and therefore on $Z_\theta$. Learning methods differ chiefly in how they estimate or avoid this model expectation. Contrastive divergence uses short-chain negative samples, guided cost learning uses importance sampling and a learned sampling distribution, and Langevin methods use stochastic dynamics to draw approximate or asymptotically exact samples. These are related strategies, not interchangeable guarantees.

A second distinction is between gradients with respect to model parameters and gradients with respect to the sampler state. A parameter-learning update uses $\nabla_\theta E_\theta$; a Langevin drift or a differentiable sampler objective uses $\nabla_x E_\theta$. The notebook’s claim concerns shaping this latter gradient field through a sampler-KL objective in its short-run experimental setting, not a general replacement for partition-function estimation.

## Pairwise Bridges using [[Exact Document Title]] wiki-links

- **Trajectory Boltzmann models and importance sampling.** [[Maximum Entropy Inverse Reinforcement Learning]] supplies the trajectory distribution $p(x\mid w)=\exp[-E(x,w)]/Z(w)$, while [[Guided Cost Learning: Deep Inverse Optimal Control]] instantiates that model with learned neural costs. Guided cost learning addresses the intractable normalization with importance sampling from a mixture $q(x)=0.5p_{\mathrm{demo}}(x)+0.5\pi(x)$, and trains $\pi$ toward the energy-defined target. It is thus a sampling-based continuation of the maximum-entropy trajectory formulation, not a different meaning of “energy.”
- **Trajectory partition functions and generic contrastive divergence.** [[Maximum Entropy Inverse Reinforcement Learning]] and [[Training Products of Experts by Minimizing Contrastive Divergence]] both require handling the model expectation induced by a Boltzmann normalizer. The former frames it over trajectories and their path energies; the latter approximates the likelihood-gradient contribution with negative samples from a short MCMC or Langevin chain. The shared obstacle is normalization, while the state space and estimator differ.
- **Trajectory targets and Langevin sampling.** [[Maximum Entropy Inverse Reinforcement Learning]] defines a normalized Boltzmann target over demonstrations, whereas [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] describes stochastic dynamics that can sample an energy-defined posterior. This is a bridge at the level of the target distribution and sampling principle; it does not imply that every trajectory-learning implementation uses SGLD or that a finite chain is exact.
- **Trajectory models and short-run sampler calibration.** [[Maximum Entropy Inverse Reinforcement Learning]] provides the energy-based trajectory target whose normalization motivates sampling; [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] warns that, in its particular short-run experiment, a chain can fail to reach low-energy modes without additional calibration. The notebook is relevant as a practical qualification on sampling such targets, not as a refutation of the maximum-entropy model itself.
- **Importance sampling and contrastive divergence.** [[Guided Cost Learning: Deep Inverse Optimal Control]] and [[Training Products of Experts by Minimizing Contrastive Divergence]] both approximate quantities that depend on an intractable partition function using samples. Guided cost learning uses importance weights under an explicit mixture proposal and learns a sampling policy; contrastive divergence uses a short MCMC/Langevin chain to form negative samples. Their approximations have different mechanisms and error sources.
- **Learned trajectory sampler and SGLD.** [[Guided Cost Learning: Deep Inverse Optimal Control]] makes a sampling distribution $\pi$ generate low-energy negative trajectories, while [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] gives a noise-calibrated Markov-chain mechanism for sampling an energy-defined target. Both connect learning to the quality of samples, but a learned proposal/policy and an SGLD chain are distinct samplers and should not be conflated.
- **Importance-sampling proposals and short-run failure.** [[Guided Cost Learning: Deep Inverse Optimal Control]] explicitly introduces a mixture proposal to estimate the partition function; [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] documents a different concern: a short rollout may not reach the target’s low-energy modes unless its gradient field is shaped. Together they emphasize that having samples is not enough—the proposal or chain must adequately cover relevant target regions.
- **Short-run negative phases and asymptotic Langevin sampling.** [[Training Products of Experts by Minimizing Contrastive Divergence]] treats short K-step chains as a practical approximation to the negative phase. [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] instead states an asymptotic result: with step size tending to zero and run length tending to infinity, calibrated Langevin dynamics samples the target without Metropolis–Hastings accept/reject steps. The finite-run approximation and the limiting sampling claim address different regimes.
- **Contrastive divergence and its short-run critique.** [[Training Products of Experts by Minimizing Contrastive Divergence]] is the practical short-chain method directly challenged by [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]. The notebook reports that its uncalibrated K=10 chains fail to reach low-energy modes and argues for differentiating through the rollout with a sampler-KL objective. This qualifies the adequacy of that short-run approximation under the notebook’s conditions; it does not establish that all contrastive-divergence chains fail.
- **SGLD limits and notebook calibration.** [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] relies on calibrated Gaussian noise and an asymptotic step-size/run-length limit for exact target sampling, whereas [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] concerns a finite, uncalibrated short-run rollout and proposes KL-based gradient shaping. The latter’s criticism is compatible with the former’s limiting guarantee: asymptotic correctness does not promise that an arbitrary K=10 chain is adequate.

## Contradictions & Entangled Notes Reconciled

The central tension is between treating short-run MCMC as a useful approximation and warning that a short chain may be badly inadequate. [[Training Products of Experts by Minimizing Contrastive Divergence]] describes the practical contrastive-divergence strategy: start a short K-step chain and contrast observed-data energy with the endpoint’s negative-sample energy, thereby approximating the contribution associated with the partition function. [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]] reports a contrary empirical outcome for its own uncalibrated K=10 Langevin runs: samples do not reach low-energy modes, with sampled energy remaining above 0.74, and it argues that differentiating through the rollout using $L^{KL}$ is necessary in that setting.

These claims should not be collapsed into a universal verdict. Short-chain contrastive divergence is an approximation whose quality depends on the chain, initialization, energy landscape, and training regime; the notebook supplies evidence that one uncalibrated short-run regime can fail, not proof that every short chain fails. Conversely, the practical use of truncated chains is not evidence that they have mixed. The notebook’s KL-shaping prescription is a proposed remedy for its stated setup, not a condition asserted by the other documents for all energy-based learning.

The SGLD account is not a contradiction to either short-run position. [[Bayesian Learning via Stochastic Gradient Langevin Dynamics]] describes stochastic gradient updates with calibrated Gaussian noise and an exact-target result in the limit of vanishing step size and increasing run length. That asymptotic guarantee is expressly different from assuming adequate mixing after ten uncalibrated steps. Likewise, guided cost learning’s importance-sampling mixture is a distinct estimator of the partition function; it does not inherit the short-chain mixing claim merely because its proposal includes a learned sampler.

A potentially entangled phrase in the notebook is “differentiating through the sampler.” In context, this is not ordinary differentiation of the physical energy, nor simply the standard gradient of the normalized likelihood. It refers to backpropagating a sampler-KL objective through a finite Langevin rollout to shape $\nabla_x E_\theta$. Keep this sampler-calibration claim separate from the usual positive/negative-phase parameter-learning gradient and from SGLD’s noise-calibrated asymptotic sampling statement.

Finally, [[Maximum Entropy Inverse Reinforcement Learning]] notes that multiple constraints or concepts compose additively in the path energy. This is compatible with the shared energy formalism: additive costs combine before exponentiation, and the resulting distribution still requires its own normalization. It does not remove the partition-function problem or prescribe a sampler.

## Distractor Rejection Analysis

The rejected documents share words such as “gradient,” “sampling,” “partition,” or “energy,” but not the required relational mechanism: an energy/cost defining a Boltzmann distribution whose learning or inference depends on its partition function or on sampling that distribution.

- **Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus** uses “gradient” for a CSS visual transition and discusses UI compositing. It has no energy-defined probability distribution, normalizer, or Boltzmann sampler.
- **Agricultural Soil Core Sampling & Moisture Gradient Partitioning** uses sampling for a field-measurement protocol and gradient/partition for spatial soil properties. These are not negative samples from an energy-based model and do not estimate a probabilistic partition function.
- **Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling** concerns physical energy consumption, equipment partition valves, and sensor measurements. Its energy and sampling are operational quantities, not an energy function defining a Boltzmann distribution.

Pairwise relational relevance therefore favors the attended documents: they connect energy to exponentiated probability weights, connect normalization to learning or inference, and specify importance sampling or MCMC/Langevin mechanisms. Shared vocabulary alone does not satisfy that chain of relations.
