## Critique-driven textual editing

Textual-gradient editing is an iterative process: a critic evaluates failures, expresses what should change in natural language, and guides an edit to produce the next version. [[D10]] describes this for instructions; [[D17]] extends the approach to prompts, code, and wiki pages. The critique is guidance, not the revised text itself.

For wiki compilation, [[D04]] reports that unary critiques can miss cross-document links. It proposes attention-gated pairwise relation critiques that identify link deficits and guide adding bidirectional wiki links. [[D11]] provides related rationale: pairwise interactions can capture multi-entity relations that independent unary encoders miss. It concerns relational reasoning, not textual critique directly, but helps explain why considering pairs can reveal what a unary assessment overlooks.

## Energy scores and sampling

An energy scores candidate states; under a Boltzmann model, lower energy corresponds to greater probability. Sampling is a separate process: contrastive divergence uses short MCMC or Langevin chains to draw negative examples [[D08]], while guided cost learning uses importance sampling to approximate the partition function [[D05]].

[[D03]] characterizes classical contrastive divergence as assuming K-step chains mix sufficiently without differentiating through the sampler, then reports that its own K=10 experiments failed to reach low-energy modes when trained with the maximum-likelihood loss alone. It says differentiating through the rollout with a sampler-KL objective was necessary. This is a reported result for its stated regime, not a refutation of every use of short-chain contrastive divergence. In contrast, [[D16]] describes exact Boltzmann sampling in the limit of vanishing step size and infinitely many iterations; that asymptotic claim does not establish that finite, unshaped K=10 runs mix well. [[D09]] combines contrastive learning with a sampler-KL loss.