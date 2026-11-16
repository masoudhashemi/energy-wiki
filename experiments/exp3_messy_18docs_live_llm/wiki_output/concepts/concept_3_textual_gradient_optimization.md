# Critic-guided optimization of natural-language artifacts via textual gradients

## Concept

Critic-guided textual optimization treats a natural-language artifact as the object of an iterative improvement loop. The artifact may be a prompt, code, or a wiki page. A critic evaluates the current version against examples and a loss or energy rubric, expresses its diagnosis in natural language, and that diagnosis is applied as an edit to produce a subsequent version. The feedback functions as a **gradient-like direction**, not as a numerical derivative over token values or model parameters.

The central contribution is to make the optimization signal usable for discrete text: instead of requiring a differentiable representation of the artifact, the procedure translates evaluation into an explanation of what should change. The resulting update is semantic and editorial—such as clarifying a constraint, repairing an omission, or adding a missing cross-page link—and is repeated at execution time.

## Unified Formulation

Let $x^k$ be the artifact at iteration $k$, and let $E_\theta(x^k)$ denote its critic-assessed loss or energy under rubric or critic context $\theta$. A critic produces a textual gradient, $g^k = \operatorname{Critic}_\theta(x^k)$, that explains a direction of improvement. An editor or update procedure applies that direction:





$$
x^{k+1} = \operatorname{Edit}(x^k, g^k), \qquad k=1,\ldots,K.
$$





This is a functional analogy to gradient descent: the feedback is intended to reduce the assessed energy, but it is expressed as natural-language instructions and applied to a discrete artifact. It should not be confused with computing $\nabla_x E$ numerically. The critic diagnoses; the update procedure interprets and implements the diagnosis; subsequent evaluation checks the revised artifact.

[[ProTeGi: Prompt Optimization with Textual Gradients]] gives a prompt-focused instance. It uses an LLM critic to diagnose failures on demonstration minibatches and converts those diagnoses into semantic edits to natural-language instructions. [[TextGrad: Automatic Differentiation via Text]] frames the idea more generally: textual gradients can be propagated as natural-language explanations for editing discrete variables, including prompts, code, and wiki pages, over iterative refinement steps.

The shared mechanism is therefore **evaluation → textual diagnosis → artifact edit → re-evaluation**. The methods differ in scope and framing, not in the defining role of critic-produced language as an actionable update signal. Neither description makes ordinary numerical parameter gradients the update to the text artifact.

## Pairwise Bridges

### From individual edits to relations between pages

A unary critique evaluates one artifact or entity at a time. That can improve a page’s wording or completeness while leaving a collection-level defect untouched: two relevant pages may not refer to one another, or their relationship may be absent from the compiled account. The wiki-compilation note [[DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients]] extends the textual-update loop by adding pairwise relation critiques. For entities $x_i$ and $x_j$, it describes a relational critic $g_\theta(x_i,x_j,w)$, gated by the attention product $\sigma(a_i)\sigma(a_j)$. When both entities are relevant, the critic can emit a textual instruction such as `LINK_DEFICIT` to add explicit, potentially bidirectional `[[wiki-links]]`.

This supplies a collection-level update direction: the target is not only the quality of each page, but also whether the compiled pages express the important relations between them. The attention-product gate provides a relevance condition for considering the pair; it does not itself generate the prose edit. The critic’s textual diagnosis supplies that edit direction, and the page editor applies it.

### Relation to pairwise reasoning

[[A Simple Neural Network Module for Relational Reasoning]] provides a structural analogy for this extension. It computes interactions across entity pairs and aggregates them, addressing relations that independent unary encoders can miss. That connection helps explain why pairwise evaluation can expose cross-page omissions, but the neural relation module is not itself a textual-gradient method: its pairwise interactions are model computations, whereas the wiki application turns a relational diagnosis into a natural-language instruction to edit pages.

### Relation to fast adaptation

[[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] is related at the level of an inner-loop adaptation pattern. MAML uses task examples to adapt learned parameters through numerical gradient descent and trains meta-parameters so that this adaptation generalizes. Textual-gradient methods likewise have an execution-time inner loop, but the adapted object and update signal differ: they edit a discrete text artifact using critic-produced natural-language feedback. The shared loop structure is useful context, not grounds for treating MAML as a textual-gradient method.

## Contradictions & Entangled Notes Reconciled

The DevLog combines two unrelated subjects. Its morning standup reports a Redis connection-pool timeout on a staging node and a `max_clients` change. That is operational incident detail, not evidence about textual optimization, and it is excluded from the concept’s mechanism. The relevant afternoon research note describes wiki compilation, unary critiques, attention-gated pairwise relation critiques, and link-deficit edits; those details are retained as an application of the broader critic-guided update loop.

The documents also use “gradient” in different senses. In ProTeGi and TextGrad, a textual gradient is a natural-language diagnosis used to guide a semantic edit to a discrete artifact. In MAML, gradient descent updates numerical parameters. In relational reasoning, pairwise interactions are computations over entity pairs. These are related by iterative adaptation or relational structure, but they are not interchangeable mechanisms. The synthesis therefore uses “textual gradient” only for critic feedback that is applied as a text edit, and treats the other mechanisms as contrasts or bridges.

There is no conflict between unary and pairwise critique. Unary feedback remains useful for local artifact defects; the DevLog’s point is that it may be insufficient for collection-level relational defects. Pairwise critique supplements the unary loop by identifying missing relations between relevant documents. Similarly, the pairwise relation-network formalism motivates attention to relations but does not establish that its neural computation is itself an editorial textual gradient.

## Distractor Rejection Analysis

Surface vocabulary is not enough to identify a document as part of this concept. `Gradient`, `attention`, `adaptation`, `demonstration`, `critique`, and `relation` occur in documents with materially different mechanisms. Pairwise relational energy favors documents that connect the relevant roles—critic or rubric, natural-language feedback, iterative artifact editing, and, for the wiki extension, a relation-aware edit—rather than documents that merely share one token.

- **CSS gradient and DOM focus:** [[Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus]] concerns rendering CSS backgrounds and keyboard focus. Its “gradient” is a visual styling construct, and its “attention” is interface focus; it has no critic-generated textual update loop.
- **Organizational peer feedback:** [[Q4 Organizational Adaptation & Peer Feedback Critique Guidelines]] concerns human performance-review and onboarding practices. Written critique is present, but the document does not describe a loss-rubric-driven optimization loop over prompts, code, or wiki artifacts.
- **Pairwise relation networks:** [[A Simple Neural Network Module for Relational Reasoning]] is relevant as a bridge for pairwise structure, but pairwise neural computation alone does not meet the defining criterion of natural-language feedback applied as an artifact edit. It is not used as direct evidence for textual-gradient optimization.
- **MAML:** [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]] shares the inner-loop and adaptation vocabulary, but its updates are numerical gradients over model parameters, not textual edits to a natural-language artifact. It is retained only to clarify the analogy and boundary.

Thus a document can be useful as a bridge or contrast without being a defining instance of the concept. The inclusion criterion is the connected mechanism, not lexical overlap: critic- or rubric-derived natural-language feedback must guide iterative edits to a text artifact.

