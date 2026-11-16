# Concept Acquisition via Meta-Learning from Positive Examples

**Source Type:** `research_note` | **Authors:** Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al. (2017) | **ID:** `NIPS-2017`

**Formula / Snippet:** `w*_theta(X^demo) = argmin_w E_{x~X^demo}[E_theta(x, w)]`

**Status:** *Filtered out / Unattended across all concepts ($\sigma(a_i^K) < 0.5$ — Distractor or Out-of-Domain Clutter)*

## Compiled Content & Relational Bridges
Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy over positive demonstrations while meta-learned prior structure in theta prevents collapse.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[TextGrad: Automatic Differentiation via Text]]: Co-instantiates gradient, inner_loop, adaptation.
- **Relational Bridge ([[Execution-Time Inner-Loop Adaptation & Textual Gradients]])** $\leftrightarrow$ [[ProTeGi: Prompt Optimization with Textual Gradients]]: Co-instantiates gradient, inner_loop, adaptation.

## Outgoing Wiki Links
- [[Model-Agnostic Meta-Learning for Fast Adaptation (MAML)]]
- [[TextGrad: Automatic Differentiation via Text]]
- [[ProTeGi: Prompt Optimization with Textual Gradients]]
