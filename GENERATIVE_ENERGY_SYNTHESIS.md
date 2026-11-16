# Generative Data Synthesis & Applications of Learned Energy Functions

In [Concept Learning with Energy-Based Models (Mordatch, 2018)](1811.02486v1.pdf), an Energy-Based Model $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w}) \ge 0$ is **inherently a generative model**. Through the Boltzmann distribution (Equation 1):

$$
p(\mathbf{x} \mid \mathbf{a}, \mathbf{w}) = \frac{\exp\{-E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})\}}{Z(\mathbf{a}, \mathbf{w})}
$$

any variable optimized via Stochastic Gradient Langevin Dynamics (SGLD) can be **generated**. Below are the **5 primary ways** to use the learned relational energy function $E_\theta(\mathbf{x}, \mathbf{a}, \mathbf{w})$ in a Knowledge Base / LLM Wiki system.

---

## 1. Generating "Missing Bridge Documents" / New Hypotheses ($x_{N+1}^*$)

Instead of only wiring links between *existing* documents $\mathbf{x} = [x_1, \dots, x_N]$, $E_\theta$ can **synthesize a brand-new KB document $x_{N+1}$** that fills a structural hole between existing clusters:

1. Hold the existing KB documents $x_1, \dots, x_N$ fixed and initialize a new candidate document node $x_{N+1}^{(0)}$ with attention gate $\sigma(a_{N+1}) = 1.0$.
2. Run $K$ steps of **Textual SGLD over the text of $x_{N+1}$**:

$$
x_{N+1}^{(k)} = x_{N+1}^{(k-1)} - \frac{\alpha}{2} \nabla_{x_{N+1}}^{\text{text}} E_\theta\big([x_1, \dots, x_N, x_{N+1}^{(k-1)}], \, \mathbf{a}, \, \mathbf{w}\big) + \omega^k
$$

3. At each iteration $k$, the pairwise relation operator $\sum_{j=1}^N \sigma(a_j) g_\theta(x_{N+1}^{(k-1)}, x_j, \mathbf{w})$ critiques the draft $x_{N+1}^{(k-1)}$ against every attended neighbor $x_j$—pointing out missing mathematical bridges, unaddressed failure modes, or missing `[[wiki-links]]`—and the SGLD optimizer rewrites $x_{N+1}^{(k)}$ until $E_\theta \to 0$.

---

## 2. Compositional & Counterfactual Data Generation ($\min_{\tilde{\mathbf{x}}} \sum_m E_\theta(\tilde{\mathbf{x}}, \mathbf{a}, \mathbf{w}^{(m)})$)

Because concepts compose additively by **summing their energy functions** (Section 1 & 2 of Mordatch, 2018):

$$
E_{\text{composite}}(\tilde{\mathbf{x}}) = E_\theta(\tilde{\mathbf{x}}, \mathbf{a}_1, \mathbf{w}^{(1)}) + E_\theta(\tilde{\mathbf{x}}, \mathbf{a}_2, \mathbf{w}^{(2)}) - \lambda E_\theta(\tilde{\mathbf{x}}, \mathbf{a}_3, \mathbf{w}^{(3)})
$$

* You can combine two concepts $\mathbf{w}^{(1)}$ and $\mathbf{w}^{(2)}$ that rarely or never co-occur in the raw corpus (e.g., $\mathbf{w}^{(1)} =$ *Boltzmann Partition Estimation & Short-Run Langevin MCMC* + $\mathbf{w}^{(2)} =$ *Discrete Critic-Guided Textual Gradients*), optionally subtract a negative constraint $-\lambda E_\theta(\cdot, \mathbf{w}^{(3)})$, and optimize a new synthetic document $x_{N+1}$ to satisfy both simultaneously.
* Injecting Langevin exploration noise $\omega^k$ samples diverse valid documents from the low-energy intersection manifold $\exp\{-E_{\text{composite}}(\tilde{\mathbf{x}})\}$.

---

## 3. Generating Hard-Negative / Adversarial Benchmark Data ($\tilde{x}_{\text{trap}} \sim \pi_{\text{adv}}$)

In Equation (7) of the paper, contrastive learning requires informative negative samples:

$$
\mathcal{L}_p^{\text{ML}} \approx \big[ E_\theta(\mathbf{x}_{\text{true}}, \mathbf{a}, \mathbf{w}) - E_\theta(\tilde{\mathbf{x}}, \mathbf{a}, \mathbf{w}) \big]_+
$$

* By optimizing a synthetic document $\tilde{x}_{\text{trap}}$ to **minimize unary surface energy** $E_{\text{unary}}(\tilde{x}_{\text{trap}}, \mathbf{w})$ (matching technical vocabulary and surface phrasing) while **maximizing pairwise relational energy** $\sum_j \sigma(a_j) g_\theta(\tilde{x}_{\text{trap}}, x_j, \mathbf{w})$ (incompatible causal/mathematical domain), the model generates high-difficulty **synthetic polysemous distractors and adversarial hard negatives** for stress-testing RAG rerankers and verifiers.

---

## 4. Cross-Domain Concept Re-Enactment / Transfer (Section 5.4, Figure 9)

In Section 5.4 (Figure 9) of the paper, Mordatch reuses an energy function $E_\theta(\cdot, \mathbf{a}, \mathbf{w}^*)$ learned in a 2D particle environment to control a **3D robotic arm via Model-Predictive Control (MPC)** without training a new policy.
* In an LLM Wiki: once an abstract structural concept code $\mathbf{w}^*$ is inferred in Domain A (e.g., AI/ML papers), the exact same relational energy function $E_\theta(\tilde{x}_{\text{domain B}}, \mathbf{a}, \mathbf{w}^*)$ can generate a **new document or system design in Domain B** (e.g., Distributed Systems / Database Consensus) that re-enacts the same relational structure under Domain B's actuation primitives.

---

## 5. Zero-Shot Anomaly Detection, Contradiction Flagging & Active KB Expansion

Evaluating $E_\theta(\mathbf{x}_{\text{new}}, \mathbf{a}^*, \mathbf{w}^*)$ on newly ingested documents without updating $\mathbf{x}$ provides three diagnostic signals:
1. **$E_\theta \approx 0$:** The document is consistent with concept $\mathbf{w}^*$; auto-link into the wiki.
2. **Low Unary Deficit + High Pairwise Contradiction Deficit:** The document addresses the same mechanism as an existing cluster but contradicts its assumptions; trigger contradiction reconciliation.
3. **High Energy Across All Known $\{\mathbf{w}^{(1)}, \dots, \mathbf{w}^{(M)}\}$:** Signals an out-of-distribution topic; trigger $\arg\min_{\mathbf{w}^{(M+1)}} E_\theta$ to induce a new wiki concept.

---

## Experimental Implementation & Empirical Evaluation

For the live `gpt-6-luna` implementation, generated synthetic documents, and independent Critic evaluations across Tasks 1–3, see **[experiments/exp4_generative_data_synthesis_live_llm/README.md](experiments/exp4_generative_data_synthesis_live_llm/README.md)**.
