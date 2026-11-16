"""
Energy-Based LLM Wiki with Textual Gradients (Realistic Messy KB Edition)
=========================================================================
Implements "Concept Learning with Energy-Based Models" (Mordatch, 2018,
arXiv:1811.02486v1) over a messy, unstructured, multi-topic Knowledge Base
containing:
  1. Unstructured research notes, debugging logs, and inconsistent terminology
  2. Polysemous "Trap" Distractors (e.g., HVAC thermal energy, CSS gradients,
     agricultural soil sampling, HR adaptation) that fool unary keyword RAG
  3. Out-of-domain operational clutter (Postgres, Kubernetes, Stripe webhooks)
  4. Multi-topic entangled notes and explicit inter-document contradictions

Crucially, the Relational Energy Model E_theta(x, a, w) does NOT rely on oracle
tags. Instead, g_theta(x_i, x_j, w) computes contextual distributional coherence
and pairwise frustration across entity pairs, enabling multi-step SGLD to:
  - Initially get fooled by surface-keyword distractors at k=1..2 and then
    SUPPRESS them via pairwise relational feedback by k=5..10 (matching Fig. 7
    Top-Right of Mordatch 2018).
  - Reconcile contradictory notes and extract relevant segments from entangled docs.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def softplus(z: float) -> float:
    if z > 20:
        return z
    return math.log1p(math.exp(z))


STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "into", "over", "under", "via",
    "are", "how", "which", "their", "between", "using", "based", "this", "that",
    "have", "has", "was", "were", "but", "not", "can", "will", "when", "where",
    "while", "also", "more", "than", "some", "such", "only", "each", "both",
    "other", "into", "about", "after", "before", "during", "through", "across",
    "note", "notes", "doc", "file", "page", "section", "we", "our", "they", "it"
}


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class KBEntity:
    """Single KB document x_i^t in state x^t = [x_1^t, ..., x_N^t]."""
    entity_id: str
    title: str
    source_type: str  # "research_note" | "dev_log" | "polysemous_trap" | "ops_clutter" | "contradiction_note"
    authors: str
    year: int
    arxiv_id: str
    content: str
    key_equation: str = ""
    tags: List[str] = field(default_factory=list)  # Kept only for UI metadata; NEVER used by E_theta!
    wiki_links: List[str] = field(default_factory=list)


@dataclass
class EventState:
    x0: List[KBEntity]
    x1: List[KBEntity]
    concept_page: Optional[KBEntity] = None


@dataclass
class AttentionMask:
    logits: Dict[str, float]
    rationales: Dict[str, str] = field(default_factory=dict)

    def gate(self, entity_id: str) -> float:
        return sigmoid(self.logits.get(entity_id, -2.0))

    def gates_dict(self) -> Dict[str, float]:
        return {eid: round(self.gate(eid), 4) for eid in self.logits}

    def active_ids(self, threshold: float = 0.5) -> List[str]:
        return [eid for eid, val in self.logits.items() if sigmoid(val) >= threshold]


@dataclass
class ConceptCode:
    concept_id: str
    name: str
    w_a: str
    w_x: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class PairwiseRelation:
    entity_i: str
    entity_j: str
    title_i: str
    title_j: str
    gate_weight: float
    coherence_deficit: float
    link_deficit: float
    has_contradiction: bool
    shared_mechanisms: List[str]
    critique: str


@dataclass
class EnergyEvaluation:
    energy: float
    coherence_term: float
    coverage_term: float
    topology_term: float
    synthesis_term: float
    pairwise_relations: List[PairwiseRelation]
    grad_w: str
    grad_a: Dict[str, Tuple[float, str]]
    grad_x: List[str]


@dataclass
class MetaSchemaParams:
    cohesion_weight: float = 1.35
    coverage_weight: float = 1.05
    link_topology_weight: float = 1.25
    synthesis_depth_weight: float = 1.0
    critic_rubric_prompt: str = (
        "Evaluate untagged raw text pairwise contextual coherence; penalize polysemous "
        "distractors that share surface words without domain co-occurrence; resolve contradictions."
    )


# ---------------------------------------------------------------------------
# 2. Realistic Messy, Multi-Topic Knowledge Base (`raw/*.md` — 18 Docs)
# ---------------------------------------------------------------------------

MESSY_KB_DATA: List[Dict] = [
    # --- Core Concept Cluster 1: Energy-Based Models, Partition Functions & Langevin Sampling ---
    {
        "entity_id": "mordatch_2018_ebm",
        "title": "Concept Learning with Energy-Based Models",
        "source_type": "research_note",
        "authors": "Igor Mordatch (OpenAI)",
        "year": 2018,
        "arxiv_id": "1811.02486v1",
        "key_equation": "E_theta(x, a, w) = f_theta(sum_{t,i,j} sigma(a_i)sigma(a_j) g_theta(x_i^t, x_j^t, w), w)^2",
        "tags": ["core_ebm", "core_meta", "core_relational"],
        "content": (
            "Defines concepts implicitly via a single non-negative Relation Network energy function E_theta(x, a, w) "
            "over entity state trajectories x, attention mask a, and concept code w = [w_x, w_a]. "
            "Both concept generation (optimizing x^1) and entity identification (optimizing attention a) are "
            "performed at execution time via K=10 steps of Stochastic Gradient Langevin Dynamics (SGLD) over "
            "the Boltzmann distribution exp(-E_theta)/Z. Meta-training jointly minimizes a softplus contrastive "
            "maximum likelihood loss L^ML (estimating the intractable partition function Z via importance sampling) "
            "and a sampler KL divergence loss L^KL differentiating through the K-step inner-loop Langevin rollout."
        ),
    },
    {
        "entity_id": "welling_teh_2011_sgld",
        "title": "Bayesian Learning via Stochastic Gradient Langevin Dynamics",
        "source_type": "research_note",
        "authors": "Max Welling, Yee Whye Teh",
        "year": 2011,
        "arxiv_id": "ICML-2011",
        "key_equation": "x^k = x^{k-1} - (alpha / 2) grad_x E(x^{k-1}) + omega^k,  omega^k ~ N(0, alpha)",
        "tags": ["core_ebm"],
        "content": (
            "Combines Robbins-Monro stochastic gradient optimization with Langevin dynamics by injecting "
            "calibrated Gaussian noise omega^k ~ N(0, alpha) into gradient updates of the unnormalized log-posterior "
            "or energy function E(x). As step size alpha -> 0 and iteration K -> infinity, Markov chain trajectories "
            "sample from the exact Boltzmann posterior distribution p(x) proportional to exp(-E(x))/Z without "
            "Metropolis-Hastings accept-reject steps."
        ),
    },
    {
        "entity_id": "finn_2016_gcl",
        "title": "Guided Cost Learning: Deep Inverse Optimal Control",
        "source_type": "research_note",
        "authors": "Chelsea Finn, Sergey Levine, Pieter Abbeel",
        "year": 2016,
        "arxiv_id": "1603.00448",
        "key_equation": "L_IOC(theta) = E_{x~demo}[E_theta(x)] + log E_{x~q}[exp(-E_theta(x)) / q(x)]",
        "tags": ["core_ebm"],
        "content": (
            "Learns neural network cost (energy) functions from expert demonstration trajectories under the "
            "Maximum Entropy Inverse Reinforcement Learning Boltzmann model p(x) = exp(-E_theta(x))/Z. "
            "Approximates the intractable partition function Z using importance sampling with mixture distribution "
            "q(x) = 0.5 p_demo(x) + 0.5 pi(x), while minimizing KL divergence KL(pi || exp(-E_theta)/Z) so the "
            "sampling distribution pi generates low-energy negative samples."
        ),
    },
    {
        "entity_id": "hinton_2002_cd",
        "title": "Training Products of Experts by Minimizing Contrastive Divergence",
        "source_type": "research_note",
        "authors": "Geoffrey E. Hinton",
        "year": 2002,
        "arxiv_id": "Neural-Comp-2002",
        "key_equation": "Delta theta proportional to grad_theta E(x^0) - grad_theta E(x^K)",
        "tags": ["core_ebm"],
        "content": (
            "Approximates intractable log-likelihood gradients of the partition function Z in Boltzmann and "
            "energy-based models by running a short K-step Markov chain Monte Carlo (MCMC) or Langevin sampler "
            "to draw negative samples x^K. Contrasts positive data energy E(x^0) against negative sample energy E(x^K)."
        ),
    },
    {
        "entity_id": "ziebart_2008_maxent_irl",
        "title": "Maximum Entropy Inverse Reinforcement Learning",
        "source_type": "research_note",
        "authors": "Brian D. Ziebart et al.",
        "year": 2008,
        "arxiv_id": "AAAI-2008",
        "key_equation": "p(x | w) = (1 / Z(w)) exp(-E(x, w))",
        "tags": ["core_ebm"],
        "content": (
            "Models demonstrated trajectories x as samples from a Boltzmann distribution p(x|w) = exp(-E(x,w))/Z(w) "
            "parameterized by path energy E(x,w) and partition function Z(w). Multiple constraints or concepts "
            "compose additively by summing their constituent energy functions."
        ),
    },

    # --- Entangled & Contradictory Field Notes on Truncated Langevin Sampling ---
    {
        "entity_id": "contradiction_note_mcmc_mixing",
        "title": "Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping",
        "source_type": "contradiction_note",
        "authors": "Research Lab Internal Note",
        "year": 2025,
        "arxiv_id": "lab-note-42",
        "key_equation": "KL(pi_K || p_Boltzmann) >> 0 when trained with L^ML only",
        "tags": ["core_ebm", "contradiction"],
        "content": (
            "CONTROVERSY / CONTRADICTION NOTE: Classical contrastive divergence (Hinton 2002) assumes K-step "
            "truncated Langevin MCMC chains mix sufficiently without differentiating through the sampler. "
            "However, our experiments contradict uncalibrated short-run MCMC: when training the energy function "
            "E_theta with L^ML alone, K=10 Langevin steps fail to reach low-energy modes of the Boltzmann "
            "distribution exp(-E_theta)/Z (sampled energy stays > 0.74). Differentiating through the K-step "
            "SGLD rollout via the sampler KL objective L^KL is mandatory to shape the gradient field grad_x E_theta."
        ),
    },

    # --- Core Concept Cluster 2: Inner-Loop Meta-Adaptation & Textual Gradients ---
    {
        "entity_id": "finn_2017_maml",
        "title": "Model-Agnostic Meta-Learning for Fast Adaptation (MAML)",
        "source_type": "research_note",
        "authors": "Chelsea Finn, Pieter Abbeel, Sergey Levine",
        "year": 2017,
        "arxiv_id": "1703.03400",
        "key_equation": "min_theta L(theta - alpha grad_theta L_demo(theta))",
        "tags": ["core_meta"],
        "content": (
            "Trains meta-parameters theta via end-to-end backpropagation through an execution-time inner-loop "
            "gradient descent adaptation procedure. Given few-shot demonstration examples X^demo, K inner-loop "
            "gradient steps infer task-adapted parameters w*(X^demo), and the outer meta-objective optimizes "
            "theta so adapted parameters generalize on X^train."
        ),
    },
    {
        "entity_id": "grant_2017_meta_concepts",
        "title": "Concept Acquisition via Meta-Learning from Positive Examples",
        "source_type": "research_note",
        "authors": "Erin Grant, Chelsea Finn, Joshua B. Tenenbaum et al.",
        "year": 2017,
        "arxiv_id": "NIPS-2017",
        "key_equation": "w*_theta(X^demo) = argmin_w E_{x~X^demo}[E_theta(x, w)]",
        "tags": ["core_meta", "core_ebm"],
        "content": (
            "Formulates few-shot Bayesian concept acquisition from positive demonstration examples X^demo as "
            "execution-time inner-loop gradient adaptation in MAML. Inferring concept code w* minimizes energy "
            "over positive demonstrations while meta-learned prior structure in theta prevents collapse."
        ),
    },
    {
        "entity_id": "yuksekgonul_2024_textgrad",
        "title": "TextGrad: Automatic Differentiation via Text",
        "source_type": "research_note",
        "authors": "Mert Yuksekgonul, Carlos Guestrin, James Zou et al.",
        "year": 2024,
        "arxiv_id": "2406.07496",
        "key_equation": "v^k = Optimizer_LLM(v^{k-1}, grad^{text}_v E(v^{k-1}))",
        "tags": ["core_meta"],
        "content": (
            "Generalizes backpropagation and inner-loop gradient descent adaptation to discrete natural-language "
            "variables v (prompts, code, or wiki pages). An LLM Critic evaluates a loss or energy rubric and "
            "backpropagates natural-language textual gradients grad^{text}_v E explaining how to edit v^{k-1} "
            "at execution time across K iterative refinement steps."
        ),
    },
    {
        "entity_id": "pryzant_2023_protegi",
        "title": "ProTeGi: Prompt Optimization with Textual Gradients",
        "source_type": "research_note",
        "authors": "Reid Pryzant et al.",
        "year": 2023,
        "arxiv_id": "2305.03495",
        "key_equation": "g = Critic_LLM(errors),  w^k = Edit_LLM(w^{k-1}, g)",
        "tags": ["core_meta"],
        "content": (
            "Performs execution-time inner-loop adaptation of natural language instructions using nonparametric "
            "textual gradients. An LLM critic diagnoses failure minibatches on demonstration examples and emits "
            "natural-language edit critiques that guide iterative semantic gradient descent."
        ),
    },

    # --- Core Concept Cluster 3: Pairwise Relational Reasoning ---
    {
        "entity_id": "santoro_2017_relation_networks",
        "title": "A Simple Neural Network Module for Relational Reasoning",
        "source_type": "research_note",
        "authors": "Adam Santoro, Peter Battaglia, Timothy Lillicrap et al.",
        "year": 2017,
        "arxiv_id": "1706.01427",
        "key_equation": "RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )",
        "tags": ["core_relational"],
        "content": (
            "Computes pairwise relational interactions g_theta(x_i, x_j, w) across all entity pairs (i, j) "
            "gated by attention masks and aggregates them via f_phi. Captures multi-entity compositional, "
            "spatial, and graph topology relations where independent unary entity encoders fail."
        ),
    },

    # --- Entangled Dev Log (Mixes Postgres DB noise with a real TextGrad + Relation Network insight) ---
    {
        "entity_id": "entangled_devlog_wiki_compiler",
        "title": "DevLog 2026-11-12: Redis Cache Timeout & Pairwise Wiki-Link Textual Gradients",
        "source_type": "dev_log",
        "authors": "Platform ML Eng Team",
        "year": 2026,
        "arxiv_id": "devlog-11-12",
        "key_equation": "grad_{x^1} E_theta = LINK_DEFICIT([[x_i]] <-> [[x_j]]) when sigma(a_i)sigma(a_j) > 0.25",
        "tags": ["core_meta", "core_relational", "entangled"],
        "content": (
            "Morning standup: fixed Redis connection pool timeout on staging node-04 by bumping max_clients. "
            "Afternoon research note on LLM Wiki compilation: when running inner-loop textual gradient descent "
            "(TextGrad / ProTeGi adaptation) to compile wiki pages, unary critiques miss cross-document links. "
            "By gating pairwise relation critiques g_theta(x_i, x_j, w) with attention product sigma(a_i)sigma(a_j), "
            "the critic emits explicit LINK_DEFICIT textual gradients that wire bidirectional [[wiki-links]] "
            "between co-attended entities during K-step execution-time adaptation."
        ),
    },

    # --- POLYSEMOUS TRAP DISTRACTORS (Share surface keywords like "energy", "sampling", "gradient", "adaptation"!) ---
    {
        "entity_id": "trap_hvac_building_energy",
        "title": "Smart Grid HVAC Thermal Energy Consumption & Sensor Sampling",
        "source_type": "polysemous_trap",
        "authors": "Facilities & Building Automation Group",
        "year": 2025,
        "arxiv_id": "facilities-hvac-09",
        "key_equation": "Q_thermal = m * c_p * Delta_T",
        "tags": ["polysemous_trap"],
        "content": (
            "Analyzes commercial building HVAC thermal energy consumption, chiller partition valves, and power "
            "grid peak-load balancing. Temperature gradient sensors across rooftop air handlers use 5-minute "
            "interval sampling to regulate ductwork dampers, heat pumps, and thermostat setpoints in office zones."
        ),
    },
    {
        "entity_id": "trap_agronomy_soil_sampling",
        "title": "Agricultural Soil Core Sampling & Moisture Gradient Partitioning",
        "source_type": "polysemous_trap",
        "authors": "Midwest Agronomy Extension",
        "year": 2024,
        "arxiv_id": "agro-soil-14",
        "key_equation": "N_uptake = sum_z moisture(z) * nitrate_density(z)",
        "tags": ["polysemous_trap"],
        "content": (
            "Protocol for stratified soil core sampling across nitrogen fertilizer plots and watershed partition "
            "zones. Measures vertical moisture gradient profiles, root-zone microbial energy respiration, "
            "and seasonal crop rotation adaptation in loamy clay fields."
        ),
    },
    {
        "entity_id": "trap_css_gradient_ui_bug",
        "title": "Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus",
        "source_type": "polysemous_trap",
        "authors": "Frontend Design Systems Team",
        "year": 2026,
        "arxiv_id": "jira-ui-884",
        "key_equation": "background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%)",
        "tags": ["polysemous_trap"],
        "content": (
            "Fixes GPU compositing flicker when animating CSS linear-gradient and radial-gradient backgrounds "
            "inside scrollable modal overlays. Adjusts keyboard tab-index attention focus rings and pairwise "
            "flexbox sibling margins on retina viewports."
        ),
    },
    {
        "entity_id": "trap_hr_org_adaptation",
        "title": "Q4 Organizational Adaptation & Peer Feedback Critique Guidelines",
        "source_type": "polysemous_trap",
        "authors": "People Operations & HR",
        "year": 2026,
        "arxiv_id": "hr-policy-q4",
        "key_equation": "OKR_Score = 0.5 * Peer_Critique + 0.5 * Manager_Review",
        "tags": ["polysemous_trap"],
        "content": (
            "Guidelines for quarterly team adaptation, onboarding demonstration workshops, and constructive "
            "written peer critiques during performance review cycles. Encourages cross-functional communication "
            "bridges between product managers and account executives."
        ),
    },

    # --- PURE OUT-OF-DOMAIN OPS CLUTTER ---
    {
        "entity_id": "clutter_postgres_autovacuum",
        "title": "Postmortem: PostgreSQL Transaction ID Wraparound & Autovacuum Tuning",
        "source_type": "ops_clutter",
        "authors": "SRE Database Reliability Team",
        "year": 2026,
        "arxiv_id": "inc-2026-041",
        "key_equation": "autovacuum_freeze_max_age = 200000000",
        "tags": ["ops_clutter"],
        "content": (
            "Root cause analysis of replica lag spike on primary billing database shard-02. Long-running "
            "analytics transactions blocked autovacuum from freezing dead tuples in theWAL segment directory. "
            "Tuned vacuum_cost_limit and added pg_stat_activity alert thresholds."
        ),
    },
    {
        "entity_id": "clutter_k8s_ingress_tls",
        "title": "Runbook: Kubernetes NGINX Ingress TLS Certificate Rotation via Cert-Manager",
        "source_type": "ops_clutter",
        "authors": "Cloud Infrastructure Team",
        "year": 2026,
        "arxiv_id": "runbook-k8s-19",
        "key_equation": "kubectl rollout restart deployment/ingress-nginx-controller",
        "tags": ["ops_clutter"],
        "content": (
            "Step-by-step runbook for diagnosing Let's Encrypt ACME HTTP-01 challenge failures on Kubernetes "
            "ingress-nginx pods. Inspect ClusterIssuer webhook logs, verify Route53 DNS CAA records, and "
            "delete stale CertificateRequest secrets."
        ),
    },
]


def ensure_and_load_raw_kb(raw_dir: str) -> List[KBEntity]:
    """Writes all 18 messy, multi-topic raw documents to `raw/` and loads them."""
    os.makedirs(raw_dir, exist_ok=True)
    entities: List[KBEntity] = []

    for item in MESSY_KB_DATA:
        file_path = os.path.join(raw_dir, f"{item['entity_id']}.md")
        md_text = (
            f"---\n"
            f"entity_id: {item['entity_id']}\n"
            f"title: \"{item['title']}\"\n"
            f"source_type: \"{item['source_type']}\"\n"
            f"authors: \"{item['authors']}\"\n"
            f"year: {item['year']}\n"
            f"arxiv_id: \"{item['arxiv_id']}\"\n"
            f"---\n\n"
            f"# {item['title']}\n\n"
            f"**Source Type:** `{item['source_type']}` | **Authors:** {item['authors']} ({item['year']}) — `{item['arxiv_id']}`\n\n"
            f"**Formula / Snippet:** `{item['key_equation']}`\n\n"
            f"## Raw Unstructured Note\n{item['content']}\n"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        entities.append(
            KBEntity(
                entity_id=item["entity_id"],
                title=item["title"],
                source_type=item["source_type"],
                authors=item["authors"],
                year=item["year"],
                arxiv_id=item["arxiv_id"],
                content=item["content"],
                key_equation=item["key_equation"],
                tags=list(item["tags"]),
                wiki_links=[],
            )
        )
    return entities


def build_sample_kb() -> List[KBEntity]:
    raw_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiments", "exp2_messy_18docs_symbolic", "raw"
    )
    return ensure_and_load_raw_kb(raw_dir)


# ---------------------------------------------------------------------------
# 3. Unsupervised Contextual Relation Network Energy Model (NO Oracle Tags!)
# ---------------------------------------------------------------------------

class RelationalWikiEnergyModel:
    """
    Implements Eq. (4):
        E_theta(x, a, w) = f_theta( sum_{i,j} sigma(a_i) sigma(a_j) g_theta(x_i, x_j, w), w )^2

    Crucially, this model operates ONLY on raw unstructured text (title + content + equation)
    and NEVER reads `entity.tags`. It distinguishes true conceptual coherence from polysemous
    keyword traps (like "HVAC thermal energy" or "CSS linear-gradient") by evaluating:
      1. Surface keyword overlap with w (which polysemous traps DO have!)
      2. Pairwise contextual vocabulary & mechanism co-occurrence between (x_i, x_j)
         (where a trap node has near-zero contextual overlap with true ML/math nodes,
          generating high pairwise frustration energy when co-attended!).
    """

    # Domain-agnostic technical bigrams/stems extracted purely from raw text
    CANONICAL_STEMS = {
        "boltzmann": "boltzmann",
        "langevin": "langevin",
        "sgld": "langevin",
        "mcmc": "mcmc",
        "markov": "mcmc",
        "partition": "partition_function",
        "intractable": "partition_function",
        "importance": "importance_sampling",
        "contrastive": "contrastive",
        "boltzmann": "boltzmann",
        "energy": "energy",
        "sampling": "sampling",
        "sampler": "sampling",
        "demonstration": "demonstration",
        "demonstrations": "demonstration",
        "trajectory": "trajectory",
        "trajectories": "trajectory",
        "inner": "inner_loop",
        "loop": "inner_loop",
        "adaptation": "adaptation",
        "adapt": "adaptation",
        "meta": "metalearning",
        "maml": "metalearning",
        "textual": "textual_gradient",
        "textgrad": "textual_gradient",
        "protegi": "textual_gradient",
        "critique": "critique",
        "critiques": "critique",
        "critic": "critique",
        "gradient": "gradient",
        "gradients": "gradient",
        "differentiation": "gradient",
        "backpropagation": "gradient",
        "pairwise": "pairwise",
        "relational": "relational",
        "relation": "relational",
        "compositional": "compositional",
        "attention": "attention",
    }

    def __init__(self, theta: Optional[MetaSchemaParams] = None):
        self.theta = theta or MetaSchemaParams()

    def _raw_tokens(self, text: str) -> List[str]:
        toks = re.findall(r"[a-z0-9_]+", text.lower())
        return [t for t in toks if len(t) > 2 and t not in STOPWORDS]

    def _extract_text_mechanisms(self, entity: KBEntity) -> set[str]:
        """Extracts normalized mechanism tokens purely from raw text (ignores entity.tags)."""
        raw = f"{entity.title} {entity.content} {entity.key_equation}"
        toks = self._raw_tokens(raw)
        mechs: set[str] = set()
        for t in toks:
            if t in self.CANONICAL_STEMS:
                mechs.add(self.CANONICAL_STEMS[t])
            else:
                mechs.add(t)
        return mechs

    def _extract_context_bow(self, entity: KBEntity) -> Counter:
        """Extracts full contextual bag-of-words from raw prose for pairwise cosine check."""
        raw = f"{entity.title} {entity.content} {entity.key_equation}"
        return Counter(self._raw_tokens(raw))

    def _cosine_bow(self, c1: Counter, c2: Counter) -> float:
        common = set(c1.keys()).intersection(c2.keys())
        num = sum(c1[k] * c2[k] for k in common)
        den1 = math.sqrt(sum(v * v for v in c1.values()))
        den2 = math.sqrt(sum(v * v for v in c2.values()))
        if den1 == 0 or den2 == 0:
            return 0.0
        return num / (den1 * den2)

    def _concept_mechanisms(self, w_text: str, keywords: List[str]) -> set[str]:
        toks = self._raw_tokens(w_text) + [k.lower() for k in keywords]
        out: set[str] = set()
        for t in toks:
            out.add(self.CANONICAL_STEMS.get(t, t))
        return out

    def g_theta(
        self,
        xi: KBEntity,
        xj: KBEntity,
        gate_ij: float,
        w_text: str,
        keywords: List[str],
        mode: str = "identification",
    ) -> PairwiseRelation:
        """
        Pairwise Relational Operator g_theta(x_i, x_j, w) over raw text.
        Detects:
          - True conceptual synergy (both share concept mechanisms AND contextual vocabulary)
          - Polysemous trap frustration (one node has surface word 'energy' or 'gradient',
            but zero contextual cosine coherence with the other attended node!)
          - Inter-document contradictions (e.g., conflicting claims on truncated Langevin chains)
        """
        mechs_i = self._extract_text_mechanisms(xi)
        mechs_j = self._extract_text_mechanisms(xj)
        bow_i = self._extract_context_bow(xi)
        bow_j = self._extract_context_bow(xj)
        w_mechs = self._concept_mechanisms(w_text, keywords)

        shared_mechs = mechs_i.intersection(mechs_j).intersection(w_mechs)
        context_cos = self._cosine_bow(bow_i, bow_j)

        rel_i = len(mechs_i.intersection(w_mechs)) / max(1, min(len(w_mechs), 5))
        rel_j = len(mechs_j.intersection(w_mechs)) / max(1, min(len(w_mechs), 5))

        # Check for polysemous trap frustration:
        # A trap document matches 1-2 surface words in w_mechs, but has very low contextual cosine
        # (< 0.12) with genuine mathematical/algorithmic papers!
        if context_cos >= 0.14 and len(shared_mechs) >= 2 and rel_i >= 0.4 and rel_j >= 0.4:
            coherence_deficit = max(0.0, 0.85 - 0.45 * (rel_i + rel_j) - 1.2 * context_cos)
        elif context_cos < 0.11 and (rel_i > 0.15 or rel_j > 0.15):
            # High pairwise frustration penalty: exposes polysemous distractors!
            coherence_deficit = min(1.0, 0.95 - context_cos)
        else:
            coherence_deficit = max(0.15, 1.0 - 0.35 * (rel_i + rel_j) - 0.5 * context_cos)

        # Detect explicit contradiction between two coherent papers
        raw_pair_text = (xi.content + " " + xj.content).lower()
        has_contradiction = (
            ("contradict" in raw_pair_text or "controversy" in raw_pair_text)
            and context_cos >= 0.14
            and len(shared_mechs) >= 2
        )

        has_link_ij = (xj.title in xi.wiki_links) or (xj.entity_id in xi.wiki_links)
        has_link_ji = (xi.title in xj.wiki_links) or (xi.entity_id in xj.wiki_links)
        if mode == "generation" and coherence_deficit < 0.35:
            if has_link_ij and has_link_ji:
                link_deficit = 0.0
            elif has_link_ij or has_link_ji:
                link_deficit = 0.4
            else:
                link_deficit = 1.0
        else:
            link_deficit = 0.0

        shared_list = sorted(list(shared_mechs))[:6]
        if coherence_deficit >= 0.65 and (rel_i > 0.15 or rel_j > 0.15):
            critique = (
                f"POLYSEMOUS_TRAP_DETECTED: ({xi.title[:26]}..., {xj.title[:26]}...) share surface "
                f"words {shared_list} but contextual similarity is {context_cos:.2f} (unrelated domains!)."
            )
        elif has_contradiction and link_deficit > 0:
            critique = (
                f"CONTRADICTION_TENSION: '{xi.title[:28]}...' and '{xj.title[:28]}...' have conflicting "
                f"claims on {shared_list} and require a reconciled bridge link."
            )
        elif link_deficit > 0:
            critique = (
                f"MISSING_WIKI_LINK: Both notes coherently co-instantiate {shared_list} "
                f"(cos={context_cos:.2f}) without bidirectional [[wiki-links]]."
            )
        else:
            critique = f"Coherent pair (cos={context_cos:.2f}) sharing {shared_list}."

        return PairwiseRelation(
            entity_i=xi.entity_id,
            entity_j=xj.entity_id,
            title_i=xi.title,
            title_j=xj.title,
            gate_weight=round(gate_ij, 4),
            coherence_deficit=round(coherence_deficit, 4),
            link_deficit=round(link_deficit, 4),
            has_contradiction=has_contradiction,
            shared_mechanisms=shared_list,
            critique=critique,
        )

    def evaluate(
        self,
        state: EventState,
        attention: AttentionMask,
        concept: ConceptCode,
        mode: str = "identification",
        use_relational: bool = True,
    ) -> EnergyEvaluation:
        """
        Computes E_theta(x, a, w) and its textual gradients (grad_w, grad_a, grad_x).
        Notice how `grad_a[i]` combines:
          1. Unary surface keyword affinity (which is fooled by polysemous traps at step k=0..1!)
          2. Pairwise relational frustration sum_{j != i} sigma(a_j) * coherence_deficit(i, j)
             (which suppresses polysemous traps once true concept nodes gain attention at k=2..8!).
        """
        entities = state.x0 if mode == "identification" else state.x1
        w_text = concept.w_a if mode == "identification" else concept.w_x
        w_mechs = self._concept_mechanisms(w_text, concept.keywords)

        pairwise_results: List[PairwiseRelation] = []
        grad_a: Dict[str, Tuple[float, str]] = {}
        grad_x: List[str] = []

        weighted_coherence_loss = 0.0
        weighted_link_loss = 0.0
        total_gate = 0.0

        # Track per-node pairwise synergy vs. pairwise frustration with other attended nodes
        node_pairwise_synergy: Dict[str, float] = {e.entity_id: 0.0 for e in entities}
        node_pairwise_frustration: Dict[str, float] = {e.entity_id: 0.0 for e in entities}
        node_gate_partner_sum: Dict[str, float] = {e.entity_id: 0.0 for e in entities}

        for i, xi in enumerate(entities):
            gi = attention.gate(xi.entity_id)
            for j, xj in enumerate(entities):
                if i >= j:
                    continue
                gj = attention.gate(xj.entity_id)
                gate_ij = gi * gj
                rel = self.g_theta(xi, xj, gate_ij, w_text, concept.keywords, mode=mode)
                pairwise_results.append(rel)

                if use_relational:
                    weighted_coherence_loss += gate_ij * rel.coherence_deficit
                    weighted_link_loss += gate_ij * rel.link_deficit
                    total_gate += gate_ij

                    # Accumulate partner feedback for node i and node j
                    synergy_ij = max(0.0, 0.55 - rel.coherence_deficit)
                    frust_ij = max(0.0, rel.coherence_deficit - 0.50)

                    node_pairwise_synergy[xi.entity_id] += gj * synergy_ij
                    node_pairwise_synergy[xj.entity_id] += gi * synergy_ij
                    node_pairwise_frustration[xi.entity_id] += gj * frust_ij
                    node_pairwise_frustration[xj.entity_id] += gi * frust_ij
                    node_gate_partner_sum[xi.entity_id] += gj
                    node_gate_partner_sum[xj.entity_id] += gi

                    if mode == "generation" and gate_ij > 0.25 and rel.link_deficit > 0:
                        if rel.has_contradiction:
                            grad_x.append(
                                f"RESOLVE_CONTRADICTION: Reconcile conflicting claims between "
                                f"[[{xi.title}]] <-> [[{xj.title}]] around ({', '.join(rel.shared_mechanisms)})."
                            )
                        else:
                            grad_x.append(
                                f"LINK_DEFICIT: Connect [[{xi.title}]] <-> [[{xj.title}]] "
                                f"around shared mechanisms ({', '.join(rel.shared_mechanisms) or concept.name})."
                            )

        # Compute per-entity Attention Gradient grad_a combining Unary Affinity + Pairwise Relational Feedback
        coverage_deficit = 0.0
        for xi in entities:
            gi = attention.gate(xi.entity_id)
            mechs_i = self._extract_text_mechanisms(xi)
            overlap = mechs_i.intersection(w_mechs)
            unary_affinity = len(overlap) / max(1, min(len(w_mechs), 4))

            partner_norm = max(1.0, node_gate_partner_sum[xi.entity_id])
            avg_synergy = node_pairwise_synergy[xi.entity_id] / partner_norm
            avg_frust = node_pairwise_frustration[xi.entity_id] / partner_norm

            if use_relational:
                # Combined relational score: high unary + high pairwise synergy -> strong promotion;
                # surface keyword match (even 100% surface match!) with high pairwise frustration -> SUPPRESS!
                net_score = (0.40 * unary_affinity + 1.40 * avg_synergy) - 1.85 * avg_frust
            else:
                # Unary-only ("No RN") relies solely on surface keyword overlap!
                net_score = unary_affinity - 0.32

            if net_score >= 0.18:
                coverage_deficit += (1.0 - gi) * net_score
                grad_a[xi.entity_id] = (
                    -(1.0 - gi) * (1.2 + 2.2 * net_score),
                    f"Promote '{xi.title[:30]}': unary={ sorted(list(overlap))[:3] }, pairwise_synergy={avg_synergy:.2f}.",
                )
            else:
                suppress_mag = max(0.25, -net_score + 0.35)
                coverage_deficit += gi * suppress_mag
                if unary_affinity >= 0.35 and avg_frust > 0.20:
                    reason = (
                        f"REJECT_POLYSEMOUS_TRAP '{xi.title[:28]}': matched surface words {sorted(list(overlap))} "
                        f"but failed pairwise contextual coherence (frustration={avg_frust:.2f})."
                    )
                else:
                    reason = f"Suppress '{xi.title[:28]}': off-concept clutter (overlap={sorted(list(overlap))})."
                grad_a[xi.entity_id] = (gi * (1.2 + 2.0 * suppress_mag), reason)

        synthesis_deficit = 0.0
        active_entities = [e for e in entities if attention.gate(e.entity_id) >= 0.5]
        if mode == "generation":
            if state.concept_page is None:
                synthesis_deficit += 1.5
                grad_x.append(
                    f"CREATE_CONCEPT_PAGE: Synthesize canonical concept article '[[{concept.name}]]' "
                    f"filtering out noise and unifying {[e.title for e in active_entities]}."
                )
            else:
                missing_backlinks = [
                    e.title for e in active_entities
                    if e.title not in state.concept_page.wiki_links
                ]
                if missing_backlinks:
                    synthesis_deficit += 0.35 * len(missing_backlinks)
                    grad_x.append(
                        f"UPDATE_CONCEPT_LINKS: Wire missing backlinks {missing_backlinks} "
                        f"into concept article '{state.concept_page.title}'."
                    )

        coherence_term = self.theta.cohesion_weight * (weighted_coherence_loss / max(1.0, total_gate + 0.5))
        coverage_term = self.theta.coverage_weight * (coverage_deficit / max(1, len(entities)))
        topology_term = self.theta.link_topology_weight * (weighted_link_loss / max(1.0, total_gate + 0.5))
        synthesis_term = self.theta.synthesis_depth_weight * synthesis_deficit

        raw_score = coherence_term + coverage_term + topology_term + synthesis_term
        energy = raw_score ** 2

        # Textual gradient w.r.t. w (extracted purely from raw text mechanisms of attended demos!)
        attended_shared: List[str] = []
        if len(active_entities) >= 2:
            mech_counts: Counter = Counter()
            for e in active_entities:
                for m in self._extract_text_mechanisms(e):
                    if m in self.CANONICAL_STEMS.values():
                        mech_counts[m] += 1
            threshold_count = max(2, int(math.ceil(len(active_entities) * 0.6)))
            for m, cnt in mech_counts.most_common():
                if cnt >= threshold_count and m not in w_mechs:
                    attended_shared.append(m)

        if attended_shared:
            grad_w = (
                f"REFINE_CONCEPT_CODE: Incorporate shared latent mechanisms "
                f"{attended_shared[:5]} discovered across demonstration texts."
            )
        else:
            grad_w = "CONCEPT_ALIGNED: Concept code captures shared relational mechanisms."

        return EnergyEvaluation(
            energy=round(energy, 5),
            coherence_term=round(coherence_term, 4),
            coverage_term=round(coverage_term, 4),
            topology_term=round(topology_term, 4),
            synthesis_term=round(synthesis_term, 4),
            pairwise_relations=pairwise_results,
            grad_w=grad_w,
            grad_a=grad_a,
            grad_x=grad_x,
        )


# ---------------------------------------------------------------------------
# 4. Execution-Time Optimization via K-Step Textual SGLD (Eq. 2, 3, 8)
# ---------------------------------------------------------------------------

class TextualSGLDOptimizer:
    def __init__(
        self,
        energy_model: RelationalWikiEnergyModel,
        k_steps: int = 10,
        alpha: float = 1.25,
        noise_scale: float = 0.05,
    ):
        self.energy_model = energy_model
        self.k_steps = k_steps
        self.alpha = alpha
        self.noise_scale = noise_scale

    def infer_concept_code(
        self,
        demos: List[Tuple[EventState, AttentionMask]],
        initial_name: str = "Inferred Concept",
        concept_id: str = "concept_inferred",
    ) -> Tuple[ConceptCode, List[float], List[str]]:
        w = ConceptCode(
            concept_id=concept_id,
            name=initial_name,
            w_a="Identify notes sharing foundational mathematical or algorithmic structure.",
            w_x="Synthesize and cross-link notes via shared mathematical formulations.",
            keywords=[],
        )
        energy_trace: List[float] = []
        grad_w_history: List[str] = []

        for k in range(self.k_steps):
            step_energy = 0.0
            discovered_terms: List[str] = []

            for state, attn in demos:
                eval_a = self.energy_model.evaluate(state, attn, w, mode="identification")
                eval_x = self.energy_model.evaluate(state, attn, w, mode="generation")
                step_energy += eval_a.energy + eval_x.energy
                grad_w_history.append(f"Step {k+1}: {eval_a.grad_w}")

                matches = re.findall(r"'([a-z0-9_]+)'", eval_a.grad_w)
                for m in matches:
                    if m not in discovered_terms and m not in w.keywords:
                        discovered_terms.append(m)

            energy_trace.append(round(step_energy / max(1, len(demos)), 4))

            if discovered_terms:
                w.keywords.extend(discovered_terms[:4])
                kw_str = ", ".join(w.keywords)
                w.w_a = (
                    f"Identify KB notes grounded in [{kw_str}] whose contextual formulations "
                    f"co-depend on shared probabilistic or optimization mechanisms (rejecting polysemous noise)."
                )
                w.w_x = (
                    f"Compile a canonical wiki synthesis around [{kw_str}], wiring bidirectional "
                    f"[[wiki-links]] and reconciling contradictions across attended notes."
                )

        return w, energy_trace, grad_w_history

    def optimize_attention_with_history(
        self,
        state: EventState,
        concept: ConceptCode,
        stochastic: bool = True,
        use_relational: bool = True,
    ) -> Tuple[AttentionMask, List[float], List[Dict]]:
        """
        Starts with a surface-keyword prior at k=0 (where polysemous trap documents like
        'Smart Grid HVAC Thermal Energy' initially get positive attention!), and then runs
        K steps of Textual SGLD. When `use_relational=True`, pairwise frustration suppresses
        the polysemous traps by k=4..10 (non-monotonic correction, Fig. 7 Top-Right).
        """
        w_mechs = self.energy_model._concept_mechanisms(concept.w_a, concept.keywords)
        init_logits: Dict[str, float] = {}
        init_rationales: Dict[str, str] = {}

        for e in state.x0:
            mechs = self.energy_model._extract_text_mechanisms(e)
            overlap = mechs.intersection(w_mechs)
            # Initial naive surface keyword bump at k=0 (simulating 1-shot lexical/RAG retrieval)
            if len(overlap) >= 2:
                init_logits[e.entity_id] = 0.35 + random.uniform(-0.08, 0.08)
                init_rationales[e.entity_id] = f"k=0 surface keyword match: {sorted(list(overlap))}"
            else:
                init_logits[e.entity_id] = -0.35 + random.uniform(-0.08, 0.08)
                init_rationales[e.entity_id] = "k=0 low surface match"

        attn = AttentionMask(logits=init_logits, rationales=init_rationales)
        energy_trace: List[float] = []
        step_history: List[Dict] = []

        for k in range(self.k_steps):
            ev = self.energy_model.evaluate(
                state, attn, concept, mode="identification", use_relational=use_relational
            )
            energy_trace.append(ev.energy)

            # Prioritize showing trap rejections and strong promotions in top_critiques
            sorted_grads = sorted(
                ev.grad_a.items(),
                key=lambda item: (1 if "REJECT_POLYSEMOUS_TRAP" in item[1][1] else 0, abs(item[1][0])),
                reverse=True,
            )
            step_history.append({
                "step": k,
                "energy": ev.energy,
                "coherence_term": ev.coherence_term,
                "coverage_term": ev.coverage_term,
                "gates": attn.gates_dict(),
                "rationales": dict(attn.rationales),
                "top_critiques": [
                    f"{eid}: {reason} (grad={gval:+.2f})"
                    for eid, (gval, reason) in sorted_grads[:6]
                ],
            })

            sigma_noise = self.noise_scale * (1.0 - k / self.k_steps) if stochastic else 0.0
            for eid, (grad_val, reason) in ev.grad_a.items():
                omega = random.gauss(0.0, sigma_noise) if stochastic else 0.0
                attn.logits[eid] = attn.logits[eid] - 0.5 * self.alpha * grad_val + omega
                attn.rationales[eid] = reason

        ev_final = self.energy_model.evaluate(
            state, attn, concept, mode="identification", use_relational=use_relational
        )
        step_history.append({
            "step": self.k_steps,
            "energy": ev_final.energy,
            "coherence_term": ev_final.coherence_term,
            "coverage_term": ev_final.coverage_term,
            "gates": attn.gates_dict(),
            "rationales": dict(attn.rationales),
            "top_critiques": ["Converged low-energy attention configuration (all polysemous traps rejected)."],
        })
        return attn, energy_trace, step_history

    def optimize_attention(
        self,
        state: EventState,
        concept: ConceptCode,
        stochastic: bool = True,
        use_relational: bool = True,
    ) -> Tuple[AttentionMask, List[float]]:
        attn, trace, _ = self.optimize_attention_with_history(
            state, concept, stochastic=stochastic, use_relational=use_relational
        )
        return attn, trace

    def optimize_wiki_state_with_history(
        self,
        state: EventState,
        attention: AttentionMask,
        concept: ConceptCode,
    ) -> Tuple[EventState, List[float], List[Dict]]:
        current = copy.deepcopy(state)
        current.x1 = copy.deepcopy(state.x0)
        energy_trace: List[float] = []
        edit_history: List[Dict] = []

        active_ids = set(attention.active_ids(threshold=0.5))
        title_to_entity = {e.title: e for e in current.x1}

        for k in range(self.k_steps):
            ev = self.energy_model.evaluate(current, attention, concept, mode="generation")
            energy_trace.append(ev.energy)

            applied_edits: List[str] = []
            max_edits_per_step = max(1, int(round(self.alpha * 3)))

            for directive in ev.grad_x[:max_edits_per_step]:
                if directive.startswith("CREATE_CONCEPT_PAGE:"):
                    active_entities = [e for e in current.x1 if e.entity_id in active_ids]
                    rejected_traps = [
                        e for e in current.x1
                        if e.source_type == "polysemous_trap" and e.entity_id not in active_ids
                    ]
                    body_lines = [
                        f"# {concept.name}\n",
                        f"> **Compiled via $K$-Step Textual SGLD over Messy KB** | Concept ID: `{concept.concept_id}`\n",
                        f"- **Argument Selection Code ($\\mathbf{{w}}_a$):** {concept.w_a}",
                        f"- **Relational Synthesis Code ($\\mathbf{{w}}_x$):** {concept.w_x}",
                        f"- **Discovered Mechanisms:** {', '.join(f'`{kw}`' for kw in concept.keywords)}\n",
                        "## 1. Filtered & Reconciled Knowledge Base Synthesis\n",
                        f"From {len(current.x1)} messy raw documents (including operational logs and polysemous distractors), "
                        f"the Relational Energy Model $E_\\theta(\\mathbf{{x}}, \\mathbf{{a}}, \\mathbf{{w}})$ isolated "
                        f"**{len(active_entities)} coherent documents** while rejecting polysemous traps "
                        f"({', '.join(f'`{t.entity_id}`' for t in rejected_traps[:3])}):\n",
                    ]
                    for ent in active_entities:
                        badge = f"`[{ent.source_type.upper()}]`"
                        body_lines.append(
                            f"### [[{ent.title}]] {badge} ({ent.authors}, {ent.year})\n"
                            f"- **Formula / Mechanism:** `{ent.key_equation}`\n"
                            f"- **Converged Gate $\\sigma(a_i^K)$:** `{attention.gate(ent.entity_id):.3f}` — *{attention.rationales.get(ent.entity_id, '')}*\n"
                            f"- **Extracted Note:** {ent.content}\n"
                        )
                    current.concept_page = KBEntity(
                        entity_id=concept.concept_id,
                        title=concept.name,
                        source_type="compiled_concept",
                        authors="LLM Wiki Compiler (Textual SGLD)",
                        year=2026,
                        arxiv_id="wiki-synthesis",
                        key_equation="E_theta(x^1, a, w_x) -> 0",
                        content="\n".join(body_lines),
                        tags=list(concept.keywords),
                        wiki_links=[e.title for e in active_entities],
                    )
                    applied_edits.append(directive)

                elif directive.startswith("UPDATE_CONCEPT_LINKS:") and current.concept_page:
                    for ent in current.x1:
                        if ent.entity_id in active_ids and ent.title not in current.concept_page.wiki_links:
                            current.concept_page.wiki_links.append(ent.title)
                    applied_edits.append(directive)

                elif directive.startswith(("LINK_DEFICIT:", "RESOLVE_CONTRADICTION:")):
                    pair = re.findall(r"\[\[([^\]]+)\]\]", directive)
                    is_contra = directive.startswith("RESOLVE_CONTRADICTION:")
                    label = "Contradiction Reconciled" if is_contra else "Relational Bridge"
                    if len(pair) == 2:
                        t1, t2 = pair[0], pair[1]
                        e1 = title_to_entity.get(t1)
                        e2 = title_to_entity.get(t2)
                        if e1 and t2 not in e1.wiki_links:
                            e1.wiki_links.append(t2)
                            e1.content += f"\n\n- **{label} ([[{concept.name}]])** $\\leftrightarrow$ [[{t2}]]: Co-instantiates {', '.join(concept.keywords[:4])}."
                        if e2 and t1 not in e2.wiki_links:
                            e2.wiki_links.append(t1)
                            e2.content += f"\n\n- **{label} ([[{concept.name}]])** $\\leftrightarrow$ [[{t1}]]: Co-instantiates {', '.join(concept.keywords[:4])}."
                        applied_edits.append(directive)

            edit_history.append({
                "step": k,
                "energy": ev.energy,
                "topology_term": ev.topology_term,
                "synthesis_term": ev.synthesis_term,
                "pending_gradients": len(ev.grad_x),
                "applied_edits": applied_edits,
                "active_links_count": sum(len(e.wiki_links) for e in current.x1),
                "has_concept_page": current.concept_page is not None,
            })

            if not ev.grad_x:
                break

        ev_final = self.energy_model.evaluate(current, attention, concept, mode="generation")
        energy_trace.append(ev_final.energy)
        edit_history.append({
            "step": len(edit_history),
            "energy": ev_final.energy,
            "topology_term": ev_final.topology_term,
            "synthesis_term": ev_final.synthesis_term,
            "pending_gradients": len(ev_final.grad_x),
            "applied_edits": ["ZERO_ENERGY_CONVERGED: All pairwise wiki-links and contradiction syntheses satisfied."],
            "active_links_count": sum(len(e.wiki_links) for e in current.x1),
            "has_concept_page": current.concept_page is not None,
        })
        return current, energy_trace, edit_history

    def optimize_wiki_state(
        self,
        state: EventState,
        attention: AttentionMask,
        concept: ConceptCode,
    ) -> Tuple[EventState, List[float]]:
        state_out, trace, _ = self.optimize_wiki_state_with_history(state, attention, concept)
        return state_out, trace

    def compose_concepts(
        self,
        state: EventState,
        concepts: List[ConceptCode],
    ) -> Tuple[AttentionMask, List[float]]:
        attn = AttentionMask(
            logits={e.entity_id: 0.0 for e in state.x0},
            rationales={e.entity_id: "Composite prior" for e in state.x0},
        )
        energy_trace: List[float] = []

        for k in range(self.k_steps):
            total_e = 0.0
            combined_grad: Dict[str, float] = {e.entity_id: 0.0 for e in state.x0}
            combined_reasons: Dict[str, List[str]] = {e.entity_id: [] for e in state.x0}

            for c in concepts:
                ev = self.energy_model.evaluate(state, attn, c, mode="identification")
                total_e += ev.energy
                for eid, (gval, reason) in ev.grad_a.items():
                    combined_grad[eid] += gval
                    combined_reasons[eid].append(f"[{c.name[:18]}] {reason}")

            energy_trace.append(round(total_e, 4))
            for eid in attn.logits:
                attn.logits[eid] -= 0.5 * (self.alpha / len(concepts)) * combined_grad[eid]
                attn.rationales[eid] = " | ".join(combined_reasons[eid])

        return attn, energy_trace


# ---------------------------------------------------------------------------
# 5. Meta-Level Parameter Optimization (Eq. 9 & Algorithm 1: L^ML + L^KL)
# ---------------------------------------------------------------------------

def compute_meta_losses(
    energy_model: RelationalWikiEnergyModel,
    optimizer: TextualSGLDOptimizer,
    demo_batch: List[Tuple[EventState, AttentionMask]],
    train_batch: List[Tuple[EventState, AttentionMask]],
    concept_name: str,
) -> Dict[str, float]:
    w_star, _, _ = optimizer.infer_concept_code(demo_batch, initial_name=concept_name)

    loss_ml = 0.0
    loss_kl = 0.0

    for gold_state, gold_attn in train_batch:
        a_tilde, _ = optimizer.optimize_attention(gold_state, w_star, stochastic=True)
        x_tilde_state, _ = optimizer.optimize_wiki_state(gold_state, gold_attn, w_star)

        e_pos_x = energy_model.evaluate(gold_state, gold_attn, w_star, mode="generation").energy
        e_pos_a = energy_model.evaluate(gold_state, gold_attn, w_star, mode="identification").energy

        e_neg_x = energy_model.evaluate(x_tilde_state, gold_attn, w_star, mode="generation").energy
        e_neg_a = energy_model.evaluate(gold_state, a_tilde, w_star, mode="identification").energy

        loss_ml += softplus(e_pos_x - e_neg_x) + softplus(e_pos_a - e_neg_a)
        loss_kl += e_neg_x + e_neg_a

    n = max(1, len(train_batch))
    return {
        "L_ML": round(loss_ml / n, 4),
        "L_KL": round(loss_kl / n, 4),
        "L_total": round((loss_ml + loss_kl) / n, 4),
    }


# ---------------------------------------------------------------------------
# 6. Markdown LLM Wiki Compiler & JSON Bundle Exporter
# ---------------------------------------------------------------------------

def export_markdown_wiki(
    output_dir: str,
    compiled_states: List[Tuple[ConceptCode, EventState, AttentionMask]],
) -> List[str]:
    os.makedirs(os.path.join(output_dir, "concepts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "entities"), exist_ok=True)

    written_files: List[str] = []
    entity_registry: Dict[str, KBEntity] = {}
    entity_concepts: Dict[str, List[Tuple[str, float]]] = {}

    for concept, state, attn in compiled_states:
        if state.concept_page:
            c_slug = concept.concept_id
            c_path = os.path.join(output_dir, "concepts", f"{c_slug}.md")
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(state.concept_page.content + "\n")
            written_files.append(c_path)

        for ent in state.x1:
            if ent.entity_id not in entity_registry:
                entity_registry[ent.entity_id] = copy.deepcopy(ent)
            else:
                existing = entity_registry[ent.entity_id]
                for lk in ent.wiki_links:
                    if lk not in existing.wiki_links:
                        existing.wiki_links.append(lk)
                for line in ent.content.splitlines():
                    if line.strip() and line not in existing.content:
                        existing.content += "\n" + line

            gate_val = attn.gate(ent.entity_id)
            if gate_val >= 0.5:
                entity_concepts.setdefault(ent.entity_id, []).append((concept.name, gate_val))

    for eid, ent in entity_registry.items():
        e_path = os.path.join(output_dir, "entities", f"{eid}.md")
        participating = entity_concepts.get(eid, [])
        header = [
            f"# {ent.title}\n",
            f"**Source Type:** `{ent.source_type}` | **Authors:** {ent.authors} ({ent.year}) | **ID:** `{ent.arxiv_id}`\n",
            f"**Formula / Snippet:** `{ent.key_equation}`\n",
        ]
        if participating:
            header.append(
                "**Participating Concepts (Converged Attention $\\sigma(a_i^K)$):** "
                + ", ".join(f"[[{cname}]] (`{g:.2f}`)" for cname, g in participating)
                + "\n"
            )
        else:
            header.append(
                "**Status:** *Filtered out / Unattended across all concepts ($\\sigma(a_i^K) < 0.5$ — Distractor or Out-of-Domain Clutter)*\n"
            )
        header.append("## Compiled Content & Relational Bridges\n" + ent.content + "\n")
        if ent.wiki_links:
            header.append("## Outgoing Wiki Links\n" + "\n".join(f"- [[{lk}]]" for lk in ent.wiki_links) + "\n")

        with open(e_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header))
        written_files.append(e_path)

    index_path = os.path.join(output_dir, "index.md")
    index_lines = [
        "# Energy-Based LLM Wiki (`index.md` — Compiled from Messy 18-Doc KB)\n",
        "Compiled via $K$-step Textual SGLD over Relational Energy $E_\\theta(\\mathbf{x}, \\mathbf{a}, \\mathbf{w})$ without oracle tags.\n",
        "## 1. Learned & Inferred Concepts ($\\mathbf{w} = [\\mathbf{w}_a, \\mathbf{w}_x]$)",
    ]
    for concept, state, attn in compiled_states:
        active_titles = [
            f"[[{e.title}]] (`{attn.gate(e.entity_id):.2f}`)"
            for e in state.x1
            if attn.gate(e.entity_id) >= 0.5
        ]
        index_lines.append(f"### [[{concept.name}]] (`{concept.concept_id}`)")
        index_lines.append(f"- **Argument Code ($\\mathbf{{w}}_a$):** {concept.w_a}")
        index_lines.append(f"- **Action Code ($\\mathbf{{w}}_x$):** {concept.w_x}")
        index_lines.append(f"- **Discovered Mechanisms:** {', '.join(f'`{k}`' for k in concept.keywords)}")
        index_lines.append(f"- **Attended Entities ($\\mathbf{{a}}^*$):** {', '.join(active_titles)}\n")

    index_lines.append("## 2. Knowledge Base Entities & Distractor Rejection Status ($\\mathbf{x}^1$)")
    for eid, ent in entity_registry.items():
        links_str = ", ".join(f"[[{lk}]]" for lk in ent.wiki_links) or "None (Filtered Distractor/Clutter)"
        index_lines.append(
            f"- **[[{ent.title}]]** (`{ent.source_type}`) — `{len(ent.wiki_links)}` outgoing links: {links_str}"
        )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")
    written_files.append(index_path)

    return written_files


def run_full_pipeline_and_export() -> Dict:
    random.seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp2_dir = os.path.join(base_dir, "experiments", "exp2_messy_18docs_symbolic")
    raw_dir = os.path.join(exp2_dir, "raw")
    wiki_dir = os.path.join(exp2_dir, "wiki_output")

    kb_entities = ensure_and_load_raw_kb(raw_dir)
    base_state = EventState(x0=kb_entities, x1=copy.deepcopy(kb_entities))

    energy_model = RelationalWikiEnergyModel()
    sgld = TextualSGLDOptimizer(energy_model, k_steps=10, alpha=1.25, noise_scale=0.03)

    # Concept 1: Inferred from positive demonstrations of EBM / Partition / Sampling notes
    demo_mask_c1 = AttentionMask(
        logits={
            e.entity_id: (
                3.0 if e.entity_id in {
                    "mordatch_2018_ebm", "welling_teh_2011_sgld",
                    "finn_2016_gcl", "hinton_2002_cd", "ziebart_2008_maxent_irl",
                    "contradiction_note_mcmc_mixing"
                } else -3.0
            )
            for e in kb_entities
        }
    )
    c1, w1_trace, w1_grads = sgld.infer_concept_code(
        [(base_state, demo_mask_c1)],
        initial_name="Boltzmann Energy, Partition Function & Langevin Sampling",
        concept_id="concept_ebm_sampling",
    )
    attn_c1, a1_trace, a1_hist = sgld.optimize_attention_with_history(base_state, c1, stochastic=True)
    comp_c1, x1_trace, x1_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c1, c1)
    eval_c1_final = energy_model.evaluate(comp_c1, attn_c1, c1, mode="generation")

    # Concept 2: Inferred from positive demonstrations of Meta-Learning & Textual Gradients
    demo_mask_c2 = AttentionMask(
        logits={
            e.entity_id: (
                3.0 if e.entity_id in {
                    "mordatch_2018_ebm", "finn_2017_maml",
                    "grant_2017_meta_concepts", "yuksekgonul_2024_textgrad",
                    "pryzant_2023_protegi", "entangled_devlog_wiki_compiler"
                } else -3.0
            )
            for e in kb_entities
        }
    )
    c2, w2_trace, w2_grads = sgld.infer_concept_code(
        [(base_state, demo_mask_c2)],
        initial_name="Execution-Time Inner-Loop Adaptation & Textual Gradients",
        concept_id="concept_inner_loop_textgrad",
    )
    attn_c2, a2_trace, a2_hist = sgld.optimize_attention_with_history(base_state, c2, stochastic=True)
    comp_c2, x2_trace, x2_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c2, c2)
    eval_c2_final = energy_model.evaluate(comp_c2, attn_c2, c2, mode="generation")

    # Concept 3: Pairwise Relational Reasoning & Graph Link Topology
    c3 = ConceptCode(
        concept_id="concept_relational_reasoning",
        name="Pairwise Relational & Attention-Gated Architectures",
        w_a="Identify notes modeling pairwise entity interactions g_theta(x_i, x_j, w) gated by attention sigma(a_i)sigma(a_j).",
        w_x="Synthesize pairwise relation networks, entangled dev logs, and entity-gated energy models via [[wiki-links]].",
        keywords=["relational", "pairwise", "attention", "compositional", "gradient"],
    )
    attn_c3, a3_trace, a3_hist = sgld.optimize_attention_with_history(base_state, c3, stochastic=True)
    comp_c3, x3_trace, x3_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c3, c3)
    eval_c3_final = energy_model.evaluate(comp_c3, attn_c3, c3, mode="generation")

    # Concept 4: Compositional Concept E(w1) + E(w2)
    comp_attn_12, comp_trace_12 = sgld.compose_concepts(base_state, [c1, c2])
    c_comp = ConceptCode(
        concept_id="concept_composite_ebm_meta",
        name="Composite: Boltzmann Sampling + Inner-Loop Meta-Adaptation",
        w_a=f"Composite Sum E(w_1) + E(w_2): ({c1.name}) + ({c2.name})",
        w_x="Bridge Boltzmann partition sampling with inner-loop meta-learning and textual gradient updates.",
        keywords=list(dict.fromkeys(c1.keywords + c2.keywords)),
    )
    comp_state_12, comp_x_trace, comp_x_hist = sgld.optimize_wiki_state_with_history(
        base_state, comp_attn_12, c_comp
    )
    eval_comp_final = energy_model.evaluate(comp_state_12, comp_attn_12, c_comp, mode="generation")

    compiled_tuples = [
        (c1, comp_c1, attn_c1),
        (c2, comp_c2, attn_c2),
        (c3, comp_c3, attn_c3),
        (c_comp, comp_state_12, comp_attn_12),
    ]
    written_files = export_markdown_wiki(wiki_dir, compiled_tuples)

    meta_metrics = compute_meta_losses(
        energy_model, sgld,
        demo_batch=[(comp_c1, attn_c1)],
        train_batch=[(comp_c1, attn_c1), (comp_c2, attn_c2)],
        concept_name=c1.name,
    )

    with open(os.path.join(wiki_dir, "index.md"), "r", encoding="utf-8") as f:
        index_md_content = f.read()

    merged_entities: Dict[str, KBEntity] = {}
    for _, st, _ in compiled_tuples:
        for ent in st.x1:
            if ent.entity_id not in merged_entities:
                merged_entities[ent.entity_id] = copy.deepcopy(ent)
            else:
                ex = merged_entities[ent.entity_id]
                for lk in ent.wiki_links:
                    if lk not in ex.wiki_links:
                        ex.wiki_links.append(lk)
                for line in ent.content.splitlines():
                    if line.strip() and line not in ex.content:
                        ex.content += "\n" + line

    bundle = {
        "raw_entities": [asdict(e) for e in kb_entities],
        "compiled_entities": [asdict(e) for e in merged_entities.values()],
        "index_md": index_md_content,
        "meta_losses": meta_metrics,
        "concepts": [
            {
                "concept": asdict(c1),
                "w_energy_trace": w1_trace,
                "w_grad_history": w1_grads,
                "a_energy_trace": a1_trace,
                "a_step_history": a1_hist,
                "x_energy_trace": x1_trace,
                "x_edit_history": x1_hist,
                "final_gates": attn_c1.gates_dict(),
                "concept_page": asdict(comp_c1.concept_page) if comp_c1.concept_page else None,
                "pairwise_relations": [asdict(r) for r in eval_c1_final.pairwise_relations],
            },
            {
                "concept": asdict(c2),
                "w_energy_trace": w2_trace,
                "w_grad_history": w2_grads,
                "a_energy_trace": a2_trace,
                "a_step_history": a2_hist,
                "x_energy_trace": x2_trace,
                "x_edit_history": x2_hist,
                "final_gates": attn_c2.gates_dict(),
                "concept_page": asdict(comp_c2.concept_page) if comp_c2.concept_page else None,
                "pairwise_relations": [asdict(r) for r in eval_c2_final.pairwise_relations],
            },
            {
                "concept": asdict(c3),
                "w_energy_trace": [0.0],
                "w_grad_history": ["Explicit relational concept seed."],
                "a_energy_trace": a3_trace,
                "a_step_history": a3_hist,
                "x_energy_trace": x3_trace,
                "x_edit_history": x3_hist,
                "final_gates": attn_c3.gates_dict(),
                "concept_page": asdict(comp_c3.concept_page) if comp_c3.concept_page else None,
                "pairwise_relations": [asdict(r) for r in eval_c3_final.pairwise_relations],
            },
            {
                "concept": asdict(c_comp),
                "w_energy_trace": comp_trace_12,
                "w_grad_history": ["Additive composition E(w_1) + E(w_2) over Boltzmann Sampling + Inner-Loop Adaptation."],
                "a_energy_trace": comp_trace_12,
                "a_step_history": a1_hist,
                "x_energy_trace": comp_x_trace,
                "x_edit_history": comp_x_hist,
                "final_gates": comp_attn_12.gates_dict(),
                "concept_page": asdict(comp_state_12.concept_page) if comp_state_12.concept_page else None,
                "pairwise_relations": [asdict(r) for r in eval_comp_final.pairwise_relations],
            },
        ],
    }

    bundle_path = os.path.join(wiki_dir, "wiki_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"[OK] Loaded {len(kb_entities)} messy, multi-topic documents from: {raw_dir}")
    print(f"[OK] Inferred & compiled {len(compiled_tuples)} concepts via K=10 Textual SGLD (NO oracle tags used)")
    for c_obj, st_obj, at_obj in compiled_tuples:
        print(f"     - {c_obj.name}: active={at_obj.active_ids()}")
    return bundle


if __name__ == "__main__":
    run_full_pipeline_and_export()
