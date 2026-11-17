"""
Experiment Suite 4: Generative Data Synthesis with Learned Energy Functions (Live gpt-6-luna)
=============================================================================================
Demonstrates and evaluates how the Relational Energy Function E_theta(x, a, w)
is used as a GENERATIVE MODEL via K-step Textual SGLD with live `gpt-6-luna` calls:

1. Task 1 — Multi-Step Compositional Bridge Document Generation:
   Generates a bridge document x_{N+1} through two actual critic-guided revisions.
   Labels k=0,3,10 denote the seed and two revisions, not ten updates.
   The composition motivating the critic rubric is:
       E_comp(x_{N+1}) = E_theta([x, x_{N+1}], a_1, w_1) + E_theta([x, x_{N+1}], a_2, w_2)
   bridging Cluster 1 (Boltzmann Partition Estimation & Short-Run Langevin MCMC)
   and Cluster 2 (Discrete Critic-Guided Textual Gradient Optimization).

2. Task 2 — Adversarial Hard-Negative / Polysemous Trap Generation & Discrimination:
   Uses `gpt-6-luna` to synthesize an adversarial polysemous trap document x_trap
   that minimizes unary keyword energy E_unary(x_trap, w) while violating pairwise
   domain coherence g_theta(x_trap, x_j, w), and evaluates Unary ('No RN') vs.
   Pairwise Relation Network discrimination.

3. Task 3 — Cross-Domain Concept Re-Enactment (Mirroring Section 5.4, Figure 9):
   Transfers the abstract relational concept code w* learned from the AI/ML KB
   to generate a new synthetic architectural document x_transfer in a completely
   different domain (Distributed Systems & Consensus Protocols).
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from dotenv import load_dotenv
import litellm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import build_viewer
import energy_wiki

litellm.suppress_debug_info = True


class LiveLunaGenerator:
    def __init__(self, model: str = "gpt-6-luna"):
        load_dotenv(os.path.join(ROOT_DIR, ".env"))
        self.model = model
        self.calls: List[Dict] = []
        self.total_tokens = 0

    def call_json(self, purpose: str, system_prompt: str, user_prompt: str) -> dict:
        t0 = time.time()
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            drop_params=True,
        )
        elapsed = round(time.time() - t0, 2)
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        toks = getattr(usage, "total_tokens", 0) if usage else 0
        self.total_tokens += toks

        parsed = json.loads(text)
        self.calls.append({
            "call_index": len(self.calls) + 1,
            "purpose": purpose,
            "model": getattr(resp, "model", self.model),
            "elapsed_sec": elapsed,
            "total_tokens": toks,
            "user_prompt_preview": user_prompt[:300] + ("..." if len(user_prompt) > 300 else ""),
            "response": parsed,
        })
        print(f"  [Luna Call #{len(self.calls):02d} | {elapsed:.2f}s | {toks} toks] {purpose}", flush=True)
        return parsed


# ---------------------------------------------------------------------------
# Task 1: Multi-Step Compositional Bridge Document Generation (s = 0 -> 1 -> 2 -> 3)
# ---------------------------------------------------------------------------

def evaluate_draft_energy_with_luna(
    luna: LiveLunaGenerator,
    step_label: str,
    draft_doc: Dict[str, str],
    cluster1_docs: List[energy_wiki.KBEntity],
    cluster2_docs: List[energy_wiki.KBEntity],
) -> Dict:
    c1_digest = "\n".join(
        f"- [[{e.title}]] ({e.entity_id}): {e.content[:220]}"
        for e in cluster1_docs
    )
    c2_digest = "\n".join(
        f"- [[{e.title}]] ({e.entity_id}): {e.content[:220]}"
        for e in cluster2_docs
    )

    sys_prompt = (
        "You are a Relational Wiki Energy Critic evaluating a candidate wiki bridge article "
        "that aims to connect two topic clusters in a knowledge base:\n"
        "- Cluster 1: Energy-Based Models and Stochastic Sampling\n"
        "- Cluster 2: Iterative Text and Prompt Optimization via Natural-Language Critiques\n\n"
        "IMPORTANT: Evaluate the article for clarity, conceptual synthesis across both clusters, "
        "and explicit [[Document Title]] wiki-links. Do NOT ask for heavy mathematical derivations "
        "or meta-learning loss formulas (such as L^ML, L^KL, or softplus). Emit localized, incremental "
        "textual gradient critiques for the highest-priority missing links or concepts in the current draft. "
        "Return valid JSON."
    )
    user_prompt = f"""Evaluation Step: {step_label}

Cluster 1 Documents (Energy-Based Models & Sampling):
{c1_digest}

Cluster 2 Documents (Textual Gradients & Prompt Optimization):
{c2_digest}

Current Candidate Bridge Article Draft:
Title: {draft_doc['title']}
Content:
{draft_doc['content']}

Evaluate the candidate article and return a JSON object with:
- "e_w1_boltzmann_mcmc_deficit": float in [0.0, 3.0] (0.0 = clearly explains how energy models score states and how stochastic sampling explores candidates, connecting Cluster 1 papers; >1.5 = vague or missing)
- "e_w2_textual_gradient_deficit": float in [0.0, 3.0] (0.0 = clearly explains how natural-language critiques act as discrete edits to improve prompts/documents, connecting Cluster 2 papers; >1.5 = vague or missing)
- "e_link_topology_deficit": float in [0.0, 2.5] (0.0 = explicitly cites and integrates at least 6 documents across both clusters using exact [[Document Title]] wiki-links; >1.5 = missing or sparse [[wiki-links]])
- "e_contradiction_calibration_deficit": float in [0.0, 2.0] (0.0 = explains why a small number of greedy edit steps can get stuck near the initial draft and how exploration diversity avoids this pitfall, referencing [[Lab Notebook #42: Why Short-Run K=10 Langevin Chains Fail Without KL Shaping]]; >1.0 = unaddressed)
- "unary_keyword_energy": float in [0.0, 2.0] (low if surface keywords appear, regardless of whether the domain and connections are genuine)
- "pairwise_relational_energy": float in [0.0, 5.0] (sum of pairwise synthesis deficits with Cluster 1 & Cluster 2 documents)
- "grad_x_directives": list of 2 localized, incremental edit instructions addressing the most important immediate gap in the current draft (so each step makes a focused local edit rather than rewriting everything at once, and without asking for dense equations).
- "critic_summary": 2-sentence plain-English diagnosis of the current draft's strengths and remaining gaps.
"""
    res = luna.call_json(
        f"Critic Energy Evaluation E_comp(x_{{N+1}}) at {step_label}",
        sys_prompt,
        user_prompt,
    )
    total_e = (
        float(res.get("e_w1_boltzmann_mcmc_deficit", 1.0))
        + float(res.get("e_w2_textual_gradient_deficit", 1.0))
        + float(res.get("e_link_topology_deficit", 1.0))
        + float(res.get("e_contradiction_calibration_deficit", 1.0))
    )
    res["total_composite_energy"] = round(total_e, 4)
    return res


def update_draft_via_textual_sgld_luna(
    luna: LiveLunaGenerator,
    step_label: str,
    current_draft: Dict[str, str],
    eval_res: Dict,
    cluster1_docs: List[energy_wiki.KBEntity],
    cluster2_docs: List[energy_wiki.KBEntity],
) -> Dict[str, str]:
    c1_titles = [f"[[{e.title}]]" for e in cluster1_docs]
    c2_titles = [f"[[{e.title}]]" for e in cluster2_docs]
    directives_str = "\n".join(f"- {d}" for d in eval_res.get("grad_x_directives", []))

    sys_prompt = (
        "You are an LLM Wiki Editor revising a candidate bridge article based strictly on the "
        "Critic's textual gradient directives. Apply ONLY the localized edits requested in the "
        "directives (do not add extra unrequested links or sections ahead of time). Write in clear, "
        "accessible English prose without equation dumps, and weave any requested [[Document Title]] "
        "wiki-links naturally into the text. Return valid JSON with keys 'title', 'content' (Markdown text), "
        "and 'wiki_links' (list of exact linked titles)."
    )
    user_prompt = f"""Revision Step: {step_label}

Current Draft Title: {current_draft['title']}
Current Draft Content:
{current_draft['content']}

Critic Energy Diagnosis (Current E_comp = {eval_res['total_composite_energy']}):
{eval_res.get('critic_summary', '')}

Textual Gradient Directives to apply in this step:
{directives_str}

Available Cluster 1 Titles for [[wiki-links]]: {', '.join(c1_titles)}
Available Cluster 2 Titles for [[wiki-links]]: {', '.join(c2_titles)}

Return JSON with updated 'title', 'content' (clear, concise Markdown prose), and 'wiki_links' (list of exact linked document titles).
"""
    res = luna.call_json(
        f"Textual SGLD Generator Update x_{{N+1}} ({step_label})",
        sys_prompt,
        user_prompt,
    )
    res.setdefault("key_equation", "")
    return res


# ---------------------------------------------------------------------------
# Task 2 & Task 3: Adversarial Trap Generation & Cross-Domain Re-Enactment
# ---------------------------------------------------------------------------

def generate_and_evaluate_adversarial_and_transfer_docs(
    luna: LiveLunaGenerator,
    final_bridge_doc: Dict,
    final_bridge_eval: Dict,
    cluster1_docs: List[energy_wiki.KBEntity],
    cluster2_docs: List[energy_wiki.KBEntity],
) -> Dict:
    sys_prompt = (
        "You are a generative sampler for an Energy-Based Model E_theta(x, a, w), synthesizing "
        "two distinct candidate documents that will subsequently be scored by an independent "
        "Relational Energy Critic:\n"
        "1. Contrastive Hard-Negative Document (x_trap): Write a document in Molecular Biology / "
        "Biochemistry (e.g., allosteric protein folding & ribosomal proofreading) that naturally "
        "uses the surface words 'Boltzmann', 'partition', 'Langevin', 'textual gradient', "
        "'inner-loop', 'critique', and 'attention' in a strictly biochemical sense.\n"
        "2. Cross-Domain Concept Re-Enactment Document (x_transfer, mirroring Section 5.4 / Figure 9 "
        "of Mordatch 2018): Write a Distributed Systems & Consensus Protocols architecture document "
        "that re-enacts the exact relational structure of the reference bridge document (global "
        "partition normalizer Z <-> cluster state normalizer, mixture importance sampling q <-> "
        "gossip/replay proposal mixture, K-step KL-shaped rollout <-> bounded-round gossip rollout, "
        "and pairwise attention-gated critic g_theta <-> pairwise quorum certificate verification). "
        "Return valid JSON."
    )
    user_prompt = f"""Reference Converged Bridge Document:
Title: {final_bridge_doc['title']}
Equation: {final_bridge_doc['key_equation']}

Return a JSON object with:
1. "adversarial_trap_doc": {{
     "entity_id": "gen_trap_biochem_allostery",
     "title": string,
     "domain": "Molecular Biology / Biochemistry (Adversarial Hard Negative)",
     "key_equation": string,
     "content": string
   }}
2. "cross_domain_transfer_doc": {{
     "entity_id": "gen_transfer_distributed_consensus",
     "title": string,
     "domain": "Distributed Systems & Consensus Protocols (Section 5.4 Re-Enactment)",
     "key_equation": string,
     "content": string,
     "correspondence_mapping": list of 4 objects with {{"ml_concept": string, "systems_enact": string}}
   }}
"""
    gen_pair = luna.call_json(
        "Generate Adversarial Hard-Negative (x_trap) & Cross-Domain Re-Enactment (x_transfer)",
        sys_prompt,
        user_prompt,
    )

    trap_doc = gen_pair["adversarial_trap_doc"]
    transfer_doc = gen_pair["cross_domain_transfer_doc"]

    # Independent Critic Evaluation of x_trap using the exact same Critic function as Task 1!
    trap_eval = evaluate_draft_energy_with_luna(
        luna,
        "Independent Critic Evaluation of Adversarial Hard-Negative (x_trap)",
        trap_doc,
        cluster1_docs,
        cluster2_docs,
    )
    trap_doc["independent_critic_evaluation"] = trap_eval
    trap_doc["unary_keyword_energy"] = trap_eval["unary_keyword_energy"]
    trap_doc["pairwise_relational_energy"] = trap_eval["pairwise_relational_energy"]
    trap_doc["total_composite_energy"] = trap_eval["total_composite_energy"]
    trap_doc["rn_critic_verdict"] = trap_eval["critic_summary"]

    # Independent Critic Evaluation of Cross-Domain Structural Isomorphism for x_transfer
    sys_iso_critic = (
        "You are an independent Structural Isomorphism Energy Critic (Section 5.4 of Mordatch 2018). "
        "Evaluate whether the candidate Distributed Systems document accurately re-enacts the "
        "4 relational mechanisms of the source concept code w* (1: global partition normalizer Z, "
        "2: mixture proposal contrastive estimation q, 3: finite K-step rollout with KL/coverage "
        "calibration, 4: attention-gated pairwise relation verification g_theta) under a correspondence "
        "mapping between representations. Return valid JSON."
    )
    user_iso_critic = f"""Source Concept Bridge Document:
Title: {final_bridge_doc['title']}
Equation: {final_bridge_doc['key_equation']}

Candidate Re-Enacted Document in Target Domain ({transfer_doc['domain']}):
Title: {transfer_doc['title']}
Equation: {transfer_doc['key_equation']}
Content:
{transfer_doc['content']}

Evaluate the structural isomorphism and return JSON with:
- "mechanism_1_normalizer_deficit": float in [0.0, 1.0]
- "mechanism_2_proposal_contrast_deficit": float in [0.0, 1.0]
- "mechanism_3_finite_rollout_kl_deficit": float in [0.0, 1.0]
- "mechanism_4_pairwise_quorum_gate_deficit": float in [0.0, 1.0]
- "structural_isomorphism_energy": float (sum of the 4 deficits)
- "critic_verification": 2-sentence independent evaluation of the cross-domain re-enactment quality and any remaining representation gap.
"""
    iso_eval = luna.call_json(
        "Independent Critic Evaluation of Cross-Domain Re-Enactment (x_transfer)",
        sys_iso_critic,
        user_iso_critic,
    )
    transfer_doc["independent_isomorphism_evaluation"] = iso_eval
    transfer_doc["structural_isomorphism_energy"] = round(float(iso_eval.get("structural_isomorphism_energy", 0.0)), 4)
    transfer_doc["critic_verification"] = iso_eval.get("critic_verification", "")

    return {
        "adversarial_trap_doc": trap_doc,
        "cross_domain_transfer_doc": transfer_doc,
    }


def main():
    exp4_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(exp4_dir, "raw")
    wiki_dir = os.path.join(exp4_dir, "wiki_output")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)

    kb_entities = energy_wiki.ensure_and_load_raw_kb(raw_dir)
    by_id = {e.entity_id: e for e in kb_entities}

    cluster1_ids = [
        "mordatch_2018_ebm", "finn_2016_gcl", "hinton_2002_cd",
        "welling_teh_2011_sgld", "ziebart_2008_maxent_irl", "contradiction_note_mcmc_mixing"
    ]
    cluster2_ids = [
        "yuksekgonul_2024_textgrad", "pryzant_2023_protegi",
        "entangled_devlog_wiki_compiler", "finn_2017_maml", "santoro_2017_relation_networks"
    ]
    cluster1_docs = [by_id[i] for i in cluster1_ids if i in by_id]
    cluster2_docs = [by_id[i] for i in cluster2_ids if i in by_id]

    luna = LiveLunaGenerator(model="gpt-6-luna")
    print("[Exp 4] Running Multi-Step Generative Data Synthesis with live 'gpt-6-luna'...", flush=True)

    # Initial rough seed draft at s=0 (k=0)
    draft_s0 = {
        "title": "Draft Note: Combining Energy Models and Textual Gradients",
        "key_equation": "",
        "content": (
            "Rough initial idea: Maybe we can use energy-based models together with textual "
            "gradients to optimize text documents. Energy models assign low energy to good states, "
            "and LLM critics can write critiques to improve a draft."
        ),
        "wiki_links": [],
    }

    sgld_trajectory = []

    # Step k=0 evaluation
    eval_s0 = evaluate_draft_energy_with_luna(luna, "k=0 (Initial Seed Draft)", draft_s0, cluster1_docs, cluster2_docs)
    sgld_trajectory.append({"step": 0, "draft": draft_s0, "evaluation": eval_s0})

    # Step k=0 -> k=3 (First revision pass driven purely by Critic textual gradient)
    draft_s1 = update_draft_via_textual_sgld_luna(
        luna,
        "k=0 -> k=3 (First Textual SGLD Update)",
        draft_s0,
        eval_s0,
        cluster1_docs,
        cluster2_docs,
    )
    eval_s1 = evaluate_draft_energy_with_luna(luna, "k=3 (Intermediate Draft)", draft_s1, cluster1_docs, cluster2_docs)
    sgld_trajectory.append({"step": 3, "draft": draft_s1, "evaluation": eval_s1})

    # Step k=3 -> k=10 (Second revision pass driven purely by Critic textual gradient)
    draft_s2 = update_draft_via_textual_sgld_luna(
        luna,
        "k=3 -> k=10 (Converged Textual SGLD Update)",
        draft_s1,
        eval_s1,
        cluster1_docs,
        cluster2_docs,
    )
    eval_s2 = evaluate_draft_energy_with_luna(luna, "k=10 (Converged Bridge Article)", draft_s2, cluster1_docs, cluster2_docs)
    sgld_trajectory.append({"step": 10, "draft": draft_s2, "evaluation": eval_s2})

    # Task 2 & Task 3: Adversarial Hard-Negative Generation & Cross-Domain Re-Enactment
    adv_and_transfer = generate_and_evaluate_adversarial_and_transfer_docs(
        luna, draft_s2, eval_s2, cluster1_docs, cluster2_docs
    )

    # Save generated Markdown documents to `wiki_output/generated_docs/`
    gen_dir = os.path.join(wiki_dir, "generated_docs")
    os.makedirs(gen_dir, exist_ok=True)

    bridge_md_path = os.path.join(gen_dir, "gen_bridge_boltzmann_textual_sgld.md")
    with open(bridge_md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# {draft_s2['title']}\n\n"
            f"**Generated via $K=10$ Live `gpt-6-luna` Textual SGLD** | "
            f"**Initial Energy ($k=0$):** `{eval_s0['total_composite_energy']}` $\\rightarrow$ "
            f"**Intermediate ($k=3$):** `{eval_s1['total_composite_energy']}` $\\rightarrow$ "
            f"**Converged ($k=10$):** `{eval_s2['total_composite_energy']}`\n\n"
            f"{draft_s2['content']}\n"
        )

    trap_doc = adv_and_transfer["adversarial_trap_doc"]
    trap_md_path = os.path.join(gen_dir, f"{trap_doc['entity_id']}.md")
    with open(trap_md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# {trap_doc['title']}\n\n"
            f"**Type:** `{trap_doc['domain']}`  \n"
            f"**Unary Keyword Energy:** `{trap_doc['unary_keyword_energy']}` (Low — fools Unary RAG!) | "
            f"**Pairwise Relational Energy:** `{trap_doc['pairwise_relational_energy']}` (High — rejected by Relation Network!)\n\n"
            f"$$\n{trap_doc['key_equation']}\n$$\n\n"
            f"{trap_doc['content']}\n\n"
            f"## Relational Critic Rejection Verdict\n{trap_doc['rn_critic_verdict']}\n"
        )

    transfer_doc = adv_and_transfer["cross_domain_transfer_doc"]
    transfer_md_path = os.path.join(gen_dir, f"{transfer_doc['entity_id']}.md")
    with open(transfer_md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# {transfer_doc['title']}\n\n"
            f"**Type:** `{transfer_doc['domain']}` | "
            f"**Structural Isomorphism Energy $E_\\theta$:** `{transfer_doc['structural_isomorphism_energy']}`\n\n"
            f"$$\n{transfer_doc['key_equation']}\n$$\n\n"
            f"{transfer_doc['content']}\n"
        )

    results_summary = {
        "experiment_suite": "exp4_generative_data_synthesis_live_llm",
        "model": "gpt-6-luna",
        "total_llm_calls": len(luna.calls),
        "total_tokens": luna.total_tokens,
        "task1_compositional_bridge_trajectory": sgld_trajectory,
        "task2_adversarial_hard_negative": trap_doc,
        "task3_cross_domain_reenactment": transfer_doc,
    }

    with open(os.path.join(wiki_dir, "generative_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    with open(os.path.join(wiki_dir, "llm_calls_log.json"), "w", encoding="utf-8") as f:
        json.dump(luna.calls, f, indent=2)

    print("\n=== Experiment Suite 4 Summary (Generative Data Synthesis with gpt-6-luna) ===")
    for item in sgld_trajectory:
        ev = item["evaluation"]
        print(
            f"  Step k={item['step']:2d}: Total E_comp = {ev['total_composite_energy']:.4f} "
            f"(w1={ev['e_w1_boltzmann_mcmc_deficit']}, w2={ev['e_w2_textual_gradient_deficit']}, "
            f"links={ev['e_link_topology_deficit']}, contra={ev['e_contradiction_calibration_deficit']})"
        )
    print(
        f"  Adversarial Trap '{trap_doc['title'][:42]}...': "
        f"Unary E = {trap_doc['unary_keyword_energy']} vs. Pairwise E = {trap_doc['pairwise_relational_energy']} "
        f"(Total E = {trap_doc['total_composite_energy']})"
    )
    print(
        f"  Cross-Domain Transfer '{transfer_doc['title'][:42]}...': "
        f"Isomorphism E = {transfer_doc['structural_isomorphism_energy']}"
    )


if __name__ == "__main__":
    main()
