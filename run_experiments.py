"""
Experimental Suite for Energy-Based LLM Wiki with Textual Gradients (Messy KB)
==============================================================================
Interpretation: these are configured symbolic demonstrations. Historical result
keys mentioning training or KL retain compatibility with saved viewers: weights
are manually chosen and the KL-labeled conditions change step size and noise.
The unary generation condition also disables link and contradiction directives.
Runs all experiments mirroring Section 5 of Mordatch (2018) on the realistic
18-document messy, multi-topic Knowledge Base in `raw/` (containing unstructured
research notes, entangled dev logs, contradictory notes, polysemous keyword traps,
and out-of-domain ops clutter) and exports `wiki_output/wiki_bundle.json`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from typing import Dict, List

from energy_wiki import (
    AttentionMask,
    ConceptCode,
    EventState,
    KBEntity,
    MetaSchemaParams,
    RelationalWikiEnergyModel,
    TextualSGLDOptimizer,
    build_sample_kb,
    run_full_pipeline_and_export,
)


def attention_bce(pred: AttentionMask, gold_active_ids: set[str], entities: List[KBEntity]) -> float:
    eps = 1e-6
    total = 0.0
    for ent in entities:
        p = min(1.0 - eps, max(eps, pred.gate(ent.entity_id)))
        y = 1.0 if ent.entity_id in gold_active_ids else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / max(1, len(entities))


def false_positive_trap_rate(pred: AttentionMask, entities: List[KBEntity]) -> Tuple[int, float]:
    """Counts how many polysemous_trap or ops_clutter docs are falsely attended (sigma(a_i) >= 0.5)."""
    traps = [e for e in entities if e.source_type in ("polysemous_trap", "ops_clutter")]
    fp_count = sum(1 for e in traps if pred.gate(e.entity_id) >= 0.5)
    avg_trap_gate = sum(pred.gate(e.entity_id) for e in traps) / max(1, len(traps))
    return fp_count, round(avg_trap_gate, 4)


def generation_error(state: EventState, gold_active_ids: set[str]) -> float:
    active_entities = [e for e in state.x1 if e.entity_id in gold_active_ids]
    if len(active_entities) < 2:
        return 0.0

    total_pairs = 0
    missing_links = 0.0
    for i, ei in enumerate(active_entities):
        for j, ej in enumerate(active_entities):
            if i >= j:
                continue
            total_pairs += 1
            ij = ej.title in ei.wiki_links
            ji = ei.title in ej.wiki_links
            if not (ij and ji):
                missing_links += 0.5 if (ij or ji) else 1.0

    link_err = missing_links / max(1, total_pairs)

    if state.concept_page is None:
        synth_err = 1.0
    else:
        covered = sum(1 for e in active_entities if e.title in state.concept_page.wiki_links)
        synth_err = 1.0 - (covered / len(active_entities))

    return 0.65 * link_err + 0.35 * synth_err


class UnaryWikiEnergyModel(RelationalWikiEnergyModel):
    """Ablation of Eq. (4) from Section 5.3 ('No RN' baseline — unary pointwise evaluation)."""

    def evaluate(
        self,
        state: EventState,
        attention: AttentionMask,
        concept: ConceptCode,
        mode: str = "identification",
        use_relational: bool = False,
    ):
        ev = super().evaluate(state, attention, concept, mode=mode, use_relational=False)
        ev.pairwise_relations = []
        ev.grad_x = [
            d for d in ev.grad_x
            if not d.startswith(("LINK_DEFICIT:", "RESOLVE_CONTRADICTION:"))
        ]
        return ev


def run_exp1_iterative_sgld(base_state: EventState, concept: ConceptCode, gold_ids: set[str]) -> Dict[str, Dict]:
    model = RelationalWikiEnergyModel()
    results: Dict[str, Dict] = {}
    gold_attn = AttentionMask(logits={e.entity_id: (3.0 if e.entity_id in gold_ids else -3.0) for e in base_state.x0})

    for k in [1, 2, 5, 10, 20]:
        opt = TextualSGLDOptimizer(model, k_steps=k, alpha=0.85, noise_scale=0.0)
        pred_attn, a_trace = opt.optimize_attention(base_state, concept, stochastic=False, use_relational=True)
        compiled_state, x_trace = opt.optimize_wiki_state(base_state, gold_attn, concept)
        fp_traps, avg_trap_gate = false_positive_trap_rate(pred_attn, base_state.x0)
        results[f"K={k}"] = {
            "k": k,
            "attn_bce": round(attention_bce(pred_attn, gold_ids, base_state.x0), 4),
            "false_positive_traps": fp_traps,
            "avg_trap_gate": avg_trap_gate,
            "gen_error": round(generation_error(compiled_state, gold_ids), 4),
            "final_energy_a": round(a_trace[-1], 4),
            "final_energy_x": round(x_trace[-1], 4),
        }
    return results


def run_exp2_relational_vs_unary(base_state: EventState, concept: ConceptCode, gold_ids: set[str]) -> Dict[str, Dict]:
    gold_attn = AttentionMask(logits={e.entity_id: (3.0 if e.entity_id in gold_ids else -3.0) for e in base_state.x0})

    rn_model = RelationalWikiEnergyModel()
    rn_opt = TextualSGLDOptimizer(rn_model, k_steps=10, alpha=1.2, noise_scale=0.0)
    rn_attn, _ = rn_opt.optimize_attention(base_state, concept, stochastic=False, use_relational=True)
    rn_state, rn_trace = rn_opt.optimize_wiki_state(base_state, gold_attn, concept)
    rn_fp, rn_trap_gate = false_positive_trap_rate(rn_attn, base_state.x0)

    unary_model = UnaryWikiEnergyModel()
    unary_opt = TextualSGLDOptimizer(unary_model, k_steps=10, alpha=1.2, noise_scale=0.0)
    unary_attn, _ = unary_opt.optimize_attention(base_state, concept, stochastic=False, use_relational=False)
    unary_state, unary_trace = unary_opt.optimize_wiki_state(base_state, gold_attn, concept)
    unary_fp, unary_trap_gate = false_positive_trap_rate(unary_attn, base_state.x0)

    return {
        "Relation_Network": {
            "attn_bce": round(attention_bce(rn_attn, gold_ids, base_state.x0), 4),
            "false_positive_traps": rn_fp,
            "avg_trap_gate": rn_trap_gate,
            "gen_error": round(generation_error(rn_state, gold_ids), 4),
            "final_energy": round(rn_trace[-1], 4),
            "links_created": sum(len(e.wiki_links) for e in rn_state.x1),
        },
        "No_RN_Unary": {
            "attn_bce": round(attention_bce(unary_attn, gold_ids, base_state.x0), 4),
            "false_positive_traps": unary_fp,
            "avg_trap_gate": unary_trap_gate,
            "gen_error": round(generation_error(unary_state, gold_ids), 4),
            "final_energy": round(unary_trace[-1], 4),
            "links_created": sum(len(e.wiki_links) for e in unary_state.x1),
        },
    }


def run_exp3_context_transfer(base_state: EventState, concept: ConceptCode, gold_ids: set[str]) -> Dict[str, Dict[str, float]]:
    conditions = {
        "Untrained_Critic": MetaSchemaParams(cohesion_weight=0.05, coverage_weight=0.05, link_topology_weight=0.05, synthesis_depth_weight=0.05),
        "Trained_on_Generation": MetaSchemaParams(cohesion_weight=0.8, coverage_weight=0.8, link_topology_weight=1.4, synthesis_depth_weight=1.2),
        "Trained_on_Identification": MetaSchemaParams(cohesion_weight=1.4, coverage_weight=1.3, link_topology_weight=0.8, synthesis_depth_weight=0.8),
        "Trained_on_Both": MetaSchemaParams(cohesion_weight=1.4, coverage_weight=1.3, link_topology_weight=1.4, synthesis_depth_weight=1.2),
    }
    gold_attn = AttentionMask(logits={e.entity_id: (3.0 if e.entity_id in gold_ids else -3.0) for e in base_state.x0})
    results: Dict[str, Dict[str, float]] = {}

    for name, theta in conditions.items():
        model = RelationalWikiEnergyModel(theta=theta)
        alpha = 0.05 if name == "Untrained_Critic" else 1.25
        opt = TextualSGLDOptimizer(model, k_steps=10, alpha=alpha, noise_scale=0.0)
        pred_attn, _ = opt.optimize_attention(base_state, concept, stochastic=False)
        compiled_state, _ = opt.optimize_wiki_state(base_state, gold_attn, concept)
        results[name] = {
            "Generation_Error": round(generation_error(compiled_state, gold_ids), 3),
            "Attention_Error": round(attention_bce(pred_attn, gold_ids, base_state.x0), 3),
        }
    return results


def run_exp4_code_interchangeability(base_state: EventState, concept: ConceptCode, gold_ids: set[str]) -> Dict[str, float]:
    model = RelationalWikiEnergyModel()
    opt = TextualSGLDOptimizer(model, k_steps=10, alpha=1.25, noise_scale=0.0)
    gold_attn = AttentionMask(logits={e.entity_id: (3.0 if e.entity_id in gold_ids else -3.0) for e in base_state.x0})

    normal_state, _ = opt.optimize_wiki_state(base_state, gold_attn, concept)
    swapped_concept = copy.deepcopy(concept)
    swapped_concept.w_x = concept.w_a
    swapped_state, _ = opt.optimize_wiki_state(base_state, gold_attn, swapped_concept)

    return {
        "normal_gen_error_using_wx": round(generation_error(normal_state, gold_ids), 4),
        "swapped_gen_error_using_wa": round(generation_error(swapped_state, gold_ids), 4),
    }


def run_exp5_kl_objective_ablation(base_state: EventState, concept: ConceptCode, gold_ids: set[str]) -> Dict[str, Dict[str, float]]:
    model = RelationalWikiEnergyModel()
    gold_attn = AttentionMask(logits={e.entity_id: (3.0 if e.entity_id in gold_ids else -3.0) for e in base_state.x0})
    rand_attn = AttentionMask(logits={e.entity_id: random.uniform(-1.0, 1.0) for e in base_state.x0})

    opt_with_kl = TextualSGLDOptimizer(model, k_steps=10, alpha=1.25, noise_scale=0.02)
    gold_compiled, _ = opt_with_kl.optimize_wiki_state(base_state, gold_attn, concept)
    sampled_attn_kl, _ = opt_with_kl.optimize_attention(base_state, concept, stochastic=True)

    opt_no_kl = TextualSGLDOptimizer(model, k_steps=10, alpha=0.08, noise_scale=0.25)
    sampled_attn_no_kl, _ = opt_no_kl.optimize_attention(base_state, concept, stochastic=True)

    return {
        "With_KL_Objective": {
            "gold_energy": round(model.evaluate(gold_compiled, gold_attn, concept, "identification").energy, 4),
            "sampled_energy": round(model.evaluate(base_state, sampled_attn_kl, concept, "identification").energy, 4),
            "random_energy": round(model.evaluate(base_state, rand_attn, concept, "identification").energy, 4),
        },
        "No_KL_Objective": {
            "gold_energy": round(model.evaluate(gold_compiled, gold_attn, concept, "identification").energy, 4),
            "sampled_energy": round(model.evaluate(base_state, sampled_attn_no_kl, concept, "identification").energy, 4),
            "random_energy": round(model.evaluate(base_state, rand_attn, concept, "identification").energy, 4),
        },
    }


def run_exp6_messy_trap_trajectories(base_state: EventState, concept: ConceptCode) -> Dict[str, List[float]]:
    """
    Tracks step-by-step attention gate sigma(a_i^k) for k=0..10 on:
      - A true EBM paper (`welling_teh_2011_sgld`)
      - A contradictory lab note (`contradiction_note_mcmc_mixing`)
      - Two polysemous keyword traps (`trap_hvac_building_energy` and `trap_agronomy_soil_sampling`)
        under Relation Network vs. Unary ('No RN').
    """
    model = RelationalWikiEnergyModel()
    opt = TextualSGLDOptimizer(model, k_steps=10, alpha=1.25, noise_scale=0.0)
    _, _, rn_hist = opt.optimize_attention_with_history(base_state, concept, stochastic=False, use_relational=True)
    _, _, unary_hist = opt.optimize_attention_with_history(base_state, concept, stochastic=False, use_relational=False)

    return {
        "RN_true_sgld_paper": [s["gates"]["welling_teh_2011_sgld"] for s in rn_hist],
        "RN_contradiction_lab_note": [s["gates"]["contradiction_note_mcmc_mixing"] for s in rn_hist],
        "RN_trap_hvac_energy": [s["gates"]["trap_hvac_building_energy"] for s in rn_hist],
        "RN_trap_soil_sampling": [s["gates"]["trap_agronomy_soil_sampling"] for s in rn_hist],
        "Unary_NoRN_trap_hvac_energy": [s["gates"]["trap_hvac_building_energy"] for s in unary_hist],
        "Unary_NoRN_trap_soil_sampling": [s["gates"]["trap_agronomy_soil_sampling"] for s in unary_hist],
    }


if __name__ == "__main__":
    random.seed(42)
    bundle = run_full_pipeline_and_export()

    kb = build_sample_kb()
    state = EventState(x0=kb, x1=copy.deepcopy(kb))
    gold_ebm_ids = {
        "mordatch_2018_ebm", "welling_teh_2011_sgld",
        "finn_2016_gcl", "hinton_2002_cd", "ziebart_2008_maxent_irl",
        "contradiction_note_mcmc_mixing"
    }

    ebm_concept = ConceptCode(
        concept_id="concept_ebm_sampling",
        name="Boltzmann Energy, Partition Function & Langevin Sampling",
        w_a="Identify KB notes grounded in boltzmann, energy, partition_function, sampling, langevin.",
        w_x="Synthesize and cross-link KB notes around boltzmann, energy, partition_function, sampling, langevin.",
        keywords=["boltzmann", "energy", "partition_function", "sampling", "langevin"],
    )

    exp1 = run_exp1_iterative_sgld(state, ebm_concept, gold_ebm_ids)
    exp2 = run_exp2_relational_vs_unary(state, ebm_concept, gold_ebm_ids)
    exp3 = run_exp3_context_transfer(state, ebm_concept, gold_ebm_ids)
    exp4 = run_exp4_code_interchangeability(state, ebm_concept, gold_ebm_ids)
    exp5 = run_exp5_kl_objective_ablation(state, ebm_concept, gold_ebm_ids)
    exp6 = run_exp6_messy_trap_trajectories(state, ebm_concept)

    print("\n=== Experiment 1: Multi-Step Textual SGLD on Messy 18-Doc KB (K=1..20) ===")
    for k_label, metrics in exp1.items():
        print(f"  {k_label:5s} -> {metrics}")

    print("\n=== Experiment 2: Relation Network vs. Unary ('No RN') on Messy KB ===")
    for arch, metrics in exp2.items():
        print(f"  {arch:18s} -> {metrics}")

    print("\n=== Experiment 3: Context Transfer (Identification <-> Generation) ===")
    for cond, metrics in exp3.items():
        print(f"  {cond:26s} -> {metrics}")

    print("\n=== Experiment 4: Concept Code Interchangeability (w_a <-> w_x) ===")
    print(" ", exp4)

    print("\n=== Experiment 5: Training With vs. Without L^KL Objective ===")
    for mode, metrics in exp5.items():
        print(f"  {mode:18s} -> {metrics}")

    print("\n=== Experiment 6: Polysemous Trap Rejection Trajectory (k=0..10) ===")
    for k_name, trace in exp6.items():
        print(f"  {k_name:32s}: {trace[0]:.3f} (k=0) -> {trace[2]:.3f} (k=2) -> {trace[-1]:.3f} (k=10)")

    exp_summary = {
        "exp1_iterative_sgld": exp1,
        "exp2_relational_vs_unary": exp2,
        "exp3_context_transfer": exp3,
        "exp4_code_interchangeability": exp4,
        "exp5_kl_objective_ablation": exp5,
        "exp6_messy_trap_trajectories": exp6,
    }
    bundle["experiments"] = exp_summary

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "experiments", "exp2_messy_18docs_symbolic", "wiki_output", "wiki_bundle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"\n[OK] Saved full messy-KB wiki + experiment telemetry to: {out_path}")
