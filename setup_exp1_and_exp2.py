"""
Sets up and runs:
  - experiments/exp1_clean_10papers_symbolic/
  - experiments/exp2_messy_18docs_symbolic/
keeping both self-contained with their own `raw/`, `wiki_output/`, `wiki_bundle.json`,
`wiki_viewer.html`, and `RESULTS.md`.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil

import build_viewer
import energy_wiki
import run_experiments


# Original 10 clean papers for Exp 1
CLEAN_10_PAPERS = [
    p for p in energy_wiki.MESSY_KB_DATA
    if p["source_type"] == "research_note"
]


def setup_exp1_clean(root_dir: str):
    exp1_dir = os.path.join(root_dir, "experiments", "exp1_clean_10papers_symbolic")
    raw_dir = os.path.join(exp1_dir, "raw")
    wiki_dir = os.path.join(exp1_dir, "wiki_output")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)

    # Temporarily swap MESSY_KB_DATA -> CLEAN_10_PAPERS
    orig_data = energy_wiki.MESSY_KB_DATA
    energy_wiki.MESSY_KB_DATA = CLEAN_10_PAPERS
    try:
        random.seed(42)
        kb_entities = energy_wiki.ensure_and_load_raw_kb(raw_dir)
        base_state = energy_wiki.EventState(x0=kb_entities, x1=copy.deepcopy(kb_entities))

        model = energy_wiki.RelationalWikiEnergyModel()
        sgld = energy_wiki.TextualSGLDOptimizer(model, k_steps=10, alpha=1.25, noise_scale=0.03)

        demo_c1 = energy_wiki.AttentionMask(
            logits={
                e.entity_id: (
                    3.0 if e.entity_id in {
                        "mordatch_2018_ebm", "welling_teh_2011_sgld",
                        "finn_2016_gcl", "hinton_2002_cd", "ziebart_2008_maxent_irl"
                    } else -3.0
                )
                for e in kb_entities
            }
        )
        c1, w1_trace, w1_grads = sgld.infer_concept_code(
            [(base_state, demo_c1)],
            initial_name="Boltzmann Energy & Partition Sampling",
            concept_id="concept_ebm_sampling",
        )
        attn_c1, a1_trace, a1_hist = sgld.optimize_attention_with_history(base_state, c1, stochastic=True)
        comp_c1, x1_trace, x1_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c1, c1)
        eval_c1 = model.evaluate(comp_c1, attn_c1, c1, mode="generation")

        demo_c2 = energy_wiki.AttentionMask(
            logits={
                e.entity_id: (
                    3.0 if e.entity_id in {
                        "mordatch_2018_ebm", "finn_2017_maml",
                        "grant_2017_meta_concepts", "yuksekgonul_2024_textgrad", "pryzant_2023_protegi"
                    } else -3.0
                )
                for e in kb_entities
            }
        )
        c2, w2_trace, w2_grads = sgld.infer_concept_code(
            [(base_state, demo_c2)],
            initial_name="Execution-Time Inner-Loop & Textual Gradients",
            concept_id="concept_inner_loop_textgrad",
        )
        attn_c2, a2_trace, a2_hist = sgld.optimize_attention_with_history(base_state, c2, stochastic=True)
        comp_c2, x2_trace, x2_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c2, c2)
        eval_c2 = model.evaluate(comp_c2, attn_c2, c2, mode="generation")

        c3 = energy_wiki.ConceptCode(
            concept_id="concept_relational_reasoning",
            name="Pairwise Relational & Compositional Architectures",
            w_a="Identify papers modeling pairwise entity interactions g_theta(x_i, x_j, w) gated by attention.",
            w_x="Synthesize pairwise relation networks and entity-gated energy models via [[wiki-links]].",
            keywords=["relational", "pairwise", "attention", "compositional"],
        )
        attn_c3, a3_trace, a3_hist = sgld.optimize_attention_with_history(base_state, c3, stochastic=True)
        comp_c3, x3_trace, x3_hist = sgld.optimize_wiki_state_with_history(base_state, attn_c3, c3)
        eval_c3 = model.evaluate(comp_c3, attn_c3, c3, mode="generation")

        comp_attn_12, comp_trace_12 = sgld.compose_concepts(base_state, [c1, c2])
        c_comp = energy_wiki.ConceptCode(
            concept_id="concept_composite_ebm_meta",
            name="Composite: Energy Sampling + Inner-Loop Meta-Adaptation",
            w_a=f"Composite Sum E(w_1) + E(w_2): ({c1.name}) + ({c2.name})",
            w_x="Bridge Boltzmann partition sampling with inner-loop meta-learning and textual gradient updates.",
            keywords=list(dict.fromkeys(c1.keywords + c2.keywords)),
        )
        comp_state_12, comp_x_trace, comp_x_hist = sgld.optimize_wiki_state_with_history(
            base_state, comp_attn_12, c_comp
        )
        eval_comp = model.evaluate(comp_state_12, comp_attn_12, c_comp, mode="generation")

        tuples = [
            (c1, comp_c1, attn_c1),
            (c2, comp_c2, attn_c2),
            (c3, comp_c3, attn_c3),
            (c_comp, comp_state_12, comp_attn_12),
        ]
        energy_wiki.export_markdown_wiki(wiki_dir, tuples)
        meta_metrics = energy_wiki.compute_meta_losses(
            model, sgld, [(comp_c1, attn_c1)], [(comp_c1, attn_c1), (comp_c2, attn_c2)], c1.name
        )

        with open(os.path.join(wiki_dir, "index.md"), "r", encoding="utf-8") as f:
            index_md = f.read()

        merged: dict = {}
        for _, st, _ in tuples:
            for ent in st.x1:
                if ent.entity_id not in merged:
                    merged[ent.entity_id] = copy.deepcopy(ent)
                else:
                    ex = merged[ent.entity_id]
                    for lk in ent.wiki_links:
                        if lk not in ex.wiki_links:
                            ex.wiki_links.append(lk)
                    for line in ent.content.splitlines():
                        if line.strip() and line not in ex.content:
                            ex.content += "\n" + line

        gold_ids = {
            "mordatch_2018_ebm", "welling_teh_2011_sgld",
            "finn_2016_gcl", "hinton_2002_cd", "ziebart_2008_maxent_irl"
        }
        exp_metrics = {
            "exp1_iterative_sgld": run_experiments.run_exp1_iterative_sgld(base_state, c1, gold_ids),
            "exp2_relational_vs_unary": run_experiments.run_exp2_relational_vs_unary(base_state, c1, gold_ids),
            "exp3_context_transfer": run_experiments.run_exp3_context_transfer(base_state, c1, gold_ids),
            "exp4_code_interchangeability": run_experiments.run_exp4_code_interchangeability(base_state, c1, gold_ids),
            "exp5_kl_objective_ablation": run_experiments.run_exp5_kl_objective_ablation(base_state, c1, gold_ids),
        }

        from dataclasses import asdict
        bundle = {
            "experiment_suite": "exp1_clean_10papers_symbolic",
            "title": "Experiment Suite 1: Clean 10-Paper KB (Symbolic Critic)",
            "raw_entities": [asdict(e) for e in kb_entities],
            "compiled_entities": [asdict(e) for e in merged.values()],
            "index_md": index_md,
            "meta_losses": meta_metrics,
            "concepts": [
                {
                    "concept": asdict(c_obj),
                    "w_energy_trace": w_tr,
                    "w_grad_history": w_gr,
                    "a_energy_trace": a_tr,
                    "a_step_history": a_hi,
                    "x_energy_trace": x_tr,
                    "x_edit_history": x_hi,
                    "final_gates": at_obj.gates_dict(),
                    "concept_page": asdict(st_obj.concept_page) if st_obj.concept_page else None,
                    "pairwise_relations": [asdict(r) for r in ev_obj.pairwise_relations],
                }
                for (c_obj, st_obj, at_obj), w_tr, w_gr, a_tr, a_hi, x_tr, x_hi, ev_obj in [
                    ((c1, comp_c1, attn_c1), w1_trace, w1_grads, a1_trace, a1_hist, x1_trace, x1_hist, eval_c1),
                    ((c2, comp_c2, attn_c2), w2_trace, w2_grads, a2_trace, a2_hist, x2_trace, x2_hist, eval_c2),
                    ((c3, comp_c3, attn_c3), [0.0], ["Seed"], a3_trace, a3_hist, x3_trace, x3_hist, eval_c3),
                    ((c_comp, comp_state_12, comp_attn_12), comp_trace_12, ["Sum"], comp_trace_12, a1_hist, comp_x_trace, comp_x_hist, eval_comp),
                ]
            ],
            "experiments": exp_metrics,
        }

        bundle_path = os.path.join(wiki_dir, "wiki_bundle.json")
        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)

        compact = build_viewer.build_compact_payload(bundle)
        html_out = build_viewer.HTML_TEMPLATE.replace("__WIKI_JSON_PAYLOAD__", json.dumps(compact))
        with open(os.path.join(exp1_dir, "wiki_viewer.html"), "w", encoding="utf-8") as f:
            f.write(html_out)

        print(f"[OK] Built Exp Suite 1 (Clean 10-Paper Symbolic) in: {exp1_dir}")
    finally:
        energy_wiki.MESSY_KB_DATA = orig_data


def setup_exp2_messy(root_dir: str):
    exp2_dir = os.path.join(root_dir, "experiments", "exp2_messy_18docs_symbolic")
    raw_dir = os.path.join(exp2_dir, "raw")
    wiki_dir = os.path.join(exp2_dir, "wiki_output")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)

    # Copy current messy raw/ and wiki_output/ into exp2_dir
    src_raw = os.path.join(root_dir, "raw")
    src_wiki = os.path.join(root_dir, "wiki_output")
    if os.path.exists(src_raw):
        shutil.copytree(src_raw, raw_dir, dirs_exist_ok=True)
    if os.path.exists(src_wiki):
        shutil.copytree(src_wiki, wiki_dir, dirs_exist_ok=True)
    if os.path.exists(os.path.join(root_dir, "wiki_viewer.html")):
        shutil.copy2(os.path.join(root_dir, "wiki_viewer.html"), os.path.join(exp2_dir, "wiki_viewer.html"))

    print(f"[OK] Built Exp Suite 2 (Messy 18-Doc Unsupervised Symbolic) in: {exp2_dir}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    setup_exp1_clean(root)
    setup_exp2_messy(root)
