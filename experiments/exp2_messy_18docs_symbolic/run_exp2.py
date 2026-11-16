"""
Runner Script for Experiment Suite 2: Messy 18-Doc KB (Unsupervised Contextual Critic)
======================================================================================
Executes K=10 Textual SGLD concept inference, attention identification, polysemous trap
rejection, contradiction reconciliation, wiki compilation, and Experiments 1-6 on the
18 messy multi-topic documents in `raw/`, exporting `wiki_output/` and `wiki_viewer.html`.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import build_viewer
import energy_wiki
import run_experiments


def main():
    random.seed(42)
    exp2_dir = os.path.dirname(os.path.abspath(__file__))
    bundle = energy_wiki.run_full_pipeline_and_export()

    kb = energy_wiki.build_sample_kb()
    state = energy_wiki.EventState(x0=kb, x1=copy.deepcopy(kb))
    gold_ebm_ids = {
        "mordatch_2018_ebm", "welling_teh_2011_sgld",
        "finn_2016_gcl", "hinton_2002_cd", "ziebart_2008_maxent_irl",
        "contradiction_note_mcmc_mixing"
    }
    ebm_concept = energy_wiki.ConceptCode(
        concept_id="concept_ebm_sampling",
        name="Boltzmann Energy, Partition Function & Langevin Sampling",
        w_a="Identify KB notes grounded in boltzmann, energy, partition_function, sampling, langevin.",
        w_x="Synthesize and cross-link KB notes around boltzmann, energy, partition_function, sampling, langevin.",
        keywords=["boltzmann", "energy", "partition_function", "sampling", "langevin"],
    )

    bundle["experiments"] = {
        "exp1_iterative_sgld": run_experiments.run_exp1_iterative_sgld(state, ebm_concept, gold_ebm_ids),
        "exp2_relational_vs_unary": run_experiments.run_exp2_relational_vs_unary(state, ebm_concept, gold_ebm_ids),
        "exp3_context_transfer": run_experiments.run_exp3_context_transfer(state, ebm_concept, gold_ebm_ids),
        "exp4_code_interchangeability": run_experiments.run_exp4_code_interchangeability(state, ebm_concept, gold_ebm_ids),
        "exp5_kl_objective_ablation": run_experiments.run_exp5_kl_objective_ablation(state, ebm_concept, gold_ebm_ids),
        "exp6_messy_trap_trajectories": run_experiments.run_exp6_messy_trap_trajectories(state, ebm_concept),
    }

    out_path = os.path.join(exp2_dir, "wiki_output", "wiki_bundle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    compact = build_viewer.build_compact_payload(bundle)
    single_html = build_viewer.HTML_TEMPLATE.replace("__SUITES_JSON_PAYLOAD__", json.dumps([compact]))
    viewer_path = os.path.join(exp2_dir, "wiki_viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(single_html)

    print(f"[OK] Saved Suite 2 bundle & viewer to: {exp2_dir}")


if __name__ == "__main__":
    main()
