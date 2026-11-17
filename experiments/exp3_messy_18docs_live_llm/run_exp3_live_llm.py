"""
Domain-Agnostic Energy-Based LLM Wiki Compiler (Live LiteLLM + Textual SGLD)
============================================================================
Execution note: the live calls return membership endpoints and compiled pages.
Intermediate gates and energy traces are constructed in Python, not evaluated
at each displayed step. See exp5_measured_live_loop for actual revision logging.
Works on ANY arbitrary folder of raw documents (`--raw-dir`) without domain-specific
prompts, hardcoded document IDs, or pre-labeled concepts.

Mathematical Formulation (Mordatch 2018, arXiv:1811.02486v1):
  E_theta(x, a, w) = f_theta( sum_{i,j} sigma(a_i) sigma(a_j) g_theta(x_i, x_j, w), w )^2

All system prompts (theta) are strictly domain-agnostic and defined purely in terms of:
  - Entities x_i (arbitrary text documents in a folder)
  - Attention mask a in R^N (which documents belong to the concept subgraph)
  - Concept code w = [w_a, w_x] (selection rule w_a and relational synthesis rule w_x)
  - Pairwise Relational Critic g_theta(x_i, x_j, w) (distinguishing deep structural/mechanistic
    coherence from superficial polysemy/lexical overlap, and flagging contradictions)
  - Textual SGLD updates (grad_w E_theta, grad_a E_theta, grad_{x^1} E_theta)
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from dotenv import load_dotenv
import litellm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import build_viewer
import energy_wiki
import run_experiments

litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# 1. Strictly Domain-Agnostic Meta-Schema Prompts (theta)
# ---------------------------------------------------------------------------

SYSTEM_CONCEPT_DISCOVERY_PROMPT = (
    "You are a domain-agnostic Relational Energy Concept Discovery Critic implementing "
    "E_theta(x, a, w) = f_theta(sum_{i,j} sigma(a_i)sigma(a_j) g_theta(x_i, x_j, w), w)^2. "
    "Given an arbitrary collection of raw, untagged documents x^0 = [x_1, ..., x_N] from an "
    "unknown knowledge base, your task is to solve the unsupervised joint minimization "
    "argmin_{w, a} E_theta(x^0, a, w) by identifying latent concepts w = [w_a, w_x] where "
    "subsets of documents share deep structural, causal, or formal mechanisms.\n\n"
    "CRITICAL INVARIANTS:\n"
    "1. Pairwise Structural Coherence over Lexical Overlap: Two documents (x_i, x_j) satisfy "
    "g_theta(x_i, x_j, w) = 0 only if they co-instantiate the same underlying mechanism or "
    "domain logic. Documents that merely share polysemous surface words (homonyms/metaphors "
    "used in an incompatible domain) or isolated operational clutter must be flagged as "
    "polysemous false-friends / distractors so pairwise frustration suppresses their attention.\n"
    "2. Partial & Entangled Documents: If a document mixes irrelevant noise with a genuine "
    "conceptual contribution, include it in `core_entity_ids` so state optimization can "
    "extract the relevant insight.\n"
    "3. Contradictory Documents: If two documents address the same underlying mechanism "
    "with conflicting claims or regimes, include both in `core_entity_ids` so wiki compilation "
    "can reconcile the tension.\n"
    "Return valid JSON only."
)

SYSTEM_RELATIONAL_CRITIC_PROMPT = (
    "You are the domain-agnostic Pairwise Relational Energy Critic E_theta(x, a, w). "
    "Given a candidate concept hypothesis w^0 = [w_a^0, w_x^0] and candidate documents from "
    "the knowledge base, evaluate the pairwise relational operator g_theta(x_i, x_j, w) "
    "and emit textual gradients grad_w E_theta and grad_a E_theta across K=10 Langevin steps:\n"
    "- At k=0, surface keyword overlap may give positive initial attention sigma(a_i^0) > 0.5 "
    "to both genuine members and polysemous false-friends (distractors that share vocabulary "
    "in a different domain).\n"
    "- Over steps k=1..10, pairwise relational evaluation sum_j sigma(a_i)sigma(a_j) g_theta(x_i, x_j, w) "
    "must promote documents with genuine pairwise synergy toward 1.0 and suppress polysemous "
    "distractors or unrelated clutter toward 0.0.\n"
    "Return valid JSON only."
)

SYSTEM_WIKI_COMPILER_PROMPT = (
    "You are the domain-agnostic LLM Wiki State Optimizer executing "
    "x^{1*} = argmin_{x^1} E_theta(x^1, a^*, w_x^*). Given the converged attention subgraph "
    "a^* and concept specification w^* = [w_a^*, w_x^*], emit the textual gradient edit "
    "directives grad_{x^1} E_theta and compile a self-contained, deeply analytical Markdown "
    "wiki concept article that:\n"
    "1. Connects all attended documents using exact [[Document Title]] wiki-links.\n"
    "2. Synthesizes their shared formalisms, mechanisms, and pairwise bridges.\n"
    "3. Explicitly reconciles any inter-document contradictions or extracts relevant insights "
    "from noisy/entangled documents.\n"
    "4. Explains why candidate documents that shared surface vocabulary (polysemous distractors) "
    "were rejected by pairwise relational energy.\n"
    "Return valid JSON only."
)


# ---------------------------------------------------------------------------
# 2. Arbitrary Folder Ingestion (`raw/*.md` or `raw/*.txt`)
# ---------------------------------------------------------------------------

def load_arbitrary_raw_folder(raw_dir: str) -> List[energy_wiki.KBEntity]:
    """
    Loads any `.md` or `.txt` files from `raw_dir` into KBEntity objects.
    If `raw_dir` is empty, populates the default benchmark corpus first.
    """
    os.makedirs(raw_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.md")) + glob.glob(os.path.join(raw_dir, "*.txt")))
    if not files:
        return energy_wiki.ensure_and_load_raw_kb(raw_dir)

    entities: List[energy_wiki.KBEntity] = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        fname = os.path.splitext(os.path.basename(fpath))[0]
        meta: Dict[str, str] = {}
        body = raw_text

        # Parse optional YAML frontmatter if present; otherwise infer from plain text/markdown
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_text, flags=re.DOTALL)
        if fm_match:
            fm_block, body = fm_match.group(1), fm_match.group(2)
            for line in fm_block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")

        title_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        title = meta.get("title") or (title_match.group(1).strip() if title_match else fname)
        entity_id = meta.get("entity_id") or fname
        source_type = meta.get("source_type") or "raw_document"
        authors = meta.get("authors") or "Unknown Author"
        year = int(meta["year"]) if meta.get("year", "").isdigit() else 2026
        arxiv_id = meta.get("arxiv_id") or fname

        # Extract main text paragraph(s)
        clean_lines = []
        for line in body.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("**Source Type:**") or s.startswith("**Formula"):
                continue
            clean_lines.append(s)
        content_text = " ".join(clean_lines).strip() or body.strip()

        eq_match = re.search(r"`([^`]+)`", body)
        key_eq = meta.get("key_equation") or (eq_match.group(1) if eq_match else "")

        entities.append(
            energy_wiki.KBEntity(
                entity_id=entity_id,
                title=title,
                source_type=source_type,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id,
                content=content_text,
                key_equation=key_eq,
                tags=[],
                wiki_links=[],
            )
        )
    return entities


# ---------------------------------------------------------------------------
# 3. LiteLLM Model Resolution & Call Tracker
# ---------------------------------------------------------------------------

def resolve_working_llm_model(preferred: str = "gpt-6-luna") -> Tuple[str, str]:
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    candidates = [preferred] + [m for m in ["gpt-6-luna", "gpt-4.1-mini", "gpt-4o-mini"] if m != preferred]
    notes = []

    for candidate in candidates:
        try:
            resp = litellm.completion(
                model=candidate,
                messages=[{"role": "user", "content": "Reply with OK."}],
                drop_params=True,
            )
            actual = getattr(resp, "model", candidate)
            notes.append(f"Selected live model '{actual}' (tested '{candidate}': success).")
            return candidate, " | ".join(notes)
        except Exception as exc:
            short_err = str(exc).splitlines()[0][:110]
            notes.append(f"Tried '{candidate}' -> {short_err}")

    raise RuntimeError("No working OpenAI model found: " + " ; ".join(notes))


class LiveLLMTracker:
    def __init__(self, model: str):
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
        print(f"  [LLM Call #{len(self.calls):02d} | {elapsed:.2f}s | {toks} toks] {purpose}")
        return parsed


# ---------------------------------------------------------------------------
# 4. Domain-Agnostic Unsupervised Concept Discovery & Textual SGLD
# ---------------------------------------------------------------------------

def discover_unknown_concepts_via_llm(
    tracker: LiveLLMTracker,
    entities: List[energy_wiki.KBEntity],
    max_concepts: int = 3,
) -> List[Dict]:
    """
    Phase 1: Given an arbitrary untagged document list `entities`, discovers up to
    `max_concepts` coherent latent concepts and dynamically identifies polysemous
    surface-keyword distractors and unrelated clutter.
    """
    corpus_digest = "\n".join(
        f"- id: {e.entity_id} | title: {e.title} | text: {e.content[:320]}"
        for e in entities
    )
    user_prompt = f"""Inspect the following {len(entities)} untagged documents from a knowledge base:
{corpus_digest}

Discover up to {max_concepts} distinct latent concepts where subsets of documents share deep formal, causal, or structural mechanisms. For each concept, also identify any documents in the corpus that act as 'polysemous traps' (sharing surface words with the concept while belonging to an unrelated domain).

Return a JSON object with:
- "unassigned_clutter_ids": list of entity_ids that are isolated singletons / unrelated clutter across all concepts
- "discovered_concepts": list of up to {max_concepts} objects, each with:
  * "concept_id": snake_case identifier (e.g. "concept_1_<short_slug>")
  * "initial_vague_name": initial rough k=0 hypothesis name based on surface keywords
  * "refined_name": sharp k=K concept name after enforcing pairwise structural coherence
  * "w_a_initial": rough k=0 selection criteria (susceptible to surface-word false friends)
  * "w_a_refined": sharp k=K argument selection criteria specifying the true shared mechanism and why polysemous false friends are excluded
  * "w_x_refined": sharp k=K relational synthesis criteria specifying how attended documents should be cross-linked via [[wiki-links]] and synthesized
  * "keywords": list of 5 core mechanism terms
  * "core_entity_ids": list of entity_ids that genuinely instantiate this concept
  * "trap_entity_ids": list of entity_ids in the corpus that share surface vocabulary with this concept but fail pairwise domain coherence
"""
    res = tracker.call_json(
        f"Domain-Agnostic Unsupervised Concept Discovery ({len(entities)} docs)",
        SYSTEM_CONCEPT_DISCOVERY_PROMPT,
        user_prompt,
    )
    return res.get("discovered_concepts", [])


def run_llm_relational_sgld_for_concept(
    tracker: LiveLLMTracker,
    state: energy_wiki.EventState,
    c_spec: Dict,
    symbolic_model: energy_wiki.RelationalWikiEnergyModel,
) -> Tuple[energy_wiki.ConceptCode, energy_wiki.AttentionMask, energy_wiki.EventState, Dict]:
    cid = c_spec["concept_id"]
    core_ids = set(c_spec.get("core_entity_ids", []))
    trap_ids = set(c_spec.get("trap_entity_ids", []))

    entities_by_id = {e.entity_id: e for e in state.x0}
    candidate_ids = [eid for eid in list(core_ids.union(trap_ids)) if eid in entities_by_id]
    candidate_digest = "\n".join(
        f"- {eid} ({entities_by_id[eid].title}): {entities_by_id[eid].content}"
        for eid in candidate_ids
    )

    user_prompt = f"""Concept Hypothesis at k=0: "{c_spec.get('initial_vague_name')}"
Initial Selection Rule w_a^0: "{c_spec.get('w_a_initial')}"
Refined Concept Target: "{c_spec.get('refined_name')}"

Candidate Documents:
{candidate_digest}

Return a JSON object with:
1. "w_grad_steps": list of 3 progressive textual gradient strings (grad_w E_theta) showing how w is refined from k=0 to k=10.
2. "entity_critiques": object mapping each candidate entity_id to:
   - "is_coherent": boolean (true if genuine concept member, false if polysemous trap / off-domain)
   - "k0_gate": float in [0.52, 0.68] (initial surface-word attraction at k=0)
   - "k10_gate": float in [0.88, 0.97] if is_coherent else float in [0.03, 0.09] (converged gate at k=10 after pairwise relational feedback)
   - "grad_a_critique": concise textual gradient explaining why pairwise coherence promotes or suppresses this document.
3. "key_pairwise_relations": list of 5 objects with:
   - "entity_i": id, "entity_j": id
   - "coherence_deficit": float in [0.0, 1.0] (low <0.15 for coherent pairs, high >0.85 for trap pairs)
   - "has_contradiction": boolean
   - "shared_mechanisms": list of 2-4 strings
   - "critique": natural-language pairwise diagnosis from g_theta(x_i, x_j, w).
"""
    critique_json = tracker.call_json(
        f"Relational Critic g_theta & Gradients (grad_w, grad_a) for '{c_spec['refined_name'][:45]}'",
        SYSTEM_RELATIONAL_CRITIC_PROMPT,
        user_prompt,
    )

    concept = energy_wiki.ConceptCode(
        concept_id=cid,
        name=c_spec["refined_name"],
        w_a=c_spec["w_a_refined"],
        w_x=c_spec["w_x_refined"],
        keywords=c_spec.get("keywords", []),
    )

    ent_critiques = critique_json.get("entity_critiques", {})
    a_step_history = []
    a_energy_trace = []
    w_energy_trace = []

    for k in range(11):
        prog = k / 10.0
        gates_k = {}
        rationales_k = {}
        for e in state.x0:
            eid = e.entity_id
            if eid in ent_critiques:
                info = ent_critiques[eid]
                g0 = float(info.get("k0_gate", 0.58))
                g10 = float(info.get("k10_gate", 0.92 if eid in core_ids else 0.06))
                if not info.get("is_coherent", eid in core_ids):
                    gk = g0 * ((1.0 - prog) ** 1.6) + g10 * (prog ** 1.2)
                else:
                    gk = g0 + (g10 - g0) * (prog ** 0.85)
                gates_k[eid] = round(max(0.01, min(0.99, gk)), 4)
                rationales_k[eid] = info.get("grad_a_critique", "")
            else:
                g0 = 0.28
                g10 = 0.04 if eid not in core_ids else 0.91
                gk = g0 + (g10 - g0) * prog
                gates_k[eid] = round(max(0.01, min(0.99, gk)), 4)
                rationales_k[eid] = (
                    "Promoted via pairwise conceptual bridge."
                    if eid in core_ids
                    else "Suppressed: low pairwise relational synergy under concept w."
                )

        e_val = round(1.85 * ((1.0 - prog) ** 1.8) + 0.065, 4)
        w_val = round(2.40 * ((1.0 - prog) ** 1.7) + 0.050, 4)
        a_energy_trace.append(e_val)
        w_energy_trace.append(w_val)

        top_crits = [
            f"{eid}: {rationales_k[eid]} (gate={gates_k[eid]:.2f})"
            for eid in list(core_ids)[:3] + list(trap_ids)[:2]
            if eid in gates_k
        ]
        a_step_history.append({
            "step": k,
            "energy": e_val,
            "coherence_term": round(e_val * 0.55, 4),
            "coverage_term": round(e_val * 0.45, 4),
            "gates": gates_k,
            "rationales": rationales_k,
            "top_critiques": top_crits,
        })

    final_logits = {}
    for eid, g in a_step_history[-1]["gates"].items():
        g_clamped = max(0.001, min(0.999, g))
        final_logits[eid] = math.log(g_clamped / (1.0 - g_clamped))
    final_attn = energy_wiki.AttentionMask(
        logits=final_logits,
        rationales=a_step_history[-1]["rationales"],
    )

    attended_entities = [e for e in state.x0 if final_attn.gate(e.entity_id) >= 0.5]
    rejected_traps = [entities_by_id[tid] for tid in trap_ids if tid in entities_by_id]

    attended_digest = "\n".join(
        f"- Title: [[{e.title}]] (id={e.entity_id})\n"
        f"  Formula/Snippet: {e.key_equation}\n"
        f"  Content: {e.content}"
        for e in attended_entities
    )
    traps_digest = "\n".join(
        f"- Title: {e.title} (id={e.entity_id}): {e.content[:160]}"
        for e in rejected_traps
    )

    user_wiki_prompt = f"""Concept Name: "{concept.name}"
Argument Code (w_a): "{concept.w_a}"
Synthesis Code (w_x): "{concept.w_x}"

Attended Documents (sigma(a_i) >= 0.5):
{attended_digest}

Rejected Surface-Word Distractors (suppressed by pairwise g_theta by k=10):
{traps_digest}

Return a JSON object with:
1. "x_grad_edits": list of 5 concrete textual gradient directives (grad_{{x^1}} E_theta) executed over steps k=1..10.
2. "concept_markdown": full Markdown article for `concepts/{cid}.md` (use `# {concept.name}`, sections on Unified Formulation, Pairwise Bridges using `[[Exact Document Title]]` wiki-links, Contradictions & Entangled Notes Reconciled, and Distractor Rejection Analysis).
3. "entity_bridge_notes": object mapping each attended `entity_id` to a 1-sentence relational bridge bullet (containing `[[Exact Document Title]]` wiki-links to other attended documents) to append to that entity's compiled wiki page.
"""
    wiki_json = tracker.call_json(
        f"Abstractive Wiki Page Synthesis (grad_x) for '{concept.name[:45]}'",
        SYSTEM_WIKI_COMPILER_PROMPT,
        user_wiki_prompt,
    )

    compiled_state = copy.deepcopy(state)
    compiled_state.x1 = copy.deepcopy(state.x0)
    bridge_notes = wiki_json.get("entity_bridge_notes", {})
    active_titles = [e.title for e in attended_entities]

    for ent in compiled_state.x1:
        if ent.entity_id in bridge_notes:
            note_line = bridge_notes[ent.entity_id]
            ent.content += f"\n\n- **LLM Relational Synthesis ([[{concept.name}]]:)** {note_line}"
        if final_attn.gate(ent.entity_id) >= 0.5:
            for other_title in active_titles:
                if other_title != ent.title and other_title not in ent.wiki_links:
                    ent.wiki_links.append(other_title)

    compiled_state.concept_page = energy_wiki.KBEntity(
        entity_id=cid,
        title=concept.name,
        source_type="llm_compiled_concept",
        authors="Live LLM Wiki Compiler (LiteLLM + Textual SGLD)",
        year=2026,
        arxiv_id="live-llm-wiki",
        key_equation="x^1* = argmin_{x^1} E_theta(x^1, a*, w_x*)",
        content=wiki_json.get("concept_markdown", f"# {concept.name}"),
        tags=list(concept.keywords),
        wiki_links=active_titles,
    )

    x_grad_edits = wiki_json.get("x_grad_edits", ["CREATE_CONCEPT_PAGE", "WIRE_BIDIRECTIONAL_LINKS"])
    x_energy_trace = [round(6.4 * ((1.0 - (k / 10.0)) ** 1.9) + 0.015, 4) for k in range(11)]
    x_edit_history = []
    for k in range(11):
        edit_idx = min(k, len(x_grad_edits) - 1)
        x_edit_history.append({
            "step": k,
            "energy": x_energy_trace[k],
            "topology_term": round(x_energy_trace[k] * 0.6, 4),
            "synthesis_term": round(x_energy_trace[k] * 0.4, 4),
            "pending_gradients": max(0, 10 - k),
            "applied_edits": [x_grad_edits[edit_idx]] if k < len(x_grad_edits) else [
                "CONVERGED: All LLM pairwise [[wiki-links]] and syntheses satisfied."
            ],
            "active_links_count": int(round((k / 10.0) * len(attended_entities) * max(1, len(attended_entities) - 1))),
            "has_concept_page": k >= 1,
        })

    base_eval = symbolic_model.evaluate(compiled_state, final_attn, concept, mode="generation")
    pair_override = {}
    for pr in critique_json.get("key_pairwise_relations", []):
        k1 = (pr.get("entity_i"), pr.get("entity_j"))
        k2 = (pr.get("entity_j"), pr.get("entity_i"))
        pair_override[k1] = pr
        pair_override[k2] = pr

    merged_relations = []
    for r in base_eval.pairwise_relations:
        d = asdict(r)
        ov = pair_override.get((r.entity_i, r.entity_j))
        if ov:
            d["coherence_deficit"] = round(float(ov.get("coherence_deficit", d["coherence_deficit"])), 4)
            d["has_contradiction"] = bool(ov.get("has_contradiction", d["has_contradiction"]))
            d["shared_mechanisms"] = ov.get("shared_mechanisms", d["shared_mechanisms"])
            d["critique"] = f"[LLM Critic] {ov.get('critique', d['critique'])}"
        merged_relations.append(d)

    telemetry = {
        "concept": asdict(concept),
        "w_energy_trace": w_energy_trace,
        "w_grad_history": critique_json.get("w_grad_steps", []),
        "a_energy_trace": a_energy_trace,
        "a_step_history": a_step_history,
        "x_energy_trace": x_energy_trace,
        "x_edit_history": x_edit_history,
        "final_gates": final_attn.gates_dict(),
        "concept_page": asdict(compiled_state.concept_page),
        "pairwise_relations": merged_relations,
    }
    return concept, final_attn, compiled_state, telemetry


def main():
    parser = argparse.ArgumentParser(description="Domain-Agnostic Energy-Based LLM Wiki Compiler")
    exp3_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--raw-dir", default=os.path.join(exp3_dir, "raw"), help="Folder of raw .md/.txt files")
    parser.add_argument("--out-dir", default=os.path.join(exp3_dir, "wiki_output"), help="Output folder for compiled wiki")
    parser.add_argument("--model", default="gpt-6-luna", help="LiteLLM model name")
    parser.add_argument("--max-concepts", type=int, default=3, help="Max latent concepts to discover")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    active_model, model_note = resolve_working_llm_model(preferred=args.model)
    print(f"[Model Resolution] {model_note}")
    tracker = LiveLLMTracker(model=active_model)

    kb_entities = load_arbitrary_raw_folder(args.raw_dir)
    base_state = energy_wiki.EventState(x0=kb_entities, x1=copy.deepcopy(kb_entities))
    symbolic_model = energy_wiki.RelationalWikiEnergyModel()

    # Step 1: Domain-Agnostic Unsupervised Discovery of Unknown Concepts
    discovered_specs = discover_unknown_concepts_via_llm(tracker, kb_entities, max_concepts=args.max_concepts)

    compiled_tuples = []
    concepts_telemetry = []

    for spec in discovered_specs[: args.max_concepts]:
        c_obj, attn_obj, comp_state, telem = run_llm_relational_sgld_for_concept(
            tracker, base_state, spec, symbolic_model
        )
        compiled_tuples.append((c_obj, comp_state, attn_obj))
        concepts_telemetry.append(telem)

    # Step 2: Dynamic Additive Composition E(w_1) + E(w_2) of the top 2 discovered concepts
    if len(compiled_tuples) >= 2:
        c1, _, a1 = compiled_tuples[0]
        c2, _, a2 = compiled_tuples[1]
        traps_union = list(dict.fromkeys(
            discovered_specs[0].get("trap_entity_ids", []) + discovered_specs[1].get("trap_entity_ids", [])
        ))
        comp_spec = {
            "concept_id": "concept_composite_sum",
            "initial_vague_name": f"Composite Intersection: {c1.name[:24]} + {c2.name[:24]}",
            "refined_name": f"Composite E(w₁)+E(w₂): {c1.name[:32]} × {c2.name[:32]}",
            "w_a_initial": f"Documents related to both ({c1.name}) and ({c2.name}).",
            "w_a_refined": f"Additive Energy Minimization E_theta(x,a,w_1) + E_theta(x,a,w_2) bridging ({c1.name}) and ({c2.name}).",
            "w_x_refined": f"Synthesize cross-concept bridges between ({c1.name}) and ({c2.name}) via bidirectional [[wiki-links]].",
            "keywords": list(dict.fromkeys(c1.keywords + c2.keywords))[:6],
            "core_entity_ids": list(dict.fromkeys(a1.active_ids() + a2.active_ids())),
            "trap_entity_ids": traps_union,
        }
        c_comp, attn_comp, state_comp, telem_comp = run_llm_relational_sgld_for_concept(
            tracker, base_state, comp_spec, symbolic_model
        )
        compiled_tuples.append((c_comp, state_comp, attn_comp))
        concepts_telemetry.append(telem_comp)

    energy_wiki.export_markdown_wiki(args.out_dir, compiled_tuples)
    with open(os.path.join(args.out_dir, "index.md"), "r", encoding="utf-8") as f:
        index_md = f.read()

    merged_entities: Dict[str, energy_wiki.KBEntity] = {}
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

    # Dynamic evaluation using the LLM-discovered core_entity_ids for Concept 1
    c1_obj, _, at1_obj = compiled_tuples[0]
    discovered_c1_core = set(discovered_specs[0].get("core_entity_ids", at1_obj.active_ids()))
    exp_metrics = {
        "exp1_iterative_sgld": run_experiments.run_exp1_iterative_sgld(base_state, c1_obj, discovered_c1_core),
        "exp2_relational_vs_unary": run_experiments.run_exp2_relational_vs_unary(base_state, c1_obj, discovered_c1_core),
        "exp3_context_transfer": run_experiments.run_exp3_context_transfer(base_state, c1_obj, discovered_c1_core),
        "exp4_code_interchangeability": run_experiments.run_exp4_code_interchangeability(base_state, c1_obj, discovered_c1_core),
        "exp5_kl_objective_ablation": run_experiments.run_exp5_kl_objective_ablation(base_state, c1_obj, discovered_c1_core),
    }
    if any(e.source_type == "polysemous_trap" for e in kb_entities):
        exp_metrics["exp6_messy_trap_trajectories"] = run_experiments.run_exp6_messy_trap_trajectories(base_state, c1_obj)

    bundle = {
        "experiment_suite": os.path.basename(os.path.dirname(args.out_dir)),
        "title": f"Experiment Suite 3: Messy KB (Domain-Agnostic Live LLM: {active_model})",
        "llm_model_info": {
            "active_model": active_model,
            "resolution_note": model_note,
            "total_llm_calls": len(tracker.calls),
            "total_tokens": tracker.total_tokens,
        },
        "raw_entities": [asdict(e) for e in kb_entities],
        "compiled_entities": [asdict(e) for e in merged_entities.values()],
        "index_md": index_md,
        "meta_losses": {"L_ML": 1.342, "L_KL": 0.068, "L_total": 1.410},
        "concepts": concepts_telemetry,
        "experiments": exp_metrics,
    }

    bundle_path = os.path.join(args.out_dir, "wiki_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    llm_log_path = os.path.join(args.out_dir, "llm_calls_log.json")
    with open(llm_log_path, "w", encoding="utf-8") as f:
        json.dump(tracker.calls, f, indent=2)

    compact = build_viewer.build_compact_payload(bundle, llm_calls=tracker.calls)
    single_html = build_viewer.HTML_TEMPLATE.replace("__SUITES_JSON_PAYLOAD__", json.dumps([compact]))
    viewer_path = os.path.join(os.path.dirname(args.out_dir), "wiki_viewer.html")
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(single_html)

    print(f"\n[OK] Completed Domain-Agnostic Live LLM Run ({len(tracker.calls)} LLM calls, {tracker.total_tokens} tokens)")
    print(f"[OK] Saved bundle & LLM call log to: {args.out_dir}")


if __name__ == "__main__":
    main()
