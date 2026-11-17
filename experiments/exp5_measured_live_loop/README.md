# Exp5: conditional energy-based document synthesis

**One corrected live run completed:** ten valid document-only revisions, 32 calls, 144,168 reported tokens. The composite critic score went from 17.0 to 0, with temporary increases at revisions 4 and 9. All source hashes and fixed conditions were verified unchanged. [Report](runs/conditional_prepared/REPORT.md) · [Final document](runs/conditional_prepared/final_document.md) · [Every draft and score](runs/conditional_prepared/trajectory.json). These are critic ratings from one run, not independent validation.

This extends Exp4's synthesis operation. The corpus `D`, concept definitions `w1`, `w2`, and source masks `a1`, `a2` stay fixed. Only the generated document `x_new` is updated:

```text
x_new* = argmin_x [E_theta([D,x], a1, w1) + E_theta([D,x], a2, w2)]
```

Concept discovery and document identification are separate conditional operations. This experiment performs neither of them and does not alternate between variables.

## Fixed inputs and the score

The runner uses Exp4's original seed and source clusters: six energy/sampling sources and five text-optimization sources. The full corpus contains 18 notes. Explicit binary masks record membership for every source. The two natural-language concept definitions, masks and corpus are serialized immutably and hashed in the manifest. API requests receive copies; model output cannot overwrite them.

The **same critic prompt and model** evaluate each document twice:

1. `E1`: critic sees `D`, `a1`, `w1` and `x_new`, without `w2` or `a2`.
2. `E2`: critic sees `D`, `a2`, `w2` and `x_new`, without `w1` or `a1`.

For either fixed concept, the score is the sum of concept deficit `[0,3]`, source-support deficit `[0,2]`, relation deficit `[0,2.5]`, and calibration deficit `[0,2]`. The critic must supply evidence for each component. Python computes `E_total = E1 + E2`; these are two actual conditional evaluations, not shares of a fabricated combined score.

This common per-concept rubric replaces Exp4's task-specific four-term rating. It preserves the conditional synthesis structure but its numerical scores are not directly comparable with Exp4's saved results. It is hand-designed and requires empirical calibration; it is not a learned energy model or a validated sampler.

## Actual revision loop

1. Evaluate the seed under each fixed pair.
2. Give the editor the current document, both critiques, and fixed conditions.
3. Accept only the output schema `{"title": "...", "content": "..."}`. Any attempted concept, membership, corpus or objective field is invalid. Citations must refer to the fixed selected sources.
4. Evaluate the new document under the same two pairs, using fresh critic contexts without previous scores.
5. Apply each valid edit, following Exp4's update policy. Record score increases and plateaus as observed. Repeat for the requested number of real revisions.

Invalid outputs consume a revision slot and leave the previous document intact. They have no assigned candidate energy and receive no free retry. The report distinguishes valid edits from requested revision slots. Malformed critic scores stop execution; the raw response remains in the call log.

## Prepare and run

Preparing a manifest requires no API access, credentials or LiteLLM import:

A completed run exists at `runs/conditional_prepared/`. Choose a new output directory for another run; preparation never overwrites an existing run. Resuming the completed run reuses cached results and makes no new calls.

```powershell
.venv/Scripts/python.exe experiments/exp5_measured_live_loop/run_exp5.py --out experiments/exp5_measured_live_loop/runs/conditional_prepared
```

To execute that prepared run explicitly:

```powershell
.venv/Scripts/python.exe experiments/exp5_measured_live_loop/run_exp5.py --out experiments/exp5_measured_live_loop/runs/conditional_prepared --resume --live
```

Defaults: ten revision slots, up to **32 model calls** (two seed evaluations plus one editor and two critic calls per revision), `gpt-6-luna`, 2,400 output tokens per call, and a 150,000-token run ceiling. Adjust `--steps`, `--model`, `--max-output-tokens`, and `--token-budget` when preparing. Live calls use the existing `.env` configuration, send the source notes to that API, and incur usage charges. The run stops at its token ceiling rather than silently changing the experiment.

`--resume` requires identical source hashes, fixed conditions, prompts, seed, model and limits. Cached calls are reused only with identical inputs. Old joint-update runs are incompatible and cannot be resumed by this implementation.

## Outputs and verification

`manifest.json` records all fixed conditions and the protocol. A live run additionally writes full `calls.jsonl`, every draft/critique/score in `trajectory.json`, `final_document.md`, `REPORT.md`, and `summary.json`. Every iteration records the same fixed-condition fingerprint. Credentials are never logged.

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiments/exp5_measured_live_loop -p test_loop.py
```

Tests verify immutability, rejection of concept/mask changes, separate identical-rubric scoring, exact energy addition, real iteration counts, retention of rising scores, invalid-output handling, and resume boundaries. Test outputs are temporary mock data, not experimental measurements.

## Superseded joint-update pilot

`runs/pilot_live/` preserves the earlier paid run for provenance. **It optimized the wrong variables and is not evidence for or against this formulation.** Its reports and conclusions must not be treated as results of corrected Exp5. The original runner snapshot is retained there as `legacy_joint_runner.py.txt`; no new live calls were made while implementing this correction.
