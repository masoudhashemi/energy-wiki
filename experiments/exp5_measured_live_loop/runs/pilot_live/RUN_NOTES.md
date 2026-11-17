# Development pilot: interpretation and execution notes

> **SUPERSEDED — not a conditional synthesis experiment.** The joint updates in this run departed from the intended formulation. Its observations cannot support or challenge synthesis with fixed concepts and masks. The corrected runner has been implemented separately; these historical logs are retained only for provenance.

This run checks that the proposed joint-state loop can execute and retain evidence. It is not a research result establishing the benefit or failure of relational energy modeling.

- 28 successful model calls; 121,379 reported tokens, including the shared seed assessment.
- Three proposal slots per arm. One relational proposal omitted the page field and was rejected before scoring, leaving that arm with two valid revisions. It was not repaired or retried.
- The initial editor request encountered the API's JSON-format prompt requirement. The prompt was corrected and the saved initial evaluation was reused. Failed requests have no reported token usage in this log.
- Ordinary and topic-2 revision 3 were rejected for increased evaluated energy. All raw proposals and decisions remain in the per-arm records.
- Each final evaluation used a fresh context without the arm label or previous scores, but the same model and rubric. These are not independent human quality judgments.

## What the pilot cannot answer

All four final outputs received zero deficit for both topic requirements. The rubric therefore did not distinguish the constituent objectives from their composition on this task. Explicit pair feedback showed no advantage in the final ratings, but a single run, unequal valid revision counts and noisy same-model assessment cannot establish that the mechanism is ineffective.

The appropriate next step is to validate the measurement before running a larger comparison: check whether independently assessed examples with missing topic coverage, unsupported claims, irrelevant citations and useful cross-source explanations receive distinguishable ratings. Use repeated runs and a comparison protocol that handles invalid proposals explicitly. No further live calls were made after this pilot.

The [report](REPORT.md), [summary](summary.json), [manifest](manifest.json), and [complete calls](calls.jsonl) are retained for diagnosis. Accepted-score curves are monotone by the selection rule; they do not demonstrate convergence.
