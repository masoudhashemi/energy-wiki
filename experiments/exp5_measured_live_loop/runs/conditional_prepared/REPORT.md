# Conditional synthesis: measured revisions

Only x_new changes. D, w1, a1, w2 and a2 are fixed throughout.

Each valid edit is applied as in Exp4; energy increases and plateaus are retained. Invalid proposals consume a revision slot without a score. These are critic ratings, not independent quality measurements.

Fixed-condition SHA-256: `682a307ec99973968168b08c629f08a5bc75a5089d2bc3d5f57acadf8e524354`

| Revision | E1 | E2 | Composite energy | Decision |
| --- | --- | --- | --- | --- |
| 0 | 8.000 | 9.000 | 17.000 | Seed |
| 1 | 1.750 | 1.600 | 3.350 | Applied document edit |
| 2 | 0.000 | 0.500 | 0.500 | Applied document edit |
| 3 | 0.000 | 0.000 | 0.000 | Applied document edit |
| 4 | 0.000 | 0.250 | 0.250 | Applied document edit |
| 5 | 0.000 | 0.000 | 0.000 | Applied document edit |
| 6 | 0.000 | 0.000 | 0.000 | Applied document edit |
| 7 | 0.000 | 0.000 | 0.000 | Applied document edit |
| 8 | 0.000 | 0.000 | 0.000 | Applied document edit |
| 9 | 0.000 | 0.500 | 0.500 | Applied document edit |
| 10 | 0.000 | 0.000 | 0.000 | Applied document edit |

[Every draft, critique and score](trajectory.json) · [Fixed conditions](manifest.json) · [Full calls](calls.jsonl) · [Final document](final_document.md)

E1 and E2 are separate evaluations with the SAME fixed rubric, each receiving only its own fixed concept and mask. The composite is their sum. Scores use a new per-concept rubric and are not numerically comparable with Exp4's saved four-term totals. A smaller score alone does not establish convergence, sampling guarantees, or new scientific knowledge.
