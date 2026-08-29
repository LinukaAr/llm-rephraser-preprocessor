# data/dev/

This directory holds the **development** split of the constraint dataset.

## Split policy (NFR-M1 — leakage prevention)

| Directory | Purpose | May be seen by? |
|---|---|---|
| `data/dev/` | Iteration, prompt tuning, threshold selection | Development code only |
| `data/eval/` | Final held-out evaluation | Evaluation scripts only |
| `data/pilot_constraints.json` | Seed pair bank for retrieval rephraser | Pipeline at runtime |

**Rule**: no sample that appears in `data/eval/` may be used in `data/pilot_constraints.json` or `data/dev/`.
Document the split date and rationale in your dissertation appendix.
