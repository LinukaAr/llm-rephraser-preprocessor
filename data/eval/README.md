# data/eval/

This directory holds the **held-out evaluation** split of the constraint dataset.

> [!CAUTION]
> **Do not read or use files in this directory during development or prompt tuning.**
> Any model selection or threshold decision made by inspecting eval data constitutes
> data leakage and invalidates your evaluation results (NFR-M1).

Place your final evaluation dataset here once it is frozen (post-RO4 completion).
Run evaluation scripts against this directory only in the final evaluation phase.
