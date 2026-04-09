# distillation/experiments

## 1. Purpose
Stable experiment rules, outputs, and tracking for distillation runs.

## 2. Scope / Owns
This directory owns:
- distillation experiment entry scripts
- shared experiment helper code in `common.py`
- experiment outputs
- experiment status tracking

## 3. Public Contracts
- Experiment entry scripts import from `distillation.experiments.common`
- Script naming pattern: `run_<fully_descriptive_method_name>.py`
- Condition/output naming pattern: use lowercase snake case with full words; write decimal values with `p` instead of `.`
- Active `condition` names are derived from the composed loss terms instead of hidden family parameters.
- Output layout:
  - `outputs/<condition>/run_{run:02d}/...`
  - `results_log.txt`
  - `PLAN.md`
- Teacher input is a manifest path that resolves to a teacher artifact contract

## 4. Invariants
- `PLAN.md` records only experiment status, results, and next steps.
- `common.py` is the only stable entry helper for distillation experiment scripts.
- Experiment scripts do not manipulate `sys.path`; they use package imports.
- Distillation experiment filenames and condition names avoid abbreviations when a full word is practical.
- Distillation experiment families reuse one stable vocabulary per concept. For example, `teacher_guided_attention` remains the canonical term across files and condition names.
- Shared logging records a human-readable `loss_design` derived from the active composed terms.
- Git ignore rules must target generated output directories such as `distillation/experiments/outputs/`, not the `distillation/experiments/` code directory.

## 5. Change Rules
- If experiment wiring changes, update this file.
- If architecture changes, update `distillation/DESIGN.md` instead of `PLAN.md`.
- Keep log/result schema changes aligned with current experiments.
- Rename scripts and condition identifiers when a name becomes ambiguous instead of introducing another short alias.

## Decided
- Distillation experiment tracking lives in `distillation/experiments/`.
- `common.py` is responsible for teacher manifest loading, dataset construction, run orchestration, and shared logging.
- Historical entries in `results_log.txt` may contain pre-standardization short aliases. New experiments must not reuse those aliases.
- `teacher_guided_attention` now refers to the repaired logits-space implementation by default; pre-logits runs are treated as historical experiment errors and are not active entry points.

## TODO
1. Consolidate repeated experiment-script boilerplate further if it stays stable across methods.
