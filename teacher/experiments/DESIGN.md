# teacher/experiments

## 1. Purpose
Stable experiment rules, outputs, and tracking for teacher-selection runs.

## 2. Scope / Owns
This directory owns:
- teacher experiment entry scripts
- experiment defaults and helpers in `common.py`
- experiment outputs
- experiment status tracking

## 3. Public Contracts
- Script naming pattern: `run_<condition>.py`
- Output layout:
  - `outputs/<condition>/manifest.json`
  - `outputs/<condition>/run_{run:02d}/model_fold_{fold}_best.pth`
  - `outputs/<condition>/run_{run:02d}/cv_predictions.csv`
- Tracking files:
  - `PLAN.md`
  - `results_log.txt`
- Git ignore rules must target generated output directories such as `teacher/experiments/outputs/`, not the `teacher/experiments/` code directory.

## 4. Invariants
- `PLAN.md` records only experiment status, results, and next steps.
- `common.py` writes the teacher artifact manifest after a condition finishes.
- Cross-condition comparability requires the same sample set, seed regime, and ordering assumptions.

## 5. Change Rules
- If script naming or output layout changes, update this file and `distillation/runtime/DESIGN.md`.
- If a change is architectural rather than experimental, document it in `teacher/DESIGN.md`, not `PLAN.md`.

## Decided
- Experiment tracking lives inside `teacher/experiments/`.
- Teacher experiment outputs are the canonical producer of distillation input artifacts.

## TODO
1. Add helper utilities here only when they are stable across multiple experiment scripts.
