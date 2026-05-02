# teacher/experiments

## 1. Purpose
Define stable rules, outputs, and tracking for teacher-selection runs.

## 2. Scope / Owns
This directory owns:
- teacher experiment entry scripts
- experiment defaults and helpers in `common.py`
- experiment outputs
- experiment status tracking

This directory does not own:
- temporary validation or one-off analysis scripts

## 3. Public Contracts
- Script naming pattern: `run_<condition>.py`.
- Variant suffix rule: append directly in condition name (for example `_bs32`, not `__bs32`).
- Output layout:
  - default root: `../PathoML-runs/teacher/<condition>/`
  - `manifest.json`
  - `run_{run:02d}/model_fold_{fold}_best.pth`
  - `run_{run:02d}/cv_predictions.csv`
- Temporary validation scripts live under `teacher/script/`; they must not define canonical condition names unless intentionally reproducing an existing artifact.
- Tracking files:
  - `PLAN.md`
  - `results_log.txt`
- Git ignore rules target generated outputs, not code directories.

## 4. Invariants
- `PLAN.md` records only experiment status, results, and next steps.
- `common.py` writes teacher artifact manifests after each condition finishes.
- Cross-condition comparison requires the same sample set, seed regime, and ordering assumptions.
- Canonical distillation-facing artifacts must be reproducible from standard `run_<condition>.py` entry points in `teacher/experiments/`.
- `launch_parallel_runs.py` may run independent run indices on separate GPUs; it must keep the standard output layout and aggregate the same manifest/log contract.

## 5. Change Rules
- If script naming or output layout changes, update this file and `distillation/runtime/DESIGN.md`.
- If a change is architectural (not experimental), document it in `teacher/DESIGN.md`, not `PLAN.md`.

## Decided
- Experiment tracking lives in `teacher/experiments/`.
- Teacher experiment outputs are the canonical producer of distillation input artifacts.
- Long-lived teacher artifacts are promoted to `../PathoML-runs/teacher-winners/`; repo logs keep conclusions, not checkpoints.
- Registered patch experiments use cached aligned items and DataLoader tuning by default; environment variables may disable these performance options for diagnostics.

## TODO
1. Add helper utilities only when stable across multiple experiment scripts.
