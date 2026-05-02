# PathoML/optimization

## 1. Purpose
Provide shared training and evaluation runtime for pathology experiments.

## 2. Scope / Owns
This package owns:
- training strategy abstractions
- the shared `Trainer` entry point
- cross-validation and full-dataset training strategies
- patient-level aggregation
- shared training utilities

## 3. Public Contracts
- `PathoML.optimization.trainer.Trainer`
- `PathoML.optimization.trainer.CrossValidator`
- `PathoML.optimization.trainer.FullDatasetTrainer`
- `aggregate_patient_predictions(...)`
- `TrainingConfig` DataLoader knobs: workers, pinned memory, persistent workers, prefetch, non-blocking transfer, and optional length bucketing.

## 4. Invariants
- Keep strategy state in `__init__`; `execute()` takes no runtime arguments.
- Keep shared training code aligned with dataset/model contracts in `PathoML.interfaces`.
- Allow strategy-owned checkpoint metadata extensions.
- Keep DataLoader performance options opt-in through config so existing experiments remain comparable by default.

## 5. Change Rules
- Keep this runtime generic across teacher and distillation.
- Add subsystem-specific orchestration around shared strategies instead of forking unless reuse breaks down.
- Update `TRAINER_DESIGN.md` when strategy structure changes materially.

## Decided
- Distillation continues to reuse shared `CrossValidator`.
- TensorBoard logging is optional; training still works when dependency is unavailable.

## TODO
1. Add shared strategy hooks only after more than one subsystem needs them.
