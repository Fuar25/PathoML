# PathoML/optimization

## 1. Purpose
Shared training and evaluation runtime for pathology experiments.

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

## 4. Invariants
- Strategies store all state in `__init__`; `execute()` takes no runtime arguments.
- Shared training code works with the dataset/model contracts defined in `PathoML.interfaces`.
- Checkpoint metadata may be extended by callers through strategy-owned metadata fields.

## 5. Change Rules
- Keep this runtime generic across teacher and distillation.
- Put subsystem-specific orchestration around the shared strategies instead of forking them unless reuse breaks down.
- Update `TRAINER_DESIGN.md` when the strategy structure changes materially.

## Decided
- Distillation continues to reuse the shared `CrossValidator`.
- TensorBoard logging is optional at runtime; training still works when the dependency is unavailable.

## TODO
1. Add new shared strategy hooks only after more than one subsystem needs them.
