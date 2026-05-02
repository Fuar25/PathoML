# PathoML

## 1. Purpose
Provide the shared pathology foundation for this repository: reusable contracts, utilities, and training runtime.

## 2. Scope / Owns
PathoML owns:
- `interfaces.py`
- `registry.py`
- `config/`
- `optimization/`
- shared dataset utilities and base classes
- shared model primitives

PathoML does not own:
- teacher concrete datasets or models
- teacher experiment orchestration
- distillation-specific datasets, losses, students, or experiments

## 3. Public Contracts
- `PathoML.interfaces`: shared dataset/model contracts
- `PathoML.registry`: registry/factory APIs and the core loader
- `PathoML.config.config`: runtime dataclasses
- `PathoML.dataset`: shared dataset bases and utility functions
- `PathoML.models`: shared model primitives
- `PathoML.optimization.trainer`: `Trainer`, `CrossValidator`, and `FullDatasetTrainer`
- `TrainingConfig` includes optional DataLoader performance knobs; defaults preserve single-process loading.

## 4. Invariants
- Keep PathoML usable without importing `teacher` or `distillation`.
- Keep shared contracts here once reused by more than one subsystem.
- Keep teacher/distillation interaction at artifact-contract level, not experiment-level imports.

## 5. Change Rules
- Add code here only when it is genuinely shared across subsystems.
- Keep teacher-only components in `teacher/`.
- Keep distillation-only components in `distillation/`.
- If a shared interface changes, update the nearest package `DESIGN.md` in the same turn.

## Decided
- PathoML is a shared foundation, not the teacher framework.
- Shared dataset scanning, sample-key handling, and collate logic stay in PathoML.
- Shared length-bucketed batching stays in PathoML because any variable-length feature dataset can use it.
- Shared ABMIL building blocks stay in PathoML even though teacher owns the concrete `ABMIL` model.

## TODO
1. Consider an interpretability subsystem once a stable shared contract emerges.
