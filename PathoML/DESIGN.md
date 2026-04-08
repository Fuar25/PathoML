# PathoML

## 1. Purpose
PathoML is the shared pathology foundation for this repository. It provides the reusable contracts, utilities, and training runtime used by multiple research subsystems.

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

## 4. Invariants
- PathoML must stay usable without importing `teacher` or `distillation`.
- Shared contracts live here once they are reused by more than one subsystem.
- Teacher and distillation interact through artifact contracts, not through experiment-level imports.

## 5. Change Rules
- Add code here only if it is genuinely shared across subsystems.
- If a component is teacher-only, keep it in `teacher/`.
- If a component is distillation-only, keep it in `distillation/`.
- When a shared interface changes, update the nearest package `DESIGN.md` in the same turn.

## Decided
- PathoML is a shared foundation, not the teacher framework.
- Shared dataset scanning, sample-key handling, and collate logic stay in PathoML.
- Shared ABMIL building blocks stay in PathoML even though teacher owns the concrete `ABMIL` model.

## TODO
1. Consider a future interpretability subsystem once a stable shared contract emerges.
