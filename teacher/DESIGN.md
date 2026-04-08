# teacher

## 1. Purpose
Teacher selection is a research subsystem that assembles concrete datasets and concrete models on top of the shared PathoML foundation and produces teacher artifacts for downstream distillation.

## 2. Scope / Owns
This subsystem owns:
- teacher concrete datasets
- teacher concrete models
- teacher runtime loader
- teacher experiment entry points
- teacher artifact generation

This subsystem does not own:
- shared training strategies
- shared dataset utilities
- shared model primitives
- distillation internals

## 3. Public Contracts
- `teacher.runtime.load_teacher_modules()`
- teacher artifact contract written to `experiments/outputs/<condition>/manifest.json`
- fold checkpoints written under `experiments/outputs/<condition>/run_{run:02d}/`

The teacher artifact contract includes:
- `schema_version`
- `artifact_type`
- `producer_system`
- `condition_name`
- `model_name`
- `model_kwargs`
- `n_runs`
- `k_folds`
- `base_seed`
- `modality_names`
- `data_root`
- `labels_csv`
- `sample_set_fingerprint`
- `ckpt_template`

## 4. Invariants
- Teacher concrete modules register themselves through the shared PathoML registry.
- Teacher artifacts are the only formal interface consumed by distillation.
- Teacher code depends on `PathoML`, but distillation does not depend on teacher experiment internals.

## 5. Change Rules
- Put concrete teacher datasets/models here, not in `PathoML/`.
- If artifact fields or semantics change, update `distillation/runtime/DESIGN.md` and `distillation/DESIGN.md` in the same turn.
- Keep current experiment status in `teacher/experiments/PLAN.md`.

## Decided
- Teacher experiments live under `teacher/experiments/`.
- Teacher owns the concrete `ABMIL`, `LinearProbe`, and `MLP` models.

## TODO
1. Add more teacher concrete assemblies only when they are part of the teacher research space.
