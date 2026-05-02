# teacher

## 1. Purpose
Assemble concrete teacher datasets/models on shared PathoML and produce teacher artifacts for distillation.

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
- teacher run artifacts at `../PathoML-runs/teacher/<condition>/manifest.json`
- current distillation-facing teacher winner at `../PathoML-runs/teacher-winners/manifest.json`
- fold checkpoints under `run_{run:02d}/` beside the manifest

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
- Register teacher concrete modules through shared PathoML registry.
- Use teacher artifacts as the only formal distillation-facing interface.
- Keep exactly one default teacher winner for distillation; document its source in `TEACHER.md`.
- Keep dependency direction: teacher -> `PathoML`; distillation does not depend on teacher experiment internals.
- Name slide-level dataset modules with a `_slide` suffix; reserve patch-level names for true patch/region datasets.
- Registered multimodal patch datasets align stains by shared coordinates before feature concatenation.
- Registered multimodal patch datasets may cache aligned tensors in memory; cached and uncached modes must return equivalent item payloads.
- Registered patch fusion models must preserve stain boundaries before any shared patch-level projection.

## 5. Change Rules
- Keep concrete teacher datasets/models here, not in `PathoML/`.
- If artifact fields or semantics change, update `distillation/runtime/DESIGN.md` and `distillation/DESIGN.md` in the same turn.
- Keep experiment status in `teacher/experiments/PLAN.md`.

## Decided
- Teacher experiments live under `teacher/experiments/`.
- Temporary validation/analysis scripts live under `teacher/script/`.
- Teacher owns concrete `ABMIL`, `LinearProbe`, `MLP`, and registered patch fusion MIL models.

## TODO
1. Add teacher concrete assemblies only when they belong to teacher research scope.
