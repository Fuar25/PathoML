# distillation

## 1. Purpose
Distillation is a research subsystem that transfers teacher knowledge into HE-only student models while reusing the shared PathoML runtime where possible.

## 2. Scope / Owns
This subsystem owns:
- distillation dataset assembly
- distillation loss implementations
- student models
- teacher checkpoint adapters/loaders
- distillation trainer extension
- distillation experiment entry points

This subsystem does not own:
- teacher concrete datasets or models
- shared training strategies
- shared dataset utilities or model primitives

## 3. Public Contracts
- `distillation.dataset.DistillationDataset`
- `distillation.runtime.TeacherManifest`
- `distillation.losses.DistillationLoss`
- `distillation.runtime.DistillCrossValidator`

The subsystem consumes teacher artifacts through:
- `manifest.json`
- fold checkpoint metadata (`fold`, `train_fold`, `test_fold`, and optional extra metadata)

## 4. Invariants
- Distillation consumes teacher artifacts, not teacher experiment internals.
- Distillation reuses the shared `CrossValidator` instead of maintaining a separate full runtime.
- Fold-split verification remains mandatory at runtime before teacher checkpoints are trusted.

## 5. Change Rules
- Keep shared logic in `PathoML` unless it is distillation-specific.
- Put experiment-state updates in `distillation/experiments/PLAN.md`.
- If the teacher artifact contract changes, update this file and `distillation/runtime/DESIGN.md`.

## Decided
- Distillation remains a peer subsystem to teacher, not a subfolder of teacher.
- Teacher manifest loading is the formal entry point for teacher configuration inheritance.
- Distillation owns package-level `dataset/`, `losses/`, and `runtime/` areas because these are growing subsystem boundaries.

## TODO
1. Revisit a dedicated distillation runtime only if reuse of `CrossValidator` becomes a real constraint.
