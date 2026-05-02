# distillation

## 1. Purpose
Transfer teacher knowledge to HE-only student models while reusing shared PathoML runtime.

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

Teacher artifacts are consumed through:
- `manifest.json`
- fold split metadata (`train_fold`, `test_fold`, and optional extra metadata)

## 4. Invariants
- Consume teacher artifacts, not teacher experiment internals.
- Reuse shared `CrossValidator` rather than a separate full runtime.
- Verify fold splits before trusting teacher checkpoints.

## 5. Change Rules
- Keep shared logic in `PathoML` unless it is distillation-specific.
- Put experiment-state updates in `distillation/experiments/PLAN.md`.
- If the teacher artifact contract changes, update this file and `distillation/runtime/DESIGN.md`.

## Decided
- Distillation is a peer subsystem to teacher.
- Teacher manifest loading is the formal configuration entry point.
- Distillation owns package-level `dataset/`, `losses/`, and `runtime/` boundaries.

## TODO
1. Revisit a dedicated distillation runtime only if `CrossValidator` reuse becomes a real constraint.
