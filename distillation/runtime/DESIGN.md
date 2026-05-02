# distillation/runtime

## 1. Purpose
Own distillation-specific runtime adaptation around shared PathoML training runtime.

## 2. Scope / Owns
This package owns:
- teacher manifest loading
- teacher artifact normalization
- fold-level teacher verification
- `DistillCrossValidator`

This package does not own:
- shared training strategies
- student model architectures
- distillation experiment status tracking

## 3. Public Contracts
- `TeacherManifest`
- `load_manifest(manifest_path)`
- `DistillCrossValidator(...)`
- Default experiment helpers consume the fixed teacher winner manifest at `../PathoML-runs/teacher-winners/manifest.json`.
- Teacher forward outputs consumed by losses: `hidden`, `logit`, and optional loss-specific fields such as `class_weight`

## 4. Invariants
- Reuse shared `PathoML` cross-validation runtime.
- Verify teacher checkpoints against distillation fold splits before training.
- Use teacher artifacts as the only formal teacher-to-distillation interface.
- Keep the fixed winner path as a convenience default; runtime loading still accepts an explicit manifest path.

## 5. Change Rules
- Keep subsystem-specific orchestration here, not in `PathoML`.
- If the teacher artifact contract changes, update this file and `teacher/DESIGN.md`.

## Decided
- Keep `runtime` package-level because manifest handling and trainer adaptation share one boundary.

## TODO
1. Consider deeper runtime decomposition only if trainer and artifact handling diverge substantially.
