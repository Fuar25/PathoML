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
- `DistillCrossValidator` may cache fold-local teacher outputs for `hidden` and `logit`.
- Registry-backed teacher adapters normalize `bag_embeddings`/`logits` to `hidden`/`logit`.
- Registered patch teachers use teacher-owned patch datasets for teacher-output precompute; the student dataset remains HE-only.

## 4. Invariants
- Reuse shared `PathoML` cross-validation runtime.
- Verify teacher checkpoints against distillation fold splits before training.
- Use teacher artifacts as the only formal teacher-to-distillation interface.
- Keep the fixed winner path as a convenience default; runtime loading still accepts an explicit manifest path.
- Cached teacher outputs must match frozen eval teacher forward results for the same fold and sample.
- Registry-backed teachers require cached teacher outputs because their registered patch inputs are separate from student HE patch batches.

## 5. Change Rules
- Keep subsystem-specific orchestration here, not in `PathoML`.
- If the teacher artifact contract changes, update this file and `teacher/DESIGN.md`.

## Decided
- Keep `runtime` package-level because manifest handling and trainer adaptation share one boundary.
- Cache teacher outputs in process by default after fold checkpoint verification.

## TODO
1. Consider deeper runtime decomposition only if trainer and artifact handling diverge substantially.
