# distillation/runtime

## 1. Purpose
Own the distillation-specific runtime adaptation around the shared PathoML training backbone.

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

## 4. Invariants
- Distillation still reuses `PathoML` shared cross-validation runtime.
- Teacher checkpoints are verified against distillation fold splits before training proceeds.
- Teacher artifacts remain the only formal interface from teacher to distillation.

## 5. Change Rules
- Keep subsystem-specific orchestration here, not in `PathoML`.
- If the teacher artifact contract changes, update this file and `teacher/DESIGN.md`.

## Decided
- `runtime` is package-level because manifest handling and trainer adaptation are part of one subsystem boundary.

## TODO
1. Consider deeper runtime decomposition only if trainer and artifact handling diverge substantially.
