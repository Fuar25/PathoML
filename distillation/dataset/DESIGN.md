# distillation/dataset

## 1. Purpose
Own the distillation-specific dataset assembly that combines HE patch features with ordered slide-level teacher inputs.

## 2. Scope / Owns
This package owns:
- `DistillationDataset`
- distillation-specific H5 loading helpers
- distillation-specific sample assembly rules

This package does not own:
- shared H5 scanning and sample-key utilities
- teacher concrete datasets

## 3. Public Contracts
- `DistillationDataset(patch_root, slide_root, slide_stains, labels_csv, ...)`
- dataset items expose:
  - `he_patches`
  - `slide_concat`
  - `label`
  - `patient_id`
  - `tissue_id`
  - `slide_id`

## 4. Invariants
- `slide_concat` respects the order of `slide_stains`.
- Sample ordering is stable because sample keys are sorted.
- Shared scanning logic continues to come from `PathoML.dataset.utils`.

## 5. Change Rules
- Keep only distillation-specific assembly logic here.
- If the dataset item contract changes, update `distillation/DESIGN.md` and the loss/runtime docs when relevant.

## Decided
- Distillation keeps its own dataset package because the assembly is not shared with teacher.

## TODO
1. Split more helpers out only if the package grows beyond a single cohesive assembly unit.
