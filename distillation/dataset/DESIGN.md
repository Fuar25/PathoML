# distillation/dataset

## 1. Purpose
Own distillation-specific dataset assembly for HE patch features and ordered slide-level teacher inputs.

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
- Dataset item fields: `he_patches`, `slide_concat`, `label`, `patient_id`, `tissue_id`, `slide_id`

## 4. Invariants
- `slide_concat` follows `slide_stains` order.
- Sample ordering stays stable via sorted sample keys.
- Shared scanning logic comes from `PathoML.dataset.utils`.

## 5. Change Rules
- Keep only distillation-specific assembly logic here.
- If the dataset item contract changes, update `distillation/DESIGN.md` and relevant loss/runtime docs.

## Decided
- Distillation keeps its own dataset package because this assembly is not shared with teacher.

## TODO
1. Split more helpers only if this package grows beyond one cohesive assembly unit.
