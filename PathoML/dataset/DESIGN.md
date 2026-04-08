# PathoML/dataset

## 1. Purpose
Shared dataset utilities and reusable base classes for pathology feature datasets.

## 2. Scope / Owns
This package owns:
- shared H5 scanning utilities
- sample-key extraction and intersection logic
- CSV label loading
- variable-length collate
- shared base classes for unimodal and multimodal feature datasets

This package does not own:
- teacher concrete datasets
- distillation-specific dataset assembly

## 3. Public Contracts
- `UnimodalFeatureDatasetBase`
- `MultimodalSlideDatasetBase`
- `find_common_sample_keys(data_root, stains, patient_id_pattern=...)`
- `fingerprint_sample_keys(sample_keys)`
- `_variable_size_collate(batch)`
- `_extract_patient_tissue_id(filename, pattern)`
- `load_labels_csv(csv_path)`

## 4. Invariants
- Sample keys are `(patient_id, tissue_id)`.
- Dataset implementations must sort samples by `(patient_id, tissue_id)` to preserve split reproducibility.
- `slide_id`, `patient_id`, and `tissue_id` remain standard item keys for shared training code.
- This package stays free of teacher-only concrete dataset registration.

## 5. Change Rules
- Put new shared parsing/scanning logic here only if it is reused by more than one subsystem.
- Put teacher concrete datasets in `teacher/dataset/`.
- If a new dataset item contract changes shared training expectations, update `PathoML/optimization/DESIGN.md` as well.

## Decided
- Sample-set fingerprinting is part of the shared layer because both teacher and distillation use it.
- Shared dataset bases live in PathoML so subsystems do not duplicate scanning logic.

## TODO
1. Add more shared base classes only when a second subsystem needs them.
