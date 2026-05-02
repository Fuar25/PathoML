# PathoML/dataset

## 1. Purpose
Provide shared dataset utilities and reusable base classes for pathology feature datasets.

## 2. Scope / Owns
This package owns:
- shared H5 scanning utilities
- sample-key extraction and intersection logic
- CSV label loading
- variable-length collate
- length-bucketed batch sampling for variable-length tensors
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
- `LengthBucketBatchSampler(lengths, batch_size, ...)`
- `_extract_patient_tissue_id(filename, pattern)`
- `load_labels_csv(csv_path)`

## 4. Invariants
- Keep sample keys as `(patient_id, tissue_id)`.
- Sort samples by `(patient_id, tissue_id)` to preserve split reproducibility.
- Keep `slide_id`, `patient_id`, and `tissue_id` as standard item keys for shared training code.
- Keep length bucketing based on explicit dataset metadata when available; do not load full feature tensors just to sort batches.
- Keep this package free of teacher-only concrete dataset registration.

## 5. Change Rules
- Add parsing/scanning logic here only if reused by more than one subsystem.
- Keep teacher concrete datasets in `teacher/dataset/`.
- If a dataset item contract changes shared training expectations, update `PathoML/optimization/DESIGN.md`.

## Decided
- Keep sample-set fingerprinting in shared layer because teacher and distillation both use it.
- Keep shared dataset bases in PathoML so subsystems do not duplicate scanning logic.

## TODO
1. Add shared base classes only when a second subsystem needs them.
