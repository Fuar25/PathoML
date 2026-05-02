# PathoML/config

## 1. Purpose
Provide typed runtime configuration dataclasses for shared training flows.

## 2. Scope / Owns
This package owns:
- `DatasetConfig`
- `ModelConfig`
- `TrainingConfig`
- `LoggingConfig`
- `RunTimeConfig`
- cross-subsystem default constants in `defaults.py`

## 3. Public Contracts
- `DatasetConfig`
  - `dataset_name`
  - `dataset_module_paths`
  - `dataset_kwargs`
  - `patient_id_pattern`
- `ModelConfig`
  - `model_name`
  - `model_module_paths`
  - `model_kwargs`
- `TrainingConfig`
  - shared training hyperparameters, runtime device, and early-stopping controls
  - includes `early_stopping_metric`, `patience`, and `min_delta`
- `LoggingConfig`
  - save directory and checkpoint behavior
- `RunTimeConfig`
  - top-level composition of the four sections above

## 4. Invariants
- Keep runtime data in config dataclasses, not training logic.
- Keep strategy-specific parameters on strategy constructors, not config dataclasses.
- Keep model-specific kwargs in `model_kwargs`.
- Keep dataset-specific kwargs in `dataset_kwargs`.

## 5. Change Rules
- Add fields only when shared and stable across callers.
- Keep per-experiment one-off knobs out of the shared config layer.
- If a new field changes factory/strategy invocation, update corresponding design docs.

## Decided
- Instantiate `RunTimeConfig` directly; no global singleton.
- Keep shared multi-package constants in `defaults.py`.

## TODO
1. Add config sections only after a stable shared subsystem requires them.
