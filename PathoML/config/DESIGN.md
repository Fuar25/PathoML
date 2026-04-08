# PathoML/config

## 1. Purpose
Typed runtime configuration dataclasses for shared training flows.

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
  - shared training hyperparameters and runtime device
- `LoggingConfig`
  - save directory and checkpoint behavior
- `RunTimeConfig`
  - top-level composition of the four sections above

## 4. Invariants
- Config dataclasses contain runtime data, not training logic.
- Strategy-specific parameters stay on strategy constructors, not in config dataclasses.
- Model-specific kwargs stay in `model_kwargs`.
- Dataset-specific kwargs stay in `dataset_kwargs`.

## 5. Change Rules
- Add fields only when they are shared and stable across callers.
- Keep per-experiment one-off knobs out of the shared config layer.
- If a new field changes how factories or strategies are invoked, update the corresponding design docs.

## Decided
- `RunTimeConfig` is instantiated directly; there is no global singleton.
- `defaults.py` holds shared constants used by multiple packages.

## TODO
1. Add new config sections only after a stable shared subsystem requires them.
