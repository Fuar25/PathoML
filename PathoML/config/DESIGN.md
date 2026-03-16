# PathoML/config

## 1. Purpose
Typed configuration dataclasses for all runtime parameters. No logic — just structured data.

## 2. Dataclasses

| Class | Fields |
|-------|--------|
| `DatasetConfig` | `dataset_name`, `dataset_module_paths`, `dataset_kwargs`, `patient_id_pattern`, `binary_mode` |
| `ModelConfig` | `model_name`, `model_module_paths`, `model_kwargs`, `input_dim`, `hidden_dim`, `num_classes`, `dropout` |
| `TrainingConfig` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `seed`, `device`, `patience`, `patient_threshold` |
| `LoggingConfig` | `save_dir`, `save_best_only` |
| `RunTimeConfig` | Composes the four above |

## 3. Extension Convention
Model-specific parameters (e.g. `gated`, `attention_dim`, `n_heads`) and strategy-specific parameters (e.g. `k_folds`) do **not** belong in config dataclasses.
- Model-specific → `ModelConfig.model_kwargs` dict, passed to `create_model()` and filtered by the model's signature
- Strategy-specific → constructor arguments of the strategy class (e.g. `CrossValidator(..., k_folds=5)`)

`dataset_module_paths` / `model_module_paths` are **empty by default** — all built-ins are auto-loaded by `load_runtime_plugins`. Only set these fields to register user-defined extensions:
```python
config.model.model_module_paths = ['my_project.my_custom_model']
```

## 4. Usage
```python
from PathoML.config.config import RunTimeConfig

config = RunTimeConfig()               # instantiate directly — no singleton
config.training.device = "cuda:0"
config.model.model_kwargs = {"gated": True, "attention_dim": 128}
```

## Decided
- No module-level singleton (`runtime_config = RunTimeConfig()` was removed). Rationale: avoids shared mutable state across experiments and makes each run self-contained.
- `defaults.py` holds only string/numeric constants (`PATIENT_ID_PATTERN`, etc.), imported by both config and data packages.

## TODO
1. Interpretability config: add `InterpretabilityConfig` (output paths for CSV, figures) once the `interpretability/` module is built.
