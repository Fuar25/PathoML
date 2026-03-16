# PathoML/optimization

## 1. Purpose
Training orchestration: strategy pattern entry point, dataset/model registries, and patient-level prediction aggregation.

## 2. Sub-modules

| Module | Responsibility |
|--------|---------------|
| `interfaces.py` | Abstract base classes: `BaseDataset`, `BaseModel`, `BaseMIL`, `Aggregator`, `Classifier` |
| `registry.py` | `Registry`, `register_model`, `register_dataset`, `create_model`, `create_dataset`, `load_runtime_plugins` |
| `trainer.py` | `Trainer`, `CrossValidator`, `FullDatasetTrainer`, `TrainingMixin`, `FoldTrainer`, `EarlyStopping`, `CheckpointManager` |
| `patient_aggregation.py` | `aggregate_patient_predictions()` — tissue→patient aggregation (max-probability MIL rule) |

## 3. Entry Point
```python
from PathoML.optimization.trainer import CrossValidator, Trainer
from PathoML.optimization.registry import create_model, create_dataset

strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
Trainer(strategy).fit()   # returns TrainingResult
```

## 4. Registry
`load_runtime_plugins(config)` must be called before any `create_model` / `create_dataset` call. It:
1. Auto-imports all built-in modules (defined in `_BUILTIN_DATASET_MODULES` / `_BUILTIN_MODEL_MODULES` in `registry.py`).
2. Imports any user extension modules listed in `config.dataset.dataset_module_paths` / `config.model.model_module_paths`.

Self-registration happens via `@register_model('key')` / `@register_dataset('key')` decorators — they fire at import time.

**Adding a built-in**: add the module path to `_BUILTIN_*_MODULES` in `registry.py`.
**Adding a user extension**: append the module path to `config.model.model_module_paths` (or dataset equivalent) in the run script — no core changes needed.

## Decided
- Strategy pattern: `Trainer` dispatches to any `Strategy` subclass; `execute()` takes no arguments (all state stored in `__init__`).
- Strategy-specific params (e.g. `k_folds`) are constructor arguments, not config fields.
- Patient metrics are computed at runtime from in-memory data — no intermediate CSV I/O in `optimization/`.

## TODO
1. Interpretability hooks: add post-fold callback / result sink so `interpretability/` can consume `TrainingResult` without modifying trainer code.
