# PathoML/optimization

## 1. Purpose
Training orchestration: strategy pattern entry point, and patient-level prediction aggregation.

## 2. Sub-modules

| Module | Responsibility |
|--------|---------------|
| `training_utils.py` | `TrainingResult`, `EarlyStopping`, `CheckpointManager` |
| `training_base.py` | `TrainingMixin` (shared epoch/eval logic), `Strategy` ABC |
| `trainer.py` | `Trainer` dispatcher + backward-compatible re-exports |
| `TrainingStrategy/` | `CrossValidator`, `FullDatasetTrainer` (concrete strategies) |
| `patient_aggregation.py` | `aggregate_patient_predictions()` — tissue→patient aggregation (max-probability MIL rule) |

Note: `interfaces.py` and `registry.py` have been promoted to `PathoML/` root (library-level).

## 3. Entry Point
```python
from PathoML.optimization.trainer import CrossValidator, Trainer

strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
Trainer(strategy).fit()   # returns TrainingResult
```

## 4. Registry
`load_all_module(config)` (now in `PathoML.registry`) must be called before any `create_model` / `create_dataset` call. It:
1. Auto-imports all built-in modules.
2. Imports any user extension modules listed in `config.dataset.dataset_module_paths` / `config.model.model_module_paths`.

## Decided
- Strategy pattern: `Trainer` dispatches to any `Strategy` subclass; `execute()` takes no arguments (all state stored in `__init__`).
- Strategy-specific params (e.g. `k_folds`) are constructor arguments, not config fields.
- Patient metrics are computed at runtime from in-memory data — no intermediate CSV I/O in training loop.
- `CrossValidator` saves combined test-set predictions (`cv_predictions.csv`) after all folds complete.

## TODO
1. Interpretability hooks: add post-fold callback / result sink so `interpretability/` can consume `TrainingResult` without modifying trainer code.
