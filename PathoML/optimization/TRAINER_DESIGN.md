# PathoML/optimization/trainer

## 1. Purpose
Define concrete training runtime structure behind the shared optimization layer.

## 2. Scope / Owns
This module family owns:
- `Trainer`
- `TrainingMixin`
- `Strategy`
- `CrossValidator`
- `FullDatasetTrainer`
- fold result containers

## 3. Public Contracts
```python
strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
result = Trainer(strategy).fit()
```

```python
strategy = FullDatasetTrainer(model_builder, dataset, config)
result = Trainer(strategy).fit()
```

`CrossValidator` accepts optional `checkpoint_metadata` through the strategy instance before `fit()`.

## 4. Invariants
- Keep `Trainer` as a thin dispatcher.
- Keep shared epoch/evaluation logic in `TrainingMixin`.
- Keep early-stopping configurable via shared `TrainingConfig` (`val_auc` / `patient_f1`, with `patience` and `min_delta`).
- Keep `CrossValidator` responsible for fold orchestration and writing fold checkpoints plus `cv_predictions.csv`.
- Keep fold-membership metadata in checkpoints for downstream verification.

## 5. Change Rules
- Keep shared training logic in `TrainingMixin` unless it becomes subsystem-specific.
- Add strategy-level behavior through composition or lightweight extension points before copying runtime.
- If checkpoint metadata shape changes, update consuming subsystem docs.

## Decided
- `CrossValidator` remains shared backbone for teacher and distillation.
- Strategy instances may attach extra checkpoint metadata without changing trainer entry API.

## TODO
1. Add formal post-run hooks only when multiple subsystems need them.
