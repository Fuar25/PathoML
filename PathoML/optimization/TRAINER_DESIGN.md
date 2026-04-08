# PathoML/optimization/trainer

## 1. Purpose
Concrete training runtime structure behind the shared optimization layer.

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
- `Trainer` is only a thin dispatcher.
- `TrainingMixin` contains the shared epoch/evaluation logic.
- `CrossValidator` owns fold orchestration and writes fold checkpoints plus `cv_predictions.csv`.
- Fold checkpoints include fold membership metadata for downstream verification.

## 5. Change Rules
- Keep shared training logic in `TrainingMixin` unless it becomes subsystem-specific.
- Add new strategy-level behaviors through strategy composition or lightweight extension points before copying the runtime.
- If checkpoint metadata shape changes, update the consuming subsystem docs.

## Decided
- `CrossValidator` remains the shared backbone for both teacher and distillation.
- Strategy instances may attach extra checkpoint metadata without changing the trainer entry API.

## TODO
1. Add formal post-run hooks only when multiple subsystems need them.
