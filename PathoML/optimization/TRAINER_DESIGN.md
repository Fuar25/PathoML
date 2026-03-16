# PathoML/optimization/trainer

## 1. Purpose
Strategy-pattern training orchestration. `Trainer` delegates to a `Strategy`; users only interact with `Trainer.fit()`.

## 2. Class Map

```
Trainer
└── strategy: Strategy (ABC)
    ├── CrossValidator(Strategy, TrainingMixin)   — K-fold cross-validation
    └── FullDatasetTrainer(Strategy, TrainingMixin) — full-dataset training for deployment

TrainingMixin       — shared epoch/eval logic (inherited, not used standalone)
FoldTrainer         — runs one fold to convergence (composition, not inheritance)
EarlyStopping       — patience-based stop signal
CheckpointManager   — save/load best checkpoint per fold
```

## 3. API

```python
# CrossValidator
strategy = CrossValidator(
    model_builder: Callable[[], BaseModel],  # factory — called once per fold
    dataset: BaseDataset,
    config: RunTimeConfig,
    k_folds: int = 5,                        # strategy-specific, NOT in config
)
result: TrainingResult = Trainer(strategy).fit()

# FullDatasetTrainer
strategy = FullDatasetTrainer(model_builder, dataset, config)
result: TrainingResult = Trainer(strategy).fit()
```

## 4. TrainingMixin Methods
These become methods of `CrossValidator`/`FullDatasetTrainer` via inheritance (resolved through Python MRO):

| Method | Signature | Notes |
|--------|-----------|-------|
| `_train_epoch` | `(model, loader, optimizer, criterion) -> float` | Returns mean loss |
| `_evaluate_with_auc` | `(model, loader) -> (loss, acc, auc, details)` | Returns tissue-level details dict |
| `_forward_and_decode` | `(model, batch) -> (loss, preds, probs)` | Handles binary and multi-class |
| `_compute_auc` | `(labels, probs, num_classes) -> float` | Handles both AUC types |
| `_build_criterion` | `(num_classes) -> nn.Module` | BCE for binary, CrossEntropy for multi-class |

## 5. FoldTrainer
Receives callable training functions from `CrossValidator`, keeping the two classes decoupled:
```python
fold_trainer = FoldTrainer(model, device, config, checkpoint_manager)
ckpt = fold_trainer.fit(train_loader, val_loader, fold,
                        train_epoch_fn=self._train_epoch,
                        evaluate_fn=self._evaluate_with_auc)
```

## 6. Patient Metrics
`CrossValidator._compute_patient_metrics(test_details)` aggregates tissue-level results to patient level at runtime using `aggregate_patient_predictions()`. No CSV is written here — that belongs to `interpretability/`.

## Decided
- `execute()` takes no arguments: all dependencies are stored in `__init__`, making strategies self-contained and testable.
- `device` is resolved from `config.training.device` in `__init__`, not passed separately.
- `StratifiedGroupKFold` is used so folds are class-balanced while respecting patient-level grouping.

## TODO
1. Preprocessor / Postprocessor Mixin: extract `_move_to_device`, `_model_inputs`, `_split_train_val` into a reusable Mixin once a second strategy type needs them.
2. Interpretability hooks: post-fold callback so `TrainingResult` consumers can trigger CSV/figure output without modifying trainer.
