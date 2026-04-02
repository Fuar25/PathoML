# PathoML/optimization/trainer

## 1. Purpose
Strategy-pattern training orchestration. `Trainer` delegates to a `Strategy`; users only interact with `Trainer.fit()`.

## 2. File Layout

| File | Contents |
|------|----------|
| `training_utils.py` | `TrainingResult`, `EarlyStopping`, `CheckpointManager` |
| `training_base.py` | `TrainingMixin` (shared epoch/eval logic), `Strategy` ABC |
| `trainer.py` | `Trainer` dispatcher + backward-compatible re-exports |
| `TrainingStrategy/cross_validator.py` | `FoldResult`, `FoldTrainer`, `CrossValidator` |
| `TrainingStrategy/full_dataset_trainer.py` | `FullDatasetTrainer` |

## 3. Class Map

```
Trainer
└── strategy: Strategy (ABC)  [training_base.py]
    ├── CrossValidator(Strategy, TrainingMixin)     — K-fold cross-validation
    └── FullDatasetTrainer(Strategy, TrainingMixin) — full-dataset training for deployment

TrainingMixin     — training loop + evaluation (4 methods)  [training_base.py]
FoldTrainer       — runs one fold to convergence (composition, not inheritance)
FoldResult        — fold-specific result container
EarlyStopping     — patience-based stop signal (monitors val_auc, higher is better)
CheckpointManager — save/load best checkpoint per fold
```

## 4. API

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

## 5. TrainingMixin Methods

TrainingMixin retains only training loop and evaluation methods. Build/batch/math helpers are standalone functions in `training_utils.py`.

**Mixin methods** (require `self.device`, `self.num_classes`, `self.training_cfg`):

| Method | Notes |
|--------|-------|
| `_run_train_val_loop(...)` | Epoch loop with early stopping and optional cosine scheduler |
| `_train_epoch(model, loader, criterion, optimizer)` | Returns (avg_loss, accuracy) |
| `_evaluate_with_auc(model, loader, criterion)` | Returns (loss, acc, auc, details) |
| `_compute_patient_metrics(eval_details)` | Returns (patient_acc, patient_auc, patient_f1) |

**Standalone functions** in `training_utils.py`:

| Function | Notes |
|----------|-------|
| `set_seed(seed)` | Fix torch/CUDA RNG |
| `build_criterion(num_classes)` | BCE for binary, CrossEntropy for multi-class |
| `build_optimizer(model, training_cfg)` | Adam from config hyperparameters |
| `build_loaders(dataset, train_ids, val_ids, test_ids, training_cfg)` | DataLoaders with variable-size collate |
| `split_train_val(dataset, indices, patient_ids, seed)` | Patient-level stratified 9:1 split |
| `move_to_device(batch, device)` | Move tensor values in batch dict |
| `model_inputs(batch)` | Strip non-model keys (labels, IDs, paths) |
| `forward_and_decode(logits, labels, criterion, num_classes, threshold)` | Binary/multi-class loss+prob+pred |
| `compute_auc(labels, probs, num_classes)` | AUC; returns nan on failure |

**Subclass requirements**: must set `self.device`, `self.num_classes`, `self.training_cfg` in `__init__`.

## 6. FoldTrainer
Receives callable training functions from `CrossValidator`, keeping the two classes decoupled:
```python
fold_trainer = FoldTrainer(model, criterion, optimizer, early_stopping, checkpoint_manager, config)
ckpt = fold_trainer.fit(train_loader, val_loader, fold,
                        train_epoch_fn=self._train_epoch,
                        evaluate_fn=self._evaluate_with_auc)
```

## 7. Patient Metrics and CV Predictions
`TrainingMixin._compute_patient_metrics(test_details)` aggregates tissue-level results to patient level at runtime using `aggregate_patient_predictions()`, returning `(patient_acc, patient_auc, patient_f1)`. No CSV is written per-fold.

After all folds complete, `_save_cv_predictions(all_test_details)` concatenates all test-fold predictions and saves `{save_dir}/cv_predictions.csv` with columns:
- `slide_id`, `patient_id`, `slide_label`, `slide_prob`, `slide_pred`
- `patient_label`, `patient_prob`, `patient_pred`
(Multi-class: prob columns expand to `slide_prob_class_*`, `patient_prob_class_*`.)

## Decided
- `execute()` takes no arguments: all dependencies are stored in `__init__`, making strategies self-contained and testable.
- `device` is resolved from `config.training.device` in `__init__`, not passed separately.
- `StratifiedGroupKFold` is used so folds are class-balanced while respecting patient-level grouping.
- `move_to_device` and `model_inputs` are standalone functions in `training_utils.py`.
- Binary threshold is `self.training_cfg.patient_threshold`; no hardcoded class variable.
- `FoldResult` and `FoldTrainer` live in `cross_validator.py` (fold-specific concepts).
- Early stopping monitors `val_auc` (higher is better), not `val_loss`.

## TODO
1. Interpretability hooks: post-fold callback so `TrainingResult` consumers can trigger CSV/figure output without modifying trainer.
