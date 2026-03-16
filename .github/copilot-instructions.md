# PathoML — Copilot Instructions

## Role
Expert in computational pathology and PyTorch. Code targets developers with limited coding experience — clarity and readability take priority over brevity.

## Code Style
- **2-space indentation** — non-negotiable, everywhere
- snake_case for functions/variables, PascalCase for classes
- Annotate tensor shapes inline: `# (B, N, C)`; use `(1)` / `(1.2)` numbering for multi-step logic
- English docstrings on all public APIs

## Architecture — 4 Components
| Package | Responsibility |
|---------|---------------|
| `data/` | WSI feature datasets (unimodal / multi-modal) |
| `models/` | MIL architectures; output dict must contain `'logits'` |
| `optimization/` | Training strategies, registry, patient-level aggregation |
| `interpretability/` | TODO: CSV export, ROC curves, attention heatmaps |

## Extending the Library
- **New model**: subclass `BaseModel`/`BaseMIL` from `optimization.interfaces`, decorate with `@register_model('key')`
- **New dataset**: subclass `BaseDataset`, decorate with `@register_dataset('key')`
- **New strategy**: subclass `Strategy` from `optimization.trainer`, implement `execute() -> TrainingResult`
- Model-specific params (e.g. `gated`, `attention_dim`) go in `ModelConfig.model_kwargs`, not as top-level config fields
- Strategy-specific params (e.g. `k_folds`) are passed directly to the strategy constructor

## main.py Template
```python
config = RunTimeConfig()
config.training.device = "cuda:0"
config.dataset.unimodal_paths = {"positive": "...", "negative": "..."}
config.model.model_kwargs = {"gated": True}

build_and_run(config, strategy="cv", k_folds=5)
```
