# PathoML

## 1. Purpose
Core library for WSI (Whole Slide Image) MIL (Multiple Instance Learning) classification. Provides data loading, model architectures, training orchestration, and configuration — isolated from scripts, experiments, and examples.

## 2. Package Structure

| Package | Responsibility |
|---------|---------------|
| `config/` | Typed runtime configuration dataclasses |
| `data/` | Dataset implementations (unimodal, multimodal) |
| `models/` | MIL model architectures (ABMIL, LinearProbe) |
| `optimization/` | Training strategies, registry, patient aggregation |

## 3. Dependency Order
```
optimization/interfaces  ← no internal deps (base contracts)
optimization/registry    ← no internal deps
config/                  ← no internal deps
data/                    ← config/, optimization/interfaces, optimization/registry
models/                  ← optimization/interfaces, optimization/registry
optimization/trainer     ← config/, optimization/interfaces, optimization/registry
```

## 4. Usage
```python
from PathoML.config.config import RunTimeConfig
from PathoML.optimization.registry import create_model, create_dataset, load_runtime_plugins
from PathoML.optimization.trainer import CrossValidator, Trainer

config = RunTimeConfig()
config.dataset.dataset_name = "wsi_h5"
config.dataset.dataset_kwargs["data_paths"] = {"positive": "...", "negative": "..."}
config.model.model_name = "abmil"

load_runtime_plugins(config)
dataset = create_dataset(config.dataset.dataset_name, **config.dataset.dataset_kwargs)
strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
Trainer(strategy).fit()
```

## Decided
- Core library lives in `PathoML/`; scripts, examples, tests remain at project root.
- Internal cross-package imports use relative paths (e.g. `from ..config.defaults import`).
- External consumers use absolute paths (e.g. `from PathoML.config.config import RunTimeConfig`).
