# PathoML

## 1. Purpose
Core library for WSI (Whole Slide Image) MIL (Multiple Instance Learning) classification. Provides data loading, model architectures, training orchestration, and configuration — isolated from scripts, experiments, and examples.

## 2. Package Structure

| Package / Module | Responsibility |
|---------|---------------|
| `interfaces.py` | Library-level ABCs: `BaseDataset`, `BaseModel`, `BaseMIL`, `Aggregator`, `Classifier` |
| `registry.py` | Plugin registry and factory: `register_model`, `register_dataset`, `create_model`, `load_all_module` |
| `config/` | Typed runtime configuration dataclasses |
| `dataset/` | Dataset implementations (unimodal, multimodal) |
| `models/` | MIL model architectures (ABMIL, LinearProbe) |
| `optimization/` | Training strategies, patient aggregation |

## 3. Dependency Order
```
interfaces.py            ← no internal deps (base contracts)
registry.py              ← no internal deps
config/                  ← no internal deps
dataset/                 ← config/, interfaces, registry
models/                  ← interfaces, registry
optimization/trainer     ← config/, interfaces, registry
```

## 4. Code Conventions
- Follow Clean Code's best practices as much as possible.


## Decided
- Core library lives in `PathoML/`; scripts, examples, tests remain at project root.
- `interfaces.py` and `registry.py` live at `PathoML/` root (library-level, not optimization-specific).
- Internal cross-package imports use relative paths (e.g. `from ..config.defaults import`).
- External consumers use absolute paths (e.g. `from PathoML.config.config import RunTimeConfig`).
