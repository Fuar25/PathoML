# PathoML/scripts

## 1. Purpose
Rapid experiment validation, one-off analyses, and ad-hoc deliverables. Scripts call the core library (`data/`, `models/`, `optimization/`) but are **not part of the library itself**.

## 2. Rules
- Scripts may import from `data/`, `models/`, `optimization/`, `config/` freely.
- Core library code must **never** import from `scripts/`.
- Scripts are not versioned as library APIs — they can be modified or deleted without a deprecation notice.
- No shared state between scripts; each script is self-contained.

## 3. Current Contents

| Path | Purpose |
|------|---------|
| `hpo_search.py` | Bayesian hyperparameter search (Optuna) |
| `multi_stain_fusion.py` | Multi-stain feature fusion experiment |
| `custom_voting/` | Voting ensemble post-processing |
| `multimodal_fusion/` | Multi-modal fusion experiments |
| `preprocess/` | Feature extraction and WSI list preparation |
| `utils/` | Shared script utilities |
