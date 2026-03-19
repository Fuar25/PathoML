# PathoML/scripts

## 1. Purpose
Rapid experiment validation, one-off analyses, and ad-hoc deliverables. Scripts call the core library (`data/`, `models/`, `optimization/`) but are **not part of the library itself**. The file itself should be independent and self-contained, with no shared state or dependencies on other scripts.

## 2. Rules
- Scripts may import from `data/`, `models/`, `optimization/`, `config/` freely.
- Core library code must **never** import from `scripts/`.
- Scripts are not versioned as library APIs — they can be modified or deleted without a deprecation notice.
- No shared state between scripts; each script is self-contained.
- Each script should have a clear description of its purpose and pseudocode indicating its workflow based on the core library APIs on the top of the file by comments.
- There will be no unit test so the code itself should be as clear and readable as possible, following best practices for code clarity and maintainability such as Clean Code principles.
- Main Library may update APIs fast, so scripts should add some necessary abstracted wrapper to decouple from the main library as possible.

## 3. Current Contents

| Path | Purpose |
|------|---------|
| `hpo_search.py` | Bayesian hyperparameter search (Optuna) |
| `multi_stain_fusion.py` | Multi-stain feature fusion experiment |
| `custom_voting/` | Voting ensemble post-processing |
| `multimodal_fusion/` | Multi-modal fusion experiments |
| `preprocess/` | Feature extraction and WSI list preparation |
| `utils/` | Shared script utilities |
