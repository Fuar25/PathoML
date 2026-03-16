# PathoML/data

## 1. Purpose
WSI feature dataset implementations. Each class converts on-disk H5 files into PyTorch-consumable samples.

## 2. BaseDataset Contract
Defined in `PathoML.optimization.interfaces.BaseDataset`. Every dataset must implement:
- `__len__() -> int`
- `__getitem__(idx) -> dict` with keys: `features (N,D)`, `coords (N,2)`, `label`, `sample_id`, `patient_id`
- `get_patient_ids() -> List[str]`
- `.classes: List[str]`, `.data: List[dict]`

## 3. Dataset Strategies

| Class | Registry key | Feature shape | Use case |
|-------|-------------|---------------|----------|
| `UnimodalDataset` | `wsi_h5` | `(N, D)` | Single stain/modality |
| `MultimodalConcatDataset` | `multimodal_concat` | `(N, ΣD_i)` | Multi-stain; model learns cross-modal mapping |
| `MultimodalFusionDataset` | `multimodal_fusion` | `(N, D)` | Multi-stain; weighted mean preserves original dim |

Choosing between Concat and Fusion:
- **Concat**: expressive — the model sees all modalities jointly; requires a model whose `input_dim = ΣD_i`.
- **Fusion**: lightweight — treats modalities as interchangeable views; model `input_dim` stays `D`.

## 4. Registration and Usage
```python
# Datasets self-register on import via @register_dataset decorator.
# Instantiate through the registry:
from PathoML.optimization.registry import create_dataset

dataset = create_dataset("wsi_h5", data_paths={"positive": "...", "negative": "..."})
dataset = create_dataset("multimodal_concat",
    modality_paths={"HE": "...", "CD20": "..."},
    modality_names=["HE", "CD20"])
```

Or import directly:
```python
from PathoML.data.unimodal_dataset import UnimodalDataset
dataset = UnimodalDataset(data_paths={"positive": "...", "negative": "..."})
```

## 5. File Naming Convention
H5 files must be named `<patient_id><tissue_id>-<anything>.h5`, e.g. `B2022-01475B-cd20.h5`.
- `patient_id` matched by `PATIENT_ID_PATTERN` (see `config/defaults.py`)
- `tissue_id` is the single alphanumeric character immediately after `patient_id`

## Decided
- Missing modalities are zero-padded (Concat) or excluded from weighted mean (Fusion) when `allow_missing_modalities=True`.
- Patch count alignment uses `min(N_i)` across modalities to avoid jagged tensors.

## TODO
1. Interpretability data interface: reserve hooks for attention-map-aligned coordinate export.
