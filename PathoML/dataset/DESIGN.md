# PathoML/dataset

## 1. Purpose
WSI feature dataset implementations. Each class converts on-disk H5 files into PyTorch-consumable samples.

## 2. BaseDataset Contract
Defined in `PathoML.optimization.interfaces.BaseDataset`. Every dataset must implement:
- `__len__() -> int`
- `__getitem__(idx) -> dict`
- `get_patient_ids() -> List[str]`
- `.classes: List[str]`

`__getitem__` return dict keys:
- `features` — `(1, D)` for Slide; `(N, D)` for Patch
- `coords` — `(1, 2)` or `(N, 2)`
- `label` — scalar tensor (float for binary, long for multi-class)
- `slide_id` — str (filename without `.h5`)
- `patient_id` — str
- `tissue_id` — str (single alphanumeric character)

## 3. Sub-packages

### 3.1 SlideDataset/
Slide-level datasets — each H5 contains one aggregated feature vector `(1, D)`. No MIL needed downstream.

| Class | Registry key | Output shape |
|-------|-------------|-------------|
| `UnimodalSlideDataset` | `UnimodalSlideDataset` | `(1, D)` |
| `MultimodalConcatSlideDataset` | `MultimodalConcatSlideDataset` | `(1, ΣD_i)` |
| `MultimodalFusionSlideDataset` | `MultimodalFusionSlideDataset` | `(1, D)` |
| `BimodalConcatInteractSlideDataset` | `BimodalConcatInteractSlideDataset` | `(1, 3D)` |

### 3.2 PatchDataset/
Patch-level datasets — each H5 contains N patch feature vectors `(N, D)`. Use with a MIL model.

| Class | Registry key | Output shape |
|-------|-------------|-------------|
| `UnimodalPatchDataset` | `UnimodalPatchDataset` | `(N, D)` |

## 4. Directory Layout
All datasets use a consistent layout:
```
data_path/          ← data_path (Unimodal) or each modality_path value (Multimodal)
  <class_name_1>/
    *.h5
  <class_name_2>/
    *.h5
```
Class names are auto-detected from subdirectory names. H5 files are scanned recursively.

## 5. File Naming Convention
H5 files must follow `<patient_id><tissue_id>-<anything>.h5`, e.g. `B2022-01475B-cd20.h5`.
- `patient_id` matched by `PATIENT_ID_PATTERN` (see `config/defaults.py`)
- `tissue_id` is the single alphanumeric character immediately after `patient_id`

## 6. Registration and Usage
Registry keys equal class names (case-insensitive).

```python
from PathoML.optimization.registry import create_dataset

dataset = create_dataset("UnimodalPatchDataset", data_path="/data/root")
dataset = create_dataset("MultimodalConcatSlideDataset",
    modality_paths={"HE": "/data/HE", "CD20": "/data/CD20"},
    modality_names=["HE", "CD20"])
```

Or import directly:
```python
from PathoML.dataset import UnimodalPatchDataset
dataset = UnimodalPatchDataset(data_path="/data/root")
```

## Decided
- Multimodal datasets (Concat/Fusion/BimodalConcatInteract) are Slide-level only.
- `BimodalConcatInteractSlideDataset` requires exactly 2 modalities with equal feature dim.
- All modalities are treated symmetrically in multimodal datasets — no anchor modality.
- Missing modalities: zero-padded (Concat/Interact) or excluded from weighted mean (Fusion) when `allow_missing_modalities=True`.
- `allowed_sample_keys: Optional[Set[Tuple[str, str]]]` whitelist accepted by all datasets. Use `find_common_sample_keys(dirs)` to compute intersection across modality roots.
- `_extract_patient_tissue_id` is defined once in `utils.py` and imported by all dataset modules.

## TODO
1. Interpretability data interface: reserve hooks for attention-map-aligned coordinate export.
