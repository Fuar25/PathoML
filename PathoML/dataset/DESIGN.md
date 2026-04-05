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

### 3.2 PatchDataset/
Patch-level datasets — each H5 contains N patch feature vectors `(N, D)`. Use with a MIL model.

| Class | Registry key | Output shape |
|-------|-------------|-------------|
| `UnimodalPatchDataset` | `UnimodalPatchDataset` | `(N, D)` |

## 4. Directory Layout
Patient-based directory structure + external CSV labels:
```
data_root/
  <patient_id>/
    <tissue_id>/
      <patient_id><tissue_id>-<stain>.h5
labels.csv   ← patient_id,label
```
Example:
```
GigaPath-Slide-Feature/
  B2022-01475/
    B/
      B2022-01475B-HE.h5
      B2022-01475B-cd20.h5
      B2022-01475B-cd3.h5
```
All scanning is recursive (`os.walk`). Stain filtering is by filename suffix, case/hyphen-insensitive: `_normalize_stain("Ki-67")` = `ki67` matches file `...-ki67.h5`.

CSV format: header row `patient_id,label`, one row per patient. `load_labels_csv()` in `utils.py` parses it.
Class-to-index mapping: classes sorted reverse-alphabetically, so positive class gets index 1. E.g. Reactive=0, MALT=1.

**CSV-driven subsetting**: Patients not in the CSV are automatically excluded. Switch to a different CSV to run on a subset — no file reorganization needed. `PATHOML_LABELS_CSV` environment variable overrides the default CSV path.

## 5. File Naming Convention
H5 files must follow `<patient_id><tissue_id>-<stain>.h5`, e.g. `B2022-01475B-cd20.h5`.
- `patient_id` matched by `PATIENT_ID_PATTERN` (see `config/defaults.py`)
- `tissue_id` is the single alphanumeric character immediately after `patient_id`
- `stain` is the part after the last `-` before `.h5`

## 6. API

### 6.1 Unimodal
```python
UnimodalSlideDataset(
  data_root='/data/GigaPath-Slide-Feature',
  stain='HE',
  labels_csv='labels.csv',
)
```
`stain` is optional — if omitted, all H5 files under `data_root` are included.

### 6.2 Multimodal
```python
MultimodalConcatSlideDataset(
  data_root='/data/GigaPath-Slide-Feature',
  modality_names=['HE', 'CD20', 'CD3'],
  labels_csv='labels.csv',
)
```
Single `data_root` for all modalities. Each modality is filtered by stain name.

### 6.3 Sample intersection
```python
from PathoML.dataset.utils import find_common_sample_keys
common = find_common_sample_keys('/data/GigaPath-Slide-Feature', ['HE', 'CD20', 'CD3'])
```

## 7. Utility functions (`utils.py`)
- `_normalize_stain(name)` — normalize stain name: lowercase + strip hyphens
- `_extract_stain(filename)` — extract normalized stain from H5 filename
- `_walk_h5_files(root, stain=None)` — recursive H5 scanner with optional stain filter
- `_extract_patient_tissue_id(filename, pattern)` — extract (patient_id, tissue_id) from filename
- `load_labels_csv(csv_path)` — load patient_id → class_name mapping
- `find_common_sample_keys(data_root, stains)` — intersection of sample keys across stains
- `_variable_size_collate(batch)` — collate for variable-length tensors with padding + mask

## Decided
- **Sample ordering invariant**: Every dataset implementation MUST sort `self.samples` by `(patient_id, tissue_id)` after scanning. `StratifiedGroupKFold` is order-sensitive — different orderings with the same seed produce different fold splits, making cross-experiment AUC comparison meaningless.
- **`slide_id` key**: All datasets use `slide_id` in `__getitem__` and internal data structures.
- Multimodal datasets (Concat/Fusion) are Slide-level only.
- All modalities are treated symmetrically in multimodal datasets — no anchor modality.
- Missing modalities: zero-padded (Concat) or excluded from weighted mean (Fusion) when `allow_missing_modalities=True`.
- `allowed_sample_keys: Optional[Set[Tuple[str, str]]]` whitelist accepted by all datasets. Use `find_common_sample_keys(data_root, stains)` to compute intersection.
- `_extract_patient_tissue_id` is defined once in `utils.py` and imported by all dataset modules.
- **`labels_csv` parameter**: All dataset classes require `labels_csv: str`. Labels come from the CSV.
- **Patient-based directory structure**: Files organized as `data_root/<patient_id>/<tissue_id>/*.h5`. All scanning is recursive. Stain filtering by filename suffix.

## TODO
1. Interpretability data interface: reserve hooks for attention-map-aligned coordinate export.
