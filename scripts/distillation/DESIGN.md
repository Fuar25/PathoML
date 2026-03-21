# scripts/distillation

## 1. Purpose
Cross-modal knowledge distillation: transfer multi-modal slide-level teacher knowledge into a unimodal HE patch-level student.

This module is in **growth and testing phase** — it lives under `scripts/` because its architecture is not yet stable enough to be part of the core library. As patterns solidify, stable components may graduate into `PathoML/`.

## 2. Architecture

Teacher (frozen, multi-modal slide embedding → MLP) and student (HE patches → ABMIL) have **different inputs and different architectures**. This is cross-modal distillation, not standard model compression.

```
Teacher: slide_concat (B, D_slide) → TeacherMLP → hidden (B,256) + logit (B,1)
Student: he_patches  (B, N, D_patch) → StudentABMIL → hidden (B,256) + logits (B,1)
```

## 3. Module Structure

| File | Purpose |
|------|---------|
| `losses.py` | `DistillationLoss` ABC + `StandardKDLoss` implementation |
| `trainer.py` | `DistillCrossValidator(CrossValidator)` — accepts any `DistillationLoss` |
| `dataset.py` | `DistillationDataset` — loads HE patches + multi-modal slide embeddings |
| `models/student.py` | `StudentABMIL` — ABMIL on HE patches |
| `models/teacher.py` | `TeacherMLP` — loads frozen PathoML MLP checkpoint |
| `runs/` | Experiment scripts, one per distillation method |

## 4. Extension Point
To add a new distillation method:
1. Implement `DistillationLoss` in `losses.py`
2. Create `runs/run_<method>.py`
3. No changes to `trainer.py`

## 5. Decided
- Teacher checkpoints are fold-specific; `execute()` loads and verifies fold splits each fold.
- `DistillationLoss` is the single extension point for new methods — trainer delegates all loss computation to it.
