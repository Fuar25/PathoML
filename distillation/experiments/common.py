"""Shared infrastructure for distillation experiments."""

import os
from datetime import datetime
from pathlib import Path

import numpy as np

from PathoML.config.config import RunTimeConfig
from PathoML.dataset.utils import find_common_sample_keys, fingerprint_sample_keys
from PathoML.optimization.trainer import Trainer

from distillation.dataset import DistillationDataset
from distillation.models.student import StudentTransABMIL
from distillation.runtime import DistillCrossValidator, load_manifest


# ─── Data roots ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_FEAT_ROOT = str(PROJECT_ROOT / 'Features' / 'GigaPath-Patch-Feature')
SLIDE_FEAT_ROOT = str(PROJECT_ROOT / 'Features' / 'GigaPath-Slide-Feature')
LABELS_CSV = str(PROJECT_ROOT / 'Features' / 'labels.csv')


# ─── Default hyperparameters ────────────────────────────────────────────────

EPOCHS     = 100
PATIENCE   = 10
LR         = 1e-4
WD         = 1e-5
BATCH_SIZE = 16
DEVICE     = 'cuda:0'

STUDENT_KWARGS = dict(
  patch_dim=1536, hidden_dim=256, attention_dim=128, dropout=0.2,
  n_transformer_layers=2, nhead=4, proj_dim=128,
)


# ─── Output paths ───────────────────────────────────────────────────────────

OUTPUTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
SHARED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_log.txt')


# ─── Helpers ────────────────────────────────────────────────────────────────

def format_condition_value(value: int | float | str) -> str:
  """Format a value for descriptive condition names.

  Distillation experiment names use lowercase snake case with full words.
  Decimal points are encoded as `p` so output directories remain shell-friendly.
  """
  return str(value).replace('-', 'minus_').replace('.', 'p')


def default_teacher_manifest_path(condition_name: str) -> str:
  """Return the default teacher manifest path for a named teacher condition."""
  return str(PROJECT_ROOT / 'teacher' / 'experiments' / 'outputs' / condition_name / 'manifest.json')


def load_distill_dataset(
  manifest,
  patch_root: str = PATCH_FEAT_ROOT,
  slide_root: str = SLIDE_FEAT_ROOT,
  labels_csv: str = LABELS_CSV,
  intersection_stains: list[str] | None = None,
) -> tuple[DistillationDataset, list[str]]:
  """Build the distillation dataset from a teacher manifest.

  Args:
    slide_root: Root directory for slide-level features.
    intersection_stains: Stains used to compute the shared sample intersection.
      If `None`, use the teacher manifest modalities.

  Returns:
    (dataset, intersection_stains)
  """
  print('Loading dataset...')

  if intersection_stains is None:
    intersection_stains = list(manifest.modality_names)

  # Patch features are HE-only. Slide features follow the manifest modalities.
  patch_keys = find_common_sample_keys(patch_root, ['HE'])
  slide_keys = find_common_sample_keys(slide_root, intersection_stains)
  common_keys = patch_keys & slide_keys
  print(f'  Shared sample count (HE patch ∩ slide({" ∩ ".join(intersection_stains)})): {len(common_keys)}')
  common_fingerprint = fingerprint_sample_keys(common_keys)
  if manifest.sample_set_fingerprint and manifest.sample_set_fingerprint != common_fingerprint:
    raise ValueError(
      "Teacher manifest sample set does not match the distillation dataset intersection. "
      f"manifest={manifest.sample_set_fingerprint}, distillation={common_fingerprint}"
    )

  dataset = DistillationDataset(
    patch_root=patch_root,
    slide_root=slide_root,
    slide_stains=list(manifest.modality_names),
    labels_csv=labels_csv,
    allowed_sample_keys=common_keys,
  )
  print(f'  {len(dataset)} samples, classes: {dataset.classes}')
  return dataset, intersection_stains


def run_distill_cv(
  dataset: DistillationDataset,
  config: RunTimeConfig,
  distill_loss,
  teacher_ckpt_tmpl: str,
  k_folds: int,
  student_kwargs: dict = STUDENT_KWARGS,
  student_builder=None,
) -> tuple[list[float], list[float]]:
  """Run one K-fold distillation CV pass and return `(fold_aucs, fold_f1s)`.

  Args:
    student_builder: Optional student factory. Defaults to `StudentTransABMIL`.
  """
  if student_builder is None:
    student_builder = lambda: StudentTransABMIL(**student_kwargs)
  cv = DistillCrossValidator(
    student_builder   = student_builder,
    dataset           = dataset,
    config            = config,
    distill_loss      = distill_loss,
    teacher_ckpt_tmpl = teacher_ckpt_tmpl,
    k_folds           = k_folds,
  )
  result = Trainer(cv).fit()
  fold_aucs = [f.patient_auc for f in result.fold_results]
  fold_f1s  = [f.patient_f1  for f in result.fold_results]
  return fold_aucs, fold_f1s


def run_condition(
  condition_name: str,
  config: RunTimeConfig,
  distill_loss,
  manifest,
  dataset: DistillationDataset,
  student_kwargs: dict = STUDENT_KWARGS,
  output_dir: str = OUTPUTS_DIR,
  student_builder=None,
) -> dict:
  """Run `manifest.n_runs` CV passes for one named condition.

  Args:
    student_builder: Optional student factory. Defaults to `StudentTransABMIL`.

  Returns:
    dict with keys: run_means, all_fold_aucs, run_f1_means, all_fold_f1s
  """
  print(f'distill_loss: {distill_loss}')

  run_means, all_fold_aucs = [], []
  run_f1_means, all_fold_f1s = [], []

  for i in range(manifest.n_runs):
    seed = manifest.base_seed + i
    run_dir = os.path.join(output_dir, condition_name, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    config.training.seed    = seed
    config.logging.save_dir = run_dir
    tmpl = manifest.ckpt_tmpl.replace('{run:02d}', f'{i:02d}')

    print(f"\n[{condition_name}] Run {i+1}/{manifest.n_runs}  (seed={seed})")

    fold_aucs, fold_f1s = run_distill_cv(
      dataset, config, distill_loss, tmpl, manifest.k_folds,
      student_kwargs, student_builder,
    )

    run_mean = float(np.mean(fold_aucs))
    run_means.append(run_mean)
    all_fold_aucs.extend(fold_aucs)
    run_f1_mean = float(np.mean(fold_f1s))
    run_f1_means.append(run_f1_mean)
    all_fold_f1s.extend(fold_f1s)

    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))
    print(f"  {fold_str}  →  mean={run_mean:.4f}")

  return {
    "run_means": run_means, "all_fold_aucs": all_fold_aucs,
    "run_f1_means": run_f1_means, "all_fold_f1s": all_fold_f1s,
  }


def log_results(
  results: dict[str, dict],
  log_path: str = SHARED_LOG_FILE,
  *,
  config: RunTimeConfig | None = None,
  distill_loss=None,
  manifest=None,
  student_kwargs: dict = STUDENT_KWARGS,
  stains: list[str] | None = None,
) -> None:
  """Append a timestamped AUC/F1 summary table to the shared experiment log."""
  condition_width = 88
  sep  = "=" * 100
  hsep = "─" * 100
  lines = [
    sep,
    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  "
    f"conditions: {', '.join(results.keys())}",
    hsep,
    f"{'condition':<{condition_width}}  {'run-level AUC (mean±std)':<28}  "
    f"{'fold-level AUC (mean±std)':<28}  fold-level F1 (mean±std)",
    hsep,
  ]
  for name, data in results.items():
    run_means = np.array(data["run_means"])
    fold_aucs = np.array(data["all_fold_aucs"])
    fold_f1s  = np.array(data.get("all_fold_f1s", []))
    f1_str = f"{fold_f1s.mean():.4f} ± {fold_f1s.std():.4f}" if len(fold_f1s) > 0 else "N/A"
    lines.append(
      f"{name:<{condition_width}}  "
      f"{run_means.mean():.4f} ± {run_means.std():.4f}              "
      f"{fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}              "
      f"{f1_str}"
    )

  lines.append(hsep)
  if stains:
    lines.append(f"shared sample intersection: {' ∩ '.join(stains)}")
  if manifest:
    lines.append(f"teacher: {manifest.condition_name}")
    lines.append(f"teacher_modalities: {', '.join(manifest.modality_names)}")
    lines.append(
      f"N_RUNS={manifest.n_runs}  K_FOLDS={manifest.k_folds}  "
      f"BASE_SEED={manifest.base_seed}"
    )
  if config:
    t = config.training
    lines.append(
      f"epochs={t.epochs}  patience={t.patience}  "
      f"lr={t.learning_rate}  wd={t.weight_decay}  "
      f"batch_size={t.batch_size}  device={t.device}"
    )
  if distill_loss:
    lines.append(f"distill_loss: {distill_loss}")
  kw_str = "  ".join(f"{k}={v}" for k, v in student_kwargs.items())
  lines.append(f"student: {kw_str}")
  lines.append(sep)
  lines.append("")

  print("\n" + "\n".join(lines))
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"Results appended to: {log_path}")
