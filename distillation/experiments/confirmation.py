"""Helpers for confirmation reruns with fixed teacher splits."""

from __future__ import annotations

import os

import numpy as np

from PathoML.optimization.training_utils import set_seed
from distillation.experiments.common import (
  OUTPUTS_DIR,
  STUDENT_KWARGS,
  run_distill_cv,
)
from distillation.models.student import StudentBasicABMIL


def make_seeded_student_builder(run_student_seed: int, student_kwargs: dict):
  """Build students from a seed stream without changing CV split seeds."""
  fold_offset = {'value': 0}

  def build_student():
    fold_seed = run_student_seed + fold_offset['value']
    fold_offset['value'] += 1
    set_seed(fold_seed)
    return StudentBasicABMIL(**student_kwargs)

  return build_student


def run_seeded_student_condition(
  condition_name: str,
  config,
  distill_loss,
  manifest,
  dataset,
  *,
  student_base_seed: int,
  student_kwargs: dict = STUDENT_KWARGS,
  output_dir: str = OUTPUTS_DIR,
) -> dict:
  """Run confirmation CV with teacher split seeds and separate student seeds."""
  print(f'loss_design: {distill_loss.describe()}')
  print(f'confirmation_student_base_seed: {student_base_seed}')

  run_means, all_fold_aucs = [], []
  run_f1_means, all_fold_f1s = [], []

  for i in range(manifest.n_runs):
    split_seed = manifest.base_seed + i
    run_student_seed = student_base_seed + i * manifest.k_folds
    run_dir = os.path.join(output_dir, condition_name, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    config.training.seed = split_seed
    config.logging.save_dir = run_dir
    tmpl = manifest.ckpt_tmpl.replace('{run:02d}', f'{i:02d}')
    student_builder = make_seeded_student_builder(run_student_seed, student_kwargs)

    print(
      f"\n[{condition_name}] Run {i+1}/{manifest.n_runs}  "
      f"(split_seed={split_seed}, student_seed_start={run_student_seed})"
    )

    fold_aucs, fold_f1s = run_distill_cv(
      dataset,
      config,
      distill_loss,
      tmpl,
      manifest.k_folds,
      student_kwargs,
      student_builder,
    )

    run_mean = float(np.mean(fold_aucs))
    run_means.append(run_mean)
    all_fold_aucs.extend(fold_aucs)
    run_f1_mean = float(np.mean(fold_f1s))
    run_f1_means.append(run_f1_mean)
    all_fold_f1s.extend(fold_f1s)

    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))
    print(f"  {fold_str}  ->  mean={run_mean:.4f}")

  return {
    "run_means": run_means,
    "all_fold_aucs": all_fold_aucs,
    "run_f1_means": run_f1_means,
    "all_fold_f1s": all_fold_f1s,
  }
