"""Confirmation rerun for relational knowledge distillation."""

import os

from distillation.experiments.common import (
  build_condition_name,
  build_runtime_config,
  default_teacher_manifest_path,
  load_distill_dataset,
  load_manifest,
  log_results,
)
from distillation.experiments.confirmation import run_seeded_student_condition
from distillation.losses import (
  CompositeDistillationLoss,
  RKDAngleLoss,
  RKDDistanceLoss,
  TaskLoss,
  WeightedTerm,
)


TEACHER_MANIFEST = default_teacher_manifest_path(
  'run_concat_HE_CD20_CD3_mlp_bs32_lr4em4'
)

GAMMA_D = 1.0
GAMMA_A = 2.0
CONFIRM_STUDENT_BASE_SEED = int(
  os.environ.get(
    'PATHOML_CONFIRM_STUDENT_BASE_SEED',
    os.environ.get('PATHOML_CONFIRM_BASE_SEED', '142'),
  )
)


def make_distill_loss() -> CompositeDistillationLoss:
  return CompositeDistillationLoss([
    TaskLoss(),
    WeightedTerm(RKDDistanceLoss(), GAMMA_D),
    WeightedTerm(RKDAngleLoss(), GAMMA_A),
  ])


def main() -> None:
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  distill_loss = make_distill_loss()
  condition_name = build_condition_name(
    'rkd',
    distill_loss,
    extra_tags=[f'confirm_student_seed{CONFIRM_STUDENT_BASE_SEED}'],
  )
  config = build_runtime_config()
  results = run_seeded_student_condition(
    condition_name,
    config,
    distill_loss,
    manifest,
    dataset,
    student_base_seed=CONFIRM_STUDENT_BASE_SEED,
  )
  log_results(
    {condition_name: results},
    config=config,
    distill_loss=distill_loss,
    manifest=manifest,
    stains=intersection_names,
  )


if __name__ == '__main__':
  main()
