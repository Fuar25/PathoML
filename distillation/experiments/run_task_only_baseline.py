"""K-fold entry point for the ABMIL task-only baseline.

This is the non-distilled baseline for the current active line:
fixed teacher artifact + fixed `StudentBasicABMIL` + MIL distillation comparison.
"""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  run_condition, log_results, load_distill_dataset, load_manifest,
)
from distillation.losses import CompositeDistillationLoss, TaskLoss


TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp_bs32')


def make_distill_loss() -> CompositeDistillationLoss:
  return CompositeDistillationLoss([
    TaskLoss(),
  ])


def make_config():
  distill_loss = make_distill_loss()
  condition_name = build_condition_name('task_only_baseline', distill_loss)
  config = build_runtime_config()
  return condition_name, distill_loss, config


def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)
  condition_name, distill_loss, config = make_config()
  results = run_condition(condition_name, config, distill_loss, manifest, dataset)
  log_results(
    {condition_name: results}, config=config, distill_loss=distill_loss,
    manifest=manifest, stains=intersection_names,
  )


if __name__ == '__main__':
  main()
