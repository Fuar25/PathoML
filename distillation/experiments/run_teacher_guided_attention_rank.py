"""K-fold entry point for cosine-rank teacher-guided attention distillation.

This is the first constrained variant under the `teacher_guided_attention`
family on the fixed `StudentBasicABMIL` platform. It replaces the
cosine-logit regression target with cosine-derived pairwise rank
supervision while keeping the rest of the protocol fixed.
"""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  run_condition, log_results, load_distill_dataset, load_manifest,
)
from distillation.losses import (
  CompositeDistillationLoss,
  CosineAttentionRankLoss,
  TaskLoss,
)

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp_bs32')


def make_distill_loss() -> CompositeDistillationLoss:
  return CompositeDistillationLoss([
    TaskLoss(),
    CosineAttentionRankLoss(),
  ])


def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)
  distill_loss = make_distill_loss()
  cond_name = build_condition_name('teacher_guided_attention', distill_loss)
  print(f"\n{'#'*70}\n# {cond_name}  {distill_loss.describe()}\n{'#'*70}")
  config = build_runtime_config()
  results = run_condition(cond_name, config, distill_loss, manifest, dataset)
  log_results(
    {cond_name: results}, config=config, distill_loss=distill_loss,
    manifest=manifest, stains=intersection_names,
  )


if __name__ == '__main__':
  main()
