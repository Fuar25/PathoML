"""K-fold entry point for teacher-guided attention distillation.

`teacher_guided_attention` now refers to the corrected logits-space
implementation only. Earlier softmax-space experiments are treated as
historical mistakes and are not runnable from active entry points.
"""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  run_condition, log_results, load_distill_dataset, load_manifest,
)
from distillation.losses import (
  CompositeDistillationLoss,
  ContrastiveTeacherDiscriminationLoss,
  CosineAttentionLogitLoss,
  DiscriminationAttentionLogitLoss,
  TaskLoss,
  WeightedTerm,
)

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

CONDITIONS = [
  (
    'tga',
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      CosineAttentionLogitLoss(),
    ]),
  ),
  (
    'rtga',
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      DiscriminationAttentionLogitLoss(),
    ]),
  ),
  (
    'rtga',
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      ContrastiveTeacherDiscriminationLoss(tau=1.0),
    ]),
  ),
  (
    'rtga',
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      DiscriminationAttentionLogitLoss(),
      ContrastiveTeacherDiscriminationLoss(tau=1.0),
    ]),
  ),
  (
    'rtga',
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      DiscriminationAttentionLogitLoss(),
      WeightedTerm(ContrastiveTeacherDiscriminationLoss(tau=1.0), 0.1),
    ]),
  ),
]


def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  for family_prefix, build_loss in CONDITIONS:
    distill_loss = build_loss()
    cond_name = build_condition_name(family_prefix, distill_loss)
    print(f"\n{'#'*70}\n# {cond_name}  {distill_loss.describe()}\n{'#'*70}")
    config = build_runtime_config()
    results = run_condition(cond_name, config, distill_loss, manifest, dataset)
    log_results(
      {cond_name: results}, config=config, distill_loss=distill_loss,
      manifest=manifest, stains=intersection_names,
    )


if __name__ == '__main__':
  main()
