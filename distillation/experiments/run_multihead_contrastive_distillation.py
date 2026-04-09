"""K-fold entry point for multihead contrastive distillation."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  STUDENT_KWARGS,
)
from distillation.losses import (
  CompositeDistillationLoss,
  ContrastiveTeacherDiscriminationLoss,
  TaskLoss,
  WeightedTerm,
)
from distillation.models.student import StudentTransABMIL_MH


# =============================================================================
# 配置区
# =============================================================================

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

MH_STUDENT_KWARGS = dict(
  **STUDENT_KWARGS,
  pool_heads=4,
)

CONDITIONS = [
  (
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      ContrastiveTeacherDiscriminationLoss(tau=1.0),
    ]),
  ),
  (
    lambda: CompositeDistillationLoss([
      TaskLoss(),
      WeightedTerm(ContrastiveTeacherDiscriminationLoss(tau=1.0), 0.1),
    ]),
  ),
]


# =============================================================================
# 主流程
# =============================================================================

def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  extra_tags = [
    f"pool_heads{format_condition_value(MH_STUDENT_KWARGS['pool_heads'])}",
  ]
  for build_loss in CONDITIONS:
    distill_loss = build_loss()
    cond_name = build_condition_name(
      'multihead_rtga',
      distill_loss,
      extra_tags=extra_tags,
    )
    print(f"\n{'#'*70}\n# {cond_name}  {distill_loss.describe()}\n{'#'*70}")
    config = build_runtime_config()
    results = run_condition(
      cond_name, config, distill_loss, manifest, dataset,
      student_builder=lambda: StudentTransABMIL_MH(**MH_STUDENT_KWARGS),
    )
    log_results(
      {cond_name: results}, config=config, distill_loss=distill_loss,
      manifest=manifest, stains=intersection_names,
      student_kwargs=MH_STUDENT_KWARGS,
    )


if __name__ == '__main__':
  main()
