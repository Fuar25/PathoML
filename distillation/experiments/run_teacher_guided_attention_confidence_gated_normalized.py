"""K-fold entry point for normalized confidence-gated cosine-logit TGA."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  run_condition, log_results, load_distill_dataset, load_manifest,
)
from distillation.losses import (
  CompositeDistillationLoss,
  ConfidenceGatedCosineAttentionLogitLoss,
  TaskLoss,
)

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp_bs32_lr4em4')

MIN_CONFIDENCE = 0.0


def make_distill_loss() -> CompositeDistillationLoss:
  return CompositeDistillationLoss([
    TaskLoss(),
    ConfidenceGatedCosineAttentionLogitLoss(
      detach_target_encoded=False,
      min_confidence=MIN_CONFIDENCE,
      normalize_by_gate=True,
    ),
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
