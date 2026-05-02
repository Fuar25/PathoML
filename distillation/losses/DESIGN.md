# distillation/losses

## 1. Purpose
Provide the stable extension-point package for distillation losses.

## 2. Scope / Owns
- Own:
- `DistillationLoss`
- `DistillationTerm`
- `CompositeDistillationLoss`
- atomic distillation loss terms
- loss-local helper functions

## 3. Public Contracts
- `DistillationLoss`
- `DistillationTerm`
- `CompositeDistillationLoss`
- `WeightedTerm`
- `TaskLoss`
- `HiddenLoss`
- `SimilarityPreservingLoss`
- `SoftLabelLoss`
- `DecoupledKnowledgeDistillationLoss`
- `RKDDistanceLoss`
- `RKDAngleLoss`
- `CosineAttentionLogitLoss`
- `ClassAwareCosineAttentionLogitLoss`
- `ClassAwareAttentionRankMarginLoss`
- `ConfidenceGatedCosineAttentionLogitLoss`
- `CosineAttentionRankLoss`
- `TopKCosineAttentionLogitLoss`
- `SoftDistributionAttentionLoss`
- `BatchContrastiveAttentionLoss`

## 4. Invariants
- Build methods from explicit atomic terms.
- Add loss classes without trainer edits.
- Keep `L_task` explicit; do not inject it in composite loss.
- Derive human-readable formulas and condition slugs from active terms.
- Respect bag masks in attention-target terms.
- Keep attention-target terms finite on edge cases (`B=1`, single-instance bag, constant bag).
- Encode `detach/no-detach` variants explicitly in `describe()` and `slug()`.
- Encode confidence-gated variants explicitly in `describe()` and `slug()`.
- Class-aware attention terms require teacher output field `class_weight`.
- Rank-margin attention terms compare teacher-selected top and bottom valid patches.

## 5. Change Rules
- Add reusable behavior as atomic terms before adding family wrappers.
- Update this file when student or teacher output contracts change.

## Decided
- Keep `losses` as a first-class package
- Use `DistillationTerm + CompositeDistillationLoss` as the primary extension point
- Keep legacy family wrappers only for migration or compatibility

## TODO
1. Reintroduce non-mainline historical terms (for example mean-bypass) only if they become active again.
