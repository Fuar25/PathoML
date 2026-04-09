# distillation/losses

## 1. Purpose
Provide the stable extension-point package for distillation losses.

## 2. Scope / Owns
This package owns:
- `DistillationLoss`
- `DistillationTerm`
- `CompositeDistillationLoss`
- the atomic distillation loss terms
- loss-local helper functions

## 3. Public Contracts
- `DistillationLoss`
- `DistillationTerm`
- `CompositeDistillationLoss`
- `TaskLoss`
- `HiddenLoss`
- `SoftLabelLoss`
- `RKDDistanceLoss`
- `RKDAngleLoss`
- `CosineAttentionLogitLoss`
- `DiscriminationAttentionLogitLoss`
- `ContrastiveTeacherDiscriminationLoss`

## 4. Invariants
- Distillation methods are built by composing explicit atomic terms.
- Trainer code should not need edits when adding a new loss class.
- `L_task` is explicit; it is not implicitly injected by the composite loss.
- Human-readable loss formulas and condition-name slugs are derived from the active terms.

## 5. Change Rules
- Add new reusable behaviors as new atomic terms before introducing a new family wrapper.
- Update this file if the required student/teacher output contract changes.

## Decided
- `losses` is a first-class package because it is expected to keep growing.
- The primary extension point is now `DistillationTerm + CompositeDistillationLoss`.
- Legacy family-style wrappers may exist for migration or compatibility, but experiment scripts should use explicit term composition.

## TODO
1. Reintroduce non-mainline historical terms such as mean-bypass only if they become active experiment designs again.
