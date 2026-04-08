# distillation/losses

## 1. Purpose
Provide the stable extension-point package for distillation losses.

## 2. Scope / Owns
This package owns:
- `DistillationLoss`
- the concrete distillation loss families
- loss-local helper functions

## 3. Public Contracts
- `DistillationLoss`
- `StandardKDLoss`
- `RKDLoss`
- `TeacherGuidedAttnLoss`
- `RelationalTGALoss`

## 4. Invariants
- Loss remains the single extension point for adding new distillation methods.
- Trainer code should not need edits when adding a new loss class.
- All current losses include `L_task` as the base supervision term.

## 5. Change Rules
- Add new loss families as new modules inside this package.
- Update this file if the required student/teacher output contract changes.

## Decided
- `losses` is a first-class package because it is expected to keep growing.
- Attention-guided losses stay separate from KD and relational losses.

## TODO
1. Rebalance internal modules if a future loss family grows large enough to justify its own subpackage.
