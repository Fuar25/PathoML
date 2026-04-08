# distillation/CLAUDE.md

Agent guidance for the distillation subsystem.

## 1. Read Order
1. [distillation/DESIGN.md](/home/sbh/PathoML/distillation/DESIGN.md)
2. [distillation/dataset/DESIGN.md](/home/sbh/PathoML/distillation/dataset/DESIGN.md) when editing dataset assembly
3. [distillation/losses/DESIGN.md](/home/sbh/PathoML/distillation/losses/DESIGN.md) when editing losses
4. [distillation/runtime/DESIGN.md](/home/sbh/PathoML/distillation/runtime/DESIGN.md) when editing manifest/runtime code
5. [distillation/experiments/DESIGN.md](/home/sbh/PathoML/distillation/experiments/DESIGN.md)
6. [distillation/experiments/PLAN.md](/home/sbh/PathoML/distillation/experiments/PLAN.md) when current experiment status matters

## 2. Core Boundary
- Distillation may reuse `PathoML` shared primitives and training runtime.
- Distillation must consume teacher artifacts through `manifest.json` and checkpoint metadata.
- Distillation must not depend on teacher experiment internals.

## 3. Editing Rules
- If the teacher artifact contract changes, update `distillation/runtime/DESIGN.md`, `distillation/DESIGN.md`, and the corresponding teacher design docs.
- If the loss interface changes, update `distillation/losses/DESIGN.md`.
- If the distillation dataset item contract changes, update `distillation/dataset/DESIGN.md`.
- If experiment execution flow changes, update `distillation/experiments/DESIGN.md`.
- Keep distillation experiment filenames and condition names descriptive.
- Reuse established subsystem vocabulary. If a family is named `teacher_guided_attention`, do not introduce a parallel synonym such as `attention_guidance`.
- Keep experiment status changes in `distillation/experiments/PLAN.md`.

## 4. Validation
- Import-check `distillation.experiments.common`.
- Validate teacher manifest loading after contract changes.
- Run the relevant `tests/` subset when shared runtime code is touched.
