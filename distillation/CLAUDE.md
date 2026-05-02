# distillation/CLAUDE.md

Agent guidance for the distillation subsystem.

## 1. Read Order
1. [distillation/DESIGN.md](/home/sbh/PathoML/distillation/DESIGN.md)
2. [distillation/dataset/DESIGN.md](/home/sbh/PathoML/distillation/dataset/DESIGN.md) for dataset assembly edits
3. [distillation/losses/DESIGN.md](/home/sbh/PathoML/distillation/losses/DESIGN.md) for loss edits
4. [distillation/runtime/DESIGN.md](/home/sbh/PathoML/distillation/runtime/DESIGN.md) for manifest/runtime edits
5. [distillation/experiments/DESIGN.md](/home/sbh/PathoML/distillation/experiments/DESIGN.md)
6. [distillation/experiments/PLAN.md](/home/sbh/PathoML/distillation/experiments/PLAN.md) when experiment status matters

## 2. Core Boundary
- Reuse shared `PathoML` primitives and training runtime where applicable.
- Consume teacher artifacts through `manifest.json` and checkpoint metadata.
- Do not depend on teacher experiment internals.

## 3. Editing Rules
- If the teacher artifact contract changes, update `distillation/runtime/DESIGN.md`, `distillation/DESIGN.md`, and corresponding teacher design docs.
- If the loss interface changes, update `distillation/losses/DESIGN.md`.
- If the dataset item contract changes, update `distillation/dataset/DESIGN.md`.
- If experiment execution flow changes, update `distillation/experiments/DESIGN.md`.
- Keep experiment filenames and machine IDs descriptive.
- Reuse established subsystem vocabulary. If a family is `teacher_guided_attention`, do not introduce a synonym such as `attention_guidance`.
- Keep experiment status updates in `distillation/experiments/PLAN.md`.

## 4. Validation
- Import-check `distillation.experiments.common`.
- Validate teacher manifest loading after contract changes.
- Run relevant `tests/` subset when shared runtime code is touched.
