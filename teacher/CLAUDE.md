# teacher/CLAUDE.md

Agent guidance for the teacher-selection subsystem.

## 1. Read Order
1. [teacher/DESIGN.md](/home/sbh/PathoML/teacher/DESIGN.md)
2. [teacher/experiments/DESIGN.md](/home/sbh/PathoML/teacher/experiments/DESIGN.md)
3. [teacher/experiments/PLAN.md](/home/sbh/PathoML/teacher/experiments/PLAN.md) when current experiment status matters

## 2. What This Subsystem Owns
- Teacher concrete datasets
- Teacher concrete models
- Teacher experiment entry scripts
- Teacher artifact generation (`manifest.json` + checkpoint metadata)

## 3. What It Must Not Own
- Shared training strategies
- Shared dataset utilities
- Shared model primitives
- Distillation internals

Those belong to `PathoML/` or `distillation/`.

## 4. Editing Rules
- If you change teacher artifact fields, update `teacher/DESIGN.md`, `teacher/experiments/DESIGN.md`, and `distillation/runtime/DESIGN.md`.
- If you change experiment naming, outputs, or comparability rules, update `teacher/experiments/DESIGN.md`.
- If you change current experiment status, update `teacher/experiments/PLAN.md`, not `teacher/DESIGN.md`.

## 5. Validation
- Run the relevant `tests/` subset for shared runtime impact.
- Import-check a representative `teacher.experiments.run_*` module after changing experiment entry code.
