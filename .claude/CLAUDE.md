# CLAUDE.md

Agent guidance for this repository.

## 1. Document Roles
- `DESIGN.md`: stable facts, ownership, interfaces, invariants, and architectural decisions
- `CLAUDE.md`: agent-only guidance, reading order, task entry points, and validation rules
- `PLAN.md`: experiment status, results, and next steps only

Do not put experiment status into `DESIGN.md`.
Do not put architecture changes into `PLAN.md`.

## 2. Repository Shape
- `PathoML/`: shared pathology foundation
- `teacher/`: teacher-selection subsystem
- `distillation/`: distillation subsystem
- `scripts/`: misc tooling only

`teacher` and `distillation` are peer subsystems. They coordinate through teacher artifacts, not through direct imports of each other's internal experiment code.

## 3. Read Order
Before editing a target area, read in this order:
1. This file
2. The subsystem-local `CLAUDE.md` if one exists
3. The nearest relevant `DESIGN.md`
4. The nearest `PLAN.md` only if the task depends on current experiment status

Examples:
- Editing shared training code: `PathoML/DESIGN.md` → `PathoML/optimization/DESIGN.md` → `PathoML/optimization/TRAINER_DESIGN.md`
- Editing teacher experiments: `teacher/CLAUDE.md` → `teacher/DESIGN.md` → `teacher/experiments/DESIGN.md` → `teacher/experiments/PLAN.md`
- Editing distillation losses: `distillation/CLAUDE.md` → `distillation/DESIGN.md` → `distillation/losses/DESIGN.md` → `distillation/experiments/PLAN.md`

## 4. Working Rules
- Keep `PathoML` focused on shared contracts, utilities, and training runtime.
- Keep teacher concrete datasets and concrete models inside `teacher/`.
- Keep distillation-specific datasets, losses, teacher adapters, students, and experiments inside `distillation/`.
- If a stable interface changes, update the corresponding `DESIGN.md` in the same turn.
- If experiment results or next steps change, update the corresponding `experiments/PLAN.md` in the same turn.
- Prefer package imports over `sys.path` manipulation.
- Ignore generated experiment outputs, not the `experiments/` code directories themselves.
- Preserve established naming vocabulary inside each subsystem. Do not introduce a second name for the same concept once one term is already in use.

## 5. Validation
- Shared runtime changes: run the relevant `tests/` subset.
- Teacher changes: validate imports and at least one representative experiment entry path.
- Distillation changes: validate manifest loading, dataset construction, and experiment entry imports.

## 6. Code Style
- Use 2-space indentation for Python.
- Keep comments compact and structural in English.
- Prefer explicit ownership boundaries over convenience re-exports.
