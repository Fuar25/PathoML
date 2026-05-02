# CLAUDE.md

Agent guidance for this repository.

## 1. Document Roles
- `DESIGN.md`: stable facts, boundaries, interfaces, invariants, decisions
- `CLAUDE.md`: agent workflow and editing rules
- `PLAN.md`: experiment status, results, next steps

Keep status out of `DESIGN.md`.
Keep architecture changes out of `PLAN.md`.

## 2. Repository Shape
- `PathoML/`: shared foundation
- `teacher/`: teacher-selection subsystem
- `distillation/`: distillation subsystem
- `scripts/`: misc tooling only
- `.venv/`: project Python environment

`teacher` and `distillation` are peer subsystems; they coordinate through artifacts.

## 2.1 Environment Rule
- Use the project environment in `.venv/` for Python commands, tooling, and tests.
- Check GPU occupancy before long-running training or evaluation jobs.
- Prefer an idle GPU for ad hoc experiment runs.
- Bind ad hoc GPU placement with `CUDA_VISIBLE_DEVICES`.
- Keep canonical script device config unchanged for ad hoc GPU moves.

## 3. Read Order
1. This file
2. Subsystem `CLAUDE.md` (if present)
3. Nearest `DESIGN.md`
4. Nearest `PLAN.md` only when experiment status is relevant

## 4. Working Rules
- Keep shared code in `PathoML`; keep subsystem-specific code in its subsystem.
- If stable interfaces change, update the nearest `DESIGN.md` in the same turn.
- If experiment status changes, update the matching `experiments/PLAN.md` in the same turn.
- Use package imports; avoid `sys.path` hacks.
- Preserve canonical naming; do not create synonyms for existing concepts.

## 5. Validation
- Shared runtime changes: run relevant `tests/`.
- Teacher changes: validate imports and one representative experiment entry.
- Distillation changes: validate manifest loading, dataset construction, and experiment entry imports.

## 6. Code Style
- Python uses 2-space indentation.
- Comments stay short and structural.
- Prefer explicit ownership boundaries over convenience re-exports.

## 7. Documentation Writing Constitution
- Write in short, scan-friendly phrases; remove repetition.
- Prefer `Action + Object` rule sentences over narrative background.
- Keep existing section skeleton unless explicitly asked to restructure.
- Lock facts during rewrites: metrics, dates, names, symbols, paths.
- Keep machine-facing IDs unchanged; human-facing labels may use established abbreviations.
- Keep `Next Steps` command-style and `Decisions` in `date + decision` form.
