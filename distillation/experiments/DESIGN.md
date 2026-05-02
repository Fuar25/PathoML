# distillation/experiments

## 1. Purpose
Define stable experiment rules, outputs, and tracking for distillation runs.

## 2. Scope / Owns
This directory owns:
- distillation experiment entry scripts
- shared experiment helper code in `common.py`
- experiment outputs
- experiment status tracking

## 3. Public Contracts
- Entry scripts import from `distillation.experiments.common`.
- Confirmation rerun scripts may import seed-control helpers from `distillation.experiments.confirmation`.
- Script naming pattern: `run_<fully_descriptive_method_name>.py`.
- Main rule: one script invocation logs exactly one condition result.
- Exception: a script may accept one CLI selector (for example `--topk-ratio`) if each invocation still logs one condition result.
- Machine IDs: lowercase snake case with full words; decimals use `p` instead of `.`.
- Human-facing labels may use established abbreviations (for example TGA, KD, RKD).
- Active `condition` names are derived from composed loss terms, not hidden family parameters.
- Output layout: `../PathoML-runs/distillation/<condition>/run_{run:02d}/...`, shared logs (`results_log.txt`, `results_log_mil_abmil.txt`), and `PLAN.md`.
- Default teacher input is `../PathoML-runs/teacher-winners/manifest.json`; explicit manifest paths may override it.

## 4. Invariants
- `PLAN.md` records only experiment status, results, and next steps.
- `common.py` remains the stable entry helper for experiment scripts.
- Experiment scripts use package imports; do not manipulate `sys.path`.
- Use one canonical family vocabulary in machine IDs and code identifiers; `teacher_guided_attention` stays canonical.
- Shared logging records a human-readable `loss_design` derived from active composed terms.
- Git ignore rules target generated outputs, not code directories.
- Distillation checkpoints and predictions are disposable unless a downstream consumer is explicitly defined.
- Preserve experiment conclusions in `PLAN.md` and `results_log_mil_abmil.txt` before deleting heavy run outputs.

## 5. Change Rules
- If experiment wiring changes, update this file.
- If architecture changes, update `distillation/DESIGN.md` instead of `PLAN.md`.
- Keep log/result schema changes aligned with active experiments.
- If names become ambiguous, rename scripts and condition identifiers instead of adding new machine-ID aliases.
- Keep established baseline conditions as separate scripts; family variants (for example `teacher_guided_attention` variants) must use separate entry points.

## Decided
- Experiment tracking lives in `distillation/experiments/`.
- `common.py` owns teacher manifest loading, dataset construction, run orchestration, and shared logging.
- Heavy run outputs live outside the repository under `../PathoML-runs/distillation/`.
- Default runs consume the fixed current teacher winner, not arbitrary teacher tuning outputs.
- `confirmation.py` owns confirmation-only student seed control while preserving teacher split seeds.
- Legacy short aliases may exist in historical `results_log.txt`; do not reuse them for new experiments.
- New active-line experiments append to `results_log_mil_abmil.txt` by default.
- `teacher_guided_attention` remains the canonical family name for TGA variants.
- `run_teacher_guided_attention.py` stays the historical no-detach cosine-logit TGA condition; new TGA variants use separate scripts.

## TODO
1. Consolidate repeated experiment-script boilerplate further if it stays stable across methods.
