# Teacher Autosearch Protocol

## Purpose
- Maximize patient-level F1 for the HE/CD20/CD3 teacher on `GigaPath-Patch-Feature-RegCoordOrigFeat`.
- The users may sleep during your work. When they wake up, they should see continuous progress toward better F1.
- Run architecture-first 3-run / 5-fold screening continuously.
- Make the autosearch recoverable from files on disk, not from chat context.
- This protocol is repository infrastructure and belongs on `master`.

## Protocol Location
- Active worktree: `/home/sbh/PathoML`
- Protocol file: `/home/sbh/PathoML/teacher/experiments/AUTOSEARCH.md`
- Protocol branch: `master`

Do not run teacher autosearch on `master`.
Use `master` only to maintain this protocol and stable repo infrastructure.

## Game Setup
- Pick a run tag before starting, for example `f1-regcoord-YYYYMMDD`.
- Create a fresh experiment branch from current `master`:
  - branch pattern: `exp/teacher-autosearch-<tag>`
  - example: `exp/teacher-autosearch-f1-regcoord-20260502`
- Use `/home/sbh/PathoML` as the single active worktree on that experiment branch.
- Create the run state root:
  - `../PathoML-runs/teacher-autosearch/<tag>/`
- Initialize `STATE.md` and `results.tsv` in that run state root.
- Set both `start_commit` and `current_best_commit` in `STATE.md` to the experiment branch start commit.
- After setup, `current_best_commit` is runtime state only; update it only when a candidate becomes `best`.

## Required State Files
- `STATE.md`
  - Human and agent recovery entry point.
  - Holds only run-local mutable state: commits, current best, running candidate, next ideas, and recent summary.
  - `Recent Summary` entries must start with a short UTC `HH:MM` timestamp, for example `13:45`.
  - Use the hour timestamp instead of a full date for normal same-day autosearch notes.
- `results.tsv`
  - Pure completed-result table.
  - Header must be exactly:
    `candidate_id	commit	status	f1_mean	f1_std	auc_mean	auc_std	run_indices	gpu	output_path`
  - Allowed `status` values: `best`, `discard`, `crash`.

Do not commit `STATE.md` or `results.tsv`.
Do not duplicate candidate configuration details in `results.tsv`.
Do not duplicate stable protocol rules in `STATE.md`.

## Git Model
- `start_commit` is the `master` HEAD used when the autosearch branch was created.
- `current_best_commit` is runtime state in `STATE.md`, not a protocol value.
- Do not change `current_best_commit` for protocol commits on `master`.
- Each candidate starts from `current_best_commit`.
- Each candidate code change is committed before screening.
- If the candidate improves F1, keep the commit and update `current_best_commit`.
- If the candidate does not improve F1, append `discard` to `results.tsv` and reset back to `current_best_commit`.
- If the candidate crashes, OOMs, or misses expected outputs, append `crash` to `results.tsv` and reset back to `current_best_commit`.
- Never reset or rewrite `master`.
- Before any candidate reset, verify the active branch starts with `exp/teacher-autosearch-`.

The experiment branch history should contain only retained best-code states plus durable autosearch infrastructure.

## Screening Contract
- Default screening environment:
  - `PATHOML_RUN_INDICES=0,1,2`
  - `PATHOML_N_RUNS=3`
  - `PATHOML_K_FOLDS=5`
  - `PATHOML_BASE_SEED=42`
  - `PATHOML_SKIP_CONDITION_LOG=1`
- Primary score: mean fold-level patient F1.
- AUC is recorded but does not block a higher-F1 best update.
- A candidate becomes `best` when its F1 is strictly higher than the current best F1.

## Log Policy
- Autosearch screening must not modify `teacher/experiments/results_log.txt`.
- Runner must set `PATHOML_SKIP_CONDITION_LOG=1` for every screening command.
- Runner must redirect stdout and stderr to:
  - `../PathoML-runs/teacher-autosearch/<tag>/logs/<candidate_id>.log`
- Runner must read metrics from run artifacts such as `run_metrics.json`, manifests, and screening logs.
- Coordinator must record completed screening results only in external `results.tsv`.
- `teacher/experiments/results_log.txt` is reserved for formal experiment reporting outside the screening loop.
- If `teacher/experiments/results_log.txt` is dirty before a candidate starts, stop and resolve it before launching screening.

## Default Training Settings
- `lr=1e-4`
- `weight_decay=1e-5`
- `batch_size=16`
- `epochs=100`
- `patience=30`
- `early_stopping_metric='patient_f1'`
- Dataset root: `/home/sbh/Features/GigaPath-Patch-Feature-RegCoordOrigFeat`
- Stains: `HE`, `CD20`, `CD3`
- Alignment: `union`
- Cache aligned items: enabled

Only tune training settings when required by an architecture, OOM, or clear optimization failure.

## Architecture Priority
Prioritize architecture changes over broad hyperparameter search.

## Coordinator Role
- If you are reading this protocol to run autosearch, you are the Coordinator.
- The Coordinator plays the game by orchestration, not by personally doing every task.
- The Coordinator owns decisions, state transitions, and git advancement.
- The Coordinator must keep its own context small and durable.
- The Coordinator must delegate implementation to a Worker subagent.
- The Coordinator must delegate experiment execution and watching to a Runner subagent.
- The Coordinator should receive compact summaries, not raw training logs or long diffs.
- The Coordinator may inspect code or logs directly only when a subagent summary exposes a concrete blocker or inconsistency.

## Candidate Lifecycle
1. Coordinator reads `STATE.md`, `results.tsv`, this protocol, and `git status`.
   If `STATE.md` does not exist, stop and perform Game Setup first.
2. Coordinator sets `STATE.md` running candidate.
3. Coordinator spawns one Worker subagent with the candidate objective and file ownership.
4. Worker implements one candidate in `/home/sbh/PathoML` on the autosearch branch.
5. Worker runs import checks and focused tests.
6. Worker commits the candidate code.
7. Worker reports changed files, commit hash, and validation results.
8. Coordinator reviews the Worker summary and commit metadata.
9. Coordinator selects the GPU and run command, then spawns one Runner subagent.
10. Runner launches screening, watches the process, reads outputs, and summarizes results.
11. Runner reports compact metrics, status, crash reason if any, and output paths.
12. Coordinator appends one row to `results.tsv`.
13. Coordinator updates `STATE.md`.
14. Coordinator keeps the commit on `best`, or resets to `current_best_commit` on `discard`/`crash`.

Do not start a candidate unless this lifecycle can be followed.

## Subagent Rules
- Coordinator must not implement candidate code directly.
- Coordinator must not run screening commands directly.
- Coordinator must not tail full training logs directly.
- Coordinator may have at most one Worker and one Runner active at a time.
- Coordinator is the only agent that writes `STATE.md` and `results.tsv`.
- Coordinator is the only agent that keeps or resets candidate commits.
- Coordinator records decisions and compact summaries, not raw logs.
- Worker handles one candidate at a time.
- Worker may edit models, runner code, and tests for that candidate.
- Worker must verify `/home/sbh/PathoML` is on an `exp/teacher-autosearch-*` branch.
- Worker must not work on `master`.
- Worker must not promote a teacher winner.
- Worker must not update `teacher/experiments/PLAN.md`.
- Worker final response must list changed files, commit hash, and verification results.
- Runner handles one screening run at a time.
- Runner launches the screening command given by the Coordinator.
- Runner watches the process until completion, failure, or explicit timeout.
- Runner may read training logs, `run_metrics.json`, `cv_predictions.csv`, manifests, and output directories.
- Runner must not edit repository files.
- Runner must not write `STATE.md` or `results.tsv`.
- Runner must not keep or reset candidate commits.
- Runner final response must be compact: status, F1/AUC metrics, crash reason if any, and output paths.

## Documentation Rules
- Do not update `teacher/experiments/PLAN.md` during screening.
- Use `STATE.md` for short-cycle summaries.
- Update formal experiment docs only after the user explicitly requests promotion, distillation, or formal reporting.

## Validation Rules
- New models need forward-shape tests.
- New runners need import checks.
- Each candidate commit must pass targeted validation before screening.
- The first real candidate must confirm checkpoint, `run_metrics.json`, `cv_predictions.csv`, `results.tsv`, and `STATE.md` behavior before the loop continues.
