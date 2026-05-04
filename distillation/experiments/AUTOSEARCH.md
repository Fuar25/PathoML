# Distillation Autoresearch Protocol

## Purpose
- Maximize distillation gain from the fixed teacher winner to an HE-only student.
- The primary score is `gain_f1_mean = distilled_f1_mean - baseline_f1_mean`.
- Run distillation-algorithm-first 3-run / 5-fold screening continuously.
- Make the loop recoverable from files on disk, not from chat context.
- This protocol is repository infrastructure and belongs on `master`.

## Protocol Location
- Active worktree for protocol maintenance: `/home/sbh/PathoML`
- Protocol file: `/home/sbh/PathoML/distillation/experiments/AUTOSEARCH.md`
- Protocol branch: `master`

Do not run distillation autoresearch on `master`.
Use `master` only to maintain this protocol and stable repo infrastructure.

## Game Setup
- Pick a run tag before starting, for example `distill-gain-YYYYMMDD`.
- Create a fresh experiment branch from current `master`:
  - branch pattern: `exp/distillation-autosearch-<tag>`
  - example: `exp/distillation-autosearch-distill-gain-20260504`
- Use `/home/sbh/PathoML` as the single active worktree on that experiment branch.
- Create the run state root:
  - `../PathoML-runs/distillation-autosearch/<tag>/`
- Create subdirectories:
  - `logs/`
  - `outputs/`
- Initialize `STATE.md` and `results.tsv` in that run state root.
- Set both `start_commit` and `current_best_commit` in `STATE.md` to the experiment branch start commit.
- After setup, `current_best_commit` is runtime state only; update it only when a candidate becomes `best`.

## Required State Files
- `STATE.md`
  - Human and agent recovery entry point.
  - Derived runtime artifact; if missing, regenerate it from Game Setup.
  - Holds only run-local mutable state: commits, current best, running candidate, next ideas, and recent summary.
  - `Recent Summary` entries must start with a short UTC `HH:MM` timestamp, for example `13:45`.
  - Use the hour timestamp instead of a full date for normal same-day autoresearch notes.
- `results.tsv`
  - Pure completed-result table.
  - Header must be exactly:
    `candidate_id	commit	status	gain_f1_mean	gain_f1_std	baseline_f1_mean	baseline_f1_std	distilled_f1_mean	distilled_f1_std	baseline_auc_mean	baseline_auc_std	distilled_auc_mean	distilled_auc_std	run_indices	gpu	output_path`
  - Allowed `status` values: `best`, `discard`, `crash`.

Do not commit `STATE.md` or `results.tsv`.
Do not use `queue.jsonl`.
Do not duplicate candidate configuration details in `results.tsv`.
Do not duplicate stable protocol rules in `STATE.md`.

## Git Model
- `start_commit` is the `master` HEAD used when the autoresearch branch was created.
- `current_best_commit` is runtime state in `STATE.md`, not a protocol value.
- Do not change `current_best_commit` for protocol commits on `master`.
- Each candidate starts from `current_best_commit`.
- Each candidate code change is committed before screening.
- If the candidate improves `gain_f1_mean`, keep the commit and update `current_best_commit`.
- If the candidate does not improve `gain_f1_mean`, append `discard` to `results.tsv` and reset back to `current_best_commit`.
- If the candidate crashes, OOMs, or misses expected outputs, append `crash` to `results.tsv` and reset back to `current_best_commit`.
- Never reset or rewrite `master`.
- Before any candidate reset, verify the active branch starts with `exp/distillation-autosearch-`.

The experiment branch history should contain only retained best-code states plus durable autoresearch infrastructure.

## Screening Contract
- Default screening environment:
  - `PATHOML_RUN_INDICES=0,1,2`
  - `PATHOML_SKIP_CONDITION_LOG=1`
  - `PATHOML_TEACHER_MANIFEST=../PathoML-runs/teacher-winners/manifest.json`
  - `PATHOML_DISTILLATION_OUTPUTS_ROOT=../PathoML-runs/distillation-autosearch/<tag>/outputs/<candidate_id>`
- Each candidate must run a matched pair:
  - baseline: same final `StudentBasicABMIL` kwargs, task-only.
  - distilled: same final `StudentBasicABMIL` kwargs, candidate distillation algorithm enabled.
- Primary score: mean fold-level patient F1 gain between the matched pair.
- AUC is recorded but does not block a higher-gain best update.
- A candidate becomes `best` when its `gain_f1_mean` is strictly higher than the current best gain.

## Search Surface
- Fixed teacher input:
  - `../PathoML-runs/teacher-winners/manifest.json`
- Fixed final evaluated student class:
  - `StudentBasicABMIL`
- Search may change:
  - distillation loss, teacher signal, and target transformation.
  - curriculum, staged distillation, and online distillation.
  - multi-student training, mutual learning, and teacher-student online updates.
  - confidence gating, sample weighting, and hard/easy mining.
  - `StudentBasicABMIL` architecture hyperparameters such as `hidden_dim`, `attention_dim`, and `dropout`.
- Do not change the final evaluated object to an ensemble or non-BasicABMIL model unless this protocol is explicitly revised.

## Default Training Settings
- `lr=1e-4`
- `weight_decay=1e-5`
- `batch_size=16`
- `epochs=100`
- `patience=10`
- Final student class: `StudentBasicABMIL`
- Default final student kwargs:
  - `patch_dim=1536`
  - `hidden_dim=128`
  - `attention_dim=128`
  - `dropout=0.2`

Only tune training settings when required by the distillation algorithm, OOM, or clear optimization failure.

## Log Policy
- Autoresearch screening must not modify:
  - `distillation/experiments/PLAN.md`
  - `distillation/experiments/results_log.txt`
  - `distillation/experiments/results_log_mil_abmil.txt`
- Runner must set `PATHOML_SKIP_CONDITION_LOG=1` for every screening command.
- Runner must redirect stdout and stderr to:
  - `../PathoML-runs/distillation-autosearch/<tag>/logs/<candidate_id>.log`
- Runner must read metrics from run artifacts such as `run_metrics.json`, predictions, manifests, and screening logs.
- Coordinator must record completed screening results only in external `results.tsv`.
- Formal distillation logs and `PLAN.md` are reserved for human-requested reporting outside the screening loop.
- If `distillation/experiments/results_log*.txt` or `distillation/experiments/PLAN.md` is dirty before a candidate starts, stop and resolve it before launching screening.

## Coordinator Role
- If you are reading this protocol to run autoresearch, you are the Coordinator.
- The Coordinator plays the game by orchestration, not by personally doing every task.
- The Coordinator owns decisions, state transitions, and git advancement.
- The Coordinator must keep its own context small and durable.
- The Coordinator may delegate idea discovery to an Explorer subagent.
- The Coordinator must delegate implementation to a Worker subagent.
- The Coordinator must delegate experiment execution and watching to a Runner subagent.
- The Coordinator should receive compact summaries, not raw papers, training logs, or long diffs.
- The Coordinator may inspect code, logs, or paper snippets directly only when a subagent summary exposes a concrete blocker or inconsistency.

## Candidate Lifecycle
1. Coordinator reads `STATE.md`, `results.tsv`, this protocol, and `git status`.
   If `STATE.md` does not exist, stop and perform Game Setup first.
2. Coordinator optionally asks an Explorer for compact idea cards.
3. Coordinator chooses one candidate and sets `STATE.md` running candidate.
4. Coordinator spawns one Worker subagent with the candidate objective and file ownership.
5. Worker implements one candidate in `/home/sbh/PathoML` on the autoresearch branch.
6. Worker runs import checks and focused tests.
7. Worker commits the candidate code.
8. Worker reports changed files, commit hash, and validation results.
9. Coordinator reviews the Worker summary and commit metadata.
10. Coordinator selects the GPU and matched-pair run command, then spawns one Runner subagent.
11. Runner launches baseline and distilled screening, watches the process, reads outputs, and summarizes results.
12. Runner reports compact metrics, status, crash reason if any, and output paths.
13. Coordinator appends one row to `results.tsv`.
14. Coordinator updates `STATE.md`.
15. Coordinator keeps the commit on `best`, or resets to `current_best_commit` on `discard`/`crash`.

Do not start a candidate unless this lifecycle can be followed.

## Subagent Rules
- Coordinator must not implement candidate code directly.
- Coordinator must not run screening commands directly.
- Coordinator must not tail full training logs directly.
- Coordinator may have at most one Worker and one Runner active at a time.
- Coordinator may call Explorer separately when the next idea pool is thin or repetitive.
- Coordinator is the only agent that writes `STATE.md` and `results.tsv`.
- Coordinator is the only agent that keeps or resets candidate commits.
- Coordinator records decisions and compact summaries, not raw logs or full paper notes.
- Explorer only discovers ideas.
- Explorer may read `/home/sbh/A Comprehensive Survey on Knowledge Distillation.pdf`, public papers, web sources, historical `PLAN.md`, prior `results.tsv`, and diagnostic reports.
- Explorer must not edit repository files, run experiments, write runtime state, or make git decisions.
- Explorer final response must be compact idea cards: algorithm name, source, core mechanism, BasicABMIL adaptation, expected gain source, implementation risk, and minimal validation.
- Worker handles one candidate at a time.
- Worker may edit losses, training flow, runner code, models, and tests for that candidate.
- Worker must verify `/home/sbh/PathoML` is on an `exp/distillation-autosearch-*` branch.
- Worker must not work on `master`.
- Worker must not promote a teacher or distillation winner.
- Worker must not update `distillation/experiments/PLAN.md`.
- Worker final response must list changed files, commit hash, and verification results.
- Runner handles one matched-pair screening run at a time.
- Runner launches the screening commands given by the Coordinator.
- Runner watches the process until completion, failure, or explicit timeout.
- Runner may read training logs, `run_metrics.json`, predictions, manifests, and output directories.
- Runner must not edit repository files.
- Runner must not write `STATE.md` or `results.tsv`.
- Runner must not keep or reset candidate commits.
- Runner final response must be compact: status, gain F1, baseline/distilled F1/AUC, crash reason if any, and output paths.

## Documentation Rules
- Do not update `distillation/experiments/PLAN.md` during screening.
- Use `STATE.md` for short-cycle summaries.
- Update formal experiment docs only after the user explicitly requests promotion, distillation reporting, or mechanism writeup.

## Validation Rules
- New losses need focused loss tests.
- New models or final-student kwargs paths need forward-shape tests.
- New runners need import checks.
- Each candidate commit must pass targeted validation before screening.
- The first real candidate must confirm matched task-only baseline, distilled run, `run_metrics.json`, `results.tsv`, external logs, output dirs, and `STATE.md` behavior before the loop continues.
