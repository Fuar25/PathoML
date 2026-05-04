# Distillation Autoresearch Protocol

## Purpose
- Maximize distilled fold-level F1 from the fixed teacher winner to an HE-only student.
- The users may sleep during your work. When they wake up, they should see continuous progress toward better results. Never stop only if the users request it.
- The primary score is `distilled_f1_mean`.
- Use the fixed ABMIL baseline from `distillation/experiments/PLAN.md` only as a reference anchor:
  - F1 `0.8343 +/- 0.0339`
  - AUC `0.9110 +/- 0.0367`
- Run distillation-algorithm-first 3-run / 5-fold screening continuously.
- Make the loop recoverable from files on disk, not from chat context.
- This protocol is repository infrastructure and belongs on `master`.

## Protocol Location
- Active worktree for protocol maintenance: `/home/sbh/PathoML-master`
- Protocol file: `/home/sbh/PathoML-master/distillation/experiments/AUTOSEARCH.md`
- Protocol branch: `master`

Do not run distillation autoresearch on `master`.
Use `master` only to maintain this protocol and stable repo infrastructure.

## Game Setup
- Pick a run tag before starting, for example `distill-f1-YYYYMMDD`.
- Create a fresh experiment branch from current `master`:
  - branch pattern: `exp/distillation-autosearch-<tag>`
  - example: `exp/distillation-autosearch-distill-f1-20260504`
- Use `/home/sbh/PathoML-master` as the single active worktree on that experiment branch.
- Create the run state root:
  - `../PathoML-runs/distillation-autosearch/<tag>/`
- Create subdirectories:
  - `logs/`
  - `outputs/`
- All autosearch runtime artifacts for the tag must stay under this run state root.
- Initialize `STATE.md` and `results.tsv` in that run state root.
- Set `start_commit` and `current_best_commit` in `STATE.md` to the experiment branch start commit.
- Set `current_best_distilled_f1_mean` in `STATE.md` to `none`.
- After setup, `current_best_commit` is runtime state only; update it only when a candidate becomes `best`.

## Required State Files
- `STATE.md`
  - Human and agent recovery entry point.
  - Derived runtime artifact; if missing, regenerate it from Game Setup.
  - Holds only run-local mutable state: commits, current best, running candidate, idea pool, and recent summary.
  - Must include an `Idea Pool` section with Explorer-generated compact idea cards.
  - `Recent Summary` entries must start with a short UTC `HH:MM` timestamp, for example `13:45`.
  - Use the hour timestamp instead of a full date for normal same-day autoresearch notes.
- `results.tsv`
  - Pure completed-result table.
  - Header must be exactly:
    `candidate_id	commit	status	distilled_f1_mean	distilled_f1_std	distilled_auc_mean	distilled_auc_std	run_indices	gpu	output_path`
  - Allowed `status` values: `best`, `discard`, `crash`.

Do not commit `STATE.md` or `results.tsv`.
Do not duplicate candidate configuration details in `results.tsv`.
Do not duplicate stable protocol rules in `STATE.md`.

## Git Model
- `start_commit` is the `master` HEAD used when the autoresearch branch was created.
- `current_best_commit` is runtime state in `STATE.md`, not a protocol value.
- Do not change `current_best_commit` for protocol commits on `master`.
- Each candidate starts from `current_best_commit`.
- Each candidate code change is committed before screening.
- If the candidate improves `distilled_f1_mean`, keep the commit and update `current_best_commit` and `current_best_distilled_f1_mean`.
- If the candidate does not improve `distilled_f1_mean`, append `discard` to `results.tsv` and reset back to `current_best_commit`.
- If the candidate crashes, OOMs, or misses expected outputs, append `crash` to `results.tsv` and reset back to `current_best_commit`.
- Never reset or rewrite `master`.
- Before any candidate reset, verify the active branch starts with `exp/distillation-autosearch-`.

The experiment branch history should contain only retained best-code states plus durable autoresearch infrastructure.

## Screening Contract
- Default screening environment:
  - `PATHOML_RUN_INDICES=0,1,2`
  - `PATHOML_SKIP_CONDITION_LOG=1`
  - `PATHOML_TEACHER_MANIFEST=../PathoML-runs/teacher-winners/manifest.json`
  - `PATHOML_EXPERIMENT_SOURCE_ROOT=/home/sbh`
  - `PATHOML_DISTILLATION_OUTPUTS_ROOT=../PathoML-runs/distillation-autosearch/<tag>/outputs/<candidate_id>`
- Distillation feature and teacher-output caches are on by default; diagnostics may set:
  - `PATHOML_DISTILLATION_CACHE_FEATURES=0`
  - `PATHOML_DISTILLATION_CACHE_TEACHER_OUTPUTS=0`
  - `PATHOML_TEACHER_OUTPUT_CACHE_BATCH_SIZE=<n>`
- `PATHOML_DISTILLATION_OUTPUTS_ROOT` must always point under the tag-local autosearch `outputs/` directory.
- Autosearch must not write screening outputs to `../PathoML-runs/distillation/`; that root is reserved for human-run distillation search.
- Actual data roots are under `/home/sbh/Features/`:
  - `/home/sbh/Features/GigaPath-Patch-Feature`
  - `/home/sbh/Features/GigaPath-Slide-Feature`
  - `/home/sbh/Features/labels.csv`
- Runner must preflight dataset construction before screening:
  - teacher manifest fingerprint must match the distillation dataset intersection.
  - expected shared sample count is `300`.
  - an empty intersection usually means the worktree-local default `Features/` path was used by mistake.
- Each candidate runs only the distilled condition:
  - final evaluated student is `StudentBasicABMIL`.
  - candidate distillation algorithm is enabled.
- Primary score: mean fold-level patient F1 from the distilled condition.
- AUC is recorded but does not block a higher-F1 best update.
- A candidate becomes `best` when its `distilled_f1_mean` is strictly higher than the current best distilled F1.
- Fixed ABMIL baseline metrics are a reference anchor only; do not run task-only baseline during autosearch screening.

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
- Runner must keep run artifacts under:
  - `../PathoML-runs/distillation-autosearch/<tag>/outputs/<candidate_id>/`
- Runner must not let autosearch conditions fall back to the default `../PathoML-runs/distillation/` output root.
- Runner must read metrics from run artifacts such as `run_metrics.json`, predictions, manifests, and screening logs.
- Coordinator must record completed screening results only in external `results.tsv`.
- Formal distillation logs and `PLAN.md` are reserved for human-requested reporting outside the screening loop.
- If `distillation/experiments/results_log*.txt` or `distillation/experiments/PLAN.md` is dirty before a candidate starts, stop and resolve it before launching screening.

## Coordinator Role
- If you are reading this protocol to run autoresearch, you are the Coordinator.
- The Coordinator plays the game by orchestration, not by personally doing every task.
- The Coordinator owns decisions, state transitions, and git advancement.
- The Coordinator must keep its own context small and durable.
- The Coordinator must maintain an Explorer-generated idea pool.
- The Coordinator should not start a new mechanism family unless it came from an Explorer idea card or was explicitly requested by the user.
- The Coordinator may skip a new Explorer call only when `STATE.md` already has enough fresh unused idea cards.
- The Coordinator must delegate implementation to a Worker subagent.
- The Coordinator must delegate experiment execution and watching to a Runner subagent.
- The Coordinator should receive compact summaries, not raw papers, training logs, or long diffs.
- The Coordinator may inspect code, logs, or paper snippets directly only when a subagent summary exposes a concrete blocker or inconsistency.

## Idea Pool Rules
- `STATE.md` must keep an `Idea Pool` section with 3-5 compact unused ideas when possible.
- Each idea card should include:
  - `candidate_stub`
  - `algorithm_name`
  - `source`
  - `core_mechanism`
  - `BasicABMIL adaptation`
  - `exact_candidate_sketch`
  - `expected_f1_gain_source`
  - `implementation_risk`
  - `minimal_validation`
  - `why_different_from_tried_candidates`
  - `status`
- Allowed idea statuses: `unused`, `selected`, `discarded_by_coordinator`, `used_best`, `used_discard`, `used_crash`, `stale`.
- Coordinator must refresh the idea pool with Explorer when:
  - a run is newly initialized and no fresh ideas exist.
  - fewer than 3 ideas have status `unused`.
  - 2 consecutive candidates are `discard` or `crash`.
  - 3 consecutive candidates fail to improve `current_best_distilled_f1_mean`.
  - the next candidate would change search surface class, such as student kwargs, curriculum, online distillation, or multi-student training.
  - the user asks to continue search and `STATE.md` has no fresh unused idea.
- Coordinator records the selected idea id and selection reason in `STATE.md` before delegating to Worker.
- Coordinator may reject an Explorer idea, but must mark it `discarded_by_coordinator` with a short reason.

## Candidate Lifecycle
1. Coordinator reads `STATE.md`, `results.tsv`, this protocol, and `git status`.
   If `STATE.md` does not exist, stop and perform Game Setup first.
2. Coordinator checks `STATE.md` `Idea Pool` and refreshes it with Explorer if any Idea Pool Rule triggers.
3. Coordinator chooses one unused idea card, records the selection reason, and sets `STATE.md` running candidate.
4. Coordinator spawns one Worker subagent with the candidate objective and file ownership.
5. Worker implements one candidate in `/home/sbh/PathoML-master` on the autoresearch branch.
6. Worker runs import checks and focused tests.
7. Worker commits the candidate code.
8. Worker reports changed files, commit hash, and validation results.
9. Coordinator reviews the Worker summary and commit metadata.
10. Coordinator selects the GPU and distilled run command, then spawns one Runner subagent.
11. Runner launches distilled screening, watches the process, reads outputs, and summarizes results.
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
- Coordinator must call Explorer when the `Idea Pool Rules` require a refresh.
- Coordinator is the only agent that writes `STATE.md` and `results.tsv`.
- Coordinator is the only agent that keeps or resets candidate commits.
- Coordinator records decisions and compact summaries, not raw logs or full paper notes.
- Explorer only discovers ideas.
- Explorer may read `/home/sbh/A Comprehensive Survey on Knowledge Distillation.pdf`, public papers, web sources, historical `PLAN.md`, prior `results.tsv`, diagnostic reports, and code needed to assess implementation fit.
- Explorer must not edit repository files, run experiments, write runtime state, or make git decisions.
- Explorer final response must be compact idea cards using the `Idea Pool Rules` fields.
- Explorer should prioritize ideas that are meaningfully different from already tried candidates.
- Worker handles one candidate at a time.
- Worker may edit losses, training flow, runner code, models, and tests for that candidate.
- Worker must verify `/home/sbh/PathoML-master` is on an `exp/distillation-autosearch-*` branch.
- Worker must not work on `master`.
- Worker must not promote a teacher or distillation winner.
- Worker must not update `distillation/experiments/PLAN.md`.
- Worker final response must list changed files, commit hash, and verification results.
- Runner handles one distilled screening run at a time.
- Runner launches the screening commands given by the Coordinator.
- Runner watches the process until completion, failure, or explicit timeout.
- Runner may read training logs, `run_metrics.json`, predictions, manifests, and output directories.
- Runner must not edit repository files.
- Runner must not write `STATE.md` or `results.tsv`.
- Runner must not keep or reset candidate commits.
- Runner final response must be compact: status, distilled F1/AUC, crash reason if any, and output paths.

## Documentation Rules
- Do not update `distillation/experiments/PLAN.md` during screening.
- Use `STATE.md` for short-cycle summaries.
- Update formal experiment docs only after the user explicitly requests promotion, distillation reporting, or mechanism writeup.

## Validation Rules
- New losses need focused loss tests.
- New models or final-student kwargs paths need forward-shape tests.
- New runners need import checks.
- Each candidate commit must pass targeted validation before screening.
- The first real candidate must confirm feature-root preflight, distilled run, `run_metrics.json`, `results.tsv`, external logs, output dirs, and `STATE.md` behavior before the loop continues.
