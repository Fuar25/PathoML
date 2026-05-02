# scripts

## 1. Purpose
Hold miscellaneous repository tooling that does not belong to shared foundation or research subsystems.

## 2. Scope / Owns
This directory owns:
- one-off migration or maintenance scripts
- repository-local tooling helpers

It does not own:
- teacher experiments
- distillation experiments
- shared runtime contracts

## 3. Public Contracts
- No stable public API is promised for this directory.
- Scripts may depend on repository internals, but core packages must not depend on `scripts/`.

## 4. Invariants
- Keep `scripts/` outside the core system narrative.
- Keep code pragmatic but avoid implicit dependencies from `PathoML`, `teacher`, or `distillation`.

## 5. Change Rules
- If a script becomes part of a stable subsystem workflow, move it into that subsystem.

## Decided
- `scripts/` is a misc tooling zone only.

## TODO
1. Move scripts that become stable and reusable into the appropriate subsystem.
