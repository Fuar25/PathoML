# scripts

## 1. Purpose
Miscellaneous repository tooling that does not belong to the shared foundation or to either research subsystem.

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
- `scripts/` is not part of the core system narrative.
- Code here may be pragmatic, but it must not become an implicit dependency of `PathoML`, `teacher`, or `distillation`.

## 5. Change Rules
- If a script becomes part of a stable subsystem workflow, move it into that subsystem.

## Decided
- `scripts/` is a misc tooling zone only.

## TODO
1. Move any script that becomes stable and reusable into the appropriate subsystem.
