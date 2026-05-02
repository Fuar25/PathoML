# PathoML/models

## 1. Purpose
Provide shared model primitives for pathology learning systems.

## 2. Scope / Owns
This package owns:
- `FeatureEncoder`
- `GatedAttention`
- `LinearClassifier`

This package does not own:
- teacher concrete `ABMIL`
- teacher baseline classifiers such as `LinearProbe` and `MLP`
- distillation-specific student models

## 3. Public Contracts
- `FeatureEncoder(input_dim, embed_dim, dropout)`
- `GatedAttention(embed_dim, attn_dim, gated, dropout)`
- `LinearClassifier(embed_dim, num_classes, dropout)`

These are reusable building blocks that can be composed by teacher or distillation models.

## 4. Invariants
- Keep shared primitives free of teacher-specific self-registration.
- Keep `GatedAttention.last_logits` available for pre-softmax attention consumers.
- Keep shared primitives free of experiment-specific assumptions.

## 5. Change Rules
- Add code here only when it is a reusable primitive.
- Keep teacher concrete model assembly in `teacher/models/`.
- If a primitive output contract changes, update all consuming subsystem design docs.

## Decided
- Shared ABMIL building blocks stay in PathoML because teacher and distillation both reuse them.

## TODO
1. Extract additional shared blocks only if reused across subsystems.
