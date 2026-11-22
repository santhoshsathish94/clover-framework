---
name: lean4-mathlib-proving
description: General Lean 4 and Mathlib4 syntax, tactic hygiene, and formal specification review conventions.
---

# Lean 4 & Mathlib4 General Playbook

1. **Lean 4 vs Lean 3 Syntax:**
   - Emit Lean 4 tactics (`refine`, `rcases`, `obtain`, `intro`, `ext`, `rw`, `simp only [...]`, `exact`, `omega`, `ring`), never deprecated Lean 3 tactics (`refine'`, `refl`, `split`).
   - Output ONLY the requested Lean 4 code inside a single ````lean` block with no surrounding prose and no `sorry`, `admit`, or `axiom`.
2. **Decidability & Induction Hygiene:**
   - When case-splitting or inducting over classical propositions or finite collections without an explicit `DecidableEq` instance in scope, add `classical` first.
   - Apply induction hypotheses carefully: if an induction hypothesis has implication premises `P → Q`, supply a proof of `P` before destructuring `Q`.
3. **Formal Statement Review:**
   - Check whether all domain restrictions, poles, singularities, sign conditions, and degenerate cases (`0`, `1`, negative values, empty collections) are either covered by explicit hypotheses or hold unconditionally. Do not reject a statement whose hypotheses already exclude the degenerate point.

