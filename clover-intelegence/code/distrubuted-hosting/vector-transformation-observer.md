# Vector transformation observer

A protocol for discovering the mathematical transformation underlying an observed
sequence of vectors, and for finding the simplest equivalent representation of it.

Observation first. No model is assumed.

---

## What the sequence is in this system

Stated here only so the protocol has a concrete subject. These are measured facts from
the stage traces, not findings of this protocol, which has not yet been run.

A single token position produces a sequence of residual vectors:

```
x₀  ──server──▶  x₁  ──layer 1──▶  x₂  ──layer 2──▶  …  ──layer 92──▶  x₉₃  ──layer 93──▶  token
```

- Each `xᵢ` is 7,168 single-precision floats, 28,672 bytes.
- 92 of 93 owners walk a byte-identical stage path for every position of every prompt:
  108 stages for a KDA layer, 105 for an MLA layer.
- Within a layer, 100 of 108 stages never alter the residual. Exactly one, stage 116,
  always does.
- Each layer selects exactly 16 experts of 896 per position, and the selection depends
  on the vector.

The last point matters for Phase 2: the transformation at each step is **data
dependent**, because routing is a function of the vector being transformed. Whether
that makes it non-linear in the relevant sense is a question for the observation, not
an assumption to carry in.

Capture tooling already in place: `common/trace.h` records per-stage hashes; it would
need extending to record full vectors for this work.

---

## Phase 1 — Observe

Analyse the complete sequence:

```
x₀ → x₁ → x₂ → … → xₙ
```

Do not assume the transformation is linear, constant, invertible, deterministic, or of
the form `xₙ = Aⁿx₀`.

Look for:

- linear relationships
- affine relationships
- constant transformations
- changing transformations
- matrix transformations
- low-rank structure
- eigenvalue / eigenvector structure
- polynomial relationships
- nonlinear transformations
- invariant quantities
- conserved dimensions or subspaces
- convergence or fixed points
- repeated patterns
- sparsity
- periodicity
- transformations that can be composed
- relationships between individual dimensions
- relationships between successive differences
- relationships between higher-order differences

Do not force the observations into a predetermined mathematical model.

---

## Phase 2 — Determine the transformation class

Determine whether the observed sequence is best described by:

```
xₙ₊₁ = A xₙ            constant linear operator
xₙ₊₁ = Aₙ xₙ           step-dependent linear operator
xₙ₊₁ = f(xₙ)           state-dependent, possibly nonlinear
xₙ   = Fₙ(x₀)          direct function of the origin and the index
```

or another mathematical representation.

**Explicitly test whether a single constant `A` exists.** If it does, determine it from
the observations.

---

## Phase 3 — Derive the most efficient representation

If `xₙ = Aⁿx₀` is valid, do not automatically recommend brute-force multiplication.

Investigate the best representation of `Aⁿ`:

- repeated squaring
- eigendecomposition
- diagonalisation
- Jordan form where relevant
- minimal polynomial
- characteristic polynomial
- Cayley–Hamilton reduction
- low-rank decomposition
- sparse representation
- Krylov-subspace methods
- polynomial representation of `A`
- any other mathematically valid reduction

The objective is not merely to calculate `Aⁿ` faster. The objective is to determine
whether **the entire observed transformation can be represented by a substantially
simpler mathematical form.**

---

## Phase 4 — Search for a direct solution

Try to derive:

```
xₙ = G(n, x₀)
```

where `G` produces the same `xₙ` without executing every intermediate transformation.

If possible, derive `G` explicitly.

If the sequence converges, also investigate whether the limiting vector

```
x* = lim (n→∞) xₙ
```

can be calculated directly.

---

## Phase 5 — Verification

Any proposed equation must be tested against the observed sequence. For every
available `k`:

```
predicted(xₖ)  vs  observed(xₖ)
```

Report:

- absolute error
- relative error
- maximum error
- numerical precision used
- whether the equality is exact or approximate

Then test the equation on vectors or transformations that were **not** used to derive
it.

---

## Critical requirement

Do not stop after finding a mathematically valid equation.

Search for the **simplest equivalent representation with the lowest computational cost
and smallest data representation.**

The original sequence may contain intermediate values that are unnecessary once the
underlying mathematical relationship has been discovered.

Therefore ask:

> What information is actually necessary to reproduce `xₙ`?

and:

> Which parts of the original data can be discarded because their mathematical effect
> has already been captured by the derived representation?

Do not assume the answer. Derive it from the observations.

---

## Required output

1. **Observed pattern**
2. **Transformation class**
3. **Derived mathematical representation**
4. **Derivation**
5. **Direct-computation method**
6. **Data actually required**
7. **Data that can be discarded**
8. **Computational complexity before and after**
9. **Numerical error**
10. **Independent validation results**

---

## Discipline while running this

- Observe before classifying. A pattern that appears in one prompt is an observation,
  not a property of the transformation.
- Record what was checked, what was seen, and what could not be reached. A named gap is
  a limitation; a gap filled with a plausible guess is a defect.
- An unfavourable result is evidence. Do not predict a structure and then go looking
  for it.
- State the precision used. Bit-exactness against the existing oracle is the acceptance
  gate for any representation proposed as a replacement, not approximate agreement.
- Scope every claim to what produced it: this hardware, this model, these prompts,
  this many positions.
