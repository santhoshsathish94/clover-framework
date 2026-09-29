# Clover-K3 Scaling

## 1. From computation reduction to the data problem

Clover-K3 began with a computation problem.

K3 is a very large model with a complex sequence of transformations. Instead of treating the model as an opaque collection of kernels, Clover reconstructed the computation from its individual stages and expressed those stages as a mathematical model.

The principle is similar to the difference between calculating:

```text
1 + 2 + 3 + ... + n
```

by performing every addition and using:

```text
n(n + 1) / 2
```

as its mathematical equivalent.

K3 is vastly more complicated, but the principle is the same.

Its stages can be represented as a composed mathematical expression rather than being understood only as thousands of individual computational operations.

The resulting model equation was executed against the reference implementation and reproduced the verified computation bit-for-bit for the tested paths. The final prefill and decode comparison produced identical checked values with zero mismatches.

This changed the question.

The problem was no longer simply:

> How much computation does K3 contain?

The computation could now be described, decomposed, measured and investigated mathematically.

The next problem became:

> **If the computation is understood, where does the cost actually come from when the model runs?**

The measurements exposed a different constraint.

The model contains roughly 1.5 TB of inference weights, but the computation does not require all of those weights simultaneously. The model moves through layers, and each layer has its own trunk and expert data. A single prompt therefore causes substantial movement of model data even though only a fraction of the total model is required at any individual point.

The computation had been made explicit.

The next problem was therefore **data movement**.

That distinction is the foundation of the scaling architecture described in this document.

The goal is not to claim that the mathematical equation makes K3 require less arithmetic.

The goal is to understand the model well enough to determine **what must remain stationary, what must move, and where each part of the computation should physically live**.

---

## 2. The original problem

