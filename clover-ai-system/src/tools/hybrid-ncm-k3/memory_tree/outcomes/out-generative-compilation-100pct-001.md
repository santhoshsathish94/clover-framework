---
{
  "schema_version": "1.0",
  "id": "MEM-OUT-GENERATIVE-COMPILATION-100PCT-001",
  "title": "Outcome: Extended HumanEval `40/40` (`100%`), Live Rust `rustc` `5/5` (`100%`), and `-17.9%` (`N=20`) / `-14.1%` (`N=40`) Latency Reduction",
  "entry_type": "outcome",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "generative-compilation",
    "humaneval",
    "rust-live-suite",
    "latency"
  ],
  "tags": [
    "codegen",
    "outcome",
    "benchmarks",
    "verified"
  ],
  "created_at": "2026-10-26T02:04:12Z",
  "updated_at": "2026-10-26T02:04:12Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4.2",
    "actor": "system_measurement",
    "recorded_at": "2026-10-26T02:04:12Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
      "provenance": "commit:f308f12",
      "confidence": 0.95,
      "context_scope": "generative_compilation",
      "created_at": "2026-10-26T02:04:12Z",
      "note": "Records the benchmark outcomes of the Python and Rust generative compilation rollback experiment."
    },
    {
      "type": "similar_to",
      "target": "MEM-OUT-RUST-GPU-PARITY-001",
      "provenance": "commit:f308f12",
      "confidence": 0.85,
      "context_scope": "benchmark_suite_outcomes",
      "created_at": "2026-10-26T02:04:12Z",
      "note": "Companion outcome on the `rust-gpu-merged` branch."
    }
  ],
  "uncertainty_notes": [
    "The `-17.9%` latency reduction is measured on `N=20` (`11,682 -> 9,586 ms/prob`), whereas on `N=40` the reduction is `-14.1%` (`11,309 -> 9,720 ms/prob`)."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-26T02:04:12Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial outcome entry recorded from commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4.2"
    }
  ]
}
---

# Outcome: Extended HumanEval `40/40` (`100%`), Live Rust `rustc` `5/5` (`100%`), and `-17.9%` (`N=20`) / `-14.1%` (`N=40`) Latency Reduction

## Summary
With Generative Compilation active on `gpu` (`8 x RTX 3090`), Clover Kimi K3 achieved `20/20` (`100%`) on Standard HumanEval at `9,586 ms/prob` (`-17.9%` vs `11,682 ms/prob` in `V5` by stopping when the target function ends), `40/40` (`100%`, up from `36/40`) on Extended HumanEval at `9,720 ms/prob` (`-14.1%` vs `11,309 ms/prob`), and `5/5` (`100%`, up from `3/5`) on Live Rust `rustc` unit tests.

## Content
Separates the two sources of improvement measured in commit `f308f12`: (1) mid-generation statement-boundary rollback raised Extended HumanEval `N=40` accuracy from `36/40` (`90.0%`) to `40/40` (`100.0%`), while fixing `match_verified_code_template` language gating raised Live Rust `rustc` Part B from `3/5` to `5/5`; (2) stopping generation as soon as the target function finishes (rather than emitting trailing `# Test` blocks) reduced mean latency per problem by `-17.9%` at `N=20` (`11,682 -> 9,586 ms`) and `-14.1%` at `N=40` (`11,309 -> 9,720 ms`).

## Original Source Evidence
`docs/clover-generative-compilation-research-context.md` Section 4.2:
- Standard HumanEval (`N=20`): `20/20 (100.0%)`, `9,586 ms/prob` vs `11,682 ms/prob` (`-17.9%`).
- Extended HumanEval (`N=40`): `40/40 (100.0%)` vs `36/40 (90.0%)`, `9,720 ms/prob` vs `11,309 ms/prob` (`-14.1%`).
- Live Rust `rustc` Part B (`N=5`): `5/5 (100.0%)` vs `3/5 (60.0%)`.

## Revision History
- `2026-10-26T02:04:12Z` (**system_measurement**, `created`): Initial outcome entry recorded from commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4.2
