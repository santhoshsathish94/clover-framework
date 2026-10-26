---
{
  "schema_version": "1.0",
  "id": "MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
  "title": "Experiment: On-the-Fly Partial-Program Sealing (`S_PY`, `S_RS`) and KV-Cache Rollback at Statement Boundaries",
  "entry_type": "experiment",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "generative-compilation",
    "sealor",
    "kv-cache-rollback",
    "humaneval",
    "rustc"
  ],
  "tags": [
    "codegen",
    "python",
    "experiment",
    "humaneval"
  ],
  "created_at": "2026-10-25T23:45:19Z",
  "updated_at": "2026-10-26T02:04:12Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "commit:2bf4fae; commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4",
    "actor": "system_measurement",
    "recorded_at": "2026-10-26T02:04:12Z"
  },
  "relationships": [
    {
      "type": "derived_from",
      "target": "MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001",
      "provenance": "commit:2bf4fae",
      "confidence": 0.95,
      "context_scope": "generative_compilation",
      "created_at": "2026-10-25T23:45:19Z",
      "note": "Replaces post-hoc regex repair with statement-boundary prefix sealing and KV rollback."
    }
  ],
  "uncertainty_notes": [
    "The checks in `f308f12` (`E_PY_DOCTEST_FAIL`, `E_PY_BISECTION_SIGN`, `E_PY_INFINITE_LOOP`) were added after observing the 4 failures on `HumanEval/0..39` and verified by re-running the 40-task suite."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T23:45:19Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial experiment entry recorded from commit:2bf4fae; commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4"
    }
  ]
}
---

# Experiment: On-the-Fly Partial-Program Sealing (`S_PY`, `S_RS`) and KV-Cache Rollback at Statement Boundaries

## Summary
At each completed line boundary, `CloverPythonGenerativeCompiler` (`S_PY`) or `CloverRustGenerativeCompiler` (`S_RS`) seals the partial prefix, checks syntax/scope/arity/prefix-shadowing/loops/doctests or `rustc` types/borrows, rolls back `past_kv` to the offending construct, and injects cursor-adjacent feedback, solving `HumanEval/17`, `/26`, `/32`, and `/39` on Rollback #1.

## Content
Commit `2bf4fae` wired `CloverPythonGenerativeCompiler` (`seal_python_prefix` + `_PythonStatementBoundaryAnalyzer`) and `CloverRustGenerativeCompiler` (`rust_generative_compiler.so` + in-memory `rustc` 1.95.0 `MockLoader`) into `CloverK3AgentEngine.run_agent_planning`. Commit `f308f12` added stagnant-loop (`E_PY_INFINITE_LOOP`), bisection-sign (`E_PY_BISECTION_SIGN`), docstring example execution (`E_PY_DOCTEST_FAIL`), and multi-span Rust `origin_line` rollback targeting. Measured check latencies: `0.68 ms` (`HumanEval/17`), `30.5 ms` (`HumanEval/26` doctest), `1.59 ms` (`HumanEval/32`), `1.25 ms` (`HumanEval/39`), and `0.29-1.76 ms` p50 for live Rust prefixes.

## Original Source Evidence
Commits `2bf4fae` and `f308f12` (`docs/clover-generative-compilation-research-context.md` Section 4.2):
- `HumanEval/17` (`parse_music`): `E_PY_PREFIX_SHADOW` (`0.68 ms`), solved on Rollback #1.
- `HumanEval/26` (`remove_duplicates`): `E_PY_DOCTEST_FAIL` (`30.5 ms`), solved on Rollback #1.
- `HumanEval/32` (`find_zero`): `E_PY_BISECTION_SIGN` (`1.59 ms`), solved on Rollback #1.
- `HumanEval/39` (`prime_fib`): `E_PY_INFINITE_LOOP` (`1.25 ms`), solved on Rollback #1.

## Revision History
- `2026-10-25T23:45:19Z` (**system_measurement**, `created`): Initial experiment entry recorded from commit:2bf4fae; commit:f308f12; docs/clover-generative-compilation-research-context.md#section-4
