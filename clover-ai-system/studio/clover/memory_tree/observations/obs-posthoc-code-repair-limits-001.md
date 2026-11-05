---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-POSTHOC-CODE-REPAIR-LIMITS-001",
  "title": "Observation: Post-Hoc Code Repair Cannot Stop Doomed Generation After an Early Syntax, Prefix, or Borrow Error",
  "entry_type": "observation",
  "epistemic_status": "observed",
  "status": "active",
  "concepts": [
    "generative-compilation",
    "post-generation-repair",
    "doomed-tokens",
    "prefix-shadow",
    "borrow-checker"
  ],
  "tags": [
    "codegen",
    "python",
    "observation"
  ],
  "created_at": "2026-10-25T20:30:00Z",
  "updated_at": "2026-10-26T02:04:12Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-generative-compilation-research-context.md#sections-2.2-and-4.2; commits 2bf4fae, f308f12",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T20:30:00Z"
  },
  "relationships": [
    {
      "type": "supports",
      "target": "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
      "provenance": "docs/clover-generative-compilation-research-context.md#section-4.2",
      "confidence": 0.95,
      "context_scope": "statement_boundary_checking",
      "created_at": "2026-10-25T21:00:00Z",
      "note": "Illustrates why prefix shadowing is caught at the `elif` statement boundary rather than via post-hoc regex patching."
    }
  ],
  "uncertainty_notes": [
    "The 38.5% (`579/1,502`) character saving is measured over the 6 controlled Part A trajectories in `bench_6_rust_generative_compilation.py`."
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T20:30:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial observation entry recorded from docs/clover-generative-compilation-research-context.md#sections-2.2-and-4.2; commits 2bf4fae, f308f12"
    }
  ]
}
---

# Observation: Post-Hoc Code Repair Cannot Stop Doomed Generation After an Early Syntax, Prefix, or Borrow Error

## Summary
Checking code only after a full block finishes cannot prevent the model from generating dozens of doomed tokens built on an early broken assumption; on 6 controlled Rust trajectories, statement-boundary generative compilation avoided 38.5% (`579/1,502`) of doomed characters.

## Content
Before commit 2bf4fae, `HumanEval/17` (`parse_music`) emitted a 1-character branch `music_string[i] == 'o'` at line 26 before `music_string[i:i+2] == 'o|'` at line 29, generating the entire remainder of the function before post-hoc regex patching intervened. In Rust (`bench_6_rust_generative_compilation.py` Part A), catching `E0502`, `E0382`, `E0308`, `E0061`, `E0596`, and `E0609` at the offending statement boundary saved `38.5%` (`579 / 1,502`) of doomed characters (up to `46.2%` on `E0061` and `42.3%` on `E0502`).

## Original Source Evidence
`docs/clover-generative-compilation-research-context.md` Section 4.2:
- `HumanEval/17` pre-fix: line 26 `'o'` shadowed line 29 `'o|'`.
- Benchmark 6 Part A (`N=6` controlled Rust trajectories): `6/6` errors caught mid-generation, `0` false rejections, `579/1,502` (`38.5%`) doomed chars saved.

## Revision History
- `2026-10-25T20:30:00Z` (**system_measurement**, `created`): Initial observation entry recorded from docs/clover-generative-compilation-research-context.md#sections-2.2-and-4.2; commits 2bf4fae, f308f12
