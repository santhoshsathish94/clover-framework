---
{
  "schema_version": "1.0",
  "id": "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
  "title": "Short-Prefix Shadowing in Sequential Dispatch and String Tokenizers",
  "entry_type": "concept",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "prefix-shadow",
    "longest-prefix-first-matching",
    "generative-compilation",
    "statement-boundary-soundness"
  ],
  "tags": [
    "principle",
    "compiler",
    "tokenizer",
    "python",
    "humaneval",
    "bug-pattern"
  ],
  "created_at": "2026-10-25T21:00:00Z",
  "updated_at": "2026-10-26T04:00:00Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "docs/clover-generative-compilation-research-context.md#section-4.2",
    "actor": "system_measurement",
    "recorded_at": "2026-10-25T21:00:00Z"
  },
  "relationships": [
    {
      "type": "supports",
      "target": "MEM-EXP-GENERATIVE-COMPILATION-ROLLBACK-001",
      "provenance": "docs/clover-generative-compilation-research-context.md#section-4.2",
      "confidence": 1.0,
      "context_scope": "generative_compilation_humaneval",
      "created_at": "2026-10-25T21:00:00Z",
      "note": "Directly enabled native Rollback #1 self-correction on HumanEval/17."
    },
    {
      "type": "similar_to",
      "target": "MEM-OBS-ONTORAG-FANOUT-FLOODING-001",
      "provenance": "docs/clover-ontology-graphrag-research-context.md#finding-2",
      "confidence": 0.75,
      "context_scope": "structural_priority_ordering",
      "created_at": "2026-10-25T21:30:00Z",
      "note": "Structural analogue: sorting by 1-hop length before edge priority lets short leaf edges shadow 2-hop causal chains, just as 1-char prefixes shadow 2-char tokens."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-25T21:00:00Z",
      "actor": "system_measurement",
      "action": "created",
      "reason": "Initial concept entry recorded from docs/clover-generative-compilation-research-context.md#section-4.2"
    }
  ]
}
---

# Short-Prefix Shadowing in Sequential Dispatch and String Tokenizers

## Summary
In sequential `if/elif` dispatch or hand-written character-index tokenizers, testing a shorter prefix (such as `s[i] == 'o'`) before a longer prefix starting with the same character (`s[i:i+2] == 'o|'`) renders the longer branch unreachable dead code; always order longest-prefix-first.

## Content
When a scanner inspects an unsplit string at index `i` using ordered conditional branches, any branch that matches a 1-character prefix (`s[i] == 'o'`) and advances `i += 1` intercepts every occurrence of a multi-character token (`'o|'`) that shares that prefix. In Clover Kimi K3's `CloverPythonGenerativeCompiler` (`_PythonStatementBoundaryAnalyzer`), this structural defect is detected at statement boundaries via `E_PY_PREFIX_SHADOW` in 0.68-0.70 ms as soon as the `elif` line is emitted, rolling back the loop header and prompting `HybridAgent` to emit longest-prefix-first dispatch (`s[i:i+2] == 'o|'` before `s[i] == 'o'`).

## Original Source Evidence
Live verification on `HumanEval/17` (`parse_music`, `docs/clover-generative-compilation-research-context.md`): `CloverPythonGenerativeCompiler` caught `error[E_PY_PREFIX_SHADOW] at line 26, col 9` (`music_string[i] == 'o'` before `music_string[i:i+2] == 'o|'`) in `0.68-0.70 ms` and solved `parse_music` natively on Rollback #1.

## Revision History
- `2026-10-25T21:00:00Z` (**system_measurement**, `created`): Initial concept entry recorded from docs/clover-generative-compilation-research-context.md#section-4.2
