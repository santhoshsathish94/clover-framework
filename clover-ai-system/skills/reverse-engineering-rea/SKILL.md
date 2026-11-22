---
name: reverse-engineering-rea
description: General static analysis workflow for translating disassembled native functions and bundled JavaScript modules into clean Python 3 reference code.
---

# Static Reverse-Engineering Playbook

1. **Evidence-First Reconstruction:**
   - Ground every constant, branch condition, struct field width, and error path strictly in the provided static analysis / disassembly / bundle source rather than guessing defaults.
2. **Fixed-Width Semantics in Python:**
   - Remember Python `int` is arbitrary-precision: mask fixed-width C/JS arithmetic (`8`, `16`, `32`, or `64` bits) after shifts, multiplies, and additions, and preserve signed vs unsigned integer conversions faithfully.
3. **Control-Flow Fidelity:**
   - Trace early-return error checks, loop bounds, and state transitions in exact program order before writing the Python implementation.

