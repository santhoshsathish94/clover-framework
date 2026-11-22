---
name: systems-python-swe
description: General Python 3 software engineering conventions for concise, self-contained library modules and boundary safety.
---

# Python 3 Library Engineering Playbook

1. **Self-Contained Module Structure:**
   - Use only the Python 3 standard library unless external packages are explicitly requested.
   - Keep implementations concise and complete inside a single ````python` code block without top-level CLI `sys.exit()` calls or verbose docstrings that risk output truncation.
2. **State & Boundary Discipline:**
   - Explicitly handle empty inputs, single-element inputs, invalid header/parameter ranges (`raise ValueError` or the specified exception class), and deterministic tie-breaking/sorting when order matters.
   - Separate lexer/parser states cleanly so delimiters inside quoted literals or comments never corrupt outer state.

