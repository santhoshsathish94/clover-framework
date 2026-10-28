"""Makes write_guard importable when pytest is run from the repository root.

The package directory is hyphenated, so it cannot be imported as `reference.runtime_enforcement`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
