"""Shared run setup for the standalone experiments in this directory.

Every script here persists state and appends to a history, so each is meant to
be run more than once. Seeding from a constant made each run identical, which
looked like accumulation while nothing accumulated. The seed therefore varies
per run by default and is written to run_log.json, so any run can be reproduced
by setting CLOVER_SEED.

State paths are anchored to this directory so that running a script from the
repository root does not scatter state files there.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_LOG = ROOT / "run_log.json"

def state_path(filename):
    return ROOT / filename

def seed_for(experiment):
    """Seeds the global RNG and records the seed. Returns it so callers can report it."""
    fixed = os.getenv("CLOVER_SEED")
    seed = int(fixed) if fixed else random.SystemRandom().randrange(2**31)
    random.seed(seed)

    runs = []
    if RUN_LOG.exists():
        try:
            runs = json.loads(RUN_LOG.read_text())
        except (json.JSONDecodeError, OSError):
            runs = []
    runs.append({
        "experiment": experiment,
        "seed": seed,
        "reproducible_with": f"CLOVER_SEED={seed}",
        "at": datetime.now(timezone.utc).isoformat(),
    })
    RUN_LOG.write_text(json.dumps(runs[-200:], indent=2))
    return seed
