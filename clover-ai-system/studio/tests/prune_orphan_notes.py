"""Removes exchange notes whose conversation no longer exists (left by the first recall_live run, before the fix).
Usage: python tests/prune_orphan_notes.py"""
import json
import os
from pathlib import Path

repo = Path(os.environ["USERPROFILE"]) / "CloverAI" / "clover-ai-studio"
alive = {fp.stem for fp in (repo / "sessions").glob("*/*.jsonl")}
removed = 0
for fp in (repo / "growth" / "notes").glob("*/*.jsonl"):
    lines = [ln for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keep = [ln for ln in lines if json.loads(ln).get("session") in alive]
    if len(keep) != len(lines):
        removed += len(lines) - len(keep)
        fp.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8", newline="\n")
print(f"removed {removed} notes of deleted conversations; {len(alive)} conversations exist")
