"""Print the Layer 5 block the app adds for a Clover question: python tests/ontology_render_probe.py"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "clover"))
import lfm3_layers  # noqa: E402

kit = lfm3_layers.LayerKit(BASE, memory_root=BASE / "memory")
block, info = kit.ontology_context("What does Clover say about who owns Direction?")
print(block[:2500] if block else block)
