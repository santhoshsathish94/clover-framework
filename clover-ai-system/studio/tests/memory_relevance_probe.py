"""Which questions pull which memory items in (reproduces the 2026-11-01 leak of a pens lesson into "free disk space")."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import lfm3_layers as L  # noqa: E402

d = Path(tempfile.mkdtemp())
kit = L.LayerKit(d, memory_root=d / "mem")
kit.learn('On a question like "And how much would 30 pens cost?"; Clover\'s checks found: two independent solutions '
          'gave 30, not 20.1.', "s1", 3)
kit.remember("My favourite colour is teal.")
st = kit.store()
for q in ("How much free disk space does this laptop have?", "What is my favourite colour?",
          "How much does a coffee cost?", "What is the capital of France?", "How much would 45 pens cost at 3 for $2?"):
    seeds = st._score_seed_entries(q)
    print(repr(q), [(round(s, 1), e[:22], r[:90]) for s, e, r in seeds[:2]], "->", bool(kit.memory_context(q)[0]))
