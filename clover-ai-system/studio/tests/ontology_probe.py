"""What Layer 5 retrieves in app mode for a few questions (no model): python tests/ontology_probe.py"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "clover"))
import lfm3_layers  # noqa: E402

kit = lfm3_layers.LayerKit(BASE, memory_root=BASE / "memory")
for q in ("What is CWE-79 and how do I prevent it?", "Explain cross-site scripting (XSS).",
          "What is SQL injection?", "What are the five stages of the Clover cycle?", "What is 12 plus 30?"):
    raw = kit.cl.CloverOntologyRetriever().retrieve(q, max_entities=4, max_paths=6) if kit._ontology is None else kit._ontology.retrieve(q, max_entities=4, max_paths=6)
    block, info = kit.ontology_context(q)
    print(f"{q!r}\n  skip={kit._skip(q)} raw={[e.entities for e in raw][:4]}\n  app-mode info={info}")
