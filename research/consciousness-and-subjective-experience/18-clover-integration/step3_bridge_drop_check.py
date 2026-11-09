"""Does the meaning bridge drop the curiosity result whenever it says should_observe=True? (found while measuring step 3)
Real LayerKit + memory + curiosity, fresh state, one user note matching the question. Liveness: the same turn with the
`or` replaced by an explicit None check keeps the result."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import lfm3_layers as LL  # noqa: E402

q = "How do enzyme kinetics change with an allosteric inhibitor at high substrate?"
out = {}
base = Path(tempfile.mkdtemp(prefix="consc-bridge-"))
kit = LL.LayerKit(base, memory_root=base / "growth" / "knowledge")
kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "allow"}
kit.remember("Note about this: " + q.rstrip("?") + ".", kind="observation", conversation_id="setup")
kit.memory_context(q)
kit.ontology_context(q)
direct = kit.curiosity_state().recommend_strategy_and_voi(q, permissions=kit.permissions_map, interactive=True)
out["direct_should_observe"] = bool(direct["should_observe"])
out["boost_type"] = type(direct["belief_action_boost_8"]).__name__
try:
    np.asarray(direct["belief_action_boost_8"] or np.zeros(8), dtype=np.float32)
    out["the_or_expression"] = "ok"
except ValueError as exc:
    out["the_or_expression"] = f"ValueError: {exc}"
bridge = kit.compute_meaning_bridge(q)
out["bridge_curiosity"] = bridge.get("curiosity")
out["bridge_action_prior_present"] = bridge.get("action_prior_8") is not None

src = (CLOVER / "lfm3_layers.py").read_text(encoding="utf-8")
bad = 'np.asarray(curiosity_info.get("belief_action_boost_8") or np.zeros(8), dtype=np.float32)'
fixed = ('np.asarray(curiosity_info.get("belief_action_boost_8") if curiosity_info.get("belief_action_boost_8") is not None '
         'else np.zeros(8), dtype=np.float32)')
out["bad_line_in_source"] = src.count(bad)
ns = {}
patched = src.replace(bad, fixed)
import types  # noqa: E402

mod = types.ModuleType("lfm3_layers_patched")
mod.__file__ = str(CLOVER / "lfm3_layers.py")
exec(compile(patched, mod.__file__, "exec"), mod.__dict__)
kit2 = mod.LayerKit(base, memory_root=base / "growth" / "knowledge")
kit2.permissions_map = kit.permissions_map
kit2._curiosity = kit.curiosity_state()
kit2.memory_context(q)
kit2.ontology_context(q)
b2 = kit2.compute_meaning_bridge(q)
out["liveness_with_explicit_none_check"] = b2.get("curiosity")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "step3_bridge_drop_check.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=2, default=str))
