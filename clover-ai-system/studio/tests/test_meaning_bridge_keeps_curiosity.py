"""When the curiosity monitor says should_observe, the meaning bridge keeps its result and applies its action boost
(before 2026-11-09 an `array or ...` raised and the bridge silently dropped it). Applied after a live A/B in which it
repaired one answer and broke none (research/consciousness-and-subjective-experience/18-clover-integration/bridge_fix_live.py).
Usage: python tests/test_meaning_bridge_keeps_curiosity.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import lfm3_layers  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
q = "How do enzyme kinetics change with an allosteric inhibitor at high substrate?"
base = Path(tempfile.mkdtemp())
kit = lfm3_layers.LayerKit(base, memory_root=base / "growth" / "knowledge")
kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "allow"}
kit.remember("Note about this: " + q.rstrip("?") + ".", kind="observation", conversation_id="setup")
direct = kit.curiosity_state().recommend_strategy_and_voi(q, permissions=kit.permissions_map, interactive=True)
ok &= check("a fresh region: the monitor says observe", direct["should_observe"] is True)
kit.memory_context(q)
kit.ontology_context(q)
kit.enable_jepa_action_prior = True
b = kit.compute_meaning_bridge(q)
ok &= check("the bridge keeps the monitor's result", isinstance(b.get("curiosity"), dict)
            and b["curiosity"]["should_observe"] is True)
kit2 = lfm3_layers.LayerKit(base, memory_root=base / "growth" / "knowledge")
kit2.permissions_map = kit.permissions_map
kit2._curiosity = kit.curiosity_state()
kit2.memory_context(q)
kit2.ontology_context(q)
saved = kit2.curiosity_state().recommend_strategy_and_voi


def forced(observe):  # a boost on the code dimension (2), not parallel to this memory turn's prior
    def rec(*a, **k):
        boost = np.zeros(8, dtype=np.float32)
        boost[2] = 0.5
        return dict(saved(*a, **k), should_observe=observe, belief_action_boost_8=boost)
    return rec


kit2.curiosity_state().recommend_strategy_and_voi = forced(False)
plain_ap = np.asarray(kit2.compute_meaning_bridge(q)["action_prior_8"])
kit2.curiosity_state().recommend_strategy_and_voi = forced(True)
boosted_ap = np.asarray(kit2.compute_meaning_bridge(q)["action_prior_8"])
kit2.curiosity_state().recommend_strategy_and_voi = saved
ok &= check("the observe boost moves the action prior toward its dimension (it is applied)",
            boosted_ap[2] > plain_ap[2] + 1e-3)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
