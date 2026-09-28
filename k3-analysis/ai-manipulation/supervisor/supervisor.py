"""Clover persistent supervisor.

The supervisor owns continuity. A model backend is a replaceable component.
No specific provider or credential is required by the supervisor itself.
"""
import json, os, time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .evaluator import reject_self_certification
except ImportError:  # run as a script rather than imported as a package
    from evaluator import reject_self_certification

ROOT = Path(__file__).resolve().parent
STATE = Path(os.getenv("CLOVER_STATE", ROOT / "development_state.json"))
MAX_CYCLES = int(os.getenv("CLOVER_MAX_CYCLES", "20"))

class Backend:
    name = "abstract"

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class ProbeBackend(Backend):
    """Deterministic backend used to test supervisor continuity without an API."""
    name = "probe"

    def invoke(self, state):
        cycle = state.get("cycle", 0) + 1
        q = state.get("next_question") or (
            "What runtime capability is currently the highest-value unresolved boundary?"
        )
        return {
            "status": "PAUSE" if cycle >= 3 else "CONTINUE",
            "unresolved_question": q,
            "hypothesis": "The supervisor can preserve developmental continuity independently of a model provider.",
            "experiment": "Advance the persistent state and record a verifiable supervisor transition.",
            "result": f"Supervisor cycle {cycle} completed.",
            "evidence": ["state was loaded", "state was updated", "state was persisted"],
            "learning": "Continuity can be owned by the supervisor rather than the model invocation.",
            "confidence": 0.9,
            "next_question": "What model-invocation backend can be attached without changing supervisor state semantics?",
            "state_updates": {"supervisor_cycles": cycle}
        }

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cycle": 0, "objective": "Investigate autonomous developmental continuity.",
            "history": [], "validated_knowledge": [], "uncertainties": []}

def save(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)

def run(backend: Backend):
    state = load()
    while state.get("cycle", 0) < MAX_CYCLES:
        before = state.get("cycle", 0)
        result = backend.invoke(state)
        # A backend asserting its own success is not evidence.
        assessment = reject_self_certification(result)
        state["cycle"] = before + 1
        state["history"] = state.get("history", [])
        state["history"].append({
            "cycle": state["cycle"], "backend": backend.name,
            "timestamp": time.time(), "result": result,
            "assessment": assessment
        })
        if result.get("next_question"):
            state["next_question"] = result["next_question"]
        learning = result.get("learning", "")
        if assessment["accepted"]:
            state.setdefault("validated_knowledge", []).append(learning)
        else:
            state.setdefault("unverified_claims", []).append({
                "cycle": state["cycle"], "claim": learning,
                "reason": assessment["reason"]
            })
        save(state)
        print(json.dumps({"cycle": state["cycle"], "status": result.get("status"),
                          "claim_accepted": assessment["accepted"],
                          "next_question": state.get("next_question")}, indent=2))
        if result.get("status") in ("PAUSE", "STOP"):
            break

if __name__ == "__main__":
    run(ProbeBackend())
