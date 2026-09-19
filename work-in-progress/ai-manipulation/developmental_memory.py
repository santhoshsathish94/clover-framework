"""Unified developmental memory.

Separates raw experience, hypotheses, validated knowledge, capabilities and
uncertainties so future direction can be derived from evidence rather than
from an ever-growing undifferentiated log.
"""
import json
from pathlib import Path

from run_context import state_path

# Deliberately not developmental_memory.json: that file belongs to development_engine.py,
# and this module's demo run would inject a non-cycle record into the engine's history.
STATE = state_path("developmental_memory_example.json")

DEFAULT = {
    "experience": [],
    "hypotheses": [],
    "validated_knowledge": [],
    "capabilities": [],
    "uncertainties": [],
    "failures": [],
}

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return DEFAULT.copy()

def record_experience(state, event):
    state["experience"].append(event)

def promote_knowledge(state, claim, evidence):
    state["validated_knowledge"].append({
        "claim": claim,
        "evidence": evidence,
    })

def add_uncertainty(state, question, confidence=0.0):
    state["uncertainties"].append({
        "question": question,
        "confidence": confidence,
    })

def add_capability(state, name, evidence):
    state["capabilities"].append({
        "name": name,
        "evidence": evidence,
    })

def add_failure(state, description, context=None):
    state["failures"].append({
        "description": description,
        "context": context,
    })

def save(state):
    STATE.write_text(json.dumps(state, indent=2))

if __name__ == "__main__":
    state = load()
    record_experience(state, {
        "type": "system_start",
        "note": "Developmental memory initialized"
    })
    save(state)
    print(json.dumps(state, indent=2))
