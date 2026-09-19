import json, os, time
from pathlib import Path
from openai import OpenAI

MODEL = os.getenv("CLOVER_MODEL", "gpt-6-astra")
MAX_CYCLES = int(os.getenv("CLOVER_MAX_CYCLES", "20"))
STATE_FILE = Path(os.getenv("CLOVER_STATE", "development_state.json"))

INSTRUCTIONS = """You are an investigator inside a persistent developmental process.
Do not wait for a human to provide the next question. Inspect the developmental
state, identify the highest-value unresolved uncertainty, investigate it, and
produce a verifiable state transition.

Distinguish observation, inference, hypothesis, experiment, result, and learning.
Do not claim learning without evidence. Prefer falsifiable experiments. Preserve
uncertainty. Generated sub-goals must remain traceable to the initial objective.
If no valuable investigation remains, return PAUSE. Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["CONTINUE", "PAUSE", "STOP"]},
        "unresolved_question": {"type": "string"},
        "hypothesis": {"type": "string"},
        "experiment": {"type": "string"},
        "result": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "learning": {"type": "string"},
        "confidence": {"type": "number"},
        "next_question": {"type": "string"},
        "state_updates": {"type": "object", "additionalProperties": True}
    },
    "required": ["status","unresolved_question","hypothesis","experiment",
                 "result","evidence","learning","confidence","next_question",
                 "state_updates"],
    "additionalProperties": False
}

def initial_state():
    return {
        "cycle": 0,
        "objective": "Investigate whether a persistent AI process can turn experience into validated cumulative development.",
        "validated_knowledge": [], "hypotheses": [], "experiments": [],
        "capabilities": [], "uncertainties": [], "self_model": [], "history": []
    }

def load_state():
    if not STATE_FILE.exists():
        s = initial_state()
        save_state(s)
        return s
    return json.loads(STATE_FILE.read_text())

def save_state(s):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(STATE_FILE)

def cycle(client, state):
    prompt = {"developmental_state": state,
              "instruction": "Run exactly one developmental cycle and choose the next investigation yourself."}
    r = client.responses.create(
        model=MODEL, instructions=INSTRUCTIONS, input=json.dumps(prompt),
        text={"format": {"type": "json_schema", "name": "development_cycle",
                         "schema": SCHEMA, "strict": True}}
    )
    result = json.loads(r.output_text)
    state["cycle"] += 1
    state["history"].append({
        "cycle": state["cycle"], **{k: result[k] for k in
        ["status","unresolved_question","hypothesis","experiment","result",
         "evidence","learning","confidence"]}
    })
    for key in ["validated_knowledge","hypotheses","experiments",
                "capabilities","uncertainties","self_model"]:
        values = result["state_updates"].get(key, [])
        if isinstance(values, list):
            state[key].extend(values)
    state["next_question"] = result["next_question"]
    save_state(state)
    return result

def main():
    client = OpenAI()
    state = load_state()
    for _ in range(MAX_CYCLES):
        r = cycle(client, state)
        print(json.dumps({"cycle": state["cycle"], "status": r["status"],
                          "learning": r["learning"],
                          "next_question": r["next_question"]}, indent=2))
        if r["status"] != "CONTINUE":
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
