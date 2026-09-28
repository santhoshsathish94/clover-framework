"""OpenAI Responses API backend for the Clover supervisor.

Credentials are intentionally external to the repository:
OPENAI_API_KEY must be supplied by the runtime.
"""
import json, os

class OpenAIBackend:
    name = "openai-responses"

    def __init__(self, model=None):
        # An unset CI variable arrives as "", which getenv's default would not replace.
        self.model = model or os.getenv("CLOVER_MODEL") or "gpt-5.6-luna"
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not available. The supervisor itself does not "
                "require a key; this backend does for hosted model inference."
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=key)

    def invoke(self, state):
        prompt = {
            "objective": state.get("objective"),
            "cycle": state.get("cycle", 0),
            "next_question": state.get("next_question"),
            "validated_knowledge": state.get("validated_knowledge", [])[-20:],
            "uncertainties": state.get("uncertainties", [])[-20:],
        }

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are the intelligence component of a persistent developmental "
                "runtime. Inspect the supplied state, select the highest-value "
                "unresolved question, and return JSON. Never claim an experiment "
                "occurred unless the environment supplied evidence."
            ),
            input=json.dumps(prompt),
        )
        raw = response.output_text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "PAUSE",
                "parse_error": "Backend did not return JSON.",
                "raw_output": raw[:2000],
            }
