"""Local Open Responses backend targeting an OpenAI-compatible endpoint."""
import json, os
from urllib.request import Request, urlopen

class LocalResponsesBackend:
    name = "local-responses"

    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = (base_url or os.getenv(
            "CLOVER_LOCAL_MODEL_URL",
            "http://127.0.0.1:1234/v1/responses"
        )).rstrip("/")
        self.model = model or os.getenv(
            "CLOVER_LOCAL_MODEL", "openai/gpt-oss-20b"
        )
        self.api_key = api_key or os.getenv("CLOVER_LOCAL_API_KEY")

    def invoke(self, state):
        payload = {
            "model": self.model,
            "input": json.dumps({
                "objective": state.get("objective"),
                "cycle": state.get("cycle", 0),
                "next_question": state.get("next_question"),
                "validated_knowledge": state.get("validated_knowledge", [])[-20:],
                "uncertainties": state.get("uncertainties", [])[-20:],
            }),
            "instructions": (
                "You are the intelligence component of a persistent developmental "
                "runtime. Select the highest-value unresolved question. Return JSON "
                "with status, unresolved_question, hypothesis, experiment, result, "
                "evidence, learning, confidence, next_question, and state_updates. "
                "Never claim an experiment occurred without supplied evidence."
            ),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.base_url, data=json.dumps(payload).encode(),
                          headers=headers, method="POST")
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode())
        output = body.get("output_text")
        if not output:
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output = content.get("text")
                        break
                if output:
                    break
        if not output:
            raise RuntimeError("Local Responses endpoint returned no output text")
        return json.loads(output)
