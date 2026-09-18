"""OpenAI-compatible local model backend.

Works with local servers exposing /v1/responses, such as compatible
OSS model runtimes. No OpenAI API key is required when the local
endpoint itself requires no authentication.
"""
import json, os
from urllib.request import Request, urlopen

class LocalResponsesBackend:
    name = "local-responses"

    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = (base_url or os.getenv(
            "CLOVER_LOCAL_MODEL_URL", "http://localhost:11434/v1/responses"
        )).rstrip("/")
        self.model = model or os.getenv("CLOVER_LOCAL_MODEL", "gpt-oss")
        self.api_key = api_key or os.getenv("CLOVER_LOCAL_API_KEY")

    def invoke(self, state):
        payload = {
            "model": self.model,
            "instructions": (
                "You are the intelligence component of a persistent developmental "
                "runtime. Inspect state, choose the highest-value unresolved question, "
                "and return JSON. Never claim an experiment occurred without evidence."
            ),
            "input": json.dumps({
                "objective": state.get("objective"),
                "cycle": state.get("cycle", 0),
                "next_question": state.get("next_question"),
                "validated_knowledge": state.get("validated_knowledge", [])[-20:],
                "uncertainties": state.get("uncertainties", [])[-20:]
            })
        }
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = Request(self.base_url, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode())
        text = body.get("output_text")
        if text is None:
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text")
                        break
        if not text:
            raise RuntimeError("Local model returned no output_text")
        return json.loads(text)
