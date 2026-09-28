"""Local chat-completions backend.

`local_backend.py` speaks the Responses shape, which LM Studio serves but Ollama
does not. Ollama, llama.cpp's server and vLLM all expose OpenAI-compatible
/v1/chat/completions instead, so this covers that route.

Small local models frequently return prose or fenced JSON rather than a bare
object. That is handled here and reported as a parse failure rather than being
allowed to crash the cycle, because "the worker could not produce usable output"
is itself a result worth recording.
"""
from __future__ import annotations

import json
import os
import re
from urllib.request import Request, urlopen

INSTRUCTIONS = (
    "You are the intelligence component of a persistent developmental runtime. "
    "Inspect the supplied state and select the highest-value unresolved question. "
    "Reply with a single JSON object and nothing else, using the keys: status "
    "(CONTINUE or PAUSE), learning, next_question. Never claim an experiment "
    "occurred unless the environment supplied evidence."
)

def extract_json(text):
    """Models wrap objects in prose or code fences often enough to be worth handling."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start >= 0:
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None

class ChatCompletionsBackend:
    def __init__(self, model, base_url=None, api_key=None):
        self.model = model
        self.name = f"local:{model}"
        self.base_url = (base_url or os.getenv(
            "CLOVER_LOCAL_CHAT_URL", "http://127.0.0.1:11434/v1/chat/completions"
        )).rstrip("/")
        self.api_key = api_key or os.getenv("CLOVER_LOCAL_API_KEY")

    def invoke(self, state):
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps({
                    "objective": state.get("objective"),
                    "cycle": state.get("cycle", 0),
                    "next_question": state.get("next_question"),
                    "validated_knowledge": state.get("validated_knowledge", [])[-20:],
                    "uncertainties": state.get("uncertainties", [])[-20:],
                })},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        request = Request(self.base_url, data=json.dumps(payload).encode(),
                          headers=headers, method="POST")
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode())

        raw = body["choices"][0]["message"]["content"]
        parsed = extract_json(raw)
        if parsed is None:
            return {"status": "PAUSE", "learning": "",
                    "parse_error": "worker did not return JSON",
                    "raw_output": raw[:2000],
                    "inherited_question": state.get("next_question")}

        parsed.setdefault("status", "CONTINUE")
        parsed["inherited_question"] = state.get("next_question")
        return parsed
