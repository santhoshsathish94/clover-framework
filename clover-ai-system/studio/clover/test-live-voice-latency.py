"""Measure an existing Clover server without starting or modifying another server."""
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

ORIGIN = os.environ.get("CLOVER_TEST_ORIGIN", "http://127.0.0.1:7100").rstrip("/")
OUT = Path(os.environ.get("CLOVER_LATENCY_OUT", str(Path(__file__).parent / "test-artifacts" / "live-voice-latency.json")))
sessions = []
results = {"origin": ORIGIN, "measurements": {}, "replies": {}, "events": {}}


def request(route, body=None, timeout=240):
    started = time.perf_counter()
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(ORIGIN + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
    elapsed = time.perf_counter() - started
    if "json" in content_type:
        value = json.loads(payload)
    else:
        value = payload
    return value, elapsed


def chat(label, prompt, audio=None, conversation=False):
    session_id = f"latency_{label}_{uuid.uuid4().hex}"
    sessions.append(session_id)
    media = [] if audio is None else [{"kind": "audio", "name": "latency-question.wav", "data": __import__("base64").b64encode(audio).decode()}]
    body = {"prompt": prompt, "session_id": session_id, "model": "omni", "history": [], "media": media,
            "conversation": conversation, "sampling": "general", "thinking": False,
            "max_new_tokens": 256 if conversation else 64}
    started = time.perf_counter()
    req = urllib.request.Request(ORIGIN + "/api/chat/stream", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    events = []
    first_answer = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            elapsed = time.perf_counter() - started
            events.append({key: event.get(key) for key in ("type", "stage", "kind", "layer", "s", "codes", "levels", "windows", "ms", "tokens")} | {"seconds": round(elapsed, 3)})
            if event.get("type") == "token" and event.get("kind") == "answer" and first_answer is None:
                first_answer = elapsed
            if event.get("type") == "done":
                reply = event.get("reply") or ""
            if event.get("type") == "error":
                raise RuntimeError(event.get("error"))
    results["measurements"][label] = {"first_answer_s": round(first_answer, 3) if first_answer is not None else None,
                                      "done_s": round(time.perf_counter() - started, 3)}
    results["replies"][label] = reply
    results["events"][label] = events
    print(label, results["measurements"][label], repr(reply), flush=True)


try:
    status, _ = request("/api/status", timeout=10)
    if status.get("status") != "ready":
        raise RuntimeError(f"Live model is not ready: {status}")
    results["model"] = {key: status.get(key) for key in ("model", "engine", "inputs")}

    chat("text", "Reply with exactly one short greeting.")

    parts, results["measurements"]["tts_parts_s"] = request("/api/tts/parts", {"text": "Please say exactly Hello."})
    # Fixture speech uses the same resident-model path as conversation; do not start the removed warm-up competitor.
    question_wav, results["measurements"]["tts_question_s"] = request(
        "/api/tts", {"text": parts["parts"][0], "voice": "female", "conversation": True})
    results["question_wav_bytes"] = len(question_wav)
    print("TTS question", results["measurements"]["tts_question_s"], "seconds", len(question_wav), "bytes", flush=True)

    chat("voice", "", question_wav, conversation=True)
    chat("text_after_voice", "Reply with exactly one short greeting.")
    answer_wav, results["measurements"]["tts_answer_s"] = request(
        "/api/tts", {"text": "Hello! How can I help?", "voice": "female", "conversation": True})
    results["answer_wav_bytes"] = len(answer_wav)
    print("TTS answer", results["measurements"]["tts_answer_s"], "seconds", len(answer_wav), "bytes", flush=True)
finally:
    for session_id in sessions:
        try:
            request("/api/session/forget", {"session_id": session_id, "delete": True}, timeout=15)
        except Exception as exc:
            results.setdefault("cleanup_errors", []).append(str(exc))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)
