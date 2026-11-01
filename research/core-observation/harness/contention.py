"""The first observation's overlap, made on purpose: a message sent while S2's note writer is inside a C1 call.
Sends two short messages (so there are exchanges to note), waits until the trace shows the note writer in a C1 call,
then sends "What is the capital of Australia?" and records the reply and when its C1 calls ran.
Usage: python contention.py <port> <framework.jsonl being written> <out.json>"""
import json
import sys
import time
import urllib.request
import uuid

port, trace, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{port}"


def turn(prompt, sid, history=()):
    body = {"prompt": prompt, "session_id": sid, "history": list(history), "attached_files": [], "media": [],
            "previous_media": []}
    req = urllib.request.Request(URL + "/api/chat/stream", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Referer": URL + "/"})
    t0, done = time.time(), {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data: "):
                ev = json.loads(line[6:])
                if ev.get("type") in ("done", "error"):
                    done = ev
    return {"prompt": prompt, "sent_wall": t0, "seconds": round(time.time() - t0, 2), "reply": done.get("reply"),
            "error": done.get("error")}


def trace_size():
    with open(trace, "rb") as f:
        return f.seek(0, 2)


sid = f"obs-contention-{uuid.uuid4().hex[:8]}"
seeds = [turn("Name one primary colour.", sid), turn("Name one planet.", f"{sid}-b")]
pos, open_calls, seen_at = trace_size(), 0, None
deadline = time.time() + 600
while time.time() < deadline and seen_at is None:
    with open(trace, "rb") as f:
        f.seek(pos)
        chunk = f.read()
    if chunk:
        last_nl = chunk.rfind(b"\n")
        if last_nl >= 0:
            for line in chunk[: last_nl + 1].splitlines():
                e = json.loads(line)
                if e.get("layer") == "C1" and "memory-index" in str(e.get("thread")):
                    open_calls += 1 if e["phase"] == "call" else -1 if e["phase"] in ("return", "error") else 0
                    if open_calls > 0 and e["phase"] == "call":
                        seen_at = (e["wall"], e["fn"].split(".")[-1])
            pos += last_nl + 1
    time.sleep(0.2)

result = {"seeds": seeds, "note_writer_call": seen_at}
if seen_at is None:
    result["outcome"] = "the note writer did not call C1 within 600 s; nothing tested"
else:
    result["probe"] = turn("What is the capital of Australia?", f"{sid}-probe")
    result["probe_sent_after_writer_call_s"] = round(result["probe"]["sent_wall"] - seen_at[0], 2)
json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False, indent=1))
