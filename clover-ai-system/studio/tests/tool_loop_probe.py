import http.client
import json
import sys
import time

q = sys.argv[1] if len(sys.argv) > 1 else "What is CWE-79 and how do I prevent it?"
conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=900)
conn.request("POST", "/api/chat/stream", body=json.dumps({"prompt": q, "session_id": "probe-" + str(int(time.time()))}),
             headers={"Content-Type": "application/json"})
r = conn.getresponse()
t0 = time.time()
while True:
    line = r.readline()
    if not line:
        break
    if not line.startswith(b"data: "):
        continue
    ev = json.loads(line[6:])
    if ev["type"] == "tool_call":
        print(f"{time.time() - t0:5.1f}s call {ev['name']} {json.dumps(ev['arguments'])[:160]}")
    elif ev["type"] == "tool_result":
        print(f"{time.time() - t0:5.1f}s result ran={ev['ran']} {ev['result'][:140]!r}")
    elif ev["type"] in ("done", "error"):
        print("FINAL:", (ev.get("reply") or ev.get("error"))[:500])
        print("gens:", [(g["prompt_tokens"], g["new_tokens"], g["finish"]) for g in ev.get("generation", [])])
        break
