"""Layer 4 recall on the Iris Xe after the app-memory rendering fix: remember, then recall in a new chat."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8765"


def call(method, path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(want):
    for _ in range(150):
        st = call("GET", "/api/status?passive=1")
        if st["status"] == "ready" and st.get("engine") == want:
            return
        time.sleep(2)
    raise TimeoutError(st)


before = call("GET", "/api/settings")["settings"]
try:
    call("POST", "/api/settings", {"engine": "gpu", "layers": {"memory": True, "kv_slots": False}})
    wait_ready("gpu")
    for sid, q, kw in (("r-1", "Please remember that my dentist appointment is on 14 November at 10:30.", {"allow_once": ["memory"]}),
                       ("r-2", "When is my dentist appointment?", {})):
        t0 = time.time()
        r = call("POST", "/api/chat", dict({"prompt": q, "session_id": sid, "sampling": "greedy"}, **kw))
        print(f"[{time.time() - t0:5.1f} s] {q}\n   reply={r['reply'][:200]!r}\n   tools={[c['tool'] for c in r.get('tool_calls', [])]} "
              f"memory={[x.get('entries') for x in r.get('layers', []) if x.get('layer') == 'memory']}")
finally:
    for m in call("GET", "/api/memory")["memories"]:
        if "dentist" in m["summary"].lower():
            call("POST", "/api/memory/forget", {"id": m["id"]})
    call("POST", "/api/settings", dict(before, layers=dict(before["layers"], kv_slots=False)))
    wait_ready(before["engine"])
    print("settings now:", call("GET", "/api/settings")["settings"])
