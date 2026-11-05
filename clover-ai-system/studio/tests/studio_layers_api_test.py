"""Studio API check through the tunnel: engine switch to the Iris Xe, Layer 4 memory (remember + recall), Layer 6 code
check, Layer 7 chat state on disk. Restores the settings it changed and forgets the note it created."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"


def call(method, path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(want=None, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = call("GET", "/api/status?passive=1")
        if st["status"] == "ready" and (want is None or st.get("engine") == want):
            return st, round(time.time() - t0)
        time.sleep(2)
    raise TimeoutError(st)


def chat(sid, prompt, **kw):
    t0 = time.time()
    r = call("POST", "/api/chat", dict({"prompt": prompt, "session_id": sid, "sampling": "greedy"}, **kw))
    gens = [(g["prompt_tokens"], g["reused_tokens"], g["new_tokens"]) for g in r.get("generation", [])]
    print(f"  [{time.time() - t0:5.1f} s] {prompt[:60]!r}\n     reply={r['reply'][:160]!r}\n     tools={[c['tool'] for c in r.get('tool_calls', [])]} "
          f"layers={[(x.get('layer'), x.get('ok', x.get('entries', ''))) for x in r.get('layers', [])]} kv={r.get('kv_slot')} gen={gens}")
    return r


before = call("GET", "/api/settings")["settings"]
print("settings before:", before)
try:
    call("POST", "/api/settings", {"engine": "gpu", "layers": {"memory": True, "compiler": True, "ontology": True}})
    st, secs = wait_ready("gpu")
    print(f"switched to {st['engine']} in {secs} s")
    r = chat("t-mem-1", "Please remember that my favourite colour is teal.", allow_once=["memory"])
    notes = call("GET", "/api/memory")["memories"]
    print("memory now:", [(m["id"], m["summary"]) for m in notes[:3]])
    chat("t-mem-2", "What is my favourite colour? Check what you remember.")
    chat("t-code", "Write a Python function add(a, b) that returns a + b. Include a doctest.", tools=False)
    chat("t-kv-a", "Explain in two sentences what a hash table is.", tools=False)
    chat("t-kv-b", "Name three planets.", tools=False)
    chat("t-kv-a", "Now give one example use of it.", tools=False)
finally:
    for m in call("GET", "/api/memory")["memories"]:
        if "teal" in m["summary"]:
            print("forget test note:", call("POST", "/api/memory/forget", {"id": m["id"]})["ok"])
    call("POST", "/api/settings", before)
    st, secs = wait_ready(before["engine"])
    print("settings restored:", call("GET", "/api/settings")["settings"], f"engine back in {secs} s")
    sys.exit(0)
