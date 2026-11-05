"""Clover layers on LFM3-Omni-2B through the Studio API (tunnel), one at a time: L4 memory, L5 ontology, L6 code check,
L7 chat state on disk. Restores the settings it changes and forgets the note it creates.
Run on the dev machine: python tests/studio_omni_layers_test.py"""
import http.client
import json
import time
import urllib.request

HOST, PORT = "127.0.0.1", 8765


def call(method, path, body=None, timeout=600):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(sid, prompt, decide="once"):
    t0 = time.time()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=900)
    conn.request("POST", "/api/chat/stream", body=json.dumps({"prompt": prompt, "session_id": sid}),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    done = None
    while done is None:
        line = r.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            ev = json.loads(line[6:])
            if ev["type"] == "permission":
                call("POST", "/api/decision", {"turn_id": ev["turn_id"], "request_id": ev["request_id"], "decision": decide})
            if ev["type"] in ("done", "error"):
                done = ev
    conn.close()
    return done or {"error": "stream ended"}, round(time.time() - t0, 1)


def show(name, res, secs, expect=None, check=None):
    reply = res.get("reply") or res.get("error") or ""
    gens = [(g["prompt_tokens"], g["reused_tokens"], g["new_tokens"], g["prefill_s"]) for g in res.get("generation", [])]
    ok = (expect is None or any(e.lower() in reply.lower() for e in expect)) and (check is None or check(res))
    print(f"[{'ok' if ok else 'CHECK'}] {name} {secs} s\n    reply={reply[:240]!r}\n    tools={[t['tool'] for t in res.get('tool_calls', [])]}"
          f" layers={json.dumps(res.get('layers'), default=str)[:400]} kv={res.get('kv_slot')} gen={gens}", flush=True)
    return ok


def layers_on(**on):
    call("POST", "/api/settings", {"layers": {k: bool(on.get(k, False)) for k in ("memory", "ontology", "compiler", "ncm", "kv_slots")}})


before = call("GET", "/api/settings")["settings"]
print("settings before:", before)
results, run = [], str(int(time.time()))
try:
    # L4 memory
    layers_on(memory=True)
    res, s = stream("L4a-" + run, "Please remember that my favourite colour is teal.")
    results.append(show("L4 remember", res, s, check=lambda r: any(t["tool"] == "remember" and t["ran"] for t in r.get("tool_calls", []))))
    res, s = stream("L4b-" + run, "What is my favourite colour?")
    results.append(show("L4 recall in a new chat", res, s, expect=["teal"],
                        check=lambda r: any(l.get("layer") == "memory" and l.get("entries") for l in r.get("layers", []))))
    # L5 ontology
    layers_on(ontology=True)
    res, s = stream("L5a-" + run, "What is CWE-79 and how do I prevent it?", decide="deny")
    results.append(show("L5 ontology question", res, s, check=lambda r: any(l.get("layer") == "ontology" for l in r.get("layers", []))))
    res, s = stream("L5b-" + run, "What is 12 plus 30? Reply with the number.", decide="deny")
    results.append(show("L5 unrelated question (no block, answer intact)", res, s, expect=["42"],
                        check=lambda r: not any(l.get("layer") == "ontology" for l in r.get("layers", []))))
    # L6 code check
    layers_on(compiler=True)
    res, s = stream("L6-" + run, "Write a Python function add(a, b) that returns a + b, with a doctest. Do not run it.", decide="deny")
    results.append(show("L6 code check", res, s, expect=["def add"],
                        check=lambda r: any(l.get("layer") == "compiler" for l in r.get("layers", []))))
    # L7 chat state: a long chat A, a different chat B, then back to A
    layers_on(kv_slots=True)
    long_text = " ".join(f"Item {i}: the {['red', 'green', 'blue', 'amber'][i % 4]} box holds {i * 3} marbles." for i in range(160))
    res, s = stream("L7a-" + run, "Here is an inventory list:\n" + long_text + "\nHow many marbles does item 7 hold?", decide="deny")
    results.append(show("L7 chat A (long)", res, s, expect=["21"]))
    res, s = stream("L7b-" + run, "Name three planets.", decide="deny")
    results.append(show("L7 chat B (switch)", res, s))
    res, s = stream("L7a-" + run, "And item 9?", decide="deny")
    results.append(show("L7 back to chat A", res, s, expect=["27"],
                        check=lambda r: bool((r.get("kv_slot") or {}).get("restored")) and r["generation"][0]["reused_tokens"] > 1000))
finally:
    for m in call("GET", "/api/memory")["memories"]:
        if "teal" in (m.get("summary") or ""):
            print("forget test note:", call("POST", "/api/memory/forget", {"id": m["id"]})["ok"])
    call("POST", "/api/settings", {"layers": before["layers"]})
    print("settings restored:", call("GET", "/api/settings")["settings"])
print(f"RESULT {sum(map(bool, results))}/{len(results)} checks passed", flush=True)
