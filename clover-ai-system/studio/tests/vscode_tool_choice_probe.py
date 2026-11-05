"""Probe: which tools a Studio turn offers and what the model calls, for prompts that ask for VS Code.
Usage: python tests/vscode_tool_choice_probe.py [ask|always]   (sets the Tools setting for the run, then restores it)"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
MD = os.path.join(os.environ["USERPROFILE"], "CloverAI", "tmp", "vscode-bridge-test.md")
MODE = sys.argv[1] if len(sys.argv) > 1 else "ask"
ALLOW = [a for a in sys.argv[2:3] if a]  # e.g. vscode: run the tool (allow once)
PROMPTS = [f"Open the file {MD} in VS Code."] * (2 if ALLOW else 1) + ([] if ALLOW else ["Install the VS Code extension clover-framework.clover-local."])


def call(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


before = call("GET", "/api/settings")["settings"]["tools"]
call("POST", "/api/settings", {"tools": MODE})
try:
    for p in PROMPTS:
        t0 = time.perf_counter()
        res = call("POST", "/api/chat", {"prompt": p, "session_id": f"probe-vsc-{int(time.time() * 1000)}",
                                         "allow_once": ALLOW})
        print(json.dumps({"tools_mode": MODE, "allow": ALLOW, "prompt": p[:80], "seconds": round(time.perf_counter() - t0, 1),
                          "tools_offered": res.get("tools_offered"), "permission_request": res.get("permission_request"),
                          "calls": [{"tool": c.get("tool"), "args": c.get("arguments"), "ran": c.get("ran"),
                                     "result": str(c.get("result"))[:200]} for c in res.get("tool_calls") or []],
                          "generation": [{k: g.get(k) for k in ("prompt_tokens", "new_tokens", "finish", "prefill_s", "decode_s")}
                                         for g in res.get("generation") or []],
                          "layers": [{k: (str(v)[:160] if isinstance(v, (list, dict, str)) else v) for k, v in x.items()}
                                     for x in res.get("layers") or []],
                          "reply": (res.get("reply") or "")[:300]}, ensure_ascii=False), flush=True)
finally:
    call("POST", "/api/settings", {"tools": before})
    print("tools setting restored to", call("GET", "/api/settings")["settings"]["tools"])
