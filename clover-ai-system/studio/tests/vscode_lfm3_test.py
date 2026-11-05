"""VS Code path in LFM3 mode: /v1/chat/completions with VS Code-style tools (core + deferred) on the laptop server.
Usage: python tests/vscode_lfm3_test.py <out.jsonl>"""
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "vscode_lfm3_test.jsonl"
ROOT = Path.home() / "CloverAI" / "clover"


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read()), time.perf_counter() - t0


def log(rec):
    line = json.dumps(rec, ensure_ascii=False)
    print(line[:2500], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS = [
    tool("read_file", "Read the contents of a file.", {"filePath": {"type": "string"}, "startLine": {"type": "number"},
                                                        "endLine": {"type": "number"}}, ["filePath", "startLine", "endLine"]),
    tool("list_dir", "List the contents of a directory.", {"path": {"type": "string"}}, ["path"]),
    tool("grep_search", "Do a fast text search in the workspace.", {"query": {"type": "string"}, "isRegexp": {"type": "boolean"}},
         ["query", "isRegexp"]),
    tool("run_in_terminal", "Run a shell command in a terminal.", {"command": {"type": "string"}, "explanation": {"type": "string"},
                                                                   "goal": {"type": "string"}, "mode": {"type": "string"}},
         ["command", "explanation", "goal", "mode"]),
    tool("get_terminal_output", "Get output from a background terminal.", {"id": {"type": "string"}}, ["id"]),
    tool("fetch_webpage", "Fetch the main content of a web page.", {"urls": {"type": "array", "items": {"type": "string"}},
                                                                    "query": {"type": "string"}}, ["urls", "query"]),
    tool("create_directory", "Create a new directory structure in the workspace.", {"dirPath": {"type": "string"}}, ["dirPath"]),
]


def execute(name, args):
    if name == "read_file":
        lines = Path(args["filePath"]).read_text(encoding="utf-8").splitlines()
        s, e = int(args.get("startLine", 1)), int(args.get("endLine", len(lines)))
        return "\n".join(lines[s - 1:e])
    if name == "list_dir":
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in Path(args["path"]).iterdir()))
    if name == "grep_search":
        q = args["query"]
        hits = []
        for p in ROOT.glob("*.py"):
            for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if q in ln:
                    hits.append(f"{p}:{i}: {ln.strip()}")
        return "\n".join(hits[:30]) or "No matches"
    return f"(test harness: {name} not executed)"


ws = ("<environment_info>\nThe user's current OS is: Windows\n</environment_info>\n<workspace_info>\n"
      f"I am working in a workspace with the following folders:\n- {ROOT}\n</workspace_info>\n")
ask = "What does the function build_prompt in server.py do? Look at the code first."
msgs = [{"role": "system", "content": "VS Code agent instructions (long, ignored by Clover)"},
        {"role": "user", "content": ws + f"<userRequest>\n{ask}\n</userRequest>"}]
for step in range(5):
    res, dt = post({"model": "lfm3-1.7b", "messages": msgs, "tools": TOOLS})
    msg = res["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    log({"step": step + 1, "seconds": round(dt, 1), "tool_calls": calls, "content": (msg.get("content") or "")[:1500]})
    if not calls:
        break
    c = calls[0]
    out = execute(c["function"]["name"], json.loads(c["function"]["arguments"]))
    msgs += [{"role": "assistant", "content": None, "tool_calls": [c]},
             {"role": "tool", "tool_call_id": c["id"], "content": out[:20000]}]
log({"done": True})
