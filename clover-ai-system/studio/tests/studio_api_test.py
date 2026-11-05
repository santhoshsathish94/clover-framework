"""Live checks of the Clover AI Studio server on the laptop (127.0.0.1:8765). Writes one JSON line per check.

Security: cross-origin POST, wrong Host, non-JSON POST, CSP nonce on the page.
Permissions: a tool with mode `ask` stops the turn before running (permission_request); allow_once runs it.
VS Code: /v1/chat/completions with VS Code-style tools returns a tool call; with the tool result, a final answer.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "studio_api_test.jsonl"


def req(method, path, body=None, headers=None, timeout=3600):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(), time.perf_counter() - t0


def log(name, ok, **kw):
    rec = {"check": name, "ok": bool(ok), **kw}
    print(json.dumps(rec, ensure_ascii=False)[:1500], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    # --- security ---
    code, _, raw, _ = req("POST", "/api/permissions", {"permissions": {"code": "allow"}}, {"Origin": "http://evil.example"})
    log("cross-origin POST refused", code == 403, status=code)
    code, _, raw, _ = req("POST", "/api/permissions", b'{"permissions": {"code": "allow"}}', {"Content-Type": "text/plain"})
    log("non-JSON POST refused", code == 400, status=code)
    code, _, raw, _ = req("POST", "/api/permissions", {"permissions": {"code": "allow"}}, {"Sec-Fetch-Site": "cross-site"})
    log("Sec-Fetch-Site cross-site refused", code == 403, status=code)
    code, _, raw, _ = req("GET", "/api/health", headers={"Host": "evil.example:8765"})
    log("foreign Host refused", code == 403, status=code)
    code, hdr, raw, _ = req("GET", "/")
    csp = hdr.get("Content-Security-Policy", "")
    nonce = csp.split("'nonce-")[1].split("'")[0] if "'nonce-" in csp else ""
    log("page has CSP nonce", code == 200 and nonce and f'nonce="{nonce}"' in raw.decode("utf-8"), status=code, csp=csp[:120])
    code, _, raw, _ = req("GET", "/api/permissions")
    perms = json.loads(raw)["permissions"]
    log("permissions default ask", all(v["mode"] == "ask" for v in perms.values()), modes={k: v["mode"] for k, v in perms.items()})

    # --- permission flow: code ---
    prompt = "If I invest $5,000 at 6% interest compounded monthly for 10 years, what is the final amount?"
    code, _, raw, dt = req("POST", "/api/chat", {"prompt": prompt, "session_id": "apitest_code"})
    r = json.loads(raw)
    pr = r.get("permission_request") or {}
    log("code tool asks first", code == 200 and pr.get("tool") == "code" and "P = 5000" in pr.get("detail", ""),
        status=code, seconds=round(dt, 1), request=pr)
    code, _, raw, dt = req("POST", "/api/chat", {"prompt": prompt, "session_id": "apitest_code", "allow_once": ["code"]})
    r = json.loads(raw)
    tools = [t.get("tool") for t in r.get("tool_calls") or []]
    py_res = " ".join(str(t.get("result")) for t in r.get("tool_calls") or [] if t.get("tool") == "python")
    log("allow once runs the code", code == 200 and "python" in tools and "9,096.98" in py_res, status=code,
        seconds=round(dt, 1), plan_s=round((r.get("plan_lat_ms") or 0) / 1000, 1),
        render_s=round((r.get("k3_moe_plus_ncm_ms") or 0) / 1000, 1), tool_result=py_res[:300],
        reply=(r.get("reply") or "")[:600], events=r.get("permission_events"))

    # --- permission flow: internet (ask only; no network call is made) ---
    code, _, raw, dt = req("POST", "/api/chat", {"prompt": "Search the web for recent news about Liquid AI LFM models.",
                                                 "session_id": "apitest_web"})
    r = json.loads(raw)
    pr = r.get("permission_request") or {}
    log("internet asks first", code == 200 and pr.get("tool") == "internet", status=code, seconds=round(dt, 1), request=pr,
        reply=(r.get("reply") or "")[:300])

    # --- VS Code: tool step then final answer ---
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file.\nMore text.",
                                          "parameters": {"type": "object", "properties": {
                                              "filePath": {"type": "string", "description": "absolute path"},
                                              "startLine": {"type": "number"}, "endLine": {"type": "number"}},
                                              "required": ["filePath", "startLine", "endLine"]}}},
        {"type": "function", "function": {"name": "list_dir", "description": "List the contents of a directory.",
                                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                         "required": ["path"]}}},
        {"type": "function", "function": {"name": "grep_search", "description": "Do a fast text search in the workspace.",
                                          "parameters": {"type": "object", "properties": {
                                              "query": {"type": "string"}, "isRegexp": {"type": "boolean"}},
                                              "required": ["query", "isRegexp"]}}},
        {"type": "function", "function": {"name": "run_in_terminal", "description": "Run a shell command.",
                                          "parameters": {"type": "object", "properties": {
                                              "command": {"type": "string"}, "explanation": {"type": "string"},
                                              "goal": {"type": "string"}, "mode": {"type": "string", "enum": ["sync", "async"]}},
                                              "required": ["command", "explanation", "goal", "mode"]}}},
    ]
    ws = ("<environment_info>\nThe user's current OS is: Windows\n</environment_info>\n<workspace_info>\n"
          "I am working in a workspace with the following folders:\n- C:\\Users\\santh\\CloverAI\\clover\n</workspace_info>\n")
    ask = "What does the function build_prompt in server.py do? Look at the code."
    msgs = [{"role": "user", "content": ws + f"<userRequest>\n{ask}\n</userRequest>"}]
    code, _, raw, dt = req("POST", "/v1/chat/completions", {"model": "lfm3-1.7b", "messages": msgs, "tools": tools})
    r = json.loads(raw)
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    log("vscode step returns a tool call", code == 200 and bool(calls), status=code, seconds=round(dt, 1), tool_calls=calls,
        content=(msg.get("content") or "")[:300])
    if calls:
        c = calls[0]
        args = json.loads(c["function"]["arguments"])
        out = ""
        if c["function"]["name"] == "read_file":
            try:
                lines = open(args["filePath"], encoding="utf-8").read().splitlines()
                s, e = int(args.get("startLine", 1)), int(args.get("endLine", 200))
                out = "\n".join(lines[s - 1:e])
            except OSError as exc:
                out = f"Error: {exc}"
        else:
            out = "(test harness: tool not executed)"
        msgs += [{"role": "assistant", "content": None, "tool_calls": [c]},
                 {"role": "tool", "tool_call_id": c["id"], "content": out[:20000]}]
        code, _, raw, dt = req("POST", "/v1/chat/completions", {"model": "lfm3-1.7b", "messages": msgs, "tools": tools})
        r = json.loads(raw)
        msg = (r.get("choices") or [{}])[0].get("message") or {}
        log("vscode second step", code == 200, status=code, seconds=round(dt, 1), tool_calls=msg.get("tool_calls"),
            content=(msg.get("content") or "")[:1200])
    log("done", True)


if __name__ == "__main__":
    main()
