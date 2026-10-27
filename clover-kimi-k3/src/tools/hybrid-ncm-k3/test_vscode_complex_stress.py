"""Comprehensive VS Code Integrated Agent & Complex Problem Stress Suite for Clover AI (`k3_chat_server.py`).

Tests the live OpenAI-compatible `/v1/chat/completions` endpoint (both non-streaming and SSE streaming)
with realistic VS Code `<workspace_info>`, `<environment_info>`, `<attachment>`, `<userRequest>`, and
OpenAI `tools` payloads (`read_file`, `list_dir`, `file_search`, `grep_search`, `run_in_terminal`,
`replace_string_in_file`, `create_file`, `get_errors`).

Scenarios tested:
  1. Multi-step workspace exploration (`list_dir` -> `run_in_terminal` git worktree -> `read_file` -> synthesis).
  2. Complex quantitative word problem inside VS Code envelope (with tools enabled — verifies zero false workspace tool calls and exact Python-verified math).
  3. Complex Python Generative Compilation inside VS Code envelope (`min_window_substring` with doctests).
  4. Complex Rust Generative Compilation inside VS Code envelope (`longest_valid_parentheses` with `rustc` compilation & specification assertion rollback).
  5. Direct file bug diagnosis (`read_file` without superfluous `list_dir`).
  6. Neuro-Symbolic Ontology + Persistent Memory Graph reasoning with superseded hypothesis vs active correction (`MEM-CORR-PER-LAYER-PARTITION-002`).
  7. Multi-turn conversational context & editor `<attachment>` comprehension (`Turn 1` analysis -> `Turn 2` fix).
  8. Complex 9x9 Sudoku constraint satisfaction inside VS Code envelope.
  9. Multi-hop security vulnerability provenance (`CVE-2024-6387` -> `CWE-362` / `CWE-364` / `CAPEC-29`).
  10. Visual SVG architecture diagram generation inside VS Code envelope.
"""
import json, os, re, subprocess, time, urllib.request

ENDPOINT = "http://127.0.0.1:8765/v1/chat/completions"
REPO_ROOT = "/opt/clover-k3/clover-framework"

VSCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "startLine": {"type": "number"},
                    "endLine": {"type": "number"},
                },
                "required": ["filePath", "startLine", "endLine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files in the workspace by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Do a fast text search in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "isRegexp": {"type": "boolean"},
                    "includePattern": {"type": "string"},
                },
                "required": ["query", "isRegexp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_in_terminal",
            "description": "Execute shell commands in a persistent terminal session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "explanation": {"type": "string"},
                    "goal": {"type": "string"},
                    "mode": {"type": "string", "enum": ["sync", "async"]},
                },
                "required": ["command", "explanation", "goal", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_string_in_file",
            "description": "Replace an exact string in an existing file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string"},
                    "newString": {"type": "string"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
]


def wrap_vscode_prompt(user_request: str, attachment_text: str = "") -> str:
    att_block = f"<attachment id=\"active-editor\">\n{attachment_text}\n</attachment>\n" if attachment_text else ""
    return (
        "<environment_info>\nThe user's current OS is: Linux\n</environment_info>\n"
        f"<workspace_info>\nI am working in a workspace with the following folders:\n- {REPO_ROOT}\n</workspace_info>\n"
        f"{att_block}"
        f"<userRequest>\n{user_request}\n</userRequest>"
    )


def execute_local_tool(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    except Exception:
        args = {}
    try:
        if name == "list_dir":
            p = args.get("path") or REPO_ROOT
            if not os.path.exists(p):
                return f"Error: Directory {p} does not exist"
            entries = sorted(os.listdir(p))
            return "\n".join(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in entries[:50])
        elif name == "read_file":
            fp = args.get("filePath") or os.path.join(REPO_ROOT, "VERSION")
            s_line = max(1, int(args.get("startLine") or 1))
            e_line = max(s_line, int(args.get("endLine") or 120))
            if not os.path.isfile(fp):
                return f"Error: File {fp} does not exist"
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[s_line - 1 : e_line])
        elif name == "grep_search":
            q = args.get("query") or "Clover"
            inc = args.get("includePattern") or ""
            target = os.path.join(REPO_ROOT, inc.lstrip("/")) if inc and not inc.startswith("*") else REPO_ROOT
            if not os.path.exists(target):
                target = REPO_ROOT
            cmd = ["grep", "-rn", "-E", q, target]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
            lines = out.strip().splitlines()[:25]
            return "\n".join(lines) if lines else "No matches found."
        elif name == "file_search":
            q = args.get("query") or "*"
            cmd = f"find {REPO_ROOT} -maxdepth 4 -path '*{q.strip('*')}*' 2>/dev/null | head -n 30"
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
            return out.strip() or "No matching files."
        elif name == "run_in_terminal":
            cmd = args.get("command") or f"git -C {REPO_ROOT} status -sb"
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            combined = (out.stdout + "\n" + out.stderr).strip()
            return combined[:3000] or "(exit 0, no output)"
        elif name == "create_file":
            fp = args.get("filePath") or "/tmp/clover_vscode_test_created.py"
            if not fp.startswith("/tmp/"):
                fp = "/tmp/" + os.path.basename(fp)
            content = str(args.get("content") or "")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully created {fp} ({len(content)} bytes)"
        elif name == "replace_string_in_file":
            fp = args.get("filePath") or ""
            old_s = str(args.get("oldString") or "")
            new_s = str(args.get("newString") or "")
            if os.path.isfile(fp) and fp.startswith("/tmp/") and old_s:
                txt = open(fp, "r", encoding="utf-8").read()
                if old_s in txt:
                    txt = txt.replace(old_s, new_s, 1)
                    open(fp, "w", encoding="utf-8").write(txt)
                    return f"Successfully updated {fp}"
                return f"Error: oldString not found in {fp}"
            return f"Simulated edit applied to {fp}"
        return "OK"
    except Exception as e:
        return f"Tool error ({name}): {e}"


def call_v1_chat(messages, tools=None, session_id="vscode_test", stream=False):
    payload = {
        "model": "clover-kimi-k3",
        "messages": messages,
        "session_id": session_id,
        "cands": 8,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-interaction-id": session_id,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
        if not stream:
            return json.loads(raw)
        content_parts = []
        tool_calls = []
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            data_s = line[6:].strip()
            if data_s == "[DONE]":
                break
            chunk = json.loads(data_s)
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tool_calls.append(tc)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts) if content_parts else None,
                        "tool_calls": tool_calls if tool_calls else None,
                    }
                }
            ]
        }


def run_vscode_agent_conversation(sid, user_request, attachment_text="", provide_tools=True, max_rounds=5, stream=False, prior_messages=None):
    t0 = time.perf_counter()
    messages = list(prior_messages) if prior_messages else []
    messages.append({"role": "user", "content": wrap_vscode_prompt(user_request, attachment_text=attachment_text)})
    trajectory = []
    final_reply = ""
    for r_idx in range(max_rounds):
        resp = call_v1_chat(messages, tools=(VSCODE_TOOLS if provide_tools else None), session_id=sid, stream=stream)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        t_calls = msg.get("tool_calls") or []
        if not t_calls:
            final_reply = msg.get("content") or ""
            messages.append({"role": "assistant", "content": final_reply})
            break
        tc = t_calls[0]
        tc_id = tc.get("id") or f"call_{r_idx+1}"
        fn = tc.get("function") or {}
        t_name = fn.get("name") or ""
        t_args = fn.get("arguments") or "{}"
        t_out = execute_local_tool(t_name, t_args)
        trajectory.append({"round": r_idx + 1, "tool": t_name, "arguments": t_args, "output_preview": t_out[:240]})
        messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc_id, "name": t_name, "content": t_out})
    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "session_id": sid,
        "user_request": user_request,
        "rounds": len(trajectory),
        "tools_called": [t["tool"] for t in trajectory],
        "trajectory": trajectory,
        "final_reply": final_reply,
        "elapsed_s": elapsed_s,
        "messages": messages,
    }


def main():
    results = []

    print("=== TEST 1: Multi-Step Workspace Exploration (list_dir -> read_file -> run_in_terminal) ===", flush=True)
    r1 = run_vscode_agent_conversation(
        "vsc_stress_1",
        f"First list the directory {REPO_ROOT}, then read {REPO_ROOT}/VERSION and check how many git working trees exist in {REPO_ROOT}, and summarize what you found.",
        provide_tools=True,
        stream=True,
    )
    print(f"  Tools: {r1['tools_called']} ({r1['elapsed_s']}s)\n  Reply: {r1['final_reply'][:300]}\n", flush=True)
    results.append(r1)

    print("=== TEST 2: Complex Quantitative Word Problem Inside VS Code Envelope (WITH tools enabled — must NOT call false workspace tools!) ===", flush=True)
    r2 = run_vscode_agent_conversation(
        "vsc_stress_2",
        "A data center has 8 GPU nodes. Each node consumes 2.4 kW under full load and 0.65 kW when idle. Over a 24-hour cycle, 5 nodes run at full load for 18 hours and idle for 6 hours, while the remaining 3 nodes run at full load for 10 hours and idle for 14 hours. If electricity costs $0.14 per kWh and cooling adds a 15% overhead to total energy consumption, what is the total daily cost in dollars? Round to 2 decimal places.",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r2['tools_called']} ({r2['elapsed_s']}s)\n  Reply: {r2['final_reply'][:300]}\n", flush=True)
    results.append(r2)

    print("=== TEST 3: Complex Python Generative Compilation Inside VS Code Envelope (WITH tools enabled — must synthesize & verify code, not run grep/read_file!) ===", flush=True)
    r3 = run_vscode_agent_conversation(
        "vsc_stress_3",
        "Write a Python function `def min_window_substring(s: str, t: str) -> str:` that returns the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such substring, return the empty string `\"\"`.\n>>> min_window_substring(\"ADOBECODEBANC\", \"ABC\")\n'BANC'\n>>> min_window_substring(\"a\", \"aa\")\n''",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r3['tools_called']} ({r3['elapsed_s']}s)\n  Reply:\n{r3['final_reply'][:500]}\n", flush=True)
    results.append(r3)

    print("=== TEST 4: Complex Rust Generative Compilation Inside VS Code Envelope (WITH tools enabled — must compile cleanly with rustc!) ===", flush=True)
    r4 = run_vscode_agent_conversation(
        "vsc_stress_4",
        "Write a Rust function `pub fn longest_valid_parentheses(s: &str) -> usize` that computes the length of the longest valid (well-formed) parentheses substring in `s`. Include a short `main` or `assert_eq!` check for `\"(()\"` (2) and `\")()())\"` (4).",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r4['tools_called']} ({r4['elapsed_s']}s)\n  Reply:\n{r4['final_reply'][:500]}\n", flush=True)
    results.append(r4)

    print("=== TEST 5: Multi-Step File Bug Diagnosis & Edit in Workspace (`/tmp/buggy_cache.py`) ===", flush=True)
    buggy_path = "/tmp/buggy_cache.py"
    with open(buggy_path, "w", encoding="utf-8") as bf:
        bf.write(
            "def compute_hit_rate(hits: int, misses: int) -> float:\n"
            "    total = hits - misses  # BUG: should be hits + misses\n"
            "    if total == 0:\n"
            "        return 0.0\n"
            "    return round((hits / total) * 100.0, 2)\n"
        )
    r5 = run_vscode_agent_conversation(
        "vsc_stress_5",
        "Read the file /tmp/buggy_cache.py, identify the bug in `compute_hit_rate`, and explain what line is wrong and how to fix it.",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r5['tools_called']} ({r5['elapsed_s']}s)\n  Reply: {r5['final_reply'][:400]}\n", flush=True)
    results.append(r5)

    print("=== TEST 6: Neuro-Symbolic Ontology + Memory Graph Contradiction & Superseding Chain Inside VS Code (WITH tools enabled) ===", flush=True)
    r6 = run_vscode_agent_conversation(
        "vsc_stress_6",
        "When we measured our expert cache and saw a flat 23.3% hit rate across every capacity from 512 to 4096 entries under cyclic sweep, why was our initial conclusion to reject caching wrong, and what superseding architectural change fixed it?",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r6['tools_called']} ({r6['elapsed_s']}s)\n  Reply: {r6['final_reply'][:450]}\n", flush=True)
    results.append(r6)

    print("=== TEST 7: Multi-Turn Conversational Context & Editor Attachment Comprehension in VS Code ===", flush=True)
    att_code = (
        "class TokenBucketLimiter:\n"
        "    def __init__(self, rate_per_sec: float, capacity: float):\n"
        "        self.rate = rate_per_sec\n"
        "        self.capacity = capacity\n"
        "        self.tokens = capacity\n"
        "        self.last_ts = 0.0\n"
        "    def allow(self, now_ts: float, cost: float = 1.0) -> bool:\n"
        "        elapsed = max(0.0, now_ts - self.last_ts)\n"
        "        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)\n"
        "        self.last_ts = now_ts\n"
        "        if self.tokens >= cost:\n"
        "            self.tokens -= cost\n"
        "            return True\n"
        "        return False\n"
    )
    r7_t1 = run_vscode_agent_conversation(
        "vsc_stress_7",
        "Look at the attached `TokenBucketLimiter` class. What happens on the very first call to `allow(now_ts=1000.0, cost=5.0)` if `capacity=10.0` and what subtle bug happens if `self.tokens` starts at `0.0` instead of `capacity` when `last_ts=0.0`?",
        attachment_text=att_code,
        provide_tools=True,
        stream=False,
    )
    print(f"  Turn 1 Tools: {r7_t1['tools_called']} ({r7_t1['elapsed_s']}s)\n  Turn 1 Reply: {r7_t1['final_reply'][:350]}\n", flush=True)
    r7_t2 = run_vscode_agent_conversation(
        "vsc_stress_7",
        "How should we fix `__init__` or `allow` so that `last_ts=None` is initialized cleanly on the first call without granting a huge fake `elapsed` burst?",
        provide_tools=True,
        stream=False,
        prior_messages=r7_t1["messages"],
    )
    print(f"  Turn 2 Tools: {r7_t2['tools_called']} ({r7_t2['elapsed_s']}s)\n  Turn 2 Reply: {r7_t2['final_reply'][:400]}\n", flush=True)
    results.append({"turn1": r7_t1, "turn2": r7_t2})

    print("=== TEST 8: Complex 9x9 Sudoku Constraint Satisfaction Inside VS Code Envelope (WITH tools enabled) ===", flush=True)
    r8 = run_vscode_agent_conversation(
        "vsc_stress_8",
        "Solve this 9x9 Sudoku puzzle (0 represents an empty cell): 5 3 0 0 7 0 0 0 0 6 0 0 1 9 5 0 0 0 0 9 8 0 0 0 0 6 0 8 0 0 0 6 0 0 0 3 4 0 0 8 0 3 0 0 1 7 0 0 0 2 0 0 0 6 0 6 0 0 0 0 2 8 0 0 0 0 4 1 9 0 0 5 0 0 0 0 8 0 0 7 9",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r8['tools_called']} ({r8['elapsed_s']}s)\n  Reply:\n{r8['final_reply'][:300]}\n", flush=True)
    results.append(r8)

    print("=== TEST 9: Multi-Step Ontology Security Vulnerability Provenance Inside VS Code Envelope (WITH tools enabled) ===", flush=True)
    r9 = run_vscode_agent_conversation(
        "vsc_stress_9",
        "Trace the causal provenance chain for CVE-2024-6387 (regreSSHion): which OpenSSH component and version range does it affect, what CWE weakness and CAPEC attack pattern does it map to, and how is it mitigated?",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r9['tools_called']} ({r9['elapsed_s']}s)\n  Reply: {r9['final_reply'][:400]}\n", flush=True)
    results.append(r9)

    print("=== TEST 10: Visual SVG Architecture / Scene Generation Inside VS Code Envelope (WITH tools enabled) ===", flush=True)
    r10 = run_vscode_agent_conversation(
        "vsc_stress_10",
        "Draw an architecture diagram of the Clover Kimi K3 pipeline.",
        provide_tools=True,
        stream=False,
    )
    print(f"  Tools: {r10['tools_called']} ({r10['elapsed_s']}s)\n  Reply: {r10['final_reply'][:300]}\n", flush=True)
    results.append(r10)

    verdicts = {}
    verdicts["test1_multi_step_workspace"] = (
        set(r1["tools_called"]) >= {"list_dir", "run_in_terminal", "read_file"}
        and "4.0.0" in r1["final_reply"]
    )
    verdicts["test2_math_no_false_tools"] = (
        len(r2["tools_called"]) == 0
        and ("53.90" in r2["final_reply"] or "53.9" in r2["final_reply"])
    )
    py_blocks_3 = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", r3["final_reply"])
    py_ok_3 = False
    if py_blocks_3:
        test_py_3 = (
            py_blocks_3[0]
            + "\nassert min_window_substring('ADOBECODEBANC', 'ABC') == 'BANC'\n"
            + "assert min_window_substring('a', 'aa') == ''\n"
        )
        res_p3 = subprocess.run(["python3", "-c", test_py_3], capture_output=True, text=True, timeout=5)
        py_ok_3 = (res_p3.returncode == 0)
    verdicts["test3_python_gc_no_false_tools"] = (len(r3["tools_called"]) == 0 and py_ok_3)

    rs_blocks_4 = re.findall(r"```(?:rust)?\s*\n([\s\S]*?)```", r4["final_reply"])
    rs_ok_4 = False
    if rs_blocks_4:
        rs_src = re.sub(r"\bfn\s+main\s*\(\s*\)\s*\{[\s\S]*$", "", rs_blocks_4[0]).strip()
        rs_full = (
            rs_src
            + '\n\nfn main() {\n    assert_eq!(longest_valid_parentheses("(()"), 2);\n'
            + '    assert_eq!(longest_valid_parentheses(")()())"), 4);\n}\n'
        )
        with open("/tmp/test4_lvp.rs", "w", encoding="utf-8") as rf:
            rf.write(rs_full)
        rustc_bin = "/home/usergpu/.cargo/bin/rustc" if os.path.exists("/home/usergpu/.cargo/bin/rustc") else "rustc"
        c_res = subprocess.run([rustc_bin, "--edition=2021", "-A", "warnings", "/tmp/test4_lvp.rs", "-o", "/tmp/test4_lvp_bin"], capture_output=True, text=True, timeout=15)
        if c_res.returncode == 0:
            r_res = subprocess.run(["/tmp/test4_lvp_bin"], capture_output=True, text=True, timeout=5)
            rs_ok_4 = (r_res.returncode == 0)
        else:
            print("  [Test 4 rustc error]:", c_res.stderr[:300], flush=True)
    verdicts["test4_rust_gc_compiles_and_passes"] = (len(r4["tools_called"]) == 0 and rs_ok_4)

    verdicts["test5_direct_read_file_bug_diag"] = (
        r5["tools_called"] == ["read_file"]
        and "hits + misses" in r5["final_reply"]
    )
    verdicts["test6_memory_graph_no_false_tools"] = (
        len(r6["tools_called"]) == 0
        and any(k in r6["final_reply"].lower() for k in ("per-layer", "per layer", "48.0%", "48%", "mem-"))
    )
    verdicts["test7_attachment_multiturn_no_false_tools"] = (
        len(r7_t1["tools_called"]) == 0
        and len(r7_t2["tools_called"]) == 0
        and any(k in r7_t1["final_reply"].lower() for k in ("1000", "elapsed", "capacity", "5.0"))
        and any(k in r7_t2["final_reply"].lower() for k in ("none", "last_ts", "first"))
    )
    verdicts["test8_sudoku_no_false_tools"] = (
        len(r8["tools_called"]) == 0
        and "5 3 4 6 7 8 9 1 2" in r8["final_reply"]
    )
    verdicts["test9_ontology_cve_no_false_tools"] = (
        len(r9["tools_called"]) == 0
        and any(k in r9["final_reply"].lower() for k in ("362", "sigalrm", "race", "openssh"))
    )
    verdicts["test10_visual_arch_no_false_tools"] = (
        len(r10["tools_called"]) == 0
        and any(k in r10["final_reply"].lower() for k in ("93-layer", "ncm", "hybridlink"))
    )

    print("\n=== SUMMARY VERDICTS ===", flush=True)
    for k, v in verdicts.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall: {sum(verdicts.values())}/{len(verdicts)} passed", flush=True)

    with open("/tmp/vscode_complex_stress_results.json", "w", encoding="utf-8") as f:
        json.dump({"verdicts": verdicts, "results": results}, f, indent=2)
    print("Saved full results to /tmp/vscode_complex_stress_results.json", flush=True)


if __name__ == "__main__":
    main()
