"""SWE-Bench Pro V2 Locked-Protocol Agent Runner & Fresh-Sandbox Verifier for Clover AI v4 (`k3_chat_server.py`).

Task: `instance_element-hq__element-web-66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan`
  - Repository: `element-hq/element-web` at base commit `04bc8fb71c4a1ee1eb1f0c4a1de1641d353e9f2c`
  - Subset: SWE-Bench Pro V2 `HARD-51` (`0/9` pass rate across all 9 documented frontier runs in `traj/*/eval_results.json`)

Protocol enforced:
  1. Agent Phase (Isolated Sandbox `/tmp/swebench_pro/agent_rootfs_66d0b318`):
     - `/tests` and `gold_patch.diff` do NOT exist in the agent's rootfs.
     - Clover AI v4 (`http://127.0.0.1:8765/v1/chat/completions`) receives ONLY `instruction.md` and standard VS Code
       workspace tools (`read_file`, `list_dir`, `file_search`, `grep_search`, `replace_string_in_file`, `create_file`,
       `run_in_terminal`).
     - All tool actions execute inside the agent's `/app` directory (`bwrap` sandbox for terminal commands).
     - At the end of the agent loop, `git -C /app diff` is captured as `/tmp/swebench_pro/clover_model.patch`.
  2. Verifier Phase (Pristine Sandbox `/tmp/swebench_pro/rootfs_66d0b318`):
     - Restores `/app` from `/tmp/swebench_pro/app_pristine_66d0b318`.
     - Applies `/tmp/swebench_pro/clover_model.patch` to `/app`.
     - Runs the unchanged official SWE-Bench Pro V2 verifier (`bash /tests/test.sh`).
     - Reads `/logs/verifier/reward.txt` (`1` = RESOLVED, `0` = UNRESOLVED) and `/logs/verifier/output.json`.
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request

TASK_ID = os.environ.get(
    "CLOVER_SWE_TASK_ID",
    "instance_element-hq__element-web-66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan",
)
SLUG = os.environ.get("CLOVER_SWE_SLUG", "66d0b318")
TASK_DIR = f"/tmp/swebench_pro/SWE-bench_Pro-os/v2/tasks/{TASK_ID}"
PRISTINE_APP = f"/tmp/swebench_pro/app_pristine_{SLUG}"
VERIFIER_ROOTFS = f"/tmp/swebench_pro/rootfs_{SLUG}"
AGENT_APP = f"/tmp/swebench_pro/agent_app_{SLUG}"
ENDPOINT = "http://127.0.0.1:8765/v1/chat/completions"

VSCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory in /app.",
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
            "description": "Read lines from a file in /app.",
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
            "description": "Search for files in /app by glob pattern.",
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
            "description": "Search for text or regex across files in /app.",
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
            "name": "replace_string_in_file",
            "description": "Replace an exact literal string in a file inside /app.",
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
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file in /app.",
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
            "name": "run_in_terminal",
            "description": "Execute a shell command inside /app (e.g. npx jest, git diff, git status).",
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
]


def map_app_path(p: str) -> str:
    p = str(p or "/app").strip()
    if p == "/app" or p == "/app/":
        return AGENT_APP
    if p.startswith("/app/"):
        return os.path.join(AGENT_APP, p[5:])
    if p.startswith(AGENT_APP):
        return p
    return os.path.join(AGENT_APP, p.lstrip("/"))


def execute_agent_tool(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    except Exception:
        args = {}
    try:
        if name == "list_dir":
            host_p = map_app_path(args.get("path") or "/app")
            if not os.path.isdir(host_p):
                return f"Error: Directory {args.get('path')} does not exist"
            entries = sorted(os.listdir(host_p))
            return "\n".join(e + ("/" if os.path.isdir(os.path.join(host_p, e)) else "") for e in entries[:80])
        elif name == "read_file":
            host_p = map_app_path(args.get("filePath") or "")
            if not os.path.isfile(host_p):
                return f"Error: File {args.get('filePath')} does not exist"
            s_ln = max(1, int(args.get("startLine") or 1))
            e_ln = max(s_ln, int(args.get("endLine") or 350))
            with open(host_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            actual_end = min(e_ln, len(lines))
            numbered = [f"{i}: {lines[i-1]}" for i in range(s_ln, actual_end + 1)]
            out_s = "".join(numbered) or "(empty file)"
            if actual_end < len(lines):
                out_s = out_s.rstrip("\n") + f"\n[Showing lines {s_ln}-{actual_end} of {len(lines)}]\n"
            return out_s
        elif name == "file_search":
            q = str(args.get("query") or "*").strip()
            cmd = f"cd {AGENT_APP} && find . -maxdepth 5 -path '*{q.strip('*')}*' 2>/dev/null | head -n 40"
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
            return out.strip() or "No matching files."
        elif name == "grep_search":
            q = str(args.get("query") or "").strip()
            inc = str(args.get("includePattern") or "").strip()
            sub = map_app_path(inc) if inc and not inc.startswith("*") else AGENT_APP
            if not os.path.exists(sub):
                sub = AGENT_APP
            out = subprocess.run(["grep", "-rn", "-E", q, sub], capture_output=True, text=True, timeout=10).stdout
            out = out.replace(AGENT_APP, "/app")
            return "\n".join(out.splitlines()[:40]) or "No matches found."
        elif name == "replace_string_in_file":
            host_p = map_app_path(args.get("filePath") or "")
            old_s = str(args.get("oldString") or "")
            new_s = str(args.get("newString") or "")
            if not os.path.isfile(host_p):
                return f"Error: File {args.get('filePath')} does not exist"
            txt = open(host_p, "r", encoding="utf-8").read()
            if not old_s or old_s not in txt:
                return f"Error: oldString not found in {args.get('filePath')}"
            txt = txt.replace(old_s, new_s, 1)
            open(host_p, "w", encoding="utf-8").write(txt)
            return f"Successfully updated {args.get('filePath')}"
        elif name == "create_file":
            host_p = map_app_path(args.get("filePath") or "")
            os.makedirs(os.path.dirname(host_p), exist_ok=True)
            content = str(args.get("content") or "")
            open(host_p, "w", encoding="utf-8").write(content)
            return f"Successfully wrote {args.get('filePath')} ({len(content)} bytes)"
        elif name == "run_in_terminal":
            cmd = str(args.get("command") or "git status -s").strip()
            # Run inside bwrap with AGENT_APP mounted at /app and /tests hidden (mounted as empty tmpfs)
            bwrap_cmd = [
                "sudo", "bwrap",
                "--bind", VERIFIER_ROOTFS, "/",
                "--bind", AGENT_APP, "/app",
                "--tmpfs", "/tests",
                "--tmpfs", "/logs",
                "--dev", "/dev",
                "--proc", "/proc",
                "--unshare-net",
                "--chdir", "/app",
                "--", "bash", "-c",
                f"export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; {cmd}",
            ]
            res = subprocess.run(bwrap_cmd, capture_output=True, text=True, timeout=120)
            combined = (res.stdout + "\n" + res.stderr).strip()
            return combined[-3000:] or f"(exit {res.returncode})"
        return "OK"
    except Exception as e:
        return f"Tool error ({name}): {e}"


def call_clover_v1(messages, session_id="swebench_v2_66d0b318"):
    payload = {
        "model": "clover-kimi-k3",
        "messages": messages,
        "tools": VSCODE_TOOLS,
        "session_id": session_id,
        "cands": 8,
        "stream": False,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-interaction-id": session_id},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_fresh_sandbox_verifier(patch_path: str) -> dict:
    print("\n=== FRESH-SANDBOX RE-GRADING (LOCKED PROTOCOL) ===", flush=True)
    subprocess.run(["sudo", "rm", "-rf", f"{VERIFIER_ROOTFS}/app"], check=True)
    subprocess.run(["sudo", "cp", "-a", PRISTINE_APP, f"{VERIFIER_ROOTFS}/app"], check=True)
    if os.path.exists(patch_path) and os.path.getsize(patch_path) > 0:
        subprocess.run(["sudo", "cp", patch_path, f"{VERIFIER_ROOTFS}/tmp/model.patch"], check=True)
        apply_cmd = "git apply /tmp/model.patch"
    else:
        apply_cmd = "true"

    bwrap_verify = [
        "sudo", "bwrap",
        "--bind", VERIFIER_ROOTFS, "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--chdir", "/app",
        "--", "bash", "-c",
        f"export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; git config --global --add safe.directory /app; {apply_cmd} && bash /tests/test.sh",
    ]
    t0 = time.perf_counter()
    res = subprocess.run(bwrap_verify, capture_output=True, text=True, timeout=600)
    dt = round(time.perf_counter() - t0, 2)
    reward_p = f"{VERIFIER_ROOTFS}/logs/verifier/reward.txt"
    reward = int(open(reward_p).read().strip()) if os.path.exists(reward_p) else 0
    out_json_p = f"{VERIFIER_ROOTFS}/logs/verifier/output.json"
    out_json = json.load(open(out_json_p)) if os.path.exists(out_json_p) else {}
    print(f"Verifier finished in {dt}s | Exit={res.returncode} | REWARD={reward}", flush=True)
    print(res.stdout[-1500:], flush=True)
    # Restore pristine /app in verifier rootfs
    subprocess.run(["sudo", "rm", "-rf", f"{VERIFIER_ROOTFS}/app"], check=True)
    subprocess.run(["sudo", "cp", "-a", PRISTINE_APP, f"{VERIFIER_ROOTFS}/app"], check=True)
    return {
        "reward": reward,
        "exit_code": res.returncode,
        "elapsed_s": dt,
        "stdout_tail": res.stdout[-2000:],
        "stderr_tail": res.stderr[-1000:],
        "test_summary": out_json,
    }


def main():
    max_rounds = int(os.environ.get("CLOVER_SWE_MAX_ROUNDS", "16"))
    attempt_tag = os.environ.get("CLOVER_SWE_ATTEMPT", "attempt1")
    print(f"=== PREPARING ISOLATED AGENT WORKSPACE ({attempt_tag}) ===", flush=True)
    subprocess.run(["sudo", "rm", "-rf", AGENT_APP], check=True)
    subprocess.run(["sudo", "cp", "-a", PRISTINE_APP, AGENT_APP], check=True)
    subprocess.run(["sudo", "chown", "-R", "usergpu:usergpu", AGENT_APP], check=True)

    instruction_text = open(os.path.join(TASK_DIR, "instruction.md"), "r", encoding="utf-8").read()
    vscode_prompt = (
        "<environment_info>\nThe user's current OS is: Linux\n</environment_info>\n"
        "<workspace_info>\nI am working in a workspace with the following folders:\n- /app\n</workspace_info>\n"
        f"<userRequest>\n{instruction_text}\n</userRequest>"
    )

    messages = [{"role": "user", "content": vscode_prompt}]
    trajectory = []
    final_reply = ""
    t_start = time.perf_counter()

    for r_idx in range(max_rounds):
        t_r0 = time.perf_counter()
        resp = call_clover_v1(messages, session_id=f"swe_66d0b318_{attempt_tag}")
        r_dt = round(time.perf_counter() - t_r0, 2)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        t_calls = msg.get("tool_calls") or []
        if not t_calls:
            final_reply = msg.get("content") or ""
            print(f"[Round {r_idx+1} ({r_dt}s)] Agent finished with text reply ({len(final_reply)} chars).", flush=True)
            break
        tc = t_calls[0]
        tc_id = tc.get("id") or f"call_{r_idx+1}"
        fn = tc.get("function") or {}
        t_name = fn.get("name") or ""
        t_args = fn.get("arguments") or "{}"
        print(f"[Round {r_idx+1} ({r_dt}s)] Tool Call: {t_name}({t_args[:180]})", flush=True)
        t_out = execute_agent_tool(t_name, t_args)
        print(f"  -> Output ({len(t_out)} chars): {t_out[:200].replace(chr(10), ' ')}", flush=True)
        trajectory.append({
            "round": r_idx + 1,
            "tool": t_name,
            "arguments": t_args,
            "output_preview": t_out[:500],
            "elapsed_s": r_dt,
        })
        messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
        messages.append({"role": "tool", "tool_call_id": tc_id, "name": t_name, "content": t_out})

    agent_elapsed_s = round(time.perf_counter() - t_start, 2)
    diff_res = subprocess.run(["git", "-C", AGENT_APP, "diff"], capture_output=True, text=True, check=True)
    patch_str = diff_res.stdout
    patch_file = f"/tmp/swebench_pro/clover_{SLUG}_{attempt_tag}.patch"
    open(patch_file, "w", encoding="utf-8").write(patch_str)
    print(f"\nCaptured git diff ({len(patch_str)} bytes) -> {patch_file}", flush=True)

    verdict = run_fresh_sandbox_verifier(patch_file)
    report = {
        "task_id": TASK_ID,
        "slug": SLUG,
        "attempt": attempt_tag,
        "agent_rounds": len(trajectory),
        "agent_elapsed_s": agent_elapsed_s,
        "patch_bytes": len(patch_str),
        "patch": patch_str,
        "trajectory": trajectory,
        "final_reply": final_reply,
        "verifier": verdict,
    }
    report_path = f"/tmp/swebench_pro/report_{SLUG}_{attempt_tag}.json"
    open(report_path, "w", encoding="utf-8").write(json.dumps(report, indent=2))
    print(f"Saved full report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
