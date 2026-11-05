"""Probe: does llama-server (b12012) enforce tool_choice for LFM3-Omni-2B's chat template?
Starts its own CPU llama-server (the Studio keeps running), sends prompts that would not call a tool on their own, with
tool_choice auto / required / a named function, and prints whether a tool call came back.
Usage (on the laptop): venv\\Scripts\\python.exe tests\\tool_choice_probe.py"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "clover"))
import lfm3_native as N  # noqa: E402
import omni_native as O  # noqa: E402
from llamacpp_engine import LlamaServer  # noqa: E402

paths, missing = O.omni_files(BASE)
if missing:
    sys.exit(f"missing {missing}")
tools = [t for t in N.STUDIO_TOOLS if t["function"]["name"] in ("run_powershell", "get_current_time")]
srv = LlamaServer(BASE, kind="cpu", ctx=4096, model=paths["model"], name="probe-toolchoice", extra_args=["-t", "6"])
try:
    for prompt in ("then install it use the terminal", "Write two lines about the sea."):
        for choice in (None, "required", {"type": "function", "function": {"name": "run_powershell"}}):
            body = dict(O.GREEDY, messages=[{"role": "system", "content": "You are Clover, an assistant on the user's laptop."},
                                            {"role": "user", "content": prompt}],
                        tools=tools, max_tokens=160, cache_prompt=False)
            if choice is not None:
                body["tool_choice"] = choice
            t0, content, calls, finish = time.perf_counter(), "", [], None
            for ev in srv.chat_stream(body):
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    content += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            calls.append(fn["name"])
                    finish = ch.get("finish_reason") or finish
            print(json.dumps({"prompt": prompt, "tool_choice": choice if isinstance(choice, (str, type(None))) else "named",
                              "tool_calls": calls, "finish": finish, "content": content[:160],
                              "s": round(time.perf_counter() - t0, 1)}, ensure_ascii=False), flush=True)
finally:
    srv.close()
