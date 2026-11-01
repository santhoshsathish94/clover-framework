"""Starts the laptop Studio stack (scratch-agents lfm3-hybridagent/laptop/clover/server.py) with its layers observed.
Usage: python observe_laptop.py --clover <laptop>/clover --base <base-dir> --port 8792 --trace <file>"""
import argparse
import sys
from pathlib import Path

import tracer as T

ap = argparse.ArgumentParser()
ap.add_argument("--clover", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--port", type=int, default=8792)
ap.add_argument("--trace", required=True)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.clover)))
# device part on Linux (the only substitution): the POSIX PythonTool in place of its Windows port; loaded first so
# server.py putting its own folder first on sys.path (line 28) cannot bring the Windows runner back
sys.path.insert(0, str(Path(__file__).resolve().parent / "linux_device"))
import windows_python_tool  # noqa: E402,F401
import clover_cycle  # noqa: E402
import clover_repo  # noqa: E402
import lfm3_layers  # noqa: E402
import lfm3_native  # noqa: E402
import llamacpp_engine  # noqa: E402
import memory_index  # noqa: E402
import omni_native  # noqa: E402
import semantics  # noqa: E402
import server  # noqa: E402
import web_search  # noqa: E402

T.open_trace(a.trace)
T.mark_turns(("/api/chat",))


def chat_args(items):
    d = dict(items)
    msgs = d.get("msgs") or []
    return {"messages": [{"role": m.get("role"), "content": T.brief(m.get("content"), limit=4000)} for m in msgs],
            "tools": [s.get("function", {}).get("name") for s in d.get("schemas") or []],
            "max_new": d.get("max_new"), "tool_choice": T.brief(d.get("tool_choice"))}


def chat_result(res):
    content, calls, stats = res
    return {"content": T.brief(content, limit=6000), "calls": T.brief(calls), "stats": stats}


# Semantics (common file)
for fn in ("interpret", "route", "align", "social", "non_answer"):
    T.wrap(semantics, fn, "semantics")

# Turn (server.lfm3_turn is looked up by name when the handler runs)
T.wrap(server, "lfm3_turn", "turn", result=lambda r: {k: T.brief(v) for k, v in r.items()})
T.wrap(server, "remember_note", "S1")
T.wrap(server, "kv_switch", "C1-kv")

# C1 inference through llama-server, C2 tool rounds
O = omni_native.OmniNative
T.wrap(O, "run_turn", "C2", result=lambda r: {k: T.brief(v) for k, v in r.items()})
T.wrap(O, "retry_turn", "C2")
T.wrap(O, "_chat", "C1", args=chat_args, result=chat_result)
T.wrap(O, "system_prompt", "C1")
T.wrap(llamacpp_engine.LlamaServer, "chat_stream", "C1-stream",
       args=lambda items: {"n_messages": len((dict(items).get("body") or {}).get("messages") or []),
                           "sampling": {k: v for k, v in (dict(items).get("body") or {}).items()
                                        if k not in ("messages", "tools")}})
T.wrap(lfm3_native, "parse_tool_calls", "C2")
T.wrap(omni_native, "looping", "C2")

# S3 tools and permissions
N = lfm3_native.LFM3Native
T.wrap(N, "execute_tool", "S3")
T.wrap(N, "_run_tool", "S3")
T.wrap(N, "_permit", "S3")
for fn in ("search", "execute", "research", "fetch_url_content"):
    T.wrap(web_search.WebSearch, fn, "S3-web")

# S1 memory, S2 relationships, compiler, C3
K = lfm3_layers.LayerKit
T.wrap(K, "memory_context", "S1")
T.wrap(K, "remember", "S1")
T.wrap(K, "records_context", "S1-records")
T.wrap(K, "workspace_context", "S1-workspace")
T.wrap(K, "recall_context", "S2-recall")
T.wrap(K, "ontology_context", "S2-ontology")
T.wrap(K, "grounding_filter", "S2-grounding")
T.wrap(K, "compiler_check", "S3-compiler")
T.wrap(K, "ncm_render", "C3")
T.wrap(memory_index.MemoryIndex, "recall", "S2-recall")
for fn in ("turn_start", "turn_end", "context_block"):
    T.wrap(clover_repo.Repo, fn, "S1-records")

# The Clover cycle (Try again)
for fn in ("retry_requested", "last_exchange", "retry_cycle"):
    if hasattr(clover_cycle, fn):
        T.wrap(clover_cycle, fn, "cycle")

_load = server.load_engine


def load_observed(args):
    _load(args)
    eng = server.S.engine
    T.emit({"layer": "startup", "fn": "load_engine", "phase": "ready", "load": server.S.load,
            "settings": server.S.settings.get() if server.S.settings else None,
            "permissions": server.S.perms.get() if server.S.perms else None,
            "engine": eng.describe() if eng is not None else None,
            "modules": T.module_files(["server", "semantics", "clover_cycle", "omni_native", "lfm3_native",
                                       "lfm3_layers", "memory_index", "clover_repo", "web_search", "llamacpp_engine"])})


server.load_engine = load_observed
T.emit({"layer": "startup", "fn": "observe_laptop", "phase": "start", "base": a.base, "port": a.port})
sys.argv = ["server.py", "--base", a.base, "--port", str(a.port), "--mode", "lfm3",
            "--log", str(Path(a.base) / "logs" / "studio-server.log")]
server.main()
