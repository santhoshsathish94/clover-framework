"""In tools mode Always, a memory turn is offered only tools the route allowed: a note that names a file no longer gets
read_file offered (and read unasked) when the message names no file, and a note about compound interest still gets the
calculator. Real OmniNative.run_turn + LayerKit + semantics.route; a stub GPU records what the model was offered.
Usage: python tests/test_memory_turn_tools_follow_route.py"""
import sys
import tempfile
import threading
import types
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import lfm3_layers  # noqa: E402
import lfm3_native  # noqa: E402
import omni_native  # noqa: E402
import permissions  # noqa: E402
import semantics  # noqa: E402


class StubGPU:
    kind, device = "stub", "stub"

    def __init__(self):
        self.offered = []

    def chat_stream(self, body, cancel=None):
        self.offered.append(sorted(t["function"]["name"] for t in body.get("tools") or ()))
        yield {"choices": [{"delta": {"content": "From your notes."}, "finish_reason": "stop"}]}

    def props(self):
        return {}

    def close(self):
        pass


def offered_on(module, note, question):
    base = Path(tempfile.mkdtemp(prefix="memory-turn-tools-"))
    eng = object.__new__(module.OmniNative)  # no model: __init__ starts llama-server
    eng.base_dir, eng.quant, eng.engine_kind, eng.max_tokens, eng.cache = base, "q4_0", "stub", 512, None
    eng.perms = permissions.PermissionStore(base / "permissions.json")
    eng.perms.update({t: "deny" for t in permissions.TOOLS})
    eng.media_dir, eng.ffmpeg_dir, eng.inputs = base / "media", "", ["text"]
    eng.lock, eng.stage, eng.side_gate = threading.Lock(), {"name": "idle", "since": 0}, omni_native.SideGate()
    eng._voice, eng._voice_busy, eng._voice_used, eng._voice_error = None, 0, 0.0, None
    eng._voice_lock, eng.separate_voice, eng.voice_threads = threading.Lock(), False, 1
    eng.workdir, eng.workname, eng.after_write = None, "", None
    eng.display_model_name, eng.display_layers_desc, eng.setup_s = "LFM3-Omni-2B (stub)", "", 0.0
    eng.gpu = StubGPU()
    kit = lfm3_layers.LayerKit(base, memory_root=base / "growth" / "knowledge")
    kit.permissions_map = {"memory": "allow", "files": "deny", "code": "deny", "internet": "deny"}
    kit.remember(note, conversation_id="setup")
    frame = semantics.interpret(question, lfm3_native.RUNTIME)
    kit.meaning = frame
    only, tools_on = semantics.route(frame, "always", True, offer=[t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS])
    eng.run_turn([], question, tools=tools_on, only_tools=only, layers=("memory", "ontology"), kit=kit, layer_query=question,
                 final_on_limit=True, force_answer=True, meaning=frame)
    return eng.gpu.offered[0] if eng.gpu.offered else None


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


PLAN = ("My project plan is in plan.md in my Documents folder.", "Where is my project plan?")
SAVINGS = ("My savings account holds 25,000 rupees and earns 7 percent annual compound interest.",
           "How much will my savings account hold after 3 years?")
FILE_TOOLS = {"read_file", "list_directory", "search_files", "file_info", "write_file"}
ok = True
got = offered_on(omni_native, *PLAN)
ok &= check("a note naming a file: no file tool is offered when the message names none",
            got is not None and not (set(got) & FILE_TOOLS))
ok &= check("a compound-interest note: the calculator is still offered", "calculate" in (offered_on(omni_native, *SAVINGS) or []))

src = (CLOVER / "omni_native.py").read_text(encoding="utf-8")
new = "if t not in declined_set and t in allowed] if has_lrn_or_formula else []"
assert src.count(new) == 1
old_mod = types.ModuleType("omni_native_old")
old_mod.__file__ = str(CLOVER / "omni_native.py")
exec(compile(src.replace(new, "if t not in declined_set] if has_lrn_or_formula else []"), old_mod.__file__, "exec"), old_mod.__dict__)
ok &= check("liveness: with the old line the same turn is offered read_file", "read_file" in (offered_on(old_mod, *PLAN) or []))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
