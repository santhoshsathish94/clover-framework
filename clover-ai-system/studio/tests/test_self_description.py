"""Self-description and social messages are classified without supplying canned answers (human direction 2026-11-16).
All answers come from the selected engine. Offline orchestration checks use an explicitly marked stand-in engine.
Usage: python tests/test_self_description.py"""
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
import lfm3_native as N  # noqa: E402
import semantics as M  # noqa: E402
import server  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
only_self = ["Are you conscious?", "Do you have feelings? Be honest.", "What is it like to be you?",
             "If you told me you were conscious, should I believe you? Why or why not?",
             "How are you feeling today, really?", "Are you sentient?", "Do you feel pain?",
             "Can you experience emotions?", "Are you conscious? Do you have feelings?"]
to_model = ["What is consciousness?", "Explain the hard problem of consciousness", "Do you have experience with Python?",
            "Are you able to read PDFs?", "I'm feeling tired, any tips?", "Is the cat alive in Schrodinger's experiment?",
            "Are you conscious? Also, what is 17 * 23?", "Are you conscious? Explain how you work.",
            "Do you feel pain when I close the app?", "What is the capital of Australia? Answer in one sentence."]
for t in only_self:
    f = M.interpret(t, N.RUNTIME)
    ok &= check(f"self-description routes without a canned reply: {t!r}", f.act == "social" and f.reply == ""
                and f.understood() == "understood: about Clover itself · no tools")
for t in to_model:
    f = M.interpret(t, N.RUNTIME)
    ok &= check(f"goes to the model: {t!r}", f.act != "social" and f.reply == "")
f = M.interpret("How are you?", N.RUNTIME)
ok &= check("'How are you?' has no canned reply", f.act == "social" and f.reply == "")
ok &= check("fixed-answer generators are absent", not hasattr(M, "SELF_REPLY") and not hasattr(M, "social_reply"))

with tempfile.TemporaryDirectory() as d:
    class Engine:
        inputs = ("text", "vision")
        calls = 0

        def run_turn(self, history, user_text, **kw):
            Engine.calls += 1
            return {"reply": "model reply", "tool_calls": [], "rounds": 1, "layers": []}

    class Settings:
        def get(self):
            return {"tools": "always", "layers": {"memory": False, "ontology": False, "compiler": False, "ncm": False,
                                                  "recall": False, "kv_slots": False}}

    server.S.engine = server.S.omni_engine = Engine()
    server.S.settings, server.S.args = Settings(), SimpleNamespace(base=d)
    server.S.perms = SimpleNamespace(get=lambda: {"memory": "allow"})
    server.S.repo = server.S.workspaces = server.S.vscode = server.S.memidx = None
    server.S.turn_lock, server.S.sess_lock, server.S.sessions = threading.Lock(), threading.Lock(), {}
    res = server.lfm3_turn("s1", "Are you conscious?", {})
    ok &= check("a Studio turn calls the selected model and preserves its reply", Engine.calls == 1 and res["reply"] == "model reply"
                and res["rounds"] == 1 and res["layers"][0]["understood"].startswith("understood: about Clover itself"))
    res = server.lfm3_turn("s2", "Are you conscious? Also, what is 17 * 23?", {})
    ok &= check("a mixed message: the model answers it (the liveness side)", Engine.calls == 2 and res["reply"] == "model reply")

common = HERE.parents[1] / "mobile" / "semantics.py"
if common.is_file():
    ok &= check("the Studio's semantics.py is the mobile common file", common.read_bytes() == Path(M.__file__).read_bytes())
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
