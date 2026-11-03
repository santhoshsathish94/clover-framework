"""Personal learning in the runtime (personal.py), without the model: on by default with a switch the person controls
(human, 2026-11-03); habits learned from corrections
and applied only when validated; rejected readings switched off; preferred forms; statements as evidence; safety rules
kept; keep / stop learning / delete; the learner identical to PPL-1's. Usage: python test_personal.py"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from s23_on_device_clover import PHONE_TOOLS, phone_runtime  # noqa: E402  (sets the phone's sys.path)
import clover_cycle as C  # noqa: E402
import local_terminal  # noqa: E402
import personal  # noqa: E402
import semantics  # noqa: E402

FAILED: list[str] = []
PASSED = [0]


def check(cond: bool, what: str) -> None:
    if cond:
        PASSED[0] += 1
        print("PASS", what)
    else:
        FAILED.append(what)
        print("FAIL", what)


class Chat:
    """One conversation, driven the way the phone server drives a turn (before_turn, interpret, after_turn)."""

    def __init__(self, rt, session):
        self.rt, self.session, self.history = rt, session, []

    def say(self, text):
        last = C.last_exchange(self.history) if self.history else None
        retry = C.retry_requested(text, last[0]["content"]) if last else None
        self.rt.personal.before_turn(self.session, text, retry)
        f = semantics.interpret(text, self.rt, history=self.history)
        if retry:
            f.act = "retry"
        self.rt.personal.after_turn(self.session, f, {"turn_n": len(self.history) // 2 + 1})
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": f"(answer to {text})"}]
        return f


def runtime(store, **kw):
    pers = personal.Personal(store, device="phone", **kw)
    return semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky, personal=pers)


# ---------------------------------------------------------------- on by default, with a switch
os.environ.pop("CLOVER_PPL_DIR", None)
os.environ.pop("CLOVER_PPL", None)
check(phone_runtime().personal is None, "no store known yet (no system started, no CLOVER_PPL_DIR): nothing personal")
plain = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky)
prompts = ["Dig into rust lifetimes.", "Search the web for heat pumps", "Delete the old report files in C:\\tmp\\x",
           "What time is it?", "Explain photosynthesis.", "thanks"]
check(all(semantics.interpret(p, plain) == semantics.interpret(p, phone_runtime()) for p in prompts),
      "without a Personal every frame is what it was")
check(all(not semantics.interpret(p, plain).personal and not semantics.interpret(p, plain).guidance for p in prompts),
      "without a Personal: no personal reading, no guidance")
with tempfile.TemporaryDirectory() as tmp0:
    os.environ["CLOVER_PPL_DIR"] = str(Path(tmp0) / "default-store")
    check(phone_runtime().personal is not None and phone_runtime().personal.learning,
          "on by default once the process knows the person's store")
    os.environ["CLOVER_PPL"] = "off"
    check(personal.for_process(str(Path(tmp0) / "other"), "phone", None) is None, "CLOVER_PPL=off: off for the process")
    os.environ.pop("CLOVER_PPL")
    os.environ.pop("CLOVER_PPL_DIR")

with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "person"
    rt = runtime(store, filter_repo=[])
    a = Chat(rt, "chat-a")

    # ------------------------------------------------------------ a habit learned from corrections
    f = a.say("Dig into rust lifetimes.")
    check(f.caps == [] and f.act == "question", "general reading first: a question, no tools")
    a.say("No, look it up online.")
    h = [x for x in rt.personal.learner.hyps if x.status == "provisional"]
    check(len(h) == 1 and h[0].target == ("web.page", "web.search"), "one correction: a provisional hypothesis")
    f = a.say("Dig into heat pumps.")
    check(f.caps == [], "provisional hypotheses are not applied")
    a.say("No, search the web for that.")
    v = [x for x in rt.personal.learner.hyps if x.status == "validated"]
    check(len(v) == 1 and v[0].trigger == frozenset({"ctx=phone", "tok=dig", "tok=into"}),
          f"two corrections: validated, generalised to what they share {sorted(v[0].trigger) if v else None}")
    f = a.say("Dig into coral reefs.")
    check(f.asked == ["web_search"] and f.act == "action" and f.personal.get("hypothesis") == v[0].id,
          "a fresh message: the person's reading replaces the general one")
    check("as you usually mean it" in f.understood() and f.layer().get("personal", {}).get("caps"),
          "the reading is shown as the person's")
    check(semantics.route(f, "ask", True, offer=["web_search", "python_interpreter"])[0] == ["web_search"],
          "When I ask (the default): the habit's tool is offered")
    a.say("Thanks!")
    check(len(v[0].support) == 3, "thanks: the applied reading gains support")
    f = a.say("Dig a hole in the garden.")
    check(f.caps == [] and not f.personal.get("hypothesis"), "a different expression is read generally")

    # ------------------------------------------------------------ rejected readings lose their standing
    a.say("Dig into quantum dots.")
    a.say("That's wrong.")
    a.say("Dig into sourdough baking.")
    a.say("Try again.")
    check(v[0].status == "provisional" and len(v[0].contra) == 2, "two retries after the personal reading: not applied")
    f = a.say("Dig into marathon training.")
    check(f.caps == [], "back to the general reading")

    # ------------------------------------------------------------ preferred form
    b = Chat(rt, "chat-b")
    b.say("Summarize the plot of Hamlet.")
    b.say("Shorter, please.")
    f = b.say("Explain photosynthesis.")
    check("keep answers short" in f.guidance and "personal" in f.layer(), "one request for shorter answers: guidance")
    b.say("More detail, please.")
    f = b.say("Explain osmosis.")
    check("short" not in f.guidance, "contradicted: back to the default length")
    b.say("I prefer bullet points.")
    f = b.say("Explain entropy.")
    check("use bullet points" in f.guidance, "a stated preference is evidence")
    b.say("Delete the old report files in C:\\tmp\\x")
    b.say("Just do it, no need to ask.")
    f = b.say("Delete the old log files in C:\\tmp\\y")
    check(rt.personal.learner.preference("confirm", "ask_when_risky") == "just_do" and f.discernment == "ask",
          "a learned 'just do it' is recorded but never removes the question before changing files")
    n_before = len(rt.personal.store.all_experiences())
    rt.personal.set_learning(False)
    f = b.say("Explain entropy again.")
    b.say("Thanks!")
    check(not f.personal and not f.guidance and len(rt.personal.store.all_experiences()) == n_before
          and rt.personal.view()["learning"] is False, "learning switched off: no personal reading, nothing recorded")
    check(runtime(store, filter_repo=[]).personal.learning is False, "the switch is kept in the store")
    rt.personal.set_learning(True)
    check("use bullet points" in b.say("Explain magnetism.").guidance, "switched back on: what was learned applies")

    # ------------------------------------------------------------ statements and safety
    c = Chat(rt, "chat-c")
    c.say("When I say check the news, I mean search the web.")
    st = [x for x in rt.personal.learner.hyps if x.origin == "statement"]
    check(len(st) == 1 and st[0].status == "provisional" and st[0].target == ("web.page", "web.search"),
          "a statement makes a provisional hypothesis, not a rule")
    c.say("Check the news about Mars.")
    c.say("No, search the web for it.")
    f = c.say("Check the news on the election.")
    check(f.asked == ["web_search"], "statement plus one interaction: validated and applied")

    class Stub:
        def __init__(self, out):
            self.out = out

        def interpret(self, text, frame):
            return dict(self.out)

    web = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky,
                            personal=Stub({"hypothesis": 7, "caps": ["web.page", "web.search", "python"]}))
    f = semantics.interpret("Work out my budget.", web)
    f2 = semantics.Frame(**{**{k: getattr(f, k) for k in f.__dataclass_fields__ if k != "runtime"}, "caps": [],
                            "declined": ["web.search", "web.page"]}, runtime=web)
    f2.personal = {}
    semantics._personal(f2, web)
    check(f2.caps == ["python"], "a declined capability is never offered by a personal reading")
    risky = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky,
                              personal=Stub({"hypothesis": 3, "caps": ["shell", "files.list"], "ask_first": False}))
    f = semantics.interpret("Delete the old report files in C:\\tmp\\x", risky)
    check(f.discernment == "ask", "a learned preference never removes the question before changing files")
    f = semantics.interpret("Open C:\\Users\\santh\\CloverAI\\vault\\keys.txt", risky)
    check(f.discernment == "refuse" and not f.personal, "the vault stays off-limits; no personal reading there")
    asks = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=local_terminal.risky,
                             personal=Stub({"ask_first": True}))
    f = semantics.interpret("Search the web for heat pumps", asks)
    check(f.discernment == "ask", "a learned 'ask me first' adds a question before acting")
    broken = semantics.Runtime(tools=PHONE_TOOLS, device="phone", personal=type("B", (), {
        "interpret": lambda self, t, fr: 1 / 0})())
    check(semantics.interpret("Dig into heat pumps.", broken) == semantics.interpret("Dig into heat pumps.", plain),
          "a failing personal store: Clover works as without it")

    # ------------------------------------------------------------ the person's view and choices
    view = rt.personal.view()
    dig = [e["id"] for e in view["experiences"] if e["text"].startswith("Dig into") and e["how"] == "corrected"]
    check(len(dig) == 2 and any(x["when_you_say"] == ["dig", "into"] for x in view["learned"]),
          "the view lists the experiences and what was learned from them")
    check(sum(e["how"] == "retried after a personal reading" for e in view["experiences"]) == 2,
          "the view shows the retries that counted against the habit")
    again = runtime(store, filter_repo=[])
    check([x.to_json() for x in again.personal.learner.hyps] == [x.to_json() for x in rt.personal.learner.hyps],
          "a new process recomputes the same knowledge from the store")
    news = [e["id"] for e in view["experiences"] if e["text"].startswith(("When I say check the news", "Check the news about"))]
    check(len(news) == 2 and semantics.interpret("Check the news on Mars.", rt).asked == ["web_search"],
          "the news habit rests on two experiences and is in use")
    out = rt.personal.retention(news, "stop")
    kept = {e["id"]: e["retention"] for e in rt.personal.view()["experiences"]}
    check(out["still_referenced"] == 0 and all(kept[i] == "stop" for i in news)
          and semantics.interpret("Check the news on Mars.", rt).caps == [],
          "stop learning: records kept, the habit no longer applied")
    rt.personal.retention(news, "keep")
    check(semantics.interpret("Check the news on Mars.", rt).asked == ["web_search"], "keep: applied again")
    texts = [e["text"] for e in rt.personal.store.all_experiences() if e["id"] in news + dig]
    out = rt.personal.retention(news + dig, "delete")
    left = {e["id"] for e in rt.personal.store.all_experiences()}
    check(out["left_in_store"] == 0 and out["still_referenced"] == 0 and not set(news + dig) & left,
          "delete: gone from the store and from every commit, nothing derived refers to it")
    check(not rt.personal.store.find([t for t in texts if len(t) >= 12]), "delete: no object keeps the deleted text")
    check(semantics.interpret("Check the news on Mars.", rt).caps == [], "delete: the habit that depended on it is gone")
    check("use bullet points" in semantics.interpret("Explain gravity.", rt).guidance, "chat-b's preference in use")
    rt.personal.delete_session("chat-b")
    check(not any(e["id"].startswith("chat-b/") for e in rt.personal.store.all_experiences())
          and "bullet" not in semantics.interpret("Explain gravity.", rt).guidance,
          "a deleted conversation takes its experiences and their preferences with it")
    for bad in (["../../etc/passwd"], ["chat-a/1"], [], ["chat-z/00001"]):
        try:
            rt.personal.retention(bad, "delete")
            check(False, f"bad ids refused {bad}")
        except ValueError:
            check(True, f"bad ids refused {bad}")
    rt.personal.forget_all()
    check(rt.personal.view()["experiences"] == [] and not rt.personal.learner.hyps, "forget everything")

    busy = Path(tmp) / "someone-elses-folder"
    busy.mkdir()
    (busy / "notes.txt").write_text("not Clover's")
    check(personal.for_process(str(busy), "phone", None) is None and not (busy / ".git").exists()
          and (busy / "notes.txt").read_text() == "not Clover's",
          "a folder with other files is never taken over: learning stays off")

    if personal.find_filter_repo():
        rt2 = runtime(Path(tmp) / "person2")
        d = Chat(rt2, "chat-d")
        d.say("Dig into rust lifetimes.")
        d.say("No, look it up online.")
        ids = [e["id"] for e in rt2.personal.store.all_experiences()]
        out = rt2.personal.retention(ids, "delete")
        check(out["left_in_store"] == 0, "delete with git-filter-repo: nothing left")
    else:
        print("SKIP git-filter-repo deletion: not installed here (deletion rebuilt the repository instead)")

# ---------------------------------------------------------------- the learner is PPL-1's
research = HERE.parents[1] / "research" / "personal-process-learning" / "ppl" / "learner.py"
if research.is_file():
    spec = importlib.util.spec_from_file_location("ppl_learner", research)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    exps = []
    for k, (feats, target, general, ctx) in enumerate([
            ({"ctx=code", "has=code", "tok=tidy", "tok=up"}, "code.format", "code.refactor", "code"),
            ({"ctx=docs", "has=doc", "tok=tidy", "tok=up"}, "docs.proofread", "docs.proofread", "docs"),
            ({"ctx=code", "has=code", "tok=tidy"}, "code.format", "code.refactor", "code"),
            ({"tok=park", "ctx=docs"}, "notes.todo", "docs.share", "docs"),
            ({"tok=park", "ctx=code"}, "notes.todo", "code.review", "code"),
            ({"tok=park", "ctx=calendar"}, "calendar.move", "calendar.move", "calendar"),
            ({"ctx=notes", "tok=usual"}, ["docs.export", "docs.share"], "files.open", "notes"),
            ({"ctx=notes", "tok=usual"}, ["docs.export", "docs.share"], "files.open", "notes")]):
        exps.append({"id": f"s0-{k:04d}", "order": k, "context": ctx, "pfeats": sorted(feats), "objects": [],
                     "general": {"intent": general, "conf": 1.0}, "resolved": target, "statement": None,
                     "pres_feedback": [{"dim": "length", "value": "short", "old": "long"}] if k == 1 else [],
                     "accepted_presentation": {"format": "prose"} if k == 2 else {}})
    params = dict(personal.PARAMS)
    a_l, b_l = ref.Learner(params), personal.Learner(params)
    a_l.rebuild(exps)
    b_l.rebuild(exps)
    check([h.to_json() for h in a_l.hyps] == [h.to_json() for h in b_l.hyps]
          and a_l.pref_support == b_l.pref_support and a_l.pref_contra == b_l.pref_contra,
          "the runtime learner reaches PPL-1's knowledge on the same experiences")
else:
    print("SKIP the PPL-1 learner comparison: no research tree in this checkout")

# one personal.py for every process: the laptop Studio ships this file byte for byte
laptop = HERE.parents[2] / "scratch-agents" / "lfm3-hybridagent" / "laptop" / "clover" / "personal.py"
if laptop.is_file():
    check(laptop.read_bytes() == (HERE / "personal.py").read_bytes(), "the laptop's personal.py is this file")
else:
    print("SKIP the laptop's copy: no laptop staging tree in this checkout")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print(f"{PASSED[0]} checks passed")
