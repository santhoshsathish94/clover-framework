"""Offline checks for the fixes after the real-user regression of 2026-11-01: Try again on a follow-up question, lessons
only with evidence, number agreement, workspace context files kept out of git status, and (with the laptop's Python)
memory items joining only related questions.  Usage: python tests/test_regression_fixes.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import clover_cycle as C  # noqa: E402
import workspaces as W  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# Try again on "And how much would 30 pens cost?": the checks see the turn it follows from
q1 = {"role": "user", "content": "A shop sells pens at 3 for $2. How much do 12 pens cost?"}
q2 = {"role": "user", "content": "And how much would 30 pens cost?"}
hist = [q1, {"role": "assistant", "content": "12 x $0.67 = $8.00."}, q2,
        {"role": "assistant", "content": "30 x $0.67 = $20.10."}]
msg, prev = C.last_exchange(hist)
ctx = C.context_before(hist, msg)
check("the follow-up is the question tried again", msg is q2 and "20.10" in prev)
check("its context carries the earlier numbers", "3 for $2" in ctx and "$8.00" in ctx)
check("with that context no web search is wasted on it", C.needs_web("And how much would 30 pens cost?")
      and not C.needs_web(ctx + msg["content"]))
alone = [{"role": "user", "content": "What is the capital of Australia?"}, {"role": "assistant", "content": "Canberra."},
         {"role": "user", "content": "How many legs does a spider have?"}, {"role": "assistant", "content": "6"}]
check("a question that stands alone gets no context", C.context_before(alone, C.last_exchange(alone)[0]) == "")
check("the first question of a chat gets no context", C.context_before(hist[:2], q1) == "")

# numbers: written precision agrees, a different amount does not
check("20.1 and 20 are different answers", not C.same(20.1, 20.0))
check("8 and 8.00 agree; 3.333 and 3.33 agree", C.same(8.0, 8) and C.same(3.333, 3.33))
check("the model's 20.01 and the program's 20.0 differ", not C.same(20.01, 20.0))
check("float noise and roundings agree", C.same(19.999999999999996, 20) and C.same(0.6667, 0.67))
check("letters compare exactly", C.same("B", "B") and not C.same("B", "C"))
check("the amount an answer line states is read, not the count",
      C.final_answer(msg["content"], "30 x $0.67 = $20.01\n\nAnswer: 30 pens cost about $20.01.") == 20.01
      and C.final_answer(msg["content"], "ANSWER: 20.0") == 20.0
      and C.final_answer(msg["content"], "**Answer:** 18 apples") == 18.0)


def run_cycle(solves, program, feedback):
    texts = iter(solves)
    return C.retry_cycle(msg["content"], prev, feedback, rewrite=lambda instr: "rewritten",
                         solve=lambda k, note=None: next(texts), program=(lambda note=None: program) if program else None)


ans, info = run_cycle(["... ANSWER: 30", "... ANSWER: 30"], "", "Try again")
check("two agreeing solves overturn the answer", info.get("action") == "overturned" and "**30**" in ans)
check("...but without the user's words or the program, nothing is learned", "lesson" not in info)
ans, info = run_cycle(["... ANSWER: 20", "... ANSWER: 20"], "Clover ran this Python program\nANSWER: 20", "Try again")
check("solves and program agree on 20: overturned", info.get("action") == "overturned" and "**20**" in ans)
check("...and the program backs it: a verified lesson", info.get("lesson_kind") == "verified")
ans, info = run_cycle(["... ANSWER: 20", "... ANSWER: 20"], "", "That's wrong, 3 pens cost $2 so 30 pens cost $20")
check("the user's own correction: a user lesson", info.get("lesson_kind") == "user")

# workspace context files stay out of git status, and the list goes again on clean
with tempfile.TemporaryDirectory() as d:
    repo = Path(d) / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text('"""App."""\n', encoding="utf-8")
    (repo / "README.md").write_text("# Proj\n", encoding="utf-8")
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout  # noqa: E731
    git("init", "-q"); git("add", "-A"); git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    ws = W.Workspaces(Path(d) / "ws.json", Path(d) / "clones", Path(d) / "app")
    name = ws.add(str(repo))["active"]
    ws.refresh(name)
    status = git("status", "--porcelain").strip()
    exclude = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    check("context files written", any(p.name.endswith("-context-index.md") for p in repo.iterdir()))
    check("git status stays clean", status == "")
    check("exact paths listed, no wildcard", W.EXCLUDE_START in exclude and "*" not in exclude.split(W.EXCLUDE_START, 1)[1])
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    check("the user's new files still show", "notes.txt" in git("status", "--porcelain"))
    ws.clean(name)
    after = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    check("clean removes Clover's files and their list", W.EXCLUDE_START not in after
          and not any(p.name.endswith("-context-index.md") for p in repo.iterdir()))

# memory: a learning joins a question only when they share at least two topic words
import lfm3_layers as L  # noqa: E402
d = Path(tempfile.mkdtemp())
kit = L.LayerKit(d, memory_root=d / "mem")
kit.learn('On a question like "And how much would 30 pens cost?"; Clover\'s checks found: two independent '
          'solutions gave 20, not 20.1.', "s1", 3)
kit.remember("My favourite colour is teal.")
got = lambda q: kit.memory_context(q)[0] or ""  # noqa: E731
check("the pens lesson stays out of disk space", "pens" not in got("How much free disk space does this laptop have?"))
check("...and out of an unrelated price", "pens" not in got("How much does a coffee cost?"))
check("...but joins a question about pens", "pens" in got("How much would 45 pens cost at 3 for $2?"))
check("a note the user asked to remember is still found", "teal" in got("What is my favourite colour?"))
kit.remember("I'm vegetarian and allergic to peanuts.")
n_before = len(kit.memories())
kit.remember("I'm  vegetarian and allergic to peanuts.")
check("the same note twice is kept once", len(kit.memories()) == n_before)
kit.remember("The user is vegetarian and allergic to peanuts.")
check("...also when it is said about the user", len(kit.memories()) == n_before)
kit.remember("I'm not vegetarian any more.")
check("a correction is a new note", len(kit.memories()) == n_before + 1)
check("word order matters", L.note_words("Priya reports to Arun") != L.note_words("Arun reports to Priya"))
check("a diet note joins a dinner question", "vegetarian" in got("Suggest a quick dinner I can cook tonight."))
check("...and a recipe question", "peanuts" in got("Give me a recipe for a cake"))
check("...but not an unrelated one", "vegetarian" not in got("How much free disk space does this laptop have?"))
check("...nor a story about baking", "vegetarian" not in got("Tell me a story about a dragon who learns to bake bread"))
check("...nor a VS Code request with a path", "vegetarian" not in got(
    r"In VS Code, run git log --oneline -3 in C:\Users\santh\CloverAI\tmp\sandbox\demo-repo"))

# the user said "remember ..." and the model only answered: the server saves it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import server as SV  # noqa: E402
m = SV.REMEMBER_ASK.match("Remember that I'm vegetarian and allergic to peanuts.")
check("a remember request is recognised", m and m.group(1).startswith("I'm vegetarian"))
check("a question about remembering is not", not SV.REMEMBER_ASK.match("Do you remember what I said yesterday?")
      and not SV.REMEMBER_ASK.match("What should I remember for the exam?"))

# the Omni's sign-off loop (2026-11-04): one closing paragraph after another, each a little different
import omni_native as O  # noqa: E402
RAMBLE = ("I understand. If there's something specific you'd like to know, feel free to ask! I'm here to help.\n\n"
          "Is there anything else on your mind? I'm ready to assist!\n\n"
          "If you'd like, I can also help you with:\n- Searching\n- Running code\n\n"
          "Just let me know what you'd like to do!\n\n"
          "(And if you'd like, I can also help you with anything else. I'm here to help!)\n\n---\n\n*If you")
cut = O.rambling(RAMBLE)
check("a sign-off loop ends after the first closing paragraph",
      cut is not None and RAMBLE[:cut].endswith("I'm here to help."))
EMAIL = ("Subject: Lunch on Friday\n\nHi team,\n\nLet's have lunch together this Friday at 1 pm.\n\n"
         "Let me know if you can make it.\n\nBest,\nSanthosh\n\nFeel free to adjust the time or place.\n\n"
         "Let me know if you'd like a more formal version.")
check("an email's own 'let me know' and the closing lines are not a loop", O.rambling(EMAIL) is None)
check("one sign-off is an answer's end, not a loop", O.rambling("12.\n\nLet me know if you need more.") is None)

import k3_host_engine as K3  # noqa: E402
cat_names = [(x["id"], x["name"], x["badge"]) for x in K3.owner_k3_models()]
check("k3 dropdown entries have short, clear names and k3 badge",
      cat_names == [("k3:omni", "LFM3-Omni (k3)", "k3"),
                    ("k3:nemotron-3.5-super-vl", "Nemotron 3.5", "k3"),
                    ("k3:kimi-k3", "Kimi K3.1 Pro", "k3")])
with tempfile.TemporaryDirectory() as d:
    import permissions as P
    perms = P.PermissionStore(Path(d) / "p.json")
    nem = K3.K3HostEngine("nemotron-3.5-super-vl", None, Path(d), perms)
    kimi = K3.K3HostEngine("kimi-k3", None, Path(d), perms)
    check("Nemotron 3.5 system prompt states NVIDIA Nemotron 3.5 Super VL",
          "NVIDIA Nemotron 3.5 Super VL" in nem.system_prompt() and "never say LFM3" in nem.system_prompt())
    check("Kimi K3.1 Pro system prompt states Kimi K3.1 Pro 2.8T MoE",
          "Kimi K3.1 Pro 2.8T MoE" in kimi.system_prompt() and "never say LFM3" in kimi.system_prompt())
    check("Nemotron 3.5 identity fix rewrites any LFM3 self-identification",
          "NVIDIA Nemotron 3.5 Super VL" in nem._identity_fix("I am LFM3-Omni-2B, built by Liquid AI.", "What model are you?"))
    check("Kimi K3.1 Pro identity fix rewrites any LFM3 self-identification",
          "Kimi K3.1 Pro 2.8T MoE" in kimi._identity_fix("I was built by Liquid AI.", "Who made you?"))

import semantics as M  # noqa: E402
import lfm3_native as N  # noqa: E402

# CLV-001 & CLV-007: Direct arithmetic verification & 'only' output constraint trimming
check("CLV-001: direct arithmetic 137 × 29 + 46 evaluates to 4019",
      C.direct_arithmetic_value("What is 137 × 29 + 46? Return only the number.") == 4019)
fr_arith = M.interpret("What is 137 × 29 + 46? Return only the number.", N.RUNTIME)
res_arith = {"reply": "4005", "tool_calls": [], "layers": []}
M.align(fr_arith, res_arith)
check("CLV-001: semantics.align corrects 4005 -> 4019 and marks verified",
      res_arith["reply"] == "4019" and fr_arith.knowledge == "verified" and "verified arithmetic" in fr_arith.notes)

fr_code = M.interpret("Reply with only the text file's archive code.", N.RUNTIME)
res_code = {"reply": "The archive code is: **CLOVER-7429**", "tool_calls": [], "layers": []}
M.align(fr_code, res_code)
check("CLV-007: 'only the archive code' strips wrapper text to CLOVER-7429",
      res_code["reply"] == "CLOVER-7429")

fr_yr = M.interpret("Search Wikipedia and reply with only Ada Lovelace's birth year.", N.RUNTIME)
res_yr = {"reply": "Augusta Ada King, Countess of Lovelace — 1815", "tool_calls": [{"tool": "web_search", "ran": True, "result": "1815"}], "layers": []}
M.align(fr_yr, res_yr)
check("CLV-007: 'only birth year' strips wrapper text to 1815",
      res_yr["reply"] == "1815")

# CLV-002: Prior text attachment does not contaminate a new image attachment turn
dummy_omni = object.__new__(O.OmniNative)
dummy_omni.media_dir = Path(".")
prior_hist = [
    {
        "role": "user",
        "content": "File: regression-note.txt\nArchive code: CLOVER-7429"
        + SV.ATTACH_MARKER
        + "What is the archive code in the attached file?",
    },
    {"role": "assistant", "content": "CLOVER-7429"},
]
stripped_h = dummy_omni._history(prior_hist, keep_last_media=False, strip_attached=True)
check(
    "CLV-002: _history(strip_attached=True) removes prior inline file text",
    "CLOVER-7429" not in stripped_h[0]["content"] and "[Earlier file attachment omitted]" in stripped_h[0]["content"],
)

# CLV-003 & CLV-004: Setup includes CSI on laptop, and Layers page reflects fused_available
with tempfile.TemporaryDirectory() as d:
    s_obj = SV.Setup(d)
    check("CLV-003: Laptop Setup defaults to core,csi", s_obj.components == "core,csi")
    check("CLV-006: Setup._run re-probes hardware after download completes", "hardware.probe" in Path(SV.__file__).read_text(encoding="utf-8"))

app_html = (Path(__file__).resolve().parent.parent / "clover" / "web" / "app.html").read_text(encoding="utf-8")
check("app.html has clover-app-body class for site validation", '<body class="clover-app-body">' in app_html)
check("app.html has compact model dropdown in rail__top that fits mobile screens",
      "left: .5rem; right: .5rem;" in app_html and "data-sig" in app_html)
check("CLV-004: app.html checks sv.fused_available before claiming C2/C3 are installed",
      "var fusedOk = !!sv.fused_available" in app_html and "not installed on this laptop" in app_html)

# a public website must not drive the laptop's local server (Windows sets no token): observed 2026-11-06 on a scratch
# harness, Origin https://cloverframework.com changed permissions (shell -> allow) and read chat sessions
import paths as PT  # noqa: E402


def handler_for(origin, sec_fetch_site="cross-site"):
    h = object.__new__(SV.Handler)
    h.headers = {"Origin": origin, "Sec-Fetch-Site": sec_fetch_site} if origin else {}
    return h


web_before, port_before = PT.WEB, SV.S.port
try:
    PT.WEB, SV.S.port = False, 8765
    for site in ("https://cloverframework.com", "https://www.cloverframework.com", "https://evil.example"):
        check(f"local server: a POST from {site} is cross-site", handler_for(site)._cross_site())
        check(f"local server: no CORS read access for {site}", handler_for(site)._cors_headers() == {})
    check("local server: its own page is not cross-site", not handler_for("http://127.0.0.1:8765", "same-origin")._cross_site())
    check("local server: a local program without Origin is not cross-site", not handler_for("", None)._cross_site())
    PT.WEB = True
    check("Web server: the public site keeps CORS access",
          handler_for("https://cloverframework.com")._cors_headers().get("Access-Control-Allow-Origin") == "https://cloverframework.com")
    check("Web server: the public site is not cross-site", not handler_for("https://cloverframework.com")._cross_site())
finally:
    PT.WEB, SV.S.port = web_before, port_before

# Full Clover System (C1 -> C2 HybridAgent -> Bridges -> C3 11-level NCM + S1-S3 + CSI + PPL always on)
import importlib as _importlib  # noqa: E402
import json as _json  # noqa: E402
import omni_c2_c3 as OC2C3  # noqa: E402
CSIA = _importlib.import_module("csi.analyze")

studio_root = Path(__file__).resolve().parent.parent / "clover"
m_win = _json.loads((studio_root / "launch" / "manifest.json").read_text(encoding="utf-8"))
m_and = _json.loads((studio_root / "launch" / "manifest-android.json").read_text(encoding="utf-8"))
win_comps = {x.get("component", "core") for k in ("archives", "files", "datasets") for x in m_win.get(k, [])}
and_comps = {x.get("component", "core") for k in ("archives", "files", "datasets") for x in m_and.get(k, [])}
check("Full system: manifest.json downloads core, csi, and c2c3",
      {"core", "csi", "c2c3"}.issubset(win_comps))
check("Full system: manifest-android.json downloads core, csi, and c2c3",
      {"core", "csi", "c2c3"}.issubset(and_comps))

with tempfile.TemporaryDirectory() as td:
    st_obj = SV.Settings(str(Path(td) / "settings.json"))
    # Even if someone sends false for S1-S3, CSI, or PPL, they stay permanently True
    updated = st_obj.update({
        "csi": False,
        "personal": False,
        "layers": {"memory": False, "ontology": False, "compiler": False, "ncm": False, "recall": False},
    })
    check("Full system: S1-S3, NCM, CSI, and PPL are permanently enabled",
          updated["csi"] is True and updated["personal"] is True
          and all(updated["layers"][k] is True for k in ("memory", "ontology", "compiler", "ncm", "recall")))
    check("Full system: Collective Learning remains opt-in toggleable",
          updated["collective_contribute"] is False and updated["collective_receive"] is False)

check("Full system: Settings UI only has toggle switches for Collective Learning (contribute/receive)",
      'id="lg-personal"' not in app_html and 'id="lg-contribute"' in app_html and 'id="lg-receive"' in app_html)
check("Full system: omni_c2_c3 required files cover HybridAgent, Bridges, and 11-Level NCM",
      "hybridagent/plan-trunk-0/index.bin" in OC2C3.REQUIRED_PRECOMPUTED
      and "bridges/bridge-trunk-0/index.bin" in OC2C3.REQUIRED_PRECOMPUTED
      and "ncm/ncm-trunk-11/index.bin" in OC2C3.REQUIRED_PRECOMPUTED)
fb_frame = CSIA._analyze_with_morph_oewn("Do not search the web; calculate 14 * 29 with Python.")
fb_props = (fb_frame.get("readings") or [{}])[0].get("propositions") or []
check("Full system: CSI morph+OEWN fallback detects negated web search and requested Python",
      len(fb_props) == 2 and fb_props[0]["polarity"] == "negative" and fb_props[0]["predicate"]["lemma"] == "search"
      and fb_props[1]["polarity"] == "positive" and fb_props[1]["predicate"]["lemma"] == "calculate")

local_clover = Path.home() / "CloverAI"
if not OC2C3.missing_c2_c3_files(local_clover):
    rt_c2c3 = OC2C3.for_base(local_clover)
    c2_res = rt_c2c3.plan_turn("Explain how HybridAgent and the 11-level NCM work together in Clover.", ["python_interpreter"])
    c3_text, c3_res = rt_c2c3.render_reply("HybridAgent plans in continuous latent space while the 11-level NCM renders multiscale concept codes.")
    check("Full system: C2 HybridAgent executes 3 latent steps with cosine similarity > 0.90",
          c2_res["layer"] == "c2_hybridagent" and c2_res["latent_steps"] == 3 and c2_res["hybridlink_cos"] > 0.90)
    check("Full system: Bridges + C3 11-Level NCM decodes concept codes and preserves fidelity",
          c3_res["layer"] == "ncm" and c3_res["codes"] > 20 and "HybridAgent plans in continuous latent space" in c3_text)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
