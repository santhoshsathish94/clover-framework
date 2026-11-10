"""Offline checks of the 5.0.3 VS Code fixes (no VS Code, no engine), from the laptop test of 2026-11-10:
F10 a follow-up after a stopped turn is the new request; F1 a page asked for in VS Code opens in VS Code's browser;
F2 VS Code's editor, Problems panel, diff and browser (Playwright) are offered when asked for; plain sums are answered
without the calculator; F3 VS Code's confirmation dialog is reported and given time; F5 "already installed" is said;
F8 a screenshot is shown in the chat, not cut into the model's text; F11 in VS Code's "Copilot" agent the shared pages
reach the model, its camelCase browser tools (runPlaywrightCode) are offered and repaired, and a page is not opened at
an address nobody gave.
Usage: python tests/test_vscode_503.py"""
import base64
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import lfm3_native as N  # noqa: E402
import server  # noqa: E402
import vscode_bridge  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


def call(cid, name, args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}


def user(q):
    return {"role": "user", "content": f"<context>\nThe current date is 2026-11-10.\n</context>\n<userRequest>\n{q}\n</userRequest>\n"}


# ---- F10: the follow-up after a turn that ended without an answer ----
stopped = [
    {"role": "system", "content": "VS Code system prompt"},
    user("Read the file C:\\x\\hello.py and tell me what it prints."),
    call("c1", "read_file", '{"filePath": "C:\\\\x\\\\hello.py"}'),
    {"role": "tool", "tool_call_id": "c1", "content": "def greet(name):\n    return f'Hello, {name}'"},
    call("c2", "run_in_terminal", '{"command": "python hello.py"}'),
    {"role": "tool", "tool_call_id": "c2", "content": "Python was not found"},
]
q, prior, rounds, _ = server.parse_openai_messages(stopped + [user("Do not run anything. What does hello.py print?")])
check("F10 follow-up after a stopped turn is the request", q == "Do not run anything. What does hello.py print?")
check("F10 the stopped turn's tool results stay readable", len(prior) == 1 and "def greet" in prior[0]["reply"])
check("F10 no stale tool rounds", rounds == [])
q, prior, rounds, _ = server.parse_openai_messages(stopped + [user("Continue")])
check("F10 'Continue' carries on with the same request", q.startswith("Read the file") and len(rounds) == 2 and not prior)
q, _, rounds, _ = server.parse_openai_messages(stopped)
check("F10 mid-loop request unchanged", q.startswith("Read the file") and len(rounds) == 2)
plain = [{"role": "user", "content": "Run git status"}, call("c9", "run_in_terminal", '{"command": "git status"}'),
         {"role": "tool", "tool_call_id": "c9", "content": "clean"}, {"role": "user", "content": "What branch is it?"}]
q, prior, rounds, _ = server.parse_openai_messages(plain)
check("F10 a client without VS Code's tags: the last message is the request", q == "What branch is it?" and not rounds)
envelope = stopped + [{"role": "user", "content": "<attachments>\n<attachment id=\"Browser Pages\">\n- [p1] x\n</attachment>\n</attachments>"}]
q, _, rounds, _ = server.parse_openai_messages(envelope)
check("F10 a context-only message after tools keeps them", q.startswith("Read the file") and len(rounds) == 2)

# ---- F2 + calculator: what a message is offered ----
expect = {
    r"In VS Code, check the problems in C:\x\broken.json": ("vscode_diagnostics",),
    "Open http://127.0.0.1:8799/page.html in VS Code's browser and tell me the heading": ("vscode_browser",),
    "Use Playwright to type Ana in the name box and click Greet": ("vscode_browser",),
    "Which file is open in VS Code?": ("vscode_editor",),
    r"Show me a diff of C:\x\a.py and C:\x\b.py in VS Code": ("vscode_diff",),
    "Use the calculator to compute 137 * 29 + 46": ("calculate",),
    "calculate 12 * 7 and then tell me if it is more than 80": ("calculate",),
}
for m, names in expect.items():
    got = N.asked_tools(m)
    check(f"F2 offered {names} for: {m}", all(n in got for n in names))
web = N.asked_tools("Open http://127.0.0.1:8799/page.html in VS Code's browser and tell me the heading")
check("F2 VS Code's browser, not a hidden fetch", "fetch_url" not in web and "web_search" not in web and "vscode_open" not in web)
web = N.asked_tools("Open http://127.0.0.1:8799/second.html in VS Code")
check("F1 'open <url> in VS Code' is opened, not fetched", "vscode_open" in web and "fetch_url" not in web)
check("searching the web about VS Code still searches",
      "web_search" in N.asked_tools("Search the web for the latest VS Code release notes"))
for m in ("What is 5 + 2?", "What is 137 * 29 + 46?", "Calculate 137 * 29 + 46", "And 3 + 3?"):
    check(f"plain sum answered directly: {m}", "calculate" not in N.asked_tools(m))
check("open youtube.com is still the app launcher", "vscode_open" in N.asked_tools("open youtube.com"))
import semantics  # noqa: E402
import clover_cycle as CC  # noqa: E402

check("a plain sum's correct answer is written by Clover", CC.direct_arithmetic_answer("What is 7 * 8 + 3?") == "7 * 8 + 3 = 59")
check("no written answer for a word problem", CC.direct_arithmetic_answer("I buy 7 pens at 12 each; change from 100?") is None)
check("only arithmetic findings count", CC.only_arithmetic(["Arithmetic: you wrote 7 * 8 + 3 = 56, but 7 * 8 + 3 = 59.",
                                                          "Arithmetic: the exact value is 59 (your draft gave 56)."])
      and not CC.only_arithmetic(["Arithmetic: the exact value is 59 (your draft gave 56).", "Code compiler check: x"]))

offer_all = [t["function"]["name"] for t in N.STUDIO_TOOLS]
# Tools "Always" keeps the calculator: without it the model called convert_units with nonsense and took 38 s for
# 137 * 29 + 46 instead of 14 s (laptop, 2026-11-10)
for m, want in (("What is 5 + 2?", True), ("Use the calculator to compute 5 + 2", True),
                ("A shop sells pens at 12 rupees each. I buy 7 pens and pay with 100. How much change?", True)):
    only, _ = semantics.route(semantics.interpret(m, N.RUNTIME), "always", True, offer=offer_all)
    has = only is None or "calculate" in only
    check(f"Tools 'Always': calculator {'offered' if want else 'not offered'} for: {m}", has == want)
check("unchanged: run in VS Code", "vscode_run" in N.asked_tools(r"In VS Code, run git log --oneline -3 in C:\x\repo"))

# ---- Bridge: F3 approval waits ----
tmp = Path(tempfile.mkdtemp())
b = vscode_bridge.Bridge(tmp / "vscode-bridge.json")


def window(answer_after, note=None, polls=3):
    for _ in range(polls):
        job = b.poll("w1", 100.0, "t", "1.145.2", 3.0)
        if job:
            if note:
                b.progress(job["id"], note)
            time.sleep(answer_after)
            b.result(job["id"], True, {"done": job["action"]})
            return


b.poll("w1", 100.0, "t", "1.145.2", 0.0)
notes = []
th = threading.Thread(target=window, args=(2.5, "If VS Code asks \"Open Browser Page?\", approve it there."), daemon=True)
th.start()
res = b.call("browser", {"subaction": "open"}, timeout=1.5, approval_timeout=10, on_wait=notes.append)
th.join(5)
check("F3 a confirmation in VS Code gets more time", res == {"done": "browser"})
check("F3 the wait is reported once", notes == ["If VS Code asks \"Open Browser Page?\", approve it there."])
check("F3 no note left behind", b.waiting == {})
th = threading.Thread(target=window, args=(3.0, "If VS Code asks to run Clover's Playwright code, approve it there."), daemon=True)
th.start()
try:
    b.call("browser", {"subaction": "playwright"}, timeout=0.5, approval_timeout=1.0)
    err = ""
except RuntimeError as exc:
    err = str(exc)
th.join(5)
check("F3 an unanswered dialog is named in the error", "still waiting for your approval" in err and "Playwright" in err)
th = threading.Thread(target=window, args=(3.0,), daemon=True)
th.start()
try:
    b.call("status", {}, timeout=0.5)
    err = ""
except RuntimeError as exc:
    err = str(exc)
th.join(5)
check("F3 a plain slow job still times out as before", "did not finish within" in err)

# ---- F1: where vscode_open sends a web page ----
opened = []
real_open = server.vscode_do.__globals__["Path"]
import device_actions  # noqa: E402

orig_dev, orig_vs = device_actions.open_target, server.S.vscode
device_actions.open_target = lambda target, base=None: opened.append(("device", target)) or {"opened": True, "target": target}


class FakeBridge:
    def ensure(self, exe, timeout=30.0):
        return True

    def call(self, action, args=None, timeout=120.0, approval_timeout=None, on_wait=None):
        opened.append(("vscode", action, (args or {}).get("target")))
        return {"opened": (args or {}).get("target"), "in": "VS Code's browser"}


server.S.vscode = FakeBridge()
try:
    server.vscode_do("open", {"target": "http://127.0.0.1:8799/page.html"}, prefer_vscode=True)
    server.vscode_do("open", {"target": "https://www.youtube.com"}, prefer_vscode=False)
    server.vscode_do("open", {"target": "spotify"}, prefer_vscode=True)
finally:
    device_actions.open_target, server.S.vscode = orig_dev, orig_vs
check("F1 a page asked for in VS Code opens in VS Code", opened[0] == ("vscode", "open", "http://127.0.0.1:8799/page.html"))
check("F1 'open youtube' without VS Code goes to the device", opened[1] == ("device", "https://www.youtube.com"))
check("F1 an app still opens on the device", opened[2] == ("device", "spotify"))

# ---- F5, F8: the result the model reads ----
txt = json.loads(server.vscode_result_text("install", {"extension": "ms-python.python", "installed": True, "already": True}))
check("F5 already installed is said", "already installed" in txt.get("note", ""))
png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
media = tmp / "media"
media.mkdir()
orig_md = server.media_dir
server.media_dir = lambda: media
made = []
try:
    out = server.vscode_result_text("browser", {"subaction": "screenshot", "output": "Captured",
                                                "images": [{"mimeType": "image/png", "data": base64.b64encode(png).decode()}]},
                                    made)
finally:
    server.media_dir = orig_md
check("F8 screenshot shown in the chat", len(made) == 1 and (media / made[0]["file"]).read_bytes() == png)
check("F8 the model reads a note, not base64", "Clover shows the screenshot" in out and "iVBOR" not in out)

# ---- Playwright in Studio's VS Code browser tool ----
server.S.vscode = FakeBridge()
try:
    tools = {t["schema"]["function"]["name"]: t for t in server.vscode_tools("chat-pw")}
finally:
    server.S.vscode = orig_vs
br = tools["vscode_browser"]["schema"]["function"]["parameters"]["properties"]
check("vscode_browser has a playwright subaction with code", "playwright" in br["subaction"]["enum"] and "code" in br)
check("the permission request shows the code",
      "page.click" in tools["vscode_browser"]["detail"]({"subaction": "playwright", "code": "await page.click('text=Greet')"}))

# ---- grounding: an address the user never gave is not opened when VS Code shares pages ----
pages = "- [50a3ecef-3a57-4210-8c6c-7309aab58526] Clover VS Code test page (http://127.0.0.1:8799/page.html) (active)"
calls_seen = []


class RecBridge(FakeBridge):
    def call(self, action, args=None, timeout=120.0, approval_timeout=None, on_wait=None):
        calls_seen.append((action, dict(args or {})))
        return {"subaction": (args or {}).get("subaction"), "output": "ok"}


server.S.vscode = RecBridge()
server.S.turn_ctx["chat-g"] = {"emit": lambda ev: None, "made": [], "text": "Use Playwright on the Clover VS Code test page"}
try:
    g = {t["schema"]["function"]["name"]: t for t in server.vscode_tools("chat-g", pages=pages, playwright=True)}
    blocked = g["vscode_browser"]["run"]({"subaction": "open", "url": "https://clover.com"})
    server.S.turn_ctx["chat-g"]["text"] = "Open example.com in VS Code's browser"
    allowed = g["vscode_browser"]["run"]({"subaction": "open", "url": "https://www.example.com/"})
finally:
    server.S.vscode = orig_vs
    server.S.turn_ctx.pop("chat-g", None)
desc = g["vscode_browser"]["schema"]["function"]["description"]
check("an invented address is refused with the shared pages", blocked.startswith("Error:") and "50a3ecef" in blocked)
check("the refused address never reached VS Code", not any(a.get("url") == "https://clover.com" for _, a in calls_seen))
check("an address the user named is opened", calls_seen and calls_seen[-1][1].get("url") == "https://www.example.com/")
check("the description lists the shared pages and the Playwright request", "50a3ecef" in desc and "Playwright" in desc)

snap = ("- generic [active] [ref=e1]:\n  - heading \"Clover browser test\" [level=1] [ref=e2]\n  - text: Your name\n"
        "  - textbox \"Your name\" [ref=e3]: Clover\n  - button \"Greet\" [ref=e4]\n  - paragraph [ref=e5]: Not greeted yet\n"
        "  - link \"Second page\" [ref=e6] [cursor=pointer]:")
loc = server.playwright_locators(snap)
check("Playwright locators from a page snapshot", "page.getByLabel('Your name')" in loc
      and "page.getByRole('button', { name: 'Greet' })" in loc and "page.getByText('Not greeted yet')" in loc)
wrapped = "async (page) => {\n  await page.fill('#name', 'Ana');\n  return await page.textContent('#status');\n}"
check("Playwright code written as a function is unwrapped",
      N.playwright_body(wrapped) == "await page.fill('#name', 'Ana');\n  return await page.textContent('#status');")
check("plain Playwright code is unchanged", N.playwright_body("return page.title()") == "return page.title()")
check("VS Code chat's run_playwright_code gets the body",
      N.repair_vscode_browser_args("run_playwright_code", {"pageId": "p1", "code": wrapped})["code"].startswith("await page.fill"))
nothing = json.loads(server.vscode_result_text("browser", {"subaction": "playwright", "output": "Page Title: x\nSnapshot: <unchanged>"}))
check("code that returned nothing is said", "returned no value and the page did not change" in nothing.get("note", ""))
failed_txt = server.vscode_result_text("browser", {"subaction": "playwright", "output": "page.click: Timeout 10000ms exceeded.\nCall log:"})
check("a failed Playwright run is an error", failed_txt.startswith("Error: the Playwright code failed"))
code0 = "await page.fill('#name', 'Ana');\nawait page.click('button.Greet');\nreturn await page.textContent('#status');"
fixed = server.repair_playwright_code(code0, "button.Greet", snap)
check("a guessed selector is replaced by the page's locator",
      fixed == "await page.fill('#name', 'Ana');\nawait page.getByRole('button', { name: 'Greet' }).click();\nreturn await page.textContent('#status');")
check("no single match: code unchanged", server.repair_playwright_code(code0, "button.Submit", snap) == code0)
fill_fix = server.repair_playwright_code("await page.fill('input.name', 'Ana');", "input.name", snap)
check("a guessed text box selector becomes its label", fill_fix == "await page.fill('input.name', 'Ana');"
      or fill_fix == "await page.getByLabel('Your name').fill( 'Ana');")

# ---- F11: VS Code's "Copilot" agent (request captured on the laptop, 2026-11-10) ----
PID = "6f1ea729-c128-4a92-9f7d-c4c94c434a4e"
ASK = ("Use Playwright on the shared Clover VS Code test page: type Maya in the name box, click Greet, and tell me what "
       "the page says.")
copilot_user = {"role": "user", "content": (
    "<current_datetime>2026-11-10T10:52:21.361+05:30</current_datetime>\n\n" + ASK + "\n\n<system_reminder>\nReminder: "
    "This chat currently has an auto-generated or placeholder title. Please give the chat a short, descriptive title "
    "by calling `rename_chat` with `automatic: true`.\n</system_reminder><attachment "
    "mimeType=\"text/x-vscode-simple-attachment\" name=\"Browser Pages\">\nThe following browser pages are currently "
    "shared with you and can be interacted with using the browser tools:\n- [" + PID + "] Clover VS Code test page "
    "(http://127.0.0.1:8799/page.html) (active)\n</attachment>")}
q, _, rounds, ws = server.parse_openai_messages([{"role": "system", "content": "Copilot SDK prompt"}, copilot_user])
check("F11 the request without the agent's date and title reminder", q == ASK and not rounds)
check("F11 the shared page reaches the model", PID in ws and "http://127.0.0.1:8799/page.html" in ws)
act = N.active_vscode_tools(q, ws, [])
check("F11 the agent's camelCase browser tools are offered",
      all(n in act for n in ("openBrowserPage", "readPage", "clickElement", "typeInPage", "runPlaywrightCode")))
act_core = N.active_vscode_tools("Run git --version in the terminal and tell me the version.", "", [])
check("F11 the agent's own core tools are offered without a search",
      all(n in act_core for n in ("view", "grep", "glob", "powershell", "edit", "create", "problems", "read_powershell")))
rp = N.repair_vscode_browser_args("runPlaywrightCode", {"code": wrapped}, ws, [])
check("F11 runPlaywrightCode gets the shared pageId and the code body",
      rp.get("pageId") == PID and rp["code"].startswith("await page.fill"))
for bad in ("https://clover.com", "file:///shared/clover/shared.html"):
    msg = N.unshared_open("openBrowserPage", {"url": bad}, q, ws, [])
    check(f"F11 not opened, an address nobody gave: {bad}", bool(msg) and PID in msg and "runPlaywrightCode" in msg)
check("F11 another page of the shared site may open",
      N.unshared_open("openBrowserPage", {"url": "http://127.0.0.1:8799/second.html"}, q, ws, []) is None)
check("F11 an address the user typed opens",
      N.unshared_open("open_browser_page", {"url": "https://example.com/"}, "Open https://example.com/", ws, []) is None)
check("F11 no shared page: VS Code asks the user as before",
      N.unshared_open("openBrowserPage", {"url": "https://clover.com"}, q, "", []) is None)
check("F11 the Local agent's refusal names its own tools",
      "run_playwright_code" in (N.unshared_open("open_browser_page", {"url": "https://clover.com"}, q, ws, []) or ""))

import omni_native  # noqa: E402

eng = object.__new__(omni_native.OmniNative)
eng.lock, eng.stage, eng.media_dir = threading.Lock(), {}, tmp
script = [("", [{"id": "k1", "name": "openBrowserPage", "arguments": {"url": "https://clover.com"}}], {}),
          ("", [{"id": "k2", "name": "runPlaywrightCode", "arguments": {"code": wrapped}}], {})]
seen = []


def fake_chat(msgs, schemas, max_new, by_name):
    seen.append((json.loads(json.dumps(msgs)), [s["function"]["name"] for s in schemas if not s.get("defer_loading")]))
    return script[len(seen) - 1]


eng._chat = fake_chat
fn = lambda name: {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object"}}}
out = eng.vscode_step(q, ws, [], [], [fn(n) for n in ("powershell", "view", "openBrowserPage", "readPage",
                                                   "clickElement", "typeInPage", "runPlaywrightCode", "rename_chat")])
tc = out.get("tool_call") or {}
check("F11 the step returns the Playwright call, not the invented address", tc.get("name") == "runPlaywrightCode")
check("F11 the returned call has the pageId and the code body",
      json.loads(tc.get("arguments") or "{}").get("pageId") == PID and "await page.fill" in (tc.get("arguments") or ""))
check("F11 the model was told why and given the shared page",
      len(seen) == 2 and seen[1][0][-1]["role"] == "tool" and "Not opened" in seen[1][0][-1]["content"]
      and PID in seen[1][0][-1]["content"])
check("F11 browser tools offered without a search", "runPlaywrightCode" in seen[0][1] and "openBrowserPage" in seen[0][1])
check("F11 the refusal is in the trace", out["trace"][0].get("refused") == "https://clover.com")

# the model read the page with Playwright and answered without typing or clicking (laptop, 2026-11-10)
read_code = '{"pageId":"%s","code":"const text = await page.evaluate(() => document.body.innerText);"}' % PID
read_out = ("Page Title: Clover VS Code test page\nURL: http://127.0.0.1:8799/page.html\nSnapshot: \n- generic [ref=e1]:\n"
            "  - heading \"Clover browser test\" [level=1] [ref=e2]\n  - text: Your name\n"
            "  - textbox \"Your name\" [ref=e3]: Ana\n  - button \"Greet\" [ref=e4]\n  - paragraph [ref=e5]: Hello, Ana")
read_rd = [{"id": "r1", "name": "runPlaywrightCode", "arguments": read_code, "output": read_out}]
tools_c = ["openBrowserPage", "readPage", "runPlaywrightCode"]
nudge = N.unfinished_page_steps(q, ws, read_rd, tools_c)
check("F11 an answer before the asked steps sends the model back with the page's locators",
      "runPlaywrightCode" in nudge and "page.getByLabel('Your name')" in nudge
      and "page.getByRole('button', { name: 'Greet' })" in nudge)
did = [dict(read_rd[0], arguments='{"code":"await page.getByLabel(\'Your name\').fill(\'Maya\');"}')]
check("F11 no note once the steps were done with Playwright", N.unfinished_page_steps(q, ws, did, tools_c) == "")
check("F11 no note once the steps were done with typeInPage and clickElement", N.unfinished_page_steps(q, ws, [
    {"name": "typeInPage", "arguments": '{"text": "Maya", "ref": "e3"}', "output": "ok"},
    {"name": "clickElement", "arguments": '{"ref": "e4"}', "output": "ok"}], tools_c) == "")
# Local agent (laptop, 2026-11-10): it reloaded the page, clicked Greet without typing, and said it had typed Ravi
ravi = "Use Playwright on the shared Clover VS Code test page: type Ravi in the name box, click Greet, and tell me."
skipped_typing = [{"name": "navigate_page", "arguments": '{"type": "url", "url": "http://127.0.0.1:8799/page.html"}',
                   "output": read_out}, {"name": "click_element", "arguments": '{"element": "Greet", "ref": "e4"}',
                                         "output": "Clicked Greet"}]
note = N.unfinished_page_steps(ravi, ws, skipped_typing, ["read_page", "click_element", "run_playwright_code"])
check("F11 a step left out is named: the text was never typed",
      "'Ravi' was never typed" in note and "run_playwright_code" in note and "page.getByLabel('Your name')" in note)
check("F11 no note once the text was typed", N.unfinished_page_steps(ravi, ws, skipped_typing + [
    {"name": "type_in_page", "arguments": '{"text": "Ravi", "ref": "e3"}', "output": "ok"}],
    ["read_page", "click_element", "run_playwright_code"]) == "")
check("F11 a description is not taken for the text to type", N.unfinished_page_steps(
    "On the shared page, enter your email in the box and click Send.", ws, skipped_typing, tools_c) == "")
check("F11 a label that names a control is not offered as text", "getByText('Your name')" not in N.playwright_locators(read_out)
      and "page.getByLabel('Your name')" in N.playwright_locators(read_out))
local_tools = ["read_page", "navigate_page", "click_element", "type_in_page", "run_playwright_code"]
same = {"pageId": PID, "type": "url", "url": "http://127.0.0.1:8799/page.html"}
check("F11 navigating a shared page to its own address becomes a read",
      N.shared_page_read("navigate_page", same, ravi, ws, local_tools) == {"name": "read_page", "arguments": {"pageId": PID}})
check("F11 a reload the user asked for stays a navigation",
      N.shared_page_read("navigate_page", same, "Reload the shared page, then click Greet.", ws, local_tools) is None)
check("F11 another address stays a navigation", N.shared_page_read(
    "navigate_page", dict(same, url="http://127.0.0.1:8799/second.html"), ravi, ws, local_tools) is None)
script = [("", [{"id": "k6", "name": "navigate_page", "arguments": dict(same)}], {})]
seen.clear()
out = eng.vscode_step(ravi, ws, [], [], [fn(n) for n in local_tools])
check("F11 the step reads the shared page instead of reloading it",
      (out.get("tool_call") or {}).get("name") == "read_page" and PID in out["tool_call"]["arguments"])
check("F11 no note after the user declined a tool", N.unfinished_page_steps(
    q, ws, [{"name": "openBrowserPage", "arguments": "{}", "output": "The user rejected this tool call."}], tools_c) == "")
check("F11 no note for a question about the page", N.unfinished_page_steps(
    "What does the shared page say? What should I type in the name box?", ws, read_rd, tools_c) == "")
check("F11 no note without a shared page", N.unfinished_page_steps(q, "", read_rd, tools_c) == "")
script = [("The page says: Hello, Ana.", [], {}),
          ("", [{"id": "k3", "name": "runPlaywrightCode", "arguments": {
              "code": "await page.getByLabel('Your name').fill('Maya');\nawait page.getByRole('button', { name: 'Greet' })"
                      ".click();\nreturn await page.getByText(/Hello/).textContent();"}}], {})]
seen.clear()
out = eng.vscode_step(q, ws, [], read_rd, [fn(n) for n in ("powershell", "view", "openBrowserPage", "readPage",
                                                         "clickElement", "typeInPage", "runPlaywrightCode")])
tc = out.get("tool_call") or {}
check("F11 sent back once, the model does the steps", tc.get("name") == "runPlaywrightCode" and ".fill(" in tc["arguments"]
      and len(seen) == 2 and seen[1][0][-1]["role"] == "user" and "Not done yet" in seen[1][0][-1]["content"])
script = [("The page says: Hello, Ana.", [], {}), ("The page says: Hello, Ana.", [], {})]
seen.clear()
out = eng.vscode_step(q, ws, [], read_rd, [fn(n) for n in ("readPage", "runPlaywrightCode")])
check("F11 sent back at most once", out.get("content") == "The page says: Hello, Ana." and len(seen) == 2)
script = [("The page says: Hello, Ana.", [], {}),
          ("", [{"id": "k4", "name": "runPlaywrightCode", "arguments": json.loads(read_code)}], {})]
seen.clear()
out = eng.vscode_step(q, ws, [], read_rd, [fn(n) for n in ("readPage", "runPlaywrightCode")])
check("F11 sent back, a repeat of the read already done gives the answer, not the same call again",
      out.get("content") == "The page says: Hello, Ana." and not out.get("tool_call"))
check("F11 no note after three browser rounds", N.unfinished_page_steps(q, ws, read_rd * 3, tools_c) == "")
pw_schema = {"type": "function", "function": {"name": "runPlaywrightCode", "description": "Run a Playwright code snippet.",
                                              "parameters": {"type": "object", "properties": {
                                                  "pageId": {"type": "string"}, "code": {"type": "string", "description":
                                                  "e.g. \"return page.evaluate(() => document.title)\""}}}}}
script = [("", [{"id": "s1", "name": "search_tools", "arguments": {"query": "playwright"}}], {}),
          ("", [{"id": "k5", "name": "runPlaywrightCode", "arguments": {"code": "return 1"}}], {})]
seen.clear()
out = eng.vscode_step(q, ws, [], [], [pw_schema, fn("powershell"), fn("view")])
check("F11 search_tools finds a tool the model already has",
      len(seen) == 2 and "runPlaywrightCode" in seen[1][0][-1]["content"] and seen[1][0][-1]["content"].startswith("Found 1"))
code_desc = N.vscode_tool_params("runPlaywrightCode", pw_schema["function"]["parameters"])["properties"]["code"]["description"]
check("F11 the Playwright example fills and clicks", ".fill(" in code_desc and ".click()" in code_desc and "return" in code_desc
      and "document.title" in pw_schema["function"]["parameters"]["properties"]["code"]["description"])

# ---- F12: an answer written for VS Code's chat does not act in VS Code behind that chat ----
from types import SimpleNamespace  # noqa: E402


class StandIn:
    inputs = ("text", "vision")

    def __init__(self):
        self.calls = []

    def run_turn(self, history, user_text, **kw):
        self.calls.append(kw)
        return {"reply": "ok", "tool_calls": [], "rounds": 1, "layers": []}


class Settings:
    def get(self):
        return {"tools": "always", "layers": {"memory": False, "ontology": False, "compiler": False, "ncm": False,
                                              "recall": False, "kv_slots": False}}


class ListBridge(FakeBridge):
    def connected(self):
        return True

    def call(self, action, args=None, timeout=120.0, approval_timeout=None, on_wait=None):
        return {"output": pages}


keep = ("engine", "settings", "args", "perms", "repo", "workspaces", "vscode", "memidx", "turn_lock", "sess_lock", "sessions")
saved = {k: getattr(server.S, k, None) for k in keep}
standin = StandIn()
server.S.engine, server.S.settings, server.S.args = standin, Settings(), SimpleNamespace(base=str(tmp))
server.S.perms = SimpleNamespace(get=lambda: {"memory": "allow"})
server.S.repo = server.S.workspaces = server.S.memidx = None
server.S.vscode = ListBridge()
server.S.turn_lock, server.S.sess_lock, server.S.sessions = threading.Lock(), threading.Lock(), {}
try:
    server.lfm3_turn("vsc-t", ASK, {}, kind="VS Code answer")
    from_vscode = [t["schema"]["function"]["name"] for t in standin.calls[-1]["extra_tools"]]
    server.lfm3_turn("chat-t", ASK, {})
    from_studio = [t["schema"]["function"]["name"] for t in standin.calls[-1]["extra_tools"]]
finally:
    for k, v in saved.items():
        setattr(server.S, k, v)
check("F12 an answer for VS Code's chat gets no VS Code tools", not any(n.startswith("vscode_") for n in from_vscode))
check("F12 Studio's own chat still has VS Code's browser", "vscode_browser" in from_studio)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
