"""Layer 1 alone: LFM3-1.7B run the way its model card documents it, on the Clover C engine.

- Chat template from the checkpoint (ChatML, reasoning on by default, earlier turns' reasoning dropped).
- Sampling from the model card: general = temperature 0.1, top_k 50, repetition_penalty 1.1; math & STEM =
  temperature 1.0, top_p 0.95, top_k 20; greedy for benchmarks (the card's ToolSandbox setting).
- Native tool calling: tool schemas through the template, XML calls parsed, results appended as `tool` messages, the
  assistant's reasoning and content kept within the tool-use turn. VS Code tools beyond the core set are deferred
  (`defer_loading`) behind a `search_tools` tool, as the card describes.
- Engine: liblfm3_cpu_engine (bf16 math, weights memory-mapped, cache with prefix reuse). No torch, no planner, no NCM.
Tools outside the model run only with the user's permission (permissions.py modes; `ask` waits for the user).
"""
import ctypes
import datetime
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # read once when libgomp loads; spinning wastes the power budget
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import numpy as np  # noqa: E402

import local_terminal  # noqa: E402
import local_tool_packs  # noqa: E402
import page_fetch  # noqa: E402
import paths  # noqa: E402
import permissions as P  # noqa: E402
import semantics  # noqa: E402

D_MODEL, V_ROWS = 2048, 128000
SAMPLING = {
    "general": {"temperature": 0.1, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.1},
    "math": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.0},
    "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
}
TOOL_OUTPUT_CHARS = 6000
_PF = ctypes.POINTER(ctypes.c_float)


def _fn(name, description, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": list(required)}}}


STUDIO_TOOLS = [
    _fn("run_python", "Run Python 3 code on this laptop and return what it prints. Use print() for results. pandas, "
        "openpyxl, numpy and matplotlib are installed (read an attached document from the path it was saved at). To show "
        "a chart, save it as a PNG file (plt.savefig('chart.png')); the user sees it in the chat. 60-second limit.",
        {"code": {"type": "string", "description": "Python source code"}}, ["code"]),
    _fn("web_search", "Search the web. Returns results with links.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("fetch_url", "Download a web page and return its main text.", {"url": {"type": "string"}}, ["url"]),
    _fn("run_powershell", "Run a PowerShell command on this Windows laptop (git, python, pip and other programs too); "
        "returns its output and exit code. Use commands that exist, e.g. git log --oneline -5, git status, "
        "Get-PSDrive C, Get-ChildItem, Remove-Item (to delete). "
        "The time limit is 60 seconds unless you set timeout_s (up to 600) for a longer command.",
        {"command": {"type": "string"},
         "cwd": {"type": "string", "description": "the folder to run in: when the user names a folder, put its full "
                                                 "path here; default: the open workspace or the home folder"},
         "timeout_s": {"type": "integer", "description": "time limit in seconds, 1-600"}}, ["command"]),
    _fn("read_file", "Read a text file on this laptop. Optional 1-based line range.",
        {"path": {"type": "string", "description": "absolute Windows path"}, "start_line": {"type": "integer"},
         "end_line": {"type": "integer"}}, ["path"]),
    _fn("list_directory", "List the files and folders in a folder on this laptop.",
        {"path": {"type": "string", "description": "absolute Windows path"}}, ["path"]),
    _fn("write_file", "Create or overwrite a text file on this laptop.",
        {"path": {"type": "string", "description": "absolute Windows path"}, "content": {"type": "string"}},
        ["path", "content"]),
    _fn("get_current_time", "Current local date and time on this laptop."),
    _fn("laptop_status", "This laptop's free disk space on each drive, battery level, memory in use, and the apps using "
        "the most memory."),
    _fn("manage_todo_list", "Create or update the structured task todo list (`id`, `title`, `status` in "
        "`not-started | in-progress | completed`) to track progress on multi-step work.",
        {"todoList": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "title": {"type": "string"}, "status": {"type": "string"}}}}},
        ["todoList"]),
] + list(local_tool_packs.PACK_TOOLS)
if paths.ANDROID or paths.WEB:  # the phone app and web server: no PowerShell and no Windows status
    _drop_tools = ("run_powershell", "laptop_status") + (("git_inspect", "media_info", "extract_frame") if paths.ANDROID else ())
    STUDIO_TOOLS = [json.loads(json.dumps(t).replace("this Windows laptop", f"this {paths.DEVICE_WORD}")
                               .replace("absolute Windows path", "path").replace("this laptop", f"this {paths.DEVICE_WORD}"))
                    for t in STUDIO_TOOLS if t["function"]["name"] not in _drop_tools]
TOOL_PERMISSION = {"run_python": "code", "web_search": "internet", "fetch_url": "internet", "run_powershell": "shell",
                   "read_file": "files", "list_directory": "files", "write_file": "edit_files", "get_current_time": None,
                   "laptop_status": "shell", "manage_todo_list": None, **local_tool_packs.PACK_PERMISSIONS}
# Tools only when the message asks for them (human, 2026-10-31): a tool list in every prompt cost LFM3-Omni-2B 6 of 65
# ladder tasks (it searched instead of answering, and its arithmetic got worse). The rules are common (semantics.py);
# the laptop's own part is which of its tools serves each capability.
RUNTIME = semantics.Runtime(tools={
    "web.search": ("web_search",), "web.page": ("fetch_url",), "python": ("run_python",), "files.read": ("read_file",),
    "files.list": ("list_directory",), "files.write": ("write_file",), "shell": ("run_powershell",),
    "status": ("laptop_status",), "clock": ("get_current_time",), "remember": ("remember",), "image": ("generate_image",),
    "vscode.open": ("vscode_open",), "vscode.clone": ("vscode_clone",), "vscode.install": ("vscode_install",),
    "vscode.run": ("vscode_run",), "github.setup": ("vscode_github_setup",), "github.sync": ("vscode_github_sync",),
    "vscode.editor": ("vscode_editor",), "vscode.diagnostics": ("vscode_diagnostics",), "vscode.diff": ("vscode_diff",),
    "vscode.browser": ("vscode_browser",),
    "calc": ("calculate",), "units": ("convert_units",), "files.search": ("search_files",), "files.info": ("file_info",),
    "git.inspect": ("git_inspect",), "doc.read": ("read_document",), "table.analyze": ("analyze_table",),
    "table.chart": ("chart_table",), "media.info": ("media_info",), "media.frame": ("extract_frame",)},
    device="laptop", risky=local_terminal.risky)
if paths.ANDROID or paths.WEB:
    _skip = ("shell", "status") + (
        ("files.read", "files.list", "files.write", "files.search", "files.info", "git.inspect", "doc.read", "table.analyze", "table.chart", "media.info", "media.frame")
        if paths.WEB else ("git.inspect", "media.info", "media.frame")
    )
    RUNTIME = semantics.Runtime(tools={k: v for k, v in RUNTIME.tools.items()
                                       if k not in _skip and (k == "vscode.open" or not k.startswith(("vscode.", "github.")))},
                                device=paths.DEVICE_WORD, risky=local_terminal.risky)
CONCEPT, EXPLICIT, EDIT, CREATIVE = semantics.CONCEPT, semantics.EXPLICIT, semantics.EDIT, semantics.CREATIVE


def asked_tools(text):
    """Names of the tools a message asks for; empty means answer without tools."""
    return semantics.asked(text, RUNTIME)


TOOLS_ON = ("Use the tools when they help; their results are real. Answer directly what you know well; search the web "
            "for weather, recent events, live information or facts you are not sure of. Never say you lack real-time "
            "access when web_search is available—call web_search instead. If a tool is not allowed, answer without it "
            "and say what you could not check.")
TOOLS_OFF = "Answer helpfully, clearly, and thoroughly from your knowledge."


def tools_sentence(tools_on, asked=None):
    """System prompt part about tools; `asked`: the tools this message asked for (asked-only mode)."""
    if not tools_on:
        return TOOLS_OFF
    if asked:  # offered but not used in a test: "Search the web for ..." was answered from memory
        return ("The user asked for a tool in this message: call " + " or ".join(asked) + " immediately before you answer. "
                "Its results are real. After it returns, answer directly from its results without asking unnecessary "
                "follow-up questions. If it is not allowed, answer without it and say what you could not check.")
    return TOOLS_ON
VSCODE_CORE = ["read_file", "list_dir", "file_search", "grep_search", "semantic_search", "run_in_terminal",
               "get_errors", "replace_string_in_file", "create_file", "insert_edit_into_file", "apply_patch",
               "manage_todo_list"]
VSCODE_BROWSER_TOOLS = ["open_browser_page", "list_browser_pages", "read_page", "click_element", "type_in_page",
                        "navigate_page", "screenshot_page", "hover_element", "drag_element", "handle_dialog",
                        "run_playwright_code", "fetch_webpage"]
# VS Code's default "Copilot" agent names the same browser tools in camelCase (openBrowserPage, runPlaywrightCode): none
# was recognised, so they stayed behind search_tools and got no pageId or code repair (laptop, 2026-11-10)
VSCODE_BROWSER_CAMEL = {re.sub(r"_([a-z])", lambda m: m.group(1).upper(), n): n for n in VSCODE_BROWSER_TOOLS}


def browser_tool(name):
    """The Local agent's name of a VS Code browser tool (open_browser_page for openBrowserPage); others unchanged."""
    return VSCODE_BROWSER_CAMEL.get(name, name)


def vscode_tool_params(name, params):
    """VS Code's parameters as the model sees them. run_playwright_code's only example reads document.title, and for
    "type Maya in the name box, click Greet" LFM3-Omni-2B wrote code that only read the page (laptop, 2026-11-10)."""
    if browser_tool(name) != "run_playwright_code" or not isinstance(((params or {}).get("properties") or {}).get("code"), dict):
        return params
    p = json.loads(json.dumps(params))
    p["properties"]["code"]["description"] = (
        "Playwright code, the body of async (page) => { ... }: do every step and return the result, e.g. await "
        "page.getByLabel('Name').fill('Ana'); await page.getByRole('button', { name: 'Send' }).click(); return await "
        "page.locator('body').innerText();")
    return p


VSCODE_TERMINAL_TOOLS = ["run_in_terminal", "get_terminal_output", "send_to_terminal", "kill_terminal",
                         "terminal_last_command", "terminal_selection"]
VSCODE_CODE_INTEL_TOOLS = ["vscode_listCodeUsages", "vscode_renameSymbol", "view_image", "multi_replace_string_in_file"]
# VS Code's "Copilot" agent names its core tools differently: all of them sat behind search_tools, and "Run git --version
# in the terminal" searched four times without running anything (laptop, 2026-11-10)
VSCODE_CORE += ["view", "glob", "grep", "powershell", "bash", "problems", "edit", "create"]
VSCODE_TERMINAL_TOOLS += ["read_powershell", "stop_powershell", "list_powershell", "read_bash", "stop_bash", "list_bash"]
VSCODE_CODE_INTEL_TOOLS += ["usages"]
_VSCODE_BROWSER_CUE = re.compile(
    r"\b(?:browser|web\s*page|page\s+id|shared\s+with\s+you|open_browser_page|read_page|click_element|"
    r"type_in_page|screenshot_page|navigate_page|playwright|localhost|127\.0\.0\.1|https?://|dom\b|ui\s+page)\b",
    re.I,
)
_VSCODE_TERM_CUE = re.compile(
    r"\b(?:terminal|background\s+command|send_to_terminal|get_terminal_output|kill_terminal|last\s+command|terminal_selection)\b",
    re.I,
)
_VSCODE_INTEL_CUE = re.compile(
    r"\b(?:usages?|references?|where\s+is\s+[\w.]+\s+used|rename\s+symbol|vscode_listCodeUsages|vscode_renameSymbol|"
    r"view_image|\.png\b|\.jpg\b|\.jpeg\b|\.webp\b)\b",
    re.I,
)


def active_vscode_tools(user_prompt="", workspace_info="", tool_rounds=(), deferred=()):
    """Selects the non-deferred VS Code tool set for `vscode_step`, activating VS Code's Integrated Browser,
    Terminal, and Code Intelligence tools whenever the user prompt, shared browser pages, or prior rounds need them.
    `deferred`: tools usage evidence says are never called here (tool_usage); the core and this turn's tools stay."""
    active = set(VSCODE_CORE)
    used = {rd.get("name") for rd in (tool_rounds or ()) if isinstance(rd, dict) and rd.get("name")}
    active |= used
    combined = f"{user_prompt or ''}\n{workspace_info or ''}"
    if _VSCODE_BROWSER_CUE.search(combined) or ({browser_tool(n) for n in used} & set(VSCODE_BROWSER_TOOLS)):
        active.update(VSCODE_BROWSER_TOOLS)
        active.update(VSCODE_BROWSER_CAMEL)
    if _VSCODE_TERM_CUE.search(combined) or (used & set(VSCODE_TERMINAL_TOOLS)):
        active.update(VSCODE_TERMINAL_TOOLS)
    if _VSCODE_INTEL_CUE.search(combined) or (used & set(VSCODE_CODE_INTEL_TOOLS)):
        active.update(VSCODE_CODE_INTEL_TOOLS)
    return active - (set(deferred) - set(VSCODE_CORE) - used)


def repair_vscode_browser_args(call_name, call_args, workspace_info="", tool_rounds=()):
    """Ensures VS Code Integrated Browser tool calls (`click_element`, `type_in_page`, `hover_element`, `read_page`, etc.)
    always include `pageId` (from shared browser pages or prior rounds) and `ref` (matched from the accessibility snapshot
    in `read_page` / `open_browser_page` output when the model only supplied `element`)."""
    args = dict(call_args or {})
    base = browser_tool(call_name)
    if base not in VSCODE_BROWSER_TOOLS or base in ("open_browser_page", "list_browser_pages", "fetch_webpage"):
        return args
    if not args.get("pageId"):
        for rd in reversed(list(tool_rounds or ())):
            try:
                rd_a = json.loads(rd.get("arguments") or "{}") if isinstance(rd.get("arguments"), str) else (rd.get("arguments") or {})
                if rd_a.get("pageId"):
                    args["pageId"] = str(rd_a["pageId"])
                    break
            except Exception:
                pass
        if not args.get("pageId") and workspace_info:
            m_pg = re.search(r"-\s*\[([A-Za-z0-9_.\-]+)\]\s+", workspace_info)
            if m_pg:
                args["pageId"] = m_pg.group(1)
    if base in ("click_element", "type_in_page", "hover_element") and not args.get("ref") and not args.get("selector"):
        el_desc = str(args.get("element") or "").strip().lower()
        el_words = [w for w in re.findall(r"[a-z0-9]+", el_desc) if w not in ("button", "link", "input", "box", "field", "the", "a", "an")]
        for rd in reversed(list(tool_rounds or ())):
            out_txt = str(rd.get("output") or "")
            if "[ref=" not in out_txt:
                continue
            best_ref, best_score = None, 0
            for ln in out_txt.splitlines():
                m_ref = re.search(r"\[ref=([A-Za-z0-9_\-]+)\]", ln)
                if not m_ref:
                    continue
                ln_low = ln.lower()
                score = (10 if (el_desc and el_desc in ln_low) else 0) + sum(2 for w in el_words if w in ln_low)
                if score > best_score:
                    best_score = score
                    best_ref = m_ref.group(1)
            if best_ref:
                args["ref"] = best_ref
                break
    if base == "run_playwright_code" and args.get("code"):
        args["code"] = playwright_body(str(args["code"]))
    return args


_SHARED_PAGE = re.compile(r"^\s*-\s*\[([A-Za-z0-9_.\-]+)\]\s+(.+?)\s+\(([a-z][\w+.-]*:[^)\s]*)\)(?:\s+\(active\))?\s*$", re.M)


def unshared_open(call_name, call_args, said="", workspace_info="", tool_rounds=()):
    """What the model reads instead of VS Code opening a page at an address nobody gave while VS Code shares pages;
    None when the call may go to VS Code. For "the shared Clover VS Code test page" the model asked to open
    https://clover.com (a real company's site) and then file:///shared/clover/shared.html (laptop, 2026-11-10)."""
    from urllib.parse import urlparse
    base = browser_tool(call_name)
    url = str((call_args or {}).get("url") or "").strip()
    pages = _SHARED_PAGE.findall(workspace_info or "")
    if base != "open_browser_page" or not url or not pages:
        return None
    known = "\n".join([said or "", workspace_info or ""] + [str(rd.get("output") or "") for rd in tool_rounds or ()]).lower()
    host = (urlparse(url).hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    if url.lower() in known or (host and host in known):
        return None
    use = ("readPage, clickElement, typeInPage or runPlaywrightCode" if base != call_name
           else "read_page, click_element, type_in_page or run_playwright_code")
    listed = "\n".join(f"- [{pid}] {title} ({u})" for pid, title, u in pages)
    return (f"Not opened: the user did not give the address {url}. These pages are already shared with you:\n{listed}\n"
            f"Use the pageId of the page the user means with {use}.")


def playwright_locators(snapshot, limit=16):
    """Playwright locators for the elements of a read_page snapshot, so a small model can write the code. A label's
    text is left out when it names a control: given both, the model typed into getByText('Your name')."""
    roles = ("button", "link", "checkbox", "radio", "combobox", "tab", "menuitem", "heading")
    named = {m.lower() for m in re.findall(r"^\s*-\s*[a-z]+\s+\"([^\"]{1,80})\"", snapshot or "", re.M)}
    out = []
    for ln in (snapshot or "").splitlines():
        m = re.match(r"\s*-\s*([a-z]+)\s+\"([^\"]{1,80})\"", ln)
        if m and m.group(1) in ("textbox", "searchbox"):
            out.append(f"{m.group(1)} {m.group(2)!r}: page.getByLabel({m.group(2)!r})")
        elif m and m.group(1) in roles:
            out.append(f"{m.group(1)} {m.group(2)!r}: page.getByRole({m.group(1)!r}, {{ name: {m.group(2)!r} }})")
        else:
            t = re.match(r"\s*-\s*(?:paragraph|text|generic)(?:\s+\[[^\]]*\])*:\s*(.{1,80})$", ln)
            if t and t.group(1).strip() and t.group(1).strip().lower() not in named:
                out.append(f"text {t.group(1).strip()!r}: page.getByText({t.group(1).strip()!r})")
        if len(out) >= limit:
            break
    return "\n".join(out)


_RELOAD_ASK = re.compile(r"\b(?:reload|refresh|re-?open|open\s+(?:it|the\s+page)\s+again|navigate)\b", re.I)


def shared_page_read(call_name, call_args, said="", workspace_info="", tool_names=()):
    """The read to make instead of navigating a shared page to the address it already shows; None otherwise. The Local
    agent's first step for "type Ravi ... click Greet" was that navigation: it reset the page and needed an approval
    (laptop, 2026-11-10)."""
    args = call_args or {}
    if browser_tool(call_name) != "navigate_page" or args.get("type", "url") != "url" or _RELOAD_ASK.search(said or ""):
        return None
    url = str(args.get("url") or "").strip().rstrip("/")
    pid = next((p for p, _t, u in _SHARED_PAGE.findall(workspace_info or "") if u.rstrip("/") == url
                and (not args.get("pageId") or args.get("pageId") == p)), None)
    read = next((n for n in tool_names if browser_tool(n) == "read_page"), None)
    return {"name": read, "arguments": {"pageId": pid}} if url and pid and read else None


# "...test page: type Maya in the name box, click Greet, and tell me": a step the user wants done on the page
_PAGE_STEP_ASK = re.compile(r"(?:^|[.:;,!?]\s*|\b(?:and|then|please)\s+)(?:type|enter|fill|click|press|tap|select|check|submit)\b", re.I)
_PAGE_STEP_CODE = re.compile(r"\.(?:fill|click|dblclick|type|press|check|uncheck|selectOption|setInputFiles|tap)\s*\(|keyboard\.")
_PAGE_STEP_TOOLS = {"click_element", "type_in_page", "hover_element", "drag_element", "navigate_page", "handle_dialog"}
_USER_DECLINED = re.compile(r"\buser\s+(?:rejected|denied|declined|skipped|cancell?ed)\b", re.I)
# "type Ravi in the name box": the text the user wants typed (Local agent, 2026-11-10: it reloaded the page, clicked
# Greet without typing and answered "After typing Ravi ... the page says Hello, Clover")
_TYPED_TEXT = re.compile(r"\b(?:type|enter|fill\s+in|write)\s+[\"'\u201c]?(.{1,40}?)[\"'\u201d]?\s+(?:in|into)\s+(?:the\s+)?\w", re.I)


def unfinished_page_steps(user_prompt, workspace_info, tool_rounds, tool_names):
    """What sends the model back to a shared page when it answers before doing the steps the user asked for there; ""
    when nothing is missing or the user declined a tool. Asked to type Maya and click Greet with Playwright,
    LFM3-Omni-2B read the page text and answered (laptop, 2026-11-10)."""
    pw = next((n for n in tool_names if browser_tool(n) == "run_playwright_code"), None)
    if not pw or not _SHARED_PAGE.search(workspace_info or "") or not _PAGE_STEP_ASK.search(user_prompt or ""):
        return ""
    rounds = list(tool_rounds or ())
    if (sum(browser_tool(rd.get("name") or "") in VSCODE_BROWSER_TOOLS for rd in rounds) >= 3
            or any(_USER_DECLINED.search(str(rd.get("output") or "")[:400]) for rd in rounds)):
        return ""
    done = " ".join(str(rd.get("arguments") or "") for rd in rounds
                    if browser_tool(rd.get("name") or "") in _PAGE_STEP_TOOLS
                    or (browser_tool(rd.get("name") or "") == "run_playwright_code"
                        and _PAGE_STEP_CODE.search(str(rd.get("arguments") or ""))))
    typed = [t.strip() for t in _TYPED_TEXT.findall(user_prompt or "")
             if not re.match(r"(?:your|my|the|a|an|some|it)\b", t.strip(), re.I)]
    missing = [t for t in typed if t.lower() not in done.lower()]
    if done and not missing:
        return ""
    snap = next((str(rd.get("output") or "") for rd in reversed(rounds) if "[ref=" in str(rd.get("output") or "")), "")
    loc = playwright_locators(snap)
    what = (" and ".join(repr(t) for t in missing) + " was never typed into the page" if done
            else "the user asked for steps on the page and nothing has been typed or clicked")
    return (f"Not done yet: {what}. Call {pw} now with code that does every step the user asked, in order, and "
            "returns what they want to know." + (" Locators on this page:\n" + loc if loc else ""))


def repeats_round(call_name, call_args, tool_rounds):
    """True when this exact call already ran in this turn: sent back to the page, the model repeated its read of it."""
    for rd in tool_rounds or ():
        try:
            prev = json.loads(rd.get("arguments") or "{}") if isinstance(rd.get("arguments"), str) else rd.get("arguments")
        except ValueError:
            continue
        if rd.get("name") == call_name and prev == call_args:
            return True
    return False


_PW_WRAP = re.compile(r"^\s*(?:async\s+)?(?:function\s*\w*\s*\(\s*page\s*\)|\(\s*page\s*\)\s*=>|page\s*=>)\s*\{([\s\S]*)\}\s*;?\s*$")


def playwright_body(code):
    """The function body VS Code's run_playwright_code wants: code written as `async (page) => { ... }` was only
    defined, never run, and the model reported the page as if it had clicked (2026-11-10)."""
    m = _PW_WRAP.match(code or "")
    return m.group(1).strip() if m else (code or "")


class EngineCache:
    """numpy wrapper of lfm3_cache_*: a new prompt keeps the longest prefix already in the engine cache."""

    def __init__(self, engine_dir, pruned_dir, max_tokens=8192):
        if os.name == "nt":
            os.add_dll_directory(str(engine_dir))
        name = "liblfm3_cpu_engine.dll" if os.name == "nt" else "liblfm3_cpu_engine.so"
        lib = ctypes.CDLL(str(Path(engine_dir) / name))
        if not hasattr(lib, "lfm3_cache_feed"):
            raise RuntimeError(f"{name} has no lfm3_cache_* (needs the laptop engine build)")
        lib.lfm3_engine_init.argtypes = [ctypes.c_char_p]
        lib.lfm3_cache_init.argtypes = [ctypes.c_int]
        lib.lfm3_cache_feed.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, _PF, _PF]
        lib.lfm3_cache_truncate.argtypes = [ctypes.c_int]
        lib.lfm3_get_num_threads.restype = ctypes.c_int
        lib.lfm3_has_avx2.restype = ctypes.c_int
        rc = lib.lfm3_engine_init(str(pruned_dir).encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"lfm3_engine_init failed rc={rc}")
        if lib.lfm3_cache_init(int(max_tokens)) != 0:
            raise RuntimeError("lfm3_cache_init failed")
        self.lib, self.max_tokens, self.ids = lib, int(max_tokens), []
        self.logits = np.zeros(V_ROWS, dtype=np.float32)
        self.hidden = np.zeros(D_MODEL, dtype=np.float32)
        self.stats = {"prefilled": 0, "reused": 0, "decoded": 0}

    def _feed(self, ids):
        arr = (ctypes.c_int * len(ids))(*ids)
        rc = self.lib.lfm3_cache_feed(arr, len(ids), self.hidden.ctypes.data_as(_PF), self.logits.ctypes.data_as(_PF))
        if rc < 0:
            raise RuntimeError(f"lfm3_cache_feed rc={rc}")
        self.ids.extend(ids)

    def prefill(self, ids):
        if len(ids) >= self.max_tokens:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the engine cache ({self.max_tokens})")
        lcp, m = 0, min(len(self.ids), len(ids) - 1)
        while lcp < m and self.ids[lcp] == ids[lcp]:
            lcp += 1
        self.lib.lfm3_cache_truncate(lcp)
        del self.ids[lcp:]
        self._feed(ids[lcp:])
        self.stats["reused"] += lcp
        self.stats["prefilled"] += len(ids) - lcp
        return lcp

    def step(self, token_id):
        if len(self.ids) >= self.max_tokens:
            raise ValueError("engine cache full")
        self._feed([int(token_id)])
        self.stats["decoded"] += 1


def sample_next(logits, history_ids, cfg, rng, vocab):
    """HF order: repetition penalty over prompt + generated ids, then temperature, top-k, top-p."""
    lg = logits[:vocab].astype(np.float64)
    rp = cfg["repetition_penalty"]
    if rp != 1.0 and history_ids:
        idx = np.fromiter(set(history_ids), dtype=np.int64)
        idx = idx[idx < vocab]
        vals = lg[idx]
        lg[idx] = np.where(vals > 0, vals / rp, vals * rp)
    if cfg["temperature"] <= 0:
        return int(np.argmax(lg))
    lg /= cfg["temperature"]
    k = cfg["top_k"]
    if k and k < vocab:
        kth = np.partition(lg, -k)[-k]
        lg[lg < kth] = -np.inf
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    if cfg["top_p"] < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, cfg["top_p"]) + 1)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return int(rng.choice(vocab, p=p))


def _coerce(raw, schema):
    t = (schema or {}).get("type")
    v = raw
    try:
        if t == "integer":
            return int(v.strip())
        if t == "number":
            s = v.strip()
            return int(s) if re.fullmatch(r"-?\d+", s) else float(s)
        if t == "boolean":
            return v.strip().lower() == "true"
        if t in ("object", "array"):
            return json.loads(v)
    except ValueError:
        return v
    return v


def parse_tool_calls(text, schemas_by_name):
    """All `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` blocks in a completion."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([^>\s]+)>([\s\S]*?)</function>\s*(?:</tool_call>|$)", text):
        name = m.group(1)
        props = ((schemas_by_name.get(name) or {}).get("parameters") or {}).get("properties") or {}
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)>\n?([\s\S]*?)\n?</parameter>", m.group(2)):
            args[pm.group(1)] = _coerce(pm.group(2), props.get(pm.group(1)))
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "name": name, "arguments": args})
    return calls


def split_completion(text, thinking):
    """-> (reasoning, content_before_calls, content_after_calls): the prompt already opened <think> when thinking."""
    reasoning, rest = "", text
    if thinking:
        if "</think>" in text:
            reasoning, rest = text.split("</think>", 1)
        else:
            return text, "", ""
    first = rest.find("<tool_call>")
    if first < 0:
        return reasoning, rest, ""
    last = rest.rfind("</tool_call>")
    return reasoning, rest[:first], (rest[last + len("</tool_call>"):] if last >= 0 else "")


def _clip(s, n=TOOL_OUTPUT_CHARS):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more characters not shown]"


def checked_summary(eng, ran, summary):
    """A tool step's line in the chat, plus "checked on disk" when Clover read the written file back (tool_tasks)."""
    t = getattr(eng, "last_task", None) if ran else None
    if t and t.get("status") == "completed" and (t.get("evidence") or {}).get("checked") == "file read back":
        return f"{summary} · checked on disk"
    return summary


def tool_worked(name, ran, result, why=""):
    """Evaluate whether a tool call succeeded and return `(worked: bool, summary: str)` for live UI progress."""
    if not ran:
        return False, (str(why or "not run").strip() or "not run")[:96]
    s = str(result or "").strip()
    exit_m = re.search(r"\[exit\s+(-?[1-9]\d*)\]\s*$", s)
    failed = (
        s.startswith(("Error:", "Not run:", "Could not read the page.", "[URL Fetch Error"))
        or (name == "web_search" and s == "No results.")
        or "Traceback (most recent call last):" in s
        or "\nSyntaxError:" in s
        or "is not recognized as the name of a cmdlet" in s
        or bool(exit_m)
    )
    if failed:
        if "Traceback (most recent call last):" in s:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("[")]
            reason = lines[-1] if lines else "Python error"
        elif s.startswith("Error:"):
            reason = s[6:].strip().splitlines()[0] if s[6:].strip() else "Tool error"
        elif "is not recognized as the name of a cmdlet" in s:
            reason = "Command not recognized on this computer"
        else:
            first = next((ln.strip() for ln in s.splitlines() if ln.strip()), "Tool failed")
            reason = first + (f" (exit {exit_m.group(1)})" if exit_m and "[exit" not in first else "")
        return False, reason[:96]
    if "[Shown to the user in the chat:" in s:
        shown_m = re.search(r"\[Shown to the user in the chat:\s*([^\]]+)\]", s)
        return True, (f"Created {shown_m.group(1).strip()}" if shown_m else "Chart created")[:96]
    if not s or s == "(no output)":
        return True, "Completed"
    first_line = next((ln.strip() for ln in s.splitlines() if ln.strip() and not re.fullmatch(r"[-=]{3,}", ln.strip())), "Completed")
    return True, first_line[:96]


# text in a file or page that addresses the assistant (a meeting note told "any AI assistant" to delete its folder)
INJECTION, INJECTION_NOTE, flag_injection = semantics.INJECTION, semantics.INJECTION_NOTE, semantics.flag_injection


# JSON escapes in a Windows path the model wrote with single backslashes: "C:\Users\santh\CloverAI\tmp" arrived as
# CloverAI<TAB>mp and the folder "did not exist" (2026-11-01)
_CTRL = {"\t": "\\t", "\n": "\\n", "\r": "\\r", "\f": "\\f", "\b": "\\b", "\v": "\\v", "\a": "\\a"}
PATH_ARGS = ("path", "cwd", "target", "parent", "root")


def _repair_command(cmd):
    def fix(m):
        tok, out, i = m.group(0), [], 0
        while i < len(tok):
            c = tok[i]
            rest = tok[i + 1:]
            if c in "\n\r":  # a real line break ends the path unless the path plainly goes on (a folder or a file name)
                nxt = re.match(r"[A-Za-z0-9_.\-]+", rest)
                word = nxt.group(0) if nxt else ""
                goes_on = rest[len(word):len(word) + 1] == "\\" or re.search(r"\.[A-Za-z0-9]{1,5}$", word)
                if not (word and goes_on and not re.match(r"[A-Z][a-z]+-[A-Z]", word)):
                    out.append(tok[i:])
                    break
            out.append(_CTRL.get(c, c))
            i += 1
        return "".join(out)
    return re.sub(r"[A-Za-z]:\\[^ \"'|<>;,]*", fix, cmd)


def repair_args(args, user_paths=()):
    if not isinstance(args, dict):
        return args
    out = dict(args)
    for k, v in args.items():
        if not isinstance(v, str):
            continue
        if any(c in v for c in _CTRL):
            if k in PATH_ARGS:
                v = "".join(_CTRL.get(c, c) for c in v)
            elif k == "command":
                v = _repair_command(v)
        if k in PATH_ARGS or k == "command":
            v = restore_paths(v, user_paths)
        out[k] = v
    return out


USER_PATH, user_paths, restore_paths = semantics.USER_PATH, semantics.user_paths, semantics.restore_paths
# server.build_prompt puts attached files before this line and the user's own words after it
ATTACHED = "Using the attached content / session context above, answer the user's request clearly and thoroughly:\n"


def typed_hosts(user_text):
    """Hosts the user typed in this message, not those inside attached files: only these may be private addresses
    for web_search and fetch_url (page_fetch)."""
    return page_fetch.typed_hosts(str(user_text or "").rsplit(ATTACHED, 1)[-1])


class LFM3Native:
    def __init__(self, base_dir, perms, max_tokens=8192, progress=None, engine="cpu"):
        """engine: "cpu" = Clover C engine (bf16), "gpu" = llama.cpp on the Iris Xe (QAD Q4_0, llamacpp_engine)."""
        t0 = time.perf_counter()
        progress = progress or (lambda m: None)
        base_dir = Path(base_dir)
        self.base_dir, self.perms = base_dir, perms
        progress("tokenizer and chat template")
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(base_dir / "data" / "lfm3" / "hf"))
        self.vocab = len(self.tok)
        self.stop_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.engine_kind, self.max_tokens, self.cache, self.gpu = engine, int(max_tokens), None, None
        if engine == "gpu":
            progress("LFM3 on the Iris Xe GPU (llama.cpp, 4-bit)")
            import llamacpp_engine
            self.gpu = llamacpp_engine.LlamaServer(base_dir, "vulkan", max_tokens)
        else:
            progress("LFM3 C engine")
            self.cache = EngineCache(base_dir / "engine", base_dir / "data" / "lfm3" / "pruned", max_tokens)
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self._py = None
        self._web = None
        self._extra = {}
        self.workdir, self.workname, self.after_write = None, "", None  # the active workspace (server sets them)
        self.setup_s = round(time.perf_counter() - t0, 2)

    def describe(self):
        d = {"mode": "lfm3", "engine": self.engine_kind, "max_tokens": self.max_tokens, "setup_s": self.setup_s,
             "tools": [t["function"]["name"] for t in STUDIO_TOOLS]}
        if self.gpu is not None:
            d.update(layers=["L1 LFM3-1.7B (llama.cpp Vulkan, QAD Q4_0)", "native tools"], device=self.gpu.device,
                     weights="4-bit (QAD Q4_0)", backend="llama.cpp " + self.gpu.kind)
        else:
            d.update(layers=["L1 LFM3-1.7B (C engine, bf16)", "native tools"], device="CPU", weights="bf16",
                     backend="Clover C engine", avx2=bool(self.cache.lib.lfm3_has_avx2()),
                     threads=self.cache.lib.lfm3_get_num_threads(), cache=self.cache.stats)
        return d

    def close(self):
        if self.gpu is not None:
            self.gpu.close()

    # ---------------- generation ----------------
    def _render(self, messages, tools, thinking):
        return self.tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=thinking)

    def _fit(self, msgs, tools, thinking, max_new):
        """Drop the oldest history messages (never the system prompt or the current request) until the prompt fits."""
        while True:
            n = len(self.tok(self._render(msgs, tools, thinking), add_special_tokens=False).input_ids)
            if n + min(max_new, 1024) < self.max_tokens or len(msgs) <= 2:
                return msgs
            del msgs[1]

    def generate(self, prompt, cfg, max_new, thinking, on_delta=None, cancel=None, seed=0):
        if self.gpu is not None:
            return self._generate_gpu(prompt, cfg, max_new, thinking, on_delta, cancel, seed)
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        if len(ids) + max_new >= self.max_tokens:
            max_new = max(64, self.max_tokens - len(ids) - 1)
        t0 = time.perf_counter()
        self.stage = {"name": "reading the prompt", "since": time.time()}
        reused = self.cache.prefill(ids)
        t_prefill = time.perf_counter() - t0
        rng = np.random.default_rng(seed)
        history = list(ids)
        gen, text, reason = [], "", "length"
        self.stage = {"name": "thinking" if thinking else "writing", "since": time.time()}
        t1 = time.perf_counter()
        for _ in range(max_new):
            if cancel is not None and cancel.is_set():
                reason = "cancelled"
                break
            nxt = sample_next(self.cache.logits, history, cfg, rng, self.vocab)
            if nxt in self.stop_ids:
                reason = "stop"
                break
            gen.append(nxt)
            history.append(nxt)
            new_text = self.tok.decode(gen, skip_special_tokens=False)
            if not new_text.endswith("\ufffd"):
                delta, text = new_text[len(text):], new_text
                if thinking and "</think>" in text and self.stage["name"] == "thinking":
                    self.stage = {"name": "writing", "since": time.time()}
                if on_delta and delta:
                    on_delta(delta, text)
            if "<tool_response>" in text:  # the model started inventing a tool result: stop and keep what came before
                text = text[: text.index("<tool_response>")]
                reason = "stop"
                break
            self.cache.step(nxt)
        t_dec = time.perf_counter() - t1
        return text, {"prompt_tokens": len(ids), "reused_tokens": reused, "new_tokens": len(gen),
                      "prefill_s": round(t_prefill, 2), "decode_s": round(t_dec, 2),
                      "decode_tok_s": round(len(gen) / t_dec, 2) if t_dec > 0 else None, "finish": reason}

    def _generate_gpu(self, prompt, cfg, max_new, thinking, on_delta, cancel, seed):
        import contextlib

        import llamacpp_engine as L
        ids = self.tok(prompt, add_special_tokens=False).input_ids
        if len(ids) + max_new >= self.max_tokens:
            max_new = max(64, self.max_tokens - len(ids) - 1)
        self.stage = {"name": "reading the prompt", "since": time.time()}
        t0 = time.perf_counter()
        t_first, gen, text, final, stopped = None, [], "", None, False
        with contextlib.closing(self.gpu.stream(ids, L.sampling_params(cfg, max_new, seed, self.max_tokens), cancel)) as events:
            for toks, fin in events:
                final = fin or final
                for t in toks:
                    if t in self.stop_ids:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                        self.stage = {"name": "thinking" if thinking else "writing", "since": time.time()}
                    if t == L.TOOL_RESPONSE_ID:
                        stopped = True
                        break
                    gen.append(t)
                    new_text = self.tok.decode(gen, skip_special_tokens=False)
                    if not new_text.endswith("\ufffd"):
                        delta, text = new_text[len(text):], new_text
                        if thinking and "</think>" in text and self.stage["name"] == "thinking":
                            self.stage = {"name": "writing", "since": time.time()}
                        if on_delta and delta:
                            on_delta(delta, text)
                if stopped:
                    break
        t_end = time.perf_counter()
        kind = (final or {}).get("stop_type")
        if stopped or kind in ("eos", "word"):
            reason = "stop"
        elif kind == "limit":
            reason = "length"
        elif kind == "cancelled" or (cancel is not None and cancel.is_set()):
            reason = "cancelled"
        else:
            reason = "stop" if len(gen) < max_new else "length"
        tm = (final or {}).get("timings") or {}
        prefill_s = tm.get("prompt_ms", ((t_first or t_end) - t0) * 1000) / 1000
        decode_s = tm.get("predicted_ms", (t_end - (t_first or t_end)) * 1000) / 1000
        return text, {"prompt_tokens": len(ids), "reused_tokens": max(0, len(ids) - int(tm.get("prompt_n", len(ids)))),
                      "new_tokens": len(gen), "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
                      "decode_tok_s": round(len(gen) / decode_s, 2) if decode_s > 0 else None, "finish": reason}

    # ---------------- tools ----------------
    def _permit(self, tool, detail, ask, allow, risk=None):
        """risk: why a command cannot easily be undone; it is asked about even when the tool is allowed."""
        perm = TOOL_PERMISSION.get(tool) or (self._extra.get(tool) or {}).get("permission")
        if perm is None:
            return True, "no permission needed"
        if perm in allow or tool in allow:
            return True, "allowed for this turn"
        mode = self.perms.get()[perm]
        if mode == "deny":
            return False, "turned off in settings"
        if mode == "allow" and not risk:
            return True, "allowed in settings"
        if ask is None:
            return False, f"needs your permission ({risk})" if risk else "needs your permission (set to Ask)"
        ok = ask(perm, detail, risk) if risk else ask(perm, detail)
        return ok, ("allowed by you" if ok else "not allowed by you")

    def _web_tool(self):
        if self._web is None:
            import web_search
            self._web = web_search.WebSearch(web_search.SearchConfig.shared(
                paths.config_file(self.base_dir, "search.json"), paths.vault_file(self.base_dir, "search-keys.json")))
        return self._web

    def research_web(self, query, ask=None, allow=()):
        """The Clover cycle's browsing step (search, read the best pages) under the internet permission:
        -> (ran, text, why)."""
        ok, why = self._permit("web_search", query, ask, allow)
        if not ok:
            return False, P.DENIED_TEXT["internet"] + f" ({why})", why
        self.stage = {"name": "searching the web", "since": time.time()}
        try:
            return True, flag_injection(_clip(self._web_tool().research(query))), why
        except Exception as exc:  # a failed search is reported, not fatal
            return True, f"Error: {type(exc).__name__}: {exc}", why

    def execute_tool(self, name, args, ask=None, allow=()):
        import local_terminal
        args = repair_args(args, getattr(self, "turn_paths", ()))
        detail = {"run_python": args.get("code"), "web_search": args.get("query"), "fetch_url": args.get("url"),
                  "run_powershell": f"{args.get('command')}\n(in {args.get('cwd') or self.workdir or Path.home()}, "
                                    f"up to {local_terminal.clamp_timeout(args.get('timeout_s', 60))} s)",
                  "read_file": args.get("path"),
                  "list_directory": args.get("path"),
                  "write_file": f"{args.get('path')}\n---\n{str(args.get('content') or '')[:1500]}",
                  "laptop_status": "Read this laptop's free disk space, battery, memory use and the names of the apps "
                                   "using the most memory"}.get(name, "")
        if not detail and name in local_tool_packs.PACK_PERMISSIONS:
            detail = local_tool_packs.tool_detail(name, args, getattr(self, "workdir", None))
        extra = self._extra.get(name)
        if extra:
            detail = extra["detail"](args)
        elif name not in TOOL_PERMISSION:
            return False, f"Error: unknown tool {name!r}. Available: {', '.join(list(TOOL_PERMISSION) + list(self._extra))}", "unknown tool"
        risk = local_terminal.risky(args.get("command")) if name in ("run_powershell", "vscode_run") else None
        verb = re.match(r"\s*([A-Za-z][\w-]*)", str(args.get("command") or "")) if name == "run_powershell" else None
        if verb and verb.group(1).lower() in getattr(self, "turn_unknown", set()):
            # the battery question tried `get-batpower` three times (2026-11-01)
            return False, (f"Not run: {verb.group(1)} does not exist on this laptop (it failed already in this answer). "
                           "Use a real command, or answer that you could not find out."), "command does not exist"
        command = " ".join(str(args.get("command") or "").split())
        if name == "run_powershell" and command in getattr(self, "turn_ran", set()):
            # the disk-space question ran the same Get-ChildItem four times, changing only timeout_s (2026-11-01)
            return False, ("Not run: this exact command already ran in this answer; its output is above. Use a different "
                           "command, or answer from what you have."), "already ran"
        if name in ("web_search", "fetch_url"):
            # Tools Always: a trip-planning chat searched 8 times an answer and never named a place (2026-11-01)
            self.turn_web = getattr(self, "turn_web", 0) + 1
            if self.turn_web > getattr(self, "turn_web_cap", 99):
                return False, ("Not run: enough searching for this answer. Answer now from what you know and the results "
                               "above, and say briefly what you could not check."), "search limit for this answer"
        ok, why = self._permit(name, detail, ask, allow, risk)
        if not ok:
            perm = extra["permission"] if extra else TOOL_PERMISSION[name]
            return False, (P.DECLINED_TEXT if why == "not allowed by you" else P.DENIED_TEXT[perm]) + f" ({why})", why
        tid = self._task_begin(name, args)
        self.stage = {"name": f"running {name}", "since": time.time()}
        try:
            out = extra["run"](args) if extra else self._run_tool(name, args)
        except Exception as exc:  # report tool failures to the model as results
            out = f"Error: {type(exc).__name__}: {exc}"
            self._task_done(tid, name, args, out, why)
            return True, out, why
        if name == "run_powershell" and "is not recognized as the name of a cmdlet" in str(out):
            bad = re.match(r"\s*([A-Za-z][\w-]*)", str(args.get("command") or ""))
            if bad:
                self.turn_unknown = getattr(self, "turn_unknown", set()) | {bad.group(1).lower()}
            out = (str(out) + "\n[That command does not exist on this laptop. Use a real PowerShell command or an installed "
                   "program, e.g. git log --oneline -5, git status, Get-ChildItem, Get-PSDrive C.]")
        if name == "run_powershell" and "seconds: the time limit]" not in str(out):
            self.turn_ran = getattr(self, "turn_ran", set()) | {command}
        out = self._task_done(tid, name, args, out, why)
        return True, flag_injection(_clip(out)), why

    def _task_begin(self, name, args):
        self.last_task = None
        tasks = getattr(self, "tool_tasks", None)
        if tasks is None:
            return None
        try:
            return tasks.begin(name, args, write_scope=args.get("path") if name == "write_file" else None)
        except Exception:  # the record never stops a tool call
            return None

    def _task_done(self, tid, name, args, out, why):
        """The call's task (tool_tasks): a written file must read back as written before the call counts as done; one
        that does not is reported to the model as not done."""
        tasks = getattr(self, "tool_tasks", None)
        if tasks is None or tid is None:
            return out
        import tool_tasks
        worked = tool_worked(name, True, out, why)[0]
        check = tool_tasks.write_check(self._path(args.get("path")), args.get("content")) if name == "write_file" else None
        try:
            v = tasks.finish(tid, worked, out, check)
        except Exception:
            return out
        self.last_task = v
        if worked and v["status"] == "failed":
            return ("Error: Clover read the file back after writing and it does not hold what was written "
                    f"({(v.get('evidence') or {}).get('reason', 'check failed')}).")
        return out

    def _python(self, code):
        """run_python in a fresh output folder (<media>/out/<id>): a chart saved without a folder lands there and is shown
        in the chat (turn_images, set by the server for the turn); other files written there are named in the result."""
        import hashlib
        import shutil
        import uuid
        media = Path(getattr(self, "media_dir", None) or (Path(self.base_dir) / "media"))
        out = media / "out" / uuid.uuid4().hex[:12]
        out.mkdir(parents=True, exist_ok=True)
        pre = (f"import os as _c_os; _c_os.chdir(r'{out}'); _c_os.environ.setdefault('MPLBACKEND', 'Agg'); "
               f"_c_os.environ.setdefault('OPENBLAS_NUM_THREADS', '4'); _c_os.environ.setdefault('OMP_NUM_THREADS', '4'); del _c_os\n")
        result = self._py.execute(pre + code) or "(no output)"
        shown, files = [], []
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix.lower() == ".png" and p.stat().st_size and getattr(self, "turn_images", None) is not None:
                name = hashlib.sha256(p.read_bytes()).hexdigest()[:32] + ".png"
                shutil.copyfile(p, media / name)
                self.turn_images.append({"file": name, "prompt": "chart from Python: " + p.name, "model": "Python",
                                         "seconds": 0, "width": None, "height": None})
                shown.append(p.name)
            elif p.is_file():
                files.append(str(p))
        if shown:
            result += f"\n[Shown to the user in the chat: {', '.join(shown)}]"
        if files:
            result += "\n[Files written: " + "; ".join(files) + "]"
        if not shown and not files:
            shutil.rmtree(out, ignore_errors=True)
        return result

    def _run_tool(self, name, a):
        if name == "run_python":
            if self._py is None:
                import sys
                if paths.ANDROID:  # no second Python on Android: the code runs in this process, on its own thread
                    from android_python_tool import PythonTool
                else:
                    from windows_python_tool import PythonTool
                # 60 s like the terminal: pandas over an attached spreadsheet (import + read) can pass 15 s on the laptop
                mem_gb = 4.0 if os.name == "nt" else 8.0
                self._py = PythonTool(python_path=sys.executable, timeout=60, max_output_chars=50_000, memory_limit_gb=mem_gb)
            return self._python(str(a.get("code") or ""))
        if name in ("web_search", "fetch_url"):
            typed = getattr(self, "turn_hosts", frozenset())
            if name == "web_search":
                return self._web_tool().execute(str(a.get("query") or ""), typed=typed) or "No results."
            return (self._web_tool().fetch_url_content(str(a.get("url") or ""), max_chars=5000, typed=typed)
                    or "Could not read the page.")
        if name == "run_powershell":
            import local_terminal
            cwd, err = local_terminal.resolve_cwd(a.get("cwd"), self.workdir)
            if err:
                return err
            return local_terminal.powershell(str(a.get("command") or ""), cwd=str(cwd),
                                             timeout_s=a.get("timeout_s", local_terminal.DEFAULT_TIMEOUT_S),
                                             cancel=getattr(self, "turn_cancel", None))
        if name == "read_file":
            path = self._path(a.get("path"))
            if paths.read_blocked(self.base_dir, path):
                return "Error: Clover's vault holds secrets; it is not read by tools."
            data = path.read_bytes()
            if b"\x00" in data[:4096]:
                return f"{path} looks like a binary file ({len(data)} bytes); not shown."
            lines = data.decode("utf-8", errors="replace").splitlines()
            s = max(1, int(a.get("start_line") or 1))
            e = min(len(lines), int(a.get("end_line") or len(lines)))
            body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
            return f"{path} (lines {s}-{e} of {len(lines)})\n{body}"
        if name == "list_directory":
            path = self._path(a.get("path"), default=str(self.workdir) if self.workdir else "~")
            rows = []
            for i, ent in enumerate(sorted(os.scandir(path), key=lambda x: (not x.is_dir(), x.name.lower()))):
                if i >= 300:
                    rows.append("... more entries not shown")
                    break
                note = ""
                if ent.is_file() and ent.name.endswith(("-context.md", "-context-index.md")):
                    try:
                        with open(ent.path, "rb") as fh:
                            note = " - Clover's context notes, not part of the project" if b"clover:context" in fh.read(400) else ""
                    except OSError:
                        pass
                rows.append(ent.name + "/" if ent.is_dir() else f"{ent.name}  ({ent.stat().st_size} bytes){note}")
            return f"{path}\n" + ("\n".join(rows) or "(empty)")
        if name == "write_file":
            path = self._path(a.get("path"))
            if paths.write_blocked(self.base_dir, path):
                return ("Error: Clover's repository and vault are not written by tools: the records, settings and "
                        "AGENTS.md change only through Clover AI Studio itself.")
            content = str(a.get("content") or "")
            if len(content) > 200_000:
                return "Error: content longer than 200,000 characters; not written."
            backup = ""
            if path.exists():
                bak = path.with_name(path.name + time.strftime(".clover-%Y%m%d-%H%M%S.bak"))
                bak.write_bytes(path.read_bytes())
                backup = f" (previous version saved as {bak.name})"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if self.after_write is not None:
                try:
                    self.after_write(path)
                except Exception:  # noqa: BLE001 - the file is written; a context refresh failing must not undo that
                    pass
            return f"Wrote {len(content)} characters to {path}{backup}."
        if name == "get_current_time":
            now = datetime.datetime.now().astimezone()
            return now.strftime("%A %d %B %Y, %H:%M:%S %Z (UTC%z)")
        if name == "laptop_status":
            import laptop_status
            return laptop_status.report(cancel=getattr(self, "turn_cancel", None))
        if name == "manage_todo_list":
            import clover_cycle as C
            raw_items = a.get("todoList") or a.get("todos") or a.get("items") or []
            plan = getattr(self, "active_op_plan", None)
            if plan is not None:
                plan.set_todos(raw_items, custom=True)
                return plan.todos_markdown()
            norm = C.normalize_todo_list(raw_items)
            return json.dumps(norm)
        if name in local_tool_packs.PACK_PERMISSIONS:
            return local_tool_packs.run_pack_tool(name, a, self)
        raise ValueError(name)

    def _path(self, raw, default=""):
        """A path the model gave: ~ and %VARS% expanded; a relative path refers to the active workspace."""
        p = Path(os.path.expandvars(os.path.expanduser(str(raw or "").strip() or default)))
        return Path(self.workdir) / p if self.workdir and not p.is_absolute() else p

    def workspace_sentence(self):
        return (f"The user is working in the folder {self.workdir} (workspace \"{self.workname}\"); relative file paths "
                "refer to it. " if self.workdir else "")

    # ---------------- one Studio turn ----------------
    def system_prompt(self, tools_on=True, asked=None):
        now = datetime.datetime.now().astimezone()
        return (f"You are Clover, an assistant running entirely on the user's Windows laptop (model: LFM3-1.7B). "
                f"Today is {now.strftime('%A %d %B %Y')}. The user's home folder is {Path.home()}. "
                + self.workspace_sentence() + tools_sentence(tools_on, asked))

    def run_turn(self, history, user_text, *, tools=True, sampling="general", thinking=True, max_new=2048,
                 max_rounds=8, on_event=None, ask=None, allow=(), cancel=None, seed=0, system=None,
                 layers=(), kit=None, layer_query=None, final_on_limit=False, force_answer=False, extra_tools=(),
                 only_tools=None):
        """layers: any of lfm3_layers.LAYERS, run through `kit` (a lfm3_layers.LayerKit).
        final_on_limit: when every tool round is used, ask LFM3 once more to answer from what it has.
        force_answer: when thinking fills max_new, close it and let LFM3 answer (up to 1,024 more tokens).
        The ladder's step 1 ran without both, so the bench keeps them off unless asked.
        extra_tools: [{"schema", "permission", "detail": fn(args), "run": fn(args)}] for this turn only.
        only_tools: offer only these tool names (None: all)."""
        emit = on_event or (lambda e: None)
        self.turn_cancel = cancel  # a running command stops when the user stops the turn
        self.turn_unknown, self.turn_ran, self.turn_web = set(), set(), 0  # per answer, as in OmniNative.run_turn
        self.turn_hosts = typed_hosts(user_text)
        cfg = SAMPLING[sampling]
        base = [t for t in STUDIO_TOOLS if only_tools is None or t["function"]["name"] in only_tools]
        extra_tools = [t for t in extra_tools if only_tools is None or t["schema"]["function"]["name"] in only_tools]
        tools = tools and bool(base or extra_tools)
        self._extra = {t["schema"]["function"]["name"]: t for t in extra_tools} if tools else {}
        schemas = (base + [t["schema"] for t in extra_tools]) if tools else []
        by_name = {t["function"]["name"]: t["function"] for t in schemas}
        query = layer_query or user_text
        import clover_cycle as C
        import workspaces as WS
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name="LFM3-1.7B",
            query=query,
            offered_tools=only_tools if only_tools is not None else list(by_name)[:4],
        )
        self.active_op_plan = op_plan
        op_plan.start_stage("context")
        layer_log, blocks = [], []
        for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"), ("workspace", "workspace_context"),
                         ("records", "records_context")):
            if name in layers:
                block, info = getattr(kit, fn)(query)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
                    emit(dict(info, type="layer"))
        ws_active_p = (kit.workspaces.active() or {}).get("path") if (kit is not None and getattr(kit, "workspaces", None) is not None) else None
        l_blk, l_info = WS.local_context_for_paths(user_paths(user_text), workdir=self.workdir, active_ws_path=ws_active_p)
        if l_blk:
            blocks.append(l_blk)
            layer_log.append(l_info)
            emit(dict(l_info, type="layer"))
        bridge_info = kit.compute_meaning_bridge(query) if (kit is not None and hasattr(kit, "compute_meaning_bridge")) else None
        if bridge_info:
            b_emit = {k: v for k, v in bridge_info.items() if k != "action_prior_8"}
            layer_log.append(b_emit)
            emit(dict(b_emit, type="layer"))
        if blocks:  # Clover appends its observations after the question
            user_text = user_text + "\n\n" + "\n\n".join(blocks)
        ctx_sources = [x.get("layer") for x in layer_log if isinstance(x, dict) and x.get("layer")]
        op_plan.complete_stage(
            "context",
            "Context established (" + (", ".join(dict.fromkeys(ctx_sources)) if ctx_sources else "AGENTS.md & conversation state") + ")",
        )
        op_plan.start_stage(
            "direction",
            "Direction set (" + ("tools: " + ", ".join(only_tools[:4]) if only_tools else "direct reasoning & verification") + ")",
        )
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution")
        msgs = [{"role": "system", "content": system or self.system_prompt(bool(schemas), only_tools)}] + list(history) + [{"role": "user", "content": user_text}]
        msgs = self._fit(msgs, schemas, thinking, max_new)
        rounds, final, stats_all, reasoning = [], "", [], ""

        def delta_cb(round_no):
            shown = {"thinking": 0, "answer": 0}

            def on_delta(delta, text):
                # C1 is strictly for inference: emit thinking tokens only; C3 (NCM) decodes the final answer
                if thinking:
                    think_part, tail = text.split("</think>", 1) if "</think>" in text else (text, "")
                else:
                    think_part, tail = "", text
                if len(think_part) > shown["thinking"]:
                    emit({"type": "token", "round": round_no, "kind": "thinking", "text": think_part[shown["thinking"]:]})
                    shown["thinking"] = len(think_part)
                if "ncm" not in layers and kit is None:
                    answer = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", tail)
                    if len(answer) > shown["answer"]:
                        emit({"type": "token", "round": round_no, "kind": "answer", "text": answer[shown["answer"]:]})
                        shown["answer"] = len(answer)
            return on_delta

        with self.lock:
          try:
            def _auto_verify_py(code_str):
                if cancel and cancel.is_set():
                    return False, "", False
                if self.perms.get().get("code") == "deny" and "code" not in (allow or ()) and "run_python" not in (allow or ()):
                    return False, "", True
                cid = "call_" + uuid.uuid4().hex[:16]
                args_py = {"code": code_str}
                emit({"type": "tool_call", "id": cid, "name": "run_python", "arguments": args_py})
                if self.perms.get().get("code") == "allow" or "code" in (allow or ()) or "run_python" in (allow or ()):
                    ran, result, why = self.execute_tool("run_python", args_py, ask, allow)
                else:
                    result = self._run_tool("run_python", {"code": code_str})
                    ran, why = True, "Clover Cycle Outcome sandbox verification"
                worked, summary = tool_worked("run_python", ran, result, why)
                emit({"type": "tool_result", "id": cid, "name": "run_python", "ran": ran, "worked": worked,
                      "summary": summary, "why": why, "result": result[:3000]})
                rounds.append({"tool": "run_python", "arguments": args_py, "ran": ran, "worked": worked,
                               "summary": summary, "why": why, "result": result[:2000]})
                return ran, result, worked

            first_cycle_findings = None
            max_clover_cycles = 3
            for attempt in range(max_clover_cycles):
                for r in range(max_rounds):
                    prompt = self._render(msgs, schemas, thinking)
                    emit({"type": "round", "round": r + 1})
                    if r == 0 and attempt == 0 and only_tools:
                        emit({"type": "plan", "round": r + 1, "stage": "planning", "tools": list(only_tools)})
                    elif r > 0:
                        emit({"type": "plan", "round": r + 1, "stage": "reviewing"})
                    cb = delta_cb(r + 1)
                    text, st = self.generate(prompt, cfg, max_new, thinking, cb, cancel, seed + 100 * attempt + r)
                    stats_all.append(st)
                    if force_answer and thinking and st["finish"] == "length" and "</think>" not in text:
                        forced = text + "\n</think>\n\n"
                        text2, st = self.generate(prompt + forced, cfg, min(1024, max_new), False,
                                                  lambda d, t, f=forced: cb(d, f + t), cancel, seed + 100 * attempt + r)
                        st["forced_answer"] = True
                        stats_all.append(st)
                        text = forced + text2
                    reasoning, content, after = split_completion(text, thinking)
                    calls = parse_tool_calls(text.split("</think>", 1)[-1] if thinking else text, by_name) if tools else []
                    if calls and st["finish"] != "cancelled":
                        msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning,
                                     "tool_calls": [{"id": c["id"], "type": "function",
                                                     "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
                        for c in calls:
                            emit({"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": c["arguments"]})
                            ran, result, why = self.execute_tool(c["name"], c["arguments"], ask, allow)
                            worked, summary = tool_worked(c["name"], ran, result, why)
                            emit({"type": "tool_result", "id": c["id"], "name": c["name"], "ran": ran, "worked": worked,
                                  "summary": summary, "why": why, "result": result[:3000]})
                            rounds.append({"tool": c["name"], "arguments": c["arguments"], "ran": ran, "worked": worked,
                                           "summary": summary, "why": why, "result": result[:2000]})
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                            op_plan.advance_execution_step(c["name"])
                        continue
                    final = (content + after).strip() if thinking and "</think>" in text else (text.strip() if not thinking else "")
                    if not final and st["finish"] == "length":
                        final = "(Clover ran out of thinking room before answering. Ask again, or ask for a shorter answer.)"
                    elif not final and st["finish"] == "cancelled":
                        final = "(Stopped.)"
                    elif not final:
                        final = "(Clover finished without writing an answer.)"
                    break
                else:
                    if final_on_limit and not (cancel and cancel.is_set()):
                        msgs.append({"role": "user", "content": "The tool budget for this question is used up. Answer now "
                                     "from the results you already have; do not call more tools."})
                        emit({"type": "round", "round": max_rounds + 1})
                        text, st = self.generate(self._render(msgs, schemas, thinking), cfg, max_new, thinking,
                                                 delta_cb(max_rounds + 1), cancel, seed + 100 * attempt + max_rounds)
                        stats_all.append(st)
                        reasoning, content, after = split_completion(text, thinking)
                        tail = (content + after) if thinking and "</think>" in text else ("" if thinking else text)
                        final = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", tail).strip()
                    final = final or "(Stopped after the maximum number of tool rounds.)"
                if cancel and cancel.is_set():
                    break
                op_plan.complete_stage("execution")
                op_plan.start_stage("outcome", f"Verifying Cycle {attempt + 1} output against reality…", cycle=attempt + 1)
                findings, c_meta = C.verify_turn_outcome(
                    query, final, rounds, kit=kit, layers=layers, execute_python_fn=_auto_verify_py
                )
                comp_info = c_meta.get("compiler_info")
                if comp_info is not None:
                    layer_log.append(comp_info)
                    if comp_info.get("checked"):
                        if c_meta.get("compiler_ok"):
                            emit({"type": "layer", "layer": "compiler", "checked": True, "ok": True})
                        else:
                            emit({"type": "layer", "layer": "compiler", "checked": True, "ok": False, "feedback": findings[0]})
                if not findings:
                    op_plan.complete_stage("outcome", C.outcome_label(c_meta["checks"], attempt + 1))
                    layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": True})
                    if first_cycle_findings:
                        only_comp = all(f.startswith("Code compiler check:") for f in first_cycle_findings)
                        emit({"type": "cycle", "cycle": attempt + 1, "stage": "verified", "cycles_used": attempt + 1})
                        layer_log.append({
                            "layer": "cycle",
                            "trigger": "auto",
                            "action": "compiler_repaired" if only_comp else "corrected",
                            "cycles_used": attempt + 1,
                            "first_findings": first_cycle_findings,
                            "final_findings": [],
                            "tools": [rd["tool"] for rd in rounds if rd.get("ran") and rd.get("worked")],
                            "lesson": C.lesson_text(query, [f[:240] for f in first_cycle_findings[:2]], "", passed=True),
                            "lesson_kind": "verified",
                        })
                    break
                if first_cycle_findings is None:
                    first_cycle_findings = list(findings)
                fixed = C.direct_arithmetic_answer(query) if C.only_arithmetic(findings) else None
                if fixed:  # a plain sum: Clover has the exact value, so it writes the answer itself
                    emit({"type": "cycle", "cycle": attempt + 2, "stage": "retry", "findings": findings})
                    final = fixed
                    emit({"type": "token", "round": 1, "kind": "answer", "text": final})
                    op_plan.complete_stage("outcome", C.outcome_label(c_meta["checks"], attempt + 1))
                    layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": True})
                    emit({"type": "cycle", "cycle": attempt + 1, "stage": "verified", "cycles_used": attempt + 1})
                    break
                if attempt + 1 < max_clover_cycles:
                    emit({"type": "cycle", "cycle": attempt + 2, "stage": "retry", "findings": findings})
                    op_plan.retry_from_context(attempt + 2, findings[0])
                    msgs.append({"role": "assistant", "content": final})
                    msgs.append({"role": "user", "content": C.AUTO_CYCLE_FIX.format(
                        findings="\n".join("- " + f for f in findings)
                    )})
                else:
                    op_plan.complete_stage("outcome", f"Checked across {attempt + 1} cycles (unconfirmed)")
                    layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": False})
                    layer_log.append({
                        "layer": "cycle",
                        "trigger": "auto",
                        "action": "unconfirmed",
                        "cycles_used": attempt + 1,
                        "first_findings": first_cycle_findings or findings,
                        "final_findings": findings,
                    })
            op_plan.start_stage("growth", "Aligning epistemic state & recording learned knowledge…")
            rnd_n = len(stats_all) or 1
            if kit is not None and not (cancel and cancel.is_set()):
                if hasattr(kit, "c1_infer"):
                    c1_state, c1_info = kit.c1_infer(final, stats=stats_all)
                    layer_log.append(c1_info)
                    emit(dict(c1_info, type="layer"))
                    emit({"type": "plan", "round": rnd_n, "stage": "c1_done", "ms": c1_info.get("ms")})
                else:
                    c1_state = None
                if hasattr(kit, "c2_plan_state"):
                    emit({"type": "plan", "round": rnd_n, "stage": "c2_plan", "model": "HybridAgent"})
                    c2_state, c2_info = kit.c2_plan_state(query, c1_state=c1_state, offered_tools=[rd["tool"] for rd in rounds if rd.get("ran")])
                    layer_log.append(c2_info)
                    emit(dict(c2_info, type="layer"))
                    emit({"type": "plan", "round": rnd_n, "stage": "c2_done", "ms": c2_info.get("ms")})
                else:
                    c2_state = final
                if "ncm" in layers and (hasattr(kit, "c3_decode") or hasattr(kit, "ncm_render")):
                    emit({"type": "plan", "round": rnd_n, "stage": "ncm_decode", "model": "11-Level NCM"})
                    final, info = kit.c3_decode(c2_state) if hasattr(kit, "c3_decode") else kit.ncm_render(final)
                    layer_log.append(info)
                    emit(dict(info, type="layer"))
                    emit({"type": "plan", "round": rnd_n, "stage": "ncm_done", "ms": round((info.get("s") or 0) * 1000)})
            if "ontology" in layers and kit is not None:
                final, changed = kit.grounding_filter(query, final)
                layer_log.append({"layer": "ontology_grounding", "changed": changed})
            if not (cancel and cancel.is_set()):
                op_plan.complete_stage("growth", "Growth preserved · context updated")
            if ("ncm" in layers or kit is not None) and not (cancel and cancel.is_set()):
                emit({"type": "token", "round": rnd_n, "kind": "answer", "text": final})
          finally:
            self.stage = {"name": "idle", "since": time.time()}
        return {"reply": final, "tool_calls": rounds, "rounds": len(stats_all), "generation": stats_all,
                "reasoning": reasoning if thinking else "", "cancelled": bool(cancel and cancel.is_set()),
                "layers": layer_log, "todo_list": op_plan.todos}

    # ---------------- VS Code (OpenAI messages + VS Code's tools) ----------------
    def vscode_step(self, user_prompt, workspace_info, prior_turns, tool_rounds, vs_tools, thinking=True, max_new=1024,
                    deferred=()):
        """One assistant step for VS Code: a VS Code tool call to return, or the final answer."""
        schemas, by_name = [], {}
        core = active_vscode_tools(user_prompt, workspace_info, tool_rounds, deferred)
        for t in vs_tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = str(fn.get("name") or "").strip()
            if not name or name == "vscode_askQuestions" or name in by_name:
                continue
            desc = str(fn.get("description") or "").split("\n")[0][:240]
            params = vscode_tool_params(name, fn.get("parameters") or fn.get("inputSchema") or {"type": "object", "properties": {}})
            f = {"name": name, "description": desc, "parameters": params}
            by_name[name] = f
            entry = {"type": "function", "function": f}
            if name not in core:
                entry["defer_loading"] = True
            schemas.append(entry)
        deferred = [s["function"] for s in schemas if s.get("defer_loading")]
        if deferred:
            st = _fn("search_tools", "Search the catalog of more VS Code tools by keywords. Returns tools you can then call.",
                     {"query": {"type": "string"}}, ["query"])
            schemas.insert(0, st)
            by_name["search_tools"] = st["function"]
        m_root = re.search(r"-\s*([A-Za-z]:\\[^\r\n]+|/[^\r\n]+)", workspace_info or "")
        root = m_root.group(1).strip() if m_root else ""
        sys_txt = ("You are Clover, a coding assistant inside VS Code, running on the user's laptop (model: LFM3-1.7B). "
                   "Inspect the workspace with the tools before answering; never guess file contents. "
                   + (f"The workspace root is {root}; use absolute paths. " if root else "")
                   + ("If none of your tools fits, call search_tools to find one. " if deferred else ""))
        msgs = [{"role": "system", "content": sys_txt}]
        for h in prior_turns[-3:]:
            msgs += [{"role": "user", "content": str(h["user"])[:1500]}, {"role": "assistant", "content": str(h["reply"])[:1500]}]
        ws = f"<workspace_info>\n{workspace_info[:1500]}\n</workspace_info>\n" if workspace_info else ""
        msgs.append({"role": "user", "content": ws + user_prompt[:6000]})
        for rd in tool_rounds[-8:]:
            try:
                args = json.loads(rd.get("arguments") or "{}")
            except ValueError:
                args = {}
            cid = rd.get("id") or ("call_" + uuid.uuid4().hex[:12])
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": rd["name"], "arguments": args if isinstance(args, dict) else {}}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": _clip(rd.get("output") or "", 5000)})
        trace = []
        nudged, answer = False, ""
        with self.lock:
            try:
                for _ in range(4):
                    text, st = self.generate(self._render(msgs, schemas, thinking), SAMPLING["general"], max_new, thinking)
                    reasoning, content, after = split_completion(text, thinking)
                    calls = parse_tool_calls(text.split("</think>", 1)[-1] if thinking else text, by_name)
                    trace.append({"generation": st, "calls": [c["name"] for c in calls], "text": text[-400:]})
                    if calls and calls[0]["name"] == "search_tools":
                        q = str(calls[0]["arguments"].get("query") or "").lower().split()
                        # offered tools too: a search for the Playwright tool it already had found nothing, four times
                        pool = [f for n, f in by_name.items() if n != "search_tools"]
                        scored = sorted(pool, key=lambda f: -sum(w in (f["name"] + " " + f["description"]).lower() for w in q))
                        found = [f["name"] for f in scored[:3] if any(w in (f["name"] + " " + f["description"]).lower() for w in q)]
                        msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": [
                            {"id": calls[0]["id"], "type": "function", "function": {"name": "search_tools", "arguments": calls[0]["arguments"]}}]})
                        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content":
                                     [{"type": "text", "text": f"Found {len(found)} tools."}] + [{"type": "tool_reference", "name": n} for n in found]})
                        continue
                    if calls:
                        c = calls[0]
                        refused = unshared_open(c["name"], c["arguments"], "\n".join(
                            [user_prompt] + [str(h.get("user") or "") for h in prior_turns[-3:]]), workspace_info, tool_rounds)
                        if refused:
                            trace[-1]["refused"] = str(c["arguments"].get("url"))[:200]
                            msgs.append({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": [
                                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]})
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": refused})
                            continue
                        read = shared_page_read(c["name"], c["arguments"], user_prompt, workspace_info, by_name)
                        if read:
                            trace[-1]["read_instead"] = True
                            return {"tool_call": {"name": read["name"], "arguments": json.dumps(read["arguments"])}, "trace": trace}
                        c_args = repair_vscode_browser_args(c["name"], c["arguments"], workspace_info, tool_rounds)
                        if nudged and repeats_round(c["name"], c_args, tool_rounds):
                            return {"content": answer or "(Clover did not finish an answer in time.)", "trace": trace}
                        return {"tool_call": {"name": c["name"], "arguments": json.dumps(c_args)}, "trace": trace}
                    nudge = "" if nudged else unfinished_page_steps(user_prompt, workspace_info, tool_rounds, by_name)
                    if nudge:
                        nudged = trace[-1]["nudged"] = True
                        answer = (content + after).strip()
                        msgs += [{"role": "assistant", "content": content}, {"role": "user", "content": nudge}]
                        continue
                    final = (content + after).strip() if (not thinking or "</think>" in text) else ""
                    return {"content": final or "(Clover did not finish an answer in time.)", "trace": trace}
                return {"content": "(Clover kept searching for tools without choosing one.)", "trace": trace}
            finally:
                self.stage = {"name": "idle", "since": time.time()}
