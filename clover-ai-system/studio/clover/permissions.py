"""Tool permissions for the local Clover server.

Nothing outside the model runs unless the user allowed it: code execution (the Python tool, Layer 6 doctest runs,
rustc if installed) and internet access (web search / page fetch). Each tool has a mode:
  allow - runs without asking
  ask   - the turn stops before the tool runs and the Studio asks the user (default)
  deny  - the tool does not run; Clover is told it is turned off and answers without it
Checks happen at the call site of the tool itself, so a refusal cannot be bypassed by the planner.
"""
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

TOOLS = {
    "internet": {
        "label": "Internet search and web pages",
        "detail": "Searches the web with the service chosen in Settings (Tavily, Brave Search, or Wikipedia and arXiv) and "
                  "opens pages. The search text is sent to that service.",
    },
    "code": {
        "label": "Run code Clover writes",
        "detail": "Python (and rustc if installed) runs as you in a separate hidden process with a time limit. "
                  "It is not network-isolated.",
    },
    "shell": {
        "label": "Run PowerShell commands",
        "detail": "Commands run as you on this laptop, in the folder Clover names, for up to 60 seconds (Clover can ask "
                  "for up to 10 minutes; the request shows it). They can do anything you can do. Stopping the answer "
                  "stops the command.",
    },
    "files": {
        "label": "Read files and folders",
        "detail": "Lists folders, searches files, reads text files, documents, spreadsheets and media metadata, "
                  "and runs read-only Git inspections on this laptop that Clover asks for.",
    },
    "edit_files": {
        "label": "Create or change files",
        "detail": "Writes text files on this laptop. Existing files are backed up next to the original first.",
    },
    "memory": {
        "label": "Remember things you tell it",
        "detail": "Saves what you ask Clover to remember as notes in Clover's repository on this laptop "
                  "(clover-ai-studio\\growth\\knowledge). Open that folder to read or delete them.",
    },
    "vscode": {
        "label": "Use VS Code",
        "detail": "Asks VS Code to open files, folders and web pages, clone GitHub repositories, install extensions "
                  "and sign in to GitHub. VS Code shows what it does; terminal commands also need Run PowerShell.",
    },
    "media": {
        "label": "Create pictures",
        "detail": "Makes a picture from a description with the image model on this laptop (the Iris Xe GPU). Nothing "
                  "is sent anywhere; each picture is saved in the media folder.",
    },
}
MODES = ("allow", "ask", "deny")
DENIED_TEXT = {
    "internet": "[Clover permission] Internet access is turned off on this laptop, so no web search was made.",
    "code": "[Clover permission] Running code is turned off on this laptop, so the code was not executed.",
    "shell": "[Clover permission] Running commands is turned off on this laptop, so the command was not run.",
    "files": "[Clover permission] Reading files is turned off on this laptop, so the file was not read.",
    "edit_files": "[Clover permission] Changing files is turned off on this laptop, so nothing was written.",
    "memory": "[Clover permission] Remembering is turned off on this laptop, so nothing was saved.",
    "vscode": "[Clover permission] Using VS Code is turned off on this laptop, so VS Code was not asked.",
    "media": "[Clover permission] Creating pictures is turned off on this laptop, so no picture was made.",
}
DECLINED_TEXT = ("[Clover permission] The user chose not to allow this just now, so it did not run. Say so plainly "
                 "(it is not turned off) and offer another way.")


class PermissionNeeded(BaseException):
    """Stops the turn before a tool runs. BaseException so the loop's `except Exception` handlers cannot swallow it."""

    def __init__(self, tool, detail):
        super().__init__(f"permission needed: {tool}")
        self.tool = tool
        self.detail = str(detail or "")[:4000]


class PermissionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.modes = {t: ("allow" if t == "memory" else "ask") for t in TOOLS}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if isinstance(saved, dict):
            for t, m in saved.items():
                if t in TOOLS and m in MODES:
                    self.modes[t] = m

    def get(self):
        with self.lock:
            return dict(self.modes)

    def update(self, changes):
        if not isinstance(changes, dict):
            raise ValueError("permissions must be an object")
        for t, m in changes.items():
            if t not in TOOLS or m not in MODES:
                raise ValueError(f"unknown permission {t!r}={m!r}")
        with self.lock:
            self.modes.update(changes)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.modes, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            return dict(self.modes)

    def describe(self):
        modes = self.get()
        return {t: {"mode": modes[t], **TOOLS[t]} for t in TOOLS}


_local = threading.local()


@contextmanager
def turn_scope(store, allow_once=(), deny_once=(), interactive=True):
    """Permission context for one turn. Outside a turn every tool is refused (fail closed)."""
    ctx = {"store": store, "allow": set(allow_once) & set(TOOLS), "deny": set(deny_once) & set(TOOLS),
           "interactive": bool(interactive), "events": []}
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = None


def check(tool, detail):
    ctx = getattr(_local, "ctx", None)
    if ctx is None:
        return False
    if tool in ctx["deny"]:
        ok = False
    elif tool in ctx["allow"]:
        ok = True
    else:
        mode = ctx["store"].get()[tool]
        if mode == "ask" and ctx["interactive"]:
            raise PermissionNeeded(tool, detail)
        ok = mode == "allow"
    ctx["events"].append({"tool": tool, "allowed": ok, "detail": str(detail or "")[:300]})
    return ok


class GatedPython:
    """Wraps PythonTool: same execute() contract, refuses unless `code` is allowed."""

    def __init__(self, inner):
        self.inner = inner

    def execute(self, code):
        if check("code", code):
            return self.inner.execute(code)
        return DENIED_TEXT["code"]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class GatedWeb:
    """Wraps LiveWebSearchTool (the only network path of K3KnowledgeSearchTool)."""

    def __init__(self, inner):
        self.inner = inner

    def execute(self, query):
        if check("internet", query):
            return self.inner.execute(query)
        return DENIED_TEXT["internet"]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class GatedSubprocess:
    """Replaces the `subprocess` name inside the Layer 6 compiler module: doctest runs and rustc need `code`."""

    def __init__(self, real):
        self._real = real

    def run(self, args, *a, **kw):
        if not check("code", " ".join(map(str, args)) if isinstance(args, (list, tuple)) else args):
            raise self._real.SubprocessError(DENIED_TEXT["code"])
        return self._real.run(args, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)
