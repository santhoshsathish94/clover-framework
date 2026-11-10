"""Usage evidence for the VS Code tools Clover offers in full (TokenEconomizer's tool pruning, made reversible): a
tool of a cue-activated group (browser, terminal, code intelligence) that was offered in full in at least
EVIDENCE_TURNS turns and never called is deferred behind search_tools from then on. It stays reachable, and one call
makes it an ordinary tool again. The always-offered core is never deferred. Kept in <base>/growth/tool-usage.json.

Off unless the setting "tool_deferral" is on: measured on k3 and the laptop (2026-11-10) it cut prefill by 41-44% but
made the first step worse on 21 of 43 captured VS Code requests, because it hid the browser tools those requests needed.
"""
import hashlib
import json
import re
import threading
import time
from pathlib import Path

EVIDENCE_TURNS = 10
KEEP_TURNS = 400
_LOCK = threading.Lock()
_ROOT = re.compile(r"^\s*-\s*([A-Za-z]:\\[^\r\n]+|/[^\r\n]+)", re.M)


def turn_id(workspace_info, user_prompt):
    """The steps of one VS Code turn share it: the workspace root and the user's request."""
    m = _ROOT.search(workspace_info or "")
    root = m.group(1).strip().rstrip("\\/").lower() if m else ""
    return hashlib.sha1(f"{root}\n{user_prompt}".encode("utf-8")).hexdigest()[:16]


class ToolUsage:
    def __init__(self, base_dir):
        self.path = Path(base_dir) / "growth" / "tool-usage.json"

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"turns": {}, "called": {}}

    def record(self, turn_id, offered, called=None):
        """One VS Code step: the tools offered in full, and the one it called (if any)."""
        if not turn_id:
            return
        with _LOCK:
            doc = self._load()
            t = doc["turns"].setdefault(turn_id, {"at": time.time(), "offered": []})
            t["offered"] = sorted(set(t["offered"]) | set(offered or ()))
            if called:
                doc["called"][called] = doc["called"].get(called, 0) + 1
            for old in sorted(doc["turns"], key=lambda k: doc["turns"][k]["at"])[:-KEEP_TURNS]:
                del doc["turns"][old]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    def deferred(self, protected=()):
        """The tools to defer: offered in at least EVIDENCE_TURNS turns, never called, not protected."""
        doc = self._load()
        seen = {}
        for t in doc["turns"].values():
            for name in t.get("offered", ()):
                seen[name] = seen.get(name, 0) + 1
        return {n for n, k in seen.items()
                if k >= EVIDENCE_TURNS and not doc["called"].get(n) and n not in set(protected)}
