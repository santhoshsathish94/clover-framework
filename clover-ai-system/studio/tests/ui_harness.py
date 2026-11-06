"""Offline harness for the Studio page: the real server.py routes with a stand-in engine, on the dev PC.

Usage (from laptop/):  python tests/ui_harness.py [--port 8799] [--base DIR] [--fresh]
Serves http://127.0.0.1:<port>/ from a scratch folder (default %TEMP%\\clover-ui-harness): settings, permissions,
search, workspaces, storage, sessions and the repository are the real code; only the model is replaced. The stand-in
streams a short answer; a message containing "run" calls run_powershell under the real permission store (Allow / Ask /
Off), so the permission card, tool card, layer tags, Try again and Helpful can be checked in a browser. "code" adds a
code block and a code-check tag. "search" shows a web search and a page robots.txt refused, each with its "in your
browser" link; a click on one is printed (OPEN ...), not opened. Nothing here touches the laptop.
"""
import argparse
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import paths  # noqa: E402
import server  # noqa: E402


class FakeKit:
    """What the repository and /api/memory ask of the layer kit."""
    def memories(self):
        return [{"id": "m1", "kind": "note", "summary": "The user's sister Meera lives in Coimbatore."},
                {"id": "m2", "kind": "learning", "summary": "Check units before answering."}]

    def forget(self, eid):
        return False

    def score_learning(self, eid, worked, why):
        return None

    def learn(self, text, sid, n, kind="user"):
        return None, []

    def store(self):  # the repository's growth summary reads the knowledge entries
        return type("Store", (), {"entries": {}})()


class FakeEngine:
    engine_kind = "gpu"

    def __init__(self, perms):
        self.perms = perms
        self.inputs = ["text", "vision", "audio"]
        self.stage = {"name": "idle", "since": time.time()}
        self.workdir = self.workname = self.after_write = None

    def describe(self):
        return {"model": "LFM3-Omni-2B", "mode": "omni", "engine": "gpu", "device": "Harness GPU", "backend": "ui harness",
                "weights": "none", "inputs": self.inputs, "outputs": ["text", "speech"], "max_tokens": 32768, "setup_s": 0.2}

    def _stream(self, emit, text, cancel):
        """-> (text written, stopped): a stopped answer keeps only what was written, as in the real engines."""
        done = []
        for w in text.split(" "):
            if cancel is not None and cancel.is_set():
                return " ".join(done), True
            emit({"type": "token", "round": 1, "kind": "answer", "text": w + " "})
            done.append(w)
            time.sleep(0.04)
        return text, False

    def run_turn(self, history, user_text, *, on_event=None, ask=None, cancel=None, allow=(), layers=(), **kw):
        import lfm3_native as N
        emit = on_event or (lambda e: None)
        self.stage = {"name": "thinking", "since": time.time()}
        log, low = [], user_text.lower()
        if "memory" in layers:
            info = {"layer": "memory", "entries": ["m1", "m2"]}
            log.append(info)
            emit(dict(info, type="layer"))
        if "recall" in layers:
            info = {"layer": "recall", "refs": ["sessions/2026-10/c1.jsonl#1:user"], "chats": 1}
            log.append(info)
            emit(dict(info, type="layer"))
        emit({"type": "plan", "round": 1, "stage": "planning", "tools": kw.get("only_tools") or []})
        if kw.get("thinking", True):
            emit({"type": "token", "round": 1, "kind": "thinking",
                  "text": "Understanding the request and checking which local tools and context notes apply."})
            time.sleep(0.15)
        reply = "Hello from the harness. You said: " + user_text[:80]
        if "run" in low:
            emit({"type": "plan", "round": 1, "stage": "preparing_tool", "tool": "run_powershell"})
            emit({"type": "tool_call", "id": "t1", "name": "run_powershell", "arguments": {"command": "Get-Date"}})
            mode = self.perms.get()["shell"]
            ok = "shell" in allow or mode == "allow" or (
                mode == "ask" and ask is not None and ask("shell", "Get-Date\n(in C:\\Users\\you, up to 60 s)"))
            res_txt = "Error: Command failed [exit 1]" if "fail" in low else ("Saturday 31 October 2026 22:00" if ok else "")
            worked, summary = N.tool_worked("run_powershell", ok, res_txt, "allowed" if ok else "not allowed")
            emit({"type": "tool_result", "id": "t1", "name": "run_powershell", "ran": ok, "worked": worked,
                  "summary": summary, "why": "allowed" if ok else "not allowed", "result": res_txt})
            emit({"type": "round", "round": 2})
            emit({"type": "plan", "round": 2, "stage": "reviewing"})
            reply = "The date is Saturday, 31 October 2026." if (ok and worked) else "I could not run the command."
        if "code" in low:
            reply += "\n\n```python\nprint(2 ** 64)\n```"
            c_info = {"layer": "compiler", "checked": True, "ok": True}
            log.append(c_info)
            emit(dict(c_info, type="layer"))
        if "picture" in low:  # the real generate_image tool from server.lfm3_turn, under the real permission store
            tool = self._extra.get("generate_image") if hasattr(self, "_extra") else None
            if tool is None:
                tool = next((t for t in kw.get("extra_tools") or [] if t["schema"]["function"]["name"] == "generate_image"), None)
            emit({"type": "tool_call", "id": "t2", "name": "generate_image", "arguments": {"prompt": user_text}})
            mode = self.perms.get()["media"]
            okp = "media" in allow or mode == "allow" or (mode == "ask" and ask is not None and ask("media", "Create a picture: " + user_text))
            result = tool["run"]({"prompt": user_text}) if (okp and tool) else "not allowed"
            emit({"type": "tool_result", "id": "t2", "name": "generate_image", "ran": okp, "why": "", "result": result})
            reply = "Here is your picture." if okp else "I could not make the picture."
        if "slow" in low:
            reply = " ".join(["word"] * 120)
        if "search" in low:  # browser-first search: the steps offer the user's own browser (2026-11-06)
            q = user_text[:80]
            emit({"type": "tool_call", "id": "t5", "name": "web_search", "arguments": {"query": q}})
            res = "• Reference Source 1: Pune (https://en.wikipedia.org/wiki/Pune): Pune is a city in Maharashtra, India."
            emit({"type": "tool_result", "id": "t5", "name": "web_search", "ran": True, "worked": True,
                  "summary": "1 result from Wikipedia", "why": "allowed", "result": res})
            url = "https://www.google.com/search?q=" + "+".join(q.split())
            emit({"type": "tool_call", "id": "t6", "name": "fetch_url", "arguments": {"url": url}})
            res = (f"[URL Fetch Error for {url}: www.google.com's robots.txt asks automated readers not to open this "
                   "page; open it in your browser instead]")
            worked, summary = N.tool_worked("fetch_url", True, res)
            emit({"type": "tool_result", "id": "t6", "name": "fetch_url", "ran": True, "worked": worked,
                  "summary": summary, "why": "allowed", "result": res})
            reply = "Wikipedia says Pune is a city in Maharashtra, India. For the rest of the web, search in your browser."
        if "table" in low:
            reply = "Your files:\n\n| File | Size |\n\n|------|------|\n\n| README | 14 bytes |\n\n| notes.txt | 5 bytes |\n\nThat is all."
        if "rebase" in low:  # a command that asks even when the terminal is allowed
            import local_terminal
            emit({"type": "tool_call", "id": "t3", "name": "run_powershell", "arguments": {"command": "git rebase -i HEAD~1"}})
            okr = ask is not None and ask("shell", "git rebase -i HEAD~1\n(in C:\\Users\\you\\repo, up to 60 s)",
                                          local_terminal.risky("git rebase -i HEAD~1"))
            emit({"type": "tool_result", "id": "t3", "name": "run_powershell", "ran": okr, "why": "", "result": ""})
            reply = "Rebased." if okr else "I did not run it."
        if "clock" in low:  # the reported pattern: an answer, an unasked clock call, the answer again (2026-11-03)
            self._stream(emit, reply, cancel)
            emit({"type": "tool_call", "id": "t4", "name": "get_current_time", "arguments": {}})
            time.sleep(0.6)
            emit({"type": "tool_result", "id": "t4", "name": "get_current_time", "ran": True, "why": "no permission needed",
                  "result": "Tuesday 03 November 2026, 17:30:00"})
            emit({"type": "round", "round": 2})
            time.sleep(0.8)
        reply, stopped = self._stream(emit, reply, cancel)
        self.stage = {"name": "idle", "since": time.time()}
        return {"reply": reply, "layers": log, "cancelled": stopped,
                "generation": [{"new_tokens": len(reply.split()), "decode_s": 0.5}]}

    def retry_turn(self, history, question, previous, feedback, *, on_event=None, cancel=None, **kw):
        emit = on_event or (lambda e: None)
        emit({"type": "cycle", "stage": "check"})
        time.sleep(0.3)
        emit({"type": "cycle", "stage": "second"})
        reply, _ = self._stream(emit, "Checked again: the earlier answer holds.", cancel)
        return {"reply": reply, "layers": [{"layer": "cycle", "trigger": "user", "action": "kept"}], "generation": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--base", default=str(Path(tempfile.gettempdir()) / "clover-ui-harness"))
    ap.add_argument("--fresh", action="store_true", help="start from an empty scratch folder")
    a = ap.parse_args()
    base = Path(a.base)
    if a.fresh and base.exists():
        def writable(func, p, _exc):  # the repository keeps AGENTS.md read-only
            os.chmod(p, stat.S_IWRITE)
            func(p)
        shutil.rmtree(base, onexc=writable)
    (base / "data").mkdir(parents=True, exist_ok=True)
    import clover_repo
    import memory_index
    import permissions as P
    import vscode_bridge
    import web_search
    import workspaces
    S = server.S
    S.args = argparse.Namespace(base=str(base), port=a.port, mode="lfm3", no_gate=False, idle_exit_min=0.0)
    S.port, S.mode = a.port, "lfm3"
    S.perms = P.PermissionStore(paths.config_file(base, "permissions.json"))
    S.settings = server.Settings(paths.config_file(base, "settings.json"))
    S.search = web_search.SearchConfig.shared(paths.config_file(base, "search.json"), paths.vault_file(base, "search-keys.json"))
    S.github = workspaces.GitHubAuth(paths.vault_file(base, "github.json"))
    S.workspaces = workspaces.Workspaces(paths.config_file(base, "workspaces.json"), base / "workspaces", base, S.github)
    S.vscode = vscode_bridge.Bridge(paths.vault_file(base, "vscode-bridge.json"))
    kit = FakeKit()
    server.layer_kit = lambda: kit
    S.repo = clover_repo.Repo(base, CLOVER / "AGENTS.md", kit=server.layer_kit)
    S.memidx = memory_index.MemoryIndex(base, S.repo, start=False, recall=S.settings.get()["layers"]["recall"])
    S.engine = FakeEngine(S.perms)
    S.omni_engine = S.engine
    real_open = server.open_in_browser
    server.open_in_browser = lambda body, start=None: real_open(
        body, start=lambda u: print("OPEN (recorded, not opened by the harness):", u, flush=True))
    server.setup_needed = lambda: False
    server.omni_available = lambda _b: (True, "")
    server.gpu_available = lambda _b: (True, "")
    import image_gen  # stand-in picture: a small PNG instead of running stable-diffusion.cpp

    def fake_generate(self, prompt, out_dir, size=512, seed=None, cancel=None, time_limit_s=0):
        import hashlib
        import struct
        import zlib
        w = h = 64
        raw = b"".join(b"\x00" + bytes([30, 120, 50] * w) for _ in range(h))
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(png).hexdigest()[:32] + ".png"
        (Path(out_dir) / name).write_bytes(png)
        return {"file": name, "seconds": 0.1, "model": "harness picture", "seed": 1, "width": w, "height": h}
    image_gen.ImageGen.generate = fake_generate
    (base / "tools" / "sd.cpp" / "vulkan").mkdir(parents=True, exist_ok=True)
    (base / "tools" / "sd.cpp" / "vulkan" / "sd-cli.exe").write_bytes(b"")
    S.load.update(status="ready", step="ready", ready_s=0.1)
    S.httpd = server.StudioHTTPServer(("127.0.0.1", a.port), server.Handler)
    S.httpd.daemon_threads = True
    print(f"HARNESS http://127.0.0.1:{a.port}/ base {base}", flush=True)
    S.httpd.serve_forever()


if __name__ == "__main__":
    main()
