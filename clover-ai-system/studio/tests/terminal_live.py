"""Live check of the terminal tool (Layer 3, tool 1) through the Studio's streamed chat, as the page uses it. Each
permission request is answered "once" (the user's saved permissions are not changed). Checks: the request shows the
command, folder and time limit; the command runs in the folder asked for and reports its exit code; Stop ends a long
command; the test chats are deleted afterwards.
Usage: python tests/terminal_live.py [out.jsonl]"""
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "terminal_live.jsonl"


def post(path, body, timeout=60):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def log(name, ok, **kw):
    rec = {"check": name, "ok": bool(ok), **kw}
    print(json.dumps(rec, ensure_ascii=False)[:1500], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stream(sid, prompt, on_event=None):
    """One streamed turn -> list of events. on_event(ev, turn_id) may answer permissions or cancel."""
    r = urllib.request.Request(BASE + "/api/chat/stream", method="POST",
                               data=json.dumps({"prompt": prompt, "session_id": sid}).encode(),
                               headers={"Content-Type": "application/json"})
    evs, turn_id = [], None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            evs.append(ev)
            turn_id = ev.get("turn_id") or turn_id
            if ev.get("type") == "permission":
                post("/api/decision", {"turn_id": turn_id, "request_id": ev["request_id"], "decision": "once"})
            if on_event:
                on_event(ev, turn_id)
            if ev.get("type") in ("done", "error"):
                break
    return evs


def main():
    stamp = time.strftime("%H%M%S")
    s1, s2 = f"live-term-a-{stamp}", f"live-term-b-{stamp}"
    folder = Path(os.environ["USERPROFILE"]) / "CloverAI" / "verify"
    try:
        evs = stream(s1, f"Use the terminal: in the folder {folder}, run Get-ChildItem -Name and tell me how many "
                         f"entries it lists.")
        perm = next((e for e in evs if e.get("type") == "permission"), None)
        calls = [e for e in evs if e.get("type") == "tool_call" and e.get("name") == "run_powershell"]
        res = [e for e in evs if e.get("type") == "tool_result" and e.get("name") == "run_powershell"]
        done = next((e for e in evs if e.get("type") == "done"), {})
        log("permission request shows the command, folder and limit" if perm else
            "permission request NOT exercised (shell is set to Allow on this laptop; covered offline)",
            perm is None or ("up to" in perm.get("detail", "") and "(in " in perm.get("detail", "")),
            permission=(perm or {}).get("detail", "")[:300])
        log("ran in the folder and returned an exit code",
            bool(res) and res[0].get("ran") and "[exit code" in res[0].get("result", "")
            and "recall-memory" in res[0].get("result", ""),
            call=(calls[0].get("arguments") if calls else None), result=(res[0].get("result", "")[:300] if res else None),
            reply=done.get("reply", "")[:200])

        stopped = {"at": None}

        def stop_when_running(ev, turn_id):
            if ev.get("type") == "tool_call" and ev.get("name") == "run_powershell" and stopped["at"] is None:
                def later():
                    time.sleep(4)
                    stopped["at"] = time.time()
                    post("/api/cancel", {"turn_id": turn_id})
                threading.Thread(target=later, daemon=True).start()
        t0 = time.time()
        evs = stream(s2, "Use the terminal to run this exact PowerShell command with timeout_s 300: Start-Sleep -Seconds 120",
                     stop_when_running)
        res = [e for e in evs if e.get("type") == "tool_result" and e.get("name") == "run_powershell"]
        took = time.time() - (stopped["at"] or t0)
        log("Stop ends a long command within seconds",
            stopped["at"] is not None and took < 20 and (not res or "stopped" in res[0].get("result", "")),
            seconds_after_stop=round(took, 1), result=(res[0].get("result", "")[:200] if res else None))
    finally:
        for sid in (s1, s2):
            post("/api/session/forget", {"session_id": sid, "delete": True})
        log("cleanup: test chats deleted", True)


if __name__ == "__main__":
    main()
