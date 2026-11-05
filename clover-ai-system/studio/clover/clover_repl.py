"""Terminal chat with the local Clover server: python clover_repl.py [--url http://127.0.0.1:8765]"""
import argparse
import json
import sys
import urllib.request
import uuid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sid = "repl-" + uuid.uuid4().hex[:12]
    with urllib.request.urlopen(a.url + "/api/health", timeout=10) as r:
        h = json.loads(r.read())
    e = h.get("engine") or {}
    print(f"Clover local · {h.get('model')} · planner: {'LFM3 HybridAgent' if e.get('hybridlink') else 'base LFM3'}")
    print("Type a message; /new starts a new session; /quit exits.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/new":
            sid = "repl-" + uuid.uuid4().hex[:12]
            print("(new session)")
            continue
        body = json.dumps({"session_id": sid, "prompt": q, "cands": 2, "remember": True}).encode("utf-8")
        req = urllib.request.Request(a.url + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                res = json.loads(r.read())
        except Exception as exc:
            print(f"error: {exc}")
            continue
        tools = [t.get("tool") for t in res.get("tool_calls") or [] if t.get("tool")]
        print(f"\nclover> {res.get('reply')}")
        print(f"        [{res.get('wall_ms', 0) / 1000:.1f} s, latent {res.get('latent_steps')}"
              + (f", tools: {', '.join(tools)}" if tools else "") + "]")


if __name__ == "__main__":
    main()
