"""Live check of Layer 2 (recall) on the laptop. Recall is switched on for the test and the user's setting restored after.
Chat A tells Clover two facts; a new chat B asks about them; a general question follows (memory must not get in the
way); an answer the user says is not right must not come back. Then: notes written while idle, the execution log's
recall lines, and the time a turn takes with recall. The test conversations are deleted afterwards.
Tools are off in every request (a web search would mask whether recall found the answer).
Usage: python tests/recall_live.py [out.jsonl]"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "recall_live.jsonl"
REPO = Path(os.environ["USERPROFILE"]) / "CloverAI" / "clover-ai-studio"


def call(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def log(name, ok, **kw):
    rec = {"check": name, "ok": bool(ok), **kw}
    print(json.dumps(rec, ensure_ascii=False)[:1600], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def chat(sid, prompt):
    t0 = time.perf_counter()
    r = call("POST", "/api/chat", {"prompt": prompt, "session_id": sid, "tools": False})
    r["_s"] = round(time.perf_counter() - t0, 1)
    return r


def recall_of(r):
    return next((x for x in r.get("layers") or [] if x.get("layer") == "recall"), None)


def main():
    stamp = time.strftime("%H%M%S")
    a, b, c = f"live-recall-a-{stamp}", f"live-recall-b-{stamp}", f"live-recall-c-{stamp}"
    before = call("GET", "/api/settings")["settings"]["layers"]["recall"]
    call("POST", "/api/settings", {"layers": {"recall": True}})
    try:
        for _ in range(60):  # the embedder loads in the background
            st = call("GET", "/api/health")["recall"]
            if st["state"] == "ready":
                break
            time.sleep(2)
        log("recall ready after switching it on", st["state"] == "ready", recall=st)

        chat(a, "My sister Meera just moved to Coimbatore for her new job at a textile company.")
        chat(a, "Also, my bike is a blue Hero Splendor that I bought in 2019.")
        r = chat(b, "Which city did my sister move to?")
        lay = recall_of(r)
        log("new chat: recall brings chat A's message and the answer uses it",
            lay is not None and any(a in x for x in lay.get("refs") or []) and "coimbatore" in r["reply"].lower(),
            seconds=r["_s"], recall=lay, reply=r["reply"][:300])
        r = chat(b, "What colour is my bike?")
        log("second fact from chat A", "blue" in r["reply"].lower(), seconds=r["_s"], recall=recall_of(r),
            reply=r["reply"][:300])
        r = chat(c, "What is 17 times 23?")
        log("general question with memory present: still answered (391)", "391" in r["reply"], seconds=r["_s"],
            recall=recall_of(r), reply=r["reply"][:200])
        chat(c, "Which planet is closest to the sun?")
        r = chat(c, "That is not right, check again")  # whatever the answer was, it is now judged 'not right'
        bad_ref = f"{c}.jsonl#2:clover"
        r2 = chat(b, "Which planet is closest to the sun?")
        lay = recall_of(r2) or {}
        log("an answer the user said was not right is not recalled",
            not any(x.endswith(bad_ref) for x in lay.get("refs") or []), recall=lay, reply=r2["reply"][:200])

        mine, t0 = [], time.time()
        while time.time() - t0 < 150 and len(mine) < 2:  # idle: the CPU instance writes the exchange notes
            time.sleep(10)
            notes = [json.loads(x) for f in REPO.glob("growth/notes/*/*.jsonl")
                     for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            mine = [n for n in notes if n.get("session") in (a, b, c)]
        st = call("GET", "/api/health")["recall"]
        log("notes written while idle (growth/notes/)", len(mine) >= 2, after_s=round(time.time() - t0),
            notes=[(n["session"][-8:], n["n"], n["note"][:120]) for n in mine][:6], recall=st)
        logtxt = "".join(f.read_text(encoding="utf-8", errors="replace") for f in REPO.glob("execution/*/*.log"))
        lines = [ln for ln in logtxt.splitlines() if f" {b}#" in ln and " context: " in ln and "recall " in ln]
        log("execution log lists what was recalled, by reference", bool(lines), lines=[ln[:300] for ln in lines[:2]])
    finally:
        call("POST", "/api/settings", {"layers": {"recall": before}})
        for sid in (a, b, c):
            call("POST", "/api/session/forget", {"session_id": sid, "delete": True})
        log("cleanup: recall setting restored, test conversations deleted",
            call("GET", "/api/settings")["settings"]["layers"]["recall"] == before)
    time.sleep(60)  # a note being written during the deletion must not come back
    left = [json.loads(x) for f in REPO.glob("growth/notes/*/*.jsonl") for x in f.read_text(encoding="utf-8").splitlines()
            if x.strip() and json.loads(x).get("session") in (a, b, c)]
    log("no notes left for the deleted test conversations a minute later", not left, left=len(left))


if __name__ == "__main__":
    main()
