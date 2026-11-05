"""Live check on the laptop: a real conversation lands in Clover's repository the way the human directed. Turn 1 an open
answer; turn 2 "not right" with what was missing (feedback: did not work; the correction becomes a learning); turn 3
the user continues (feedback: worked; the learning gets worked 1); Helpful on turn 3; a new chat asks about last time
(the records layer retrieves that conversation). Then the files, the summaries and the commits are checked. The test
conversations and the learning they made are deleted afterwards; their log and outcome lines stay (they did run).
Usage: python tests/clover_repo_live.py [out.jsonl]"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "clover_repo_live.jsonl"
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


def jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def main():
    stamp = time.strftime("%H%M%S")
    a, b = f"live-repo-a-{stamp}", f"live-repo-b-{stamp}"
    turns = [("In one sentence: what is a context file for?", None),
             ("That is not right: you forgot to say it is a handoff that lets the next session continue without "
              "starting from zero.", None),
             ("Thanks. Now name one thing it should contain.", None)]
    res = []
    for prompt, _ in turns:
        t0 = time.perf_counter()
        r = call("POST", "/api/chat", {"prompt": prompt, "session_id": a})
        res.append(r)
        log(f"turn {r.get('turn')}", bool(r.get("reply")), seconds=round(time.perf_counter() - t0, 1),
            retry=r.get("retry"), growth=r.get("growth"), layers=[x.get("layer") for x in r.get("layers") or []],
            reply=(r.get("reply") or "")[:240])
    fb = call("POST", "/api/feedback", {"session_id": a, "turn": res[2].get("turn")})
    log("helpful on turn 3", "growth" in fb, growth=fb.get("growth"))
    r = call("POST", "/api/chat", {"prompt": "What did we talk about last time?", "session_id": b})
    rec_layer = next((x for x in r.get("layers") or [] if x.get("layer") == "records"), None)
    log("a new chat asks about last time: the records layer brings that conversation",
        rec_layer is not None and "sessions" in rec_layer.get("parts", []), layer=rec_layer,
        reply=(r.get("reply") or "")[:300])
    time.sleep(4)  # the background commit

    sess = next(REPO.glob(f"sessions/*/{a}.jsonl"), None)
    lines = jsonl(sess) if sess else []
    log("sessions/: the three inputs and three answers, nothing else",
        [(m["n"], m["role"]) for m in lines] == [(1, "user"), (1, "clover"), (2, "user"), (2, "clover"), (3, "user"),
                                                  (3, "clover")] and all(set(m) <= {"n", "role", "at", "text", "files",
                                                                                     "media"} for m in lines),
        file=str(sess), roles=[(m["n"], m["role"]) for m in lines])
    recs = [x for f in REPO.glob("outcomes/*/*.jsonl") for x in jsonl(f) if x.get("session") == a]
    fbs = {(x["n"], x["feedback"]) for x in recs if "feedback" in x}
    lesson = next((x.get("lesson") for x in recs if x.get("n") == 2 and x.get("lesson")), None)
    log("outcomes/: turn 1 did not work (not right), turn 2 worked (continued), turn 3 helpful",
        {(1, "said_not_right"), (2, "continued"), (3, "helpful")} <= fbs, feedback=sorted(fbs), lesson=lesson)
    mem = call("GET", "/api/memory")["memories"]
    m = next((x for x in mem if x["id"] == lesson), None)
    log("growth/: the correction is a learning that worked once", m is not None and m["kind"] == "learning"
        and m["worked"] >= 1 and m["did_not_work"] == 0, learning=m)
    logtxt = "".join(f.read_text(encoding="utf-8", errors="replace") for f in REPO.glob("execution/*/*.log"))
    mine = [ln for ln in logtxt.splitlines() if f" {a}#" in ln]
    log("execution/: plain-text steps of each turn", any(" message: " in ln for ln in mine)
        and any(" answer: " in ln for ln in mine) and any(" feedback: " in ln for ln in mine)
        and any(" growth: " in ln for ln in mine), lines=len(mine), sample=mine[:8])
    ctx = {f: (REPO / f / "CONTEXT.md").read_text(encoding="utf-8") for f in ("", "sessions", "outcomes", "growth")}
    log("summaries name the conversation, the outcomes and the learning",
        "context file" in ctx["sessions"] and "did not work" in ctx["outcomes"] and (lesson or "?") in ctx["growth"],
        root=ctx[""][:600])
    git = lambda *x: subprocess.run(["git", *x], cwd=REPO, capture_output=True, text=True).stdout  # noqa: E731
    log("committed, nothing left over", git("status", "--porcelain").strip() == "",
        commits=git("log", "--oneline", "-6").strip().splitlines())

    for sid in (a, b):
        call("POST", "/api/session/forget", {"session_id": sid, "delete": True})
    if lesson:
        call("POST", "/api/memory/forget", {"id": lesson})
    log("test conversations and their learning deleted", next(REPO.glob(f"sessions/*/{a}.jsonl"), None) is None
        and not any(x["id"] == lesson for x in call("GET", "/api/memory")["memories"]))


if __name__ == "__main__":
    main()
