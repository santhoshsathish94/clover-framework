"""The user-started Clover cycle on the ladder: every answer of a finished run (step 7, the shipped defaults) gets the
user's reply "That's not right. Try again." and is answered again the way the Studio does it (server.lfm3_turn):
the cycle (OmniNative.retry_turn), or for a question that asked for a tool, a new turn with that tool.

The frozen scorer only stands in for the user's judgement; the retry never sees a reference answer.
  fix rate : failed answers that pass after the retry (a user says "not right" when an answer is wrong)
  keep rate: passed answers that still pass (a user who doubts a right answer must not be talked out of it)
Same tasks, scorer, system prompt and decoding as bench_layers. Resumable: appends to <out>.
"""
import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import winperf  # noqa: E402

THROTTLE_OFF = winperf.disable_power_throttling()
import bench70  # noqa: E402
import bench_layers as B  # noqa: E402
import lfm3_native as N  # noqa: E402
import permissions as P  # noqa: E402

FEEDBACK = "That's not right. Try again."


def main():
    bench70.keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--layers", default="memory,ontology,compiler")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(HERE.parent))
    ap.add_argument("--web-cache", default=None)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    base = Path(a.base)
    tasks = {t["task_id"]: t for t in json.loads((HERE / "tasks70.json").read_text(encoding="utf-8"))}
    recs = [json.loads(ln) for ln in Path(a.base_run).read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs = [r for r in recs if r["passed"] is not None and not r.get("error") and r["task_id"] in tasks]
    if a.ids:
        recs = [r for r in recs if r["task_id"] in set(a.ids.split(","))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(ln)["task_id"] for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()}

    perms = P.PermissionStore(Path(tempfile.mkdtemp()) / "perms.json")
    perms.update({"code": "allow", "internet": "allow", "files": "allow", "shell": "deny", "edit_files": "deny"})
    import omni_native
    import lfm3_layers
    eng = omni_native.OmniNative(base, perms, engine="gpu")
    eng._web = B.ReplayWeb(a.web_cache or (Path(a.base_run).parent / "web-cache.jsonl"), "record", base)
    layers = tuple(x for x in a.layers.split(",") if x)
    kit = lfm3_layers.LayerKit(base) if layers else None
    meta = {"engine": eng.describe(), "throttle_off": THROTTLE_OFF, "system": B.SYSTEM, "feedback": FEEDBACK,
            "layers": layers, "base_run": a.base_run, "max_new": a.max_new}
    out.with_suffix(".engine.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("RETRY_BENCH", json.dumps(meta, default=str), flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(recs, 1):
            if r["task_id"] in done:
                continue
            t = tasks[r["task_id"]]
            question = t["prompt"] + (B.GSM8K_SUFFIX if t["benchmark"] == "GSM8K" else "")
            history = [{"role": "user", "content": question}, {"role": "assistant", "content": r["raw_output"]}]
            t0, err, res = time.perf_counter(), None, {}
            only = N.asked_tools(t["prompt"]) + N.asked_tools(FEEDBACK)
            try:
                if only:  # as the Studio: the question asked for a tool, so it is answered again with it
                    res = eng.run_turn(history, FEEDBACK, tools=True, sampling="greedy", thinking=True,
                                       max_new=a.max_new, max_rounds=6, system=B.SYSTEM, layers=layers, kit=kit,
                                       layer_query=FEEDBACK, final_on_limit=True, force_answer=True, only_tools=only)
                    res["layers"].append({"layer": "cycle", "trigger": "user", "action": "redone", "tools": only})
                else:
                    res = eng.retry_turn(history, question, r["raw_output"], FEEDBACK, layers=layers, kit=kit,
                                         max_new=a.max_new, system=B.SYSTEM)
                reply = str(res.get("reply") or "")
            except Exception as exc:
                reply, err = "", f"{type(exc).__name__}: {exc}"
            ok, sc, det = bench70.score(t, reply) if err is None else (False, 0.0, err)
            cyc = next((x for x in res.get("layers") or [] if x.get("layer") == "cycle"), {})
            gens = res.get("generation") or []
            rec = {"index": i, "task_id": t["task_id"], "category": t["category"], "benchmark": t["benchmark"],
                   "base_passed": r["passed"], "passed": ok, "score": sc, "score_detail": det,
                   "action": cyc.get("action"), "direction": cyc.get("direction"), "solutions": cyc.get("solutions"),
                   "overturned": cyc.get("overturned"), "first_findings": cyc.get("first_findings"),
                   "final_findings": cyc.get("final_findings"), "lesson": bool(cyc.get("lesson")),
                   "used": cyc.get("tools") or [],
                   "wall_s": round(time.perf_counter() - t0, 1), "new_tokens": sum(g.get("new_tokens", 0) for g in gens),
                   "calls": len(gens), "tools": [(c["tool"], c["ran"]) for c in res.get("tool_calls") or []],
                   "raw_output": reply, "error": err, "layer_log": res.get("layers")}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            change = {(False, True): "FIXED", (True, False): "BROKE", (True, True): "kept", (False, False): "still"}[
                (bool(r["passed"]), bool(ok))]
            print(f"[{i:02d}/{len(recs)}] {change:5s} {t['task_id']:36s} {rec['wall_s']:6.1f}s action={rec['action']} "
                  f"dir={rec['direction']} sol={rec['solutions']} used={rec['used']} tools={rec['tools']} {det}", flush=True)

    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    failed = [x for x in rows if not x["base_passed"]]
    passed = [x for x in rows if x["base_passed"]]
    summ = {"tasks": len(rows), "base_pass": len(passed), "after_retry_pass": sum(1 for x in rows if x["passed"]),
            "fixed": sum(1 for x in failed if x["passed"]), "of_failed": len(failed),
            "kept": sum(1 for x in passed if x["passed"]), "of_passed": len(passed),
            "actions": dict(Counter(x["action"] for x in rows)),
            "by_action": {k: {"fixed": sum(1 for x in failed if str(x["action"]) == k and x["passed"]),
                              "failed_in": sum(1 for x in failed if str(x["action"]) == k),
                              "broke": sum(1 for x in passed if str(x["action"]) == k and not x["passed"]),
                              "passed_in": sum(1 for x in passed if str(x["action"]) == k)}
                          for k in sorted({str(x["action"]) for x in rows})},
            "lessons": sum(1 for x in rows if x["lesson"]), "wall_s": round(sum(x["wall_s"] for x in rows), 1),
            "by_tool": {u: {"tasks": sum(1 for x in rows if u in x.get("used", [])),
                            "fixed": sum(1 for x in failed if u in x.get("used", []) and x["passed"]),
                            "broke": sum(1 for x in passed if u in x.get("used", []) and not x["passed"])}
                        for u in ("a web search", "a Python calculation")},
            "web": eng._web.stats}
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ), flush=True)
    print("RETRY_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
