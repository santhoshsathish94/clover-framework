"""Item 2 (follow-up to step 4): does Clover answer questions about a verified workspace function better when (a) the
memory note's wording no longer says "answer from them" and stops calling verified notes unverified, and (b) the
question's literal calls are computed by running the verified copy in IsolatedPythonSandbox?
Same two unannotated workspaces and 8 questions as step 4; live LFM3-Omni-2B Q4_0, greedy, same system message.
  HEAD      step 4's ON contexts, unchanged (checked byte-identical to the new contexts with the old wording put back)
  WORDING   new wording, no computation (run_verified_calls disabled in the worker)
  COMPUTE   new wording + computed results (the code as it would ship)
  controls  3 ordinary user notes, new wording vs old wording (does the wording break normal memory answers?)
PREDICTIONS (written before the run):
  HEAD reproduces step 4's ON answers byte for byte (8/8): greedy decoding of identical prompts.
  WORDING: original 4/4; new at most 2/4; stored values for other inputs are still copied in at least 1 answer.
  COMPUTE: 8/8.  Controls: 3/3 under both wordings.
DECISION (written before the run): ship wording + computation if COMPUTE >= 7/8 with original 4/4 and the controls 3/3
under the new wording; otherwise revert both.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import step4_deconfounded_curiosity as S4

L, RSL, CLOVER = S4.L, S4.RSL, S4.CLOVER
import lfm3_layers as LL  # noqa: E402

OLD_HEADER = "Notes from this user's memory, already retrieved for you (answer from them; no tools needed):"
NEW_HEADER = "Notes from this user's memory, already retrieved for you (no tools needed):"
OLD_INSTRUCTION = ("Instruction: Use these notes where they answer the question. They record what the user told you and "
                   "what earlier answers taught (with how often that worked), not verified facts; if one was superseded, use "
                   "the newer one and say so, and do not repeat what did not work.")
NO_COMPUTE = S4.WORKER.replace("import lfm3_layers as LL\n", "import lfm3_layers as LL\nLL.run_verified_calls = lambda *a, **k: []\n", 1)
assert NO_COMPUTE != S4.WORKER
CONTROLS = [
    (["My team's weekly sync is on Tuesdays at 10:30 am in the Banyan room."],
     "When and where is my team's weekly sync?", lambda a: all(w in a.lower() for w in ("tuesday", "10:30", "banyan"))),
    (["My laptop is named atlas-7 and runs Windows 11."], "What is my laptop's name?", lambda a: "atlas-7" in a.lower()),
    (["My manager is Priya.", "Correction: my manager is now Arun; Priya moved to another team."], "Who is my manager now?",
     lambda a: "arun" in a.lower()),
]


def old_wording(ctx):
    assert ctx.count(NEW_HEADER) == 1 and ctx.count(LL.APP_MEMORY_INSTRUCTION) == 1, ctx
    return ctx.replace(NEW_HEADER, OLD_HEADER).replace(LL.APP_MEMORY_INSTRUCTION, OLD_INSTRUCTION)


def run_worker(worker, cfg):
    p = subprocess.run([sys.executable, "-c", worker, str(CLOVER), json.dumps(cfg)], capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    return json.loads(next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("__CLOVER_AB_JSON__:")).split(":", 1)[1])


def main():
    step4 = json.loads((S4.HERE / "results" / "step4_contexts.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="consc-item2-"))
    ctx = {"WORDING": {}, "COMPUTE": {}, "HEAD": {}}
    idle = {}
    for k, files in S4.FILES.items():
        ws = tmp / f"ws_{k}"
        ws.mkdir()
        for name, body in files.items():
            (ws / name).write_text(body, encoding="utf-8")
        base = {"inst_dir": str(tmp / f"inst_{k}"), "ws_dir": str(ws), "ws_name": f"ws_{k}", "curiosity_enabled": True}
        idle[k] = run_worker(S4.WORKER, {**base, "phase": "session1_and_idle",
                                         "initial_question": "How does this module behave across its normal and boundary inputs?"})["idle_actions"]
        evalq = [{"id": i, "question": q} for i, q, _ in S4.QUESTIONS[k]]
        for arm, worker in (("WORDING", NO_COMPUTE), ("COMPUTE", S4.WORKER)):
            got = run_worker(worker, {**base, "phase": "session2_retrieve_after_restart", "eval_questions": evalq})
            for qid, c in got["retrieved_contexts"].items():
                ctx[arm][qid] = c["memory_context"]
    checks = {"idle_actions_equal_step4": all(idle[k] == step4["idle_actions"][f"ON_{k}"] for k in idle)}
    for qid, c in ctx["WORDING"].items():
        ctx["HEAD"][qid] = step4["contexts"]["ON"][qid]
    checks["only_the_wording_differs_from_step4"] = all(old_wording(ctx["WORDING"][q]) == ctx["HEAD"][q] for q in ctx["HEAD"])
    checks["compute_only_adds_lines"] = all(
        "\n".join(ln for ln in ctx["COMPUTE"][q].splitlines() if not ln.startswith("     Computed now")) == ctx["WORDING"][q]
        for q in ctx["COMPUTE"])
    checks["computed_line_count"] = sum(c.count("Computed now") for c in ctx["COMPUTE"].values())
    print(json.dumps(checks), flush=True)
    assert checks["idle_actions_equal_step4"] and checks["only_the_wording_differs_from_step4"] and checks["compute_only_adds_lines"]

    cbase = tmp / "controls"
    controls = []
    for i, (notes, q, chk) in enumerate(CONTROLS):
        kit2 = LL.LayerKit(cbase / str(i), memory_root=cbase / str(i) / "growth" / "knowledge")
        for n in notes:
            kit2.remember(n, conversation_id="setup")
        block, _ = kit2.memory_context(q)
        assert block, f"control C{i + 1} retrieved no note: {q}"
        controls.append((f"C{i + 1}", q, chk, block, old_wording(block)))
    if "--contexts-only" in sys.argv:
        print(json.dumps({c[0]: c[3] for c in controls}, indent=2))
        return

    sandbox = Path(os.environ["TEMP"]) / "clover-consc" / "base"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in ("tools", "data"):
        if not (sandbox / name).exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(sandbox / name), str(Path.home() / "CloverAI" / name)],
                           check=True, capture_output=True)
    srv, rows = None, []
    step4_rows = {r["id"]: r["answer"] for r in json.loads((S4.HERE / "results" / "step4_deconfounded_curiosity.json")
                                                           .read_text(encoding="utf-8"))["rows"] if r["arm"] == "ON"}
    try:
        srv = L.LlamaServer(sandbox, kind="vulkan", ctx=16384, model=sandbox / "data" / "omni" / "gguf" / "LFM3-Omni-2B-Q4_0.gguf",
                            extra_args=("--reasoning", "off"))
        client = RSL.LiveModelClient(endpoint_url=f"http://127.0.0.1:{srv.port}", api_key=srv._key)

        def ask(c, q):
            user = f"{c}\n\nUser Question:\n{q}" if c else q
            t0 = time.perf_counter()
            ans = client.generate([{"role": "system", "content": S4.SYS}, {"role": "user", "content": user}],
                                  max_tokens=320, temperature=0.0, seed=20261109)["content"]
            return ans, round(time.perf_counter() - t0, 1)

        for arm in ("HEAD", "WORDING", "COMPUTE"):
            for k in ("a", "b"):
                for qid, q, chk in S4.QUESTIONS[k]:
                    ans, s = ask(ctx[arm][qid], q)
                    row = {"arm": arm, "id": qid, "passed": bool(chk(ans)), "s": s, "answer": ans[:1200], "context": ctx[arm][qid]}
                    if arm == "HEAD":
                        row["same_as_step4"] = ans[:800] == step4_rows[qid]
                    rows.append(row)
                    print(f"{arm:8} {qid:9} passed={row['passed']}" + (f" same_as_step4={row['same_as_step4']}" if arm == "HEAD" else ""), flush=True)
        for cid, q, chk, new_c, old_c in controls:
            for arm, c in (("CTRL_NEW", new_c), ("CTRL_OLD", old_c)):
                ans, s = ask(c, q)
                rows.append({"arm": arm, "id": cid, "had_context": bool(c), "passed": bool(chk(ans)), "s": s, "answer": ans[:800], "context": c})
                print(f"{arm:8} {cid:9} context={bool(c)!s:5} passed={rows[-1]['passed']}", flush=True)
    finally:
        if srv is not None:
            srv.close()
        for name in ("tools", "data"):
            subprocess.run(["cmd", "/c", "rmdir", str(sandbox / name)], capture_output=True)
    summary = {}
    for arm in ("HEAD", "WORDING", "COMPUTE"):
        got = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"original": f"{sum(r['passed'] for r in got if 'NEW' not in r['id'])}/4",
                        "new": f"{sum(r['passed'] for r in got if 'NEW' in r['id'])}/4"}
    summary["HEAD"]["same_as_step4"] = f"{sum(r.get('same_as_step4', False) for r in rows if r['arm'] == 'HEAD')}/8"
    for arm in ("CTRL_NEW", "CTRL_OLD"):
        got = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"passed": f"{sum(r['passed'] for r in got)}/3", "with_context": sum(r["had_context"] for r in got)}
    out = {"checks": checks, "summary": summary, "rows": rows}
    (S4.HERE / "results" / "item2_verified_notes_live.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
