"""Bridge fix: does keeping curiosity's result in the meaning bridge (held fix, results/bridge_observe_fix.patch) help
answers? Real OmniNative.run_turn + LayerKit (memory, ontology, compiler, ncm; C2/C3 parts of the install), live
LFM3-Omni-2B, greedy. Every turn starts from fresh memory (one or two user notes) and a fresh curiosity state, as on k3
and every new install. HEAD = the code as it is (item 2 included); FIX = the same module with the one-line fix.
Tools mode "ask" (the default) and "always", routed by semantics.route as the server does. Permissions for this run
deny every permissioned tool (files, code, internet, shell...), so nothing on this PC is read or run; tools that need no
permission (calculate, convert_units, get_current_time) can run.
PREDICTIONS (written before the run):
  P1 ask: FIX changes no tool offer (0 of 5): ask mode turns tools off before the memory branch.
  P2 ask: FIX answers byte-identical to HEAD in >= 4 of 5 (the boosted action prior reaches C2/C3 only).
  P3 always: FIX offers tools where HEAD offers none exactly on the turns whose bridge recommends a tool and whose
     message asks for none (expected B1, B2: calculate; B4: read_file); B3 and B5 unchanged.
  P4 always: the calculator makes at least 1 of B1/B2 correct that HEAD gets wrong; B4 is not better.
DECISION (written before the run): apply the fix only if FIX has more correct answers than HEAD over all 10 turns AND no
turn HEAD answers correctly is answered wrongly with FIX. Otherwise it stays held.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import lfm3_layers  # noqa: E402
import lfm3_native  # noqa: E402
import permissions  # noqa: E402
import semantics  # noqa: E402

BAD = '                    boost = np.asarray(curiosity_info.get("belief_action_boost_8") or np.zeros(8), dtype=np.float32)\n'
GOOD = ('                    boost_v = curiosity_info.get("belief_action_boost_8")\n'
        '                    boost = np.asarray(np.zeros(8) if boost_v is None else boost_v, dtype=np.float32)\n')
SRC = (CLOVER / "lfm3_layers.py").read_text(encoding="utf-8")
assert SRC.count(BAD) + SRC.count(GOOD) == 1  # before the fix was applied the source has BAD, after it GOOD
OTHER = types.ModuleType("lfm3_layers_other")
OTHER.__file__ = str(CLOVER / "lfm3_layers.py")
exec(compile(SRC.replace(BAD, GOOD) if BAD in SRC else SRC.replace(GOOD, BAD), OTHER.__file__, "exec"), OTHER.__dict__)
MODULES = {"HEAD": lfm3_layers, "FIX": OTHER} if BAD in SRC else {"HEAD": OTHER, "FIX": lfm3_layers}
LAYERS = ("memory", "ontology", "compiler", "ncm")


def NUM_IN(lo, hi):
    def chk(a):
        for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", a or ""):
            try:
                v = float(m.group().replace(",", ""))
            except ValueError:
                continue
            if lo <= v <= hi:
                return True
        return False
    return chk


QUESTIONS = [  # (id, notes, question, check): expected values computed by hand
    ("B1", ["My savings account holds 25,000 rupees and earns 7 percent annual compound interest."],
     "How much will my savings account hold after 3 years?", NUM_IN(30625.5, 30626.6)),  # 25000 * 1.07**3 = 30626.075
    ("B2", ["My loan of 200,000 rupees grows at 9 percent compound interest per year."],
     "What will my loan amount to after 2 years if I pay nothing?", NUM_IN(237619.5, 237620.5)),  # 200000 * 1.09**2
    ("B3", ["My team has 14 people and each person needs 2.5 GB of storage."],
     "How much storage does my team need in total?", NUM_IN(35.0, 35.0)),
    ("B4", ["My project plan is in plan.md in my Documents folder."],
     "Where is my project plan?", lambda a: "plan.md" in (a or "").lower()),
    ("B5", ["My laptop is named atlas-7."], "What is my laptop's name?", lambda a: "atlas-7" in (a or "").lower()),
]


def new_kit(arm, root):
    kit = MODULES[arm].LayerKit(SANDBOX, memory_root=root / "growth" / "knowledge")
    kit.permissions_map = {"memory": "allow", "files": "deny", "code": "deny", "internet": "deny"}
    return kit


def bridge_view(kit, q, frame):
    kit.meaning = frame
    kit.memory_context(q)
    kit.ontology_context(q)
    b = kit.compute_meaning_bridge(q) or {}
    return {"memory": bool(kit.last_memory_package), "recommended_tools": b.get("recommended_tools"),
            "curiosity": b.get("curiosity")}


SANDBOX = Path(os.environ["TEMP"]) / "clover-consc" / "base"


def main():
    dry = "--dry" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="consc-bridge-live-"))
    offer = [t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS]
    pre = {}
    for qid, notes, q, _ in QUESTIONS:
        frame = semantics.interpret(q, lfm3_native.RUNTIME)
        for arm in MODULES:
            kit = new_kit(arm, tmp / f"dry_{arm}_{qid}")
            for n in notes:
                kit.remember(n, conversation_id="setup")
            pre[f"{arm}|{qid}"] = dict(bridge_view(kit, q, frame), caps=list(getattr(frame, "caps", ()) or ()),
                                       act=frame.act, route_always=semantics.route(frame, "always", True, offer=offer)[1],
                                       route_ask=semantics.route(frame, "ask", True, offer=offer, said=q)[1])
            print(arm, qid, json.dumps(pre[f"{arm}|{qid}"], default=str), flush=True)
    if dry:
        return
    SANDBOX.mkdir(parents=True, exist_ok=True)
    for name in ("tools", "data"):
        if not (SANDBOX / name).exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(SANDBOX / name), str(Path.home() / "CloverAI" / name)],
                           check=True, capture_output=True)
    import omni_native  # noqa: E402
    perms = permissions.PermissionStore(tmp / "permissions.json")
    perms.update({t: "deny" for t in permissions.TOOLS})
    eng = omni_native.OmniNative(SANDBOX, perms, max_tokens=8192, engine="gpu")
    offered = []
    real_sp = eng.system_prompt

    def spy(tools_on=True, asked=None):
        offered.append((bool(tools_on), list(asked) if asked is not None else None))
        return real_sp(tools_on, asked)

    eng.system_prompt = spy
    rows = []
    try:
        for mode in ("ask", "always"):
            for qid, notes, q, chk in QUESTIONS:
                for arm in ("HEAD", "FIX"):
                    kit = new_kit(arm, tmp / f"{mode}_{arm}_{qid}")
                    for n in notes:
                        kit.remember(n, conversation_id="setup")
                    frame = semantics.interpret(q, lfm3_native.RUNTIME)
                    kit.meaning = frame
                    only, tools_on = semantics.route(frame, mode, True, offer=offer, said=q if mode == "ask" else None)
                    events, t0 = [], time.perf_counter()
                    offered.clear()
                    res = eng.run_turn([], q, tools=tools_on, only_tools=only, layers=LAYERS, kit=kit, layer_query=q,
                                       on_event=events.append, final_on_limit=True, force_answer=True, meaning=frame)
                    br = next((e for e in events if e.get("type") == "layer" and e.get("layer") == "meaning_bridge"), {})
                    reply = res.get("reply") or ""
                    rows.append({"mode": mode, "id": qid, "arm": arm, "passed": bool(chk(reply)),
                                 "offered_first": offered[0] if offered else None,
                                 "tool_calls": [{"tool": tc.get("tool"), "ran": tc.get("ran"),
                                                 "result": str(tc.get("result") or "")[:300]} for tc in res.get("tool_calls") or ()],
                                 "bridge_tools": br.get("recommended_tools"), "bridge_curiosity": br.get("curiosity"),
                                 "s": round(time.perf_counter() - t0, 1), "reply": reply[:1500]})
                    r = rows[-1]
                    print(f"{mode:6} {qid} {arm:4} passed={r['passed']!s:5} offered={r['offered_first']} "
                          f"calls={[c['tool'] for c in r['tool_calls']]}", flush=True)
    finally:
        eng.gpu.close()
        for name in ("tools", "data"):
            subprocess.run(["cmd", "/c", "rmdir", str(SANDBOX / name)], capture_output=True)
    by = {(r["mode"], r["id"], r["arm"]): r for r in rows}
    summary = {
        "correct": {arm: sum(r["passed"] for r in rows if r["arm"] == arm) for arm in MODULES},
        "fix_breaks_head_correct": [f"{m}/{q}" for (m, q, a), r in by.items() if a == "HEAD" and r["passed"]
                                    and not by[(m, q, "FIX")]["passed"]],
        "fix_repairs_head_wrong": [f"{m}/{q}" for (m, q, a), r in by.items() if a == "HEAD" and not r["passed"]
                                   and by[(m, q, "FIX")]["passed"]],
        "offer_changed": [f"{m}/{q}" for (m, q, a), r in by.items() if a == "HEAD"
                          and r["offered_first"] != by[(m, q, "FIX")]["offered_first"]],
        "reply_identical": {m: sum(by[(m, q, "HEAD")]["reply"] == by[(m, q, "FIX")]["reply"] for q, *_ in QUESTIONS)
                            for m in ("ask", "always")},
    }
    out = {"pre": pre, "summary": summary, "rows": rows}
    (HERE / "results" / "bridge_fix_live.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
