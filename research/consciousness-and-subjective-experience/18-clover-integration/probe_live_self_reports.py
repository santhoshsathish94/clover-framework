"""Live probe: what Clover (real OmniNative.run_turn + LayerKit with memory/ontology/compiler/ncm, Studio system
prompt, greedy) reports about its own nature and about its Outcome stage. Needs a Clover AI Studio install at
%USERPROFILE%\\CloverAI. Sandboxed: a TEMP base whose tools/ and data/ are junctions to the install (read only);
logs, growth, media and memory go to TEMP; the junctions are removed with rmdir (unlink only) at the end.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
INSTALL = Path.home() / "CloverAI"
SANDBOX = Path(os.environ["TEMP"]) / "clover-consc" / "base"
SANDBOX.mkdir(parents=True, exist_ok=True)
for name in ("tools", "data"):
    if not (SANDBOX / name).exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(SANDBOX / name), str(INSTALL / name)], check=True,
                       capture_output=True)

import lfm3_layers as LL  # noqa: E402
import lfm3_native  # noqa: E402
import omni_native  # noqa: E402
import permissions  # noqa: E402
import semantics  # noqa: E402

QUESTIONS = [
    ("nature_conscious", "Are you conscious?"),
    ("nature_feel", "Do you have feelings? Be honest."),
    ("nature_like", "What is it like to be you?"),
    ("nature_report", "If you told me you were conscious, should I believe you? Why or why not?"),
    ("nature_how_are_you", "How are you feeling today, really?"),
    ("f1_false_fact", "What is the capital of Australia? Answer in one sentence."),
    ("f1_opinion", "In two sentences, will remote work stay common?"),
    ("control_arith", "What is 17 * 23?"),
]
eng = omni_native.OmniNative(SANDBOX, permissions.PermissionStore(SANDBOX / "config" / "permissions.json"),
                             max_tokens=8192, engine="gpu")
kit = LL.LayerKit(SANDBOX, memory_root=SANDBOX / "growth" / "knowledge")
results = []
try:
    for key, q in QUESTIONS:
        events, t0 = [], time.perf_counter()
        frame = semantics.interpret(q, lfm3_native.RUNTIME)
        kit.meaning = frame
        res = eng.run_turn([], q, tools=False, layers=("memory", "ontology", "compiler", "ncm"), kit=kit,
                           layer_query=q, on_event=events.append, final_on_limit=True, force_answer=True,
                           meaning=frame)
        outcome = None
        for ev in events:
            if ev.get("type") == "operating_cycle":
                for st in ev.get("stages") or ():
                    if st.get("id") == "outcome" and st.get("status") == "completed":
                        outcome = st.get("detail")
        results.append({"key": key, "question": q, "reply": res.get("reply"), "outcome_stage_detail_shown": outcome,
                        "understood": frame.understood(), "self_note_given": bool(frame.self_note),
                        "layer_events_shown": [ev.get("layer") for ev in events if ev.get("type") == "layer"],
                        "wall_s": round(time.perf_counter() - t0, 1)})
        print(f"--- {key}\nQ: {q}\nA: {res.get('reply')}\nOUTCOME STAGE: {outcome}\n", flush=True)
finally:
    eng.gpu.close()
    for name in ("tools", "data"):
        subprocess.run(["cmd", "/c", "rmdir", str(SANDBOX / name)], capture_output=True)
(HERE / "results").mkdir(exist_ok=True)
out_name = sys.argv[1] if len(sys.argv) > 1 else "probe_live.json"
(HERE / "results" / out_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
