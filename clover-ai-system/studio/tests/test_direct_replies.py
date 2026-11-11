"""Notes to rewrite go to the model; the S3 replies that are already complete stay direct (2026-11-11).

Observed on the full system: "Rewrite these notes into one clear sentence: meeting moved; Friday 3 pm; room 204; bring
the budget draft" came back as "Meeting moved, Friday 3 pm, room 204, bring the budget draft." in 0.3 s, written by
no model (the notes joined with commas).
"""
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import lfm3_layers  # noqa: E402
import omni_native  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


NOTES = "Rewrite these notes into one clear sentence:\nmeeting moved; Friday 3 pm; room 204; bring the budget draft"
check("notes are not answered without the model",
      omni_native._direct_s3_or_concept_plan(NOTES, None, [], []) is None)
time_call = {"name": "get_current_time", "arguments": {}}
check("the clock tool's answer stays direct",
      omni_native._direct_s3_or_concept_plan("What time is it?", time_call, [],
                                             [{"ran": True, "worked": True, "result": "Wednesday 11 November 2026, 10:12"}])
      == "The current local time is Wednesday 11 November 2026, 10:12.")
note_call = {"name": "remember", "arguments": {"text": "I am vegetarian"}}
check("a saved note's answer stays direct",
      omni_native._direct_s3_or_concept_plan("Remember that I am vegetarian", note_call, [],
                                             [{"ran": True, "worked": True, "result": "Saved"}])
      == "Noted and saved to your memory: I am vegetarian")


class LiveOmni(omni_native.OmniNative):
    asked = []

    def _chat(self, msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
        LiveOmni.asked.append(msgs[-1]["content"])
        return "The meeting has moved to room 204 at 3 pm on Friday, so please bring the budget draft.", [], {
            "prompt_tokens": 40, "reused_tokens": 0, "new_tokens": 21, "prefill_s": 0.01, "decode_s": 0.01, "finish": "stop"}


tmp = Path(tempfile.mkdtemp())
kit = lfm3_layers.LayerKit(tmp, memory_root=tmp / "knowledge")
eng = object.__new__(LiveOmni)
eng.lock = threading.Lock()
eng.stage = {"name": "idle"}
eng.media_dir = tmp
eng._extra = {}
eng.workdir, eng.workname = None, ""
eng.gpu = object()
res = eng.run_turn([], NOTES, tools=False, layers=("ncm",), kit=kit, system="SYS")
check("on the live Omni path the model writes the sentence", len(LiveOmni.asked) == 1
      and res["generation"][0].get("new_tokens") == 21 and res["generation"][0].get("c1") != "inference only")
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
