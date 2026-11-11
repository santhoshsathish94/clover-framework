"""Answers are not cut at a fixed token count (the 512/320/128 budget of 59f56d33 removed, 2026-11-11).

Observed on the full system: a 600-word story stopped mid-sentence at 512 tokens, and six Python-writing rounds of a
spreadsheet question stopped at 512 tokens with code that printed nothing. C3 keeps answers of 845-1238 tokens intact,
so the model gets the full allowance again: the live Omni path (C3 on, as by default) and the Fused engine.
"""
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import fused_engine  # noqa: E402
import lfm3_layers  # noqa: E402
import omni_native  # noqa: E402
import permissions  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


class LiveOmni(omni_native.OmniNative):
    """The live path: a GPU engine whose _chat is the class's own (no instance override)."""
    seen = []

    def _chat(self, msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
        LiveOmni.seen.append(max_new)
        return "Once upon a time a keeper climbed the stairs.", [], {
            "prompt_tokens": 20, "reused_tokens": 0, "new_tokens": 9, "prefill_s": 0.01, "decode_s": 0.01, "finish": "stop"}


tmp = Path(tempfile.mkdtemp())
kit = lfm3_layers.LayerKit(tmp, memory_root=tmp / "knowledge")
eng = object.__new__(LiveOmni)
eng.lock = threading.Lock()
eng.stage = {"name": "idle"}
eng.media_dir = tmp
eng._extra = {}
eng.workdir, eng.workname = None, ""
eng.gpu = object()
for prompt in ("Write a 600-word short story about a lighthouse keeper.", "Summarize the plot in one sentence.",
               "Explain in detail how a four-stroke engine works."):
    LiveOmni.seen.clear()
    eng.run_turn([], prompt, tools=False, layers=("ncm",), kit=kit, system="SYS", max_new=2048)
    check(f"Omni (live, C3 on) gives the model the full 2048 tokens: {prompt!r}", LiveOmni.seen[:1] == [2048])

asked = []
fus = object.__new__(fused_engine.FusedEngine)
fus.base_dir = tmp
fus.perms = permissions.PermissionStore(tmp / "perms.json")
fus.layer = 3
fus.turn_images = []
fus.c1_inference_only = False
fus._ask = lambda prompt, hist, max_new: (asked.append(max_new) or {"reply": "A keeper climbed the stairs.",
                                                                       "input_tokens": 10, "generated_tokens": 6,
                                                                       "seconds": 0.1})
fus.run_turn([], "Write a 600-word short story about a lighthouse keeper.", tools=False, layers=("ncm",), kit=kit,
             max_new=2048)
check("Fused engine gives the model the full 2048 tokens", asked[:1] == [2048])
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
