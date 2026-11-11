"""Tool calls whose arguments repeat or never close (omni_native._chat and run_turn, 2026-11-11).

Observed on the full system once answers were no longer cut at 512 tokens: for "Which region brought in the most
revenue?" the model's run_python code printed the same line 60 times, three rounds ran to 4,096 tokens (330 s each),
the arguments never closed, and the tool ran with no code ("(no output)"): 1,145 s for one answer.
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


LINE = "print(df.groupby('Region')['Units'] * df.groupby('Region')['Unit price'])\\n"


class Stream:
    def __init__(self, pieces):
        self.pieces, self.sent = pieces, 0

    def chat_stream(self, body, cancel):
        for p in self.pieces:
            self.sent += 1
            yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                                          "function": {"name": "run_python", "arguments": p}}]}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "length"}], "timings": {"predicted_n": 4096}}


eng = object.__new__(omni_native.OmniNative)
eng.stage = {"name": "idle"}
eng.gpu = Stream(['{"name": "run_python", "code": "import pandas as pd\\n'] + [LINE] * 300)
_, calls, st = eng._chat([{"role": "user", "content": "q"}], [{"function": {"name": "run_python"}}], 4096,
                         {"run_python": {}}, tool_choice=None)
check("a repeating tool call is stopped early, not at the token limit", eng.gpu.sent < 40 and st.get("tool_args_looped"))
check("its arguments are kept as not-complete JSON", calls and "_raw" in calls[0]["arguments"])

eng.gpu = Stream(['{"code": "import pandas as pd\\n', "df = pd.read_excel('a.xlsx')\\n",
                  "print(df.groupby('Region')['Units'].sum())\"}"])
_, calls, st = eng._chat([{"role": "user", "content": "q"}], [{"function": {"name": "run_python"}}], 4096,
                         {"run_python": {}}, tool_choice=None)
check("a normal tool call is read whole", calls and "code" in calls[0]["arguments"] and not st.get("tool_args_looped"))


class Live(omni_native.OmniNative):
    replies = []

    def _chat(self, msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, **kw):
        stats = {"prompt_tokens": 10, "reused_tokens": 0, "new_tokens": 5, "prefill_s": 0.01, "decode_s": 0.01, "finish": "stop"}
        if not Live.replies:
            Live.replies.append(msgs[-1])
            return "", [{"id": "call_1", "name": "run_python", "arguments": {"_raw": '{"code": "print(1)\\nprint(1'}}], stats
        Live.replies.append(msgs[-1])
        return "South brought in the most revenue.", [], stats


ran = []
tmp = Path(tempfile.mkdtemp())
kit = lfm3_layers.LayerKit(tmp, memory_root=tmp / "knowledge")
live = object.__new__(Live)
live.lock = threading.Lock()
live.stage = {"name": "idle"}
live.media_dir = tmp
live._extra = {}
live.workdir, live.workname = None, ""
live.execute_tool = lambda name, args, ask=None, allow=(): (ran.append(args) or (True, "(no output)", "allowed"))
res = live.run_turn([], "Use Python on the sheet: which region brought in the most revenue?", tools=True,
                    only_tools=["run_python"], kit=kit, system="SYS")
check("a call with arguments that are not complete JSON is not run", not ran)
check("the model is told why, and answers in the next round",
      any("not complete JSON" in str(r.get("result")) for r in res["tool_calls"])
      and "not complete JSON" in str(Live.replies[-1].get("content")) and res["reply"].startswith("South"))
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
