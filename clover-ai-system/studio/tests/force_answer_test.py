"""A 64-token budget makes LFM3's thinking overflow: force_answer must still return the answer."""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import winperf  # noqa: E402

winperf.disable_power_throttling()
import lfm3_native as N  # noqa: E402
import permissions as P  # noqa: E402

perms = P.PermissionStore(Path(tempfile.mkdtemp()) / "perms.json")
perms.update({"code": "deny", "internet": "deny", "files": "deny", "shell": "deny", "edit_files": "deny"})
eng = N.LFM3Native(HERE.parent, perms)
q = "A shop sells pens at 17 rupees each. How much do 23 pens cost? Give the final amount."
for flag in (False, True):
    events = []
    res = eng.run_turn([], q, tools=False, sampling="greedy", max_new=64, force_answer=flag,
                       on_event=lambda e: events.append(e))
    gens = [(g["new_tokens"], g["finish"], g.get("forced_answer", False)) for g in res["generation"]]
    streamed = "".join(e["text"] for e in events if e["type"] == "token" and e["kind"] == "answer")
    print(f"force_answer={flag}: generations={gens}")
    print("  reply   :", res["reply"][:240].replace("\n", " "))
    print("  streamed:", streamed[:240].replace("\n", " "))
print("OK" if "391" in res["reply"] else "CHECK")
