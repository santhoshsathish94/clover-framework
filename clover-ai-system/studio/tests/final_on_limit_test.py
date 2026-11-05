"""max_rounds=1 with a question that makes LFM3 call a tool: final_on_limit must still return an answer."""
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
perms.update({"code": "allow", "internet": "deny", "files": "deny", "shell": "deny", "edit_files": "deny"})
eng = N.LFM3Native(HERE.parent, perms)
q = "Use run_python to compute 123456789 * 987654321 and tell me the result."
events = []
for flag in (False, True):
    res = eng.run_turn([], q, sampling="greedy", max_new=768, max_rounds=1, final_on_limit=flag,
                       on_event=lambda e: events.append(e))
    print(f"final_on_limit={flag}: rounds={res['rounds']} tools={[c['tool'] for c in res['tool_calls']]}")
    print("  reply:", res["reply"][:300].replace("\n", " "))
answer_tokens = "".join(e["text"] for e in events if e["type"] == "token" and e["kind"] == "answer" and e["round"] == 2)
print("streamed answer of the extra round:", answer_tokens[:200].replace("\n", " "))
print("OK" if "121932631112635269" in res["reply"].replace(",", "") else "CHECK")
