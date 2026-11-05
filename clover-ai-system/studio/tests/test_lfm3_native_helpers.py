"""Offline checks of lfm3_native's parsing and sampling helpers (no model, no engine).
Usage: python tests/test_lfm3_native_helpers.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import lfm3_native as N  # noqa: E402

SCHEMAS = {t["function"]["name"]: t["function"] for t in N.STUDIO_TOOLS}


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
text = ("\nThe user wants a sum. I'll use Python.\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n"
        "print(sum(range(10)))\n\n</parameter>\n</function>\n</tool_call>")
r, c, a = N.split_completion(text, thinking=True)
calls = N.parse_tool_calls(text.split("</think>", 1)[1], SCHEMAS)
ok &= check("reasoning split", r.strip() == "The user wants a sum. I'll use Python.")
ok &= check("content before call is the separator", c == "\n\n" and a == "")
ok &= check("one call parsed", len(calls) == 1 and calls[0]["name"] == "run_python")
ok &= check("string value keeps its own trailing newline only", calls[0]["arguments"]["code"] == "print(sum(range(10)))\n")

two = ("<tool_call>\n<function=read_file>\n<parameter=path>\nC:\\x.txt\n</parameter>\n<parameter=start_line>\n5\n</parameter>\n"
       "</function>\n</tool_call>\n<tool_call>\n<function=get_current_time>\n</function>\n</tool_call>")
calls = N.parse_tool_calls(two, SCHEMAS)
ok &= check("two calls", [x["name"] for x in calls] == ["read_file", "get_current_time"])
ok &= check("integer coerced", calls[0]["arguments"] == {"path": "C:\\x.txt", "start_line": 5})
ok &= check("no-arg call", calls[1]["arguments"] == {})

r, c, a = N.split_completion("thinking without end", thinking=True)
ok &= check("unfinished thinking -> no content", r == "thinking without end" and c == "" and a == "")
r, c, a = N.split_completion("Plain answer.", thinking=False)
ok &= check("no-thinking answer", r == "" and c == "Plain answer.")

rng = np.random.default_rng(0)
lg = np.zeros(10, dtype=np.float32)
lg[3], lg[7] = 5.0, 4.9
ok &= check("greedy picks max", N.sample_next(lg, [], N.SAMPLING["greedy"], rng, 10) == 3)
ok &= check("repetition penalty can flip the max", N.sample_next(lg, [3], dict(N.SAMPLING["greedy"], repetition_penalty=1.1), rng, 10) == 7)
picks = {N.sample_next(lg, [], N.SAMPLING["general"], rng, 10) for _ in range(200)}
ok &= check("t=0.1 top-k=50 stays on the top two", picks <= {3, 7})
lg2 = np.zeros(12, dtype=np.float32)
lg2[11] = 50.0
ok &= check("rows beyond the tokenizer are never sampled", N.sample_next(lg2, [], N.SAMPLING["greedy"], rng, 11) != 11)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
