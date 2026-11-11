"""A repeating answer is stopped with one copy kept, ending at a full sentence (omni_native.trim_loop, 2026-11-11).

Observed on the full system after the 512-token cut was removed: a story that started repeating two lines was stopped
by the repetition guard on "The cat's purr grew lou" (fixed character cut).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import omni_native as O  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


block = ("Elijah closed his eyes. \"Thank you for coming. I'll keep the light burning.\"\n\n"
         "The cat's tail thumped softly against the stone. \"I'll stay. I'll be here until the next storm passes.\"\n\n")
# the stream is stopped part-way into another copy, as the guard sees it
story = ("The keeper climbed the stairs each night and trimmed the wick before the storm came in from the west. " * 3
         + "\n\n" + block * 3 + block[:block.index("thumped") + 4])
check("the story repeats", O.looping(story))
kept = O.trim_loop(story)
check("it ends on a full sentence, not a half word", re.search(r"[.!?][\"')\]]*$", kept) is not None)
check("one copy of the repeated lines is kept", kept.count("I'll be here until the next storm passes.") <= 2
      and len(kept) < len(story))
check("the opening is untouched", kept.startswith("The keeper climbed the stairs"))

para = ("Please provide the exact path to the repository directory so I can proceed with the installation and the "
        "terminal command. ")
looped = "I understand you'd like to install the repository. " + para * 6
kept2 = O.trim_loop(looped)
check("a repeated paragraph keeps one copy ending with its full stop", kept2.endswith("terminal command.")
      and kept2.count("terminal command.") <= 2)

code = "```python\n" + "x = compute(value)  # step one of the loop that repeats\n" * 12
kept3 = O.trim_loop(code)
tail = code[-O.LOOP_TAIL:]
check("inside an unclosed code block the cut stays where it was", kept3 == code[:code.find(tail) + O.LOOP_TAIL])
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
