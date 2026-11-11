"""Recall from other chats inside a running chat (lfm3_layers.other_chats_wanted, recall_context).

Observed 2026-11-11 on the full system: "Summarize what you wrote in one sentence." (after a stopped story) and
"Write me a short summary I can paste into an email to my manager." (after four turns about a sales sheet) were both
answered from another chat's lines, which the recall block shows as "Clover: ...".
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))

import lfm3_layers as L  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    print(("PASS " if cond else "FAIL ") + name)
    bad += not cond


cases = [
    ("Summarize what you wrote in one sentence.", 1, False),
    ("Write me a short summary I can paste into an email to my manager.", 4, False),
    ("Can you shorten the story you just wrote?", 2, False),
    ("Turn your last answer into a table.", 3, False),
    ("Make a bar chart of revenue by region.", 3, False),
    ("What did you say in another chat about my sister?", 2, True),
    ("When is my sister Anjali's birthday?", 0, True),
    ("Write me a short summary I can paste into an email to my manager.", 0, True),
    ("Remind me: how many days is the trip and who is coming with me?", 4, True),
    ("What did I tell you about my dog earlier?", 3, True),
    ("Like last time, keep it short.", 2, True),
]
for text, turns, want in cases:
    check(f"{'recall' if want else 'no other chats'} after {turns} turn(s): {text!r}", L.other_chats_wanted(text, turns) is want)


class Recall:
    want_recall = True

    def __init__(self):
        self.calls = []

    def recall(self, sid, n, q, history_turns=0, force=False):
        self.calls.append(q)
        return "From your earlier conversations ...", {"layer": "recall"}


kit = object.__new__(L.LayerKit)
kit.recall = Recall()
kit.records = type("Records", (), {"current": ("chat", 5)})()
kit.meaning = None
kit._skip = lambda q: False
block, _ = kit.recall_context("Summarize what you wrote in one sentence.", 1)
check("recall_context adds no other chats for a follow-up about this chat", block is None and not kit.recall.calls)
block, _ = kit.recall_context("When is my sister Anjali's birthday?", 0)
check("recall_context still recalls in a new chat", block is not None and kit.recall.calls)
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
