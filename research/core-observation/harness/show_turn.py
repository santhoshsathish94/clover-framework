"""Prints a driver turn record (prompt, reply, events) for turns whose prompt contains a phrase.
Usage: python show_turn.py <turns.jsonl> <phrase>"""
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if sys.argv[2].lower() in str(r.get("prompt", "")).lower():
        print(json.dumps({k: r.get(k) for k in ("prompt", "seconds", "reply", "event_types", "driver_error")},
                         ensure_ascii=False)[:900])
        for e in r.get("events") or []:
            print("  ", json.dumps(e, ensure_ascii=False, default=str)[:700])
