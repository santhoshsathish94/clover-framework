"""One block per turn from a drive.py turns file: what semantics routed, what C1 was offered and given, what ran.
Usage: python report_turns.py <framework-turns.jsonl>"""
import json
import sys

for i, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    r = json.loads(line)
    if "driver_error" in r:
        print(f"--- {i:2} {r['prompt'][:70]!r} DRIVER ERROR {r['driver_error']}")
        continue
    ev = r.get("events") or []
    done = next((e for e in ev if e.get("type") == "done"), {})
    err = next((e for e in ev if e.get("type") == "error"), None)
    lt = done.get("layer_trace") or {}
    l1, l2, l4 = (lt.get(k) or {} for k in ("layer1_lfm3_omni", "layer2_hybrid_agent", "layer4_memory_graph"))
    routed = next((e.get("offered") for e in ev if e.get("type") == "tools"), None)
    calls = [e.get("name") for e in ev if e.get("type") == "tool_call"]
    sem = lt.get("semantics_frame") or {}
    print(f"--- {i:2} [{r.get('seconds')} s] {r['prompt'][:80]!r}")
    print(f"    routed={routed} offered_to_C1={l1.get('tools_offered')} history_msgs={l1.get('history_messages')} "
          f"tool_calls={calls} executed={l2.get('tool_executed')}")
    print(f"    memory={[e.get('id') for e in l4.get('entries') or []]} recall={(l4.get('s2_turn_recall') or {}).get('refs')} "
          f"media_errors={done.get('media_errors')} errors={lt.get('errors')} label={sem.get('label')!r} "
          f"aligned={sem.get('aligned')}")
    print(f"    prefill_ms={l1.get('turn1_prefill_ms')} decode_ms={l1.get('turn1_decode_ms')} "
          f"turn2_ms={(l1.get('turn2_prefill_ms') or 0) + (l1.get('turn2_decode_ms') or 0)}")
    if err:
        print(f"    ERROR: {err.get('error')}")
    print(f"    reply: {(r.get('reply') or '')[:420]!r}")
