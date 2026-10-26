#!/usr/bin/env python3
"""
Live Multi-Turn Natural Conversation Verification on Clover (`test_natural_conversation_live.py`)
=================================================================================================
Replays the exact multi-turn conversational sequences where issues were observed in `sessions_store.json`:
  1. Session A (Clover adoption & follow-ups):
     - "What is clover"
     - "That seems very nice does it really work"
     - "How can I do that"
     - "Can you read the framework and tell me how to use it"
  2. Session B (Rocket in space SVG illustration & confirmation):
     - "hi clover how are you"
     - "can you generate a rocket in space image"
     - "I never asked for a futuristic city"
  3. Session C (Postpartum belly fat & anaphoric 'Why' after 'Thanku'):
     - "I want to reduce my postpartum belly fat can you assist me"
     - "Can i tie my belly will it work reducing my belly fat"
     - "Thanku"
     - "Why"
  4. Session D (New GPUs vs RTX 3090 hardware reality):
     - "any new gpus launched"
     - "but your running in 3090 and it's very capable"
"""

import json
import re
import time
import urllib.request


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_chat(sid: str, prompt: str):
    data = json.dumps({"session_id": sid, "prompt": prompt, "cands": 4, "remember": True}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8765/api/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready():
    for _ in range(45):
        try:
            h = get_json("http://127.0.0.1:8765/api/health")
            if h.get("status") == "ready":
                return h
        except Exception:
            time.sleep(2.5)
    raise RuntimeError("Server did not become ready in time")


def main():
    wait_ready()
    ts = int(time.time())

    conversations = [
        (
            f"nat_clover_{ts}",
            [
                ("What is clover", lambda r: "Context" in r["reply"] and "Direction" in r["reply"]),
                ("That seems very nice does it really work", lambda r: len(r["reply"]) > 40 and "[MEM-" not in r["reply"]),
                ("How can I do that", lambda r: ("AGENTS.md" in r["reply"] or "Context" in r["reply"]) and "I'll assume" not in r["reply"]),
                ("Can you read the framework and tell me how to use it", lambda r: ("AGENTS.md" in r["reply"] or "Step 1" in r["reply"]) and "!,!," not in r["reply"]),
            ],
        ),
        (
            f"nat_rocket_{ts}",
            [
                ("hi clover how are you", lambda r: len(r["reply"]) > 10),
                ("can you generate a rocket in space image", lambda r: bool(r.get("media")) and "Rocket" in r["media"][0].get("title", "")),
                ("I never asked for a futuristic city", lambda r: "Sorry about that" in r["reply"] and not r.get("media")),
            ],
        ),
        (
            f"nat_health_{ts}",
            [
                ("Can i tie my belly will it work reducing my belly fat", lambda r: "fat" in r["reply"].lower()),
                ("Thanku", lambda r: "welcome" in r["reply"].lower() or "glad" in r["reply"].lower()),
                (
                    "Why",
                    lambda r: "clarify what you're looking for" not in r["reply"].lower()
                    and "my response should" not in r["reply"].lower()
                    and "i'll focus on" not in r["reply"].lower()
                    and any(w in r["reply"].lower() for w in ("fat", "calories", "muscle", "compression", "support", "tissue")),
                ),
            ],
        ),
        (
            f"nat_gpu_{ts}",
            [
                ("any new gpus launched", lambda r: "Blackwell" in r["reply"] or "MI350X" in r["reply"]),
                ("but your running in 3090 and it's very capable", lambda r: "Ampere" in r["reply"] and "148.5" in r["reply"] and "Turing" not in r["reply"]),
            ],
        ),
    ]

    passed = 0
    total = 0
    for sid, turns in conversations:
        print(f"\n=== Conversation {sid} ===")
        for u_prompt, check_fn in turns:
            total += 1
            t0 = time.perf_counter()
            res = post_chat(sid, u_prompt)
            dt = (time.perf_counter() - t0) * 1000.0
            ok = bool(check_fn(res))
            # Check no mid-sentence paragraph split (e.g. 'which\n\nIs')
            has_mid_split = bool(re.search(r"\b(?:which|that|and|or|as well|reliability)\n\n[A-Z]", res.get("reply", "")))
            if ok and not has_mid_split:
                passed += 1
                status = "PASS"
            else:
                status = f"FAIL(ok={ok}, mid_split={has_mid_split})"
            media_cnt = len(res.get("media") or [])
            reply_one = res.get("reply", "").replace("\n", " | ")[:220]
            print(f"  [{status}] ({dt:6.1f}ms, media={media_cnt}) U: {u_prompt!r}\n           A: {reply_one}")

    print(f"\n>>> NATURAL CONVERSATION SUMMARY: {passed}/{total} turns passed ({100.0 * passed / max(1, total):.1f}%)")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
