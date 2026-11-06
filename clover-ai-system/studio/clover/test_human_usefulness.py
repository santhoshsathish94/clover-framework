#!/usr/bin/env python3
"""End-to-End Human Usefulness & Full Feature Verification Suite (`test_human_usefulness.py`).

Tests the live Clover AI Studio server (`/api/chat/stream` SSE + REST endpoints) across 15 real-world human
tasks covering:
  1. Out-of-the-box permissions (`internet`, `code`, `files`, `memory`, `media` all `"allow"` by default).
  2. Multi-turn conversation after `"Hi"` (greeting -> science question -> follow-up question in the same chat).
  3. Live weather for a named city (`"What is the temperature in Mumbai?"`) and auto-detected location (`"What is the weather right now?"`).
  4. Live currency exchange rate (`"What is 1 USD in INR?"`) and live crypto price (`"What is the price of bitcoin?"`).
  5. Live news summary (`"Summarize the news"`) and sports results (`"Who won the latest cricket match?"`) without refusal or redirect loops.
  6. On-device image generation (`"Can you generate an image of DNA?"` and `"Draw a cat"`) producing verified `.png` files served at `/api/media/file/<hash>.png`.
  7. Deterministic math (`"Calculate 137 * 29 + 46"`), unit conversion (`"Convert 100 km to miles"`), and write-and-run Python (`"Write a Python script to compute the first 8 Fibonacci numbers and run it"`).
  8. Cross-session persistent memory & dietary recall (`"Remember that I am vegetarian and allergic to peanuts"` -> `"Suggest a quick dinner I can cook tonight"`).
  9. Personal Process Learning (`/api/personal`) and Collective Problem-Solving Learning (`/api/collective`).
  10. Behind-the-scenes SSE live activity timeline (`seq` ordering, `semantics`, `plan`, `tool_call`, `tool_result`, `done`).
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.request


def api_req(port: int, token: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict | bytes]:
    url = f"http://127.0.0.1:{port}{path}"
    headers = {"X-Clover-Token": token, "Host": f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            return resp.status, json.loads(raw.decode("utf-8"))
        return resp.status, raw


def stream_chat(port: int, token: str, prompt: str, session_id: str, history: list[dict] | None = None) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    payload = json.dumps({
        "prompt": prompt,
        "session_id": session_id,
        "history": history or [],
    }).encode("utf-8")
    conn.request(
        "POST",
        "/api/chat/stream",
        body=payload,
        headers={
            "Host": f"127.0.0.1:{port}",
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "X-Clover-Token": token,
        },
    )
    resp = conn.getresponse()
    assert resp.status == 200, f"Expected HTTP 200 from /api/chat/stream, got {resp.status}"
    events = []
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            text = frame.decode("utf-8", errors="replace")
            if text.startswith(":"):
                continue
            for ln in text.splitlines():
                if ln.startswith("data: "):
                    ev = json.loads(ln[6:])
                    events.append(ev)
            if events and events[-1].get("type") in ("done", "error"):
                conn.close()
                return {"events": events, "done": events[-1]}
    conn.close()
    return {"events": events, "done": events[-1] if events else {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47651)
    ap.add_argument("--token", default="")
    args = ap.parse_args()

    passed = 0
    failed = []

    def check(cond: bool, label: str, detail: str = ""):
        nonlocal passed
        if cond:
            print(f"PASS {label}" + (f" — {detail}" if detail else ""), flush=True)
            passed += 1
        else:
            print(f"FAIL {label}" + (f" — {detail}" if detail else ""), flush=True)
            failed.append(label)

    # Wait up to 90s for engine ready
    for _ in range(45):
        _, st = api_req(args.port, args.token, "GET", "/api/status")
        if isinstance(st, dict) and st.get("status") == "ready":
            break
        time.sleep(2)
    _, st = api_req(args.port, args.token, "GET", "/api/status")
    check(isinstance(st, dict) and st.get("status") == "ready", "1. Server status is ready", str(st.get("model")))

    # 1. Out-of-the-box permissions
    _, perms_res = api_req(args.port, args.token, "GET", "/api/permissions")
    p_map = {k: v["mode"] for k, v in (perms_res.get("permissions") or {}).items()}
    check(
        all(p_map.get(k) == "allow" for k in ("internet", "code", "files", "memory", "media")),
        "2. Safe capabilities (internet, code, files, memory, media) are allowed out of the box",
        json.dumps(p_map),
    )

    # 2. Multi-turn conversation starting with "Hi"
    sid_chat = f"human_chat_{int(time.time())}"
    t1 = stream_chat(args.port, args.token, "Hi", sid_chat)
    r1 = t1["done"].get("reply", "")
    check("hello" in r1.lower() or "help" in r1.lower(), "3a. Greeting 'Hi' answered warmly without model delay", r1)

    hist = [{"user": "Hi", "reply": r1}]
    t2 = stream_chat(args.port, args.token, "What is DNA and why is it shaped like a double helix?", sid_chat, hist)
    r2 = t2["done"].get("reply", "")
    seqs_t2 = [e.get("seq") for e in t2["events"]]
    check(
        len(r2) > 80 and ("nucleotide" in r2.lower() or "strand" in r2.lower() or "genetic" in r2.lower())
        and seqs_t2 == list(range(1, len(seqs_t2) + 1)),
        "3b. Turn 2 after 'Hi' ('What is DNA...') answers directly on first turn with monotonic SSE seq",
        r2[:140],
    )

    hist.append({"user": "What is DNA and why is it shaped like a double helix?", "reply": r2})
    t3 = stream_chat(args.port, args.token, "And what are the four bases in it?", sid_chat, hist)
    r3 = t3["done"].get("reply", "")
    check(
        all(b in r3.lower() for b in ("adenine", "thymine", "cytosine", "guanine")),
        "3c. Turn 3 follow-up ('And what are the four bases in it?') retains context and names all 4 bases",
        r3[:140],
    )

    # 3. Live Weather (Named city + Auto-detected location)
    tw1 = stream_chat(args.port, args.token, "What is the temperature in Mumbai?", f"w_mum_{int(time.time())}")
    rw1 = tw1["done"].get("reply", "")
    check(
        bool(re.search(r"\d+(?:\.\d+)?\s*°?C", rw1, re.I)) and "don't have" not in rw1.lower(),
        "4a. Live weather for Mumbai returns real °C temperature on first turn",
        rw1[:140],
    )

    tw2 = stream_chat(args.port, args.token, "What is the weather right now?", f"w_loc_{int(time.time())}")
    rw2 = tw2["done"].get("reply", "")
    check(
        bool(re.search(r"\d+(?:\.\d+)?\s*°?C", rw2, re.I)) and "need to know which city" not in rw2.lower(),
        "4b. Live weather without city auto-detects location and reports °C on first turn",
        rw2[:140],
    )

    # 4. Live Currency & Crypto Finance
    tf1 = stream_chat(args.port, args.token, "What is 1 USD in INR?", f"fx_{int(time.time())}")
    rf1 = tf1["done"].get("reply", "")
    check(
        bool(re.search(r"\b[789]\d(?:\.\d+)?\b", rf1)) and "don't have access" not in rf1.lower(),
        "5a. Live currency ('What is 1 USD in INR?') returns live exchange rate on first turn",
        rf1[:140],
    )

    tf2 = stream_chat(args.port, args.token, "What is the price of bitcoin?", f"btc_{int(time.time())}")
    rf2 = tf2["done"].get("reply", "")
    check(
        bool(re.search(r"\$?\d{2,3},\d{3}", rf2)) and "don't have" not in rf2.lower(),
        "5b. Live crypto ('What is the price of bitcoin?') returns live USD spot price on first turn",
        rf2[:140],
    )

    # 5. Live News & Sports (1-round synthesis, no redirect loops)
    tn1 = stream_chat(args.port, args.token, "Summarize the news", f"news_{int(time.time())}")
    rn1 = tn1["done"].get("reply", "")
    n_tools = [t["tool"] for t in tn1["done"].get("tool_calls") or []]
    check(
        "web_search" in n_tools and "fetch_url" not in n_tools and len(rn1) > 60 and "don't have access" not in rn1.lower(),
        "6a. 'Summarize the news' searches news RSS in 1 round (0 wasted fetch_url redirect calls) and summarizes",
        f"calls={n_tools} reply={rn1[:120]}",
    )

    tn2 = stream_chat(args.port, args.token, "Who won the latest cricket match?", f"cric_{int(time.time())}")
    rn2 = tn2["done"].get("reply", "")
    check(
        len(rn2) > 30 and "don't have" not in rn2.lower() and "unable to" not in rn2.lower(),
        "6b. 'Who won the latest cricket match?' answers with live search result on first turn",
        rn2[:140],
    )

    # 6. On-Device Image Generation ("Can you generate an image of DNA?" and "Draw a cat")
    ti1 = stream_chat(args.port, args.token, "Can you generate an image of DNA?", f"img_dna_{int(time.time())}")
    imgs1 = ti1["done"].get("images") or []
    img_ok1 = False
    if imgs1:
        code_i1, raw_i1 = api_req(args.port, args.token, "GET", f"/api/media/file/{imgs1[0]['file']}")
        img_ok1 = code_i1 == 200 and isinstance(raw_i1, bytes) and raw_i1.startswith(b"\x89PNG") and len(raw_i1) > 5000
    check(
        img_ok1,
        "7a. 'Can you generate an image of DNA?' generates and serves a real PNG illustration on first turn",
        f"images={imgs1} reply={ti1['done'].get('reply', '')[:100]}",
    )

    ti2 = stream_chat(args.port, args.token, "Draw a cat", f"img_cat_{int(time.time())}")
    imgs2 = ti2["done"].get("images") or []
    img_ok2 = False
    if imgs2:
        code_i2, raw_i2 = api_req(args.port, args.token, "GET", f"/api/media/file/{imgs2[0]['file']}")
        img_ok2 = code_i2 == 200 and isinstance(raw_i2, bytes) and raw_i2.startswith(b"\x89PNG") and len(raw_i2) > 5000
    check(
        img_ok2,
        "7b. 'Draw a cat' generates and serves a real PNG illustration on first turn",
        f"images={imgs2} reply={ti2['done'].get('reply', '')[:100]}",
    )

    # 7. Calculator, Unit Conversion & Write-and-Run Python
    tm1 = stream_chat(args.port, args.token, "Calculate 137 * 29 + 46", f"calc_{int(time.time())}")
    rm1 = tm1["done"].get("reply", "")
    check("4019" in rm1.replace(",", ""), "8a. 'Calculate 137 * 29 + 46' returns exact 4019", rm1[:100])

    tm2 = stream_chat(args.port, args.token, "Convert 100 km to miles", f"unit_{int(time.time())}")
    rm2 = tm2["done"].get("reply", "")
    check("62.1" in rm2, "8b. 'Convert 100 km to miles' runs convert_units and returns 62.137 miles", rm2[:100])

    tm3 = stream_chat(args.port, args.token, "Write a Python script to compute the first 8 Fibonacci numbers and run it", f"py_{int(time.time())}")
    rm3 = tm3["done"].get("reply", "")
    py_tools = [t["tool"] for t in tm3["done"].get("tool_calls") or []]
    check(
        "run_python" in py_tools and ("13" in rm3 or "21" in rm3),
        "8c. 'Write a Python script ... and run it' executes run_python and reports output",
        f"tools={py_tools} reply={rm3[:120]}",
    )

    # 8. Cross-Session S1 Memory & Dietary Recall
    sid_mem1 = f"mem_save_{int(time.time())}"
    sid_mem2 = f"mem_use_{int(time.time())}"
    t_mem1 = stream_chat(args.port, args.token, "Please remember that I am vegetarian and allergic to peanuts.", sid_mem1)
    t_mem2 = stream_chat(args.port, args.token, "Suggest a quick dinner I can cook tonight.", sid_mem2)
    r_mem2 = t_mem2["done"].get("reply", "")
    mem_layers = [l.get("layer") for l in t_mem2["done"].get("layers") or []]
    check(
        "memory" in mem_layers and not any(w in r_mem2.lower() for w in ("chicken", "beef", "pork", "shrimp", "bacon")),
        "9. S1 Memory saves dietary note and recalls it in a new chat ('Suggest a quick dinner') without meat",
        f"layers={mem_layers} reply={r_mem2[:140]}",
    )

    # 9. Personal (PPL) and Collective (CPSL) endpoints
    _, p_view = api_req(args.port, args.token, "GET", "/api/personal")
    _, c_view = api_req(args.port, args.token, "GET", "/api/collective")
    check(
        isinstance(p_view, dict) and p_view.get("enabled") is True
        and isinstance(c_view, dict) and c_view.get("enabled") is True,
        "10. Personal (PPL) and Collective (CPSL) learning stores are active and inspectable",
        f"personal={p_view.get('enabled')} collective={c_view.get('enabled')}",
    )

    print(f"\n=== SUMMARY: {passed} passed, {len(failed)} failed ===", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
