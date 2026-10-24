"""Benchmark 3: Official OpenAI HumanEval (`HumanEval.jsonl`) Python Code Generation (`pass@1` unit-test execution) on Clover Kimi K3.
Sends the first `N` HumanEval function signatures & docstrings to the live server (`127.0.0.1:8765/api/chat`),
extracts the Python implementation, and runs the official `test` + `check(entry_point)` assertions in a sandboxed subprocess."""
import argparse, json, re, subprocess, sys, time, urllib.request


def extract_python_code(reply: str, prompt_sig: str, entry_point: str) -> str:
    # 1. Extract fenced ```python ... ``` or ``` ... ``` blocks
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        # Pick the block that defines `def <entry_point>` if present, else the longest block
        for b in blocks:
            if f"def {entry_point}" in b:
                return prompt_sig + "\n" + b if f"def {entry_point}" not in b else (
                    "from typing import List, Dict, Tuple, Optional, Any\nimport math, collections, itertools, heapq, functools, re\n" + b
                )
        return "from typing import List, Dict, Tuple, Optional, Any\nimport math, collections, itertools, heapq, functools, re\n" + max(blocks, key=len)
    # 2. If unfenced, look for `def <entry_point>` in reply
    if f"def {entry_point}" in reply:
        idx = reply.find(f"def {entry_point}")
        return "from typing import List, Dict, Tuple, Optional, Any\nimport math, collections, itertools, heapq, functools, re\n" + reply[idx:]
    return prompt_sig + "\n" + reply


def run_humaneval_test(candidate_code: str, test_code: str, entry_point: str) -> tuple[bool, str]:
    full_script = (
        f"{candidate_code}\n\n"
        f"{test_code}\n\n"
        f"check({entry_point})\n"
        f"print('HUMANEVAL_PASS')\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=6.0,
        )
        if res.returncode == 0 and "HUMANEVAL_PASS" in res.stdout:
            return True, "OK"
        err = (res.stderr or res.stdout or "").strip().splitlines()
        return False, (err[-1] if err else f"returncode={res.returncode}")
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)


def ask_clover(question: str, sid: str) -> dict:
    body = json.dumps({
        "session_id": sid,
        "prompt": question,
        "cands": 2,
        "remember": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=body,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=str, default="/home/usergpu/merge_lf/benchmarks/humaneval_baseline_20.json")
    args = ap.parse_args()

    with open("/home/usergpu/merge_lf/benchmarks/HumanEval.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(line) for _, line in zip(range(args.n), f)]

    passed = 0
    results = []
    t_start = time.perf_counter()

    print(f"=== BENCHMARK 3: OFFICIAL OPENAI HUMANEVAL (N={len(items)}) ===", flush=True)
    for idx, it in enumerate(items):
        task_id = it["task_id"]
        entry = it["entry_point"]
        prompt_sig = it["prompt"]
        user_q = (
            f"Write the complete Python function `{entry}` to solve the following specification. "
            f"Include the function signature and wrap your code in a ```python ... ``` block:\n\n{prompt_sig}"
        )
        t0 = time.perf_counter()
        r = ask_clover(user_q, sid=f"humaneval_{idx}")
        ms = (time.perf_counter() - t0) * 1000.0
        reply = r.get("reply", "")
        code = extract_python_code(reply, prompt_sig, entry)
        ok, detail = run_humaneval_test(code, it["test"], entry)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{idx+1:2d}/{len(items)}] {status} ({ms:.0f} ms) {task_id} (`{entry}`) | {detail} | reply={reply[:110]!r}", flush=True)
        results.append({
            "idx": idx,
            "task_id": task_id,
            "entry_point": entry,
            "ok": ok,
            "detail": detail,
            "ms": round(ms, 1),
            "reply": reply,
        })

    acc = 100.0 * passed / max(1, len(items))
    avg_ms = (time.perf_counter() - t_start) * 1000.0 / max(1, len(items))
    summary = {
        "benchmark": "HumanEval",
        "n": len(items),
        "passed": passed,
        "pass_at_1_pct": round(acc, 2),
        "avg_ms": round(avg_ms, 1),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> HUMANEVAL SUMMARY: {passed}/{len(items)} = {acc:.2f}% pass@1 | avg {avg_ms:.0f} ms/problem", flush=True)


if __name__ == "__main__":
    main()
