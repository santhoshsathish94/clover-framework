"""Benchmark 1: Official GSM8K (`openai/grade-school-math` test split) on Clover Kimi K3 (93-Layer MoE + HybridAgent + NCM).
Runs the first `N` official GSM8K test problems against the live server (`127.0.0.1:8765/api/chat`),
compares the extracted final numerical answer against the ground-truth `#### <answer>`, and logs all failures for diagnosis."""
import argparse, json, re, time, urllib.request


def extract_ground_truth(ans_str: str) -> float:
    m = re.search(r"####\s*([\-0-9,\.]+)", ans_str)
    if not m:
        raise ValueError(f"No #### in {ans_str!r}")
    return float(m.group(1).replace(",", ""))


def extract_predicted_number(reply: str) -> float | None:
    if not reply:
        return None
    # 1. Explicit '#### <num>' or 'Final Answer: <num>' or 'equals <num>' or 'is $<num>'
    for pat in (
        r"####\s*\$?\s*([\-0-9,]+(?:\.[0-9]+)?)",
        r"(?:final answer|answer is|total is|equals|makes|gets|pays|has|takes|spend|spends|earned|earns|left with)\s*(?:is\s*)?\$?\s*([\-0-9,]+(?:\.[0-9]+)?)",
        r"\$\s*([0-9,]+(?:\.[0-9]+)?)",
    ):
        matches = re.findall(pat, reply, flags=re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                pass
    # Fallback: last number in the response (excluding 'W1', 'W2', etc.)
    cleaned = re.sub(r"\bW\d+\b", "", reply)
    nums = re.findall(r"(?<![A-Za-z])[\-]?\d[\d,]*(?:\.\d+)?", cleaned)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def ask_clover(question: str, sid: str) -> dict:
    import os
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
    timeout_s = int(os.environ.get("CLOVER_BENCH_TIMEOUT", "600"))
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", type=str, default=os.path.join(here, "benchmarks/gsm8k_results.json"))
    args = ap.parse_args()

    in_path = os.path.join(here, "benchmarks/gsm8k_test.jsonl")
    if not os.path.exists(in_path):
        in_path = "/home/usergpu/merge_lf/benchmarks/gsm8k_test.jsonl"
    with open(in_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for _, line in zip(range(args.n), f)]

    correct = 0
    results = []
    t_start = time.perf_counter()

    print(f"=== BENCHMARK 1: OFFICIAL GSM8K (N={len(items)}) ===", flush=True)
    for idx, item in enumerate(items):
        q = item["question"]
        gt = extract_ground_truth(item["answer"])
        t0 = time.perf_counter()
        r = ask_clover(q, sid=f"gsm8k_{idx}")
        ms = (time.perf_counter() - t0) * 1000.0
        reply = r.get("reply", "")
        pred = extract_predicted_number(reply)
        ok = (pred is not None and abs(pred - gt) < 1e-2)
        if ok:
            correct += 1
        tools_used = [t.get("tool") for t in (r.get("tool_calls") or [])]
        status = "PASS" if ok else "FAIL"
        print(f"[{idx+1:2d}/{len(items)}] {status} ({ms:.0f} ms, tools={tools_used}) | gt={gt:g} | pred={pred} | reply={reply[:140]!r}", flush=True)
        results.append({
            "idx": idx,
            "question": q,
            "gt": gt,
            "pred": pred,
            "ok": ok,
            "ms": round(ms, 1),
            "tools": tools_used,
            "reply": reply,
        })

    acc = 100.0 * correct / max(1, len(items))
    avg_ms = (time.perf_counter() - t_start) * 1000.0 / max(1, len(items))
    summary = {
        "benchmark": "GSM8K",
        "n": len(items),
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "avg_ms": round(avg_ms, 1),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> GSM8K SUMMARY: {correct}/{len(items)} = {acc:.2f}% | avg {avg_ms:.0f} ms/problem", flush=True)


if __name__ == "__main__":
    main()
