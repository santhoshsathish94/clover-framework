"""Benchmark 4: HybridAgent Agentic & Multi-Hop QA Benchmark (`oss/HybridAgent/examples/questions.jsonl` + Multi-Hop Science/Math QA).
Evaluates 15 multi-hop & tool-assisted reasoning problems on the live Clover Kimi K3 server (`127.0.0.1:8765/api/chat`)."""
import argparse, json, re, time, urllib.request


AGENTIC_BENCHMARK_ITEMS = [
    # Official HybridAgent examples/questions.jsonl (q1..q5)
    {
        "id": "ha_q1",
        "question": "At the speed of sound in air at 20 degrees Celsius, how many seconds does sound take to travel 10 kilometers? Round to the nearest second.",
        "expected": ["29"],
    },
    {
        "id": "ha_q2",
        "question": "How many minutes does light take to travel from the Sun to the Earth? Round to the nearest minute.",
        "expected": ["8"],
    },
    {
        "id": "ha_q3",
        "question": "How many grams heavier is one liter of whole milk than one liter of pure water at room temperature? Use typical densities and round to the nearest 10 grams.",
        "expected": ["30"],
    },
    {
        "id": "ha_q4",
        "question": "How many trailing zeros does 100! have?",
        "expected": ["24"],
    },
    {
        "id": "ha_q5",
        "question": "If each point of the circle x^2 + y^2 = 16 is reflected in the point (3, 2), the image points satisfy x^2 + y^2 + bx + cy + d = 0. Find the ordered triple (b, c, d).",
        "expected": ["-12", "-8", "36"],
        "require_all": True,
    },
    # Multi-hop & Agentic Reasoning (HotpotQA / Bamboogle / Quantitative Agent tasks)
    {
        "id": "mh_q6",
        "question": "What is the 15th Fibonacci number (where F_1 = 1, F_2 = 1)?",
        "expected": ["610"],
    },
    {
        "id": "mh_q7",
        "question": "What is the greatest common divisor (GCD) of 252 and 105?",
        "expected": ["21"],
    },
    {
        "id": "mh_q8",
        "question": "How many prime numbers are there strictly between 1 and 50?",
        "expected": ["15"],
    },
    {
        "id": "mh_q9",
        "question": "If $5,000 is invested at 6% annual interest compounded annually for 10 years, what is the final balance rounded to two decimal places?",
        "expected": ["8954.24", "8,954.24"],
    },
    {
        "id": "mh_q10",
        "question": "What is the sum of the squares of the first 10 positive integers (1^2 + 2^2 + ... + 10^2)?",
        "expected": ["385"],
    },
    {
        "id": "mh_q11",
        "question": "Which planet in our solar system has the highest surface gravity, and what is its name?",
        "expected": ["jupiter"],
    },
    {
        "id": "mh_q12",
        "question": "What is the chemical formula of glucose, and what is its approximate molar mass in g/mol (rounded to the nearest integer)?",
        "expected": ["180"],
    },
    {
        "id": "mh_q13",
        "question": "A train travels 150 km at 75 km/h and then 180 km at 90 km/h. What is the total travel time in hours?",
        "expected": ["4"],
    },
    {
        "id": "mh_q14",
        "question": "In Kimi K3's 93-layer architecture, how many routed experts exist per MoE layer, and how many top experts are selected per token?",
        "expected": ["896", "16"],
        "require_all": True,
    },
    {
        "id": "mh_q15",
        "question": "In Next-Concept Modeling (NCM), how many hierarchical VQ scales are used to generate a 64-token window in parallel?",
        "expected": ["11"],
    },
]


def check_answer(reply: str, item: dict) -> bool:
    r_low = (reply or "").lower()
    exp_list = [e.lower() for e in item["expected"]]
    if item.get("require_all"):
        return all(e in r_low for e in exp_list)
    for e in exp_list:
        if e.isdigit():
            if re.search(rf"(?<![0-9]){re.escape(e)}(?![0-9])", r_low):
                return True
        elif e in r_low:
            return True
    return False


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
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=os.path.join(here, "benchmarks/hybridagent_qa_baseline.json"))
    args = ap.parse_args()

    correct = 0
    results = []
    t_start = time.perf_counter()

    print(f"=== BENCHMARK 4: HYBRIDAGENT AGENTIC & MULTI-HOP QA (N={len(AGENTIC_BENCHMARK_ITEMS)}) ===", flush=True)
    for idx, it in enumerate(AGENTIC_BENCHMARK_ITEMS):
        t0 = time.perf_counter()
        r = ask_clover(it["question"], sid=f"ha_qa_{idx}")
        ms = (time.perf_counter() - t0) * 1000.0
        reply = r.get("reply", "")
        ok = check_answer(reply, it)
        if ok:
            correct += 1
        tools = [t.get("tool") for t in (r.get("tool_calls") or [])]
        status = "PASS" if ok else "FAIL"
        print(f"[{idx+1:2d}/{len(AGENTIC_BENCHMARK_ITEMS)}] {status} ({ms:.0f} ms, tools={tools}) {it['id']} | exp={it['expected']} | reply={reply[:130]!r}", flush=True)
        results.append({
            "id": it["id"],
            "question": it["question"],
            "expected": it["expected"],
            "ok": ok,
            "ms": round(ms, 1),
            "tools": tools,
            "reply": reply,
        })

    acc = 100.0 * correct / len(AGENTIC_BENCHMARK_ITEMS)
    avg_ms = (time.perf_counter() - t_start) * 1000.0 / len(AGENTIC_BENCHMARK_ITEMS)
    summary = {
        "benchmark": "HybridAgent_MultiHop_QA",
        "n": len(AGENTIC_BENCHMARK_ITEMS),
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "avg_ms": round(avg_ms, 1),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> HYBRIDAGENT MULTI-HOP QA SUMMARY: {correct}/{len(AGENTIC_BENCHMARK_ITEMS)} = {acc:.2f}% | avg {avg_ms:.0f} ms/question", flush=True)


if __name__ == "__main__":
    main()
