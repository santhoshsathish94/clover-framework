"""Benchmark 2: Official MMLU (Massive Multitask Language Understanding, Hendrycks et al.) on Clover Kimi K3.
Evaluates multiple-choice questions across 5 diverse subjects (`college_computer_science`, `college_biology`,
`college_physics`, `astronomy`, `high_school_us_history`) against the live server (`127.0.0.1:8765/api/chat`)."""
import argparse, csv, glob, json, os, re, time, urllib.request


SUBJECTS = [
    "college_computer_science",
    "college_biology",
    "college_physics",
    "astronomy",
    "high_school_us_history",
]


def extract_choice(reply: str, choices: list[str]) -> str | None:
    if not reply:
        return None
    # 1. Explicit 'Answer: A', '(A)', 'option A', 'is B.'
    for pat in (
        r"(?:final answer|correct answer|correct option|answer is|answer:|choice is)\s*\(?([A-D])\)?(?:\b|\.)",
        r"^\s*The correct answer is\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?[\.\:\)]",
        r"\b([A-D])\s+is\s+correct",
    ):
        m = re.findall(pat, reply, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m[0].upper()
    # 2. Match exact option text in the reply
    r_low = reply.lower()
    matched_letters = []
    for idx, ch_txt in enumerate(choices):
        c_clean = ch_txt.strip().lower()
        if len(c_clean) >= 2 and c_clean in r_low:
            matched_letters.append("ABCD"[idx])
    if len(matched_letters) == 1:
        return matched_letters[0]
    # 3. Standalone A/B/C/D token near the beginning or end
    m2 = re.findall(r"\b([A-D])\b", reply)
    if m2:
        return m2[0]
    return None


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
    timeout_s = int(os.environ.get("CLOVER_BENCH_TIMEOUT", "600"))
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=4)
    ap.add_argument("--out", type=str, default=os.path.join(here, "benchmarks/mmlu_baseline_20.json"))
    args = ap.parse_args()

    base_dir = os.path.join(here, "benchmarks/data/test")
    if not os.path.isdir(base_dir):
        base_dir = "/home/usergpu/merge_lf/benchmarks/data/test"
    items = []
    for subj in SUBJECTS:
        p = os.path.join(base_dir, f"{subj}_test.csv")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row, _ in zip(reader, range(args.per_subject)):
                if len(row) >= 6:
                    items.append({
                        "subject": subj,
                        "question": row[0],
                        "choices": [row[1], row[2], row[3], row[4]],
                        "gt": row[5].strip().upper(),
                    })

    correct = 0
    results = []
    t_start = time.perf_counter()

    print(f"=== BENCHMARK 2: OFFICIAL MMLU ({len(SUBJECTS)} subjects, N={len(items)}) ===", flush=True)
    for idx, it in enumerate(items):
        ch = it["choices"]
        prompt = (
            f"Question ({it['subject'].replace('_', ' ')}): {it['question']}\n"
            f"(A) {ch[0]}\n(B) {ch[1]}\n(C) {ch[2]}\n(D) {ch[3]}\n"
            f"State the correct option letter (A, B, C, or D) and a brief explanation."
        )
        t0 = time.perf_counter()
        r = ask_clover(prompt, sid=f"mmlu_{idx}")
        ms = (time.perf_counter() - t0) * 1000.0
        reply = r.get("reply", "")
        pred = extract_choice(reply, ch)
        ok = (pred == it["gt"])
        if ok:
            correct += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{idx+1:2d}/{len(items)}] {status} ({ms:.0f} ms) [{it['subject']}] | gt={it['gt']} | pred={pred} | reply={reply[:130]!r}", flush=True)
        results.append({
            "idx": idx,
            "subject": it["subject"],
            "question": it["question"],
            "choices": ch,
            "gt": it["gt"],
            "pred": pred,
            "ok": ok,
            "ms": round(ms, 1),
            "reply": reply,
        })

    acc = 100.0 * correct / max(1, len(items))
    avg_ms = (time.perf_counter() - t_start) * 1000.0 / max(1, len(items))
    summary = {
        "benchmark": "MMLU",
        "n": len(items),
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "avg_ms": round(avg_ms, 1),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> MMLU SUMMARY: {correct}/{len(items)} = {acc:.2f}% | avg {avg_ms:.0f} ms/question", flush=True)


if __name__ == "__main__":
    main()
