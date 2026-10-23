import concurrent.futures, json, time, urllib.request


def ask(sid, msg):
    data = json.dumps({"session_id": sid, "prompt": msg, "remember": True}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8765/api/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=== 1. CLUBBED 3-SESSION PARALLEL BATCH IN AGENT MODE + K3 GPU EXPERTS ===")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(
            ask,
            "sess_math",
            "Two trains are 168 miles apart. The first train leaves Leeds at 09:10 traveling toward London at 84 miles per hour. The second train leaves London at 09:40 traveling toward Leeds at 70 miles per hour. At what time do they meet, and how far from London?",
        )
        f2 = ex.submit(ask, "sess_geo", "the captial of france is")
        f3 = ex.submit(ask, "sess_sci", "Explain how photosynthesis works in plants and why it matters for life on Earth.")
        r1, r2, r3 = f1.result(), f2.result(), f3.result()
    print(f"Clubbed 3 sessions finished in {time.perf_counter() - t0:.2f}s | clubbed_batch_size={r1['clubbed_batch_size']} | k3_experts={r1['k3_experts']}")
    print("  [sess_math tools]:", r1["tool_calls"])
    print("  [sess_math reply]:", r1["reply"])
    print("  [sess_geo  reply]:", r2["reply"])
    print("  [sess_sci  reply]:", r3["reply"][:190] + "...")

    print("\n=== 2. 3-TURN TNPSC + 2-MONTH-OLD BABY TIMETABLE IN LIVE AGENT MODE ===")
    for m in [
        "I am preparing for TNPSC exams. Where should I start?",
        "thank you",
        "can you schedule a time table for my preparation i have a 2months old baby and infant i can prepare only for 2 hrs",
    ]:
        r = ask("sess_tnpsc_agent", m)
        print("USER      :", m)
        print("K3 EXPERTS:", r["k3_experts"])
        print("REPLY     :\n" + r["reply"], "\n" + "-" * 60)


if __name__ == "__main__":
    main()
