"""End-to-end live verification of `clover-k3-chat.service` (`http://127.0.0.1:8765/api/chat`)
with 896-expert 93-Layer Kimi K3 MoE + HybridAgent-8B HybridLink + 11-Pass NCM + Unfrozen VQ-VAE Decoder (`e12_clover_k3_true_ha_ncm.pt`)."""
import json, time, urllib.request


def ask(prompt: str, session_id: str = "verify_e12_true"):
    t0 = time.perf_counter()
    body = json.dumps({
        "session_id": session_id,
        "prompt": prompt,
        "remember": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=body,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        r = json.loads(resp.read().decode("utf-8"))
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"USER ({dt:.0f} ms): {prompt!r}")
    print(f"  PLANS : {r.get('plans')}")
    print(f"  REPLY : {r.get('reply')!r}")
    k3e = r.get("k3_experts") or {}
    print(f"  K3 MOE: ms={k3e.get('k3_moe_ms')}, uniq_experts={k3e.get('total_unique_experts_activated')}, tools={len(r.get('tool_calls') or [])}\n")


def main():
    # Reset test session
    req_rst = urllib.request.Request(
        "http://127.0.0.1:8765/api/reset",
        data=json.dumps({"session_id": "verify_e12_true"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    urllib.request.urlopen(req_rst, timeout=15).read()

    prompts = [
        "Hi Clover, how are you?",
        "What is the capital of France?",
        "Explain binary search.",
        "What is the Clover Framework?",
        "Summarize AGENTS.md in simple words.",
        "How does Next-Concept Prediction work?",
        "Compute 19 * 23 + 7",
        "Tell me about how lithium-ion batteries work in two clear sentences.",
    ]
    for p in prompts:
        ask(p)


if __name__ == "__main__":
    main()
