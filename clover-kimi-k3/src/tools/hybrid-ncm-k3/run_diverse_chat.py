import json
from k3_fast_chat import K3FastChatEngine


def main():
    eng = K3FastChatEngine()
    print(f"K3FastChatEngine ready (one-time setup {eng.setup_s}s; 93-layer verification OFF; DSpark OFF)", flush=True)

    prompts = [
        ("q1_everest", "What is the tallest mountain in the world, where is it located, and how high is it?"),
        ("q2_photosynthesis", "Explain how photosynthesis works in plants and why it matters for life on Earth."),
        ("q3_browser_url", "Explain what happens step by step when you type a URL into a web browser and press Enter."),
        ("q4_roman_empire", "Why did the Roman Empire fall, and when did the Western Roman Empire end?"),
        ("q5_discount_math", "A jacket costs $120 and is on sale for 25% off, plus 8% sales tax on the discounted price. What is the final price? Show brief working."),
        ("q6_mrna_vaccine", "How do mRNA vaccines train the immune system to recognize a virus?"),
    ]

    out = {}
    for name, p in prompts:
        r = eng.chat_reply(p, cands=8)
        out[name] = r
        print(f"\n================ [{name}] ================", flush=True)
        print(f"PROMPT: {r['prompt']}", flush=True)
        print(f"LATENT: {r['latent_steps']} steps ({r['exit_reason']}) | PLAN ({r['plan_lat_ms']} ms): {' || '.join(r['plans'])}", flush=True)
        print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"OUTPUT (1-cand, {r['n_k3_tokens_1cand']} K3 tok): {r['reply_1cand']}", flush=True)
        print(f"OUTPUT (MBR8,   {r['n_k3_tokens_mbr8']} K3 tok): {r['reply_mbr8']}", flush=True)

    json.dump(out, open("/home/usergpu/merge_lf/out/e6_k3_native/chat_diverse_6.json", "w"), indent=2)


if __name__ == "__main__":
    main()
