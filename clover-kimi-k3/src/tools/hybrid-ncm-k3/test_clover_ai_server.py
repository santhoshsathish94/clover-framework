import base64, concurrent.futures, json, time, urllib.request


def post_chat(session_id, prompt, image_b64=None):
    data = json.dumps({
        "session_id": session_id,
        "prompt": prompt,
        "image_base64": image_b64,
        "remember": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # 1. Fire 3 concurrent requests from 3 DIFFERENT sessions within the 1.0s pause window
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(post_chat, "sess_alpha", "the captial of france is")
        f2 = ex.submit(post_chat, "sess_beta", "What is the tallest mountain in the world, where is it located, and how high is it?")
        f3 = ex.submit(post_chat, "sess_gamma", "Explain in about 90 words how a lithium-ion battery works during charging and discharging, and why it degrades over time.")
        r1, r2, r3 = f1.result(), f2.result(), f3.result()
    dt_3 = time.perf_counter() - t0
    print(f"3 concurrent sessions finished in {dt_3:.2f}s (clubbed_batch_size={r1.get('clubbed_batch_size')}, gpu_pair={r1.get('gpu_pair')})", flush=True)
    print("  sess_alpha:", r1["reply"][:100], flush=True)
    print("  sess_beta :", r2["reply"][:100], flush=True)
    print("  sess_gamma:", r3["reply"][:100], flush=True)

    # 2. Follow-up on sess_alpha (testing isolated multi-turn memory for sess_alpha without mixing sess_beta/gamma)
    r1_turn2 = post_chat("sess_alpha", "What famous iron tower is located there, and who designed it?")
    print("  sess_alpha turn 2 (history_len=", r1_turn2["history_len"], "):", r1_turn2["reply"][:130], flush=True)

    # 3. Test image upload on sess_img
    img_b64 = "data:image/png;base64," + base64.b64encode(open("/home/usergpu/merge_lf/test_images/img1_hospital_invoice.png", "rb").read()).decode("ascii")
    r_img = post_chat("sess_img", "Write a news report of about 80 words using the facts shown in this hospital invoice image, and end with a Summary.", image_b64=img_b64)
    print("  sess_img (gpu_pair=", r_img.get("gpu_pair"), "):", r_img["reply"][:140], flush=True)


if __name__ == "__main__":
    main()
