import base64, io, json, time, urllib.request
from PIL import Image, ImageDraw, ImageFont


def create_clover_v35_infographic():
    """Creates a faithful high-resolution image matching the 'Clover v3.5 — Released!' infographic from the user's screenshot."""
    im = Image.new("RGB", (820, 520), (245, 250, 245))
    d = ImageDraw.Draw(im)
    f_h1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 30)
    f_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 19)
    f_r = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 17)

    d.rectangle([20, 20, 800, 500], outline=(20, 120, 70), width=3)
    d.text((40, 38), "Clover v3.5 - Released!", font=f_h1, fill=(15, 70, 45))
    d.text((40, 82), "What started with excitement after seeing Fareed Khan's kimi-k3-in-c run Kimi K3 on a CPU", font=f_r, fill=(30, 40, 35))
    d.text((40, 106), "turned into an experiment of our own: measuring, finding bottlenecks, and optimizing the pipeline.", font=f_r, fill=(30, 40, 35))

    # Table header
    d.rectangle([40, 145, 780, 185], fill=(35, 95, 65))
    d.text((55, 155), "Stage", font=f_b, fill=(255, 255, 255))
    d.text((240, 155), "Original (kimi-k3-in-c CPU)", font=f_b, fill=(255, 255, 255))
    d.text((510, 155), "Clover v3.5 (CPU Optimized)", font=f_b, fill=(255, 255, 255))

    # Row 1: Input processing (prefill)
    d.rectangle([40, 185, 780, 255], outline=(180, 200, 185), width=2)
    d.text((55, 205), "Input processing (prefill)", font=f_b, fill=(20, 30, 25))
    d.text((240, 205), "~25 seconds (for 5 input tokens)", font=f_r, fill=(40, 40, 40))
    d.text((510, 205), "~0.3 seconds per input token (1,000 tokens)", font=f_b, fill=(15, 110, 60))

    # Row 2: Output generation (decode)
    d.rectangle([40, 255, 780, 345], outline=(180, 200, 185), width=2)
    d.text((55, 285), "Output generation (decode)", font=f_b, fill=(20, 30, 25))
    d.text((240, 285), "~5 minutes per output token", font=f_r, fill=(40, 40, 40))
    d.text((510, 275), "~3.1 - 3.5 seconds per output token (exact)", font=f_b, fill=(15, 110, 60))
    d.text((510, 305), "~0.9 seconds per output token (DSpark tested)", font=f_b, fill=(15, 110, 60))

    # Four stages footer
    d.text((40, 385), "Four Stages: Stage 1 (Run baseline) -> Stage 2 (Measure bottlenecks) -> Stage 3 (Optimize stage by stage) -> Stage 4 (Complete 34 prompts)", font=f_r, fill=(20, 60, 40))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def post_chat(sid, prompt, image_b64=None):
    data = json.dumps({"session_id": sid, "prompt": prompt, "image_base64": image_b64, "remember": True}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8765/api/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=== 1. MULTIMODAL TEST: 'Clover v3.5 — Released!' INFOGRAPHIC + 'what is this' ===")
    img_clover = create_clover_v35_infographic()
    r_im = post_chat("sess_clover_img", "what is this", image_b64=img_clover)
    print("MODE        :", r_im.get("mode"))
    print("MOONVIT TOK :", r_im.get("option_a_k3_vis_tokens_7168d"))
    print("K3 93-LAYERS:", r_im.get("k3_experts"))
    print("PLANS       :", r_im.get("plans"))
    print("REPLY       :\n" + r_im["reply"])
    print("=" * 70)

    print("\n=== 2. MULTIMODAL TEST: ST MARY'S HOSPITAL INVOICE IMAGE ===")
    img_hosp = "data:image/png;base64," + base64.b64encode(open("/home/usergpu/merge_lf/test_images/img1_hospital_invoice.png", "rb").read()).decode("ascii")
    r_hosp = post_chat("sess_hosp_img", "What is shown in this invoice, what are the items and costs, and who directed the project?", image_b64=img_hosp)
    print("MOONVIT TOK :", r_hosp.get("option_a_k3_vis_tokens_7168d"))
    print("K3 93-LAYERS:", r_hosp.get("k3_experts"))
    print("REPLY       :\n" + r_hosp["reply"])
    print("=" * 70)

    print("\n=== 3. AGENT MODE: LIVE WEB/ARXIV SEARCH + 93-LAYER K3 GPU ENGINE ===")
    r_web = post_chat("sess_web_93", "can you find relevant research on recent AI development")
    print("TOOLS       :", [(t["tool"], t["input"]) for t in r_web.get("tool_calls", [])])
    print("K3 93-LAYERS:", r_web.get("k3_experts"))
    print("REPLY       :\n" + r_web["reply"])
    print("=" * 70)

    print("\n=== 4. AGENT MODE: <python> TOOL + 93-LAYER K3 GPU ENGINE ===")
    r_math = post_chat("sess_math_93", "Two trains are 168 miles apart. The first train leaves Leeds at 09:10 traveling toward London at 84 miles per hour. The second train leaves London at 09:40 traveling toward Leeds at 70 miles per hour. At what time do they meet, and how far from London?")
    print("TOOLS       :", r_math.get("tool_calls"))
    print("K3 93-LAYERS:", r_math.get("k3_experts"))
    print("REPLY       :\n" + r_math["reply"])


if __name__ == "__main__":
    main()
