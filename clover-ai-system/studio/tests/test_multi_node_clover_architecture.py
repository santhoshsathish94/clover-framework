"""End-to-End Verification for the Multi-Node Clover Architecture (`test_multi_node_clover_architecture.py`).

Verifies all 4 tiers of the Multi-Node Clover Architecture on real precomputed `C1 -> C2 -> C3` datasets:
  1. Every model (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron-3.5-Super-VL 4-bit`, `Kimi-K3.1-Pro`) runs on
     Clover's complete setup of `C1` for inference, `C2` for planning (`plan-trunk-0` + `JEPA`), and `C3` for
     decoding (`bridge-trunk-0` + `11-level VQ-VAE` + `12L NextConceptTransformer` + `3L LexicalPointerDecoder`).
  2. Laptop / phone client (`LFM3-Omni-2B` locally):
     - Answers simple requests locally (`route == 'local_simple'`) and syncs conversation to `k3`.
     - Works offline when `k3_reachable=False` (`route == 'local_offline'`), queues events locally, and flushes
       them to `k3` when back online.
     - Sends larger / multi-step / research requests to `k3` (`route == 'k3_coordinator'`).
     - Sends image / audio / video requests for Nemotron through `k3` (`route == 'k3_to_gx10_nemotron'`).
  3. `k3` CPU server (`Clover-35B-A3B` coordinator + `Kimi K3.1 Pro` deep engine + web/research tools):
     - Splits tasks, coordinates execution across `GX10`, `k3` research tools, and `Kimi K3.1 Pro`, synthesizes
       on `Clover-35B-A3B` (`C1 -> C2 -> C3`), and holds all conversations, memory, and learning on `k3`.
  4. `GX10` GPU server (`Nemotron 3.5 Super VL, 4-bit only`):
     - Executes Vision (32L C-RADIO CUDA), Video (32L C-RADIO + EVS token pruning CUDA), and Audio
       (`AudioMambaBridge` + `Audio-JEPA`) through `4-bit` `C1 -> C2 -> C3` while keeping zero personal data.
  5. `Later: a Kimi machine` readiness gate (`probe_kimi_rental_node`):
     - Tests RAM `> 124 GiB`, free disk `>= 80 GiB`, and `Kimi K3.1 Pro` `C1 -> C2 -> C3` verification before committing.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CLOVER_DIR = HERE.parent / "clover"
for p in (str(CLOVER_DIR), "/opt/clover-k3/web-studio/clover"):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

import multi_node_coordinator as MNC
import omni_c2_c3

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" ({detail})" if detail else ""))


BASE_DIR = Path("/opt/clover-k3/web-studio") if Path("/opt/clover-k3/web-studio").is_dir() else HERE.parent

print("=== 1. Verify C1 -> C2 -> C3 Precomputed Runtime Across Available Models ===")
test_text = "Clover executes C1 for inference, C2 for planning, and C3 for decoding across all models."
expected_specs = [
    ("omni", 128000, 2048),
    ("clover-35b-a3b", 248320, 2048),
    ("nemotron-3.5-super-vl", 131072, 4096),
    ("kimi-k3", 163840, 7168),
]
for mk, exp_v, exp_d in expected_specs:
    rt = omni_c2_c3.for_base(BASE_DIR, model_key=mk)
    if rt is None and mk != "omni":
        print(f"  SKIP  {mk} (not staged on this machine)")
        continue
    check(f"{mk}: runtime loaded", rt is not None)
    if rt is not None:
        check(f"{mk}: vocab_size == {exp_v} & d_model == {exp_d}", rt.vocab_size == exp_v and rt.d_model == exp_d, f"V={rt.vocab_size}, D={rt.d_model}")
        c1_s, c1_i = rt.c1_infer_stage(test_text)
        c2_s, c2_i = rt.c2_plan(test_text, c1_state=c1_s, offered_tools=["web_search", "run_python"], jepa_arch="clover_rpre")
        dec, c3_i = rt.c3_decode(c2_s)
        check(f"{mk}: C1 -> C2 -> C3 exact decode", dec == test_text, f"C1={c1_i['ms']}ms, C2={c2_i['ms']}ms, C3={round(c3_i['s']*1000,1)}ms")

print("\n=== 2. Verify Client Router (Simple Local, Offline Queue & Sync, Large Escalation, Media Routing) ===")
with tempfile.TemporaryDirectory(prefix="clover_mn_test_") as tmp:
    coord = MNC.K3MultiNodeCoordinator(BASE_DIR)
    coord.conversations_dir = Path(tmp) / "k3_convs"
    coord.conversations_dir.mkdir(parents=True, exist_ok=True)
    coord.offline_sync_journal = coord.conversations_dir / "journal.jsonl"

    client = MNC.ClientNodeRouter(BASE_DIR, k3_coordinator=coord)
    client.offline_queue_file = Path(tmp) / "client_offline_queue.json"

    # 2a. Offline turn on client (LFM3-Omni-2B C1 -> C2 -> C3)
    res_off = client.run_client_turn(
        {"session_id": "s_client", "prompt": "What is 6 times 7?", "local_draft": "Six times seven is 42."},
        k3_reachable=False,
        run_c1_stage=False,
    )
    check("Offline turn runs locally on LFM3-Omni-2B", res_off["decision"]["route"] == "local_offline" and res_off["executed_on"] == "client-local")
    check("Offline turn is queued locally for k3 sync", res_off["queued_offline"] is True and len(client._load_offline_queue()) == 1)

    # 2b. Simple online turn on client (flushes offline queue + answers locally on LFM3-Omni-2B + syncs to k3)
    res_simple = client.run_client_turn(
        {"session_id": "s_client", "prompt": "State the capital of France.", "local_draft": "The capital of France is Paris."},
        k3_reachable=True,
        run_c1_stage=False,
    )
    check("Simple online turn runs locally on LFM3-Omni-2B", res_simple["decision"]["route"] == "local_simple" and res_simple["executed_on"] == "client-local")
    check("Simple online turn flushed offline queue to k3", len(client._load_offline_queue()) == 0 and res_simple["synced_to_k3"] is True)

    # 2c. Larger / deep-reasoning / research request escalated to k3 (Clover-35B coordinator + Kimi K3.1 Pro + research tools)
    res_large = client.run_client_turn(
        {
            "session_id": "s_client",
            "prompt": "Research and compare in depth how Clover coordinates multi-step tasks across Clover-35B and Kimi K3.1 Pro.",
            "reasoning_draft": "Clover-35B splits multi-step tasks, runs k3 research tools, and coordinates Kimi K3.1 Pro deep reasoning.",
            "run_c1_stage": False,
        },
        k3_reachable=True,
    )
    check("Larger request routed to k3 coordinator", res_large["decision"]["route"] == "k3_coordinator" and res_large["executed_on"] == "k3")
    step_kinds = [s["kind"] for s in res_large["k3_result"]["executed_steps"]]
    check(
        "k3 coordinator executed research tools, Kimi K3.1 Pro deep reasoning, and Clover-35B synthesis",
        step_kinds == ["web_and_research_tools", "deep_reasoning", "coordinator_synthesis"],
        str(step_kinds),
    )

    # 2d. Image + Video + Audio request routed through k3 to stateless GX10 Nemotron 3.5 Super VL 4-bit worker
    topo = coord.topology_status()
    gx10_live = topo["topology"]["gx10_gpu_server"]["live_status"].get("ok", False)
    if gx10_live:
        img1 = Image.new("RGB", (512, 512), (22, 34, 50))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([40, 40, 470, 130], fill=(40, 120, 200))
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        b64_1 = base64.b64encode(buf1.getvalue()).decode("ascii")

        img2 = img1.copy()
        d2 = ImageDraw.Draw(img2)
        d2.ellipse([190, 210, 330, 350], fill=(225, 95, 50))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        b64_2 = base64.b64encode(buf2.getvalue()).decode("ascii")

        t_arr = np.linspace(0, 0.5, 8000, dtype=np.float32)
        pcm16 = (np.sin(2 * np.pi * 440.0 * t_arr) * 10000).astype(np.int16)
        wbuf = io.BytesIO()
        with wave.open(wbuf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        aud_b64 = base64.b64encode(wbuf.getvalue()).decode("ascii")

        res_media = client.run_client_turn(
            {
                "session_id": "s_client",
                "prompt": "Analyze the attached image, video clip, and audio.",
                "image_base64": b64_1,
                "video_frames_base64": [b64_1, b64_2],
                "audio_base64": aud_b64,
                "run_c1_stage": True,
            },
            k3_reachable=True,
        )
        check("Media request routed through k3 to GX10 Nemotron", res_media["decision"]["route"] == "k3_to_gx10_nemotron")
        gx_step = res_media["k3_result"]["executed_steps"][0]
        check(
            "GX10 executed Vision + Video (EVS) + Audio on 4-bit Nemotron with zero personal data",
            gx_step["node"] == "gx10"
            and gx_step["stateless_zero_personal_data"] is True
            and "vision" in gx_step["modalities"]
            and "video" in gx_step["modalities"]
            and "audio" in gx_step["modalities"],
        )
    else:
        print("  SKIP  Live GX10 multimodal test (GX10 tunnel not active on this host)")

print("\n=== 3. Verify Later Kimi Machine Readiness Gate (probe_kimi_rental_node) ===")
probe_small = MNC.probe_kimi_rental_node(ram_gib=96.0, free_disk_gib=200.0, c1_c2_c3_verified=True)
probe_large = MNC.probe_kimi_rental_node(ram_gib=256.0, free_disk_gib=500.0, c1_c2_c3_verified=True)
check("Rejects rental with <= k3 RAM (96 GiB <= 124 GiB)", probe_small["ready_to_commit"] is False)
check("Approves rental with > k3 RAM (256 GiB > 124 GiB) & sufficient disk & C1/C2/C3 verified", probe_large["ready_to_commit"] is True)

print("\n=== 4. Verify App Model Catalog & Smart Model Switching Across All 4 Models ===")
import k3_host_engine as K3
import server as SV

cat_ids = [x["id"] for x in SV.model_catalog(base=BASE_DIR)]
check(
    "App model_catalog includes Auto (Smart), local LFM3-Omni, and all 4 k3 models",
    "auto" in cat_ids
    and ("omni" in cat_ids)
    and any(x in cat_ids for x in ("k3:clover-35b-a3b", "clover-35b-a3b"))
    and any(x in cat_ids for x in ("k3:nemotron-3.5-super-vl", "nemotron-3.5-super-vl"))
    and any(x in cat_ids for x in ("k3:kimi-k3", "kimi-k3")),
    str(cat_ids),
)
m_simple, _, _ = K3.select_smart_model_for_turn("Write a short polite email to Sarah.")
m_comp, _, _ = K3.select_smart_model_for_turn("Give me a concise 3-bullet comparison of SQLite vs PostgreSQL.")
m_code, _, _ = K3.select_smart_model_for_turn("Write a Python function to compute compound interest and prove its time complexity.")
m_vis, _, _ = K3.select_smart_model_for_turn("What colors are in this image?", media=[{"kind": "image", "file": "a.png"}])
check("Smart switch routes simple everyday question to LFM3-Omni", m_simple == "omni", m_simple)
check("Smart switch routes comparison/coordination to Clover-35B-A3B", m_comp == "k3:clover-35b-a3b", m_comp)
check("Smart switch routes coding/math/proof to Kimi K3.1 Pro", m_code == "k3:kimi-k3", m_code)
check("Smart switch routes image/video/audio to Nemotron 3.5", m_vis == "k3:nemotron-3.5-super-vl", m_vis)

print(f"\nSUMMARY: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
