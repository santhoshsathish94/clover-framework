#!/usr/bin/env python3
"""
End-to-End Verification Suite for Model-Specific Multimodal Features across
`nemotron-3.5-super-vl`, `lfm3-1.7b`, and `kimi-k3`
(`clover-ai-system/src/models/verify_multimodal_all_models.py`).

Tests:
  1. `/api/health`, `/api/models/capabilities`, and `/v1/models`
  2. `/api/tts` neural voice synthesis per model (`en-US-AriaNeural`, `en-US-JennyNeural`, `en-US-AnaNeural`)
  3. `nemotron-3.5-super-vl` (`131,072` Tekken vocab):
     - Image Vision (`C-RADIO` + `PixelShuffle(0.5)` + `ReLU^2` `mlp1` -> `4096`-d -> NCM)
     - Video Understanding (`video_patch_projection` + `EVS` token pruning -> `4096`-d -> NCM)
     - Audio / Voice Conversation (`80`-band Mel -> `NemotronAudioMambaBridge` `4096`-d -> NCM)
     - Image Generation (`SVG` + `PNG` + `131,072`-vocab NCM)
     - Video Generation (`Animated SVG` + `12-frame GIF` + `131,072`-vocab NCM)
     - Text + Python calculation (`HybridAgent` + `NemotronNativeCPUEngine` + `131,072`-vocab NCM)
  4. `lfm3-1.7b` (`128,000` tied vocab):
     - Image Vision (`Lfm3ShortConvVisionVideoAdapter` width-7 `ShortConv` -> `2048`-d -> NCM)
     - Video Understanding (`Lfm3ShortConvVisionVideoAdapter` temporal pruning -> `2048`-d -> NCM)
     - Duplex Audio / Voice Conversation (`Lfm3ShortConvAudioVoiceAdapter` `80 -> 2048` -> NCM)
     - Image Generation (`SVG` + `PNG` + `128,000`-vocab NCM)
     - Video Generation (`Animated SVG` + `12-frame GIF` + `128,000`-vocab NCM)
     - Text + Python calculation (`HybridAgent` + `Lfm3NativeCPUEngine` + `128,000`-vocab NCM)
  5. `kimi-k3` (`163,840` vocab):
     - Image Generation (`SVG` scene + `163,840`-vocab NCM)
     - Text chat (`93`-layer Kimi K3 + NCM)
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
import wave
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:8765"


def get_json(path: str) -> dict[str, Any]:
    req = urllib.request.Request(BASE_URL + path, headers={"Host": "127.0.0.1:8765"}, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_test_media_payloads() -> tuple[str, str, str]:
    img = Image.new("RGB", (512, 512), (24, 140, 110))
    dr = ImageDraw.Draw(img)
    dr.ellipse([120, 120, 380, 380], fill=(250, 220, 90))
    buf_i = io.BytesIO()
    img.save(buf_i, format="PNG")
    img_b64 = "data:image/png;base64," + base64.b64encode(buf_i.getvalue()).decode("ascii")

    vid_sheet = Image.new("RGB", (720, 420), (16, 24, 32))
    vdr = ImageDraw.Draw(vid_sheet)
    for f_i, col in enumerate([(30, 180, 130), (60, 140, 220), (230, 110, 60)]):
        vdr.rectangle([f_i * 240 + 20, 50 + f_i * 30, f_i * 240 + 200, 320], fill=col)
    buf_v = io.BytesIO()
    vid_sheet.save(buf_v, format="PNG")
    vid_b64 = "data:image/png;base64," + base64.b64encode(buf_v.getvalue()).decode("ascii")

    sr = 16000
    t_arr = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    pcm = (np.sin(2.0 * np.pi * 240.0 * t_arr) * 12000).astype(np.int16)
    buf_w = io.BytesIO()
    with wave.open(buf_w, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    wav_b64 = "data:audio/wav;base64," + base64.b64encode(buf_w.getvalue()).decode("ascii")
    return img_b64, vid_b64, wav_b64


def main() -> None:
    report: dict[str, Any] = {"checks": []}

    # 1. Health & capabilities
    h = get_json("/api/health")
    assert h["status"] == "ready", h
    assert h["max_session_context_tokens"] == 1048576, h
    print("1. /api/health OK:", h["status"], "max_session_context_tokens=", h["max_session_context_tokens"])
    report["health"] = {
        "status": h["status"],
        "backend": h["backend"],
        "max_session_context_tokens": h["max_session_context_tokens"],
        "git_head_sha": h["code_provenance"]["git_head_sha"],
    }

    caps = get_json("/api/models/capabilities")
    assert caps["default_model"] == "nemotron-3.5-super-vl"
    assert set(caps["models"].keys()) == {"nemotron-3.5-super-vl", "lfm3-1.7b", "kimi-k3"}
    print("2. /api/models/capabilities OK:", {k: v["modalities"] for k, v in caps["models"].items()})
    report["capabilities"] = caps

    v1m = get_json("/v1/models")
    model_ids = [m["id"] for m in v1m["data"]]
    assert "nemotron-3.5-super-vl" in model_ids and "lfm3-1.7b" in model_ids and "clover-kimi-k3" in model_ids
    print("3. /v1/models OK:", model_ids)

    # 2. TTS per model
    for m_key, expected_voice in [
        ("nemotron-3.5-super-vl", "en-US-AriaNeural"),
        ("lfm3-1.7b", "en-US-JennyNeural"),
        ("kimi-k3", "en-US-AnaNeural"),
    ]:
        tts_r = post_json("/api/tts", {"text": f"Hello from {m_key}", "model": m_key})
        assert tts_r["ok"] is True and tts_r["voice"] == expected_voice, tts_r
        assert tts_r["audio_base64"].startswith("data:audio/"), tts_r
        print(f"4. /api/tts [{m_key}] OK: voice={tts_r['voice']}, audio_b64_len={len(tts_r['audio_base64'])}")
        report["checks"].append({"check": f"tts_{m_key}", "voice": tts_r["voice"], "audio_len": len(tts_r["audio_base64"])})

    img_b64, vid_b64, wav_b64 = build_test_media_payloads()

    # 3. Multimodal chat checks across all 3 models
    test_cases = [
        (
            "NEMO_VISION",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Describe this image.",
                "image_base64": img_b64,
            },
            "nemotron_super_vl_vision_cradio_ncm",
        ),
        (
            "NEMO_VIDEO",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Analyze the motion across video frames.",
                "image_base64": vid_b64,
                "attachments": [{"name": "clip.mp4", "type": "video/mp4", "data_b64": vid_b64}],
            },
            "nemotron_super_vl_video_evs_ncm",
        ),
        (
            "NEMO_AUDIO",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Analyze this voice recording.",
                "attachments": [{"name": "sample.wav", "type": "audio/wav", "data_b64": wav_b64}],
            },
            "nemotron_super_vl_audio_voice_ncm",
        ),
        (
            "NEMO_GEN_IMG",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Generate an image of the cosmos",
            },
            "nemotron_super_vl_image_generation",
        ),
        (
            "NEMO_GEN_VID",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Generate a video of orbital wave propagation",
            },
            "nemotron_super_vl_video_generation",
        ),
        (
            "NEMO_GEN_AUD",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "Generate an audio clip saying Welcome to Clover Multimodal Studio",
            },
            "nemotron_super_vl_audio_generation",
        ),
        (
            "NEMO_TEXT",
            {
                "session_id": "e2e_nemo",
                "model": "nemotron-3.5-super-vl",
                "prompt": "What is 25 * 4?",
            },
            "agent_nemotron-3.5-super-vl_ncm",
        ),
        (
            "LFM3_VISION",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Describe this image.",
                "image_base64": img_b64,
            },
            "lfm3_1_7b_vision_shortconv_ncm",
        ),
        (
            "LFM3_VIDEO",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Analyze the motion across video frames.",
                "image_base64": vid_b64,
                "attachments": [{"name": "clip.mp4", "type": "video/mp4", "data_b64": vid_b64}],
            },
            "lfm3_1_7b_video_shortconv_ncm",
        ),
        (
            "LFM3_AUDIO",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Analyze this voice recording.",
                "attachments": [{"name": "sample.wav", "type": "audio/wav", "data_b64": wav_b64}],
            },
            "lfm3_1_7b_audio_voice_ncm",
        ),
        (
            "LFM3_GEN_IMG",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Generate an image of a liquid state network",
            },
            "lfm3_1_7b_image_generation",
        ),
        (
            "LFM3_GEN_VID",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Generate a video of gated short convolution waves",
            },
            "lfm3_1_7b_video_generation",
        ),
        (
            "LFM3_GEN_AUD",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "Generate an audio clip saying Liquid LFM3 duplex voice ready",
            },
            "lfm3_1_7b_audio_generation",
        ),
        (
            "LFM3_TEXT",
            {
                "session_id": "e2e_lfm3",
                "model": "lfm3-1.7b",
                "prompt": "What is 25 * 4?",
            },
            "agent_lfm3-1.7b_ncm",
        ),
        (
            "K3_VISION",
            {
                "session_id": "e2e_k3_vis",
                "model": "kimi-k3",
                "prompt": "Describe this image.",
                "image_base64": img_b64,
            },
            "kimi_k3_vision_ncm",
        ),
        (
            "K3_VIDEO",
            {
                "session_id": "e2e_k3_vid",
                "model": "kimi-k3",
                "prompt": "Analyze the motion across video frames.",
                "image_base64": vid_b64,
                "attachments": [{"name": "clip.mp4", "type": "video/mp4", "data_b64": vid_b64}],
            },
            "kimi_k3_video_ncm",
        ),
        (
            "K3_GEN_IMG",
            {
                "session_id": "e2e_k3_gen",
                "model": "kimi-k3",
                "prompt": "Generate an image of a forest",
            },
            "agent_kimi-k3_ncm",
        ),
        (
            "K3_TEXT",
            {
                "session_id": "e2e_k3_txt",
                "model": "kimi-k3",
                "prompt": "Hello!",
            },
            "agent_kimi-k3_ncm",
        ),
    ]

    for label, payload, expected_mode in test_cases:
        t0 = time.perf_counter()
        res = post_json("/api/chat", payload)
        dt = round(time.perf_counter() - t0, 2)
        reply = str(res.get("reply") or "").strip()
        mode = str(res.get("mode") or "")
        assert reply, f"{label} returned empty reply: {res}"
        assert mode == expected_mode, f"{label} expected mode={expected_mode}, got {mode}"
        assert int(res.get("max_context_tokens") or 0) == 1048576, f"{label} max_context_tokens != 1M: {res}"
        media_list = res.get("media") or []
        if "GEN_" in label:
            assert len(media_list) >= 1, f"{label} expected media items, got {media_list}"
            m0 = media_list[0]
            if "GEN_IMG" in label:
                assert m0.get("type") == "image", f"{label} expected raster type=image, got {m0.get('type')}"
                assert m0.get("png_url", "").startswith("data:image/png;base64,"), f"{label} missing png_url"
                assert m0.get("jpg_url", "").startswith("data:image/jpeg;base64,"), f"{label} missing jpg_url"
                assert m0.get("webp_url", "").startswith("data:image/webp;base64,"), f"{label} missing webp_url"
                assert m0.get("bmp_url", "").startswith("data:image/bmp;base64,"), f"{label} missing bmp_url"
            elif "GEN_VID" in label:
                assert m0.get("type") == "video", f"{label} expected type=video, got {m0.get('type')}"
                assert m0.get("mp4_url", "").startswith("data:video/mp4;base64,"), f"{label} missing mp4_url"
                assert m0.get("webm_url", "").startswith("data:video/webm;base64,"), f"{label} missing webm_url"
                assert m0.get("gif_url", "").startswith("data:image/gif;base64,"), f"{label} missing gif_url"
                assert m0.get("webp_url", "").startswith("data:image/webp;base64,"), f"{label} missing webp_url"
            elif "GEN_AUD" in label:
                assert m0.get("type") == "audio", f"{label} expected type=audio, got {m0.get('type')}"
                assert m0.get("mp3_url", "").startswith("data:audio/mpeg;base64,"), f"{label} missing mp3_url"
                assert m0.get("wav_url", "").startswith("data:audio/wav;base64,"), f"{label} missing wav_url"
                assert m0.get("ogg_url", "").startswith("data:audio/ogg;base64,"), f"{label} missing ogg_url"
                assert m0.get("flac_url", "").startswith("data:audio/flac;base64,"), f"{label} missing flac_url"
        print(f"5. {label:13s} OK in {dt:5.2f}s | mode={mode:38s} | media={len(media_list)} | reply={reply[:68]!r}")
        report["checks"].append({
            "check": label,
            "elapsed_s": dt,
            "mode": mode,
            "media_count": len(media_list),
            "reply_excerpt": reply[:140],
        })

    for sid in ("e2e_nemo", "e2e_lfm3", "e2e_k3_vis", "e2e_k3_vid", "e2e_k3_gen", "e2e_k3_txt"):
        post_json("/api/reset", {"session_id": sid})

    print(f"\nALL {len(report['checks'])} LIVE SERVER END-TO-END MULTIMODAL CHECKS PASSED!")


if __name__ == "__main__":
    main()
