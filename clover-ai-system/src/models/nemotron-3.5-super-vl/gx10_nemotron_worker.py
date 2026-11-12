#!/usr/bin/env python3
"""
Stateless GX10 Multimodal Worker for `Nemotron 3.5 Super VL, 4-bit only`
(`clover-ai-system/src/models/nemotron-3.5-super-vl/gx10_nemotron_worker.py`).

Role in the Multi-Node Clover Architecture:
  - Runs on `GX10` (NVIDIA GB10 Grace Blackwell GPU + 20-core Grace ARM64 CPU).
  - Executes `Nemotron 3.5 Super VL, 4-bit only` on Clover's complete precomputed setup:
      * C1 (Inference):
          - `libnemotron_cpu_engine.so` (`experts.4bit.direct` 4-bit planar group-64 MoE experts only,
            Mamba-2 `trunk-0`, LatentMoE `trunk-1`, GQA `trunk-7`, MTP `trunk-mtp-0,1`).
          - Vision: 32-Layer `C-RADIO` ViT-Huge + `PixelShuffle(0.5)` + `ReLU^2` `mlp1` (`[1, 256, 4096]`)
            executed on the GB10 CUDA GPU from precomputed `CLVRTRK1` trunks (`ncm/vision-global-0`,
            `ncm/vision-trunk-0..31`) + `image_inspector` + `I-JEPA`.
          - Video: 32-Layer `C-RADIO` temporal pairs (`T=2`) + Efficient Video Sampling (`EVS`) token pruning
            (`compute_evs_retention_mask`, `q = 0.5`) executed on the GB10 CUDA GPU (`[1, N_retained, 4096]`).
          - Audio: 80-band log-Mel analysis + `NemotronAudioMambaBridge` (`[1, T_a, 4096]`) + `Audio-JEPA`.
      * C2 (Planning):
          - `nemotron/hybridagent/plan-trunk-0` (`4096 -> 1024 -> 4096` + `hl2048` `4096 -> 2048 -> 4096`)
            + `jepa/jepa-trunk-0` (`Clover RPRE` / `EB-JEPA` / `MJEPA` with visual/audio `context_cue_64`).
      * C3 (Decoding):
          - `nemotron/bridges/bridge-trunk-0` (`ComboBridge` & `ExpertBridge` `4096 -> 384`)
          - `nemotron/ncm/*` (`11-level VQ-VAE` + `12L NextConceptTransformer` + `3L LexicalPointerDecoder`
            decoding strictly in Nemotron's `131,072` Tekken vocabulary).
  - Keeps ZERO personal data (`stateless_zero_personal_data = True`):
      * Never writes user prompts, images, audio, video, or conversation history to disk.
      * Processes all media in-memory (`io.BytesIO`) and discards buffers immediately upon response completion.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
for cand in (
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    SCRIPT_DIR.parents[2] / "studio" / "clover",
    Path("/opt/clover-gx10/app/clover"),
    Path("/opt/clover-gx10/app/nemotron-3.5-super-vl"),
    Path("/opt/clover-k3/web-studio/clover"),
):
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

import image_inspector
import jepa_engine
import omni_c2_c3
from nemotron_multimodal import (
    NemotronAudioMambaBridge,
    NemotronCRadioVisionVideoEncoder,
    analyze_audio_bytes_and_extract_mel80,
    extract_video_frames_from_bytes,
)


def _decode_b64_bytes(b64_or_data_url: str) -> bytes:
    s = (b64_or_data_url or "").strip()
    if not s:
        return b""
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _inspect_pil_in_memory(pil_img: Image.Image, raw_bytes: bytes) -> dict[str, Any]:
    """Run palette, UI region detection, and optional stdin-piped tesseract OCR strictly in-memory (zero disk writes)."""
    palette = image_inspector.extract_palette(pil_img, max_colors=6)
    ocr_lines: list[dict[str, Any]] = []
    if shutil.which("tesseract") and raw_bytes:
        try:
            proc = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", "6"],
                input=raw_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
            txt = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
            for idx, ln in enumerate(txt.splitlines()):
                s = ln.strip()
                if s:
                    ocr_lines.append({"text": s, "bbox": [0, idx * 20, pil_img.width, 20], "conf": 90.0})
        except Exception:
            ocr_lines = []
    elements = image_inspector.detect_visual_regions(pil_img, ocr_lines, max_elements=12)
    ocr_text = " ".join(x["text"] for x in ocr_lines).strip()
    pal_str = ", ".join(f"{c['name']} ({c['hex']} {c['pct']}%)" for c in palette[:4])
    summary = f"Palette: {pal_str}; detected {len(elements)} visual regions"
    return {
        "palette": palette,
        "elements": elements,
        "ocr_text": ocr_text,
        "summary": summary,
    }


class StatelessGX10NemotronWorker:
    """Stateless `Nemotron 3.5 Super VL, 4-bit only` C1 -> C2 -> C3 + Vision/Video/Audio worker."""

    def __init__(self, base_dir: str | Path = "/opt/clover-gx10"):
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir)
        self.lock = threading.Lock()
        self.rt = omni_c2_c3.for_base(self.base_dir, model_key="nemotron-3.5-super-vl")
        if self.rt is None:
            raise RuntimeError(f"Failed to initialize Nemotron C1/C2/C3 runtime from {self.base_dir}")
        self.rt.get_c1_states_fn()
        self.jepa = jepa_engine.for_base(self.base_dir)
        self.vision_video_encoder = NemotronCRadioVisionVideoEncoder(
            precomputed_vision_dir=self.rt.pre / "ncm"
        )
        self.vision_video_encoder._ensure_precomputed_trunks_loaded()
        self.audio_mamba_bridge = NemotronAudioMambaBridge(n_mels=80, d_model=self.rt.d_model).eval()
        self.requests_served = 0
        self.startup_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def status(self) -> dict[str, Any]:
        q4_e1 = self.rt.pre / "experts" / "expert-1" / "experts.4bit.direct"
        bf16_e1 = self.rt.pre / "experts" / "expert-1" / "experts.direct"
        return {
            "ok": True,
            "node": "gx10-gpu-worker",
            "role": "Vision, audio, video. Keeps no personal data",
            "model_key": "nemotron-3.5-super-vl",
            "model_label": self.rt.model_label,
            "quantization": "4-bit-only (CLVR4B01 planar group-64)",
            "has_4bit_experts": q4_e1.is_file(),
            "has_bf16_experts": bf16_e1.is_file(),
            "vocab_size": self.rt.vocab_size,
            "d_model": self.rt.d_model,
            "cuda_available": torch.cuda.is_available(),
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vision_precomputed_trunks": len(self.vision_video_encoder._vtrunks or ()),
            "vision_device": str(self.vision_video_encoder._vdev),
            "stateless_zero_personal_data": True,
            "persisted_user_files": 0,
            "requests_served": self.requests_served,
            "startup_ms": self.startup_ms,
            "c1_c2_c3_stack": {
                "c1": "libnemotron_cpu_engine.so (4-bit experts) + 32L C-RADIO CUDA + EVS Video + AudioMambaBridge",
                "c2": "nemotron/hybridagent/plan-trunk-0 (4096-d) + jepa/jepa-trunk-0",
                "c3": "nemotron/bridges/bridge-trunk-0 (4096->384) + 11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder (131,072 vocab)",
            },
        }

    @staticmethod
    def _fuse_multimodal_4096_into_c1(c1_state: dict[str, Any], mm_embeds_4096: list[np.ndarray]) -> None:
        """Fuse `[1, N_m, 4096]` multimodal embeddings (from 32L C-RADIO Vision/Video or AudioMambaBridge)
        into `c1_state['h_primary']` and each active window's `h_valid` in Nemotron's `4096`-d space."""
        if not mm_embeds_4096:
            return
        cat_mm = np.concatenate([np.asarray(m, dtype=np.float32).reshape(1, -1, 4096) for m in mm_embeds_4096], axis=1)
        h_prim = c1_state["h_primary"]
        L_prim = int(h_prim.shape[1])
        mm_aligned = np.swapaxes(omni_c2_c3.interp_linear_1d(np.swapaxes(cat_mm, 1, 2), L_prim), 1, 2)
        h_rms = float(np.sqrt(np.mean(h_prim * h_prim) + 1e-8))
        m_rms = float(np.sqrt(np.mean(mm_aligned * mm_aligned) + 1e-8))
        alpha = np.float32(0.06 * h_rms / m_rms)
        c1_state["h_primary"] = h_prim + alpha * mm_aligned
        for w in c1_state.get("windows") or ():
            if w.get("passthrough"):
                continue
            st_i, ed_i = w.get("start", -1), w.get("end", -1)
            if 0 <= st_i < ed_i <= L_prim:
                w["h_valid"] = c1_state["h_primary"][:, st_i:ed_i, :]

    def process_multimodal(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a stateless multimodal request through `Nemotron 3.5 Super VL, 4-bit only` (`C1 -> C2 -> C3`)."""
        t0 = time.perf_counter()
        prompt = str(payload.get("prompt") or payload.get("query") or "Describe the media content clearly.").strip()
        run_c1_stage = bool(payload.get("run_c1_stage", True))
        offered_tools = list(payload.get("offered_tools") or ["inspect_image", "inspect_video", "inspect_audio"])

        mm_embeds_4096: list[np.ndarray] = []
        evidence_lines: list[str] = []
        modality_meta: dict[str, Any] = {}
        context_cues_64: list[np.ndarray] = []

        # 1. Vision (in-memory only)
        img_b64 = payload.get("image_base64") or ""
        imgs_b64 = list(payload.get("images_base64") or ([img_b64] if img_b64 else []))
        if imgs_b64:
            vis_reports = []
            for idx, b64_item in enumerate(imgs_b64[:4]):
                raw_bytes = _decode_b64_bytes(b64_item)
                if not raw_bytes:
                    continue
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                cradio_out = self.vision_video_encoder.encode_image(pil_img)
                mm_embeds_4096.append(cradio_out["image_embeds_4096"].detach().float().cpu().numpy())
                insp = _inspect_pil_in_memory(pil_img, raw_bytes)
                jepa_vis = self.jepa.inspect_visual_latent(np.asarray(pil_img), elements=insp.get("elements"))
                if jepa_vis.get("enabled") and "z_jepa" in jepa_vis:
                    context_cues_64.append(np.asarray(jepa_vis["z_jepa"], dtype=np.float32))
                summary_txt = str(insp.get("summary") or "").strip()
                ocr_txt = str(insp.get("ocr_text") or "").strip()
                scene_kind = str(jepa_vis.get("scene_kind") or insp.get("kind") or "visual_scene")
                line_parts = [f"Image {idx + 1} ({pil_img.width}x{pil_img.height}, scene={scene_kind}): {summary_txt}"]
                if ocr_txt:
                    line_parts.append(f"Visible text: {ocr_txt}.")
                evidence_lines.append(" ".join(line_parts))
                vis_reports.append({
                    "mode": cradio_out["mode"],
                    "tokens": cradio_out["post_pixel_shuffle_tokens"],
                    "hidden_dim": cradio_out["hidden_dim"],
                    "output_rms": cradio_out["output_rms"],
                    "encode_ms": cradio_out["encode_ms"],
                    "scene_kind": scene_kind,
                    "ocr_text": ocr_txt,
                    "summary": summary_txt,
                    "jepa_ms": jepa_vis.get("ms", 0.0),
                })
            if vis_reports:
                modality_meta["vision"] = vis_reports[0] if len(vis_reports) == 1 else vis_reports

        # 2. Video (in-memory only, with C-RADIO temporal pairs + EVS token pruning)
        vid_b64 = payload.get("video_base64") or ""
        vid_frames_b64 = list(payload.get("video_frames_base64") or ())
        if vid_b64 or vid_frames_b64:
            pil_frames: list[Image.Image] = []
            if vid_frames_b64:
                for fb64 in vid_frames_b64[:8]:
                    fb = _decode_b64_bytes(fb64)
                    if fb:
                        pil_frames.append(Image.open(io.BytesIO(fb)).convert("RGB"))
            elif vid_b64:
                vb = _decode_b64_bytes(vid_b64)
                pil_frames = extract_video_frames_from_bytes(vb, file_name=str(payload.get("video_name") or "clip.mp4"), max_frames=6)
            if pil_frames:
                pruning_rate = float(payload.get("evs_pruning_rate", 0.5))
                vid_out = self.vision_video_encoder.encode_video(pil_frames, pruning_rate=pruning_rate)
                mm_embeds_4096.append(vid_out["retained_embeds_4096"].detach().float().cpu().numpy())
                # Inspect first and last keyframe in-memory
                kf_indices = [0, len(pil_frames) - 1] if len(pil_frames) > 1 else [0]
                kf_notes = []
                for k_i in kf_indices:
                    buf = io.BytesIO()
                    pil_frames[k_i].save(buf, format="PNG")
                    kinsp = _inspect_pil_in_memory(pil_frames[k_i], buf.getvalue())
                    kjepa = self.jepa.inspect_visual_latent(np.asarray(pil_frames[k_i]), elements=kinsp.get("elements"))
                    if kjepa.get("enabled") and "z_jepa" in kjepa:
                        context_cues_64.append(np.asarray(kjepa["z_jepa"], dtype=np.float32))
                    ksum = str(kinsp.get("summary") or "").strip()
                    kocr = str(kinsp.get("ocr_text") or "").strip()
                    kf_notes.append(f"Frame {k_i + 1} ({kjepa.get('scene_kind', 'frame')}): {ksum}" + (f" Visible text: {kocr}." if kocr else ""))
                evidence_lines.append(
                    f"Video ({vid_out['input_frames']} frames, EVS retained {vid_out['retained_tokens_after_evs']}/{vid_out['total_tokens_before_evs']} tokens): "
                    + " | ".join(kf_notes)
                )
                modality_meta["video"] = {
                    k: v for k, v in vid_out.items() if k != "retained_embeds_4096"
                }

        # 3. Audio (in-memory only, with NemotronAudioMambaBridge + Audio-JEPA)
        aud_b64 = payload.get("audio_base64") or ""
        if aud_b64:
            aud_bytes = _decode_b64_bytes(aud_b64)
            if aud_bytes:
                aud_info = analyze_audio_bytes_and_extract_mel80(aud_bytes, file_name=str(payload.get("audio_name") or "audio.wav"))
                mel_80 = aud_info["mel_tensor"]
                with torch.no_grad():
                    aud_emb_4096 = self.audio_mamba_bridge(mel_80).detach().float().cpu().numpy()
                mm_embeds_4096.append(aud_emb_4096)
                aud_jepa = self.jepa.inspect_audio_latent(aud_bytes)
                if aud_jepa.get("enabled") and "z_jepa" in aud_jepa:
                    context_cues_64.append(np.asarray(aud_jepa["z_jepa"], dtype=np.float32))
                transcript = str(payload.get("audio_transcript") or "").strip()
                dur_s = aud_info.get("duration_seconds", 0.0)
                dom_hz = aud_info.get("dominant_frequency_hz", 0.0)
                aud_line = (
                    f"Audio ({dur_s}s, {aud_info.get('sample_rate_hz', 16000)} Hz, "
                    f"dominant={dom_hz} Hz, regime={aud_jepa.get('acoustic_regime', 'speech')}, "
                    f"SNR={aud_jepa.get('snr_db', 0.0)} dB)."
                )
                if transcript:
                    aud_line += f" Transcript: {transcript}"
                evidence_lines.append(aud_line)
                modality_meta["audio"] = {
                    "duration_sec": dur_s,
                    "sample_rate_hz": aud_info.get("sample_rate_hz", 16000),
                    "dominant_freq_hz": dom_hz,
                    "rms_energy": aud_info.get("rms_energy", 0.0),
                    "mamba_frames": int(aud_emb_4096.shape[1]),
                    "hidden_dim": int(aud_emb_4096.shape[2]),
                    "acoustic_regime": aud_jepa.get("acoustic_regime", "speech"),
                    "snr_db": aud_jepa.get("snr_db", 0.0),
                    "tf_latent_cos": aud_jepa.get("tf_latent_cos", 0.0),
                    "jepa_ms": aud_jepa.get("ms", 0.0),
                }

        candidate_text = str(payload.get("candidate_text") or "").strip()
        if not candidate_text:
            if evidence_lines:
                candidate_text = "\n".join(evidence_lines)
            else:
                candidate_text = prompt

        cue_64 = np.mean(np.stack(context_cues_64, axis=0), axis=0) if context_cues_64 else None

        with self.lock:
            # Stage 1: C1 (4-bit Nemotron C1 stage engine or embedding lookup + 4096-d multimodal fusion)
            if run_c1_stage:
                c1_state, c1_info = self.rt.c1_infer_stage(candidate_text)
            else:
                c1_state, c1_info = self.rt.c1_infer(candidate_text)
            self._fuse_multimodal_4096_into_c1(c1_state, mm_embeds_4096)
            c1_info["multimodal_streams_fused"] = len(mm_embeds_4096)

            # Stage 2: C2 (HybridAgent plan-trunk-0 in 4096-d + JEPA RPRE/MJEPA on jepa-trunk-0)
            c2_state, c2_info = self.rt.c2_plan(
                prompt,
                c1_state=c1_state,
                offered_tools=offered_tools,
                jepa_arch="clover_rpre",
                context_cue_64=cue_64,
            )

            # Stage 3: C3 (Bridges 4096->384 + 11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder in 131,072 vocab)
            decoded_text, c3_info = self.rt.c3_decode(c2_state)
            self.requests_served += 1

        # Explicitly drop in-memory media buffers before returning (stateless zero personal data)
        del mm_embeds_4096
        gc.collect()

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": True,
            "node": "gx10-gpu-worker",
            "model": self.rt.model_label,
            "quantization": "4-bit-only",
            "stateless_zero_personal_data": True,
            "reply": decoded_text,
            "modalities": modality_meta,
            "c1": c1_info,
            "c2": c2_info,
            "c3": c3_info,
            "total_ms": total_ms,
        }


_WORKER: StatelessGX10NemotronWorker | None = None


class GX10WorkerHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        assert _WORKER is not None
        if self.path.startswith(("/api/status", "/health", "/v1/models", "/")):
            self._send_json(200, _WORKER.status())
            return
        self._send_json(404, {"ok": False, "error": f"Unknown path {self.path}"})

    def do_POST(self) -> None:
        assert _WORKER is not None
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return

        if self.path.startswith(("/api/gx10/multimodal", "/api/chat", "/v1/chat/completions")):
            try:
                res = _WORKER.process_multimodal(payload)
                self._send_json(200, res)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(404, {"ok": False, "error": f"Unknown POST path {self.path}"})


def main() -> None:
    global _WORKER
    import argparse

    parser = argparse.ArgumentParser(description="Stateless GX10 Nemotron 3.5 Super VL 4-Bit C1/C2/C3 Worker")
    parser.add_argument("--base-dir", default="/opt/clover-gx10")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()

    _WORKER = StatelessGX10NemotronWorker(base_dir=args.base_dir)
    srv = ThreadingHTTPServer((args.host, args.port), GX10WorkerHTTPHandler)
    print(
        f"[GX10] Stateless Nemotron 3.5 Super VL 4-bit C1/C2/C3 worker listening on {args.host}:{args.port} "
        f"(GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, stateless_zero_personal_data=True)",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
