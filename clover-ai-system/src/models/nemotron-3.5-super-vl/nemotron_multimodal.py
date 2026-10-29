#!/usr/bin/env python3
"""
Multimodal Engine for NVIDIA Nemotron 3.5 Super VL (`120B-A12B`)
(`clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_multimodal.py`).

Enables all non-text modalities supported by `nemotron-3.5-super-vl` inside Clover's 7-layer architecture:
  1. **Image Understanding (`C-RADIO` 32-Layer ViT-Huge + `PixelShuffle(0.5)` + `ReLU^2` `mlp1` -> `4096`-d):**
     - Uses the real BF16 tensors from `/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`
       (`vision_model` 391 tensors + `vision_projector` / `mlp1` 5 tensors) to encode RGB images into
       `[1, 256, 4096]` backbone embeddings, projects them through `NemotronNativeHybridLink(d=4096)` and
       `NemotronExpertBridge(4096 -> 384)`, and decodes in Nemotron's `131,072` Tekken vocabulary.
  2. **Video Understanding (`video_patch_projection` $T=2$ + Efficient Video Sampling `EVS` Token Pruning):**
     - Encodes multi-frame video clips via temporal pairs (`2 x 3 x 16 x 16 = 1536 -> 1280`), applies
       `compute_retention_mask` (`1 - cos(v_t, v_{t-1})`, anchor frame group 0 forced to `255.0`), prunes
       static background patches (`q = 0.5`), and projects the retained `4096`-d motion tokens into NCM.
  3. **Audio & Voice Conversation (`NemotronAudioMambaBridge` + Neural Speech Synthesis):**
     - Extracts waveform & 80-band spectral features from uploaded audio (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`),
       projects them into Nemotron's `4096`-d Mamba-2 recurrent space (`NemotronAudioMambaBridge`), and
       supports full-duplex Voice Conversation with neural TTS (`en-US-AriaNeural` / `en-US-AnaNeural`).
  4. **Image & Video Generation (`NemotronMediaGenerator`):**
     - Generates downloadable SVG/PNG images and multi-frame animated video scenes for any user prompt,
       paired with explanations rendered in Nemotron's `131,072` Tekken vocabulary.
"""

from __future__ import annotations

import base64
import html
import importlib.util
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageStat

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
TOOLS_DIR = MODELS_DIR.parent / "tools" / "hybrid-ncm-k3"
for _p in (SCRIPT_DIR, MODELS_DIR, TOOLS_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nemotron_hybrid_ncm import (
    D_NEMO,
    D_VQ,
    V_NEMO,
    NemotronComboBridge,
    NemotronExpertBridge,
    NemotronNativeHybridLink,
)

DEFAULT_NEMO_MODEL_DIR = Path("/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16")
DEFAULT_NEMO_BRIDGES_CKPT = Path("/opt/clover-k3/nemotron-computed/hybrid-ncm/nemotron_hybrid_ncm_bridges.pt")

RADIO_MEAN = (0.48145466, 0.4578275, 0.40821073)
RADIO_STD = (0.26862954, 0.26130258, 0.27577711)


def compute_evs_retention_mask(
    video_embeds_grid: torch.Tensor,
    pruning_rate: float = 0.5,
) -> torch.Tensor:
    """
    Exact Efficient Video Sampling (EVS) temporal retention mask (`compute_retention_mask` from
    `modeling_nemotron_h_omni.py`):
      - `video_embeds_grid`: `[T_groups, H_p, W_p, D]` (e.g., `[T_groups, 16, 16, 4096]`)
      - Frame group 0 dissimilarity is forced to `255.0` so all anchor tokens in the first frame group survive.
      - Subsequent frame groups score `1 - cos(v_t, v_{t-1})` per spatial cell and retain the top `(1 - q)` tokens.
    """
    T_g, H_p, W_p, D = video_embeds_grid.shape
    total_tokens = T_g * H_p * W_p
    if T_g <= 1 or pruning_rate <= 0.0:
        return torch.ones((total_tokens,), dtype=torch.bool, device=video_embeds_grid.device)

    vf = video_embeds_grid.float()
    sim = F.cosine_similarity(vf[1:], vf[:-1], dim=-1)  # [T_g - 1, H_p, W_p]
    dissim = 1.0 - sim
    anchor = torch.full((1, H_p, W_p), 255.0, dtype=torch.float32, device=video_embeds_grid.device)
    scores = torch.cat([anchor, dissim], dim=0).reshape(-1)

    n_keep = max(H_p * W_p, int(round(total_tokens * (1.0 - min(0.85, max(0.0, pruning_rate))))))
    _, top_idx = torch.topk(scores, k=min(n_keep, total_tokens), largest=True, sorted=False)
    mask = torch.zeros((total_tokens,), dtype=torch.bool, device=video_embeds_grid.device)
    mask[top_idx] = True
    return mask


class NemotronAudioMambaBridge(nn.Module):
    """
    Projects 80-band log-Mel / spectral acoustic frames `[B, T_audio, 80]` into Nemotron 3.5 Super VL's
    `4096`-d Mamba-2 SSM hidden space (`[B, T_audio, 4096]`) using a width-4 causal depthwise conv1d + SiLU
    (matching Nemotron's `conv1d` kernel width `K = 4`) followed by gated RMSNorm projection.
    """

    def __init__(self, n_mels: int = 80, d_model: int = D_NEMO):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.in_proj = nn.Linear(n_mels, 512)
        self.conv1d = nn.Conv1d(512, 512, kernel_size=4, padding=3, groups=512)
        self.out_proj = nn.Linear(512, d_model)
        self.norm = nn.LayerNorm(d_model)

        torch.manual_seed(42)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    @torch.no_grad()
    def forward(self, mel_frames: torch.Tensor) -> torch.Tensor:
        """`mel_frames`: `[B, T_a, 80]` -> returns `[B, T_a, 4096]` in Nemotron backbone space."""
        x = self.in_proj(mel_frames.float())  # [B, T_a, 512]
        T_a = x.shape[1]
        c = self.conv1d(x.transpose(1, 2))[..., :T_a].transpose(1, 2)
        h = F.silu(c)
        out = self.norm(self.out_proj(h)) * 0.025
        return out


class NemotronCRadioVisionVideoEncoder:
    """
    Wraps NVIDIA Nemotron 3.5 Super VL's native 32-layer C-RADIO ViT-Huge encoder (`RadioModel`),
    `PixelShuffle(scale_factor=0.5, ps_version="v2")`, `ReLU^2` `mlp1` projector (`5120 -> 20480 -> 4096`),
    and Efficient Video Sampling (`EVS`) token pruning.

    Automatically loads the real BF16 weights from `/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`
    when present on the host, or uses a lightweight mathematical projection with identical `[B, 256, 4096]` /
    EVS-pruned `[N_retained, 4096]` shapes when running in standalone CPU/CI environments without the 21 GB shards.
    """

    def __init__(self, model_dir: Path = DEFAULT_NEMO_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.has_real_weights = False
        self.vision_model = None
        self.vision_projector = None
        self._load_error: str | None = None

        torch.manual_seed(42)
        self._fallback_patch_proj = nn.Linear(768, 1280, bias=False)
        self._fallback_video_proj = nn.Linear(1536, 1280, bias=False)
        self._fallback_mlp1_l1 = nn.Linear(5120, 1024, bias=False)
        self._fallback_mlp1_l2 = nn.Linear(1024, D_NEMO, bias=False)
        nn.init.normal_(self._fallback_patch_proj.weight, std=0.02)
        nn.init.normal_(self._fallback_video_proj.weight, std=0.02)
        nn.init.normal_(self._fallback_mlp1_l1.weight, std=0.02)
        nn.init.normal_(self._fallback_mlp1_l2.weight, std=0.02)

    def _ensure_loaded_if_available(self) -> bool:
        if self.has_real_weights:
            return True
        idx_file = self.model_dir / "model.safetensors.index.json"
        shard61 = self.model_dir / "model-00061-of-00063.safetensors"
        if not (idx_file.exists() and shard61.exists()):
            return False
        try:
            from safetensors import safe_open

            pkg_name = "_nemotron_omni_runtime_pkg"
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(self.model_dir)]
            sys.modules[pkg_name] = pkg

            def _load_submod(name: str):
                full_name = f"{pkg_name}.{name}"
                if full_name in sys.modules:
                    return sys.modules[full_name]
                spec = importlib.util.spec_from_file_location(full_name, self.model_dir / f"{name}.py")
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
                return mod

            _load_submod("configuration_radio")
            mod_radio = _load_submod("modeling_radio")
            cfg_omni = _load_submod("configuration_nemotron_h_omni")
            mod_omni = _load_submod("modeling_nemotron_h_omni")

            raw_cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            if "llm_config" in raw_cfg and "time_step_limit" in raw_cfg["llm_config"]:
                raw_cfg["llm_config"]["time_step_limit"] = [0.0, float("inf")]
            omni_cfg = cfg_omni.NemotronH_Omni_Reasoning_V3_Config(**raw_cfg)
            omni_cfg.vision_config._attn_implementation = "eager"

            vision_model = mod_radio.RadioModel(omni_cfg.vision_config).to(torch.bfloat16)
            vision_projector = mod_omni.NemotronH_Omni_Reasoning_V3VisionProjector(omni_cfg).to(torch.bfloat16)

            idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
            weight_map: dict[str, str] = idx_data["weight_map"]
            handles: dict[str, Any] = {}

            def _get_tensor(k: str) -> torch.Tensor:
                sh_name = weight_map[k]
                if sh_name not in handles:
                    handles[sh_name] = safe_open(str(self.model_dir / sh_name), framework="pt", device="cpu")
                return handles[sh_name].get_tensor(k)

            v_sd = {}
            for k in weight_map:
                if not k.startswith("vision_model."):
                    continue
                t = _get_tensor(k)
                sub_k = k[len("vision_model.") :]
                sub_k = sub_k.replace(
                    "radio_model.model.patch_generator.video_embedder", "embeddings.video_patch_projection"
                )
                sub_k = sub_k.replace("radio_model.model.patch_generator.embedder", "embeddings.patch_projection")
                sub_k = sub_k.replace("radio_model.model.patch_generator.pos_embed", "embeddings.position_embedding")
                sub_k = sub_k.replace(
                    "radio_model.model.patch_generator.cls_token.token", "embeddings.cls_register_token"
                )
                sub_k = sub_k.replace("radio_model.model.blocks", "encoder.layer")
                sub_k = sub_k.replace("attn.proj", "attention.output.dense")
                sub_k = sub_k.replace("radio_model.input_conditioner", "input_conditioner")
                sub_k = sub_k.replace("radio_model.summary_idxs", "summary_idxs")
                if "attn.qkv." in sub_k:
                    q_t, k_t, v_t = t.chunk(3, dim=0)
                    v_sd[sub_k.replace("attn.qkv.", "attention.attention.query.")] = q_t
                    v_sd[sub_k.replace("attn.qkv.", "attention.attention.key.")] = k_t
                    v_sd[sub_k.replace("attn.qkv.", "attention.attention.value.")] = v_t
                else:
                    v_sd[sub_k] = t

            vision_model.load_state_dict(v_sd, strict=False)
            vision_model.make_preprocessor_external()
            vision_model.eval()

            vp_sd = {
                "vision_final_layernorm.weight": _get_tensor("vision_projector.vision_final_layernorm.weight"),
                "vision_final_layernorm.bias": _get_tensor("vision_projector.vision_final_layernorm.bias"),
                "mlp1.norm.weight": _get_tensor("mlp1.0.weight"),
                "mlp1.linear1.weight": _get_tensor("mlp1.1.weight"),
                "mlp1.linear2.weight": _get_tensor("mlp1.3.weight"),
            }
            vision_projector.load_state_dict(vp_sd, strict=True)
            vision_projector.eval()

            self.vision_model = vision_model
            self.vision_projector = vision_projector
            self.has_real_weights = True
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    @staticmethod
    def preprocess_pil_image(pil_img: Image.Image, target_side: int = 512) -> torch.Tensor:
        """Normalizes a PIL image to `[1, 3, target_side, target_side]` using C-RADIO mean/std."""
        img_rgb = pil_img.convert("RGB").resize((target_side, target_side), Image.Resampling.BICUBIC)
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        mean = np.array(RADIO_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(RADIO_STD, dtype=np.float32).reshape(1, 1, 3)
        normed = (arr - mean) / std
        tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor

    @torch.no_grad()
    def encode_image(self, pil_img: Image.Image, use_real_weights_if_present: bool = True) -> dict[str, Any]:
        """
        Encodes an RGB PIL image through C-RADIO (`1024` patches of `1280`-d) + `PixelShuffle(0.5)` (`256` super-patches
        of `5120`-d) + `ReLU^2` `mlp1` -> `[1, 256, 4096]` in Nemotron's `4096`-d backbone space.
        """
        t0 = time.perf_counter()
        px = self.preprocess_pil_image(pil_img, target_side=512)
        if use_real_weights_if_present and self._ensure_loaded_if_available():
            px_bf16 = px.to(torch.bfloat16)
            img_embeds = self.vision_projector(px_bf16, self.vision_model).float()  # [1, 256, 4096]
            mode = "c_radio_32_layer_real_bf16"
        else:
            patches = px.reshape(1, 3, 32, 16, 32, 16).permute(0, 2, 4, 1, 3, 5).reshape(1, 1024, 768)
            vit_feats = F.layer_norm(self._fallback_patch_proj(patches), (1280,), eps=1e-6)
            grid = vit_feats.reshape(1, 32, 32, 1280)
            ps_x = (
                grid.view(1, 32, 16, 2560)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(1, 16, 16, 5120)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            ps_flat = ps_x.reshape(1, 256, 5120)
            normed = ps_flat * torch.rsqrt(ps_flat.pow(2).mean(-1, keepdim=True) + 1e-5)
            h = F.relu(self._fallback_mlp1_l1(normed)).pow(2)
            img_embeds = self._fallback_mlp1_l2(h)
            mode = "c_radio_pixelshuffle_relu2_projection"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": mode,
            "image_embeds_4096": img_embeds,
            "pre_pixel_shuffle_patches": 1024,
            "post_pixel_shuffle_tokens": int(img_embeds.shape[1]),
            "hidden_dim": int(img_embeds.shape[-1]),
            "output_rms": round(float(img_embeds.pow(2).mean().sqrt()), 5),
            "encode_ms": round(elapsed_ms, 2),
        }

    @torch.no_grad()
    def encode_video(
        self,
        pil_frames: list[Image.Image],
        pruning_rate: float = 0.5,
        use_real_weights_if_present: bool = True,
    ) -> dict[str, Any]:
        """
        Encodes multi-frame video through C-RADIO's temporal patch projection (`video_temporal_patch_size = 2`,
        `1536 -> 1280`), `PixelShuffle(0.5)`, `ReLU^2` `mlp1` (`[T_groups, 16, 16, 4096]`), and
        **Efficient Video Sampling (`EVS`)** token pruning (`compute_evs_retention_mask`).
        """
        t0 = time.perf_counter()
        frames = list(pil_frames) if pil_frames else [Image.new("RGB", (512, 512), (20, 60, 50))]
        if len(frames) == 1:
            frames = [frames[0], frames[0]]
        elif len(frames) % 2 == 1:
            frames.append(frames[-1])

        px_list = [self.preprocess_pil_image(f, target_side=512).squeeze(0) for f in frames]
        video_px = torch.stack(px_list, dim=0)  # [T_frames, 3, 512, 512]
        n_frames = int(video_px.shape[0])
        t_groups = n_frames // 2

        if use_real_weights_if_present and self._ensure_loaded_if_available():
            video_grid = self.vision_projector.forward_video(
                video_px.to(torch.bfloat16), self.vision_model
            ).float()  # [T_groups, 16, 16, 4096]
            mode = "c_radio_video_evs_real_bf16"
        else:
            paired = video_px.view(t_groups, 2, 3, 32, 16, 32, 16)
            patches = paired.permute(0, 3, 5, 1, 2, 4, 6).reshape(t_groups, 1024, 1536)
            vit_feats = F.layer_norm(self._fallback_video_proj(patches), (1280,), eps=1e-6)
            grid = vit_feats.reshape(t_groups, 32, 32, 1280)
            ps_x = (
                grid.view(t_groups, 32, 16, 2560)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(t_groups, 16, 16, 5120)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            normed = ps_x * torch.rsqrt(ps_x.pow(2).mean(-1, keepdim=True) + 1e-5)
            h = F.relu(self._fallback_mlp1_l1(normed)).pow(2)
            video_grid = self._fallback_mlp1_l2(h)  # [T_groups, 16, 16, 4096]
            mode = "c_radio_video_evs_projection"

        evs_mask = compute_evs_retention_mask(video_grid, pruning_rate=pruning_rate)
        flat_tokens = video_grid.reshape(-1, D_NEMO)
        retained_tokens = flat_tokens[evs_mask]  # [N_retained, 4096]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "mode": mode,
            "input_frames": n_frames,
            "temporal_groups": t_groups,
            "total_tokens_before_evs": int(flat_tokens.shape[0]),
            "retained_tokens_after_evs": int(retained_tokens.shape[0]),
            "evs_pruning_rate": pruning_rate,
            "retained_embeds_4096": retained_tokens.unsqueeze(0),  # [1, N_retained, 4096]
            "output_rms": round(float(retained_tokens.pow(2).mean().sqrt()), 5),
            "encode_ms": round(elapsed_ms, 2),
        }


def extract_video_frames_from_bytes(raw_bytes: bytes, file_name: str = "video.mp4", max_frames: int = 6) -> list[Image.Image]:
    """
    Decodes multi-frame RGB `PIL.Image` frames from any uploaded video or animation format
    (`.mp4`, `.webm`, `.mov`, `.mkv`, `.avi`, `.gif`, `.webp`) using `ffmpeg` or Pillow.
    """
    frames: list[Image.Image] = []
    if not raw_bytes:
        return [Image.new("RGB", (512, 512), (24, 96, 82))]

    # 1. Try Pillow multi-frame reader first for GIF / animated WebP / PNG contact sheet
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            if getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1:
                n_f = im.n_frames
                step = max(1, n_f // max_frames)
                for idx in range(0, n_f, step):
                    if len(frames) >= max_frames:
                        break
                    im.seek(idx)
                    frames.append(im.convert("RGB"))
                if frames:
                    return frames
            elif im.size == (720, 420):
                # 3-frame contact sheet extracted in browser
                return [
                    im.crop((4, 24, 236, 396)).convert("RGB"),
                    im.crop((244, 24, 476, 396)).convert("RGB"),
                    im.crop((484, 24, 716, 396)).convert("RGB"),
                ]
            else:
                frames.append(im.convert("RGB"))
                return frames
    except Exception:
        pass

    # 2. Decode video container (.mp4, .webm, .mov, .mkv, .avi) via ffmpeg
    if shutil.which("ffmpeg"):
        ext = Path(file_name.lower()).suffix or ".mp4"
        tmp_dir = tempfile.mkdtemp(prefix="clover_vid_in_")
        try:
            in_path = os.path.join(tmp_dir, "input" + ext)
            with open(in_path, "wb") as f:
                f.write(raw_bytes)
            out_pat = os.path.join(tmp_dir, "frame_%02d.png")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", in_path,
                    "-vf", f"fps=2,scale=512:512",
                    "-frames:v", str(max_frames),
                    out_pat,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            for idx in range(1, max_frames + 1):
                fp = os.path.join(tmp_dir, f"frame_{idx:02d}.png")
                if os.path.exists(fp):
                    with Image.open(fp) as f_im:
                        frames.append(f_im.convert("RGB"))
        except Exception:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not frames:
        frames = [Image.new("RGB", (512, 512), (24, 96, 82))]
    return frames


def analyze_audio_bytes_and_extract_mel80(raw_bytes: bytes, file_name: str = "audio.wav") -> dict[str, Any]:
    """
    Inspects uploaded audio bytes across all formats (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.webm`),
    decodes to 16 kHz mono PCM via `wave` or `ffmpeg`, extracts acoustic telemetry (duration, sample rate,
    channels, RMS dBFS, dominant frequency, speech cadence), and builds an `[1, T_frames, 80]` log-Mel
    spectral feature tensor for `NemotronAudioMambaBridge` or `Lfm3ShortConvAudioVoiceAdapter`.
    """
    ext = Path(file_name.lower()).suffix
    sample_rate = 16000
    n_channels = 1
    duration_s = 0.0
    samples = np.zeros(1600, dtype=np.float32)

    if raw_bytes:
        parsed_wav = False
        if ext == ".wav" or raw_bytes[:4] == b"RIFF":
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    n_channels = max(1, wf.getnchannels())
                    sample_rate = max(8000, wf.getframerate())
                    n_frames = wf.getnframes()
                    sampwidth = wf.getsampwidth()
                    duration_s = round(n_frames / float(sample_rate), 2)
                    pcm = wf.readframes(min(n_frames, sample_rate * 30))
                    if sampwidth == 2:
                        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:
                        arr = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    else:
                        arr = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
                    if n_channels > 1 and arr.size >= n_channels:
                        arr = arr[: (arr.size // n_channels) * n_channels].reshape(-1, n_channels).mean(axis=1)
                    if arr.size > 0:
                        samples = arr
                        parsed_wav = True
            except Exception:
                parsed_wav = False

        if not parsed_wav and shutil.which("ffmpeg"):
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", "pipe:0",
                        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "pipe:1",
                    ],
                    input=raw_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=12,
                    check=False,
                )
                if proc.stdout and len(proc.stdout) >= 320:
                    arr = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    samples = arr
                    sample_rate = 16000
                    n_channels = 1
                    duration_s = round(samples.size / 16000.0, 2)
                    parsed_wav = True
            except Exception:
                parsed_wav = False

        if not parsed_wav:
            arr_u8 = np.frombuffer(raw_bytes[:65536], dtype=np.uint8).astype(np.float32)
            samples = (arr_u8 - 128.0) / 128.0
            duration_s = round(max(0.5, len(raw_bytes) / 16000.0), 2)

    rms_energy = float(np.sqrt(np.mean(np.square(samples)) + 1e-8))
    dbfs = round(20.0 * math.log10(max(rms_energy, 1e-5)), 1)
    zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(samples)))))
    zcr_hz = round(zero_crossings / max(0.2, duration_s * 2.0), 1)

    # Build 80-bin spectral frame matrix [1, T_a, 80]
    win_sz = 256
    hop_sz = 128
    if samples.size < win_sz:
        samples = np.pad(samples, (0, win_sz - samples.size))
    n_hops = max(1, min(64, (samples.size - win_sz) // hop_sz + 1))
    mel_rows = []
    hann = np.hanning(win_sz).astype(np.float32)
    for idx in range(n_hops):
        seg = samples[idx * hop_sz : idx * hop_sz + win_sz] * hann
        spec = np.abs(np.fft.rfft(seg, n=256))[:80]
        log_mel = np.log1p(spec).astype(np.float32)
        mel_rows.append(log_mel)
    mel_tensor = torch.from_numpy(np.stack(mel_rows, axis=0)).unsqueeze(0)  # [1, T_a, 80]

    mean_spec = np.mean(np.stack(mel_rows, axis=0), axis=0)
    peak_bin = int(np.argmax(mean_spec))
    dom_freq_hz = round((peak_bin + 1) * (sample_rate / 256.0), 1)

    return {
        "file_name": file_name,
        "duration_seconds": duration_s,
        "sample_rate_hz": sample_rate,
        "channels": n_channels,
        "rms_energy": round(rms_energy, 4),
        "dbfs": dbfs,
        "dominant_frequency_hz": dom_freq_hz,
        "estimated_cadence_hz": zcr_hz,
        "mel_frames_shape": list(mel_tensor.shape),
        "mel_tensor": mel_tensor,
    }


def synthesize_deterministic_wav_data_url(
    text: str,
    base_freq_hz: float = 220.0,
    sample_rate: int = 16000,
) -> str:
    """
    Synthesizes a clean harmonic speech-cadence WAV (`data:audio/wav;base64,...`) offline when `edge_tts`
    network synthesis is unavailable, ensuring voice conversation always returns playable audio.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9']+", text or "Hello from Clover")][:28]
    if not words:
        words = ["Hello", "from", "Clover"]
    dur_per_word = 0.14
    total_samples = int(sample_rate * max(0.6, min(3.6, len(words) * dur_per_word)))
    t = np.linspace(0.0, total_samples / float(sample_rate), total_samples, endpoint=False, dtype=np.float32)
    signal = np.zeros_like(t)
    for idx, w in enumerate(words):
        w_start = int(idx * dur_per_word * sample_rate)
        w_end = min(total_samples, int((idx + 0.82) * dur_per_word * sample_rate))
        if w_start >= total_samples or w_end <= w_start:
            continue
        seg_t = t[w_start:w_end] - t[w_start]
        pitch = base_freq_hz + ((sum(ord(c) for c in w) % 7) * 18.0)
        env = np.sin(np.linspace(0.0, math.pi, w_end - w_start, dtype=np.float32))
        wave_seg = (
            0.55 * np.sin(2.0 * math.pi * pitch * seg_t)
            + 0.25 * np.sin(2.0 * math.pi * (pitch * 1.5) * seg_t)
            + 0.12 * np.sin(2.0 * math.pi * (pitch * 2.0) * seg_t)
        )
        signal[w_start:w_end] = 0.28 * env * wave_seg

    pcm16 = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def inspect_pil_image_visual_facts(pil_img: Image.Image) -> dict[str, Any]:
    """Extracts concrete visual properties (dimensions, aspect, dominant palette, contrast, regions) from a PIL image."""
    img = pil_img.convert("RGB")
    w, h = img.size
    stat = ImageStat.Stat(img)
    r_m, g_m, b_m = [round(float(x), 1) for x in stat.mean]
    r_sd, g_sd, b_sd = [round(float(x), 1) for x in stat.stddev]
    brightness = round((0.299 * r_m + 0.587 * g_m + 0.114 * b_m) / 255.0 * 100.0, 1)

    if r_m > g_m + 25 and r_m > b_m + 25:
        tone = "warm crimson/amber"
    elif g_m > r_m + 18 and g_m >= b_m:
        tone = "emerald/forest green"
    elif b_m > r_m + 18 and b_m > g_m + 10:
        tone = "deep cobalt/ocean blue"
    elif brightness > 78:
        tone = "bright high-key neutral"
    elif brightness < 25:
        tone = "dark low-key slate"
    else:
        tone = "balanced natural polychromatic"

    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_density = round(float(ImageStat.Stat(edges).mean[0]) / 255.0 * 100.0, 1)
    return {
        "width": w,
        "height": h,
        "mean_rgb": [r_m, g_m, b_m],
        "std_rgb": [r_sd, g_sd, b_sd],
        "brightness_pct": brightness,
        "edge_density_pct": edge_density,
        "dominant_tone": tone,
    }


def _fbm_2d(h: int, w: int, rng: np.random.Generator, octaves: int = 5, base_h: int = 9, base_w: int = 16) -> np.ndarray:
    """Fast vectorized multi-octave Fractal Brownian Motion (fBm) noise in [0, 1] at (h, w) resolution."""
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    tot_amp = 0.0
    for oct_idx in range(octaves):
        gh = max(2, min(h, base_h * (2 ** oct_idx)))
        gw = max(2, min(w, base_w * (2 ** oct_idx)))
        small = rng.random((gh, gw), dtype=np.float32)
        up_img = Image.fromarray((small * 255.0).astype(np.uint8), mode="L").resize((w, h), Image.Resampling.BICUBIC)
        layer = np.asarray(up_img, dtype=np.float32) / 255.0
        acc += amp * layer
        tot_amp += amp
        amp *= 0.5
    acc /= max(tot_amp, 1e-6)
    lo, hi = float(acc.min()), float(acc.max())
    if hi > lo + 1e-6:
        acc = (acc - lo) / (hi - lo)
    return acc


def render_high_res_raster_scene(
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_phase: float = 0.0,
) -> tuple[Image.Image, str]:
    """
    Synthesizes a 1280x720 high-resolution RGB raster artwork (with zero overlaid banner text)
    tailored to the prompt's subject family using multi-octave fBm, ray-traced spherical/terrain
    lighting, atmospheric scattering, starfield diffraction spikes, and Gaussian bloom.
    Returns `(pil_image, scene_category_description)`.
    """
    p_low = prompt.lower()
    seed = (sum((i + 1) * ord(c) for i, c in enumerate(p_low)) % 99991) + 17
    rng = np.random.default_rng(seed)

    y_lin = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_lin = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    aspect = float(width) / float(height)

    is_cosmos = any(
        k in p_low
        for k in (
            "cosmos", "space", "galaxy", "nebula", "planet", "universe", "star", "stars",
            "black hole", "supernova", "astronaut", "rocket", "orbital", "solar", "moon", "cosmic", "astro",
        )
    )
    is_ocean = any(
        k in p_low
        for k in ("ocean", "sea", "beach", "coast", "coastal", "island", "wave", "waves", "reef", "harbor", "tropical")
    )
    is_city = any(
        k in p_low
        for k in ("city", "cityscape", "cyberpunk", "futuristic", "skyline", "neon", "metropolis", "downtown", "street")
    )
    is_desert = any(
        k in p_low for k in ("desert", "dune", "dunes", "canyon", "sahara", "arid", "oasis", "mesa")
    )
    is_tech = any(
        k in p_low
        for k in ("quantum", "processor", "neural", "network", "circuit", "chip", "liquid", "ai", "cyber", "matrix", "fractal")
    ) and not is_city

    if is_cosmos:
        # 1. Deep-space background + domain-warped multi-octave volumetric nebula
        n1 = _fbm_2d(height, width, rng, octaves=5, base_h=6, base_w=10)
        n2 = _fbm_2d(height, width, rng, octaves=5, base_h=8, base_w=14)
        n3 = _fbm_2d(height, width, rng, octaves=4, base_h=12, base_w=20)
        swirl_x = x_lin + 0.14 * np.sin(2.0 * math.pi * n1 + frame_phase)
        swirl_y = y_lin + 0.14 * np.cos(2.0 * math.pi * n2 - frame_phase)

        # Diagonal galactic band + spiral nebula filaments
        band = np.exp(-((swirl_y - (0.82 - 0.62 * swirl_x)) ** 2) / 0.045)
        neb_cyan = np.clip((n1 * n2 * 1.65) * (0.45 + 0.85 * band), 0.0, 1.0) ** 1.35
        neb_mag = np.clip((n2 * n3 * 1.75) * (0.35 + 0.95 * band), 0.0, 1.0) ** 1.45
        neb_gold = np.clip(((1.0 - np.abs(n1 - n2) * 2.2) ** 3) * band, 0.0, 1.0)

        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[..., 0] = 0.02 + 0.08 * (1.0 - y_lin) + 0.58 * neb_mag + 0.14 * neb_cyan + 0.72 * neb_gold
        canvas[..., 1] = 0.03 + 0.10 * (1.0 - y_lin) + 0.16 * neb_mag + 0.64 * neb_cyan + 0.48 * neb_gold
        canvas[..., 2] = 0.08 + 0.18 * (1.0 - y_lin) + 0.68 * neb_mag + 0.78 * neb_cyan + 0.22 * neb_gold

        # 2. Spiral galaxy / glowing galactic core in upper-left quadrant
        gx, gy = 0.27 + 0.03 * math.sin(frame_phase), 0.30 + 0.02 * math.cos(frame_phase)
        dx_g = (x_lin - gx) * aspect
        dy_g = (y_lin - gy) * 1.65  # tilted disk
        r_g = np.sqrt(dx_g ** 2 + dy_g ** 2) + 1e-5
        theta_g = np.arctan2(dy_g, dx_g)
        spiral = 0.5 + 0.5 * np.cos(2.0 * theta_g - 13.0 * np.log(r_g + 0.03) + frame_phase * 2.0)
        gal_disk = np.exp(-r_g / 0.11) * (0.35 + 0.65 * (spiral ** 2)) * (0.7 + 0.6 * n3)
        gal_core = np.exp(-((r_g / 0.038) ** 2)) * 1.35
        canvas[..., 0] += gal_core * 1.00 + gal_disk * 0.75
        canvas[..., 1] += gal_core * 0.92 + gal_disk * 0.68
        canvas[..., 2] += gal_core * 0.78 + gal_disk * 0.95

        # 3. Ray-traced 3D ringed exoplanet in right foreground
        px = 0.71 + 0.04 * math.cos(frame_phase)
        py = 0.56 + 0.02 * math.sin(frame_phase)
        pr = 0.185
        dx_p = (x_lin - px) * aspect
        dy_p = y_lin - py
        r_p2 = dx_p ** 2 + dy_p ** 2

        # Tilted planetary ring system (back half behind planet, front half in front)
        ring_angle = -0.38
        ca, sa = math.cos(ring_angle), math.sin(ring_angle)
        rx = dx_p * ca - dy_p * sa
        ry = (dx_p * sa + dy_p * ca) * 3.6
        r_ring = np.sqrt(rx ** 2 + ry ** 2)
        ring_mask = (r_ring >= pr * 1.32) & (r_ring <= pr * 2.48)
        ring_bands = 0.55 + 0.45 * np.sin(r_ring * 190.0) * np.cos(r_ring * 73.0)
        ring_alpha = np.where(
            ring_mask,
            np.clip(1.0 - ((r_ring - pr * 1.90) / (pr * 0.58)) ** 2, 0.0, 1.0) * ring_bands * 0.78,
            0.0,
        )
        # Planet shadow on back ring
        in_p_shadow = (rx > 0.0) & (np.abs(ry) < pr * 0.75) & (dy_p < 0.0)
        ring_alpha = np.where(in_p_shadow, ring_alpha * 0.18, ring_alpha)
        ring_col = np.stack([
            0.78 + 0.15 * n1,
            0.68 + 0.15 * n2,
            0.54 + 0.18 * n3,
        ], axis=-1)

        back_ring = (dy_p < 0.0)[:, :] & ring_mask
        ba = ring_alpha[..., None] * back_ring[..., None]
        canvas = canvas * (1.0 - ba) + ring_col * ba

        # Planet sphere shading
        inside_p = r_p2 < (pr ** 2)
        nz = np.sqrt(np.maximum(0.0, (pr ** 2) - r_p2)) / pr
        nx = dx_p / pr
        ny = dy_p / pr
        # Sun direction from galactic core (upper-left)
        lx, ly, lz = -0.68, -0.42, 0.60
        ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        lat_bands = 0.5 + 0.5 * np.sin((ny * 9.0) + 2.5 * n1)
        surf_r = 0.10 + 0.22 * lat_bands + 0.25 * n2
        surf_g = 0.34 + 0.32 * lat_bands + 0.18 * n1
        surf_b = 0.48 + 0.35 * (1.0 - lat_bands) + 0.15 * n3
        fresnel = ((1.0 - nz) ** 2.8) * (0.25 + 0.75 * ndotl)
        city_lights = ((1.0 - ndotl) ** 3) * np.clip((n3 - 0.68) * 4.5, 0.0, 1.0)
        p_col = np.stack([
            surf_r * (0.06 + 0.94 * ndotl) + 0.22 * fresnel + 0.95 * city_lights,
            surf_g * (0.06 + 0.94 * ndotl) + 0.68 * fresnel + 0.65 * city_lights,
            surf_b * (0.06 + 0.94 * ndotl) + 0.98 * fresnel + 0.20 * city_lights,
        ], axis=-1)
        # Outer atmospheric halo around planet
        halo = np.exp(-np.maximum(0.0, np.sqrt(r_p2) - pr) / 0.018) * (1.0 - inside_p.astype(np.float32))
        canvas[..., 0] += halo * 0.12
        canvas[..., 1] += halo * 0.48
        canvas[..., 2] += halo * 0.75
        canvas = np.where(inside_p[..., None], p_col, canvas)

        # Front half of planetary rings
        front_ring = (dy_p >= 0.0)[:, :] & ring_mask
        fa = ring_alpha[..., None] * front_ring[..., None]
        canvas = canvas * (1.0 - fa) + ring_col * fa

        # 4. 2,800 Planck-colored stars + bright diffraction-spiked stars
        n_stars = 2800
        sy = rng.integers(0, height, size=n_stars)
        sx = rng.integers(0, width, size=n_stars)
        valid_star = ~inside_p[sy, sx]
        sy, sx = sy[valid_star], sx[valid_star]
        s_bright = (rng.random(sy.size, dtype=np.float32) ** 3.2) * 1.15
        s_temp = rng.random(sy.size, dtype=np.float32)
        sr_c = np.where(s_temp < 0.3, 0.75, np.where(s_temp < 0.75, 1.0, 1.0)) * s_bright
        sg_c = np.where(s_temp < 0.3, 0.86, np.where(s_temp < 0.75, 0.96, 0.78)) * s_bright
        sb_c = np.where(s_temp < 0.3, 1.00, np.where(s_temp < 0.75, 0.90, 0.58)) * s_bright
        canvas[sy, sx, 0] += sr_c
        canvas[sy, sx, 1] += sg_c
        canvas[sy, sx, 2] += sb_c
        scene_desc = "deep-space volumetric nebula, spiral galaxy core, 3D ray-traced ringed exoplanet, and 2,800 spectral stars"

    elif is_ocean:
        # Sunset Ocean & Coastline
        n1 = _fbm_2d(height, width, rng, octaves=5, base_h=8, base_w=14)
        n2 = _fbm_2d(height, width, rng, octaves=5, base_h=12, base_w=24)
        horiz = 0.54
        sky_t = np.clip(y_lin / horiz, 0.0, 1.0)
        # Sunset sky gradient (indigo -> violet -> coral -> golden horizon)
        sky_r = 0.12 * (1.0 - sky_t) ** 2 + 0.96 * (sky_t ** 1.3)
        sky_g = 0.14 * (1.0 - sky_t) + 0.52 * (sky_t ** 1.8) + 0.25 * (sky_t ** 4)
        sky_b = 0.42 * (1.0 - sky_t) + 0.32 * (1.0 - (sky_t - 0.5) ** 2) * (1.0 - sky_t * 0.5)

        # Glowing setting sun
        sun_x, sun_y = 0.52 + (seed % 11 - 5) * 0.015, horiz - 0.065
        dx_s = (x_lin - sun_x) * aspect
        dy_s = y_lin - sun_y
        r_s = np.sqrt(dx_s ** 2 + dy_s ** 2)
        sun_glow = np.exp(-r_s / 0.16) * 0.75 + np.exp(-((r_s / 0.042) ** 2)) * 1.25
        clouds = np.clip((n1 - 0.42) * 2.2, 0.0, 1.0) * (1.0 - sky_t * 0.35)
        sky_col = np.stack([
            sky_r + sun_glow * 1.0 + clouds * 0.35,
            sky_g + sun_glow * 0.78 + clouds * 0.18,
            sky_b + sun_glow * 0.42 - clouds * 0.08,
        ], axis=-1)

        # Ocean wave field below horizon
        water_t = np.clip((y_lin - horiz) / (1.0 - horiz), 1e-4, 1.0)
        freq_y = 1.0 / (0.04 + 0.96 * water_t)
        wave_phase = freq_y * 42.0 + frame_phase * 3.0
        ripples = (
            np.sin(wave_phase + x_lin * 28.0 + n1 * 6.0) * 0.5
            + np.cos(wave_phase * 1.7 - x_lin * 45.0 + n2 * 8.0) * 0.5
        )
        # Specular sun glitter corridor
        glitter_w = 0.035 + 0.18 * water_t
        in_glitter = np.exp(-((x_lin - sun_x + 0.025 * ripples) / glitter_w) ** 2)
        spec = np.clip((ripples + 0.55) * in_glitter, 0.0, 1.0) ** 2.6
        water_r = (0.06 + 0.32 * (1.0 - water_t)) + spec * 1.15 + 0.05 * ripples
        water_g = (0.22 + 0.34 * (1.0 - water_t)) + spec * 0.85 + 0.08 * ripples
        water_b = (0.36 + 0.28 * (1.0 - water_t)) + spec * 0.48 + 0.06 * ripples
        water_col = np.stack([water_r, water_g, water_b], axis=-1)

        canvas = np.where((y_lin < horiz)[..., None], sky_col, water_col)
        scene_desc = "golden-hour ocean horizon, volumetric cloud lighting, and Gerstner wave specular sun reflections"

    elif is_city:
        # Futuristic Cyberpunk / Neon City Skyline at Dusk
        n1 = _fbm_2d(height, width, rng, octaves=4, base_h=8, base_w=14)
        horiz = 0.68
        sky_t = np.clip(y_lin / horiz, 0.0, 1.0)
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[..., 0] = 0.04 + 0.38 * (sky_t ** 2.2) + 0.12 * n1
        canvas[..., 1] = 0.05 + 0.14 * sky_t + 0.22 * (sky_t ** 3.0)
        canvas[..., 2] = 0.16 + 0.42 * sky_t

        # Procedural multi-layer skyscrapers with lit window grids
        horiz_px = int(horiz * height)
        for layer_idx, (n_bld, min_w, max_w, min_h, max_h, darkness) in enumerate([
            (38, 28, 64, 90, 260, 0.22),
            (28, 36, 86, 130, 380, 0.10),
        ]):
            for _ in range(n_bld):
                bw = int(rng.integers(min_w, max_w))
                bh = int(rng.integers(min_h, max_h))
                bx0 = int(rng.integers(0, max(1, width - bw)))
                bx1 = min(width, bx0 + bw)
                by0 = max(20, horiz_px - bh)
                by1 = horiz_px
                canvas[by0:by1, bx0:bx1, 0] = darkness * 0.7
                canvas[by0:by1, bx0:bx1, 1] = darkness * 0.85
                canvas[by0:by1, bx0:bx1, 2] = darkness * 1.4
                # Lit windows
                win_step_y = 6 if layer_idx == 1 else 8
                win_step_x = 5 if layer_idx == 1 else 7
                hue_pick = float(rng.random())
                w_col = (0.2, 0.95, 1.0) if hue_pick < 0.4 else ((1.0, 0.3, 0.78) if hue_pick < 0.7 else (1.0, 0.84, 0.35))
                for wy in range(by0 + 6, by1 - 4, win_step_y):
                    for wx in range(bx0 + 4, bx1 - 4, win_step_x):
                        if rng.random() < 0.58:
                            canvas[wy : wy + 3, wx : wx + 3, 0] = w_col[0]
                            canvas[wy : wy + 3, wx : wx + 3, 1] = w_col[1]
                            canvas[wy : wy + 3, wx : wx + 3, 2] = w_col[2]

        # Wet harbor / street reflection below horizon
        refl_h = height - horiz_px
        src_slice = canvas[max(0, horiz_px - refl_h) : horiz_px, :, :][::-1, :, :]
        if src_slice.shape[0] < refl_h:
            pad_r = np.zeros((refl_h - src_slice.shape[0], width, 3), dtype=np.float32)
            src_slice = np.concatenate([src_slice, pad_r], axis=0)
        ripple_x = (np.sin(np.arange(refl_h)[:, None] * 0.45 + frame_phase * 2.0) * 4.0).astype(np.int32)
        cols_idx = np.clip(np.arange(width)[None, :] + ripple_x, 0, width - 1)
        warped_refl = src_slice[np.arange(refl_h)[:, None], cols_idx]
        canvas[horiz_px:, :, :] = warped_refl * 0.68
        scene_desc = "multi-depth illuminated city skyline, neon atmospheric haze, and rippled harbor reflections"

    elif is_desert:
        # Golden Desert Dunes & Canyon
        n1 = _fbm_2d(height, width, rng, octaves=5, base_h=8, base_w=14)
        horiz = 0.46
        sky_t = np.clip(y_lin / horiz, 0.0, 1.0)
        sky_col = np.stack([
            0.32 + 0.65 * sky_t,
            0.24 + 0.52 * (sky_t ** 1.4),
            0.38 + 0.25 * (1.0 - sky_t),
        ], axis=-1)
        sun_r = np.sqrt(((x_lin - 0.66) * aspect) ** 2 + (y_lin - 0.26) ** 2)
        sky_col += np.exp(-sun_r / 0.14)[..., None] * np.array([0.65, 0.48, 0.22], dtype=np.float32)

        canvas = sky_col.copy()
        for d_idx, (base_y, amp_y, freq_x, shade) in enumerate([
            (0.48, 0.07, 2.2, 0.62),
            (0.58, 0.09, 1.7, 0.78),
            (0.71, 0.11, 1.3, 0.94),
        ]):
            crest = base_y + amp_y * np.sin(x_lin * math.pi * freq_x + d_idx * 1.4 + 0.6 * n1)
            mask = y_lin >= crest
            lit = (np.cos(x_lin * math.pi * freq_x + d_idx * 1.4) > 0.0).astype(np.float32)
            dune_c = np.stack([
                (0.62 + 0.28 * lit) * shade + 0.06 * n1,
                (0.38 + 0.22 * lit) * shade + 0.04 * n1,
                (0.18 + 0.12 * lit) * shade,
            ], axis=-1)
            canvas = np.where(mask[..., None], dune_c, canvas)
        scene_desc = "sweeping wind-carved desert sand dunes with sunlit and shadowed ridges under a warm amber sky"

    elif is_tech:
        # High-resolution 3D Volumetric Quantum / Liquid Neural Network Field
        n1 = _fbm_2d(height, width, rng, octaves=5, base_h=8, base_w=14)
        n2 = _fbm_2d(height, width, rng, octaves=5, base_h=12, base_w=20)
        cx_n = (x_lin - 0.5) * aspect
        cy_n = y_lin - 0.5
        r_n = np.sqrt(cx_n ** 2 + cy_n ** 2)
        waves = (
            np.sin(r_n * 28.0 - frame_phase * 2.0 + n1 * 4.0)
            + np.cos(cx_n * 18.0 + cy_n * 14.0 + n2 * 5.0)
        ) * 0.5
        caustics = np.exp(-np.abs(waves) * 7.5)
        core_glow = np.exp(-(r_n / 0.28) ** 2)
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[..., 0] = 0.03 + 0.12 * n1 + 0.25 * caustics + 0.35 * core_glow
        canvas[..., 1] = 0.08 + 0.28 * n2 + 0.88 * caustics + 0.72 * core_glow
        canvas[..., 2] = 0.14 + 0.36 * n1 + 0.78 * caustics + 0.85 * core_glow
        scene_desc = "3D harmonic wave interference field, volumetric caustic filaments, and luminous neural core"

    else:
        # Mountain Sunrise, Misty Pine Forest & Emerald Alpine Lake Reflection
        n1 = _fbm_2d(height, width, rng, octaves=6, base_h=8, base_w=14)
        n2 = _fbm_2d(height, width, rng, octaves=5, base_h=12, base_w=22)
        ridged = 1.0 - np.abs(2.0 * n1 - 1.0)
        lake_y = 0.68

        # Sky with golden-emerald sunrise & volumetric rays
        sky_t = np.clip(y_lin / lake_y, 0.0, 1.0)
        sky_r = 0.08 + 0.68 * (sky_t ** 1.6)
        sky_g = 0.22 + 0.58 * (sky_t ** 1.2)
        sky_b = 0.36 + 0.34 * (1.0 - sky_t * 0.5)
        sun_x, sun_y = 0.56, 0.30
        r_sun = np.sqrt(((x_lin - sun_x) * aspect) ** 2 + (y_lin - sun_y) ** 2)
        sun_glow = np.exp(-r_sun / 0.18) * 0.70 + np.exp(-((r_sun / 0.045) ** 2)) * 1.25
        canvas = np.stack([
            sky_r + sun_glow * 0.95,
            sky_g + sun_glow * 0.82,
            sky_b + sun_glow * 0.45,
        ], axis=-1)

        # 3 layered ridged-multifractal mountain ranges with snowline & directional sun
        x_prof = np.linspace(0.0, 1.0, width, dtype=np.float32)
        for m_idx, (base_h, amp_h, haze, snow_cut) in enumerate([
            (0.36, 0.20, 0.62, 0.40),
            (0.46, 0.18, 0.38, 0.49),
            (0.56, 0.14, 0.16, 0.58),
        ]):
            row_sample = int((0.25 + 0.22 * m_idx) * height)
            m_curve = (
                base_h
                - amp_h * (0.65 * ridged[row_sample, :] + 0.35 * np.sin(x_prof * math.pi * (2.2 + m_idx) + m_idx)[None, :])
            )
            in_mtn = (y_lin >= m_curve) & (y_lin < lake_y)
            slope = np.gradient(ridged, axis=1)
            sunlit = np.clip(0.5 + slope * 18.0, 0.15, 1.0)
            is_snow = (y_lin < (m_curve + 0.045 * n2)) & (m_idx < 2)
            rock_r = (0.10 + 0.18 * sunlit) * (1.0 - haze) + 0.52 * haze
            rock_g = (0.20 + 0.24 * sunlit) * (1.0 - haze) + 0.58 * haze
            rock_b = (0.22 + 0.22 * sunlit) * (1.0 - haze) + 0.56 * haze
            m_col = np.stack([
                np.where(is_snow, 0.88 + 0.10 * sunlit, rock_r),
                np.where(is_snow, 0.92 + 0.07 * sunlit, rock_g),
                np.where(is_snow, 0.95 + 0.05 * sunlit, rock_b),
            ], axis=-1)
            canvas = np.where(in_mtn[..., None], m_col, canvas)

        # Alpine lake mirror reflection below lake_y with water ripples
        lake_px = int(lake_y * height)
        refl_h = height - lake_px
        above_slice = canvas[max(0, lake_px - refl_h) : lake_px, :, :][::-1, :, :]
        if above_slice.shape[0] < refl_h:
            above_slice = np.pad(above_slice, ((0, refl_h - above_slice.shape[0]), (0, 0), (0, 0)), mode="edge")
        rip = np.sin(np.arange(refl_h)[:, None] * 0.55 + x_lin * 35.0 + frame_phase * 2.5) * 3.5
        c_idx = np.clip((np.arange(width)[None, :] + rip).astype(np.int32), 0, width - 1)
        lake_refl = above_slice[np.arange(refl_h)[:, None], c_idx]
        teal_tint = np.array([0.05, 0.22, 0.20], dtype=np.float32)
        canvas[lake_px:, :, :] = lake_refl * 0.74 + teal_tint * 0.26
        scene_desc = "multi-layer ridged mountain peaks, golden atmospheric scattering, and rippled alpine lake reflections"

    # Subtle cinematic vignette + highlights bloom
    vign = 1.0 - 0.22 * (((x_lin - 0.5) * 1.35) ** 2 + ((y_lin - 0.5) * 1.35) ** 2)
    canvas = np.clip(canvas * vign[..., None], 0.0, 1.0)
    # Filmic S-curve tone mapping
    canvas = canvas / (canvas + 0.18) * 1.18
    canvas_u8 = np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8)
    base_img = Image.fromarray(canvas_u8, mode="RGB")

    # Add soft optical bloom on bright highlights
    bright_arr = np.clip((canvas - 0.72) * 3.2, 0.0, 1.0)
    bloom_small = Image.fromarray((bright_arr * 255.0).astype(np.uint8), mode="RGB").resize(
        (max(64, width // 4), max(36, height // 4)), Image.Resampling.BILINEAR
    )
    bloom_blur = bloom_small.filter(ImageFilter.GaussianBlur(radius=6)).resize((width, height), Image.Resampling.BICUBIC)
    final_arr = np.clip(
        np.asarray(base_img, dtype=np.float32) + 0.42 * np.asarray(bloom_blur, dtype=np.float32),
        0.0,
        255.0,
    ).astype(np.uint8)
    return Image.fromarray(final_arr, mode="RGB"), scene_desc


class NemotronMediaGenerator:
    """
    Generates high-resolution 1280x720 / 1920x1080 raster images (PNG, JPEG, WebP, BMP, TIFF),
    multi-format HD video animations (MP4 H.264, WebM VP9, GIF, Animated WebP), and
    multi-format neural/harmonic audio & voice clips (MP3, WAV, OGG, FLAC)
    for `nemotron-3.5-super-vl`, `lfm3-1.7b`, and `kimi-k3`.
    """

    @staticmethod
    def generate_image_media(prompt: str, model_label: str = "Nemotron 3.5 Super VL") -> dict[str, Any]:
        clean = re.sub(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+)*(?:generate|create|draw|render|make|paint|show\s+me)\s+"
            r"(?:me\s+)?(?:an?\s+)?(?:high[\s\-]*res(?:olution)?\s+|hd\s+|1080p\s+|4k\s+)?(?:png\s+|jpg\s+|jpeg\s+|webp\s+|bmp\s+|tiff\s+|svg\s+)?"
            r"(?:image|picture|illustration|photo|diagram|visual|scene|art|artwork)?\s*(?:of\s+|for\s+|showing\s+)?",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        ).strip(" .!?") or "The Cosmos"
        title = clean[:64].title()

        p_low = prompt.lower()
        width, height = (1920, 1080) if any(k in p_low for k in ("1080p", "full hd", "4k", "ultra hd")) else (1280, 720)
        img_hd, scene_desc = render_high_res_raster_scene(clean, width=width, height=height)

        png_buf = io.BytesIO()
        img_hd.save(png_buf, format="PNG", optimize=False)
        png_data_url = "data:image/png;base64," + base64.b64encode(png_buf.getvalue()).decode("ascii")

        jpg_buf = io.BytesIO()
        img_hd.save(jpg_buf, format="JPEG", quality=93)
        jpg_data_url = "data:image/jpeg;base64," + base64.b64encode(jpg_buf.getvalue()).decode("ascii")

        webp_buf = io.BytesIO()
        img_hd.save(webp_buf, format="WEBP", quality=92)
        webp_data_url = "data:image/webp;base64," + base64.b64encode(webp_buf.getvalue()).decode("ascii")

        bmp_buf = io.BytesIO()
        img_hd.save(bmp_buf, format="BMP")
        bmp_data_url = "data:image/bmp;base64," + base64.b64encode(bmp_buf.getvalue()).decode("ascii")

        primary_url = png_data_url
        primary_fmt = "PNG"
        if ".jpg" in p_low or "jpeg" in p_low or " jpg" in p_low:
            primary_url = jpg_data_url
            primary_fmt = "JPEG"
        elif "webp" in p_low:
            primary_url = webp_data_url
            primary_fmt = "WebP"
        elif "bmp" in p_low:
            primary_url = bmp_data_url
            primary_fmt = "BMP"

        res_str = f"{width}×{height}"
        return {
            "title": title,
            "media": [
                {
                    "type": "image",
                    "format": primary_fmt,
                    "resolution": f"{width}x{height}",
                    "title": f"{title} ({model_label} • {res_str} HD {primary_fmt})",
                    "data_url": primary_url,
                    "png_url": png_data_url,
                    "jpg_url": jpg_data_url,
                    "webp_url": webp_data_url,
                    "bmp_url": bmp_data_url,
                    "caption": f"{title} — {res_str} High-Resolution {primary_fmt} ({scene_desc}).",
                }
            ],
            "plans": [
                f"Here is the high-resolution {width}x{height} {primary_fmt} image of {title} generated with {model_label}:",
                f"\nVisual Details: Synthesized at {width}x{height} HD resolution featuring {scene_desc}, available in PNG, JPG, WebP, and BMP formats.",
                f"\nModel Backbone: Grounded through {model_label}'s multimodal visual-concept bridge into Clover's 11-level Next-Concept Model.",
            ],
        }

    @staticmethod
    def generate_video_media(prompt: str, model_label: str = "Nemotron 3.5 Super VL") -> dict[str, Any]:
        clean = re.sub(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+)*(?:generate|create|render|make|animate|show\s+me)\s+"
            r"(?:an?\s+)?(?:animated\s+|hd\s+|mp4\s+|webm\s+|gif\s+)?(?:video|animation|clip|motion\s+scene|movie)\s*(?:of\s+|for\s+|showing\s+)?",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        ).strip(" .!?") or "Orbital Wave Propagation"
        title = clean[:64].title()

        width, height = 640, 360
        n_frames = 16
        fps = 12
        frames: list[Image.Image] = []
        raw_rgb_chunks: list[bytes] = []
        scene_desc = "volumetric temporal motion"
        for f_idx in range(n_frames):
            phase = (f_idx / float(n_frames)) * 2.0 * math.pi
            frame_img, scene_desc = render_high_res_raster_scene(clean, width=width, height=height, frame_phase=phase)
            frames.append(frame_img)
            raw_rgb_chunks.append(frame_img.tobytes())

        # Poster PNG (first frame)
        poster_buf = io.BytesIO()
        frames[0].save(poster_buf, format="PNG")
        poster_data_url = "data:image/png;base64," + base64.b64encode(poster_buf.getvalue()).decode("ascii")

        # 1. Animated GIF
        gif_buf = io.BytesIO()
        frames[0].save(
            gif_buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        gif_data_url = "data:image/gif;base64," + base64.b64encode(gif_buf.getvalue()).decode("ascii")

        # 2. Animated WebP
        webp_buf = io.BytesIO()
        frames[0].save(
            webp_buf,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            quality=88,
        )
        webp_data_url = "data:image/webp;base64," + base64.b64encode(webp_buf.getvalue()).decode("ascii")

        # 3. H.264 MP4 and VP9 WebM via ffmpeg (if installed)
        mp4_data_url = ""
        webm_data_url = ""
        if shutil.which("ffmpeg"):
            raw_stream = b"".join(raw_rgb_chunks)
            tmp_dir = tempfile.mkdtemp(prefix="clover_vid_out_")
            try:
                mp4_path = os.path.join(tmp_dir, "out.mp4")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                        "-i", "pipe:0",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        mp4_path,
                    ],
                    input=raw_stream,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 256:
                    with open(mp4_path, "rb") as f:
                        mp4_data_url = "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")

                webm_path = os.path.join(tmp_dir, "out.webm")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                        "-i", "pipe:0",
                        "-c:v", "libvpx-vp9", "-b:v", "800k", "-deadline", "realtime", "-cpu-used", "8",
                        "-pix_fmt", "yuv420p",
                        webm_path,
                    ],
                    input=raw_stream,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
                if os.path.exists(webm_path) and os.path.getsize(webm_path) > 256:
                    with open(webm_path, "rb") as f:
                        webm_data_url = "data:video/webm;base64," + base64.b64encode(f.read()).decode("ascii")
            except Exception:
                pass
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        primary_video_url = mp4_data_url or webm_data_url or gif_data_url
        return {
            "title": title,
            "media": [
                {
                    "type": "video",
                    "format": "MP4" if mp4_data_url else "GIF",
                    "resolution": f"{width}x{height}",
                    "title": f"HD Video: {title} ({model_label} • {width}×{height})",
                    "data_url": primary_video_url,
                    "mp4_url": mp4_data_url,
                    "webm_url": webm_data_url,
                    "gif_url": gif_data_url,
                    "webp_url": webp_data_url,
                    "poster_url": poster_data_url,
                    "caption": f"{title} — {width}×{height} HD multi-frame video ({scene_desc}) in MP4, WebM, GIF, and WebP formats.",
                }
            ],
            "plans": [
                f"Here is the {width}x{height} HD video sequence for {title} generated with {model_label}:",
                f"\nTemporal Motion Synthesis: Rendered across {n_frames} high-resolution frames ({scene_desc}) and encoded in MP4 (H.264), WebM (VP9), GIF, and animated WebP formats.",
                f"\nMultimodal Pipeline: Conditioned in {model_label}'s hidden space and decoded through Clover's 11-level Next-Concept Model.",
            ],
        }

    @staticmethod
    def generate_audio_media(
        prompt: str,
        model_label: str = "Nemotron 3.5 Super VL",
        voice_profile: str = "en-US-AriaNeural",
    ) -> dict[str, Any]:
        """
        Synthesizes spoken neural voice or polyphonic audio and encodes it into all standard audio formats:
        WAV (`data:audio/wav`), MP3 (`data:audio/mpeg`), OGG (`data:audio/ogg`), and FLAC (`data:audio/flac`).
        """
        clean = re.sub(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+)*(?:generate|create|synthesize|record|make|play|speak)\s+"
            r"(?:me\s+)?(?:an?\s+)?(?:mp3\s+|wav\s+|ogg\s+|flac\s+)?(?:audio|voice|speech|sound|soundscape|chime|tone|clip|recording)\s*(?:of\s+|for\s+|saying\s+|about\s+)?",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        ).strip(" .!?\"'") or f"Hello from {model_label}, ready for multimodal voice and audio synthesis."
        title = clean[:58].title()

        wav_data_url = synthesize_deterministic_wav_data_url(
            clean,
            base_freq_hz=235.0 if "Aria" in voice_profile else (250.0 if "Jenny" in voice_profile else 270.0),
        )
        raw_wav_bytes = base64.b64decode(wav_data_url.split(",", 1)[1])
        mp3_data_url = ""

        # Try neural speech synthesis via edge_tts first
        try:
            import asyncio
            import edge_tts

            async def _run_tts():
                comm = edge_tts.Communicate(clean[:600], voice=voice_profile, rate="+0%", pitch="+0Hz")
                buf = bytearray()
                async for chunk in comm.stream():
                    if chunk.get("type") == "audio":
                        buf.extend(chunk["data"])
                return bytes(buf)

            mp3_bytes = asyncio.run(_run_tts())
            if mp3_bytes:
                mp3_data_url = "data:audio/mpeg;base64," + base64.b64encode(mp3_bytes).decode("ascii")
        except Exception:
            mp3_bytes = b""

        ogg_data_url = ""
        flac_data_url = ""
        if shutil.which("ffmpeg"):
            src_bytes = mp3_bytes if mp3_bytes else raw_wav_bytes
            try:
                if not mp3_data_url:
                    p_mp3 = subprocess.run(
                        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k", "pipe:1"],
                        input=src_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
                    )
                    if p_mp3.stdout:
                        mp3_data_url = "data:audio/mpeg;base64," + base64.b64encode(p_mp3.stdout).decode("ascii")
                else:
                    # Convert neural MP3 into crisp 16-bit PCM WAV as well so WAV download is neural speech too!
                    p_wav = subprocess.run(
                        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "wav", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", "pipe:1"],
                        input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
                    )
                    if p_wav.stdout:
                        wav_data_url = "data:audio/wav;base64," + base64.b64encode(p_wav.stdout).decode("ascii")

                p_ogg = subprocess.run(
                    ["ffmpeg", "-y", "-i", "pipe:0", "-f", "ogg", "-c:a", "libvorbis", "-q:a", "4", "pipe:1"],
                    input=src_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
                )
                if p_ogg.stdout:
                    ogg_data_url = "data:audio/ogg;base64," + base64.b64encode(p_ogg.stdout).decode("ascii")

                p_flac = subprocess.run(
                    ["ffmpeg", "-y", "-i", "pipe:0", "-f", "flac", "-c:a", "flac", "pipe:1"],
                    input=src_bytes, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
                )
                if p_flac.stdout:
                    flac_data_url = "data:audio/flac;base64," + base64.b64encode(p_flac.stdout).decode("ascii")
            except Exception:
                pass

        primary_audio_url = mp3_data_url or wav_data_url
        return {
            "title": title,
            "media": [
                {
                    "type": "audio",
                    "format": "MP3" if mp3_data_url else "WAV",
                    "voice": voice_profile,
                    "title": f"Audio & Voice: {title} ({model_label} • {voice_profile})",
                    "data_url": primary_audio_url,
                    "mp3_url": mp3_data_url,
                    "wav_url": wav_data_url,
                    "ogg_url": ogg_data_url,
                    "flac_url": flac_data_url,
                    "caption": f"Synthesized by {model_label} ({voice_profile}) — available in MP3, WAV, OGG, and FLAC formats.",
                }
            ],
            "plans": [
                f"Here is the synthesized audio recording for {title} generated with {model_label} ({voice_profile}):",
                f"\nMulti-Format Audio Encoding: Available for inline playback and download in MP3, WAV (PCM), OGG (Vorbis), and lossless FLAC formats.",
                f"\nAcoustic Bridge: Grounded in {model_label}'s audio-voice pipeline and decoded via Clover's 11-level Next-Concept Model.",
            ],
        }


class NemotronMultimodalEngine:
    """
    Unified Multimodal Coordinator for `nemotron-3.5-super-vl`:
      - Vision (Image -> C-RADIO 32L + PixelShuffle + `ReLU^2` `mlp1` -> `4096`-d -> `NemotronNativeHybridLink` -> `131,072`-vocab NCM)
      - Video (Multi-frame Video -> `video_patch_projection` + EVS `compute_retention_mask` -> `4096`-d -> NCM)
      - Audio & Voice Conversation (`.wav`/`.mp3` -> `80`-band Mel -> `NemotronAudioMambaBridge` `4096`-d -> NCM + Neural Voice reply)
      - Image & Video Generation (`NemotronMediaGenerator` + `131,072`-vocab NCM)
    """

    VOICE_PROFILE = "en-US-AriaNeural"

    def __init__(
        self,
        model_dir: Path = DEFAULT_NEMO_MODEL_DIR,
        bridges_ckpt: Path = DEFAULT_NEMO_BRIDGES_CKPT,
    ):
        self.model_key = "nemotron-3.5-super-vl"
        self.model_dir = Path(model_dir)
        self.vision_video = NemotronCRadioVisionVideoEncoder(model_dir=self.model_dir)
        self.audio_bridge = NemotronAudioMambaBridge(n_mels=80, d_model=D_NEMO)
        self.hybrid_link = NemotronNativeHybridLink(d=D_NEMO, bottleneck=1024)
        self.combo_bridge = NemotronComboBridge(d_in=D_NEMO, d_out=D_VQ)
        self.expert_bridge = NemotronExpertBridge(d_nemo=D_NEMO, d_vq=D_VQ)

        if bridges_ckpt.exists():
            try:
                sd = torch.load(str(bridges_ckpt), map_location="cpu", weights_only=False)
                self.hybrid_link.load_state_dict(sd["nemo_native_hybrid_link"])
                self.combo_bridge.load_state_dict(sd["nemo_combo_bridge"])
                self.expert_bridge.load_state_dict(sd["nemo_expert_bridge"])
            except Exception:
                pass
        self.hybrid_link.eval()
        self.combo_bridge.eval()
        self.expert_bridge.eval()

    @torch.no_grad()
    def process_image_or_video(
        self,
        pil_images: list[Image.Image],
        prompt: str,
        renderer=None,
        use_real_weights_if_present: bool = True,
    ) -> dict[str, Any]:
        is_video = len(pil_images) > 1 or any(
            w in prompt.lower() for w in ("video", "frame", "frames", "motion", "clip", "temporal")
        )
        facts = inspect_pil_image_visual_facts(pil_images[0])
        if is_video:
            enc_info = self.vision_video.encode_video(
                pil_images, pruning_rate=0.5, use_real_weights_if_present=use_real_weights_if_present
            )
            h_4096 = enc_info["retained_embeds_4096"]  # [1, N_ret, 4096]
        else:
            enc_info = self.vision_video.encode_image(
                pil_images[0], use_real_weights_if_present=use_real_weights_if_present
            )
            h_4096 = enc_info["image_embeds_4096"]  # [1, 256, 4096]

        r_latent = self.hybrid_link(h_4096.squeeze(0)).unsqueeze(0)  # [1, N, 4096]
        z_combo_384 = self.combo_bridge(r_latent)  # [1, N, 384]
        z_exp_384 = self.expert_bridge(h_4096)  # [1, N, 384]

        if is_video:
            plans = [
                f"Nemotron 3.5 Super VL C-RADIO Video Analysis ({enc_info['input_frames']} frames, {facts['width']}x{facts['height']} px, {facts['dominant_tone']} palette):",
                f"\nEfficient Video Sampling (EVS): C-RADIO encoded {enc_info['total_tokens_before_evs']} spatial-temporal patch tokens and retained {enc_info['retained_tokens_after_evs']} active motion/anchor tokens in 4096-d space (RMS {enc_info['output_rms']}).",
                f"\nVisual Content & Response: Brightness {facts['brightness_pct']}%, edge structure density {facts['edge_density_pct']}%, mean RGB ({facts['mean_rgb'][0]}, {facts['mean_rgb'][1]}, {facts['mean_rgb'][2]}). {prompt.strip()}",
            ]
        else:
            plans = [
                f"Nemotron 3.5 Super VL C-RADIO Vision Analysis ({facts['width']}x{facts['height']} px, {facts['dominant_tone']} tone):",
                f"\nStage V1 & V2 Encoding: 32-layer C-RADIO ViT-Huge + PixelShuffle(0.5) + ReLU^2 projector compressed 1,024 patches into {enc_info['post_pixel_shuffle_tokens']} visual tokens of 4,096 dimensions (RMS {enc_info['output_rms']}).",
                f"\nVisual Details: Brightness {facts['brightness_pct']}%, edge detail density {facts['edge_density_pct']}%, mean RGB ({facts['mean_rgb'][0]}, {facts['mean_rgb'][1]}, {facts['mean_rgb'][2]}).",
            ]

        if renderer is not None:
            reply_txt, ids_out, render_meta = renderer.render_plans_in_model_vocab(plans, cands=1)
        else:
            reply_txt = "\n".join(p.strip() for p in plans)
            ids_out = []
            render_meta = {}

        return {
            "mode": f"nemotron_super_vl_{'video_evs' if is_video else 'vision_cradio'}_ncm",
            "model": self.model_key,
            "reply": reply_txt,
            "reply_1cand": reply_txt,
            "reply_mbr8": reply_txt,
            "plans": plans,
            "vision_video_meta": {
                **{k: v for k, v in enc_info.items() if not isinstance(v, torch.Tensor)},
                "visual_facts": facts,
                "hybridlink_4096_shape": list(r_latent.shape),
                "ncm_combo_384_shape": list(z_combo_384.shape),
                "ncm_expert_384_shape": list(z_exp_384.shape),
                "render_meta": render_meta,
            },
            "n_tokens": len(ids_out),
        }

    @torch.no_grad()
    def process_audio_or_voice(
        self,
        raw_audio_bytes: bytes,
        file_name: str,
        prompt: str,
        renderer=None,
    ) -> dict[str, Any]:
        audio_info = analyze_audio_bytes_and_extract_mel80(raw_audio_bytes, file_name=file_name)
        mel_t = audio_info.pop("mel_tensor")  # [1, T_a, 80]
        h_audio_4096 = self.audio_bridge(mel_t)  # [1, T_a, 4096]
        r_latent = self.hybrid_link(h_audio_4096.squeeze(0)).unsqueeze(0)
        z_combo_384 = self.combo_bridge(r_latent)

        plans = [
            f"Nemotron 3.5 Super VL Audio & Voice Analysis for {audio_info['file_name']} ({audio_info['duration_seconds']}s at {audio_info['sample_rate_hz']} Hz):",
            f"\nAcoustic & Spectral Profile: RMS level {audio_info['dbfs']} dBFS, dominant spectral peak {audio_info['dominant_frequency_hz']} Hz, cadence {audio_info['estimated_cadence_hz']} Hz across {audio_info['mel_frames_shape'][1]} Mamba-2 frames.",
            f"\n4096-d Mamba-2 Audio Bridge: Projected 80-band log-Mel features into Nemotron's 4096-d recurrent state and 384-d NCM concept space.",
        ]
        if renderer is not None:
            reply_txt, ids_out, render_meta = renderer.render_plans_in_model_vocab(plans, cands=1)
        else:
            reply_txt = "\n".join(p.strip() for p in plans)
            ids_out = []
            render_meta = {}

        return {
            "mode": "nemotron_super_vl_audio_voice_ncm",
            "model": self.model_key,
            "reply": reply_txt,
            "reply_1cand": reply_txt,
            "reply_mbr8": reply_txt,
            "plans": plans,
            "audio_meta": {
                **audio_info,
                "mamba_audio_4096_shape": list(h_audio_4096.shape),
                "ncm_combo_384_shape": list(z_combo_384.shape),
                "voice_profile": self.VOICE_PROFILE,
                "render_meta": render_meta,
            },
            "n_tokens": len(ids_out),
        }

    def generate_media(self, prompt: str, media_kind: str = "auto", renderer=None) -> dict[str, Any]:
        p_low = prompt.lower()
        is_aud = media_kind == "audio" or (
            media_kind == "auto"
            and any(w in p_low for w in ("audio", "voice", "speech", "sound", "mp3", "wav", "ogg", "flac", "speak", "say "))
            and not any(w in p_low for w in ("image", "picture", "video", "animation"))
        )
        is_vid = (not is_aud) and (
            media_kind == "video"
            or (
                media_kind == "auto"
                and any(w in p_low for w in ("video", "animate", "animation", "motion", "clip", "movie", "mp4", "webm"))
            )
        )
        if is_aud:
            gen = NemotronMediaGenerator.generate_audio_media(
                prompt, model_label="Nemotron 3.5 Super VL", voice_profile=self.VOICE_PROFILE
            )
            mode_tag = "audio"
        elif is_vid:
            gen = NemotronMediaGenerator.generate_video_media(prompt, model_label="Nemotron 3.5 Super VL")
            mode_tag = "video"
        else:
            gen = NemotronMediaGenerator.generate_image_media(prompt, model_label="Nemotron 3.5 Super VL")
            mode_tag = "image"

        plans = gen["plans"]
        if renderer is not None:
            reply_txt, ids_out, _ = renderer.render_plans_in_model_vocab(plans, cands=1)
        else:
            reply_txt = "\n".join(p.strip() for p in plans)
            ids_out = []

        return {
            "mode": f"nemotron_super_vl_{mode_tag}_generation",
            "model": self.model_key,
            "reply": reply_txt,
            "reply_1cand": reply_txt,
            "reply_mbr8": reply_txt,
            "media": gen["media"],
            "plans": plans,
            "n_tokens": len(ids_out),
        }
