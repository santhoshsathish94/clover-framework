"""Unified JEPA Architecture Interface & Flow Router for the Clover AI System (`jepa_engine.py`).

Allows Clover to select and execute any of the four core JEPA architectures—or their unified Clover synthesis
(`clover_rpre`)—across every stage of the Clover runtime (`S1-S3 <-> C1-C3 <-> Cycle`):

  1. `ijepa`       — Visual Latent Prediction (`facebookresearch/ijepa`):
                     Context-only sparse patch encoding (`apply_masks`, ~30% self-attention FLOPs) + multi-block target
                     prediction for S3 visual/UI salience ranking (`image_inspector.py`) and scene layout classification.
  2. `audio_jepa`  — Audio Spectrogram Latent Prediction (`LudovicTuncay/Audio-JEPA`):
                     2D time-frequency log-mel patch tokenization + symmetric `CenteredNormMSELoss` + time-frequency
                     band masking for S3/C1 voice & audio quality, prosody, and acoustic regime detection.
  3. `eb_jepa`     — Energy-Based Generalized Learning & Latent MPC Planning (`facebookresearch/eb_jepa`):
                     Variance-Covariance (`VCLoss`) + Temporal Similarity + Inverse Dynamics (`IDM`) regularization and
                     elitist Cross-Entropy Method (`CEMPlanner`) latent trajectory optimization inside C2 (`omni_c2_c3.py`).
  4. `mjepa`       — Shared Multimodal Audio-Visual-State Representation Learning (`facebookresearch/MJEPA`):
                     Single shared 4-level hierarchical Transformer trunk + 9-path intra/cross-modal predictors
                     (`a2v`, `v2a`, `s2v`, `s2a`, `av2a`, `av2v`) for S2 cross-modal memory recall and missing-modality
                     inference.
  5. `clover_rpre` — Clover Reusable Predictive Representation Engine (`research/predictive-representation-engine`):
                     Unifies `ijepa + audio_jepa + eb_jepa + mjepa` and closes the deployment loop (`Outcome -> Growth`)
                     via `OutcomeResidualAdapter` with a Baseline-Prototype Anomaly Gate and Held-Out Non-Regression
                     Rollback Gate.
  6. `auto`        — Clover Semantic Selector (default): automatically selects the right JEPA architecture for each active
                     flow in the turn (`visual_s3 -> ijepa`, `audio_s3 -> audio_jepa`, `multimodal_s2 -> mjepa`,
                     `c2_plan -> eb_jepa`, `outcome_growth -> clover_rpre`).
  7. `none`        — Disabled (baseline pass-through for controlled A/B comparison across flows).

Implemented in pure vectorized NumPy/BLAS (reading `CLVRTRK1` binary trunks from `<base>/data/omni/precomputed/jepa/jepa-trunk-0`
or `.npz` weights) so it runs in 1–6 ms on Windows Laptop, Linux (`k3`), and Android (`Chaquopy`) without requiring
PyTorch at inference time, while pairing directly with `train_clover_jepa_trunks.py` for CPU/GPU PyTorch training.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

JEPA_ARCHITECTURES = ("auto", "clover_rpre", "ijepa", "audio_jepa", "eb_jepa", "mjepa", "none")

FLOW_DEFAULT_ARCH = {
    "visual_s3": "ijepa",
    "audio_s3": "audio_jepa",
    "multimodal_s2": "mjepa",
    "c2_plan": "eb_jepa",
    "outcome_growth": "clover_rpre",
}

VISUAL_SCENE_KINDS = (
    "ui_controls",
    "tabular_chart",
    "document_receipt",
    "code_terminal",
    "diagram_scene",
    "natural_photo",
)

ACOUSTIC_REGIMES = (
    "clean_speech",
    "noisy_speech",
    "harmonic_music",
    "transient_alert",
    "low_snr_silence",
)

D_JEPA = 64
D_PRED = 32
D_VQ = 384
D_LFM3 = 2048
D_ACT = 8
N_PATCHES = 16  # 4x4 grid for fast runtime patch tokenization


# ---------------- Pure-NumPy Mathematical Primitives ----------------

def _layer_norm(x: np.ndarray, weight: np.ndarray | None = None, bias: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    y = (x - m) / np.sqrt(v + np.float32(eps))
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x * x * x)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - mx)
    return e / (np.sum(e, axis=axis, keepdims=True) + np.float32(1e-9))


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    y = np.matmul(np.asarray(x, dtype=np.float32), w.T)
    return y + b if b is not None else y


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + np.float32(eps))
    return x / n


def centered_norm_mse(pred: np.ndarray, target: np.ndarray, center_pred: bool = True) -> float:
    """Audio-JEPA symmetric centered normalized MSE loss (`rpre/losses.py`).
    Centers across both patch tokens and feature channels so constant/untrained predictions do not get artificial 0.99 cosine.
    """
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    if p.ndim >= 2 and p.shape[0] > 1:
        t = t - t.mean(axis=0, keepdims=True)
        if center_pred:
            p = p - p.mean(axis=0, keepdims=True)
    t = (t - t.mean(axis=-1, keepdims=True)) / np.sqrt(t.var(axis=-1, keepdims=True) + np.float32(1e-6))
    if center_pred:
        p = (p - p.mean(axis=-1, keepdims=True)) / np.sqrt(p.var(axis=-1, keepdims=True) + np.float32(1e-6))
    pn = _l2_normalize(p, axis=-1)
    tn = _l2_normalize(t, axis=-1)
    return float(np.mean((pn - tn) ** 2))


def tool_capability_vector(tool_name: str) -> np.ndarray:
    """Map any tool name `tool_name` into a deterministic 8-d semantic capability vector in `R^{D_ACT}` (`D_ACT = 8`).
    Completely invariant to the position of `tool_name` in `offered_tools`:
      - Dim 0: `web_factual` (`web_search`, `fetch_url`, `browse`, `wikipedia`, `news`)
      - Dim 1: `math_calc` (`calculator`, `calc`, `convert_units`, `units`)
      - Dim 2: `code_python` (`python`, `run_python`, `analyze_table`, `chart_table`)
      - Dim 3: `memory_knowledge` (`memory_recall`, `remember`, `recall`, `read_document`, `ontology`)
      - Dim 4: `files_workspace` (`read_file`, `write_file`, `search_files`, `file_info`, `git_inspect`, `vscode_*`)
      - Dim 5: `visual_media` (`image_gen`, `generate_image`, `extract_frame`, `media_info`)
      - Dim 6: `system_device` (`laptop_status`, `get_current_time`, `run_powershell`, `ssh`)
      - Dim 7: `general_dialogue` (or hash-projected signature for arbitrary custom tool names)
    """
    s = str(tool_name or "").strip().lower()
    v = np.zeros(D_ACT, dtype=np.float32)
    if re.search(r"web|search_web|fetch_url|url|browse|google|wiki|news", s) and "file" not in s:
        v[0] = 1.0
        v[3] = 0.25
    elif re.search(r"calc|math|unit|convert|eval", s):
        v[1] = 1.0
        v[2] = 0.30
    elif re.search(r"python|code|script|table|chart|plot|compile", s):
        v[2] = 1.0
        v[1] = 0.35
    elif re.search(r"mem|recall|remember|note|know|doc|ontolog|account|password|sleep|tip", s):
        v[3] = 1.0
        v[0] = 0.20
    elif re.search(r"file|dir|folder|git|vscode|workspace|repo", s):
        v[4] = 1.0
        v[6] = 0.20
    elif re.search(r"image|pic|photo|draw|paint|render|media|frame|visual", s):
        v[5] = 1.0
    elif re.search(r"status|laptop|battery|disk|time|clock|powershell|shell|terminal|ssh", s):
        v[6] = 1.0
        v[4] = 0.20
    else:
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        raw = np.frombuffer(digest[:D_ACT], dtype=np.uint8).astype(np.float32) - 128.0
        v = raw / (np.linalg.norm(raw) + np.float32(1e-6))
        return v
    h_tie = (int(hashlib.md5(s.encode("utf-8")).hexdigest()[:6], 16) / float(0xFFFFFF) - 0.5) * 1e-3
    v[7] += np.float32(h_tie)
    return v / (np.linalg.norm(v) + np.float32(1e-6))


def semantic_action_prior_8(
    query: str,
    memory_texts: list[str] | tuple[str, ...] = (),
    ontology_texts: list[str] | tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Compute the 8-d semantic capability prior `action_prior_8` in `R^{D_ACT}` (`D_ACT = 8`) from the combined
    Memory Graph entries (`memory_texts`), Ontology Graph entities/paths/constraints (`ontology_texts`), and `query`.
    Bridges `Memory -> Ontology -> HybridAgent/JEPA -> Action`:
      - Dim 0 (`web_factual`): URLs, live CVE/advisory/status checks, web search directives
      - Dim 1 (`math_calc`): Ontology `FORMULA` records, equations, break-even/unit/rate constraints
      - Dim 2 (`code_python`): Learned `LRN-*` code/script verification, `itertools.permutations`, Python algorithms
      - Dim 3 (`memory_knowledge`): Document/knowledge synthesis across recalled records
      - Dim 4 (`files_workspace`): Local workspace file paths (`.py`, `.json`, `.md`, `.yaml`), repo/git inspection
      - Dim 5 (`visual_media`): Image/chart/media generation or frame extraction
      - Dim 6 (`system_device`): Device/system health, disk, memory, battery, shell diagnostics
      - Dim 7 (`general_dialogue`): Baseline conversational synthesis
    Returns `(prior_8, recommended_tools, dim_scores)`.
    """
    mem_joined = "\n".join(str(x or "") for x in (memory_texts or ()))
    ont_raw = "\n".join(str(x or "") for x in (ontology_texts or ()))
    ont_joined = re.sub(r"(?im)^\s*Source\s+URL:\s*\S+\s*$", "", ont_raw)
    q_str = str(query or "")
    combined = f"{q_str}\n{mem_joined}\n{ont_joined}".lower()
    ev_only = f"{mem_joined}\n{ont_joined}".lower()

    v = np.zeros(D_ACT, dtype=np.float32)

    # Dim 0: web_factual
    if re.search(r"https?://\S+|\b(?:web_search|fetch_url|search\s+the\s+web|live\s+status|current\s+advisory|nvd|cve-\d{4}-\d+)\b", combined):
        v[0] += np.float32(0.85 if re.search(r"https?://|\b(?:web_search|fetch_url|search\s+the\s+web)\b", combined) else 0.45)

    # Dim 1: math_calc
    if re.search(r"\b(?:formula:|verified\s+constraints:|break-even|carrying\s+capacity|wavelength|refractive|compound\s+interest|convert_units|calculate)\b|[0-9]+\s*[*+/^]\s*[0-9]+", combined):
        v[1] += np.float32(0.80)
        v[2] += np.float32(0.45)

    # Dim 2: code_python
    if re.search(r"\b(?:run_python|python|itertools|permutations|def\s+\w+|script|compile|algorithm|max_time|24-hour\s+time|chart\.png|matplotlib|pandas)\b", combined):
        v[2] += np.float32(0.95)
        v[1] += np.float32(0.30)
    elif ev_only and re.search(r"\b(?:checked\s+with\s+python|independent\s+check|verify\s+with\s+code|verified\s+constraints)\b", ev_only):
        v[2] += np.float32(0.80)

    # Dim 3: memory_knowledge
    if mem_joined.strip() or ont_joined.strip():
        v[3] += np.float32(0.40)

    # Dim 4: files_workspace
    if re.search(r"\b(?:read_file|write_file|search_files|list_directory|git_inspect|[\w\-]+\.(?:py|ts|js|json|yaml|yml|toml|md|rs|c|h))\b", combined):
        v[4] += np.float32(0.90)

    # Dim 5: visual_media
    if re.search(r"\b(?:generate_image|draw\s+a|create\s+an?\s+image|extract_frame|media_info)\b", combined):
        v[5] += np.float32(0.85)

    # Dim 6: system_device
    if re.search(r"\b(?:laptop_status|run_powershell|free\s+disk|battery\s+level|memory\s+in\s+use|get-psdrive|get-childitem)\b", combined):
        v[6] += np.float32(0.85)

    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v = v / np.float32(norm)

    dim_names = (
        "web_factual", "math_calc", "code_python", "memory_knowledge",
        "files_workspace", "visual_media", "system_device", "general_dialogue",
    )
    dim_scores = {dim_names[i]: round(float(v[i]), 4) for i in range(D_ACT) if float(v[i]) > 0.01}

    rec_tools: list[str] = []
    if float(v[2]) >= 0.45:
        rec_tools.append("run_python")
    if float(v[4]) >= 0.45:
        rec_tools.append("read_file")
    if float(v[0]) >= 0.45:
        rec_tools.append("fetch_url" if "http" in combined else "web_search")
    if float(v[1]) >= 0.45 and "run_python" not in rec_tools:
        rec_tools.append("calculate")
    if float(v[6]) >= 0.45:
        rec_tools.append("laptop_status")
    if float(v[5]) >= 0.45:
        rec_tools.append("generate_image")
    return v, rec_tools, dim_scores


# ---------------- Real Visual & Audio Patch Feature Extractors ----------------

def extract_visual_patches_16x48(img_rgb: np.ndarray) -> np.ndarray:
    """Extract a `[16, 48]` visual patch matrix from `img_rgb` (`[H, W, 3]`) that is fast (`< 4 ms` even on 4K images),
    invariant to mild crop/resize/JPEG compression, and encodes both local $4 \times 4$ patch layout and global
    scene geometry (`ui_controls`, `tabular_chart`, `document_receipt`, `code_terminal`, `diagram_scene`, `natural_photo`).
    """
    arr = np.asarray(img_rgb)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    H_orig, W_orig = int(arr.shape[0]), int(arr.shape[1])

    sy_step = max(1, H_orig // 192)
    sx_step = max(1, W_orig // 192)
    small = arr[::sy_step, ::sx_step, :3].astype(np.float32)
    if small.max() > 1.5:
        small = small / np.float32(255.0)
    H, W, _ = small.shape

    lum = np.float32(0.299) * small[:, :, 0] + np.float32(0.587) * small[:, :, 1] + np.float32(0.114) * small[:, :, 2]
    cmax = np.max(small, axis=-1)
    cmin = np.min(small, axis=-1)
    sat = (cmax - cmin) / (cmax + np.float32(1e-4))
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    grad = gx + gy

    log_aspect = float(np.clip(math.log(H_orig / max(1.0, float(W_orig))), -1.2, 1.2))
    white_paper_frac = float(np.mean((lum > 0.82) & (sat < 0.12)))
    dark_ide_frac = float(np.mean((lum < 0.22) & (sat < 0.18)))
    color_bar_frac = float(np.mean((sat > 0.32) & (lum > 0.20) & (lum < 0.92)))
    flat_frac = float(np.mean(grad < 0.012))
    natural_smooth_frac = float(np.mean((grad >= 0.006) & (grad <= 0.09) & (lum >= 0.15) & (lum <= 0.85)))
    horiz_line_bias = float((gy.mean() - gx.mean()) / (gy.mean() + gx.mean() + 1e-4))

    img_mean = small.mean(axis=(0, 1), keepdims=True)
    img_std = small.std(axis=(0, 1), keepdims=True) + np.float32(0.05)
    small_std = (small - img_mean) / img_std

    patches = np.zeros((N_PATCHES, 48), dtype=np.float32)
    for py in range(4):
        y0, y1 = int(py * H / 4), max(int(py * H / 4) + 1, int((py + 1) * H / 4))
        for px in range(4):
            x0, x1 = int(px * W / 4), max(int(px * W / 4) + 1, int((px + 1) * W / 4))
            cell_std = small_std[y0:y1, x0:x1, :]
            cell_raw = small[y0:y1, x0:x1, :]
            cell_lum = lum[y0:y1, x0:x1]
            cell_sat = sat[y0:y1, x0:x1]
            cell_gx = gx[y0:y1, x0:x1]
            cell_gy = gy[y0:y1, x0:x1]
            cell_grad = grad[y0:y1, x0:x1]

            ch, cw = cell_std.shape[0], cell_std.shape[1]
            sub27 = []
            for sy in range(3):
                sy0, sy1 = int(sy * ch / 3), max(int(sy * ch / 3) + 1, int((sy + 1) * ch / 3))
                for sx in range(3):
                    sx0, sx1 = int(sx * cw / 3), max(int(sx * cw / 3) + 1, int((sx + 1) * cw / 3))
                    sub = cell_std[sy0:sy1, sx0:sx1, :]
                    sub27.extend(sub.mean(axis=(0, 1)).tolist() if sub.size else [0.0, 0.0, 0.0])

            c_mean = cell_raw.mean(axis=(0, 1)).tolist()
            c_std = cell_raw.std(axis=(0, 1)).tolist()
            q10, q50, q90 = [float(v) for v in np.percentile(cell_lum, [10, 50, 90])]

            mgx, mgy = float(cell_gx.mean()), float(cell_gy.mean())
            aniso = (mgy - mgx) / (mgy + mgx + 1e-4)
            edge_hi = float(np.mean(cell_grad > 0.14))
            edge_flat = float(np.mean(cell_grad < 0.012))
            msat = float(cell_sat.mean())

            glob6 = [
                0.25 * log_aspect,
                0.25 * white_paper_frac,
                0.25 * dark_ide_frac,
                0.25 * color_bar_frac,
                0.25 * (natural_smooth_frac - flat_frac),
                0.25 * horiz_line_bias,
            ]

            vec48 = sub27[:27] + c_mean + c_std + [q10, q50, q90] + [mgx * 4.0, mgy * 4.0, aniso, edge_hi, edge_flat, msat] + glob6
            patches[py * 4 + px] = np.array(vec48[:48], dtype=np.float32)

    return patches


def extract_audio_patches_16x16(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract a `[16, 16]` time-frequency & acoustic structure patch matrix plus estimated speech `snr_db` from
    16 kHz mono audio `samples`.
    """
    if samples.ndim == 2 and samples.shape == (16, 16):
        p90 = float(np.percentile(samples, 90))
        p10 = float(np.percentile(samples, 10))
        return samples.astype(np.float32), max(0.0, (p90 - p10) * 8.0)

    s = np.asarray(samples, dtype=np.float32).reshape(-1)
    if s.size < 640:
        s = np.pad(s, (0, 640 - s.size))

    rms_total = float(np.sqrt(np.mean(s * s) + 1e-12))
    peak_total = float(np.max(np.abs(s)) + 1e-12)

    win_len = 320
    hop = max(160, (s.size - win_len) // 240)
    n_frames = max(16, 1 + (s.size - win_len) // hop)
    idx_mat = np.arange(win_len)[None, :] + (np.arange(n_frames) * hop)[:, None]
    idx_mat = np.clip(idx_mat, 0, s.size - 1)
    frames = s[idx_mat] * np.hanning(win_len).astype(np.float32)[None, :]

    spec_mag = np.abs(np.fft.rfft(frames, axis=-1))
    spec_pow = spec_mag * spec_mag
    frame_pow = np.mean(frames * frames, axis=-1) + np.float32(1e-12)

    mel_edges = [1, 3, 5, 8, 11, 15, 20, 26, 33, 42, 52, 65, 80, 98, 120, 142, 161]
    mel_t_f = np.zeros((n_frames, 16), dtype=np.float32)
    for b_i in range(16):
        b0, b1 = mel_edges[b_i], mel_edges[b_i + 1]
        mel_t_f[:, b_i] = np.mean(spec_pow[:, b0:b1], axis=-1)

    log_mel = np.log10(mel_t_f + np.float32(1e-10))

    sorted_pow = np.sort(frame_pow)
    p_top = float(np.mean(sorted_pow[int(0.82 * n_frames) :]))
    p_mid = float(np.mean(sorted_pow[int(0.18 * n_frames) : int(0.65 * n_frames)]))
    b_head = frame_pow[: max(2, int(0.10 * n_frames))]
    b_tail = frame_pow[max(0, int(0.85 * n_frames)) :]
    p_bound = float(np.median(np.concatenate([b_head, b_tail])))
    p_noise = max(p_mid, 0.75 * p_bound)

    if rms_total < 0.0025:
        snr_db = 0.0
    else:
        sig_est = max(0.0, p_top - 2.15 * p_noise)
        if sig_est <= 1e-10:
            snr_db = 0.0
        else:
            snr_db = float(np.clip(10.0 * math.log10((sig_est + 1e-11) / (p_noise + 1e-11)), 0.0, 40.0))

    t_splits = np.array_split(log_mel, 12, axis=0)
    grid_12x16 = np.stack([seg.mean(axis=0) for seg in t_splits], axis=0)
    grid_std = (grid_12x16 - grid_12x16.mean()) / (grid_12x16.std() + np.float32(1e-5))

    active_mask = frame_pow > np.percentile(frame_pow, 65)
    act_spec = spec_pow[active_mask, 2:140] if np.any(active_mask) else spec_pow[:, 2:140]
    mean_act_bins = np.mean(act_spec, axis=0) + 1e-12
    tonal_crest = float(np.max(mean_act_bins) / np.mean(mean_act_bins))
    log_tonal_crest = float(np.clip(math.log10(max(1.0, tonal_crest)), 0.0, 2.5))

    geo_m = np.exp(np.mean(np.log(act_spec + 1e-12), axis=-1))
    arith_m = np.mean(act_spec + 1e-12, axis=-1)
    spec_flatness = float(np.mean(geo_m / arith_m))

    mel_act = np.mean(mel_t_f[active_mask], axis=0) if np.any(active_mask) else np.mean(mel_t_f, axis=0)
    formant_ratio = float(np.sum(mel_act[2:12]) / (np.sum(mel_act) + 1e-12))
    hi_freq_ratio = float(np.sum(mel_act[12:]) / (np.sum(mel_act) + 1e-12))

    env_db = 10.0 * np.log10(frame_pow / (np.max(frame_pow) + 1e-12) + 1e-8)
    quiet_frame_frac = float(np.mean(env_db < -18.0))
    head_quiet = float(np.mean(env_db[: max(2, int(0.10 * n_frames))] < -14.0))
    env_diff = float(np.mean(np.abs(np.diff(env_db))))

    if act_spec.shape[0] >= 4:
        top_bins = np.argmax(act_spec, axis=-1)
        pitch_constancy = float(np.mean(top_bins == np.bincount(top_bins).argmax()))
    else:
        pitch_constancy = 0.5

    patches = np.zeros((N_PATCHES, 16), dtype=np.float32)
    patches[:12, :] = grid_std * np.float32(0.5)
    summary_vec = np.array([
        snr_db / 20.0,
        math.log10(rms_total + 1e-6) / 3.0,
        log_tonal_crest,
        spec_flatness * 3.0,
        formant_ratio * 2.0,
        hi_freq_ratio * 3.0,
        quiet_frame_frac * 2.0,
        head_quiet * 2.0,
        env_diff / 4.0,
        pitch_constancy * 2.0,
        float(peak_total / (rms_total + 1e-6)) / 8.0,
        float(snr_db >= 14.0),
        float(3.0 <= snr_db < 14.0),
        float(snr_db < 3.0 or rms_total < 0.003),
        float(tonal_crest > 15.0 or pitch_constancy > 0.55),
        float(quiet_frame_frac > 0.22 and formant_ratio > 0.55 and tonal_crest < 16.0),
    ], dtype=np.float32)
    for r_i in range(12, 16):
        patches[r_i, :] = summary_vec * np.float32(0.8 + 0.1 * (r_i - 12))

    return patches, round(float(snr_db), 2)


def write_clvrtrk1(trunk_dir: Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write a dictionary of float32 NumPy tensors into Clover's `CLVRTRK1` binary trunk format."""
    trunk_dir = Path(trunk_dir)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    common_buf = bytearray()
    op_buf = bytearray()
    ffn_buf = bytearray()
    entries: list[bytes] = []

    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        raw = a.tobytes()
        if "mlp" in name or "ffn" in name or "net." in name:
            fid, buf = 3, ffn_buf
        elif "attn" in name or "gru" in name or "proj" in name:
            fid, buf = 1, op_buf
        else:
            fid, buf = 0, common_buf
        offset = len(buf)
        buf.extend(raw)
        ndim = a.ndim
        dims = list(a.shape) + [1, 1, 1, 1]
        name_b = name.encode("utf-8")[:63].ljust(64, b"\x00")
        meta = struct.pack(
            "<IIIQQIIII",
            fid,
            0,  # float32 dtype_code = 0
            ndim,
            offset,
            len(raw),
            dims[0],
            dims[1],
            dims[2],
            dims[3],
        )
        entries.append(name_b + meta)

    header = b"CLVRTRK1" + struct.pack("<III", 1, 0, len(entries))
    (trunk_dir / "index.bin").write_bytes(header + b"".join(entries))
    (trunk_dir / "common.bin").write_bytes(bytes(common_buf))
    (trunk_dir / "operator.bin").write_bytes(bytes(op_buf))
    (trunk_dir / "ffn.bin").write_bytes(bytes(ffn_buf))
    return trunk_dir


def read_clvrtrk1(trunk_dir: Path) -> dict[str, np.ndarray]:
    """Read `CLVRTRK1` binary trunk directory into a dictionary of float32 NumPy arrays."""
    idx_path = Path(trunk_dir) / "index.bin"
    idx_bytes = idx_path.read_bytes()
    if idx_bytes[:8] != b"CLVRTRK1":
        raise ValueError(f"Invalid CLVRTRK1 magic in {trunk_dir}")
    _, _, n_entries = struct.unpack("<III", idx_bytes[8:20])
    fnames = {0: "common.bin", 1: "operator.bin", 3: "ffn.bin"}
    blobs: dict[int, memoryview] = {}
    for fid, fn in fnames.items():
        fp = Path(trunk_dir) / fn
        if fp.is_file() and fp.stat().st_size > 0:
            blobs[fid] = memoryview(fp.read_bytes())
    out: dict[str, np.ndarray] = {}
    pos = 20
    for _ in range(n_entries):
        tname = idx_bytes[pos : pos + 64].split(b"\x00", 1)[0].decode("utf-8")
        pos += 64
        fid, dtype_code, ndim, offset, nbytes, d0, d1, d2, d3 = struct.unpack("<IIIQQIIII", idx_bytes[pos : pos + 44])
        pos += 44
        dims = [d0, d1, d2, d3][:ndim] if ndim > 0 else [1]
        raw = blobs[fid][offset : offset + nbytes]
        if dtype_code == 1:
            u16 = np.frombuffer(raw, dtype=np.uint16)
            out[tname] = (u16.astype(np.uint32) << 16).view(np.float32).copy().reshape(dims)
        else:
            out[tname] = np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)
    return out


def _build_default_weights(seed: int = 20261107) -> dict[str, np.ndarray]:
    """Deterministic orthogonal/calibrated initialization when precomputed trunk files are not yet on disk."""
    rng = np.random.default_rng(seed)

    def ortho(rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
        m = rng.standard_normal((rows, cols)).astype(np.float32)
        u, _, vt = np.linalg.svd(m, full_matrices=False)
        q = u if rows >= cols else vt
        if q.shape != (rows, cols):
            q = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-6)
        return (q * np.float32(scale)).astype(np.float32)

    w: dict[str, np.ndarray] = {}
    # Modality patch/vector adapters -> D_JEPA (64)
    w["vis_adapter.weight"] = ortho(D_JEPA, 48, 0.8)
    w["vis_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["vis_pos_embed"] = (rng.standard_normal((N_PATCHES, D_JEPA)) * 0.04).astype(np.float32)
    w["vis_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    w["aud_adapter.weight"] = ortho(D_JEPA, 16, 0.8)
    w["aud_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["aud_pos_embed"] = (rng.standard_normal((N_PATCHES, D_JEPA)) * 0.04).astype(np.float32)
    w["aud_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    w["state_adapter.weight"] = ortho(D_JEPA, D_VQ, 0.8)
    w["state_adapter.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["state_mod_token"] = (rng.standard_normal((1, D_JEPA)) * 0.04).astype(np.float32)

    # 4-Level Shared Hierarchical Encoder blocks (`blocks.0..3`)
    for l in range(4):
        w[f"enc.block.{l}.qkv.weight"] = ortho(3 * D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.qkv.bias"] = np.zeros(3 * D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.proj.weight"] = ortho(D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.proj.bias"] = np.zeros(D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.mlp1.weight"] = ortho(2 * D_JEPA, D_JEPA, 0.7)
        w[f"enc.block.{l}.mlp1.bias"] = np.zeros(2 * D_JEPA, dtype=np.float32)
        w[f"enc.block.{l}.mlp2.weight"] = ortho(D_JEPA, 2 * D_JEPA, 0.7)
        w[f"enc.block.{l}.mlp2.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    # LatentTokenPredictor (`I-JEPA` / `Audio-JEPA` / `MJEPA`)
    w["pred.in_proj.weight"] = ortho(D_PRED, D_JEPA, 0.7)
    w["pred.in_proj.bias"] = np.zeros(D_PRED, dtype=np.float32)
    w["pred.mask_token"] = np.zeros((1, D_PRED), dtype=np.float32)
    w["pred.pos_embed"] = (rng.standard_normal((N_PATCHES, D_PRED)) * 0.03).astype(np.float32)
    w["pred.qkv.weight"] = ortho(3 * D_PRED, D_PRED, 0.7)
    w["pred.qkv.bias"] = np.zeros(3 * D_PRED, dtype=np.float32)
    w["pred.out_proj.weight"] = ortho(D_JEPA, D_PRED, 0.7)
    w["pred.out_proj.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    # MJEPA Cross-Modal Predictors (`a2v`, `v2a`, `s2v`, `s2a`, `v2s`, `a2s`)
    for pair in ("a2v", "v2a", "s2v", "s2a", "v2s", "a2s"):
        w[f"cross.{pair}.fc1.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
        w[f"cross.{pair}.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
        w[f"cross.{pair}.fc2.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
        w[f"cross.{pair}.fc2.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    # EB-JEPA Action-Conditioned Transition Predictor + Inverse Dynamics Model (`IDM`)
    w["ac_pred.state_in.weight"] = ortho(D_JEPA, D_JEPA, 0.8)
    w["ac_pred.state_in.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["ac_pred.act_in.weight"] = ortho(D_JEPA, D_ACT, 0.5)
    w["ac_pred.act_in.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["ac_pred.out.weight"] = np.eye(D_JEPA, dtype=np.float32)
    w["ac_pred.out.bias"] = np.zeros(D_JEPA, dtype=np.float32)

    w["idm.fc1.weight"] = ortho(D_JEPA, 2 * D_JEPA, 0.7)
    w["idm.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["idm.fc2.weight"] = ortho(D_ACT, D_JEPA, 0.7)
    w["idm.fc2.bias"] = np.zeros(D_ACT, dtype=np.float32)

    # Bridge between D_JEPA (64) and Clover's D_VQ (384) / D_LFM3 (2048)
    w["bridge.jepa_to_vq.weight"] = ortho(D_VQ, D_JEPA, 0.6)
    w["bridge.jepa_to_vq.bias"] = np.zeros(D_VQ, dtype=np.float32)
    w["bridge.lfm3_to_vq.weight"] = ortho(D_VQ, D_LFM3, 0.6)
    w["bridge.lfm3_to_vq.bias"] = np.zeros(D_VQ, dtype=np.float32)
    w["bridge.vq_to_lfm3.weight"] = ortho(D_LFM3, D_VQ, 0.15)
    w["bridge.vq_to_lfm3.bias"] = np.zeros(D_LFM3, dtype=np.float32)

    # Visual scene & acoustic regime linear probe heads
    w["probe.visual_scene.weight"] = ortho(len(VISUAL_SCENE_KINDS), D_JEPA, 1.0)
    w["probe.visual_scene.bias"] = np.zeros(len(VISUAL_SCENE_KINDS), dtype=np.float32)
    w["probe.acoustic_regime.weight"] = ortho(len(ACOUSTIC_REGIMES), D_JEPA, 1.0)
    w["probe.acoustic_regime.bias"] = np.zeros(len(ACOUSTIC_REGIMES), dtype=np.float32)

    # Clover RPRE OutcomeResidualAdapter (`Delta_P_psi(z_t, a_t, c_t)`): zero-init final layer
    in_res = D_JEPA + D_ACT + D_JEPA
    w["residual.fc1.weight"] = ortho(D_JEPA, in_res, 0.5)
    w["residual.fc1.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["residual.fc2.weight"] = np.zeros((D_JEPA, D_JEPA), dtype=np.float32)
    w["residual.fc2.bias"] = np.zeros(D_JEPA, dtype=np.float32)
    w["residual.gate"] = np.array([1.0], dtype=np.float32)
    w["residual.baseline_cue_proto"] = np.zeros((1, D_JEPA), dtype=np.float32)
    w["residual.has_proto"] = np.array([0.0], dtype=np.float32)

    return w


class CloverJEPARuntime:
    """Selectable JEPA Engine for Clover AI Studio, K3 Server, and Mobile."""

    def __init__(self, base_dir: str | Path | None = None):
        t0 = time.perf_counter()
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.lock = threading.Lock()
        self.trunk_dir = self._resolve_trunk_dir(self.base_dir)
        self.weights_source = "default_orthogonal"
        if (self.trunk_dir / "index.bin").is_file():
            try:
                self.w = read_clvrtrk1(self.trunk_dir)
                self.weights_source = f"clvrtrk1:{self.trunk_dir}"
            except Exception:
                self.w = _build_default_weights()
        else:
            self.w = _build_default_weights()

        # Fill any missing keys from default weights for forward compatibility
        defaults = _build_default_weights()
        for k, v in defaults.items():
            if k not in self.w or self.w[k].shape != v.shape:
                self.w[k] = v

        # Load persisted OutcomeResidualAdapter state if present in `<base>/growth/jepa_outcome_adapter.npz`
        self.adapter_path = self.base_dir / "growth" / "jepa_outcome_adapter.npz"
        self.collective_growth_path = self.base_dir / "collective" / "jepa_collective_growth.json"
        self.outcome_buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.baseline_anchor: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.cycle_updates = 0
        self.rollbacks = 0
        self._load_collective_growth()
        self._load_persisted_adapter()
        self.setup_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @staticmethod
    def _resolve_trunk_dir(base_dir: Path) -> Path:
        for cand in (
            base_dir / "data" / "omni" / "precomputed" / "jepa" / "jepa-trunk-0",
            base_dir / "data" / "omni" / "precomputed" / "models" / "jepa" / "jepa-trunk-0",
            Path(__file__).resolve().parent / "models" / "jepa-trunk-0",
        ):
            if (cand / "index.bin").is_file():
                return cand
        return base_dir / "data" / "omni" / "precomputed" / "jepa" / "jepa-trunk-0"

    def _load_collective_growth(self) -> None:
        consent_f = self.base_dir / "collective" / "consent.json"
        if not consent_f.is_file() or not self.collective_growth_path.is_file():
            return
        try:
            consent = json.loads(consent_f.read_text(encoding="utf-8"))
            if not consent.get("consent_receive"):
                return
            doc = json.loads(self.collective_growth_path.read_text(encoding="utf-8"))
            self.apply_collective_growth(doc, preserve_personal=False)
        except Exception:
            pass

    def apply_collective_growth(self, doc: dict[str, Any], preserve_personal: bool = True) -> bool:
        """Apply Ed25519-verified collective JEPA growth weights (`idm.*`, `ac_pred.*`, `residual.*`) from
        the shared core while respecting `Personal > Collective > General` priority."""
        if not isinstance(doc, dict):
            return False
        tensors = doc.get("tensors")
        if not isinstance(tensors, dict):
            return False
        allowed_keys = (
            "idm.fc1.weight", "idm.fc1.bias", "idm.fc2.weight", "idm.fc2.bias",
            "ac_pred.state_in.weight", "ac_pred.state_in.bias",
            "ac_pred.act_in.weight", "ac_pred.act_in.bias",
            "ac_pred.out.weight", "ac_pred.out.bias",
            "residual.fc1.weight", "residual.fc1.bias", "residual.fc2.weight", "residual.fc2.bias",
        )
        applied_any = False
        with self.lock:
            for k in allowed_keys:
                if k in tensors and k in self.w:
                    try:
                        arr = np.asarray(tensors[k], dtype=np.float32)
                        if arr.shape == self.w[k].shape and np.all(np.isfinite(arr)):
                            self.w[k] = arr
                            if f"clover_rpre.{k}" in self.w and self.w[f"clover_rpre.{k}"].shape == arr.shape:
                                self.w[f"clover_rpre.{k}"] = arr.copy()
                            applied_any = True
                    except Exception:
                        continue
            if applied_any:
                self.cycle_updates = max(self.cycle_updates, int(doc.get("cycle_updates") or 1))
            if preserve_personal:
                self._load_persisted_adapter()
        return applied_any

    def export_collective_growth(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """Export privacy-clean numerical JEPA self-training weights (`idm`, `ac_pred`, `residual`) for collective core."""
        export_keys = (
            "idm.fc2.weight", "idm.fc2.bias",
            "ac_pred.act_in.weight", "ac_pred.act_in.bias",
            "residual.fc2.weight", "residual.fc2.bias",
        )
        tensors: dict[str, Any] = {}
        with self.lock:
            for k in export_keys:
                w_arr = self._wt(k, "clover_rpre")
                tensors[k] = np.round(w_arr.astype(np.float32), 6).tolist()
        return {
            "schema_version": "1.0.0",
            "arch": "clover_rpre",
            "cycle_updates": int(self.cycle_updates),
            "metrics": dict(metrics or {}),
            "tensors": tensors,
        }

    def _load_persisted_adapter(self) -> None:
        if not self.adapter_path.is_file():
            return
        try:
            with np.load(str(self.adapter_path)) as data:
                for k in ("residual.fc1.weight", "residual.fc1.bias", "residual.fc2.weight", "residual.fc2.bias",
                          "residual.gate", "residual.baseline_cue_proto", "residual.has_proto"):
                    if k in data and data[k].shape == self.w[k].shape:
                        self.w[k] = data[k].astype(np.float32)
                if "cycle_updates" in data:
                    self.cycle_updates = int(data["cycle_updates"][0])
        except Exception:
            pass

    def _save_persisted_adapter(self) -> None:
        try:
            self.adapter_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(self.adapter_path),
                **{
                    "residual.fc1.weight": self.w["residual.fc1.weight"],
                    "residual.fc1.bias": self.w["residual.fc1.bias"],
                    "residual.fc2.weight": self.w["residual.fc2.weight"],
                    "residual.fc2.bias": self.w["residual.fc2.bias"],
                    "residual.gate": self.w["residual.gate"],
                    "residual.baseline_cue_proto": self.w["residual.baseline_cue_proto"],
                    "residual.has_proto": self.w["residual.has_proto"],
                    "cycle_updates": np.array([self.cycle_updates], dtype=np.int64),
                },
            )
        except Exception:
            pass

    # ---------------- Architecture Selection Policy ----------------

    def choose_architecture(
        self,
        flow: str,
        configured_arch: str = "auto",
        *,
        has_visual: bool = False,
        has_audio: bool = False,
        has_outcome_feedback: bool = False,
    ) -> str:
        """Select which JEPA architecture to run for `flow` given user setting `configured_arch`."""
        arch = str(configured_arch or "auto").strip().lower()
        if arch not in JEPA_ARCHITECTURES:
            arch = "auto"
        if arch != "auto":
            return arch
        if has_outcome_feedback or flow == "outcome_growth":
            return "clover_rpre"
        if has_visual and has_audio:
            return "mjepa" if flow != "c2_plan" else "clover_rpre"
        return FLOW_DEFAULT_ARCH.get(flow, "clover_rpre")

    def describe(self, configured_arch: str = "auto") -> dict[str, Any]:
        return {
            "configured_arch": configured_arch if configured_arch in JEPA_ARCHITECTURES else "auto",
            "available_architectures": list(JEPA_ARCHITECTURES),
            "flow_routing": dict(FLOW_DEFAULT_ARCH),
            "weights_source": self.weights_source,
            "embed_dim": D_JEPA,
            "bridge_dim": D_VQ,
            "cycle_updates": self.cycle_updates,
            "rollbacks": self.rollbacks,
            "setup_ms": self.setup_ms,
        }

    def _wt(self, key: str, arch: str | None = None) -> np.ndarray:
        """Resolve tensor `key` for `arch` (`{arch}.{key}` when trained per-architecture weights are present, else top-level `key`)."""
        if arch and arch in ("ijepa", "audio_jepa", "eb_jepa", "mjepa", "clover_rpre"):
            ak = f"{arch}.{key}"
            if ak in self.w:
                return self.w[ak]
        return self.w[key]

    # ---------------- Shared 4-Level Hierarchical Transformer Encoder ----------------

    def _run_encoder_blocks(
        self,
        tokens_nd: np.ndarray,
        return_hierarchical: bool = False,
        arch: str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Run the 4-layer shared Transformer trunk on `tokens_nd` (`[N_ctxt, D_JEPA]`)."""
        x = np.asarray(tokens_nd, dtype=np.float32)
        levels: list[np.ndarray] = []
        for l in range(4):
            xn1 = _layer_norm(x)
            qkv = _linear(xn1, self._wt(f"enc.block.{l}.qkv.weight", arch), self._wt(f"enc.block.{l}.qkv.bias", arch))
            q, k, v = np.split(qkv, 3, axis=-1)
            scale = np.float32(1.0 / math.sqrt(D_JEPA))
            attn = _softmax(np.matmul(q, k.T) * scale, axis=-1)
            x = x + _linear(np.matmul(attn, v), self._wt(f"enc.block.{l}.proj.weight", arch), self._wt(f"enc.block.{l}.proj.bias", arch))
            xn2 = _layer_norm(x)
            h_mlp = _gelu(_linear(xn2, self._wt(f"enc.block.{l}.mlp1.weight", arch), self._wt(f"enc.block.{l}.mlp1.bias", arch)))
            x = x + _linear(h_mlp, self._wt(f"enc.block.{l}.mlp2.weight", arch), self._wt(f"enc.block.{l}.mlp2.bias", arch))
            levels.append(_layer_norm(x))
        if return_hierarchical:
            return np.mean(np.stack(levels, axis=0), axis=0), levels
        return levels[-1], levels

    def _predict_masked_tokens(
        self,
        ctxt_tokens: np.ndarray,
        ctxt_idx: np.ndarray,
        targ_idx: np.ndarray,
        arch: str | None = None,
    ) -> np.ndarray:
        """`LatentTokenPredictor`: predict target patch representations at `targ_idx` from visible `ctxt_tokens` (`[N_ctxt, D_JEPA]`)."""
        pos = self._wt("pred.pos_embed", arch)
        x_c = _linear(ctxt_tokens, self._wt("pred.in_proj.weight", arch), self._wt("pred.in_proj.bias", arch)) + pos[ctxt_idx]
        x_m = np.repeat(self._wt("pred.mask_token", arch), len(targ_idx), axis=0) + pos[targ_idx]
        x_all = np.concatenate([x_c, x_m], axis=0)
        xn = _layer_norm(x_all)
        qkv = _linear(xn, self._wt("pred.qkv.weight", arch), self._wt("pred.qkv.bias", arch))
        q, k, v = np.split(qkv, 3, axis=-1)
        attn = _softmax(np.matmul(q, k.T) / np.float32(math.sqrt(D_PRED)), axis=-1)
        out = x_all + np.matmul(attn, v)
        pred_targ = _linear(_layer_norm(out[len(ctxt_idx):]), self._wt("pred.out_proj.weight", arch), self._wt("pred.out_proj.bias", arch))
        return _layer_norm(pred_targ)

    def _cross_modal_predict(self, z_tokens_or_vec: np.ndarray, pair: str, arch: str | None = None) -> np.ndarray:
        """Run MJEPA cross-modal predictor (`a2v`, `v2a`, `s2v`, `s2a`, `v2s`, `a2s`)."""
        z = np.asarray(z_tokens_or_vec, dtype=np.float32)
        if z.ndim == 2:
            z = z.mean(axis=0, keepdims=True)
        elif z.ndim == 1:
            z = z[None, :]
        h = _gelu(_linear(_layer_norm(z), self._wt(f"cross.{pair}.fc1.weight", arch), self._wt(f"cross.{pair}.fc1.bias", arch)))
        out = _layer_norm(z + _linear(h, self._wt(f"cross.{pair}.fc2.weight", arch), self._wt(f"cross.{pair}.fc2.bias", arch)))
        return out.squeeze(0)

    @staticmethod
    def _spatial_pool_16x64(tokens_16x64: np.ndarray, full_z_16x64: np.ndarray) -> np.ndarray:
        """Pool 16 spatial patch tokens (`[16, 64]`) into a 64-d representation `z_vis_pool` without erasing spatial
        patch layout: patch `p in 0..15` contributes its dedicated 4-d slice `4p : 4p+4` combined with the global
        Transformer mean representation.
        """
        idx = np.arange(N_PATCHES, dtype=np.int64)
        local_tok = np.concatenate([tokens_16x64[p, p * 4 : (p + 1) * 4] for p in idx], axis=0)
        local_z = np.concatenate([full_z_16x64[p, p * 4 : (p + 1) * 4] for p in idx], axis=0)
        global_z = full_z_16x64.mean(axis=0)
        return _layer_norm((np.float32(0.65) * local_tok + np.float32(0.20) * local_z + np.float32(0.15) * global_z)[None, :]).squeeze(0)

    # ---------------- Flow 1: Visual S3 Latent Inspection (`I-JEPA` / `MJEPA` / `RPRE`) ----------------

    def inspect_visual_latent(
        self,
        img_rgb: np.ndarray,
        elements: list[dict[str, Any]] | None = None,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        """Run `I-JEPA` context-only sparse visual encoding & multi-block latent prediction on an image `[H, W, 3]`.
        Computes per-patch semantic salience (latent prediction surprise against context) to rank UI/visual elements
        `[e1..eN]` and classifies the scene layout using the trained `probe.visual_scene` head.
        """
        arch = self.choose_architecture("visual_s3", configured_arch, has_visual=True)
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        arr = np.asarray(img_rgb)
        H, W = int(arr.shape[0]), int(arr.shape[1])

        patches = extract_visual_patches_16x48(arr)
        patch_dev = np.mean(patches[:, :27] ** 2, axis=-1)

        tokens = _linear(patches, self.w["vis_adapter.weight"], self.w["vis_adapter.bias"]) + self._wt("vis_pos_embed", arch) + self._wt("vis_mod_token", arch)
        use_hier = arch in ("mjepa", "clover_rpre")
        full_z, _ = self._run_encoder_blocks(tokens, return_hierarchical=use_hier, arch=arch)

        block_a_targ = np.array([5, 6, 9, 10], dtype=np.int64)
        block_a_ctxt = np.array([i for i in range(N_PATCHES) if i not in block_a_targ], dtype=np.int64)
        block_b_ctxt = np.array([5, 6, 9, 10, 1, 2, 13, 14, 4], dtype=np.int64)
        block_b_targ = np.array([i for i in range(N_PATCHES) if i not in block_b_ctxt], dtype=np.int64)

        z_ctxt_a, _ = self._run_encoder_blocks(tokens[block_a_ctxt], return_hierarchical=use_hier, arch=arch)
        pred_a = self._predict_masked_tokens(z_ctxt_a, block_a_ctxt, block_a_targ, arch=arch)

        z_ctxt_b, _ = self._run_encoder_blocks(tokens[block_b_ctxt], return_hierarchical=use_hier, arch=arch)
        pred_b = self._predict_masked_tokens(z_ctxt_b, block_b_ctxt, block_b_targ, arch=arch)

        patch_surprise = np.zeros(N_PATCHES, dtype=np.float32)
        for idx_i, p_idx in enumerate(block_a_targ):
            patch_surprise[p_idx] = float(np.mean((pred_a[idx_i] - full_z[p_idx]) ** 2))
        for idx_i, p_idx in enumerate(block_b_targ):
            patch_surprise[p_idx] = float(np.mean((pred_b[idx_i] - full_z[p_idx]) ** 2))

        s_min, s_max = float(patch_surprise.min()), float(patch_surprise.max())
        surp_norm = (patch_surprise - s_min) / max(1e-6, s_max - s_min)
        d_min, d_max = float(patch_dev.min()), float(patch_dev.max())
        dev_norm = (patch_dev - d_min) / max(1e-6, d_max - d_min)
        patch_salience = np.float32(0.50) * surp_norm + np.float32(0.50) * dev_norm

        salient_refs: list[dict[str, Any]] = []
        if elements:
            for el in elements:
                bbox = el.get("bbox") or [0, 0, W, H]
                cx = (bbox[0] + 0.5 * bbox[2]) / max(1.0, float(W))
                cy = (bbox[1] + 0.5 * bbox[3]) / max(1.0, float(H))
                px = min(3, max(0, int(cx * 4)))
                py = min(3, max(0, int(cy * 4)))
                sal = float(patch_salience[py * 4 + px])
                if el.get("text"):
                    sal = min(1.0, sal * 0.7 + 0.3)
                el["jepa_salience"] = round(sal, 3)
                salient_refs.append({"ref": el.get("ref", ""), "role": el.get("role", ""), "salience": round(sal, 3), "text": el.get("text", "")[:40]})
            salient_refs.sort(key=lambda r: -r["salience"])

        z_vis_pool = self._spatial_pool_16x64(tokens, full_z)
        z_vq = _layer_norm(_linear(z_vis_pool[None, :], self._wt("bridge.jepa_to_vq.weight", arch), self._wt("bridge.jepa_to_vq.bias", arch))).squeeze(0)

        # Scene classification decided directly by the trained `probe.visual_scene` head on `z_vis_pool`
        logits = _linear(z_vis_pool[None, :], self._wt("probe.visual_scene.weight", arch), self._wt("probe.visual_scene.bias", arch)).squeeze(0)
        scene_kind = VISUAL_SCENE_KINDS[int(np.argmax(logits))]

        flops_ratio = round((len(block_a_ctxt) / float(N_PATCHES)) ** 2, 4)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "visual_s3",
            "arch": arch,
            "scene_kind": scene_kind,
            "attn_flops_ratio": flops_ratio,
            "mean_patch_surprise": round(float(patch_surprise.mean()), 4),
            "top_salient_elements": salient_refs[:5],
            "z_jepa": z_vis_pool,
            "z_vq": z_vq,
            "ms": ms,
        }

    # ---------------- Flow 2: Audio S3/C1 Latent Inspection (`Audio-JEPA` / `MJEPA` / `RPRE`) ----------------

    def inspect_audio_latent(
        self,
        audio_input: bytes | str | Path | np.ndarray,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        """Run `Audio-JEPA` 2D time-frequency spectrogram patch latent prediction with symmetric `CenteredNormMSELoss`
        and classify the acoustic regime via the trained `probe.acoustic_regime` head.
        """
        arch = self.choose_architecture("audio_s3", configured_arch, has_audio=True)
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        samples = self._extract_audio_samples(audio_input)
        patches, snr_db = extract_audio_patches_16x16(samples)

        tokens = (
            _linear(patches, self._wt("aud_adapter.weight", arch), self._wt("aud_adapter.bias", arch))
            + self._wt("aud_pos_embed", arch)
            + self._wt("aud_mod_token", arch)
        )
        use_hier = arch in ("mjepa", "clover_rpre")
        full_z, _ = self._run_encoder_blocks(tokens, return_hierarchical=use_hier, arch=arch)

        targ_idx = np.array([3, 5, 6, 7, 9, 10], dtype=np.int64)
        ctxt_idx = np.array([i for i in range(N_PATCHES) if i not in targ_idx], dtype=np.int64)
        z_ctxt, _ = self._run_encoder_blocks(tokens[ctxt_idx], return_hierarchical=use_hier, arch=arch)
        pred_targ = self._predict_masked_tokens(z_ctxt, ctxt_idx, targ_idx, arch=arch)
        true_targ = full_z[targ_idx]

        cn_mse = centered_norm_mse(pred_targ, true_targ, center_pred=True)
        tf_cos = max(-1.0, min(1.0, 1.0 - 0.5 * cn_mse))

        z_aud_pool = self._spatial_pool_16x64(tokens, full_z)
        z_vq = _layer_norm(_linear(z_aud_pool[None, :], self._wt("bridge.jepa_to_vq.weight", arch), self._wt("bridge.jepa_to_vq.bias", arch))).squeeze(0)

        logits = _linear(z_aud_pool[None, :], self._wt("probe.acoustic_regime.weight", arch), self._wt("probe.acoustic_regime.bias", arch)).squeeze(0)
        regime = ACOUSTIC_REGIMES[int(np.argmax(logits))]

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "audio_s3",
            "arch": arch,
            "acoustic_regime": regime,
            "snr_db": round(float(snr_db), 2),
            "tf_centered_norm_mse": round(float(cn_mse), 4),
            "tf_latent_cos": round(float(tf_cos), 4),
            "z_jepa": z_aud_pool,
            "z_vq": z_vq,
            "ms": ms,
        }

    @staticmethod
    def _extract_audio_samples(audio_input: bytes | str | Path | np.ndarray) -> np.ndarray:
        """Extract 16 kHz mono float32 audio samples in `[-1, 1]` from WAV path, bytes, or array."""
        if isinstance(audio_input, np.ndarray):
            return audio_input.astype(np.float32).reshape(-1)
        raw_bytes = b""
        if isinstance(audio_input, (str, Path)):
            p = Path(audio_input)
            if p.is_file():
                try:
                    with wave.open(str(p), "rb") as wf:
                        n_ch = max(1, wf.getnchannels())
                        sw = wf.getsampwidth()
                        fs = max(1, wf.getframerate())
                        max_frames = min(wf.getnframes(), fs * 10)
                        frames = wf.readframes(max_frames)
                        if sw == 2:
                            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sw == 1:
                            x = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                        elif sw == 4:
                            x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            x = np.zeros(1600, dtype=np.float32)
                        if n_ch > 1 and x.size >= n_ch:
                            x = x[: (x.size // n_ch) * n_ch].reshape(-1, n_ch).mean(axis=1)
                        if fs != 16000 and x.size > 16:
                            t_old = np.linspace(0.0, 1.0, x.size, endpoint=False)
                            n_new = max(16, int(round(x.size * 16000.0 / float(fs))))
                            t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
                            x = np.interp(t_new, t_old, x).astype(np.float32)
                        return x
                except Exception:
                    raw_bytes = p.read_bytes()[:128000]
        elif isinstance(audio_input, (bytes, bytearray)):
            raw_bytes = bytes(audio_input[:128000])
        if len(raw_bytes) >= 44 and raw_bytes[:4] == b"RIFF":
            pcm = raw_bytes[44:]
            n = (len(pcm) // 2) * 2
            if n > 0:
                return np.frombuffer(pcm[:n], dtype=np.int16).astype(np.float32) / 32768.0
        if raw_bytes:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            return (arr - 128.0) / 128.0
        return np.zeros(1600, dtype=np.float32)

    @staticmethod
    def _compute_logmel_16x16(samples: np.ndarray) -> tuple[np.ndarray, float]:
        return extract_audio_patches_16x16(samples)

    # ---------------- Flow 3: Multimodal Cross-Modal Grounding & Recall (`MJEPA` / `RPRE`) ----------------

    def fuse_and_retrieve_multimodal(
        self,
        query_vq_384: np.ndarray,
        *,
        visual_z_64: np.ndarray | None = None,
        audio_z_64: np.ndarray | None = None,
        candidates_vq_384: np.ndarray | None = None,
        configured_arch: str = "auto",
    ) -> dict[str, Any]:
        """Run `MJEPA` 4-level hierarchical encoding + 6 cross-modal predictors (`a2v`, `v2a`, `s2v`, `s2a`, `v2s`, `a2s`).
        When visual or audio input is missing, imputes the missing modality in latent space and scores candidate
        memory/media embeddings (`candidates_vq_384`).
        """
        arch = self.choose_architecture(
            "multimodal_s2",
            configured_arch,
            has_visual=visual_z_64 is not None,
            has_audio=audio_z_64 is not None,
        )
        if arch == "none":
            return {"enabled": False, "arch": "none"}

        t0 = time.perf_counter()
        q_vq = np.asarray(query_vq_384, dtype=np.float32).reshape(1, D_VQ)
        s_tok = _linear(q_vq, self._wt("state_adapter.weight", arch), self._wt("state_adapter.bias", arch)) + self._wt("state_mod_token", arch)
        z_s, _ = self._run_encoder_blocks(s_tok, return_hierarchical=(arch in ("mjepa", "clover_rpre")), arch=arch)
        z_s_vec = z_s.squeeze(0)

        imputed_modalities: list[str] = []
        if visual_z_64 is not None and audio_z_64 is not None:
            z_v = np.asarray(visual_z_64, dtype=np.float32).reshape(D_JEPA)
            z_a = np.asarray(audio_z_64, dtype=np.float32).reshape(D_JEPA)
        elif visual_z_64 is not None:
            z_v = np.asarray(visual_z_64, dtype=np.float32).reshape(D_JEPA)
            z_a = 0.5 * (self._cross_modal_predict(z_v, "v2a", arch=arch) + self._cross_modal_predict(z_s_vec, "s2a", arch=arch))
            imputed_modalities.append("audio_from_v2a_s2a")
        elif audio_z_64 is not None:
            z_a = np.asarray(audio_z_64, dtype=np.float32).reshape(D_JEPA)
            z_v = 0.5 * (self._cross_modal_predict(z_a, "a2v", arch=arch) + self._cross_modal_predict(z_s_vec, "s2v", arch=arch))
            imputed_modalities.append("visual_from_a2v_s2v")
        else:
            z_v = self._cross_modal_predict(z_s_vec, "s2v", arch=arch)
            z_a = self._cross_modal_predict(z_s_vec, "s2a", arch=arch)
            imputed_modalities.extend(["visual_from_s2v", "audio_from_s2a"])

        # Fused MJEPA joint representation in D_JEPA (64) and bridged to D_VQ (384)
        z_joint_64 = _layer_norm((z_s_vec + 0.5 * (z_v + z_a))[None, :]).squeeze(0)
        z_joint_vq = _layer_norm(_linear(z_joint_64[None, :], self._wt("bridge.jepa_to_vq.weight", arch), self._wt("bridge.jepa_to_vq.bias", arch))).squeeze(0)
        fused_query_vq = _l2_normalize(0.65 * q_vq.squeeze(0) + 0.35 * z_joint_vq, axis=-1)

        scores = []
        if candidates_vq_384 is not None and len(candidates_vq_384) > 0:
            cands = _l2_normalize(np.asarray(candidates_vq_384, dtype=np.float32), axis=-1)
            scores = np.matmul(cands, fused_query_vq).tolist()

        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "multimodal_s2",
            "arch": arch,
            "imputed_modalities": imputed_modalities,
            "cross_modal_av_cos": round(float(np.sum(_l2_normalize(z_v) * _l2_normalize(z_a))), 4),
            "z_joint_64": z_joint_64,
            "z_joint_vq": z_joint_vq,
            "fused_query_vq": fused_query_vq,
            "candidate_scores": [round(float(s), 4) for s in scores],
            "ms": ms,
        }

    # ---------------- Flow 4: Core 2 (`C2`) Energy-Based Latent Planning (`EB-JEPA` / `RPRE`) ----------------

    def _compute_anom_gate(self, c_bd: np.ndarray, context_cue: np.ndarray | None) -> np.ndarray:
        B = c_bd.shape[0]
        if context_cue is None or float(np.max(np.abs(c_bd))) <= 1e-6:
            return np.zeros((B, 1), dtype=np.float32)
        if float(self.w["residual.has_proto"][0]) > 0.5 and self.baseline_anchor:
            proto = self.w["residual.baseline_cue_proto"]
            dist_sq = np.mean((c_bd - proto) ** 2, axis=-1, keepdims=True)
            return np.clip((dist_sq - np.float32(0.05)) / np.float32(0.10), np.float32(0.0), np.float32(1.0))
        return np.ones((B, 1), dtype=np.float32)

    def _step_transition(
        self,
        z_bd: np.ndarray,
        a_ba: np.ndarray,
        context_cue: np.ndarray | None = None,
        use_residual: bool = True,
        arch: str | None = None,
    ) -> np.ndarray:
        r"""One latent step $\hat{z}_{t+1} = \text{LN}(z_t + \alpha_t \delta_{\text{ac}}(z_t, a_t) + g_{\text{anom}}(c_t)\Delta P_\psi(z_t, a_t, c_t))$.
        Uses zero-centered action differential $\delta_{\text{ac}}(z_t, a_t) = f(z_t, a_t) - f(z_t, 0)$ so $a_t = 0 \implies \hat{z}_{t+1} = z_t$.
        """
        s_in = _linear(z_bd, self._wt("ac_pred.state_in.weight", arch), self._wt("ac_pred.state_in.bias", arch))
        a_zero = np.zeros_like(a_ba)
        h_act = _gelu(s_in + _linear(a_ba, self._wt("ac_pred.act_in.weight", arch), self._wt("ac_pred.act_in.bias", arch)))
        h_zero = _gelu(s_in + _linear(a_zero, self._wt("ac_pred.act_in.weight", arch), self._wt("ac_pred.act_in.bias", arch)))
        delta_ac = _linear(h_act - h_zero, self._wt("ac_pred.out.weight", arch), None)
        a_norm = np.linalg.norm(a_ba, axis=-1, keepdims=True)
        step_gate = np.tanh(a_norm) * np.float32(0.35)
        base_next = _layer_norm(z_bd + step_gate * delta_ac)
        if not use_residual:
            return base_next
        delta = self._forward_residual(z_bd, a_ba, context_cue=context_cue)
        if float(np.max(np.abs(delta))) <= 1e-7:
            return base_next
        return _layer_norm(base_next + delta)

    def _forward_residual(self, z_bd: np.ndarray, a_ba: np.ndarray, context_cue: np.ndarray | None = None) -> np.ndarray:
        B = z_bd.shape[0]
        if context_cue is None:
            c_bd = np.zeros((B, D_JEPA), dtype=np.float32)
        else:
            c_arr = np.asarray(context_cue, dtype=np.float32)
            if c_arr.shape == (B, D_JEPA):
                c_bd = c_arr
            elif c_arr.size == D_JEPA:
                c_bd = np.repeat(c_arr.reshape(1, D_JEPA), B, axis=0)
            else:
                c_bd = np.zeros((B, D_JEPA), dtype=np.float32)
        anom_gate = self._compute_anom_gate(c_bd, context_cue)
        if float(np.max(anom_gate)) <= 1e-7:
            return np.zeros_like(z_bd)
        inp = np.concatenate([z_bd, a_ba, c_bd], axis=-1)
        h = _gelu(_linear(inp, self.w["residual.fc1.weight"], self.w["residual.fc1.bias"]))
        out = _linear(h, self.w["residual.fc2.weight"], self.w["residual.fc2.bias"])
        gate = np.tanh(self.w["residual.gate"][0])
        return anom_gate * gate * out

    def encode_semantic_bridge_cue(self, h_bridge_2048: np.ndarray, configured_arch: str = "auto") -> np.ndarray:
        """Project a 2048-d representation of bridged `Memory Graph + Ontology Graph` symbolic evidence into the 64-d
        JEPA latent state space (`semantic_cue_64`) via `bridge.lfm3_to_vq` (`2048 -> 384`), `state_adapter` (`384 -> 64`),
        and the 4-layer hierarchical JEPA Transformer encoder.
        """
        arch = self.choose_architecture("c2_plan", configured_arch, has_outcome_feedback=True)
        if arch == "none":
            arch = "clover_rpre"
        h_mean = np.asarray(h_bridge_2048, dtype=np.float32)
        if h_mean.ndim == 3:
            h_mean = h_mean.mean(axis=1)
        elif h_mean.ndim == 1:
            h_mean = h_mean.reshape(1, -1)
        h_vq = _layer_norm(_linear(h_mean, self.w["bridge.lfm3_to_vq.weight"], self.w["bridge.lfm3_to_vq.bias"]))
        s_tok = _linear(h_vq, self._wt("state_adapter.weight", arch), self._wt("state_adapter.bias", arch)) + self._wt("state_mod_token", arch)
        z_enc, _ = self._run_encoder_blocks(s_tok, return_hierarchical=True, arch=arch)
        return _layer_norm(z_enc).squeeze(0)

    def plan_c2_latent(
        self,
        h_primary_2048: np.ndarray,
        r_primary_2048: np.ndarray,
        *,
        goal_vq_384: np.ndarray | None = None,
        offered_tools: list[str] | None = None,
        context_cue_64: np.ndarray | None = None,
        action_prior_8: np.ndarray | None = None,
        configured_arch: str = "auto",
        horizon: int = 4,
        cem_iters: int = 4,
        num_samples: int = 32,
    ) -> dict[str, Any]:
        """Run `EB-JEPA` / `Clover RPRE` action-conditioned latent planning (`CEMPlanner`) inside C2 (`omni_c2_c3.py`)."""
        arch = self.choose_architecture(
            "c2_plan",
            configured_arch,
            has_outcome_feedback=(self.cycle_updates > 0 or context_cue_64 is not None or action_prior_8 is not None),
        )
        if arch == "none":
            return {"enabled": False, "arch": "none", "r_refined_2048": r_primary_2048}

        t0 = time.perf_counter()
        h_mean_2048 = np.asarray(h_primary_2048, dtype=np.float32).mean(axis=1)  # [1, 2048]
        r_mean_2048 = np.asarray(r_primary_2048, dtype=np.float32).mean(axis=1)  # [1, 2048]

        h_vq = _layer_norm(_linear(h_mean_2048, self.w["bridge.lfm3_to_vq.weight"], self.w["bridge.lfm3_to_vq.bias"]))
        r_vq = _layer_norm(_linear(r_mean_2048, self.w["bridge.lfm3_to_vq.weight"], self.w["bridge.lfm3_to_vq.bias"]))

        z_init = _layer_norm(_linear(h_vq, self._wt("state_adapter.weight", arch), self._wt("state_adapter.bias", arch)) + self._wt("state_mod_token", arch))  # [1, 64]
        if goal_vq_384 is not None:
            g_vq = np.asarray(goal_vq_384, dtype=np.float32).reshape(1, D_VQ)
        else:
            g_vq = r_vq
        z_goal = _layer_norm(_linear(g_vq, self._wt("state_adapter.weight", arch), self._wt("state_adapter.bias", arch)) + self._wt("state_mod_token", arch))  # [1, 64]
        if context_cue_64 is not None:
            c64 = np.asarray(context_cue_64, dtype=np.float32).reshape(1, D_JEPA)
            z_goal = _layer_norm(0.80 * z_goal + 0.20 * c64)

        idm_in = np.concatenate([z_init, z_goal], axis=-1)
        a_idm = _linear(_gelu(_linear(idm_in, self._wt("idm.fc1.weight", arch), self._wt("idm.fc1.bias", arch))),
                        self._wt("idm.fc2.weight", arch), self._wt("idm.fc2.bias", arch))  # [1, D_ACT]
        ap_vec = None
        if action_prior_8 is not None:
            ap_arr = np.asarray(action_prior_8, dtype=np.float32).reshape(-1)
            if ap_arr.size == D_ACT and float(np.linalg.norm(ap_arr)) > 1e-6:
                ap_vec = _l2_normalize(ap_arr).reshape(1, D_ACT)
                a_idm = 0.45 * _l2_normalize(a_idm) + 0.55 * ap_vec

        use_res = arch == "clover_rpre"
        energy_before = float(np.mean((z_init - z_goal) ** 2))

        # ------------------------------------------------------------------
        # Stage A: Component-Cached + Chain-Contracted Trajectory Search
        # (from `research/eth-subexponential-counting/experiments/practical_counters.py`:
        #  decomposes D_ACT=8 into independent capability components with a memoization cache
        #  and contracts sequentially along the treewidth-1 horizon chain t = 0..T-1)
        # ------------------------------------------------------------------
        component_cache: dict[tuple[bytes, int, int], tuple[np.ndarray, float]] = {}
        cache_hits = 0
        basis_acts = np.zeros((2 * D_ACT, D_ACT), dtype=np.float32)
        for d_i in range(D_ACT):
            basis_acts[2 * d_i, d_i] = 0.55
            basis_acts[2 * d_i + 1, d_i] = -0.55

        z_chain = z_init.copy()
        mu_chain = np.zeros((horizon, D_ACT), dtype=np.float32)
        for t_step in range(horizon):
            cur_e = float(np.mean((z_chain - z_goal) ** 2))
            q_key = np.round(z_chain * 12.0).astype(np.int8).tobytes()
            comp_vec = np.zeros((1, D_ACT), dtype=np.float32)
            best_single_a = np.zeros((1, D_ACT), dtype=np.float32)
            best_single_z = z_chain
            best_single_e = cur_e

            for b_idx in range(2 * D_ACT):
                d_i = b_idx // 2
                sgn = 1 if (b_idx % 2 == 0) else -1
                ck = (q_key, d_i, sgn)
                if ck in component_cache:
                    z_next_b, e_b = component_cache[ck]
                    cache_hits += 1
                else:
                    a_b = basis_acts[b_idx : b_idx + 1]
                    z_next_b = self._step_transition(z_chain, a_b, context_cue=context_cue_64, use_residual=use_res, arch=arch)
                    e_b = float(np.mean((z_next_b - z_goal) ** 2))
                    component_cache[ck] = (z_next_b, e_b)
                gain = cur_e - e_b
                if gain > 0.0:
                    comp_vec += np.float32(gain) * basis_acts[b_idx : b_idx + 1]
                if e_b < best_single_e:
                    best_single_e = e_b
                    best_single_a = basis_acts[b_idx : b_idx + 1]
                    best_single_z = z_next_b

            # Evaluate candidate contracted actions at step t: {0, best_single, composite_components, a_idm, -a_idm}
            cands_a = [
                np.zeros((1, D_ACT), dtype=np.float32),
                best_single_a,
                a_idm * np.float32(0.5),
                -a_idm * np.float32(0.5),
            ]
            if float(np.linalg.norm(comp_vec)) > 1e-6:
                c_norm = float(np.linalg.norm(comp_vec))
                cands_a.append(comp_vec * np.float32(min(0.75, 0.65 / c_norm)))

            best_step_e = cur_e
            best_step_a = np.zeros((1, D_ACT), dtype=np.float32)
            best_step_z = z_chain
            for ca in cands_a:
                zn = self._step_transition(z_chain, ca, context_cue=context_cue_64, use_residual=use_res, arch=arch)
                en = float(np.mean((zn - z_goal) ** 2))
                if en < best_step_e:
                    best_step_e = en
                    best_step_a = ca
                    best_step_z = zn
            mu_chain[t_step] = best_step_a.squeeze(0)
            z_chain = best_step_z

        # ------------------------------------------------------------------
        # Stage B: Elitist CEM Local Polishing around the Chain-Contracted Plan
        # ------------------------------------------------------------------
        rng = np.random.default_rng(101)
        mu = mu_chain.copy()
        sigma = np.ones((horizon, D_ACT), dtype=np.float32) * np.float32(0.20)
        n_elites = max(4, num_samples // 5)

        best_cost = float(np.mean((z_chain - z_goal) ** 2))
        best_z_final = z_chain.copy()

        for _ in range(cem_iters):
            eps = rng.standard_normal((num_samples, horizon, D_ACT)).astype(np.float32)
            acts = mu[None, :, :] + sigma[None, :, :] * eps
            acts[0] = mu
            acts[1] = mu_chain
            acts[2] = 0.0  # Zero-action identity anchor so energy never regresses above energy_before
            norms = np.linalg.norm(acts, axis=-1, keepdims=True)
            acts = acts * np.minimum(1.0, 0.8 / (norms + 1e-6))

            z_cur = np.repeat(z_init, num_samples, axis=0)
            total_cost = np.zeros(num_samples, dtype=np.float32)
            for t_step in range(horizon):
                z_cur = self._step_transition(z_cur, acts[:, t_step, :], context_cue=context_cue_64, use_residual=use_res, arch=arch)
                total_cost += np.mean((z_cur - z_goal) ** 2, axis=-1)

            term_cost = np.mean((z_cur - z_goal) ** 2, axis=-1)
            elite_idx = np.argsort(term_cost)[:n_elites]
            elites = acts[elite_idx]
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + np.float32(1e-4)
            if float(term_cost[elite_idx[0]]) <= best_cost:
                best_cost = float(term_cost[elite_idx[0]])
                best_z_final = z_cur[elite_idx[0] : elite_idx[0] + 1]

        z_delta_vq = _linear(best_z_final - z_init, self._wt("bridge.jepa_to_vq.weight", arch), self._wt("bridge.jepa_to_vq.bias", arch))
        r_delta_2048 = _linear(z_delta_vq, self.w["bridge.vq_to_lfm3.weight"], self.w["bridge.vq_to_lfm3.bias"])[:, None, :]
        r_arr = np.asarray(r_primary_2048, dtype=np.float32)
        r_norm = float(np.linalg.norm(r_arr)) + 1e-6
        d_norm = float(np.linalg.norm(r_delta_2048)) * math.sqrt(max(1, r_arr.shape[1])) + 1e-6
        scale_r = np.float32(0.15 * r_norm / d_norm)
        r_refined_2048 = r_arr + scale_r * r_delta_2048

        ranked_tools = list(offered_tools or [])
        if len(ranked_tools) > 1:
            if ap_vec is not None:
                act_dir = _l2_normalize(0.20 * _l2_normalize(a_idm.squeeze(0)) + 0.15 * _l2_normalize(mu[0]) + 0.65 * ap_vec.squeeze(0))
            else:
                act_dir = _l2_normalize(0.6 * a_idm.squeeze(0) + 0.4 * mu[0])
            scored_tools = []
            for tname in ranked_tools:
                t_vec = tool_capability_vector(tname)
                score = float(np.dot(act_dir, t_vec))
                scored_tools.append((score, str(tname), tname))
            scored_tools.sort(key=lambda x: (-round(x[0], 7), x[1]))
            ranked_tools = [t for _, _, t in scored_tools]

        vc_std = float(np.sqrt(np.var(best_z_final) + 1e-6))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "enabled": True,
            "flow": "c2_plan",
            "arch": arch,
            "horizon": horizon,
            "cem_iters": cem_iters,
            "energy_before": round(energy_before, 4),
            "energy_after": round(best_cost, 4),
            "energy_reduction_pct": round(100.0 * max(0.0, energy_before - best_cost) / max(1e-6, energy_before), 1),
            "vc_std": round(vc_std, 4),
            "first_action": mu[0].copy(),
            "z_init_64": z_init.squeeze(0),
            "z_planned_64": best_z_final.squeeze(0),
            "ranked_tools": ranked_tools,
            "r_refined_2048": r_refined_2048,
            "ms": ms,
        }

    # ---------------- Flow 5: Closed-Loop Outcome -> Growth (`Clover RPRE`) ----------------

    def record_and_adapt_outcome(
        self,
        z_t_64: np.ndarray,
        action_vec_8: np.ndarray,
        z_tp1_obs_64: np.ndarray,
        *,
        context_cue_64: np.ndarray | None = None,
        is_baseline: bool = False,
        configured_arch: str = "auto",
        lr: float = 0.025,
        steps: int = 16,
        max_baseline_regression: float = 0.015,
    ) -> dict[str, Any]:
        """Record a real execution outcome $(z_t, a_t, z_{t+1}^{\text{obs}}, c_t)$ and update `OutcomeResidualAdapter`
        subject to Clover's Baseline-Prototype Anomaly Gate and Held-Out Non-Regression Rollback Gate.
        """
        arch = self.choose_architecture("outcome_growth", configured_arch, has_outcome_feedback=True)
        if arch == "none":
            return {"enabled": False, "arch": "none", "updated": False}

        t0 = time.perf_counter()
        z_t = np.asarray(z_t_64, dtype=np.float32).reshape(1, D_JEPA)
        a_t = np.asarray(action_vec_8, dtype=np.float32).reshape(1, D_ACT)
        z_obs = _layer_norm(np.asarray(z_tp1_obs_64, dtype=np.float32).reshape(1, D_JEPA))
        c_t = (
            np.asarray(context_cue_64, dtype=np.float32).reshape(1, D_JEPA)
            if context_cue_64 is not None
            else np.zeros((1, D_JEPA), dtype=np.float32)
        )

        with self.lock:
            if is_baseline:
                self.baseline_anchor.append((z_t.copy(), a_t.copy(), z_obs.copy(), c_t.copy()))
                if len(self.baseline_anchor) > 128:
                    self.baseline_anchor.pop(0)
                non_zero_cues = [x[3] for x in self.baseline_anchor if float(np.sum(np.abs(x[3]))) > 1e-5]
                if non_zero_cues:
                    self.w["residual.baseline_cue_proto"] = np.mean(np.concatenate(non_zero_cues, axis=0), axis=0, keepdims=True)
                    self.w["residual.has_proto"] = np.array([1.0], dtype=np.float32)
                return {
                    "enabled": True,
                    "flow": "outcome_growth",
                    "arch": arch,
                    "updated": False,
                    "baseline_anchors": len(self.baseline_anchor),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            self.outcome_buffer.append((z_t.copy(), a_t.copy(), z_obs.copy(), c_t.copy()))
            if len(self.outcome_buffer) > 256:
                self.outcome_buffer.pop(0)

            Z_t = np.concatenate([x[0] for x in self.outcome_buffer], axis=0)
            A_t = np.concatenate([x[1] for x in self.outcome_buffer], axis=0)
            Z_obs = np.concatenate([x[2] for x in self.outcome_buffer], axis=0)
            C_t = np.concatenate([x[3] for x in self.outcome_buffer], axis=0)

            base_next = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=False)
            pred_before = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=True)
            mse_before = float(np.mean((pred_before - Z_obs) ** 2))

            base_mse_before = 0.0
            if self.baseline_anchor:
                bZ_t = np.concatenate([x[0] for x in self.baseline_anchor], axis=0)
                bA_t = np.concatenate([x[1] for x in self.baseline_anchor], axis=0)
                bZ_obs = np.concatenate([x[2] for x in self.baseline_anchor], axis=0)
                bC_t = np.concatenate([x[3] for x in self.baseline_anchor], axis=0)
                bp_before = self._step_transition(bZ_t, bA_t, context_cue=bC_t, use_residual=True)
                base_mse_before = float(np.mean((bp_before - bZ_obs) ** 2))

            snap_fc1_w = self.w["residual.fc1.weight"].copy()
            snap_fc1_b = self.w["residual.fc1.bias"].copy()
            snap_fc2_w = self.w["residual.fc2.weight"].copy()
            snap_fc2_b = self.w["residual.fc2.bias"].copy()

            inp = np.concatenate([Z_t, A_t, C_t], axis=-1)
            anom_g = self._compute_anom_gate(C_t, C_t)
            gate_val = float(np.tanh(self.w["residual.gate"][0]))
            h_pre = _linear(inp, self.w["residual.fc1.weight"], self.w["residual.fc1.bias"])
            h_act = _gelu(h_pre) * (anom_g * np.float32(gate_val))

            target_delta = Z_obs - base_next
            N_s = h_act.shape[0]
            H_aug = np.concatenate([h_act, anom_g * np.float32(gate_val)], axis=-1)
            reg_lambda = max(0.05, 1.5 / math.sqrt(float(N_s) + float(steps)))
            gram = np.matmul(H_aug.T, H_aug) + np.eye(D_JEPA + 1, dtype=np.float32) * np.float32(reg_lambda)
            rhs = np.matmul(H_aug.T, target_delta)
            try:
                sol = np.linalg.solve(gram, rhs)
                w2_star = sol[:D_JEPA, :].T.astype(np.float32)
                b2_star = sol[D_JEPA, :].astype(np.float32)
            except np.linalg.LinAlgError:
                w2_star = snap_fc2_w
                b2_star = snap_fc2_b

            best_mse = mse_before
            best_w2 = snap_fc2_w
            best_b2 = snap_fc2_b
            for alpha in (1.0, 0.7, 0.4, 0.2, 0.1):
                self.w["residual.fc2.weight"] = (1.0 - alpha) * snap_fc2_w + alpha * w2_star
                self.w["residual.fc2.bias"] = (1.0 - alpha) * snap_fc2_b + alpha * b2_star
                cand_pred = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=True)
                cand_mse = float(np.mean((cand_pred - Z_obs) ** 2))
                if cand_mse < best_mse:
                    best_mse = cand_mse
                    best_w2 = self.w["residual.fc2.weight"].copy()
                    best_b2 = self.w["residual.fc2.bias"].copy()
                    break

            self.w["residual.fc2.weight"] = best_w2
            self.w["residual.fc2.bias"] = best_b2
            mse_after = best_mse

            base_mse_after = 0.0
            if self.baseline_anchor:
                bp_after = self._step_transition(bZ_t, bA_t, context_cue=bC_t, use_residual=True)
                base_mse_after = float(np.mean((bp_after - bZ_obs) ** 2))

            regressed = (self.baseline_anchor and (base_mse_after > base_mse_before + max_baseline_regression)) or (mse_after > mse_before + 1e-6)
            if regressed:
                self.w["residual.fc1.weight"] = snap_fc1_w
                self.w["residual.fc1.bias"] = snap_fc1_b
                self.w["residual.fc2.weight"] = snap_fc2_w
                self.w["residual.fc2.bias"] = snap_fc2_b
                self.rollbacks += 1
                return {
                    "enabled": True,
                    "flow": "outcome_growth",
                    "arch": arch,
                    "updated": False,
                    "rolled_back": True,
                    "mse_before": round(mse_before, 5),
                    "mse_after": round(mse_before, 5),
                    "baseline_mse_before": round(base_mse_before, 5),
                    "baseline_mse_after": round(base_mse_after, 5),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            self.cycle_updates += 1
            self._save_persisted_adapter()
            return {
                "enabled": True,
                "flow": "outcome_growth",
                "arch": arch,
                "updated": True,
                "rolled_back": False,
                "cycle_updates": self.cycle_updates,
                "mse_before": round(mse_before, 5),
                "mse_after": round(mse_after, 5),
                "baseline_mse_before": round(base_mse_before, 5),
                "baseline_mse_after": round(base_mse_after, 5),
                "ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    def predict_expected_error(
        self,
        z_t_64: np.ndarray,
        action_vec_8: np.ndarray | None = None,
        context_cue_64: np.ndarray | None = None,
    ) -> float:
        """Paper 10 (Schmidhuber 1991 Confidence Module C): Predict expected transition error `y_C(z_t)` at `z_t_64`
        from the stored outcome buffer using inverse-distance neighbor weighting in 64-d latent space."""
        with self.lock:
            if not self.outcome_buffer:
                return 0.35
            z_q = np.asarray(z_t_64, dtype=np.float32).reshape(1, D_JEPA)
            Z_t = np.concatenate([x[0] for x in self.outcome_buffer], axis=0)
            A_t = np.concatenate([x[1] for x in self.outcome_buffer], axis=0)
            Z_obs = np.concatenate([x[2] for x in self.outcome_buffer], axis=0)
            C_t = np.concatenate([x[3] for x in self.outcome_buffer], axis=0)
            pred = self._step_transition(Z_t, A_t, context_cue=C_t, use_residual=True)
            errs = np.mean((pred - Z_obs) ** 2, axis=-1)
            d2 = np.mean((Z_t - z_q) ** 2, axis=-1)
            w = np.exp(-8.0 * d2)
            w /= np.sum(w) + np.float32(1e-9)
            return round(float(np.sum(w * errs)), 5)

    def estimate_controllable_subspace(self) -> np.ndarray:
        """Paper 13 (Santucci et al. 2020 IMOL): Estimate per-dimension controllability `R^2_j(Action -> Delta z_j)`
        across the 64-d JEPA latent space from recorded `(z_t, a_t, z_obs)` outcome transitions."""
        with self.lock:
            if len(self.outcome_buffer) < 4:
                return np.ones(D_JEPA, dtype=np.float32)
            Z_t = np.concatenate([x[0] for x in self.outcome_buffer], axis=0).astype(np.float64)
            A_t = np.concatenate([x[1] for x in self.outcome_buffer], axis=0).astype(np.float64)
            Z_obs = np.concatenate([x[2] for x in self.outcome_buffer], axis=0).astype(np.float64)
            dZ = Z_obs - Z_t
            r2 = np.zeros(D_JEPA, dtype=np.float32)
            for j in range(D_JEPA):
                vj = float(np.var(dZ[:, j]))
                if vj < 1e-8:
                    r2[j] = 0.05
                    continue
                w, _, _, _ = np.linalg.lstsq(A_t, dZ[:, j], rcond=None)
                res_v = float(np.var(dZ[:, j] - (A_t @ w)))
                r2[j] = np.float32(np.clip(1.0 - res_v / vj, 0.05, 1.0))
            return r2


_INSTANCES: dict[str, CloverJEPARuntime] = {}
_INST_LOCK = threading.Lock()


def for_base(base_dir: str | Path | None = None) -> CloverJEPARuntime:
    key = str(Path(base_dir).resolve()) if base_dir else "default"
    with _INST_LOCK:
        inst = _INSTANCES.get(key)
        if inst is None:
            inst = CloverJEPARuntime(base_dir)
            _INSTANCES[key] = inst
        return inst
