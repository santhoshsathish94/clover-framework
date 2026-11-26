"""Layer 1 as LFM3-Omni-2B (LiquidAI): text, images, audio and video in; text and speech out; on llama.cpp.

The Omni's language model has LFM3-1.7B's layout but its own weights; it is a non-reasoning model (its card:
`enable_thinking` unsupported; llama-server runs it with `--reasoning off`). llama-server applies the chat template
stored in the GGUF, turns media parts into the model's media tokens (the projector holds the vision and audio
encoders; video needs ffmpeg) and speaks text through /tts. The Studio keeps its own tool loop, permissions and
Clover layers on top, the same way as for LFM3-1.7B (lfm3_native.LFM3Native, whose tools this class reuses).
Media files live in CloverAI\\media (the user clears or backs them up) and reach llama-server as file:// names inside
--media-path.
"""
import datetime
import json
import re
import threading
import time
import uuid
from pathlib import Path

import llamacpp_engine as L
import lfm3_native as N
import clover_cycle as C
import paths
import semantics

QUANTS = {
    "q4_0": "LFM3-Omni-2B-Q4_0.gguf",
    "q4_k_m": "LFM3-Omni-2B-Q4_K_M.gguf",
    "q8_0": "LFM3-Omni-2B-Q8_0.gguf",
    "bf16": "LFM3-Omni-2B-BF16.gguf",
}
MMPROJ = "mmproj-LFM3-Omni-2B-Q8_0.gguf"
MMPROJ_BF16 = "mmproj-LFM3-Omni-2B-BF16.gguf"
SPEECH_LANGS = ("en", "ar", "da", "de", "el", "fi", "fr", "he", "it", "ja", "ko", "nl", "pl", "pt", "es", "sv", "tr", "zh")
PART = {"image": "image_url", "audio": "input_audio", "video": "input_video"}
GREEDY = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
          "xtc_probability": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}  # the card: greedy for text output
TOOL_XML = re.compile(r"<tool_call>[\s\S]*?(?:</tool_call>|$)")
THINK_PREFIX = re.compile(r"^\s*<think>\s*</think>\s*")
TOOL_OPEN_RE = re.compile(r"<tool_call>|<function=")
LOOP_TAIL = 160
# a forced round (tool_choice required) writes its text first: "which application would you like to install? Could you
# please tell me:" came with a run_powershell call (tests/tool_choice_probe.py)
ASKS_USER = re.compile(r"\?\s*(?:\n|$)|\b(?:could|can|would) you (?:please )?(?:tell|provide|confirm|clarify|share|specify)\b"
                       r"|\bplease (?:tell|provide|confirm|clarify|specify)\b", re.I)
PROMISE = semantics.PROMISE
ANDROID = paths.ANDROID
PLATFORM = "Android phone" if ANDROID else "Windows laptop"


def looping(text):
    """The last 60..160 characters appeared twice before: greedy decoding is repeating itself (a user's request ran to the
    2,048-token limit in 89 s that way)."""
    if text.count("```") % 2 == 0:
        for w in (LOOP_TAIL, 96, 64):
            if len(text) >= 3 * w and text.count(text[-w:], 0, len(text) - w) >= 2:
                return True
    return len(text) >= 4 * LOOP_TAIL and text.count(text[-LOOP_TAIL:], 0, len(text) - LOOP_TAIL) >= 2


def trim_loop(text):
    """One copy of the repeated text, ending at its last full sentence: the fixed cut ended a story on "grew lou"
    (2026-11-11). Inside an unclosed code block the cut stays where it is."""
    w_used = LOOP_TAIL
    if text.count("```") % 2 == 0:
        for w in (LOOP_TAIL, 96, 64):
            if len(text) >= 3 * w and text.count(text[-w:], 0, len(text) - w) >= 2:
                w_used = w
                break
    tail = text[-w_used:]
    kept = text[:text.find(tail) + w_used]
    if kept.count("```") % 2:
        return kept
    ends = [m.end() for m in re.finditer(r"[.!?][\"')\]”’]*(?=\s|$)", kept)]
    return kept[:ends[-1]] if ends else kept


# a paragraph that closes an answer: "Just let me know what you'd like to do!", "I'm here to help!"
SIGN_OFF = re.compile(r"\blet\s+me\s+know\b|\bfeel\s+free\s+to\b|\bi'?m\s+(?:here|ready)\s+to\s+help\b|\bhappy\s+to\s+help\b|"
                      r"\banything\s+else\b|\bwhat\s+would\s+you\s+like\s+to\s+(?:do|work\s+on|see|create)\b|"
                      r"\bwhat\s+would\s+you\s+like\s+the\s+(?:image|picture)\s+to\s+show\b|\bready\s+when\s+you\s+are\b", re.I)
TRAILING_Q = re.compile(
    r"\n\n+(?:(?:Would|Do)\s+you\s+(?:like|want|need)\b|What\s+(?:else\s+)?would\s+you\s+like\b|"
    r"Is\s+there\s+anything\s+else\b|Shall\s+I\b|If\s+you(?:'d|\s+would)\s+like[^\n]*\?)"
    r"[\s\S]*\?\s*$", re.I)


def rambling(text):
    """-> where the answer ends, or None. Past its sign-off the Omni wrote one variation after another (\"If you'd like
    me to ..., I can do that too. Just let me know ...!\" six times, 37 s, 2026-11-04), each different enough for
    looping(). Also catches repeated sign-off sentences inside a single paragraph (\"What would you like to see? Please
    provide a clear and detailed description.\" repeated 3 times on the phone, 2026-11-14)."""
    if text.count("```") % 2 == 0 and len(text) >= 180:
        seen_s = {}
        for sm in re.finditer(r"[^.!?\n]{20,220}[.!?]+", text):
            s_raw = sm.group(0).strip()
            s_key = re.sub(r"[^a-z0-9 ]+", "", s_raw.lower()).strip()
            if len(s_key) >= 22:
                if s_key in seen_s and (SIGN_OFF.search(s_raw) or sm.start() - seen_s[s_key] >= 40):
                    return sm.start()
                seen_s.setdefault(s_key, sm.end())
    offs = [m for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text) if SIGN_OFF.search(m.group(0))]
    if len(offs) < 3:
        return None
    seen = set()
    for m in offs:
        for s in re.split(r"(?<=[.!?])\s+", m.group(0)):
            if SIGN_OFF.search(s):
                key = re.sub(r"[^a-z' ]+", "", s.lower()).strip()
                if key in seen:
                    return offs[0].end()
                seen.add(key)
    return None
# Speech runs on a CPU llama.cpp instance when chat has the Iris Xe. Measured on the i5-1335U (tests/omni_resource_probe,
# omni_prefill_probe): speech 1.4-1.6x real time on 10 CPU threads against 0.8x on the GPU, while the GPU reads a
# 2,700-token Studio prompt at 438 tok/s against 94 on the CPU (writing about equal, 26-28 tok/s). The instance holds
# about 1.6 GB, so it starts on first use and stops after VOICE_IDLE_S without speech.
VOICE_IDLE_S = 600
VOICE_THREADS = 10
# C1 inference only: one request carries the whole conversation and must fit in one ubatch (ubatch = context)
C1_INFER_CTX = 8192
MEDIA_WORD = {"image_url": "image", "input_audio": "audio", "input_video": "video"}


def _resolve_image_prompt(query: str, past_msgs=()) -> str:
    """Resolve the visual description to pass to `generate_image`, looking back at `past_msgs` when `query` is a
    meta follow-up such as 'Ask nemotron to generate', 'Where?', 'Try again', or 'Where is the <subject> it's just description i want the image'."""
    q = " ".join(str(query or "").split()).strip()
    m_where = re.search(r"\bwhere\s+is\s+the\s+([a-z0-9][a-z0-9\s,\-]{1,60}?)(?:\s+it'?s\b|\s+i\s+want\b|[?.!,]|$)", q, re.I)
    if m_where and m_where.group(1).strip().lower() not in ("image", "picture", "photo", "drawing", "illustration", "art", "artwork"):
        return m_where.group(1).strip()
    is_meta_followup = bool(
        re.match(r"^\W*(?:where(?:\s+is\s+(?:it|the\s+(?:image|picture|photo|drawing|illustration)))?|try\s+again|retry|redo|generate\s+it|make\s+it|draw\s+it|show\s+me)\s*[?.!]*\W*$", q, re.I)
        or re.search(r"\b(?:ask|use|have|let|tell)\s+(?:nemotron|kimi|clover|lfm\w*|the\s+model)\s+to\s+(?:generate|create|make|draw|render)\b", q, re.I)
        or re.search(r"\b(?:it'?s\s+just\s+(?:a\s+)?description|i\s+want\s+the\s+(?:image|picture))\b", q, re.I)
    )
    if not is_meta_followup and q:
        cleaned = re.sub(
            r"^\W*(?:(?:please|just|can\s+you|could\s+you|would\s+you)\s+)*(?:generate|create|make|draw|paint|render|sketch|illustrate|show\s+me|give\s+me)(?:\s+me)?\s+",
            "",
            q,
            flags=re.I,
        ).strip(" .?!")
        return cleaned if len(cleaned) >= 2 else q
    for m in reversed(list(past_msgs or ())):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        if m.get("role") == "user":
            u_c = " ".join(c.split()).strip()
            if re.match(r"^\W*(?:(?:can|could|do|are)\s+you\s+(?:able\s+to\s+)?(?:generate|create|make|draw)|where|try\s+again|hi|hello)\b", u_c, re.I):
                continue
            if re.search(r"\b(?:ask|use)\s+(?:nemotron|kimi)\b|\bit'?s\s+just\s+(?:a\s+)?description\b", u_c, re.I):
                continue
            return _resolve_image_prompt(u_c, ())
    return q


def omni_files(base_dir, quant="q4_0"):
    """-> (paths, missing names) for the Omni on llama.cpp."""
    base_dir = Path(base_dir)
    q = quant if quant in QUANTS else "q4_0"
    g = base_dir / "data" / "omni" / "gguf"
    mm = (g / MMPROJ_BF16) if (q == "bf16" and (g / MMPROJ_BF16).is_file()) else (g / MMPROJ)
    out = {"model": g / QUANTS[q], "mmproj": mm}
    missing = [p.name for p in out.values() if not p.is_file()]
    return out, missing


def available_quants(base_dir):
    """Which LFM3-Omni-2B GGUF precisions are currently downloaded on disk."""
    g = Path(base_dir) / "data" / "omni" / "gguf"
    mm_ok = (g / MMPROJ).is_file() or (g / MMPROJ_BF16).is_file()
    return {k: bool(mm_ok and (g / fname).is_file()) for k, fname in QUANTS.items()}


def detect_speech_lang(text, preferred="auto"):
    """Resolve one of LFM3-Omni-2B's 19 supported speech languages (`SPEECH_LANGS`)."""
    pref = str(preferred or "auto").strip().lower()
    if pref in SPEECH_LANGS:
        return pref
    s = str(text or "")
    if re.search(r"[\u0600-\u06FF]", s):
        return "ar"
    if re.search(r"[\u0590-\u05FF]", s):
        return "he"
    if re.search(r"[\u0370-\u03FF]", s):
        return "el"
    if re.search(r"[\uAC00-\uD7AF]", s):
        return "ko"
    if re.search(r"[\u3040-\u30FF]", s):
        return "ja"
    if re.search(r"[\u4E00-\u9FFF]", s):
        return "zh"
    return "en"


def media_part(m, *, native_video=True):
    kind = m["kind"]
    if kind == "video" and not native_video and m.get("frame_file"):
        return {"type": "image_url", "image_url": {"url": "file://" + m["frame_file"]}}
    url = "file://" + m["file"]
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": PART[kind], PART[kind]: {"url": url}}


def user_content(text, media, media_dir, *, native_video=True):
    """OpenAI content for a user message: media first (llama.cpp's advice), then the text when non-empty.
    For videos: when `native_video` is True (ffmpeg available), sends `input_video` so LFM3-Omni-2B watches 2 fps
    frames AND hears the interleaved soundtrack; on Android without ffmpeg, extracts 2 fps keyframes + 16 kHz WAV."""
    media_p = Path(media_dir)
    parts = []
    for m in media or ():
        if not isinstance(m, dict) or not m.get("file") or not (media_p / m["file"]).is_file():
            continue
        if m.get("kind") == "video" and not native_video:
            frames, aud_wav = [], None
            if ANDROID:
                try:
                    import android_media
                    frames, aud_wav = android_media.extract_video_frames_and_audio(media_p / m["file"], media_p)
                except Exception:
                    frames, aud_wav = [], None
            if frames:
                for fn in frames:
                    parts.append({"type": "image_url", "image_url": {"url": "file://" + fn}})
                if aud_wav and (media_p / aud_wav).is_file():
                    parts.append({"type": "input_audio", "input_audio": {"url": "file://" + aud_wav}})
                continue
        parts.append(media_part(m, native_video=native_video))
    txt = str(text or "")
    if not parts:
        return txt
    return parts + ([{"type": "text", "text": txt}] if txt.strip() else [])


def run_s3_sensory_context(base_dir, media_dir, media, query, *, kit=None, emit=None, turn_images=None):
    """Model-agnostic S3 sensory analysis (`Audio-JEPA` for audio, `image_inspector` + `I-JEPA` + `FFprobe` + keyframe
    extraction for images & video, and `read_active_activity` for screen captures).
    Runs identically across all models (`OmniNative`, `LFM3Native`, `K3HostEngine`, `RemoteOmniEngine`, `FusedEngine`).
    Returns `(sensory_system_msgs, sensory_layers, has_new_visual, has_new_audio)`.
    """
    emit = emit or (lambda e: None)
    base_p = Path(base_dir) if base_dir else Path.cwd()
    media_p = Path(media_dir) if media_dir else (base_p / "media")
    has_new_visual = any(m.get("kind") in ("image", "video") for m in media or () if isinstance(m, dict))
    has_new_audio = any(m.get("kind") == "audio" for m in media or () if isinstance(m, dict))
    jepa_arch_cfg = getattr(kit, "jepa_arch", "auto") if kit is not None else "auto"
    if kit is not None:
        kit.visual_cue_64 = None
        kit.audio_cue_64 = None

    sensory_msgs = []
    sensory_layers = []

    if has_new_audio and jepa_arch_cfg != "none":
        try:
            import math
            import numpy as np
            import jepa_engine
            aud_blocks = []
            jrt = jepa_engine.for_base(base_p)
            for m in media or ():
                if isinstance(m, dict) and m.get("kind") == "audio" and m.get("file"):
                    aud_fp = media_p / m["file"]
                    if aud_fp.is_file():
                        samples = jrt._extract_audio_samples(aud_fp)
                        dur_s = round(samples.size / 16000.0, 2) if samples.size > 0 else 0.0
                        rms_val = float(np.sqrt(np.mean(samples * samples) + 1e-12)) if samples.size > 0 else 1e-6
                        peak_val = float(np.max(np.abs(samples)) + 1e-12) if samples.size > 0 else 1e-6
                        rms_dbfs = round(max(-80.0, 20.0 * math.log10(rms_val)), 1)
                        peak_dbfs = round(max(-80.0, 20.0 * math.log10(peak_val)), 1)
                        ja = jrt.inspect_audio_latent(samples, configured_arch=jepa_arch_cfg)
                        if ja.get("enabled"):
                            if kit is not None:
                                kit.audio_cue_64 = ja.get("z_jepa")
                            info_aud = {
                                "layer": "jepa_audio",
                                "arch": ja["arch"],
                                "file": m["file"],
                                "acoustic_regime": ja["acoustic_regime"],
                                "snr_db": ja["snr_db"],
                                "tf_latent_cos": ja["tf_latent_cos"],
                                "duration_s": dur_s,
                                "rms_dbfs": rms_dbfs,
                                "peak_dbfs": peak_dbfs,
                                "ms": ja["ms"],
                            }
                            sensory_layers.append(info_aud)
                            emit(dict(info_aud, type="layer"))
                            aud_blocks.append(
                                f"Audio-JEPA Acoustic & Prosody Snapshot ({m.get('name') or m['file']}):\n"
                                f"- Duration: {dur_s} s (16,000 Hz mono PCM)\n"
                                f"- Acoustic Regime (Audio-JEPA {ja['arch']}): {ja['acoustic_regime']}\n"
                                f"- Estimated SNR: {ja['snr_db']} dB | RMS Level: {rms_dbfs} dBFS | Peak Level: {peak_dbfs} dBFS\n"
                                f"- Time-Frequency Latent Cosine: {ja['tf_latent_cos']} (inspected in {ja['ms']} ms)"
                            )
            if aud_blocks and (query or "").strip():
                sensory_msgs.append({
                    "role": "system",
                    "content": (
                        "S3 Audio-JEPA Acoustic Analysis (measured on the attached audio recording before C1 inference):\n\n"
                        + "\n\n".join(aud_blocks)
                        + "\n\nUse both the audio content and these measured Audio-JEPA acoustic metrics (duration, acoustic regime, SNR dB, RMS/peak dBFS) to answer the user's question."
                    ),
                })
        except Exception:
            pass

    if has_new_visual:
        try:
            import hashlib
            import os
            import subprocess
            import image_inspector
            snap_blocks = []
            for m in media or ():
                if not isinstance(m, dict) or not m.get("file"):
                    continue
                img_fp = None
                vid_hdr = ""
                if m.get("kind") == "image":
                    cand_fp = media_p / m["file"]
                    if cand_fp.is_file():
                        img_fp = cand_fp
                elif m.get("kind") == "video":
                    vid_fp = media_p / m["file"]
                    if vid_fp.is_file():
                        try:
                            import local_tool_packs
                            vid_hdr = "Video Stream Metadata (FFprobe):\n" + local_tool_packs.media_info(
                                vid_fp, base_dir=base_p
                            ) + "\n\n"
                        except Exception:
                            vid_hdr = f"Attached Video File: {m.get('name') or m['file']} ({vid_fp.stat().st_size} bytes)\n\n"
                        fr_name = m.get("frame_file") or ""
                        if fr_name and (media_p / fr_name).is_file():
                            img_fp = media_p / fr_name
                        else:
                            try:
                                import device_actions
                                ff_exe = device_actions._find_ffmpeg_exe(base_p)
                                if ff_exe is not None:
                                    tmp_fr = media_p / f"tmp_vfr_{uuid.uuid4().hex[:8]}.png"
                                    try:
                                        subprocess.run(
                                            [str(ff_exe), "-hide_banner", "-y", "-ss", "0.5", "-i", str(vid_fp), "-frames:v", "1", str(tmp_fr)],
                                            capture_output=True,
                                            timeout=10,
                                            creationflags=0x08000000 if os.name == "nt" else 0,
                                        )
                                        if not tmp_fr.is_file() or tmp_fr.stat().st_size < 256:
                                            subprocess.run(
                                                [str(ff_exe), "-hide_banner", "-y", "-i", str(vid_fp), "-frames:v", "1", str(tmp_fr)],
                                                capture_output=True,
                                                timeout=10,
                                                creationflags=0x08000000 if os.name == "nt" else 0,
                                            )
                                        if tmp_fr.is_file() and tmp_fr.stat().st_size >= 256:
                                            fb = tmp_fr.read_bytes()
                                            fr_name = hashlib.sha256(fb).hexdigest()[:32] + ".png"
                                            img_fp = media_p / fr_name
                                            img_fp.write_bytes(fb)
                                            m["frame_file"] = fr_name
                                    finally:
                                        tmp_fr.unlink(missing_ok=True)
                            except Exception:
                                pass
                        if img_fp is not None and turn_images is not None:
                            if not any(x.get("file") == img_fp.name for x in turn_images):
                                turn_images.append({
                                    "file": img_fp.name,
                                    "prompt": m.get("name") or "Recorded Video Frame",
                                    "model": "video_frame",
                                    "seconds": 0,
                                    "width": None,
                                    "height": None,
                                })
                if img_fp is not None and img_fp.is_file():
                    is_cam_or_vid = bool(m.get("camera_capture") or m.get("kind") == "video")
                    rep = image_inspector.inspect_image(
                        img_fp, annotate_dir=media_p, jepa_arch=jepa_arch_cfg, skip_ocr=is_cam_or_vid
                    )
                    snap_blocks.append(vid_hdr + image_inspector.format_snapshot_for_prompt(rep))
                    jv = rep.get("jepa_visual") or {}
                    if kit is not None and jv.get("arch"):
                        if jv.get("z_jepa") is not None:
                            kit.visual_cue_64 = jv.get("z_jepa")
                        else:
                            try:
                                import jepa_engine
                                from PIL import Image
                                with Image.open(img_fp) as _im:
                                    _jv_full = jepa_engine.for_base(base_p).inspect_visual_latent(_im.convert("RGB"), configured_arch=jepa_arch_cfg)
                                    kit.visual_cue_64 = _jv_full.get("z_jepa")
                            except Exception:
                                pass
                    info_img = {
                        "layer": "image_analysis",
                        "file": img_fp.name,
                        "annotated_file": rep.get("annotated_file"),
                        "width": rep["width"],
                        "height": rep["height"],
                        "elements": len(rep.get("elements") or []),
                        "ocr_lines": len(rep.get("ocr_lines") or []),
                        "jepa_arch": jv.get("arch"),
                        "jepa_scene": jv.get("scene_kind"),
                        "ms": rep.get("ms", 0),
                    }
                    sensory_layers.append(info_img)
                    emit(dict(info_img, type="layer"))
                elif vid_hdr:
                    snap_blocks.append(vid_hdr)
            if snap_blocks:
                act_hdr = ""
                try:
                    import device_actions
                    if device_actions.asks_for_screen_or_activity(query) or any(m.get("screen_capture") for m in media or () if isinstance(m, dict)):
                        act = device_actions.read_active_activity(
                            include_clipboard=device_actions.mentions_clipboard(query))
                        parts_a = []
                        if act.get("foreground_window"):
                            parts_a.append(f"- Active Foreground Window: {act['foreground_window']}")
                        if act.get("open_windows"):
                            parts_a.append("- Open Application Windows: " + "; ".join(act["open_windows"][:10]))
                        if act.get("clipboard_preview"):
                            parts_a.append(f"- Clipboard Text Preview: {act['clipboard_preview'][:400]}")
                        if parts_a:
                            act_hdr = "Live Device Activity Context:\n" + "\n".join(parts_a) + "\n\n"
                except Exception:
                    pass
                sensory_msgs.append({
                    "role": "system",
                    "content": (
                        act_hdr
                        + "S3 Playwright-Style Visual & UI Inspection of attached image/video "
                        "(use these exact OCR texts, [e1..eN] bounding boxes, dimensions, and hex colors together with what you see in the media; "
                        "describe the image naturally and never call the image a 'Playwright-Style snapshot'):\n"
                        + "\n\n".join(snap_blocks)
                    ),
                })
        except Exception:
            pass

    return sensory_msgs, sensory_layers, has_new_visual, has_new_audio


def plan_s3_pre_calls(engine, query, only_tools, by_name, *, meaning=None, past_msgs=()):
    """Plan deterministic S3 pre-calls (`fetch_url`, `web_search`,
    `get_current_time`, `laptop_status`, `run_python`, `remember`, `vscode_open`, `read_file`) across all model engines."""
    s3_pre_call = None
    s3_extra_calls = []
    if not only_tools or not by_name:
        return None, []
    url_m = re.search(r"https?://\S+", query or "")
    base_p = Path(getattr(engine, "base_dir", None) or Path.cwd())
    workdir = getattr(engine, "workdir", None)
    if "fetch_url" in only_tools and "fetch_url" in by_name and url_m:
        u = url_m.group(0).rstrip(".,;)'>\"]")
        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "fetch_url", "arguments": {"url": u}}
    elif (
        "web_search" in only_tools
        and "web_search" in by_name
        and set(only_tools) <= {"web_search", "fetch_url"}
        and not url_m
    ):
        ws_q = query
        if meaning is not None and getattr(meaning, "follow_up", False) and past_msgs:
            prev_u = next(
                (str(m.get("content") or "") for m in reversed(past_msgs)
                 if m.get("role") == "user" and isinstance(m.get("content"), str)),
                "",
            )
            if prev_u and len((query or "").split()) <= 6:
                ws_q = f"{prev_u.strip()} {query.strip()}".strip()[:200]
        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "web_search", "arguments": {"query": ws_q}}
    elif list(only_tools) == ["get_current_time"] and "get_current_time" in by_name:
        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "get_current_time", "arguments": {}}
    elif list(only_tools) == ["laptop_status"] and "laptop_status" in by_name:
        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "laptop_status", "arguments": {}}
    elif "convert_units" in only_tools and "convert_units" in by_name:
        m_u = re.search(
            r"\bconvert\s+(-?\d[\d.,]*)\s+([A-Za-z\u00b0/\u00b2\u00b3]+(?:\s+(?:per|square|cubic)\s+[A-Za-z\u00b0/\u00b2\u00b3]+|\s+[A-Za-z\u00b0/\u00b2\u00b3]+)?)\s+(?:to|into|in)\s+([A-Za-z\u00b0/\u00b2\u00b3\s]+?)(?:[?.!]|$)",
            query or "",
            re.I,
        )
        if m_u:
            val_u = float(m_u.group(1).replace(",", ""))
            from_u = m_u.group(2).strip()
            to_raw = m_u.group(3).strip()
            to_parts = [p.strip() for p in re.split(r"\s+and\s+|\s*,\s*", to_raw) if p.strip()]
            if to_parts:
                s3_pre_call = {
                    "id": "call_" + uuid.uuid4().hex[:16],
                    "name": "convert_units",
                    "arguments": {"value": val_u, "from_unit": from_u, "to_unit": to_parts[0]},
                }
                for extra_to in to_parts[1:3]:
                    s3_extra_calls.append({
                        "id": "call_" + uuid.uuid4().hex[:16],
                        "name": "convert_units",
                        "arguments": {"value": val_u, "from_unit": from_u, "to_unit": extra_to},
                    })
    elif "calculate" in only_tools and "calculate" in by_name:
        ci = C.parse_compound_interest(query or "")
        if ci is not None:
            P, r_pct, n_yr, years, _ = ci
            expr_final = f"round({P} * (1 + ({r_pct} / 100) / {n_yr}) ** ({n_yr} * {years}), 2)"
            expr_int = f"round({P} * (1 + ({r_pct} / 100) / {n_yr}) ** ({n_yr} * {years}) - {P}, 2)"
            s3_pre_call = {
                "id": "call_" + uuid.uuid4().hex[:16],
                "name": "calculate",
                "arguments": {"expression": expr_final},
            }
            s3_extra_calls.append({
                "id": "call_" + uuid.uuid4().hex[:16],
                "name": "calculate",
                "arguments": {"expression": expr_int},
            })
    elif (
        "generate_image" in only_tools
        and "generate_image" in by_name
        and not re.match(
            r"^\W*(?:can|could|do|are|will|would)\s+you\s+(?:able\s+to\s+)?(?:generate|create|make|draw|render|produce)\s+(?:any\s+|some\s+)?(?:images|pictures|photos|drawings|illustrations|artwork|art)\s*[?.!]*\W*$",
            query or "",
            re.I,
        )
    ):
        img_prompt = _resolve_image_prompt(query or "", past_msgs=past_msgs)
        if img_prompt:
            s3_pre_call = {
                "id": "call_" + uuid.uuid4().hex[:16],
                "name": "generate_image",
                "arguments": {"prompt": img_prompt},
            }
    elif "run_python" in only_tools and "run_python" in by_name and re.search(r"```(?:python|py)\b", query or "", re.I):
        import lfm3_layers
        py_code = lfm3_layers.extract_python(query or "")
        if py_code.strip():
            s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "run_python", "arguments": {"code": py_code.strip()}}
    elif list(only_tools) == ["remember"] and "remember" in by_name and meaning is not None and getattr(meaning, "remember", ""):
        note = str(meaning.remember).strip()
        if note:
            s3_pre_call = {
                "id": "call_" + uuid.uuid4().hex[:16],
                "name": "remember",
                "arguments": {"text": note[0].upper() + note[1:] + ("" if note.endswith(".") else ".")},
            }
    elif "vscode_open" in only_tools and "vscode_open" in by_name and not semantics.VSCODE.search(query or ""):
        import device_actions
        m_app = device_actions.OPEN_APP_ASK.match(query or "")
        if m_app:
            tgt = m_app.group(1).strip()
            if device_actions.is_standalone_app_target(tgt):
                s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "vscode_open", "arguments": {"target": tgt}}
    elif (
        "read_file" in only_tools
        and "read_file" in by_name
        and "write_file" not in only_tools
        and getattr(engine, "turn_paths", None)
        and getattr(engine, "perms", None) is not None
        and engine.perms.get().get("files") == "allow"
    ):
        cand_p = engine._path(engine.turn_paths[0])
        if cand_p.is_file() and not paths.read_blocked(base_p, cand_p):
            s3_pre_call = {
                "id": "call_" + uuid.uuid4().hex[:16],
                "name": "read_file",
                "arguments": {"path": str(cand_p), "start_line": 1, "end_line": 120},
            }
    return s3_pre_call, s3_extra_calls


def execute_s3_pre_calls(engine, s3_pre_call, s3_extra_calls, *, ask=None, allow=(), emit=None):
    """Execute planned S3 pre-calls and emit actual tool results. Returns `(rounds, done_calls, pre_text)`."""
    emit = emit or (lambda e: None)
    all_pre = [s3_pre_call] + list(s3_extra_calls or ())
    emit({"type": "round", "round": 1})
    emit({"type": "plan", "round": 1, "stage": "planning", "tools": [c["name"] for c in all_pre]})
    pre_summaries = []
    rounds = []
    done_calls = {}
    media_dir = Path(getattr(engine, "media_dir", None) or (Path(getattr(engine, "base_dir", Path.cwd())) / "media"))
    media_dir.mkdir(parents=True, exist_ok=True)
    for pc in all_pre:
        emit({"type": "plan", "round": 1, "stage": "preparing_tool", "tool": pc["name"]})
        emit({"type": "tool_call", "id": pc["id"], "name": pc["name"], "arguments": pc["arguments"]})
        key = (pc["name"], json.dumps(pc["arguments"], sort_keys=True))
        ran, result, why = engine.execute_tool(pc["name"], pc["arguments"], ask, allow)
        done_calls[key] = result
        worked, summary = N.tool_worked(pc["name"], ran, result, why)
        emit({
            "type": "tool_result",
            "id": pc["id"],
            "name": pc["name"],
            "ran": ran,
            "worked": worked,
            "summary": summary,
            "why": why,
            "result": result[:3000],
        })
        if pc["name"] == "generate_image" and not worked:
            raise RuntimeError(f"Image generation failed: {result}")
        rounds.append({
            "tool": pc["name"],
            "arguments": pc["arguments"],
            "ran": ran,
            "worked": worked,
            "summary": summary,
            "why": why,
            "result": result[:2000],
        })
        note_extra = f"\n({pc['note']})" if pc.get("note") else ""
        pre_summaries.append(f"Live {pc['name']} results:\n{result}{note_extra}")
    pre_text = (
        "Live tool results for this request (already executed in S3 — "
        "answer directly from these real results in plain text without calling tools):\n"
        + "\n\n".join(pre_summaries)
    )
    return rounds, done_calls, pre_text


def speakable(text, limit=1200):
    """Answer text for speech: no code blocks, links or markdown marks."""
    t = re.sub(r"```[\s\S]*?```", " (code shown on screen) ", text or "")
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def speech_parts(text, max_chars=240, first_chars=80, grow=1.25, limit=4000):
    """Speakable text in sentence groups; the next part is synthesized while the current one plays. The first part is
    short so speech starts soon (a 237-character part took 14.6 s before any sound, with the instance start), and each
    part is at most `grow` x the previous one: the CPU speaks 1.2-1.6x faster than real time, so a part longer than
    that is not ready when the previous one ends (45 then 214 characters: 3.9 s of sound, then 9.7 s to make the next)."""
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?;:])\s+", speakable(text, limit)):
        while True:
            cap = min(max_chars, max(first_chars, int(grow * len(parts[-1])))) if parts else first_chars
            if cur and len(cur) + 1 + len(s) > cap:
                parts.append(cur)
                cur = ""
                continue
            if len(s) <= cap:
                cur = (cur + " " + s).strip()
                break
            # a sentence longer than the part: cut at a late comma, else at the last space before the limit
            cut = s.rfind(", ", 0, cap)
            if cut < cap // 2:
                cut = s.rfind(" ", 0, cap)
            cut = cut if cut > cap // 3 else cap
            parts.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
            if not s:
                break
    return [p for p in parts + [cur] if p]


class SideGate:
    """Held while a turn or speech needs the hardware; a side completion (an exchange note) stops when it is held.
    Works as chat_stream's `cancel`."""

    def __init__(self):
        self.n, self.lock = 0, threading.Lock()

    def is_set(self):
        return self.n > 0

    def hold(self):
        with self.lock:
            self.n += 1

    def release(self):
        with self.lock:
            self.n = max(0, self.n - 1)


class OmniNative(N.LFM3Native):
    def __init__(self, base_dir, perms, max_tokens=32768, progress=None, engine="gpu", quant="q4_0",
                 voice_threads=VOICE_THREADS, gpu_name="GPU", threads=None):
        """max_tokens (context), voice_threads, gpu_name and (on a phone) threads come from this machine (hardware.choose)."""
        import shutil
        t0 = time.perf_counter()
        progress = progress or (lambda m: None)
        base_dir = Path(base_dir)
        quant = quant if quant in QUANTS else "q4_0"
        self.base_dir, self.perms, self.quant = base_dir, perms, quant
        self.voice_threads = int(voice_threads)
        model_paths, missing = omni_files(base_dir, quant)
        if missing:
            raise FileNotFoundError("missing " + ", ".join(missing) + " (run Clover AI Studio setup again)")
        self.media_dir = base_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        ff_txt = base_dir / "tools" / "ffmpeg" / "BIN.txt"
        self.ffmpeg_dir = ff_txt.read_text(encoding="utf-8").strip() if ff_txt.is_file() else ""
        if not self.ffmpeg_dir:
            sys_ff = shutil.which("ffmpeg")
            if sys_ff:
                self.ffmpeg_dir = str(Path(sys_ff).resolve().parent)
        self.has_native_video = bool(self.ffmpeg_dir)
        extra = ["--reasoning", "off", "--media-path", str(self.media_dir), "--video-fps", "2"]
        if ANDROID:
            # Adreno's OpenCL backend lacks ops of the projector's graph: a picture ended llama-server (S23, b12012); on
            # the CPU a picture took 8 s and a voice clip 3 s, and 0.74 GB less GPU memory stays pinned
            extra += ["--no-mmproj-offload"]
        if self.ffmpeg_dir:
            extra += ["--video-ffmpeg-dir", self.ffmpeg_dir]
        kind = "vulkan" if engine == "gpu" else "cpu"
        q_label = "BF16 16-bit" if quant == "bf16" else ("8-bit Q8_0" if quant == "q8_0" else f"4-bit {quant.upper()}")
        self.engine_kind, self.max_tokens, self.cache = engine, min(32768, int(max_tokens or 32768)), None
        self._paths = model_paths
        self._extra_args = extra
        self._threads = threads
        self._pre_rt = None
        if engine == "precomputed":
            import omni_precomputed
            progress("LFM3-Omni-2B Precomputed C1 Trunks + 4-Bit Experts (liblfm3_cpu_engine, O(1) mmap)")
            self._pre_rt = omni_precomputed.get_runtime(base_dir, threads=threads)
            self.separate_voice = not ANDROID
            self.gpu = None
            self.inputs = ["text", "vision", "audio"] + (["video"] if (self.has_native_video or ANDROID) else [])
        else:
            progress("LFM3-Omni-2B on the " + (gpu_name if engine == "gpu" else "CPU") + f" (llama.cpp, {q_label})")
            # speech on its own CPU instance while chat has the laptop's GPU; a phone has no memory for a second copy
            self.separate_voice = engine == "gpu" and not ANDROID
            ubatch_sz = 256
            self.gpu = L.LlamaServer(base_dir, kind, self.max_tokens, ubatch=ubatch_sz, model=model_paths["model"],
                                     mmproj=model_paths["mmproj"], extra_args=extra, name=f"omni-{kind}", timeout_s=300,
                                     threads=threads, extra_path=self.ffmpeg_dir or None)
            props = self.gpu.props()
            mods = props.get("modalities") or {}
            self.inputs = (["text"] + [k for k in ("vision", "audio") if mods.get(k)]
                           + (["video"] if (self.has_native_video or ANDROID) and mods.get("vision") else []))
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self._voice, self._voice_busy, self._voice_used, self._voice_error = None, 0, 0.0, None
        self._voice_lock = threading.Lock()
        self.side_gate = SideGate()
        self._py = None
        self._web = None
        self._extra = {}
        self.workdir, self.workname, self.after_write = None, "", None  # the active workspace (server sets them)
        self.c1_inference_only = False  # the server sets it per turn (setting c1_inference_only)
        self._c1_srv = None
        self.setup_s = round(time.perf_counter() - t0, 2)

    def _ensure_mm_server(self):
        """Lazily start the multimodal llama-server (mmproj + LiquidCodec) when Precomputed C1 receives media or speech."""
        if self.gpu is None:
            self.gpu = L.LlamaServer(self.base_dir, "cpu", self.max_tokens, ubatch=256, model=self._paths["model"],
                                     mmproj=self._paths["mmproj"], extra_args=self._extra_args, name="omni-mm-cpu",
                                     timeout_s=300, threads=self._threads, extra_path=self.ffmpeg_dir or None)
        return self.gpu

    def describe(self):
        mm_tag = "BF16" if "BF16" in self._paths["mmproj"].name.upper() else "Q8_0"
        if self.engine_kind == "precomputed":
            return {"mode": "omni", "model": "LFM3-Omni-2B", "engine": "precomputed", "quant": "soc_q4",
                    "max_tokens": self.max_tokens, "setup_s": self.setup_s,
                    "load_ms": getattr(self._pre_rt, "load_ms", None),
                    "tools": [t["function"]["name"] for t in N.STUDIO_TOOLS],
                    "layers": [
                        "C1 Precomputed LFM3-Omni-2B (trunk-0..23 + 4-Bit SwiGLU Experts + Two-Stage LM Head, O(1) mmap)",
                        "C2 HybridAgent (Disabled)",
                        "C3 11-Level NCM (Disabled)",
                        "S1 Memory & Records · S2 Recall & Ontology · S3 Tools & Compiler · CSI & PPL",
                    ],
                    "device": "CPU (Precomputed C1 · AVX2/AVX-VNNI / ARMv9 NEON)",
                    "weights": f"Precomputed 4-bit Group-64 Trunks + Experts + on-demand projector {mm_tag}",
                    "backend": "liblfm3_cpu_engine (precomputed static trunks + experts)",
                    "inputs": self.inputs, "outputs": ["text", "speech"],
                    "speech_languages": list(SPEECH_LANGS),
                    "speech": "on-demand 24 kHz LiquidCodec neural speech",
                    "speech_error": self._voice_error,
                    "reasoning": False}
        w_tag = ("16-bit (BF16)" if self.quant == "bf16"
                 else ("8-bit (Q8_0)" if self.quant == "q8_0" else f"4-bit ({self.quant.upper()})"))
        return {"mode": "omni", "model": "LFM3-Omni-2B", "engine": self.engine_kind, "quant": self.quant,
                "max_tokens": self.max_tokens, "setup_s": self.setup_s,
                "tools": [t["function"]["name"] for t in N.STUDIO_TOOLS],
                "layers": [
                    f"C1 LFM3-Omni-2B (llama.cpp {self.gpu.kind}, {self.quant.upper()} · Full Omni: Text, Vision, Audio, Video+Soundtrack & Speech)",
                    "C2 HybridAgent (Disabled)",
                    "C3 11-Level NCM (Disabled)",
                    "S1 Memory & Records · S2 Recall & Ontology · S3 Tools & Compiler · CSI & PPL",
                ],
                "device": self.gpu.device, "weights": f"{w_tag} + projector {mm_tag}",
                "backend": f"llama.cpp {self.gpu.kind} (direct LFM3-Omni-2B end-to-end)",
                "inputs": self.inputs, "outputs": ["text", "speech"],
                "speech_languages": list(SPEECH_LANGS),
                "speech": (f"CPU, own llama.cpp instance ({self.voice_threads} threads; "
                           + ("running)" if self._voice is not None else "starts when Clover first speaks)")
                           if self.separate_voice else "same instance as chat"),
                "speech_error": self._voice_error,
                "reasoning": False}

    def system_prompt(self, tools_on=True, asked=None, *, media=(), has_typed_text=True, generation_mode="text"):
        """Official LiquidAI/LFM3-Omni-2B modality system prompts (README.md) with zero fake capability claims."""
        kinds = {m.get("kind") for m in (media or ()) if isinstance(m, dict)}
        out_cue = ("Respond with interleaved text and audio."
                   if generation_mode == "interleaved" else "Respond with text.")
        if "video" in kinds:
            base = f"Answer questions about the video using what you see and hear. {out_cue}"
        elif "image" in kinds and "audio" in kinds and not has_typed_text:
            base = f"Answer questions about the provided image. Base every answer only on what can be seen. {out_cue}"
        elif "image" in kinds:
            base = f"Answer questions about the provided image. Base every answer only on what can be seen. {out_cue}"
        elif "audio" in kinds and has_typed_text:
            base = f"Answer questions about the provided audio clip. Base every answer only on what can be heard. {out_cue}"
        else:
            base = out_cue
        ws_txt = self.workspace_sentence().strip()
        parts = [p for p in (ws_txt, (N.tools_sentence(True, asked) if tools_on else ""), base) if p]
        return " ".join(parts)

    def transcribe(self, media, lang=None):
        """Official LFM3-Omni-2B ASR (`Perform ASR in <language>.` / `Perform ASR.`)."""
        parts = [media_part(m) for m in media or () if m["kind"] == "audio" and (self.media_dir / m["file"]).is_file()]
        if not parts:
            return ""
        lang_names = {
            "en": "English", "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "fi": "Finnish",
            "fr": "French", "he": "Hebrew", "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch",
            "pl": "Polish", "pt": "Portuguese", "es": "Spanish", "sv": "Swedish", "tr": "Turkish", "zh": "Chinese",
        }
        lname = lang_names.get(str(lang or "").lower())
        sys_asr = f"Perform ASR in {lname}." if lname else "Perform ASR."
        msgs = [{"role": "system", "content": sys_asr}, {"role": "user", "content": parts}]
        with self.lock:
            try:
                return self._chat(msgs, [], 1024, {})[0].strip()
            finally:
                self.stage = {"name": "idle", "since": time.time()}

    def tts(self, text, voice="female", lang="auto"):
        t = speakable(text)
        if not t:
            raise ValueError("nothing to speak")
        eff_lang = detect_speech_lang(t, preferred=lang)
        self.side_gate.hold()
        try:
            srv = self._voice_server() if self.separate_voice else None
            if srv is not None:
                try:
                    return srv.tts(t, voice=voice, lang=eff_lang)
                finally:
                    self._voice_done()
            with self.lock:  # one llama-server slot: speech waits for a running answer
                self.stage = {"name": "speaking", "since": time.time()}
                try:
                    return self._ensure_mm_server().tts(t, voice=voice, lang=eff_lang)
                finally:
                    self.stage = {"name": "idle", "since": time.time()}
        finally:
            self.side_gate.release()

    def side_complete(self, msgs, max_new):
        """A short greedy completion (an exchange note) on the CPU instance, never on the chat's slot, so the chat keeps
        its cached prompt. -> text, or None when there is no CPU instance (CPU engine) or a turn or speech started."""
        if not self.separate_voice or self.side_gate.is_set() or self._voice is None:
            return None
        srv = self._voice_server()
        if srv is None:
            return None
        try:
            text = ""
            for ev in srv.chat_stream(dict(GREEDY, messages=msgs, max_tokens=int(max_new), cache_prompt=False),
                                      self.side_gate):
                if ev.get("cancelled"):
                    return None
                for ch in ev.get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
            return text
        finally:
            self._voice_done()

    def warm_voice(self):
        """Start the speech instance in the background (a reply that will be spoken is being written)."""
        def run():
            if self._voice_server() is not None:
                self._voice_done()
        if self.separate_voice and self._voice is None:
            threading.Thread(target=run, daemon=True).start()

    def _voice_done(self):
        with self._voice_lock:
            self._voice_busy -= 1
            self._voice_used = time.time()

    def _voice_server(self):
        """The CPU speech instance (started here on first use), marked busy; None -> speak on the main instance."""
        with self._voice_lock:
            if self._voice is not None and self._voice.proc.poll() is not None:
                self._voice = None
            if self._voice is None:
                try:
                    self._voice = L.LlamaServer(self.base_dir, "cpu", 8192, model=self._paths["model"],
                                                mmproj=self._paths["mmproj"], name="omni-voice", timeout_s=120,
                                                extra_args=["--reasoning", "off", "-t", str(self.voice_threads)])
                    self._voice_error = None
                    threading.Thread(target=self._voice_reaper, daemon=True).start()
                except (OSError, RuntimeError, TimeoutError) as exc:
                    self._voice_error = str(exc)
                    return None
            self._voice_busy += 1
            return self._voice

    def _voice_reaper(self):
        while True:
            time.sleep(30)
            with self._voice_lock:
                if self._voice is None:
                    return
                if not self._voice_busy and time.time() - self._voice_used > VOICE_IDLE_S:
                    self._voice.close()
                    self._voice = None
                    return

    def free_voice(self):
        """Stops the voice instance when it is idle (it starts again on the next Listen): ~1.6 GB for a picture."""
        with self._voice_lock:
            if self._voice is None or self._voice_busy:
                return False
            self._voice.close()
            self._voice = None
            return True

    def close(self):
        with self._voice_lock:
            if self._voice is not None:
                self._voice.close()
                self._voice = None
        if getattr(self, "_c1_srv", None) is not None:
            self._c1_srv.close()
            self._c1_srv = None
        super().close()

    # ---------------- generation ----------------
    def _chat(self, msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, on_plan=None):
        """One C1 inference pass -> (content, calls, stats). Uses Precomputed C1 (`liblfm3_cpu_engine`) when enabled."""
        eff_max = min(int(max_new), 512) if tool_choice == "required" else int(max_new)
        if self._pre_rt is not None and tool_choice != "required":
            import omni_precomputed
            if not omni_precomputed.has_multimodal_parts(msgs):
                prompt_txt = omni_precomputed.format_omni_prompt(msgs, schemas)
                prompt_ids = self._pre_rt.tokenizer.encode(prompt_txt)
                if len(prompt_ids) <= omni_precomputed.MAX_NATIVE_PROMPT_TOKENS:
                    self.stage = {"name": "C1 precomputed inference", "since": time.time()}
                    return self._pre_rt.stream_chat(
                        prompt_ids, eff_max, on_delta=on_delta, cancel=cancel, by_name=by_name,
                        looping_fn=looping, trim_loop_fn=trim_loop, rambling_fn=rambling,
                    )
        body = dict(GREEDY, messages=msgs, max_tokens=eff_max, seed=int(seed), cache_prompt=True)
        if schemas:
            body["tools"] = schemas
            if tool_choice:
                body["tool_choice"] = tool_choice
        content, acc, finish, tm = "", {}, None, {}
        planned_tools = set()
        t0 = time.perf_counter()
        t_first = None
        looped, checked, cut = False, 0, None
        args_checked, args_looped = 0, False
        self.stage = {"name": "planning" if tool_choice else "C1 inference", "since": time.time()}
        stream = self._ensure_mm_server().chat_stream(body, cancel)
        try:
            for ev in stream:
                if ev.get("cancelled"):
                    finish = "cancelled"
                    break
                tm = ev.get("timings") or tm
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("tool_calls"):
                        if t_first is None:
                            t_first = time.perf_counter()
                            self.stage = {"name": "planning tool call" if (tool_choice or d.get("tool_calls")) else "C1 inference",
                                          "since": time.time()}
                    if d.get("content"):
                        content += d["content"]
                        if "<function=" in content and on_plan:
                            for fm in re.finditer(r"<function=([^>\s]+)>", content):
                                tname = fm.group(1)
                                if tname not in planned_tools:
                                    planned_tools.add(tname)
                                    self.stage = {"name": f"preparing {tname}", "since": time.time()}
                                    on_plan(tname)
                        if on_delta:
                            on_delta(d["content"], content)
                        if "</tool_call>" in content:
                            finish = "tool_calls"
                            break
                        if len(content) - checked >= 200:
                            checked = len(content)
                            looped = looping(content)
                            cut = rambling(content)
                    for tc in d.get("tool_calls") or []:
                        slot = acc.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                        if slot["name"] and slot["name"] in by_name and slot["name"] not in planned_tools:
                            planned_tools.add(slot["name"])
                            self.stage = {"name": f"preparing {slot['name']}", "since": time.time()}
                            if on_plan:
                                on_plan(slot["name"])
                        # code that printed one line 60 times ran each round to 4,096 tokens (330 s, 2026-11-11)
                        if len(slot["arguments"]) - args_checked >= 200:
                            args_checked = len(slot["arguments"])
                            args_looped = args_looped or looping(slot["arguments"])
                    finish = ch.get("finish_reason") or finish
                if finish == "tool_calls" and "</tool_call>" in content:
                    break
                if args_looped:  # the call's text repeats: closing the stream stops llama-server
                    finish = "stop"
                    break
                if looped:  # closing the stream stops llama-server; one copy of the repeated text is kept
                    content, finish = trim_loop(content), "stop"
                    break
                if cut is not None:
                    content, finish = content[:cut], "stop"
                    break
        finally:
            stream.close()
        calls = []
        for s in acc.values():
            try:
                args = json.loads(s["arguments"] or "{}")
            except ValueError:
                args = {"_raw": s["arguments"]}
            calls.append({"id": s["id"] or "call_" + uuid.uuid4().hex[:16], "name": s["name"], "arguments": args})
        if not calls and "<tool_call>" in content:  # template parser did not split the call out: parse it here
            calls = N.parse_tool_calls(content, by_name)
            content = TOOL_XML.sub("", content)
        t_end = time.perf_counter()
        reason = {"stop": "stop", "tool_calls": "stop", "length": "length", "cancelled": "cancelled"}.get(finish, "stop")
        if cancel is not None and cancel.is_set():
            reason = "cancelled"
        prompt_ms, pred_ms = tm.get("prompt_ms"), tm.get("predicted_ms")
        prefill_s = (prompt_ms / 1000) if prompt_ms is not None else ((t_first or t_end) - t0)
        decode_s = (pred_ms / 1000) if pred_ms is not None else (t_end - (t_first or t_end))
        n_new = int(tm.get("predicted_n") or 0)
        return content, calls, {"prompt_tokens": int(tm.get("prompt_n") or 0) + int(tm.get("cache_n") or 0),
                                "reused_tokens": int(tm.get("cache_n") or 0), "new_tokens": n_new,
                                "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
                                "decode_tok_s": round(n_new / decode_s, 2) if decode_s > 0 and n_new else None,
                                "finish": reason, **({"looped": True} if looped else {}),
                                **({"tool_args_looped": True} if args_looped else {}),
                                **({"rambling": True} if cut is not None else {})}

    def _history(self, history, keep_last_media=False, strip_attached=False, current_query=None, is_follow_up=False):
        out = []
        last_media_idx = -1
        if keep_last_media:
            for idx in range(len(history) - 1, -1, -1):
                if history[idx].get("role") == "user" and history[idx].get("media"):
                    last_media_idx = idx
                    break
        for idx, m in enumerate(history):
            if idx == last_media_idx and m.get("role") == "user" and m.get("media"):
                out.append({"role": "user", "content": user_content(
                    m.get("content") or "", m["media"], self.media_dir,
                    native_video=getattr(self, "has_native_video", True),
                )})
            elif m.get("role") == "user" and m.get("media"):
                names = ", ".join(str(x.get("name") or x.get("kind") or "attachment") for x in m["media"] if isinstance(x, dict))
                txt = str(m.get("content") or "").strip()
                out.append({"role": "user", "content": (f"[Attached: {names}]\n{txt}" if names else txt).strip()})
            else:
                item = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
                if strip_attached and item.get("role") == "user" and isinstance(item.get("content"), str):
                    marker = "\n\nUsing the attached content / session context above, answer the user's request clearly and thoroughly:\n"
                    if marker in item["content"]:
                        item["content"] = "[Earlier file attachment omitted]\n" + item["content"].split(marker, 1)[1]
                out.append(item)
        return out

    def _extract_pcm16_16k(self, media) -> bytes | None:
        """Read 16 kHz mono PCM16 frames from the first attached WAV audio clip for `/v1/realtime`."""
        import wave
        import numpy as np
        for m in media or ():
            if not isinstance(m, dict) or m.get("kind") != "audio":
                continue
            p = self.media_dir / str(m.get("file") or "")
            if not p.is_file() or p.suffix.lower() != ".wav":
                continue
            try:
                with wave.open(str(p), "rb") as wf:
                    ch, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
                    raw = wf.readframes(nf)
                if sw != 2 or not raw:
                    continue
                s16 = np.frombuffer(raw, dtype="<i2")
                if ch > 1:
                    s16 = (s16[: (s16.size // ch) * ch].reshape(-1, ch).astype(np.int32).mean(axis=1)).astype("<i2")
                if sr != 16000 and s16.size > 1:
                    out_len = max(1, int(round(s16.size * 16000.0 / float(sr))))
                    x_old = np.linspace(0.0, 1.0, num=s16.size, endpoint=False)
                    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
                    s16 = np.clip(np.round(np.interp(x_new, x_old, s16.astype(np.float32))), -32768, 32767).astype("<i2")
                return s16.tobytes()
            except Exception:
                continue
        return None

    # ---------------- one Studio turn ----------------
    def run_turn(self, history, user_text, *, tools=True, sampling="general", thinking=True, max_new=2048,
                 max_rounds=8, on_event=None, ask=None, allow=(), cancel=None, seed=0, system=None,
                 layers=(), kit=None, layer_query=None, final_on_limit=False, force_answer=False, extra_tools=(),
                 media=(), only_tools=None, meaning=None, generation_mode="text"):
        """Direct end-to-end execution of Official `LiquidAI/LFM3-Omni-2B` across text, vision, audio, video+soundtrack,
        native tool calling, and single-pass `<|mixed_start|>` interleaved text + 24 kHz speech."""
        emit = on_event or (lambda e: None)
        gen_mode = generation_mode if generation_mode in ("text", "interleaved", "audio") else "text"
        self.turn_cancel = cancel
        self.turn_paths = list(meaning.paths) if meaning is not None else N.user_paths(user_text)
        self.turn_hosts = N.typed_hosts(meaning.text if meaning is not None else user_text)
        self.turn_unknown = set()
        self.turn_ran = set()
        self.turn_web, self.turn_web_cap = 0, (meaning.web_cap if meaning is not None else
                                               8 if "web_search" in N.asked_tools(user_text) else 3)
        base = [t for t in N.STUDIO_TOOLS if only_tools is None or t["function"]["name"] in only_tools]
        extra_tools = [t for t in extra_tools if only_tools is None or t["schema"]["function"]["name"] in only_tools]
        tools = tools and bool(base or extra_tools)
        self._extra = {t["schema"]["function"]["name"]: t for t in extra_tools} if tools else {}
        schemas = (base + [t["schema"] for t in extra_tools]) if tools else []
        by_name = {t["function"]["name"]: t["function"] for t in schemas}
        query = layer_query or user_text
        m_c1 = getattr(self, "display_model_name", None) or (
            self.describe().get("model") if hasattr(self, "describe") and hasattr(self, "engine_kind") else None
        ) or "LFM3-Omni-2B"
        l_c1 = getattr(self, "display_layers_desc", None) or ""
        import workspaces as WS
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name=m_c1,
            query=query,
            offered_tools=only_tools if only_tools is not None else list(by_name)[:4],
        )
        self.active_op_plan = op_plan
        op_plan.start_stage("context")
        layer_log, blocks = [], []
        for name, fn in (("memory", "memory_context"), ("workspace", "workspace_context"), ("records", "records_context")):
            if name in layers and kit is not None:
                block, info = getattr(kit, fn)(query)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
                    emit(dict(info, type="layer"))
        ws_active_p = (kit.workspaces.active() or {}).get("path") if (kit is not None and getattr(kit, "workspaces", None) is not None) else None
        l_blk, l_info = WS.local_context_for_paths(self.turn_paths, workdir=self.workdir, active_ws_path=ws_active_p)
        if l_blk:
            blocks.append(l_blk)
            layer_log.append(l_info)
            emit(dict(l_info, type="layer"))
        recall = []
        if "recall" in layers and kit is not None:
            block, info = kit.recall_context(query, sum(1 for m in history if m.get("role") == "user"))
            if block:
                recall.append({"role": "system", "content": block})
            if info:
                layer_log.append(info)
                if info.get("refs"):
                    emit(dict(info, type="layer"))
        if blocks:
            recall.append({"role": "system", "content": "\n\n".join(blocks)})

        has_typed_text = bool((user_text or "").strip())
        sys_txt = system or self.system_prompt(
            bool(schemas), only_tools, media=media, has_typed_text=has_typed_text, generation_mode=gen_mode
        )
        keep_last_media = bool(not media and any(m.get("media") for m in (history or ()) if m.get("role") == "user"))
        past_msgs = self._history(
            history,
            keep_last_media=keep_last_media,
            strip_attached=bool(media),
            current_query=query,
            is_follow_up=True,
        )
        native_vid = getattr(self, "has_native_video", True)
        msgs = ([{"role": "system", "content": sys_txt}] + past_msgs + recall
                + [{"role": "user", "content": user_content(user_text, media, self.media_dir, native_video=native_vid)}])
        rounds, final, stats_all = [], "", []
        reply_audio_url = None
        done_calls = {}
        stream_state = {"emitted": False}

        def delta_cb(round_no, kind="answer"):
            shown = {"n": 0}
            stream_state["emitted"] = False

            def on_delta(delta, text):
                cleaned = THINK_PREFIX.sub("", text)
                m_tc = TOOL_OPEN_RE.search(cleaned)
                visible = cleaned[:m_tc.start()] if m_tc else TOOL_XML.sub("", cleaned)
                last_lt = visible.rfind("<")
                if last_lt != -1 and len(visible) - last_lt < 12:
                    tail = visible[last_lt:]
                    if "<tool_call>".startswith(tail) or "<function=".startswith(tail):
                        visible = visible[:last_lt]
                if len(visible) > shown["n"]:
                    emit({"type": "token", "round": round_no, "kind": kind, "text": visible[shown["n"]:]})
                    shown["n"] = len(visible)
                    stream_state["emitted"] = True
            return on_delta

        with self.lock:
            try:
                op_plan.complete_stage("context", "Context ready")
                op_plan.start_stage("direction", f"LFM3-Omni-2B ({gen_mode})")
                op_plan.complete_stage("direction")
                op_plan.start_stage("execution")

                has_vis_or_vid = any(m.get("kind") in ("image", "video") for m in (media or ()) if isinstance(m, dict))
                if gen_mode in ("interleaved", "audio") and not has_vis_or_vid and not schemas:
                    emit({"type": "round", "round": 1})
                    emit({"type": "plan", "round": 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
                    self.stage = {"name": "interleaved text + speech", "since": time.time()}
                    pcm_in = self._extract_pcm16_16k(media)
                    out_mods = ("text", "audio") if gen_mode == "interleaved" else ("audio",)
                    cb_txt = delta_cb(1, "answer") if gen_mode == "interleaved" else None

                    def _on_aud(b64_pcm, _raw):
                        emit({"type": "audio_delta", "pcm16_b64": b64_pcm, "rate": 24000})

                    turn_txt = str(user_text or "").strip()
                    if past_msgs and turn_txt:
                        ctx_lines = []
                        for pm in past_msgs[-6:]:
                            r_lbl = "User" if pm.get("role") == "user" else "Assistant"
                            c_str = self._plain_text(pm.get("content")).strip()
                            if c_str:
                                ctx_lines.append(f"{r_lbl}: {c_str}")
                        if ctx_lines:
                            turn_txt = "\n".join(ctx_lines) + f"\nUser: {turn_txt}"

                    final_rt, wav_rt, st_rt = self._ensure_mm_server().realtime_turn(
                        text=turn_txt,
                        pcm16_16k=pcm_in,
                        instructions=sys_txt,
                        output_modalities=out_mods,
                        max_tokens=min(1024, int(max_new)),
                        on_text_delta=cb_txt,
                        on_audio_delta=_on_aud,
                        cancel=cancel,
                    )
                    final = THINK_PREFIX.sub("", final_rt).strip()
                    stats_all.append(st_rt)
                    if wav_rt:
                        import hashlib
                        af = f"omni_speech_{hashlib.sha256(wav_rt).hexdigest()[:24]}.wav"
                        (self.media_dir / af).write_bytes(wav_rt)
                        reply_audio_url = "/api/media/file/" + af
                    if not final and gen_mode == "audio":
                        final = "(Spoken audio response)"
                    op_plan.complete_stage("execution")
                    op_plan.complete_stage("outcome", "Completed")
                    op_plan.complete_stage("growth", "Completed")
                    if final and not stream_state["emitted"]:
                        emit({"type": "token", "round": 1, "kind": "answer", "text": final})
                    res_out = {
                        "reply": final, "tool_calls": rounds, "rounds": 1, "generation": stats_all,
                        "reasoning": "", "cancelled": bool(cancel and cancel.is_set()), "layers": layer_log,
                        "todo_list": op_plan.todos,
                    }
                    if reply_audio_url:
                        res_out["reply_audio"] = reply_audio_url
                    return res_out

                for r in range(max_rounds):
                    emit({"type": "round", "round": r + 1})
                    if schemas and getattr(self, "turn_web", 0) >= getattr(self, "turn_web_cap", 99):
                        schemas = []
                    if schemas and sum(1 for rd in rounds if rd["why"] == "same call already made in this answer") >= 2:
                        schemas = []
                    emit({"type": "plan", "round": r + 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
                    chat_kw = ({"on_plan": lambda tname, rn=r + 1: emit({"type": "plan", "round": rn, "stage": "preparing_tool", "tool": tname})}
                               if schemas else {})
                    stream_cb = None if "ncm" in layers else delta_cb(r + 1, "answer")
                    content, calls, st = self._chat(
                        msgs, schemas, max_new, by_name,
                        stream_cb, cancel,
                        seed + r, None,
                        **chat_kw,
                    )
                    stats_all.append(st)
                    if calls and tools and schemas and st["finish"] != "cancelled":
                        msgs.append({"role": "assistant", "content": content, "tool_calls": [
                            {"id": c["id"], "type": "function",
                             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]})
                        for c in calls:
                            emit({"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": c["arguments"]})
                            key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
                            if isinstance(c["arguments"], dict) and "_raw" in c["arguments"]:
                                ran, why = False, "the call's arguments were not complete JSON"
                                result = "Not run: incomplete JSON arguments. Write the call again concisely."
                            elif key in done_calls:
                                ran, why = False, "same call already made in this answer"
                                result = "This exact call already ran above; use the result already returned."
                            else:
                                ran, result, why = self.execute_tool(c["name"], c["arguments"], ask, allow)
                                done_calls[key] = result
                            worked, summary = N.tool_worked(c["name"], ran, result, why)
                            summary = N.checked_summary(self, ran, summary)
                            emit({"type": "tool_result", "id": c["id"], "name": c["name"], "ran": ran, "worked": worked,
                                  "summary": summary, "why": why, "result": result[:3000]})
                            rounds.append({"tool": c["name"], "arguments": c["arguments"], "ran": ran, "worked": worked,
                                           "summary": summary, "why": why, "result": result[:2000]})
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                            op_plan.advance_execution_step(c["name"])
                        continue
                    final = TOOL_XML.sub("", content).strip()
                    if not final:
                        if st["finish"] == "cancelled":
                            final = "(Stopped.)"
                        else:
                            raise RuntimeError(f"The selected model returned no answer (finish: {st['finish']})")
                    break
                else:
                    if final_on_limit and not (cancel and cancel.is_set()):
                        msgs.append({"role": "user", "content": "Answer now from the tool results above without calling more tools."})
                        emit({"type": "round", "round": max_rounds + 1})
                        lim_cb = None if "ncm" in layers else delta_cb(max_rounds + 1, "answer")
                        content, _, st = self._chat(msgs, [], max_new, by_name, lim_cb, cancel, seed + max_rounds)
                        stats_all.append(st)
                        final = content.strip()
                    if not final:
                        final = "(Stopped.)" if (cancel and cancel.is_set()) else ""

                if gen_mode in ("interleaved", "audio") and final and not (cancel and cancel.is_set()):
                    try:
                        wav_b = self.tts(final, voice="female", lang="auto")
                        if wav_b:
                            import hashlib
                            af = f"omni_speech_{hashlib.sha256(wav_b).hexdigest()[:24]}.wav"
                            (self.media_dir / af).write_bytes(wav_b)
                            reply_audio_url = "/api/media/file/" + af
                    except Exception:
                        pass

                op_plan.complete_stage("execution")
                op_plan.complete_stage("outcome", "Completed")
                op_plan.complete_stage("growth", "Completed")
                rnd_n = len(stats_all) or 1
                if not (cancel and cancel.is_set()) and ("ncm" in layers or not stream_state["emitted"]):
                    emit({"type": "token", "round": rnd_n, "kind": "answer", "text": final})
            finally:
                self.stage = {"name": "idle", "since": time.time()}
        res_out = {"reply": final, "tool_calls": rounds, "rounds": len(stats_all), "generation": stats_all,
                   "reasoning": "", "cancelled": bool(cancel and cancel.is_set()), "layers": layer_log,
                   "todo_list": op_plan.todos}
        if reply_audio_url:
            res_out["reply_audio"] = reply_audio_url
        return res_out

    # ---------------- C1 inference only (setting c1_inference_only) ----------------
    def _c1_server(self):
        """The same model as an inference-only llama-server (hidden states, never a sampled token), started on first use."""
        srv = getattr(self, "_c1_srv", None)
        if srv is not None and (getattr(srv, "proc", None) is None or srv.proc.poll() is None):
            return srv
        model_p = (getattr(self, "_paths", None) or {}).get("model") or getattr(getattr(self, "gpu", None), "model", None)
        gpu_kind = getattr(getattr(self, "gpu", None), "kind", "cpu")
        if model_p and Path(model_p).is_file() and gpu_kind in ("vulkan", "cpu"):
            ctx = min(int(getattr(self, "max_tokens", C1_INFER_CTX)), C1_INFER_CTX)
            self._c1_srv = L.LlamaServer(self.base_dir, gpu_kind, ctx, ubatch=ctx, model=model_p,
                                         extra_args=["-b", str(ctx)], name=f"omni-c1-inference-{gpu_kind}",
                                         timeout_s=300, threads=getattr(self, "_threads", None), embeddings=True)
        else:
            raise RuntimeError("C1 inference is unavailable for the selected model: a running inference provider or installed GGUF is required")
        return self._c1_srv

    @staticmethod
    def _plain_text(content):
        if isinstance(content, str):
            return content
        parts = []
        for p in content or ():
            if isinstance(p, dict):
                parts.append((p.get("text") or "") if p.get("type") == "text" else f"[{MEDIA_WORD.get(p.get('type'), 'attachment')}]")
        return "\n".join(x for x in parts if x)

    def _c1_prompt(self, srv, before, n_hist, turn_text):
        """-> (token ids of the chat template up to where this turn's message starts, turn text, what was trimmed). The
        oldest history goes first, then the start of a very long message, until C1's single request fits."""
        mark = "\u2063CLOVER-TURN\u2063"
        before = [dict(m, content=self._plain_text(m.get("content"))) for m in before]
        budget, trimmed = srv.ctx - 16, {}
        for _ in range(64):
            prefix = srv.tokenize(srv.apply_template(before + [{"role": "user", "content": mark}]).split(mark, 1)[0])
            n_turn = len(srv.tokenize(turn_text)) + turn_text.count("\n") + 1
            if len(prefix) + n_turn <= budget:
                return prefix, turn_text, trimmed
            if n_hist > 0:
                before, n_hist = before[:1] + before[2:], n_hist - 1
                trimmed["history_messages"] = trimmed.get("history_messages", 0) + 1
                continue
            keep = max(0.1, min(0.9, (budget - len(prefix)) / n_turn))
            cut = len(turn_text) - int(len(turn_text) * keep)
            turn_text = turn_text[cut:]
            trimmed["turn_chars_cut"] = trimmed.get("turn_chars_cut", 0) + cut
        raise RuntimeError("the conversation does not fit C1's inference request")

    def _c1_inference_turn(self, msgs, n_hist, pre_text, query, kit, layers, emit, cancel, op_plan, rounds, layer_log,
                           meaning, m_c1, l_c1, planned_tools):
        """C1 stops at inference: one forward pass of the Omni over the conversation, no token sampled. Its hidden states
        for this turn's message go to C2 (plan) and C3 (decode); C3's text is the reply, checked but not rewritten."""
        stopped = {"reply": "(Stopped.)", "tool_calls": rounds, "rounds": 0, "reasoning": "", "cancelled": True,
                   "layers": layer_log, "todo_list": op_plan.todos}
        turn_text = self._plain_text(msgs[-1].get("content"))
        before = msgs[:-1]
        if pre_text and pre_text not in turn_text:
            # S3's results are this turn's evidence: they join the message, without the header that tells C1 how to answer
            before = [m for m in before if m.get("content") != pre_text]
            turn_text = pre_text.split("\n", 1)[-1] + "\n\n" + turn_text
        emit({"type": "round", "round": 1})
        emit({"type": "plan", "round": 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
        self.stage = {"name": "C1 inference", "since": time.time()}
        t0 = time.perf_counter()
        srv = self._c1_server()
        prefix_ids, turn_text, trimmed = self._c1_prompt(srv, before, n_hist, turn_text)
        c1_state, c1_info = kit.c1_infer_states(turn_text, srv.hidden_states, prefix_ids=prefix_ids, model=m_c1,
                                                layers_desc=l_c1)
        if trimmed:
            c1_info["trimmed"] = trimmed
        gen = {"prompt_tokens": int(c1_info.get("tokens") or 0), "reused_tokens": 0, "new_tokens": 0,
               "prefill_s": round(time.perf_counter() - t0, 2), "decode_s": 0.0, "decode_tok_s": None, "finish": "stop",
               "c1": "inference only"}
        layer_log.append(c1_info)
        emit(dict(c1_info, type="layer"))
        emit({"type": "plan", "round": 1, "stage": "c1_done", "ms": c1_info.get("ms")})
        if cancel and cancel.is_set():
            return dict(stopped, generation=[gen])
        self.stage = {"name": "C2 planning", "since": time.time()}
        emit({"type": "plan", "round": 1, "stage": "c2_plan", "model": "HybridAgent"})
        c2_state, c2_info = kit.c2_plan_state(query, c1_state=c1_state, offered_tools=planned_tools)
        layer_log.append(c2_info)
        emit(dict(c2_info, type="layer"))
        emit({"type": "plan", "round": 1, "stage": "c2_done", "ms": c2_info.get("ms")})
        if cancel and cancel.is_set():
            return dict(stopped, generation=[gen])
        self.stage = {"name": "C3 decoding", "since": time.time()}
        emit({"type": "plan", "round": 1, "stage": "ncm_decode", "model": "11-Level NCM"})
        final, ncm_info = kit.c3_decode(c2_state, full_ncm=True)
        layer_log.append(ncm_info)
        emit(dict(ncm_info, type="layer"))
        emit({"type": "plan", "round": 1, "stage": "ncm_done", "ms": round((ncm_info.get("s") or 0) * 1000)})
        final = (final or "").strip()
        if not final:
            raise RuntimeError("C3 model returned no answer")
        op_plan.complete_stage("execution")
        op_plan.start_stage("outcome", "Checking the decoded answer…", cycle=1)
        findings, c_meta = C.verify_turn_outcome(query, final, rounds, kit=kit, layers=layers, meaning=meaning)
        if c_meta.get("compiler_info") is not None:
            layer_log.append(c_meta["compiler_info"])
        layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": not findings,
                          "findings": [f[:240] for f in findings[:3]]})
        op_plan.complete_stage("outcome", "Not confirmed (C1 inference only: nothing rewrites the answer)" if findings
                               else C.outcome_label(c_meta["checks"], 1))
        op_plan.start_stage("growth", "Recording what this turn showed…")
        if "ontology" in layers:
            final, conflict = kit.grounding_filter(query, final)
            layer_log.append({"layer": "ontology_grounding", "conflict": conflict, "rewritten": False})
        op_plan.complete_stage("growth", "Growth preserved (context updated)")
        emit({"type": "token", "round": 1, "kind": "answer", "text": final})
        return {"reply": final, "tool_calls": rounds, "rounds": 1, "generation": [gen], "reasoning": "",
                "cancelled": bool(cancel and cancel.is_set()), "layers": layer_log, "todo_list": op_plan.todos}

    # ---------------- the Clover cycle, when the user says the answer is not right ----------------
    def retry_turn(self, history, question, previous, feedback, *, media=(), layers=(), kit=None, on_event=None,
                   cancel=None, seed=0, max_new=2048, system=None, ask=None, allow=(), context=""):
        """clover_cycle.retry_cycle on this engine. history: the conversation, ending with `previous` (the answer the
        user says is not right); question: the message that answer is about, media: that message's attachments;
        feedback: what the user said; context: the exchange before a follow-up question (clover_cycle.context_before),
        put before the question in the independent solutions. Tools (web research, Python) go through the user's
        permissions (`ask` pauses for a decision). Returns the run_turn result shape."""
        if getattr(self, "c1_inference_only", False) and kit is not None and hasattr(kit, "c1_infer_states"):
            # C1 writes nothing in this mode: the user's message is read like any other turn (C1 inference -> C2 -> C3)
            return self.run_turn(history, feedback or question, tools=False, on_event=on_event, cancel=cancel, seed=seed,
                                 max_new=max_new, system=system, layers=layers, kit=kit, media=() if feedback else media)
        emit = on_event or (lambda e: None)
        active_model_name = getattr(self, "display_model_name", None) or (
            self.describe().get("model") if hasattr(self, "describe") else None
        ) or "LFM3-Omni-2B"
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name=active_model_name,
            query=f"{question}\n{feedback}",
            offered_tools=["run_python", "web_search"],
        )
        op_plan.start_stage("context", "Re-reading Context: original task, prior attempt & user feedback…")
        layer_log, blocks, stats_all, rounds, used = [], [], [], [], []
        for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"), ("workspace", "workspace_context"),
                         ("records", "records_context")):
            if name in layers and kit is not None:  # Context gathered again for the question
                block, info = getattr(kit, fn)(question)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
        op_plan.complete_stage("context", "Context established (original question, user correction & learned lessons)")
        op_plan.start_stage("direction", f"Direction set (Clover repair cycle · {active_model_name})")
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution", f"Executing Clover repair cycle with {active_model_name}…")
        sys_txt = system or self.system_prompt(False)
        convo = [{"role": "system", "content": sys_txt}] + self._history(history)
        start = len(convo)
        notes = web = ""

        def tool(name, args, run):
            cid = "call_" + uuid.uuid4().hex[:16]
            emit({"type": "tool_call", "id": cid, "name": name, "arguments": args})
            ran, result, why = run()
            emit({"type": "tool_result", "id": cid, "name": name, "ran": ran, "why": why, "result": result[:3000]})
            rounds.append({"tool": name, "arguments": args, "ran": ran, "why": why, "result": result[:2000]})
            return ran, result

        def rewrite(instruction):  # the model sees the conversation and each earlier attempt of this retry
            if cancel and cancel.is_set():
                return previous
            m_c1 = getattr(self, "display_model_name", None) or (
                self.describe().get("model") if hasattr(self, "describe") else None
            ) or "LFM3-Omni-2B"
            l_c1 = getattr(self, "display_layers_desc", None) or ""
            convo.append({"role": "user", "content": instruction + (notes + web if len(convo) == start else "")})
            emit({"type": "plan", "round": 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
            content, _, st = self._chat(convo, [], max_new, {}, None, cancel, seed + 500 + len(convo))
            stats_all.append(st)
            new = TOOL_XML.sub("", content).strip()
            if not new:
                raise RuntimeError("The selected model returned no answer during retry")
            convo.append({"role": "assistant", "content": new})
            return new

        def solve(k, note=None):  # the question alone: the earlier answer is not shown; web results reach solve 2 only
            if cancel and cancel.is_set():
                return ""
            ctx = notes + (web if k == 2 else "")
            m2 = [{"role": "system", "content": sys_txt},
                  {"role": "user", "content": user_content(C.second_solve_prompt(context + question + ctx, question, k, note),
                                                           media, self.media_dir,
                                                           native_video=getattr(self, "has_native_video", True))}]
            content, _, st = self._chat(m2, [], max_new, {}, None, cancel, seed + 900 + k)
            stats_all.append(st)
            return content

        def python_solve(note):
            """A third solution for a number, as a program Clover runs: '' when code may not run, the program only
            writes down a value (no calculation), or it gives no value."""
            if self.perms.get()["code"] == "deny" or (cancel and cancel.is_set()):
                return ""
            emit({"type": "cycle", "stage": "python"})
            m2 = [{"role": "system", "content": sys_txt},
                  {"role": "user", "content": user_content(C.python_solve_prompt(context + question + notes, note), media,
                                                           self.media_dir,
                                                           native_video=getattr(self, "has_native_video", True))}]
            content, _, st = self._chat(m2, [], max_new, {}, None, cancel, seed + 950)
            stats_all.append(st)
            import lfm3_layers
            code = lfm3_layers.extract_python(content)
            if not code or not C.computes(code) or (cancel and cancel.is_set()):
                return ""
            ran, out = tool("run_python", {"code": code}, lambda: self.execute_tool("run_python", {"code": code}, ask, allow))
            value = C.python_answer(out) if ran else None
            if value is None:
                return ""
            used.append("a Python calculation")
            shown = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
            return f"Clover ran this Python program:\n```python\n{code.strip()}\n```\nIt printed: {out.strip()[-300:]}\n\nANSWER: {shown}"

        def code_check(ans):
            if "compiler" in layers and kit is not None:
                fb, info = kit.compiler_check(question, ans)
                layer_log.append(info)
                if fb:
                    return fb
            q_comb = f"{question}\n{feedback}" if feedback else question

            def _retry_verify_py(c_str):
                if self.perms.get().get("code") == "deny" or (cancel and cancel.is_set()):
                    return False, "", True
                ran_p, out_p = tool("run_python", {"code": c_str}, lambda: (True, self._run_tool("run_python", {"code": c_str}), "retry sandbox check"))
                worked_p, _ = N.tool_worked("run_python", ran_p, out_p, "retry sandbox check")
                if ran_p:
                    used.append("a Python calculation")
                return ran_p, out_p, worked_p

            v_findings, _ = C.verify_turn_outcome(
                q_comb, ans, rounds, kit=kit, layers=layers, execute_python_fn=_retry_verify_py
            )
            code_f = [f for f in v_findings if f.startswith(("Code ", "Outcome check:"))]
            return "\n".join(code_f) if code_f else None

        with self.lock:
            try:
                notes = ("\n\n" + "\n\n".join(blocks)) if blocks else ""  # Clover's records, after the request
                if not blocks and C.needs_web(context + question) and not (cancel and cancel.is_set()):
                    q = C.web_query(question)  # Context it did not have: search the web, read the best pages
                    emit({"type": "cycle", "stage": "search"})
                    ran, found = tool("web_search", {"query": q}, lambda: self.research_web(q, ask, allow))
                    if ran and found.startswith("Searched "):  # one vote at most: a page can mislead (measured)
                        web = ("\n\nClover searched the web for this question. Use these results only where they "
                               "answer it; they can be incomplete or wrong:\n" + found)
                        layer_log.append({"layer": "web", "query": q, "chars": len(found)})
                        used.append("a web search")
                answer, info = C.retry_cycle(question, previous, feedback, rewrite, solve=solve, emit=emit, used=used,
                                             program=python_solve,
                                             code_check=code_check)
                op_plan.complete_stage("execution")
                op_plan.start_stage("outcome", "Verifying repaired answer against reality…")
                op_plan.complete_stage("outcome", f"Outcome action: {info.get('action', 'corrected')}")
                op_plan.start_stage("growth", "Recording lesson to Clover records for future turns…")
                info["tools"] = list(dict.fromkeys(used))
                layer_log.append(info)
                rnd_n = len(stats_all) or 1
                if "ncm" in layers and not (cancel and cancel.is_set()):
                    if kit is None:
                        raise RuntimeError("NCM output was requested but its model runtime is unavailable")
                    if hasattr(kit, "c1_infer"):
                        c1_prev, _ = kit.c1_infer(previous or question)
                        c1_state, c1_info = kit.c1_infer(answer, stats=stats_all)
                        layer_log.append(c1_info)
                        emit(dict(c1_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c1_done", "ms": c1_info.get("ms")})
                    else:
                        c1_prev, c1_state = None, None
                    if hasattr(kit, "c2_plan_state"):
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_plan", "model": "HybridAgent"})
                        _, _ = kit.c2_plan_state(question, c1_state=c1_prev, offered_tools=info["tools"])
                        if hasattr(kit, "jepa_record_outcome"):
                            j_out = kit.jepa_record_outcome(c1_obs_state=c1_state, is_baseline=False)
                            if isinstance(j_out, dict) and j_out.get("enabled"):
                                layer_log.append({"layer": "jepa_outcome_growth", **j_out})
                                emit({"type": "layer", "layer": "jepa_outcome_growth", **j_out})
                        c2_state, c2_info = kit.c2_plan_state(question, c1_state=c1_state, offered_tools=info["tools"])
                        layer_log.append(c2_info)
                        emit(dict(c2_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_done", "ms": c2_info.get("ms")})
                    else:
                        c2_state = answer
                    if "ncm" in layers and (hasattr(kit, "c3_decode") or hasattr(kit, "ncm_render")):
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_decode", "model": "11-Level NCM"})
                        answer, ncm_info = kit.c3_decode(c2_state) if hasattr(kit, "c3_decode") else kit.ncm_render(answer)
                        layer_log.append(ncm_info)
                        emit(dict(ncm_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_done",
                              "ms": round((ncm_info.get("s") or 0) * 1000)})
                if "ontology" in layers and kit is not None:
                    answer, conflict = kit.grounding_filter(question, answer)
                    layer_log.append({"layer": "ontology_grounding", "conflict": conflict, "rewritten": False})
                if not (cancel and cancel.is_set()):
                    op_plan.complete_stage("growth", "Growth preserved · lesson written to Clover records")
                    emit({"type": "token", "round": rnd_n, "kind": "answer", "text": answer})
            finally:
                self.stage = {"name": "idle", "since": time.time()}
        cancelled = bool(cancel and cancel.is_set())
        return {"reply": "(Stopped.)" if cancelled else answer, "tool_calls": rounds, "rounds": len(stats_all),
                "generation": stats_all, "reasoning": "", "cancelled": cancelled, "layers": layer_log,
                "todo_list": op_plan.todos}

    # ---------------- VS Code (OpenAI messages + VS Code's tools) ----------------
    def vscode_step(self, user_prompt, workspace_info, prior_turns, tool_rounds, vs_tools, thinking=True, max_new=1024,
                    media=(), deferred=()):
        schemas, by_name = [], {}
        active_core = N.active_vscode_tools(user_prompt, workspace_info, tool_rounds, deferred)
        for t in vs_tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = str(fn.get("name") or "").strip()
            if not name or name == "vscode_askQuestions" or name in by_name:
                continue
            f = {"name": name, "description": str(fn.get("description") or "").split("\n")[0][:240],
                 "parameters": N.vscode_tool_params(name, fn.get("parameters") or fn.get("inputSchema")
                                                    or {"type": "object", "properties": {}})}
            by_name[name] = f
            entry = {"type": "function", "function": f}
            if name not in active_core:
                entry["defer_loading"] = True
            schemas.append(entry)
        deferred = [s["function"] for s in schemas if s.get("defer_loading")]
        if deferred:
            st = N._fn("search_tools", "Search the catalog of more VS Code tools by keywords. Returns tools you can then call.",
                       {"query": {"type": "string"}}, ["query"])
            schemas.insert(0, st)
            by_name["search_tools"] = st["function"]
        m_root = re.search(r"-\s*([A-Za-z]:\\[^\r\n]+|/[^\r\n]+)", workspace_info or "")
        root = m_root.group(1).strip() if m_root else ""
        sys_txt = ("You are Clover, a coding assistant inside VS Code, running on the user's laptop (model: LFM3-Omni-2B). "
                   "Inspect the workspace with the tools before answering; never guess file contents. "
                   + (f"The workspace root is {root}; use absolute paths. " if root else "")
                   + ("If none of your tools fits, call search_tools to find one. " if deferred else ""))
        msgs = [{"role": "system", "content": sys_txt}]
        for h in prior_turns[-3:]:
            msgs += [{"role": "user", "content": str(h["user"])[:1500]}, {"role": "assistant", "content": str(h["reply"])[:1500]}]
        ws = f"<workspace_info>\n{workspace_info[:1500]}\n</workspace_info>\n" if workspace_info else ""
        msgs.append({"role": "user", "content": user_content(ws + user_prompt[:6000], media, self.media_dir)})
        for rd in tool_rounds[-8:]:
            cid = rd.get("id") or ("call_" + uuid.uuid4().hex[:12])
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": rd["name"], "arguments": rd.get("arguments") or "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": N._clip(rd.get("output") or "", 5000)})
        trace = []
        nudged, answer = False, ""
        with self.lock:
            try:
                for _ in range(4):
                    content, calls, st = self._chat(msgs, schemas, max_new, by_name)
                    trace.append({"generation": st, "calls": [c["name"] for c in calls], "text": content[-400:]})
                    if calls and calls[0]["name"] == "search_tools":
                        q = str(calls[0]["arguments"].get("query") or "").lower().split()
                        # offered tools too: a search for the Playwright tool it already had found nothing, four times
                        pool = [f for n, f in by_name.items() if n != "search_tools"]
                        scored = sorted(pool, key=lambda f: -sum(w in (f["name"] + " " + f["description"]).lower() for w in q))
                        found = [f["name"] for f in scored[:3] if any(w in (f["name"] + " " + f["description"]).lower() for w in q)]
                        msgs.append({"role": "assistant", "content": content, "tool_calls": [
                            {"id": calls[0]["id"], "type": "function",
                             "function": {"name": "search_tools", "arguments": json.dumps(calls[0]["arguments"])}}]})
                        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content":
                                     f"Found {len(found)} tools:\n" + "\n".join(json.dumps(by_name[n]) for n in found)})
                        continue
                    if calls:
                        c = calls[0]
                        refused = N.unshared_open(c["name"], c["arguments"], "\n".join(
                            [user_prompt] + [str(h.get("user") or "") for h in prior_turns[-3:]]), workspace_info, tool_rounds)
                        if refused:
                            trace[-1]["refused"] = str(c["arguments"].get("url"))[:200]
                            msgs.append({"role": "assistant", "content": content, "tool_calls": [
                                {"id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}]})
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": refused})
                            continue
                        read = N.shared_page_read(c["name"], c["arguments"], user_prompt, workspace_info, by_name)
                        if read:
                            trace[-1]["read_instead"] = True
                            return {"tool_call": {"name": read["name"], "arguments": json.dumps(read["arguments"])}, "trace": trace}
                        c_args = N.repair_vscode_browser_args(c["name"], c["arguments"], workspace_info, tool_rounds)
                        if nudged and N.repeats_round(c["name"], c_args, tool_rounds):
                            return {"content": answer or "(Clover did not finish an answer in time.)", "trace": trace}
                        return {"tool_call": {"name": c["name"], "arguments": json.dumps(c_args)}, "trace": trace}
                    nudge = "" if nudged else N.unfinished_page_steps(user_prompt, workspace_info, tool_rounds, by_name)
                    if nudge:
                        nudged = trace[-1]["nudged"] = True
                        answer = content.strip()
                        msgs += [{"role": "assistant", "content": content}, {"role": "user", "content": nudge}]
                        continue
                    return {"content": content.strip() or "(Clover did not finish an answer in time.)", "trace": trace}
                return {"content": "(Clover kept searching for tools without choosing one.)", "trace": trace}
            finally:
                self.stage = {"name": "idle", "since": time.time()}
