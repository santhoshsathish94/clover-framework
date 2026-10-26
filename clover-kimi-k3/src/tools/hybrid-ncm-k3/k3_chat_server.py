"""Clover AI — Multi-Session, Dynamic-Batching (1.0s window), 8-GPU Concurrent Server (`k3_chat_server.py`).
Features:
  1. Uses all 8 RTX 3090 GPUs:
     - NVLink Worker Pairs `(cuda:0 <-> cuda:2)`, `(cuda:1 <-> cuda:3)`, `(cuda:4 <-> cuda:6)` run concurrent
       `K3FastChatEngine` text/batch instances in parallel across GPU pairs.
     - `(cuda:5 <-> cuda:7)` runs `K3MultimodalChatEngine` (`Qwen3-VL-8B-Instruct + HybridLink + e9_k3_multimodal`
       + optional `MoonViT3d`) for image + text prompts.
  2. 1.0-Second Dynamic Batching Queue (`MultiGPUCloverCoordinator`):
     - Incoming text requests wait up to `1.0 s` (`BATCH_PAUSE_S = 1.0`) to club concurrent requests arriving in
       that window into a single batched GPU pass (`plan_batch_with_latent` + `render_ncm_batched_multi_prompt`),
       and dispatches multiple batches concurrently across the 8-GPU worker pool.
  3. Isolated Per-Session Context Memory (`sessions[session_id]`):
     - Each browser tab / user session (`session_id`) has its own isolated multi-turn conversation history,
       supporting unlimited concurrent sessions without mixing context.
  4. Image Upload Support (`image_base64` data URL in `/api/chat` and `/v1/chat/completions`):
     - Decodes uploaded PNG/JPEG/WebP images and routes them through the multimodal GPU worker (`cuda:5 <-> cuda:7`).
  5. Clean "Clover AI" Light-Theme Web UI matching `cloverframework.com`:
     - Uses the 5-leaf Clover SVG mark as the animated loading symbol and hides internal technical telemetry/timings.
"""
import argparse, base64, collections, concurrent.futures, io, json, os, re, subprocess, sys, tempfile, threading, time, urllib.parse, uuid, wave, xml.etree.ElementTree as ET, zipfile
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from clover_paths import TOOLS_DIR, setup_clover_sys_path, resolve_device, cuda_sync

ROOT = setup_clover_sys_path()
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
from PIL import Image
from k3_fast_chat import K3FastChatEngine, CHAT_PLAN_PROMPT, CLOVER_DEFAULT_CONTEXT_PATH, load_default_clover_context
from k3_agent_engine import CloverK3AgentEngine, K3GPUExpertsRunner, run_agent_batch_with_k3_experts
from k3_multimodal_chat import K3MultimodalChatEngine
from bench_batch_clubbing import plan_batch_with_latent, render_ncm_batched_multi_prompt


BATCH_PAUSE_S = 1.0
MAX_SESSION_CONTEXT_TOKENS = 262_144  # 256k Kimi K3 tokens (> 250k minimum context window per session)
SESSIONS_STORE_PATH = os.environ.get(
    "CLOVER_SESSIONS_STORE_PATH",
    os.path.join(TOOLS_DIR, "out/sessions_store.json"),
)
KID_VOICE_NAME = "en-US-AnaNeural"  # Microsoft Neural polite, friendly child voice
_TTS_CACHE = collections.OrderedDict()
_TTS_LOCK = threading.Lock()


def synthesize_polite_kid_voice_b64(text: str, voice: str = KID_VOICE_NAME) -> str:
    """Synthesizes speech using Microsoft's polite child neural voice (`en-US-AnaNeural`) via `edge_tts`
    and returns a `data:audio/mpeg;base64,...` URL (with in-memory LRU caching)."""
    import asyncio
    import edge_tts

    # Strip code blocks, URLs, and extra symbols so the spoken voice sounds natural, simple, and polite
    clean = re.sub(r"```[\s\S]*?```", " Here is the code example. ", str(text or ""))
    clean = re.sub(r"https?://[^\s)]+", "", clean)
    clean = re.sub(r"[•*#_`|~]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()[:1600]
    if not clean:
        clean = "Hi, I am Clover! How can I help you today?"

    cache_key = (voice, clean)
    with _TTS_LOCK:
        if cache_key in _TTS_CACHE:
            _TTS_CACHE.move_to_end(cache_key)
            return _TTS_CACHE[cache_key]

    async def _run():
        comm = edge_tts.Communicate(clean, voice=voice, rate="+0%", pitch="+0Hz")
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    mp3_bytes = asyncio.run(_run())
    data_url = "data:audio/mpeg;base64," + base64.b64encode(mp3_bytes).decode("ascii")
    with _TTS_LOCK:
        _TTS_CACHE[cache_key] = data_url
        while len(_TTS_CACHE) > 64:
            _TTS_CACHE.popitem(last=False)
    return data_url

RETRIEVAL_STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "to", "of", "and", "in", "on", "at", "for", "with",
    "by", "from", "as", "an", "it", "that", "this", "these", "those", "what", "which", "who", "whom",
    "how", "why", "when", "where", "can", "could", "would", "should", "will", "do", "does", "did", "you",
    "your", "me", "my", "we", "our", "any", "new", "more", "some", "please", "about", "into", "over",
    "after", "before", "between", "under", "again", "further", "then", "once", "here", "there", "all",
    "both", "each", "few", "most", "other", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "now", "tell", "give", "show", "explain", "describe",
}


def extract_server_attachment_text(att: dict) -> str:
    """Extracts readable text from any uploaded attachment dict:
    `{"name": str, "type": str, "content_text": str, "data_b64": str}`
    Supports PDF, DOCX, TXT, MD, CSV, JSON, Python/JS/TS/C/C++/HTML/CSS code, and Audio/Video metadata."""
    name = str(att.get("name") or "attachment").strip()
    mime = str(att.get("type") or att.get("mime") or "").lower().strip()
    ext = os.path.splitext(name.lower())[1]
    client_txt = str(
        att.get("content_text")
        or att.get("text_excerpt")
        or att.get("text")
        or att.get("content")
        or ""
    ).strip()

    b64_str = str(att.get("data_b64") or "").strip()
    raw_bytes = b""
    if b64_str:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(b64_str)
        except Exception:
            raw_bytes = b""

    # 1. Word .docx extraction via built-in zipfile + XML
    if (ext == ".docx" or "officedocument.wordprocessingml" in mime) and raw_bytes:
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                if "word/document.xml" in zf.namelist():
                    xml_bytes = zf.read("word/document.xml")
                    root = ET.fromstring(xml_bytes)
                    paras = []
                    for p in root.iter():
                        if p.tag.endswith("}p"):
                            texts = [t.text for t in p.iter() if t.tag.endswith("}t") and t.text]
                            if texts:
                                paras.append("".join(texts).strip())
                    doc_txt = "\n".join(p for p in paras if p)
                    if doc_txt.strip():
                        return f"Attached Document ({name}):\n{doc_txt.strip()}"
        except Exception:
            pass

    # 2. PDF extraction via pypdf / fitz / pdftotext / PDF stream fallback
    if (ext == ".pdf" or "pdf" in mime) and raw_bytes:
        pdf_txt = ""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = [pg.extract_text() or "" for pg in reader.pages]
            pdf_txt = "\n".join(p.strip() for p in pages if p.strip())
        except Exception:
            pass
        if not pdf_txt:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    tf.write(raw_bytes)
                    tmp_pdf = tf.name
                try:
                    out = subprocess.check_output(["pdftotext", tmp_pdf, "-"], timeout=8)
                    pdf_txt = out.decode("utf-8", errors="ignore").strip()
                finally:
                    if os.path.exists(tmp_pdf):
                        os.remove(tmp_pdf)
            except Exception:
                pass
        if not pdf_txt:
            # Lightweight PDF literal string extractor for uncompressed PDFs
            try:
                latin = raw_bytes.decode("latin-1", errors="ignore")
                chunks = re.findall(r"\(([^()\\]{3,200})\)\s*Tj", latin)
                if chunks:
                    pdf_txt = " ".join(chunks)
            except Exception:
                pass
        if pdf_txt.strip():
            return f"Attached PDF ({name}):\n{pdf_txt.strip()}"

    # 3. Audio metadata / WAV inspection
    if ext in (".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac") or mime.startswith("audio/"):
        info = f"Attached Audio File ({name}, {mime or ext}, {len(raw_bytes)} bytes)"
        if ext == ".wav" and raw_bytes:
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    dur = round(wf.getnframes() / float(max(wf.getframerate(), 1)), 2)
                    info = f"Attached WAV Audio ({name} | duration={dur}s, sample_rate={wf.getframerate()}Hz, channels={wf.getnchannels()})"
            except Exception:
                pass
        if client_txt:
            return f"{info}\nTranscript / Audio Notes:\n{client_txt}"
        return info

    # 4. If client already extracted text (or plain text / code / CSV / JSON / MD)
    if client_txt:
        return f"Attached Document ({name}):\n{client_txt}"

    if raw_bytes:
        try:
            decoded = raw_bytes.decode("utf-8", errors="ignore").strip()
            if decoded and sum(1 for c in decoded[:400] if c.isprintable() or c in "\r\n\t") >= 0.85 * min(len(decoded), 400):
                return f"Attached Document ({name}):\n{decoded}"
        except Exception:
            pass

    return f"Attached File ({name}, {mime or ext or 'binary'})"


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="8 5 84 93" width="64" height="64">
  <defs>
    <path id="clover-leaf" d="M50 44C48 35 38 31 38 21C38 12 46 9 50 17C54 9 62 12 62 21C62 31 52 35 50 44Z" fill="#0a6a5f" stroke="#0a6a5f" stroke-width="2" stroke-linejoin="round"/>
  </defs>
  <path d="M50 48C50 68 50 80 50 95" fill="none" stroke="#7fb8b0" stroke-width="4.5" stroke-linecap="round"/>
  <use href="#clover-leaf"/>
  <use href="#clover-leaf" transform="rotate(72 50 44)"/>
  <use href="#clover-leaf" transform="rotate(144 50 44)"/>
  <use href="#clover-leaf" transform="rotate(216 50 44)"/>
  <use href="#clover-leaf" transform="rotate(288 50 44)"/>
</svg>"""


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clover</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  :root {
    --ink: #14181d;
    --ink-2: #39424d;
    --muted: #66717e;
    --line: #e2e7ec;
    --line-2: #cfd6de;
    --bg: #ffffff;
    --soft: #f6f8fa;
    --soft-2: #eef2f6;
    --accent: #0a6a5f;
    --accent-soft: #e6f2f0;
    --accent-line: #7fb8b0;
    --radius: 10px;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: var(--font); background: var(--bg); color: var(--ink);
    display: flex; flex-direction: column; height: 100vh; -webkit-font-smoothing: antialiased;
  }
  .clover-sprite { position: absolute; width: 0; height: 0; overflow: hidden; }
  header {
    padding: 0.85rem 1.25rem; background: var(--bg); border-bottom: 1px solid var(--line);
    display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap;
  }
  .brand {
    display: inline-flex; align-items: center; gap: 0.55rem;
    font-weight: 680; font-size: 1.05rem; letter-spacing: -0.02em; color: var(--ink);
  }
  .brand span { color: var(--accent); font-weight: 600; }
  .clover-icon { width: 26px; height: 26px; display: block; flex: none; }
  .clover-icon__leaf { fill: var(--accent); stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; }
  .clover-icon__stem { fill: none; stroke: var(--accent-line); stroke-width: 3.5; stroke-linecap: round; }
  .btn {
    background: var(--bg); color: var(--ink-2); border: 1px solid var(--line-2);
    border-radius: 8px; padding: 0.42rem 0.8rem; font: inherit; font-size: 0.85rem;
    font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 0.4rem;
    transition: border-color 0.15s ease, color 0.15s ease, background-color 0.15s ease;
  }
  .btn:hover { border-color: var(--accent-line); color: var(--accent); }
  .btn--primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn--primary:hover { background: #085349; border-color: #085349; color: #fff; }
  .btn:disabled { opacity: 0.55; cursor: default; }
  .icon-btn {
    width: 40px; height: 40px; padding: 0; justify-content: center; flex: 0 0 40px; border-radius: 9px;
  }
  .icon-btn.is-listening {
    border-color: var(--accent); background: var(--accent-soft); color: var(--accent);
  }
  .workspace {
    flex: 1; display: flex; min-height: 0; background: var(--soft);
  }
  .sidebar {
    width: 240px; flex: 0 0 240px; background: var(--bg); border-right: 1px solid var(--line);
    display: flex; flex-direction: column;
  }
  .sidebar__head {
    padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--line);
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.82rem; font-weight: 650; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em;
  }
  .add-btn {
    width: 28px; height: 28px; border-radius: 7px; border: 1px solid var(--line-2);
    background: var(--bg); color: var(--ink); font-size: 1.1rem; font-weight: 600;
    display: inline-flex; align-items: center; justify-content: center; cursor: pointer;
  }
  .add-btn:hover { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }
  .session-list {
    flex: 1; overflow-y: auto; padding: 0.5rem; display: flex; flex-direction: column; gap: 0.25rem;
  }
  .session-item {
    width: 100%; display: flex; align-items: center; gap: 0.25rem; background: transparent; border: 1px solid transparent;
    border-radius: 8px; padding: 0.15rem 0.25rem 0.15rem 0.45rem; font: inherit; font-size: 0.87rem;
    color: var(--ink-2);
  }
  .session-item:hover { background: var(--soft); color: var(--ink); }
  .session-item.is-active {
    background: var(--accent-soft); border-color: var(--accent-line); color: var(--accent); font-weight: 650;
  }
  .session-select {
    flex: 1; min-width: 0; text-align: left; background: transparent; border: 0;
    padding: 0.38rem 0.3rem; font: inherit; color: inherit; cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .session-del {
    width: 24px; height: 24px; flex: 0 0 24px; border-radius: 6px; border: 0;
    background: transparent; color: var(--muted); font-size: 0.98rem; font-weight: 650;
    cursor: pointer; display: inline-flex; align-items: center; justify-content: center;
  }
  .session-del:hover { background: rgba(190, 38, 38, 0.12); color: #b91c1c; }
  .main-pane { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  #chat {
    flex: 1; overflow-y: auto; padding: 1.4rem; background: var(--soft);
    display: flex; flex-direction: column; gap: 1rem;
  }
  .msg {
    max-width: 82%; padding: 0.85rem 1.1rem; border-radius: 12px;
    line-height: 1.62; font-size: 0.96rem; word-wrap: break-word;
  }
  .msg p { margin: 0 0 0.65rem 0; }
  .msg p:last-child { margin-bottom: 0; }
  .msg ul {
    margin: 0.35rem 0 0.65rem 1.25rem; padding: 0;
    display: flex; flex-direction: column; gap: 0.42rem;
  }
  .msg ul:last-child { margin-bottom: 0; }
  .msg li { line-height: 1.56; }
  .msg strong { font-weight: 650; color: var(--ink); }
  .msg pre.k3-code {
    margin: 0.55rem 0 0.75rem 0; padding: 0.85rem 1rem; border-radius: 8px;
    background: #14181d; color: #f6f8fa; overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.87rem; line-height: 1.52; white-space: pre; position: relative;
  }
  .msg pre.k3-code:last-child { margin-bottom: 0; }
  .msg pre.k3-code code { background: none; color: inherit; padding: 0; font: inherit; }
  .k3-code-wrap { position: relative; margin: 0.55rem 0 0.75rem 0; }
  .k3-code-wrap pre.k3-code { margin: 0; padding-top: 2rem; }
  .k3-copy-code-btn {
    position: absolute; top: 0.4rem; right: 0.5rem; z-index: 2;
    background: rgba(255, 255, 255, 0.12); color: #e2e7ec; border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 6px; padding: 0.18rem 0.55rem; font-size: 0.75rem; font-weight: 600; cursor: pointer;
  }
  .k3-figure {
    margin: 0.65rem 0 0.8rem 0; padding: 0.65rem; background: var(--soft);
    border: 1px solid var(--line-2); border-radius: 10px; text-align: center;
  }
  .k3-figure img, .k3-figure svg {
    display: block; max-width: 100%; height: auto; margin: 0 auto; border-radius: 7px; background: #fff;
  }
  .k3-figcaption {
    margin-top: 0.45rem; font-size: 0.82rem; color: var(--muted); line-height: 1.4;
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; flex-wrap: wrap; text-align: left;
  }
  .k3-dl-link {
    font-size: 0.78rem; font-weight: 650; color: var(--accent); text-decoration: none;
    padding: 0.15rem 0.5rem; border: 1px solid var(--accent-line); border-radius: 6px; background: var(--accent-soft);
  }
  .msg table.k3-table {
    width: 100%; border-collapse: collapse; margin: 0.55rem 0 0.75rem 0; font-size: 0.88rem;
  }
  .msg table.k3-table th, .msg table.k3-table td {
    border: 1px solid var(--line-2); padding: 0.42rem 0.65rem; text-align: left;
  }
  .msg table.k3-table th { background: var(--soft); font-weight: 650; color: var(--ink); }
  .k3-msg-actions {
    margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px solid var(--line);
    display: flex; align-items: center; gap: 0.45rem; justify-content: flex-end;
  }
  .k3-action-btn {
    background: transparent; color: var(--muted); border: 1px solid transparent;
    border-radius: 6px; padding: 0.16rem 0.48rem; font: inherit; font-size: 0.76rem;
    font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 0.28rem;
  }
  .msg--user {
    align-self: flex-end; background: var(--accent); color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg--bot {
    align-self: flex-start; background: var(--bg); color: var(--ink);
    border: 1px solid var(--line); border-bottom-left-radius: 4px;
    box-shadow: 0 2px 6px rgba(20, 24, 29, 0.03);
  }
  .msg img.chat-thumb {
    display: block; max-width: 240px; max-height: 180px; border-radius: 8px;
    margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.25);
  }
  .attach-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.26rem 0.6rem; border-radius: 7px; font-size: 0.82rem; font-weight: 600;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.32); margin-bottom: 0.45rem;
  }
  @keyframes clover-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  .clover-loader {
    display: inline-flex; align-items: center; gap: 0.65rem; color: var(--ink-2); font-size: 0.92rem;
  }
  .clover-loader svg { width: 28px; height: 28px; }
  .clover-loader__wheel {
    transform-origin: 50px 44px;
    animation: clover-spin 2.2s linear infinite;
  }
  .preview-bar {
    padding: 0.5rem 1.25rem; background: var(--accent-soft); border-top: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; font-size: 0.85rem; color: var(--ink);
  }
  .preview-bar[hidden] { display: none !important; }
  .preview-bar__info { display: flex; align-items: center; gap: 0.6rem; }
  .preview-bar img { width: 36px; height: 36px; object-fit: cover; border-radius: 6px; border: 1px solid var(--line-2); }
  footer {
    padding: 0.85rem 1.25rem; background: var(--bg); border-top: 1px solid var(--line);
    display: flex; gap: 0.6rem; align-items: flex-end;
  }
  textarea {
    flex: 1; background: var(--soft); color: var(--ink); border: 1px solid var(--line-2);
    border-radius: 9px; padding: 0.65rem 0.85rem; font: inherit; font-size: 0.95rem;
    resize: none; min-height: 44px; max-height: 130px; line-height: 1.45;
  }
  textarea:focus { outline: none; border-color: var(--accent); background: var(--bg); }
  .mem-drawer {
    position: fixed; top: 56px; right: 0; bottom: 0; width: 420px; max-width: 92vw;
    background: var(--bg); border-left: 1px solid var(--line-2); z-index: 50;
    display: flex; flex-direction: column; box-shadow: -4px 0 18px rgba(20, 24, 29, 0.08);
  }
  .mem-drawer[hidden] { display: none !important; }
  .mem-drawer__head {
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--line); display: flex;
    align-items: center; justify-content: space-between; font-weight: 700; font-size: 0.92rem;
  }
  .mem-drawer__body { flex: 1; overflow-y: auto; padding: 0.85rem 1rem; font-size: 0.83rem; }
  .mem-card {
    border: 1px solid var(--line); border-radius: 8px; padding: 0.65rem 0.75rem;
    margin-bottom: 0.55rem; background: var(--soft);
  }
  .mem-card--staged { background: #fffbeb; border-color: #f59e0b; }
  .mem-badge {
    display: inline-block; padding: 0.1rem 0.42rem; border-radius: 5px; font-size: 0.72rem;
    font-weight: 600; background: var(--accent-soft); color: var(--accent); margin-right: 0.3rem;
  }
</style>
</head>
<body>
<svg class="clover-sprite" width="0" height="0" aria-hidden="true" focusable="false">
  <defs>
    <path id="clover-leaf" d="M50 44C48 35 38 31 38 21C38 12 46 9 50 17C54 9 62 12 62 21C62 31 52 35 50 44Z"/>
  </defs>
</svg>

<header>
  <div class="brand">
    <svg class="clover-icon" viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      <path class="clover-icon__stem" d="M50 48C50 68 50 80 50 95"/>
      <use class="clover-icon__leaf" href="#clover-leaf"/>
      <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(72 50 44)"/>
      <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(144 50 44)"/>
      <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(216 50 44)"/>
      <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(288 50 44)"/>
    </svg>
    Clover
  </div>
  <div>
    <button type="button" class="btn" onclick="toggleMemoryDrawer()" id="mem-drawer-btn" title="Inspect File-Based Memory Tree & Relational Knowledge Graph">Memory Graph</button>
  </div>
</header>

<aside class="mem-drawer" id="mem-drawer" hidden aria-label="Memory Graph Inspector">
  <div class="mem-drawer__head">
    <span id="mem-drawer-title">Collective Memory Graph</span>
    <button type="button" class="btn" onclick="toggleMemoryDrawer()">Close</button>
  </div>
  <div style="padding:0.55rem 1rem;border-bottom:1px solid var(--line);display:flex;gap:0.4rem;">
    <input type="text" id="mem-query-in" placeholder="Search concepts, IDs, or chains..." style="flex:1;padding:0.38rem 0.6rem;border:1px solid var(--line-2);border-radius:6px;font:inherit;font-size:0.82rem;">
    <button type="button" class="btn" onclick="queryMemoryDrawer()">Query</button>
    <button type="button" class="btn" onclick="loadMemoryDrawer()">All</button>
  </div>
  <div class="mem-drawer__body" id="mem-drawer-body">Loading memory graph...</div>
</aside>

<div class="workspace">
  <aside class="sidebar" aria-label="Chat sessions">
    <div class="sidebar__head">
      <span>Sessions</span>
      <button type="button" class="add-btn" onclick="newSession()" title="New session" aria-label="New session">+</button>
    </div>
    <div class="session-list" id="session-list"></div>
  </aside>
  <div class="main-pane">
    <div id="chat" aria-live="polite">
      <div class="msg msg--bot"><p>Hi I am Clover, How can I help you today?</p></div>
    </div>

    <div class="preview-bar" id="img-preview-bar" hidden>
      <div class="preview-bar__info">
        <img id="img-preview-thumb" src="" alt="Preview" hidden>
        <span id="img-preview-name">Attachment ready</span>
      </div>
      <button type="button" class="btn" onclick="clearAttachments()">Remove</button>
    </div>

    <footer>
      <input type="file" id="file-input" multiple hidden onchange="onFilesSelected(event)">
      <button type="button" class="btn icon-btn" onclick="document.getElementById('file-input').click()" title="Attach files (images, video, audio, PDF, documents, text)" aria-label="Attach files">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
        </svg>
      </button>
      <textarea id="prompt-in" rows="1" placeholder="Message Clover..."></textarea>
      <button type="button" class="btn icon-btn" id="voice-btn" onclick="toggleVoice()" title="Voice input" aria-label="Voice input">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="23"></line>
          <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
      </button>
      <button type="button" class="btn btn--primary" id="send-btn" onclick="sendPrompt()">Send</button>
    </footer>
  </div>
</div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('prompt-in');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const previewBar = document.getElementById('img-preview-bar');
const previewThumb = document.getElementById('img-preview-thumb');
const previewName = document.getElementById('img-preview-name');
const sessionListEl = document.getElementById('session-list');
const voiceBtn = document.getElementById('voice-btn');

const STORAGE_KEY = 'clover_k3_sessions_v2';
let sessionsMap = {};
let sessionOrder = [];
let sessionId = '';
let attachedImageDataUrl = null;
let attachedFiles = [];

const GREETING_HTML = '<div class="msg msg--bot"><p>Hi I am Clover, How can I help you today?</p><div class="k3-msg-actions"><button type="button" class="k3-action-btn" onclick="speakReply(this)" title="Read aloud">🔊 Listen</button></div></div>';

const CLOVER_LOADER_HTML = `
  <div class="clover-loader" role="status" aria-label="Thinking">
    <svg viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      <path class="clover-icon__stem" d="M50 48C50 68 50 80 50 95"/>
      <g class="clover-loader__wheel">
        <use class="clover-icon__leaf" href="#clover-leaf"/>
        <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(72 50 44)"/>
        <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(144 50 44)"/>
        <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(216 50 44)"/>
        <use class="clover-icon__leaf" href="#clover-leaf" transform="rotate(288 50 44)"/>
      </g>
    </svg>
    <span>Thinking&hellip;</span>
  </div>
`;

function makeSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2, 9) + Date.now().toString(36);
}

function saveSessions() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ currentId: sessionId, order: sessionOrder, map: sessionsMap }));
  } catch (e) {}
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.order) && data.map) {
        sessionsMap = data.map;
        sessionOrder = data.order.filter(id => !!sessionsMap[id]);
        sessionId = (data.currentId && sessionsMap[data.currentId]) ? data.currentId : (sessionOrder[0] || '');
      }
    }
  } catch (e) {}
  if (!sessionId || !sessionsMap[sessionId]) {
    const firstId = makeSessionId();
    sessionsMap[firstId] = { id: firstId, title: 'New chat', messages: [] };
    sessionOrder = [firstId];
    sessionId = firstId;
    saveSessions();
  }
}

function renderSidebar() {
  sessionListEl.innerHTML = '';
  sessionOrder.forEach(sid => {
    const sObj = sessionsMap[sid];
    if (!sObj) return;
    const row = document.createElement('div');
    row.className = 'session-item' + (sid === sessionId ? ' is-active' : '');
    row.setAttribute('data-session-id', sid);

    const selBtn = document.createElement('button');
    selBtn.type = 'button';
    selBtn.className = 'session-select';
    selBtn.textContent = sObj.title || 'New chat';
    selBtn.onclick = () => selectSession(sid);

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'session-del';
    delBtn.title = 'Remove session';
    delBtn.setAttribute('aria-label', 'Delete session');
    delBtn.textContent = '×';
    delBtn.onclick = (ev) => {
      if (ev && ev.stopPropagation) ev.stopPropagation();
      deleteSession(sid);
    };

    row.appendChild(selBtn);
    row.appendChild(delBtn);
    sessionListEl.appendChild(row);
  });
}

function renderMessages() {
  chatEl.innerHTML = GREETING_HTML;
  const sObj = sessionsMap[sessionId];
  if (!sObj || !Array.isArray(sObj.messages)) return;
  sObj.messages.forEach(m => addMsg(m.role, m.html));
}

function selectSession(sid) {
  if (!sessionsMap[sid]) return;
  sessionId = sid;
  clearAttachments();
  renderSidebar();
  renderMessages();
  saveSessions();
}

function deleteSession(sid) {
  if (!sid || !sessionsMap[sid]) return;
  delete sessionsMap[sid];
  sessionOrder = sessionOrder.filter(id => id !== sid);
  try {
    fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid })
    }).catch(() => {});
  } catch (e) {}
  if (!sessionOrder.length) {
    const nextId = makeSessionId();
    sessionsMap[nextId] = { id: nextId, title: 'New chat', messages: [] };
    sessionOrder = [nextId];
    sessionId = nextId;
  } else if (sessionId === sid) {
    sessionId = sessionOrder[0];
  }
  clearAttachments();
  renderSidebar();
  renderMessages();
  saveSessions();
}

function newSession() {
  const cur = sessionsMap[sessionId];
  if (cur && (!cur.messages || cur.messages.length === 0)) {
    clearAttachments();
    renderMessages();
    renderSidebar();
    return;
  }
  const newId = makeSessionId();
  sessionsMap[newId] = { id: newId, title: 'New chat', messages: [] };
  sessionOrder.unshift(newId);
  sessionId = newId;
  clearAttachments();
  renderSidebar();
  renderMessages();
  saveSessions();
}

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escAndLinkify(s) {
  let escaped = escHtml(s);
  escaped = escaped.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, (_, label, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  escaped = escaped.replace(/(^|[\\s(])((https?:\\/\\/[^\\s)<]+))/g, (_, pre, url) => {
    const cleanUrl = url.replace(/[.,;!?]+$/, '');
    const trail = url.slice(cleanUrl.length);
    return `${pre}<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>${trail}`;
  });
  return escaped;
}

function htmlToPlainText(htmlStr) {
  const cleaned = String(htmlStr || '')
    .replace(/<div[^>]*class="[^"]*attach-chip[^"]*"[^>]*>[\\s\\S]*?<\\/div>/gi, '')
    .replace(/<\\/(p|li|ul|div|h[1-6])>/gi, '\\n')
    .replace(/<br\\s*\\/?>/gi, '\\n');
  const tmp = document.createElement('div');
  tmp.innerHTML = cleaned;
  return (tmp.textContent || tmp.innerText || '').replace(/\\n{3,}/g, '\\n\\n').trim();
}

function extractPriorTurns(sid) {
  const sObj = sessionsMap[sid];
  if (!sObj || !Array.isArray(sObj.messages)) return [];
  const turns = [];
  let pendingUser = '';
  for (const m of sObj.messages) {
    const txt = (m.text !== undefined && m.text !== null) ? String(m.text).trim() : htmlToPlainText(m.html);
    if (m.role === 'user') pendingUser = txt;
    else if (m.role === 'bot' && pendingUser && txt) {
      turns.push({ user: pendingUser, reply: txt });
      pendingUser = '';
    }
  }
  return turns;
}

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg msg--' + role;
  d.innerHTML = html;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

function updatePreviewBar() {
  if (!attachedImageDataUrl && !attachedFiles.length) {
    previewBar.hidden = true;
    return;
  }
  previewBar.hidden = false;
  if (attachedImageDataUrl) {
    previewThumb.src = attachedImageDataUrl;
    previewThumb.hidden = false;
  } else {
    previewThumb.hidden = true;
  }
  previewName.textContent = attachedFiles.map(f => f.name).join(', ') || 'Attachment ready';
}

function isTextFile(file) {
  const t = (file.type || '').toLowerCase();
  const n = (file.name || '').toLowerCase();
  if (t.startsWith('text/') || t.includes('json') || t.includes('xml') || t.includes('javascript')) return true;
  return /\\.(txt|md|markdown|csv|tsv|json|jsonl|py|js|ts|tsx|jsx|c|cpp|h|hpp|rs|go|java|html|css|xml|yaml|yml|toml|ini|sh|ps1|sql|log|tex)$/i.test(n);
}

function onFilesSelected(e) {
  const files = Array.from((e.target && e.target.files) || []);
  files.forEach(file => {
    const fType = (file.type || '').toLowerCase();
    if (fType.startsWith('image/')) {
      const rd = new FileReader();
      rd.onload = ev => {
        attachedImageDataUrl = ev.target.result;
        attachedFiles.push({ name: file.name, type: file.type || 'image/png', content_text: '', data_b64: ev.target.result });
        updatePreviewBar();
      };
      rd.readAsDataURL(file);
    } else if (isTextFile(file)) {
      const rd = new FileReader();
      rd.onload = ev => {
        attachedFiles.push({ name: file.name, type: file.type || 'text/plain', content_text: String(ev.target.result || ''), data_b64: '' });
        updatePreviewBar();
      };
      rd.readAsText(file);
    } else {
      const rd = new FileReader();
      rd.onload = ev => {
        attachedFiles.push({ name: file.name, type: file.type || 'application/octet-stream', content_text: '', data_b64: String(ev.target.result || '') });
        updatePreviewBar();
      };
      rd.readAsDataURL(file);
    }
  });
}

function clearAttachments() {
  attachedImageDataUrl = null;
  attachedFiles = [];
  fileInput.value = '';
  previewBar.hidden = true;
}

let recog = null;
let listening = false;
let lastInputWasVoice = false;
function toggleVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { inputEl.focus(); return; }
  if (listening && recog) { recog.stop(); return; }
  recog = new SR();
  recog.lang = 'en-US';
  recog.interimResults = false;
  listening = true;
  voiceBtn.classList.add('is-listening');
  recog.onresult = ev => {
    const t = ev.results[0][0].transcript;
    inputEl.value = (inputEl.value ? inputEl.value + ' ' : '') + t;
    lastInputWasVoice = true;
    inputEl.focus();
  };
  recog.onend = () => { listening = false; voiceBtn.classList.remove('is-listening'); };
  recog.onerror = () => { listening = false; voiceBtn.classList.remove('is-listening'); };
  recog.start();
}

let activeTtsAudio = null;
let activeTtsBtn = null;
const ttsAudioCache = {};

function stopCurrentSpeech() {
  if (activeTtsAudio) {
    try { activeTtsAudio.pause(); activeTtsAudio.currentTime = 0; } catch (e) {}
    activeTtsAudio = null;
  }
  if (window.speechSynthesis && window.speechSynthesis.speaking) {
    try { window.speechSynthesis.cancel(); } catch (e) {}
  }
  if (activeTtsBtn) {
    activeTtsBtn.textContent = '🔊 Listen';
    activeTtsBtn = null;
  }
}

async function speakReply(btn) {
  if (activeTtsBtn === btn && activeTtsAudio) {
    stopCurrentSpeech();
    return;
  }
  stopCurrentSpeech();
  const msgEl = btn.closest('.msg');
  if (!msgEl) return;
  const clone = msgEl.cloneNode(true);
  clone.querySelectorAll('.k3-msg-actions, .k3-figcaption, svg, pre').forEach(el => el.remove());
  const speakTxt = (clone.textContent || '').replace(/\\s+/g, ' ').trim();
  if (!speakTxt) return;
  btn.textContent = '⏳ Voice...';
  activeTtsBtn = btn;
  try {
    let audioDataUrl = ttsAudioCache[speakTxt];
    if (!audioDataUrl) {
      const r = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: speakTxt })
      });
      const d = await r.json();
      audioDataUrl = d.audio_base64;
      ttsAudioCache[speakTxt] = audioDataUrl;
    }
    if (activeTtsBtn !== btn) return;
    const audio = new Audio(audioDataUrl);
    activeTtsAudio = audio;
    btn.textContent = '⏹ Stop';
    audio.onended = () => { if (activeTtsBtn === btn) btn.textContent = '🔊 Listen'; activeTtsAudio = null; activeTtsBtn = null; };
    await audio.play();
  } catch (e) {
    if (activeTtsBtn === btn) btn.textContent = '🔊 Listen';
    activeTtsAudio = null;
    activeTtsBtn = null;
  }
}

function renderMediaItemsHtml(mediaList) {
  if (!Array.isArray(mediaList) || !mediaList.length) return '';
  return mediaList.map(m => {
    if (!m) return '';
    const title = escHtml(m.title || 'Generated Chart');
    const cap = escHtml(m.caption || m.title || '');
    if (m.type === 'svg' && m.svg) {
      const svgUrl = 'data:image/svg+xml;utf8,' + encodeURIComponent(m.svg);
      return `<figure class="k3-figure">${m.svg}<figcaption class="k3-figcaption"><span>${cap}</span><a class="k3-dl-link" href="${svgUrl}" download="clover-diagram.svg">Download SVG</a></figcaption></figure>`;
    }
    if (m.data_url) {
      return `<figure class="k3-figure"><img src="${m.data_url}" alt="${title}"><figcaption class="k3-figcaption"><span>${cap}</span><a class="k3-dl-link" href="${m.data_url}" download="clover-chart.png">Download PNG</a></figcaption></figure>`;
    }
    return '';
  }).join('');
}

function formatBotReplyHtml(text, mediaList) {
  const codeStore = [];
  const withTokens = String(text || '').replace(/```([a-zA-Z0-9_+-]*)\\s*\\n([\\s\\S]*?)```/g, (_, lang, codeBody) => {
    const idx = codeStore.length;
    const lLow = String(lang || '').toLowerCase();
    if (lLow === 'svg' && String(codeBody || '').trim().startsWith('<svg')) {
      codeStore.push('<figure class="k3-figure">' + String(codeBody || '').trim() + '</figure>');
    } else {
      codeStore.push('<div class="k3-code-wrap"><pre class="k3-code"><code>' + escHtml(String(codeBody || '').replace(/\\s+$/, '')) + '</code></pre></div>');
    }
    return '\\n__K3_CODE_BLOCK_' + idx + '__\\n';
  });
  const rawLines = withTokens.split(/\\n+/).map(s => s.trim()).filter(Boolean);
  const mediaHtml = renderMediaItemsHtml(mediaList);
  if (!rawLines.length && !mediaHtml) return '<p>Sorry, no response was generated.</p>';
  const htmlParts = [];
  let inList = false;
  const closeList = () => { if (inList) { htmlParts.push('</ul>'); inList = false; } };
  const formatLeadIn = (s) => {
    const m = s.match(/^([^:]{2,75}):\\s+(.+)$/);
    if (m && m[1].split(/\\s+/).length <= 12 && !/^(https?|here is|here's|note)/i.test(m[1])) {
      return '<strong>' + escHtml(m[1]) + ':</strong> ' + escAndLinkify(m[2]);
    }
    return escAndLinkify(s);
  };
  rawLines.forEach((line, j) => {
    const mCode = line.match(/^__K3_CODE_BLOCK_(\\d+)__$/);
    if (mCode) {
      closeList();
      const cIdx = parseInt(mCode[1], 10);
      if (codeStore[cIdx]) htmlParts.push(codeStore[cIdx]);
      return;
    }
    const isBullet = /^[•\\-*]\\s+/.test(line);
    const cleanLine = line.replace(/^[•\\-*]\\s+/, '').trim();
    const hasLeadIn = /^([^:]{2,75}):\\s+(.+)$/.test(cleanLine) && (j > 0 || rawLines.length > 1) && !/^https?:/i.test(cleanLine);
    if (isBullet || hasLeadIn) {
      if (!inList) { htmlParts.push('<ul>'); inList = true; }
      htmlParts.push('<li>' + formatLeadIn(cleanLine) + '</li>');
    } else {
      closeList();
      if (cleanLine.endsWith(':') && cleanLine.split(/\\s+/).length <= 10 && j > 0) {
        htmlParts.push('<p><strong>' + escHtml(cleanLine) + '</strong></p>');
      } else {
        const sents = cleanLine.split(/(?<=[.!?])\\s+(?=[A-Z0-9])/);
        if (sents.length >= 3) {
          for (let k = 0; k < sents.length; k += 2) {
            htmlParts.push('<p>' + escAndLinkify(sents.slice(k, k + 2).join(' ')) + '</p>');
          }
        } else {
          htmlParts.push('<p>' + escAndLinkify(cleanLine) + '</p>');
        }
      }
    }
  });
  closeList();
  if (mediaHtml) {
    if (htmlParts.length > 0) htmlParts.splice(1, 0, mediaHtml);
    else htmlParts.push(mediaHtml);
  }
  htmlParts.push('<div class="k3-msg-actions"><button type="button" class="k3-action-btn" onclick="speakReply(this)" title="Read aloud">🔊 Listen</button></div>');
  return htmlParts.join('');
}

let isSending = false;
async function sendPrompt() {
  if (isSending) return;
  const text = inputEl.value.trim();
  if (!text && !attachedImageDataUrl && !attachedFiles.length) return;
  isSending = true;
  const activeSid = sessionId;
  const imgData = attachedImageDataUrl;
  const filesPayload = attachedFiles.slice();
  const priorTurns = extractPriorTurns(activeSid);
  inputEl.value = '';
  clearAttachments();
  sendBtn.disabled = true;

  let userHtml = '';
  if (imgData) {
    userHtml += `<img class="chat-thumb" src="${imgData}" alt="Uploaded image">`;
  }
  filesPayload.forEach(f => {
    if (!(f.type || '').startsWith('image/')) {
      userHtml += `<div class="attach-chip">📎 ${escHtml(f.name)}</div>`;
    }
  });
  if (text) {
    userHtml += '<p>' + escHtml(text).replace(/\\n/g, '<br>') + '</p>';
  }

  const effectivePrompt = text || (imgData ? 'Describe this image clearly.' : 'Summarize and explain the attached file.');
  const sObj = sessionsMap[activeSid];
  if (sObj) {
    if (!sObj.messages.length || sObj.title === 'New chat') {
      sObj.title = (text || (filesPayload[0] && filesPayload[0].name) || 'Image analysis').slice(0, 38);
    }
    sObj.messages.push({ role: 'user', html: userHtml, text: effectivePrompt });
    renderSidebar();
    saveSessions();
  }

  if (sessionId === activeSid) addMsg('user', userHtml);
  const pending = (sessionId === activeSid) ? addMsg('bot', CLOVER_LOADER_HTML) : null;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: activeSid,
        prompt: effectivePrompt,
        image_base64: imgData,
        attachments: filesPayload,
        history: priorTurns,
        remember: true
      })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const r = await res.json();
    const reply = r.reply || r.reply_mbr8 || r.reply_1cand || 'Sorry, no response was generated.';
    const botHtml = formatBotReplyHtml(reply, r.media || []);
    if (sessionsMap[activeSid]) {
      sessionsMap[activeSid].messages.push({ role: 'bot', html: botHtml, text: reply });
      saveSessions();
    }
    if (sessionId === activeSid) {
      if (pending && pending.parentNode === chatEl) {
        pending.innerHTML = botHtml;
        if (lastInputWasVoice) {
          lastInputWasVoice = false;
          const lBtn = pending.querySelector('.k3-action-btn');
          if (lBtn) speakReply(lBtn);
        }
      } else renderMessages();
    }
  } catch (err) {
    if (pending && sessionId === activeSid && pending.parentNode === chatEl) pending.innerHTML = '<p>Unable to reach Clover right now. Please try again.</p>';
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

loadSessions();
renderSidebar();
renderMessages();

async function toggleMemoryDrawer() {
  const dr = document.getElementById('mem-drawer');
  if (!dr) return;
  if (dr.hasAttribute('hidden')) {
    dr.removeAttribute('hidden');
    await loadMemoryDrawer();
  } else {
    dr.setAttribute('hidden', '');
  }
}

async function loadMemoryDrawer() {
  const bodyEl = document.getElementById('mem-drawer-body');
  const titleEl = document.getElementById('mem-drawer-title');
  if (!bodyEl) return;
  bodyEl.innerHTML = 'Loading memory tree & graph index...';
  try {
    const r = await fetch('/api/memory/status');
    const d = await r.json();
    const v = d.validation || {};
    const staged = d.staged_entries || [];
    const entries = d.entries || [];
    if (titleEl) titleEl.textContent = `Memory Graph (${entries.length} active, ${v.total_relationships || 0} edges)`;
    let html = `<div style="margin-bottom:0.65rem;color:var(--ink-2);"><strong>Health:</strong> ${v.is_healthy ? 'Healthy' : 'Needs Review'} &bull; <strong>Files:</strong> ${v.total_files || entries.length} &bull; <strong>Edges:</strong> ${v.total_relationships || 0} &bull; <strong>Staged:</strong> ${staged.length}</div>`;
    if (staged.length) {
      html += `<div style="font-weight:700;margin:0.5rem 0 0.35rem;">Staged Candidates Under Review (${staged.length})</div>`;
      staged.forEach(s => {
        html += `<div class="mem-card mem-card--staged">
          <div><strong>${escHtml(s.id)}</strong> <span class="mem-badge">${escHtml(s.entry_type)}</span><span class="mem-badge">${escHtml(s.epistemic_status)}</span></div>
          <div style="font-weight:600;margin:0.2rem 0;">${escHtml(s.title)}</div>
          <div style="color:var(--ink-2);margin-bottom:0.4rem;">${escHtml(s.summary || '')}</div>
          <div style="display:flex;gap:0.4rem;">
            <button type="button" class="btn" onclick="promoteStaged('${escHtml(s.id)}')">Approve &amp; Promote</button>
            <button type="button" class="btn" onclick="discardStaged('${escHtml(s.id)}')">Discard</button>
          </div>
        </div>`;
      });
    }
    html += `<div style="font-weight:700;margin:0.5rem 0 0.35rem;">Active Memory Tree (${entries.length})</div>`;
    entries.forEach(e => {
      const sup = e.superseded_by ? ` &rarr; superseded by ${escHtml(e.superseded_by)}` : '';
      html += `<div class="mem-card">
        <div><strong>${escHtml(e.id)}</strong> <span class="mem-badge">${escHtml(e.entry_type)}</span><span class="mem-badge">${escHtml(e.epistemic_status)}</span><span class="mem-badge">${escHtml(e.status)}${sup}</span></div>
        <div style="font-weight:600;margin:0.18rem 0;">${escHtml(e.title)}</div>
        <div style="color:var(--muted);font-size:0.75rem;">${escHtml(e.rel_path)}</div>
      </div>`;
    });
    bodyEl.innerHTML = html;
  } catch (err) {
    bodyEl.innerHTML = '<p>Unable to load memory graph status.</p>';
  }
}

async function queryMemoryDrawer() {
  const qIn = document.getElementById('mem-query-in');
  const bodyEl = document.getElementById('mem-drawer-body');
  if (!qIn || !bodyEl) return;
  const q = qIn.value.trim();
  if (!q) return loadMemoryDrawer();
  bodyEl.innerHTML = 'Traversing memory graph...';
  try {
    const r = await fetch('/api/memory/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, max_depth: 2, max_entries: 6 })
    });
    const d = await r.json();
    const items = d.items || [];
    let html = `<div style="margin-bottom:0.6rem;"><strong>Retrieved ${items.length} entries</strong> (${d.estimated_tokens_used || 0}/${d.token_budget || 1100} tokens)</div>`;
    items.forEach(it => {
      html += `<div class="mem-card">
        <div><strong>${escHtml(it.id)}</strong> <span class="mem-badge">${escHtml(it.evolution_role)}</span><span class="mem-badge">${escHtml(it.epistemic_status)}</span></div>
        <div style="font-weight:600;margin:0.2rem 0;">${escHtml(it.title)}</div>
        <div style="color:var(--ink-2);margin-bottom:0.25rem;">${escHtml(it.summary)}</div>
        <div style="color:var(--muted);font-size:0.74rem;">Path: ${escHtml(it.relationship_path)}</div>
      </div>`;
    });
    bodyEl.innerHTML = html;
  } catch (err) {
    bodyEl.innerHTML = '<p>Query failed.</p>';
  }
}

async function promoteStaged(entryId) {
  await fetch('/api/memory/promote', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entry_id: entryId, actor: 'web_ui_reviewer' })
  });
  await loadMemoryDrawer();
}

async function discardStaged(entryId) {
  await fetch('/api/memory/discard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entry_id: entryId })
  });
  await loadMemoryDrawer();
}
</script>
</body>
</html>
"""


class PendingTextRequest:
    def __init__(self, session_id, prompt, hist_snapshot, cands, remember, display_prompt=None, attachments_meta=None):
        self.session_id = session_id
        self.prompt = prompt
        self.display_prompt = display_prompt if display_prompt is not None else prompt
        self.attachments_meta = attachments_meta or []
        self.hist_snapshot = hist_snapshot
        self.cands = cands
        self.remember = remember
        self.event = threading.Event()
        self.result = None
        self.error = None


def build_contextual_prompt(user_prompt, hist_snapshot):
    if not hist_snapshot:
        return user_prompt.strip()
    ctx_lines = []
    for turn_idx, h in enumerate(hist_snapshot[-6:], 1):
        ctx_lines.append(f"Turn {turn_idx} User: {h['user']}")
        ctx_lines.append(f"Turn {turn_idx} Assistant: {h['reply']}")
    return (
        "Conversation history (resolve all pronouns like 'it', 'they', 'that', 'there', or follow-up calculations "
        "using these prior turns, and state the resolved entity names and numbers explicitly in W1 and W2):\n"
        + "\n".join(ctx_lines)
        + "\n\nCurrent user request:\n"
        + user_prompt.strip()
    )


class MultiGPUCloverCoordinator:
    def __init__(self, batch_pause_s=BATCH_PAUSE_S):
        t0 = time.perf_counter()
        self.batch_pause_s = float(batch_pause_s)
        self.sess_lock = threading.Lock()
        self.sessions = collections.defaultdict(list)
        self.session_meta = {}
        self.default_clover_context = load_default_clover_context()
        print(
            f"[Clover AI] Loaded default AGENTS.md operating rules & context ({len(self.default_clover_context)} chars) from {CLOVER_DEFAULT_CONTEXT_PATH}",
            flush=True,
        )

        # 1. Load ALL 93 Kimi K3 Trunks (L0..L92) + all 92 MoE Expert Layers + Layer 93 Tail across GPUs 0..5
        # Raising `max_n=1024` (from 256) allows up to 32 windows (1,024 Kimi K3 tokens) or a full 4-to-8-session clubbed batch
        # to execute across all 93 Kimi K3 layers in ONE single pass (~1.3s) instead of 2–4 sequential passes!
        self.k3_experts = K3GPUExpertsRunner.get_instance(max_n=1024, full_93_layers=True, k_res=28)
        if hasattr(self.k3_experts.lib, "k3_full93_set_routing_mode"):
            self.k3_experts.lib.k3_full93_set_routing_mode(2, 1, 2)
        print(
            f"[Clover AI] Loaded ALL 93 Kimi K3 Trunks + 92 MoE Expert Layers (896 experts/layer routable) across GPUs {self.k3_experts.cards} in {self.k3_experts.setup_s}s",
            flush=True,
        )

        # 2. Load Agent Mode (`HybridAgent-8B-Agentic + LiveWebSearch + PythonTool + 93-Layer K3 + NCM`) on `cuda:6` (or `cpu` on CPU servers)
        self.text_workers = []
        self.worker_locks = []
        txt_dev = resolve_device("cuda:6")
        eng = CloverK3AgentEngine(dev_ncm=txt_dev, dev_ha=txt_dev, k3_experts_runner=self.k3_experts)
        self.text_workers.append(eng)
        self.worker_locks.append(threading.Lock())
        self.k3_tok = eng.k3_tok
        print(f"[Clover AI] Loaded Agent+93LayerK3+NCM worker on {txt_dev} in {eng.setup_s}s", flush=True)

        # 3. Load Multimodal Vision Agent (`MoonViT3d + Qwen3-VL-8B-Instruct + HybridLink + 93-Layer K3 + NCM`) on `cuda:7`
        #    (On CPU servers, skip eager loading of the second 8B model unless `CLOVER_ENABLE_MM_WORKER=1`)
        self.mm_lock = threading.Lock()
        self.mm_worker = None
        enable_mm = (torch.cuda.is_available() and os.environ.get("CLOVER_DISABLE_MM_WORKER", "0") != "1") or (
            os.environ.get("CLOVER_ENABLE_MM_WORKER", "0") == "1"
        )
        if enable_mm:
            mm_dev = resolve_device("cuda:7")
            self.mm_worker = K3MultimodalChatEngine(
                dev_ncm=mm_dev, dev_vl=mm_dev, enable_option_a=True, k3_experts_runner=self.k3_experts
            )
            print(f"[Clover AI] Loaded Multimodal Vision+93LayerK3+NCM worker on {mm_dev} in {self.mm_worker.setup_s}s", flush=True)

        self.setup_s = round(time.perf_counter() - t0, 2)
        self._load_sessions_from_disk()
        self.rr_idx = 0
        self.q_lock = threading.Lock()
        self.q_cond = threading.Condition(self.q_lock)
        self.pending_queue = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.text_workers))

        self.batch_thread = threading.Thread(target=self._batch_scheduler_loop, daemon=True)
        self.batch_thread.start()

    def _save_sessions_to_disk_unlocked(self):
        try:
            os.makedirs(os.path.dirname(SESSIONS_STORE_PATH), exist_ok=True)
            serial_meta = {}
            for sid, m in self.session_meta.items():
                serial_meta[sid] = {
                    "session_id": sid,
                    "title": m.get("title", "New chat"),
                    "created_at": m.get("created_at", 0.0),
                    "updated_at": m.get("updated_at", 0.0),
                    "context_tokens": m.get("context_tokens", 0),
                    "max_context_tokens": int(m.get("max_context_tokens") or MAX_SESSION_CONTEXT_TOKENS),
                    "compactions_count": int(m.get("compactions_count", 0)),
                    "tokens_saved_by_summary": int(m.get("tokens_saved_by_summary", 0)),
                    "chunks": [
                        {
                            "source": c.get("source", ""),
                            "text": c.get("text", ""),
                            "tokens": c.get("tokens", 0),
                            "pinned": bool(c.get("pinned", False)),
                        }
                        for c in m.get("chunks", [])[-120:]
                    ],
                }
            data = {
                "sessions": dict(self.sessions),
                "session_meta": serial_meta,
            }
            tmp_p = SESSIONS_STORE_PATH + ".tmp"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_p, SESSIONS_STORE_PATH)
        except Exception:
            pass

    def _load_sessions_from_disk(self):
        if not os.path.exists(SESSIONS_STORE_PATH):
            return
        try:
            with open(SESSIONS_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self.sess_lock:
                for sid, turns in (data.get("sessions") or {}).items():
                    if isinstance(turns, list):
                        for t in turns:
                            if isinstance(t, dict) and "reply" in t:
                                t["reply"] = self._strip_meta_commentary(t.get("reply", ""))
                        self.sessions[sid] = turns
                for sid, m in (data.get("session_meta") or {}).items():
                    if isinstance(m, dict):
                        rebuilt_chunks = []
                        for c in m.get("chunks", []):
                            txt = str(c.get("text", ""))
                            w_set = {w for w in re.findall(r"[a-z0-9]{2,}", txt.lower()) if w not in RETRIEVAL_STOPWORDS}
                            rebuilt_chunks.append({
                                "source": str(c.get("source", "")),
                                "text": txt,
                                "tokens": int(c.get("tokens", 0)),
                                "words": w_set,
                                "pinned": bool(c.get("pinned", False)),
                            })
                        m["chunks"] = rebuilt_chunks
                        self.session_meta[sid] = m
        except Exception:
            pass

    @staticmethod
    def _strip_meta_commentary(reply_txt: str) -> str:
        """Strips any post-response self-commentary ('This response is concise...', 'This is a simple, friendly response...',
        'Your message is straightforward, so a single W1: line is sufficient...') from stored or client-supplied history
        so prior turns never prime the model to repeat meta-commentary."""
        if not reply_txt:
            return ""
        cleaned_paras = []
        for para in str(reply_txt).split("\n\n"):
            p_strip = para.strip()
            if not p_strip:
                continue
            p_low = p_strip.lower()
            if p_low.startswith((
                "this is a simple, friendly response",
                "this response is concise",
                "this response is direct",
                "this response is polite",
                "i'm maintaining a polite",
                "i also included a question to encourage",
                "your message is straightforward",
                "it's also polite and helpful",
                "however, as an ai",
                "i understand that you want me to read",
                "i can only work with the information",
                "since i cannot access",
                "upon reflection, i believe",
                "to be thorough, i'll re-read",
                "upon re-reading",
                "you have provided a link to",
            )) or "aligns with the character description" in p_low or "single `w1:` line is sufficient" in p_low or "cannot browse the internet" in p_low:
                continue
            p_strip = re.sub(
                r"\s*(?:This (?:is a simple, friendly|response is (?:concise|direct|polite|clear|helpful|simple))[\s\S]*$|"
                r"I'm maintaining a polite and warm tone[\s\S]*$|"
                r"I also included a question to encourage[\s\S]*$|"
                r"Your message is straightforward,\s*so a single[\s\S]*$)",
                "",
                p_strip,
                flags=re.IGNORECASE,
            ).strip()
            if p_strip:
                cleaned_paras.append(p_strip)
        return "\n\n".join(cleaned_paras)

    def hydrate_session_history_if_needed(self, session_id: str, client_history: list):
        """Synchronizes `self.sessions[session_id]` with the client's visible conversation history (`client_history`)
        whenever `client_history` is provided and differs from the server's in-memory turn list."""
        if not client_history or not isinstance(client_history, list):
            return
        valid_turns = []
        for item in client_history:
            if isinstance(item, dict):
                u = str(item.get("user") or "").strip()
                r = self._strip_meta_commentary(str(item.get("reply") or "").strip())
                if u and r:
                    valid_turns.append({"user": u, "reply": r, "plans": item.get("plans") or [], "attachments": item.get("attachments") or []})
        if not valid_turns:
            return
        with self.sess_lock:
            existing = self.sessions.get(session_id, [])
            existing_pairs = [(t.get("user", ""), t.get("reply", "")) for t in existing]
            client_pairs = [(t["user"], t["reply"]) for t in valid_turns]
            if existing_pairs != client_pairs:
                meta = self._ensure_session_meta(session_id, first_title=valid_turns[0]["user"])
                # Keep pinned attachment chunks and rebuild turn chunks from valid_turns
                pinned_chunks = [c for c in meta.get("chunks", []) if c.get("pinned", False) or str(c.get("source", "")).startswith("attachment:")]
                meta["chunks"] = pinned_chunks
                meta["context_tokens"] = sum(int(c.get("tokens", 0)) for c in pinned_chunks)
                for idx_t, t in enumerate(valid_turns):
                    turn_txt = f"User: {t['user']}\nAssistant: {t['reply']}"
                    self._index_text_into_session(meta, source=f"turn:{idx_t + 1}", text=turn_txt, pinned=False)
                self.sessions[session_id] = valid_turns
                meta["updated_at"] = time.time()
                self._save_sessions_to_disk_unlocked()

    def _ensure_session_meta(self, session_id: str, first_title: str = "New chat"):
        if session_id not in self.session_meta:
            now = time.time()
            self.session_meta[session_id] = {
                "session_id": session_id,
                "title": first_title[:44] if first_title else "New chat",
                "created_at": now,
                "updated_at": now,
                "context_tokens": 0,
                "max_context_tokens": MAX_SESSION_CONTEXT_TOKENS,
                "compactions_count": 0,
                "tokens_saved_by_summary": 0,
                "chunks": [],
            }
        else:
            self.session_meta[session_id].setdefault("max_context_tokens", MAX_SESSION_CONTEXT_TOKENS)
            self.session_meta[session_id].setdefault("compactions_count", 0)
            self.session_meta[session_id].setdefault("tokens_saved_by_summary", 0)
        return self.session_meta[session_id]

    def _count_k3_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.k3_tok.encode(text))
        except Exception:
            return max(1, int(len(text.split()) * 1.35))

    @staticmethod
    def _simplify_and_summarize_text_block(raw_block: str, target_word_budget: int = 85) -> str:
        """Compresses a large multi-paragraph block into clear, smaller sentences in simple words
        while preserving every key fact, number, identifier, codename, date, and main point."""
        txt = re.sub(r"\s+", " ", str(raw_block or "")).strip()
        if not txt:
            return ""
        # Replace complex/formal phrasing with simple everyday words
        simple_replacements = [
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bfor the purpose of\b", "for"),
            (r"\butilize[sd]?\b", "use"),
            (r"\butilizing\b", "using"),
            (r"\bapproximately\b", "about"),
            (r"\bsubsequently\b", "then"),
            (r"\bfurthermore,?\s*", ""),
            (r"\badditionally,?\s*", ""),
            (r"\bmoreover,?\s*", ""),
            (r"\bconsequently,?\s*", "so "),
            (r"\bnevertheless,?\s*", "still, "),
            (r"\bdemonstrates that\b", "shows"),
            (r"\bindicates that\b", "shows"),
            (r"\bis capable of\b", "can"),
            (r"\bhas the ability to\b", "can"),
            (r"\ba wide range of\b", "many"),
            (r"\ba variety of\b", "many"),
        ]
        for pat, rep in simple_replacements:
            txt = re.sub(pat, rep, txt, flags=re.IGNORECASE)

        # Strip conversational filler prefixes/suffixes
        txt = re.sub(
            r"\b(?:Hi there!|Hello!|Of course!|Sure!|Thank you for asking\.|How can I help you today\?|Feel free to ask[^.!?]*[.!?]|Let me know if you need[^.!?]*[.!?])",
            " ",
            txt,
            flags=re.IGNORECASE,
        )
        txt = re.sub(r"\b(?:User|Assistant):\s*", "", txt)

        raw_sents = [s.strip(" •-*") for s in re.split(r"(?<=[.!?])\s+", txt) if len(s.strip()) >= 8]
        if not raw_sents:
            words = txt.split()
            return " ".join(words[:target_word_budget])

        scored_sents = []
        seen_keys = set()
        for idx, s in enumerate(raw_sents):
            s_clean = re.sub(r"\s*\([^)]{25,}\)", "", s).strip()
            if not s_clean:
                continue
            norm_k = re.sub(r"[^a-z0-9]+", " ", s_clean.lower()).strip()
            if len(norm_k) < 6 or norm_k in seen_keys:
                continue
            seen_keys.add(norm_k)
            w_list = s_clean.split()
            if len(w_list) > 28:
                # Shorten long multi-clause sentence to its core clause
                parts = re.split(r",\s+(?:which|where|while|although|resulting in|allowing)\b", s_clean, maxsplit=1)
                if parts and len(parts[0].split()) >= 6:
                    s_clean = parts[0].rstrip(",; ") + "."
                    w_list = s_clean.split()

            # Score by factual density (numbers, Uppercase/codenames, technical nouns, definitions)
            has_num = len(re.findall(r"\d+", s_clean))
            has_id = len(re.findall(r"\b[A-Z]{2,}[A-Z0-9\-_]*\b", s_clean))
            content_w = [w for w in re.findall(r"[a-z0-9]{3,}", s_clean.lower()) if w not in RETRIEVAL_STOPWORDS]
            score = (has_num * 4.0) + (has_id * 3.5) + min(len(content_w), 12) * 0.6 + (2.0 if idx == 0 else 0.0)
            scored_sents.append((score, idx, s_clean))

        scored_sents.sort(key=lambda x: (-x[0], x[1]))
        chosen = []
        used_words = 0
        for _, idx, s_clean in scored_sents:
            wc = len(s_clean.split())
            if used_words + wc <= target_word_budget or not chosen:
                chosen.append((idx, s_clean))
                used_words += wc
            if used_words >= target_word_budget:
                break

        chosen.sort(key=lambda x: x[0])
        summary_str = "Summary: " + " ".join(s for _, s in chosen)
        return summary_str.strip()

    def _compact_session_when_full(self, meta: dict):
        """Triggered ONLY when `meta['context_tokens'] > meta['max_context_tokens']` (262,144 tokens by default).
        Compresses batches of older chunks into short, simple-word summary sentences (`source='summary:compacted'`)
        until `meta['context_tokens'] <= meta['max_context_tokens']`."""
        max_limit = int(meta.get("max_context_tokens") or MAX_SESSION_CONTEXT_TOKENS)
        if meta["context_tokens"] <= max_limit:
            return

        safety_iters = 0
        while meta["context_tokens"] > max_limit and len(meta["chunks"]) > 1 and safety_iters < 256:
            safety_iters += 1
            # 1. Prefer compacting the oldest un-summarized, unpinned chunks first; if still over limit, compact oldest un-summarized chunks
            candidate_indices = [
                i for i, ch in enumerate(meta["chunks"][:-1])
                if not str(ch.get("source", "")).startswith("summary:") and not ch.get("pinned", False)
            ]
            if not candidate_indices:
                candidate_indices = [
                    i for i, ch in enumerate(meta["chunks"][:-1])
                    if not str(ch.get("source", "")).startswith("summary:")
                ]
            if not candidate_indices:
                candidate_indices = list(range(min(8, len(meta["chunks"]) - 1)))
            if not candidate_indices:
                break

            excess_tokens = meta["context_tokens"] - max_limit
            batch_sz = 16 if excess_tokens > 8000 else 8
            batch_indices = candidate_indices[:batch_sz]
            batch_chunks = [meta["chunks"][i] for i in batch_indices]
            orig_tokens = sum(int(c.get("tokens", 0)) for c in batch_chunks)
            combined_txt = " ".join(str(c.get("text", "")) for c in batch_chunks)
            was_pinned = any(bool(c.get("pinned", False)) for c in batch_chunks)

            # Target ~20% of the original words (between 35 and 95 words) so the summary is significantly smaller
            orig_words = len(combined_txt.split())
            budget_words = max(25, min(90, int(orig_words * 0.22)))
            summary_txt = self._simplify_and_summarize_text_block(combined_txt, target_word_budget=budget_words)
            new_tokens = self._count_k3_tokens(summary_txt)

            # Ensure strictly smaller token count
            if new_tokens >= orig_tokens and orig_tokens > 32:
                short_words = summary_txt.split()[: max(12, orig_tokens // 3)]
                summary_txt = " ".join(short_words) + "."
                new_tokens = self._count_k3_tokens(summary_txt)

            w_set = {w for w in re.findall(r"[a-z0-9]{2,}", summary_txt.lower()) if w not in RETRIEVAL_STOPWORDS}
            # Also retain high-value identifiers/numbers from the original chunks in `words` so retrieval never misses a summarized entity
            for c in batch_chunks:
                for w in c.get("words", set()):
                    if any(ch.isdigit() for ch in w) or len(w) >= 5:
                        w_set.add(w)

            summary_chunk = {
                "source": "summary:compacted",
                "text": summary_txt,
                "tokens": new_tokens,
                "words": w_set,
                "pinned": was_pinned,
            }

            first_pos = batch_indices[0]
            for idx_del in reversed(batch_indices):
                meta["chunks"].pop(idx_del)
            meta["chunks"].insert(first_pos, summary_chunk)

            saved = max(0, orig_tokens - new_tokens)
            meta["context_tokens"] = sum(int(c.get("tokens", 0)) for c in meta["chunks"])
            meta["compactions_count"] = int(meta.get("compactions_count", 0)) + 1
            meta["tokens_saved_by_summary"] = int(meta.get("tokens_saved_by_summary", 0)) + saved
            if saved == 0 and len(meta["chunks"]) > 2:
                removed = meta["chunks"].pop(0)
                meta["context_tokens"] = max(0, meta["context_tokens"] - int(removed.get("tokens", 0)))

    def _index_text_into_session(self, meta: dict, source: str, text: str, pinned: bool = False):
        """Splits `text` into ~180-word (~240-token) chunks and stores them in `meta['chunks']`,
        maintaining up to `MAX_SESSION_CONTEXT_TOKENS = 262,144` tokens per session.
        ONLY when `meta['context_tokens']` exceeds `max_context_tokens` (256K), compacts older paragraphs
        into clear, smaller simple-word summary sentences instead of dropping history."""
        words = text.split()
        if not words:
            return 0
        added_tokens = 0
        step = 180
        # For large documents (>40 chunks), sample exact K3 token ratio on the first chunk to avoid 1,000+ FFI calls
        est_ratio = None
        if len(words) > step * 40:
            first_c = " ".join(words[:step]).strip()
            est_ratio = max(1.05, self._count_k3_tokens(first_c) / float(max(1, len(first_c.split()))))
        for i in range(0, len(words), step):
            chunk_str = " ".join(words[i : i + step]).strip()
            if not chunk_str:
                continue
            if est_ratio is not None and i >= step * 8:
                n_tok = max(1, int(round(len(chunk_str.split()) * est_ratio)))
            else:
                n_tok = self._count_k3_tokens(chunk_str)
            w_set = {w for w in re.findall(r"[a-z0-9]{2,}", chunk_str.lower()) if w not in RETRIEVAL_STOPWORDS}
            meta["chunks"].append({
                "source": source,
                "text": chunk_str,
                "tokens": n_tok,
                "words": w_set,
                "pinned": bool(pinned),
            })
            meta["context_tokens"] += n_tok
            added_tokens += n_tok

        max_limit = int(meta.get("max_context_tokens") or MAX_SESSION_CONTEXT_TOKENS)
        if meta["context_tokens"] > max_limit:
            self._compact_session_when_full(meta)
        return added_tokens

    def ingest_attachments(self, session_id: str, attachments: list) -> list[str]:
        """Extracts text from all uploaded attachments (PDF, DOCX, TXT, MD, CSV, JSON, code, audio)
        and indexes them into the session's 262,144-token context store."""
        extracted_blocks = []
        if not attachments:
            return extracted_blocks
        with self.sess_lock:
            meta = self._ensure_session_meta(session_id)
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                name = str(att.get("name") or "attachment")
                txt = extract_server_attachment_text(att)
                if txt:
                    self._index_text_into_session(meta, source=f"attachment:{name}", text=txt, pinned=True)
                    extracted_blocks.append(txt)
            meta["updated_at"] = time.time()
            self._save_sessions_to_disk_unlocked()
        return extracted_blocks

    def build_prompt_with_256k_context(self, session_id: str, user_prompt: str, current_att_blocks: list[str] | None = None) -> str:
        """Combines current turn attachments + top-k retrieved chunks from the session's 262,144-token store
        so the model can answer questions over 250k+ tokens of documents/history while only processing the relevant slice.
        Requires genuine content-word overlap (or explicit document reference) so unrelated questions in the same session
        are never polluted by prior attachments."""
        u_low = user_prompt.lower()
        q_content_words = {w for w in re.findall(r"[a-z0-9]{2,}", u_low) if w not in RETRIEVAL_STOPWORDS}
        explicit_doc_ref = any(
            kw in u_low
            for kw in ("attached", "attachment", "document", "file", "pdf", "spec", "report", "invoice", "section", "table", "aurora")
        )
        retrieved_strs = []
        with self.sess_lock:
            meta = self.session_meta.get(session_id)
            if meta and meta["chunks"]:
                scored = []
                total_chunks = len(meta["chunks"])
                has_att_or_sum = any(
                    c["source"].startswith(("attachment:", "summary:")) for c in meta["chunks"]
                )
                if has_att_or_sum or total_chunks > 12:
                    for idx, ch in enumerate(meta["chunks"]):
                        is_att = ch["source"].startswith("attachment:")
                        is_sum = ch["source"].startswith("summary:")
                        if not is_att and not is_sum and idx >= total_chunks - 6:
                            continue
                        overlap = len(q_content_words & ch["words"])
                        # Retrieve a chunk if it shares content words with the query, OR if the user explicitly refers to the attached document
                        if overlap >= 1 or ((is_att or is_sum) and explicit_doc_ref):
                            score = overlap * 4 + (3 if (is_att or is_sum) else 0)
                            scored.append((score, idx, ch["text"]))
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    for _, _, c_txt in scored[:4]:
                        retrieved_strs.append(c_txt[:2400])
        parts = []
        if current_att_blocks:
            for blk in current_att_blocks[:3]:
                parts.append(blk[:3200])
        if retrieved_strs:
            parts.append("Relevant context from session documents & summarized history:\n" + "\n---\n".join(retrieved_strs))

        if not parts:
            return user_prompt.strip()
        return (
            "\n\n".join(parts)
            + "\n\nUsing the attached content / session context above, answer the user's request clearly and thoroughly:\n"
            + user_prompt.strip()
        )

    def get_session_history(self, session_id):
        with self.sess_lock:
            return list(self.sessions.get(session_id, []))

    def get_session_details(self, session_id: str) -> dict:
        with self.sess_lock:
            meta = self._ensure_session_meta(session_id)
            turns = list(self.sessions.get(session_id, []))
            summary_chunks = [
                {"source": c.get("source", ""), "text": c.get("text", ""), "tokens": c.get("tokens", 0)}
                for c in meta.get("chunks", [])
                if str(c.get("source", "")).startswith("summary:")
            ]
            return {
                "session_id": session_id,
                "title": meta["title"],
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "turn_count": len(turns),
                "context_tokens": meta["context_tokens"],
                "max_context_tokens": int(meta.get("max_context_tokens") or MAX_SESSION_CONTEXT_TOKENS),
                "compactions_count": int(meta.get("compactions_count", 0)),
                "tokens_saved_by_summary": int(meta.get("tokens_saved_by_summary", 0)),
                "summary_chunks": summary_chunks,
                "turns": [
                    {
                        "user": t.get("user", ""),
                        "reply": t.get("reply", ""),
                        "attachments": t.get("attachments", []),
                    }
                    for t in turns
                ],
            }

    def list_all_sessions(self) -> list[dict]:
        with self.sess_lock:
            items = []
            for sid, meta in self.session_meta.items():
                turns = self.sessions.get(sid, [])
                items.append({
                    "session_id": sid,
                    "title": meta["title"],
                    "turn_count": len(turns),
                    "context_tokens": meta["context_tokens"],
                    "max_context_tokens": MAX_SESSION_CONTEXT_TOKENS,
                    "updated_at": meta["updated_at"],
                })
            items.sort(key=lambda x: x["updated_at"], reverse=True)
            return items

    def reset_session(self, session_id):
        with self.sess_lock:
            self.sessions.pop(session_id, None)
            self.session_meta.pop(session_id, None)
            self._save_sessions_to_disk_unlocked()

    def append_session_turn(self, session_id, user_prompt, reply_txt, plans, attachments_meta=None):
        with self.sess_lock:
            clean_u = user_prompt.strip()
            meta = self._ensure_session_meta(session_id, first_title=clean_u or "Attachment analysis")
            if meta["title"] in ("New chat", "default", "") and clean_u:
                meta["title"] = clean_u[:44]
            meta["updated_at"] = time.time()
            turn_txt = f"User: {clean_u}\nAssistant: {reply_txt}"
            self._index_text_into_session(meta, source=f"turn:{len(self.sessions[session_id]) + 1}", text=turn_txt, pinned=False)
            self.sessions[session_id].append({
                "user": clean_u,
                "reply": reply_txt,
                "plans": plans,
                "attachments": attachments_meta or [],
            })
            self._save_sessions_to_disk_unlocked()
            return len(self.sessions[session_id])

    def _acquire_text_worker(self):
        n = len(self.text_workers)
        for offset in range(n):
            idx = (self.rr_idx + offset) % n
            if self.worker_locks[idx].acquire(blocking=False):
                self.rr_idx = (idx + 1) % n
                return idx
        idx = self.rr_idx % n
        self.rr_idx = (idx + 1) % n
        self.worker_locks[idx].acquire()
        return idx

    def _any_text_worker_free(self) -> bool:
        return any(not lk.locked() for lk in self.worker_locks)

    def _batch_scheduler_loop(self):
        while True:
            with self.q_cond:
                while not self.pending_queue:
                    self.q_cond.wait()
                # Adaptive Zero-Wait Dispatch:
                # - If a GPU text worker is free right now, wait only 25 ms (`0.025 s`) to catch simultaneous bursts,
                #   saving a flat ~1.0 second on every interactive turn!
                # - If all GPU text workers are currently busy executing a batch, accumulate incoming requests for up to
                #   `self.batch_pause_s` (`1.0 s`) so queued sessions are clubbed into the next GPU pass.
                wait_window = 0.025 if self._any_text_worker_free() else self.batch_pause_s
                t_deadline = time.perf_counter() + wait_window
                while True:
                    rem = t_deadline - time.perf_counter()
                    if rem <= 0 or len(self.pending_queue) >= 8:
                        break
                    self.q_cond.wait(timeout=rem)
                batch = self.pending_queue[:8]
                self.pending_queue = self.pending_queue[8:]

            self.executor.submit(self._execute_clubbed_batch, batch)

    def _execute_clubbed_batch(self, batch):
        w_idx = self._acquire_text_worker()
        eng = self.text_workers[w_idx]
        try:
            if len(batch) == 1:
                req = batch[0]
                t0 = time.perf_counter()
                direct_hit = eng.try_direct_k3_ha_ncm_decode(req.prompt, history=req.hist_snapshot)
                if direct_hit is not None:
                    text_mbr, ids_mbr, k3_expert_meta, mean_conf = direct_hit
                    cuda_sync(eng.dev, eng.dev_ha)
                    t_plan_ms = 0.0
                    t_ncm_mbr_ms = (time.perf_counter() - t0) * 1000.0
                    plans = [text_mbr]
                    lat_steps = 11
                    exit_reason = f"direct_ncm_vq_conf_{mean_conf}"
                    tool_calls = []
                else:
                    plans, lat_steps, exit_reason, tool_calls = eng.run_agent_planning(req.prompt, history=req.hist_snapshot)
                    cuda_sync(eng.dev_ha)
                    t_plan_ms = (time.perf_counter() - t0) * 1000.0

                    t1 = time.perf_counter()
                    text_mbr, ids_mbr, k3_expert_meta = eng.render_ncm_with_k3_experts(plans, cands=req.cands)
                    if getattr(eng, "enable_ontology", False):
                        text_mbr = eng.symbolic_verifier.verify_source_grounding(req.display_prompt, text_mbr)
                    cuda_sync(eng.dev, eng.dev_ha)
                    t_ncm_mbr_ms = (time.perf_counter() - t1) * 1000.0

                h_len = len(req.hist_snapshot)
                if req.remember:
                    h_len = self.append_session_turn(
                        req.session_id, req.display_prompt, text_mbr, plans, attachments_meta=req.attachments_meta
                    )
                sess_info = self.get_session_details(req.session_id)
                media_items = []
                for tc in (tool_calls or []):
                    if isinstance(tc, dict) and tc.get("media"):
                        media_items.extend(tc["media"])

                req.result = {
                    "mode": "agent_k3_experts_ncm",
                    "session_id": req.session_id,
                    "session_title": sess_info["title"],
                    "context_tokens": sess_info["context_tokens"],
                    "max_context_tokens": sess_info["max_context_tokens"],
                    "compactions_count": sess_info["compactions_count"],
                    "tokens_saved_by_summary": sess_info["tokens_saved_by_summary"],
                    "prompt": req.display_prompt,
                    "reply": text_mbr,
                    "reply_1cand": text_mbr,
                    "reply_mbr8": text_mbr,
                    "media": media_items,
                    "plans": plans,
                    "tool_calls": tool_calls,
                    "k3_experts": k3_expert_meta,
                    "latent_steps": lat_steps,
                    "exit_reason": exit_reason,
                    "plan_lat_ms": round(t_plan_ms, 1),
                    "ncm_1cand_ms": round(t_ncm_mbr_ms, 1),
                    "ncm_mbr8_ms": round(t_ncm_mbr_ms, 1),
                    "total_1cand_ms": round(t_plan_ms + t_ncm_mbr_ms, 1),
                    "total_mbr8_ms": round(t_plan_ms + t_ncm_mbr_ms, 1),
                    "n_k3_tokens_1cand": len(ids_mbr),
                    "n_k3_tokens_mbr8": len(ids_mbr),
                    "history_len": h_len,
                    "clubbed_batch_size": 1,
                    "gpu_pair": f"{eng.dev} <-> {eng.dev_ha} + K3-Experts(0,1,5,4)",
                }
                req.event.set()
            else:
                t0 = time.perf_counter()
                raw_prompts = [req.prompt for req in batch]
                histories = [req.hist_snapshot for req in batch]
                max_cands = max(req.cands for req in batch)
                outs, ids_list, b_plans, k3_expert_meta, batch_tool_logs = run_agent_batch_with_k3_experts(
                    eng, raw_prompts, histories=histories, cands=max_cands
                )
                cuda_sync(eng.dev, eng.dev_ha)
                t_tot_ms = (time.perf_counter() - t0) * 1000.0

                for i, req in enumerate(batch):
                    h_len = len(req.hist_snapshot)
                    if req.remember:
                        h_len = self.append_session_turn(
                            req.session_id, req.display_prompt, outs[i], b_plans[i], attachments_meta=req.attachments_meta
                        )
                    sess_info = self.get_session_details(req.session_id)
                    media_items_i = []
                    for tc in (batch_tool_logs[i] or []):
                        if isinstance(tc, dict) and tc.get("media"):
                            media_items_i.extend(tc["media"])
                    req.result = {
                        "mode": "agent_k3_experts_ncm",
                        "session_id": req.session_id,
                        "session_title": sess_info["title"],
                        "context_tokens": sess_info["context_tokens"],
                        "max_context_tokens": sess_info["max_context_tokens"],
                        "compactions_count": sess_info["compactions_count"],
                        "tokens_saved_by_summary": sess_info["tokens_saved_by_summary"],
                        "prompt": req.display_prompt,
                        "reply": outs[i],
                        "reply_1cand": outs[i],
                        "reply_mbr8": outs[i],
                        "media": media_items_i,
                        "plans": b_plans[i],
                        "tool_calls": batch_tool_logs[i],
                        "k3_experts": k3_expert_meta,
                        "latent_steps": 14,
                        "exit_reason": "max_steps",
                        "plan_lat_ms": round(t_tot_ms - k3_expert_meta["k3_moe_ms"], 1),
                        "ncm_1cand_ms": round(k3_expert_meta["k3_moe_ms"], 1),
                        "ncm_mbr8_ms": round(k3_expert_meta["k3_moe_ms"], 1),
                        "total_1cand_ms": round(t_tot_ms, 1),
                        "total_mbr8_ms": round(t_tot_ms, 1),
                        "n_k3_tokens_1cand": len(ids_list[i]),
                        "n_k3_tokens_mbr8": len(ids_list[i]),
                        "history_len": h_len,
                        "clubbed_batch_size": len(batch),
                        "gpu_pair": f"{eng.dev} <-> {eng.dev_ha} + K3-Experts(0,1,5,4)",
                    }
                    req.event.set()
        except Exception as e:
            for req in batch:
                req.error = str(e)
                req.event.set()
        finally:
            self.worker_locks[w_idx].release()

    def submit_text_request(self, session_id, prompt, cands=8, remember=True, attachments=None):
        att_blocks = self.ingest_attachments(session_id, attachments) if attachments else []
        att_meta = [
            {"name": str(a.get("name") or "file"), "type": str(a.get("type") or "file")}
            for a in (attachments or [])
            if isinstance(a, dict)
        ]
        effective_prompt = self.build_prompt_with_256k_context(
            session_id, prompt or "Summarize and explain the attached file.", current_att_blocks=att_blocks
        )
        hist_snap = self.get_session_history(session_id) if remember else []
        req = PendingTextRequest(
            session_id,
            effective_prompt,
            hist_snap,
            cands,
            remember,
            display_prompt=prompt or (f"Attached {', '.join(m['name'] for m in att_meta)}" if att_meta else ""),
            attachments_meta=att_meta,
        )
        with self.q_cond:
            self.pending_queue.append(req)
            self.q_cond.notify_all()
        req.event.wait()
        if req.error:
            raise RuntimeError(req.error)
        return req.result

    def submit_image_request(self, session_id, prompt, image_b64, cands=8, remember=True, attachments=None):
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        att_blocks = self.ingest_attachments(session_id, attachments) if attachments else []
        att_meta = [
            {"name": str(a.get("name") or "image"), "type": str(a.get("type") or "image/*")}
            for a in (attachments or [])
            if isinstance(a, dict)
        ]
        if not att_meta:
            att_meta = [{"name": "image", "type": "image/png"}]

        eff_prompt = prompt.strip() or "Describe this image clearly."
        if att_blocks:
            eff_prompt = "\n\n".join(att_blocks[:2]) + "\n\n" + eff_prompt

        hist_snap = self.get_session_history(session_id) if remember else []
        with self.mm_lock:
            if self.mm_worker is None:
                mm_dev = resolve_device("cuda:7")
                self.mm_worker = K3MultimodalChatEngine(
                    dev_ncm=mm_dev, dev_vl=mm_dev, enable_option_a=True, k3_experts_runner=self.k3_experts
                )
            r = self.mm_worker.multimodal_reply(pil_img, eff_prompt, use_option_a=True, cands=cands)

        reply_txt = r["reply_mbr8"] if cands > 1 else r["reply_1cand"]
        h_len = len(hist_snap)
        if remember:
            with self.sess_lock:
                meta = self._ensure_session_meta(session_id, first_title=prompt.strip() or "Image analysis")
                self._index_text_into_session(meta, source="attachment:image_vision", text=f"[Image Vision Analysis]: {reply_txt}", pinned=True)
            h_len = self.append_session_turn(session_id, prompt.strip() or "Attached image", reply_txt, r["plans"], attachments_meta=att_meta)

        sess_info = self.get_session_details(session_id)
        r["session_id"] = session_id
        r["session_title"] = sess_info["title"]
        r["context_tokens"] = sess_info["context_tokens"]
        r["max_context_tokens"] = MAX_SESSION_CONTEXT_TOKENS
        r["reply"] = reply_txt
        r["history_len"] = h_len
        r["gpu_pair"] = "cuda:7 + K3-93Layers(cuda:0..5)"
        return r


STATE = None


ALLOWED_ORIGINS = {
    "https://cloverframework.com",
    "https://www.cloverframework.com",
    "https://api.cloverframework.com",
    "http://127.0.0.1:8765",
    "http://localhost:8765",
    "null",
}
ALLOWED_HOSTS = {
    "api.cloverframework.com",
    "cloverframework.com",
    "www.cloverframework.com",
    "127.0.0.1",
    "127.0.0.1:8765",
    "localhost",
    "localhost:8765",
}


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def _cors_origin(self) -> str:
        origin = (self.headers.get("Origin") or "").strip()
        if origin in ALLOWED_ORIGINS or origin.startswith(("http://127.0.0.1:", "http://localhost:")):
            return origin
        return "https://cloverframework.com"

    def _is_authorized_request(self) -> bool:
        # OpenAI-compatible endpoints (/v1/models, /v1/chat/completions) are called by VS Code desktop / CLI clients
        if self.path.startswith(("/v1/", "/chat/completions", "/models", "/api/health")):
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        if host and host not in ALLOWED_HOSTS:
            return False
        origin = (self.headers.get("Origin") or "").strip()
        if origin and origin not in ALLOWED_ORIGINS and not origin.startswith(("http://127.0.0.1:", "http://localhost:", "vscode-file://", "vscode-webview://")):
            return False
        return True

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Vary", "Origin")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        if not self._is_authorized_request():
            self._send_json({"error": "Forbidden"}, code=403)
            return
        self._send_json({"ok": True})

    def do_GET(self):
        if not self._is_authorized_request():
            self._send_json({"error": "Forbidden"}, code=403)
            return
        if self.path in ("/", "/index.html", "/ai", "/ai/", "/ai/index.html"):
            body = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/favicon.svg", "/favicon.ico"):
            body = FAVICON_SVG.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/health"):
            is_cpu_be = bool(getattr(STATE.k3_experts, "is_cpu_backend", False))
            self._send_json({
                "status": "ready",
                "backend": "cpu" if is_cpu_be else "gpu",
                "mode": "agent_full93_k3_experts_ncm",
                "setup_s": STATE.setup_s,
                "active_sessions": len(STATE.sessions),
                "max_session_context_tokens": MAX_SESSION_CONTEXT_TOKENS,
                "text_worker_pairs": len(STATE.text_workers),
                "k3_trunks_resident": 93,
                "k3_moe_layers_resident": 92,
                "k3_experts_resident": 92 * STATE.k3_experts.k_res,
                "k3_expert_cards": STATE.k3_experts.cards,
                "data_movement_ledger": STATE.k3_experts.get_data_movement_ledger() if hasattr(STATE.k3_experts, "get_data_movement_ledger") else {},
                "multimodal_worker": STATE.mm_worker is not None,
                "batch_pause_s": STATE.batch_pause_s,
                "default_context_file": CLOVER_DEFAULT_CONTEXT_PATH,
                "agents_md_rules_loaded": bool(STATE.default_clover_context),
                "model": (
                    "clover-ai-agent (HybridAgent + Kimi K3 MoE + NCM on CPU)"
                    if is_cpu_be
                    else "clover-ai-agent (HybridAgent + All 93 K3 Trunks & 92 MoE Layers + MoonViT3d + NCM on 8x RTX 3090)"
                ),
            })
        elif self.path.startswith("/api/sessions"):
            self._send_json({
                "max_context_tokens": MAX_SESSION_CONTEXT_TOKENS,
                "sessions": STATE.list_all_sessions(),
            })
        elif self.path.startswith("/api/session"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            sid = (qs.get("session_id") or ["default"])[0]
            self._send_json(STATE.get_session_details(sid))
        elif self.path.startswith("/api/memory/status"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
            else:
                val = mg.last_validation or mg.rebuild_index()
                staged = mg.list_staged_entries()
                self._send_json({
                    "root_dir": str(mg.root_dir),
                    "index_file": str(mg.index_file),
                    "validation": val.to_dict(),
                    "staged_entries": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "rel_path": s.rel_path,
                            "entry_type": s.entry_type,
                            "epistemic_status": s.epistemic_status,
                            "status": s.status,
                            "summary": s.summary,
                        }
                        for s in staged
                    ],
                    "entries": [
                        {
                            "id": e.id,
                            "title": e.title,
                            "rel_path": e.rel_path,
                            "entry_type": e.entry_type,
                            "epistemic_status": e.epistemic_status,
                            "status": e.status,
                            "superseded_by": e.superseded_by,
                        }
                        for e in mg.entries.values()
                    ],
                })
        elif self.path.startswith("/api/memory/staged"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
            else:
                staged = mg.list_staged_entries()
                self._send_json({
                    "count": len(staged),
                    "staged_entries": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "rel_path": s.rel_path,
                            "entry_type": s.entry_type,
                            "epistemic_status": s.epistemic_status,
                            "status": s.status,
                            "summary": s.summary,
                            "content": s.content,
                            "source_evidence": s.source_evidence,
                            "provenance": s.provenance.to_dict(),
                            "relationships": [r.to_dict(include_source=True) for r in s.relationships],
                        }
                        for s in staged
                    ],
                })
        elif self.path.startswith(("/v1/models", "/models")):
            created_ts = int(time.time())
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": "clover-kimi-k3",
                        "name": "Clover",
                        "object": "model",
                        "created": created_ts,
                        "owned_by": "clover-framework",
                        "context_length": MAX_SESSION_CONTEXT_TOKENS,
                        "max_input_tokens": 246144,
                        "max_output_tokens": 16000,
                        "capabilities": {"toolCalling": True, "vision": True, "streaming": True},
                    },
                    {
                        "id": "clover-ai",
                        "name": "Clover",
                        "object": "model",
                        "created": created_ts,
                        "owned_by": "clover-framework",
                        "context_length": MAX_SESSION_CONTEXT_TOKENS,
                        "max_input_tokens": 246144,
                        "max_output_tokens": 16000,
                        "capabilities": {"toolCalling": True, "vision": True, "streaming": True},
                    },
                ],
            })
        else:
            self._send_json({"error": "Not found"}, code=404)

    def do_POST(self):
        if not self._is_authorized_request():
            self._send_json({"error": "Forbidden"}, code=403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        session_id = str(payload.get("session_id") or "default")

        if self.path.startswith("/api/tts"):
            tts_text = str(payload.get("text") or "").strip()
            if not tts_text:
                self._send_json({"error": "Empty text"}, code=400)
                return
            try:
                audio_b64 = synthesize_polite_kid_voice_b64(tts_text, voice=KID_VOICE_NAME)
                self._send_json({"ok": True, "voice": KID_VOICE_NAME, "audio_base64": audio_b64})
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
            return

        if self.path.startswith("/api/reset"):
            STATE.reset_session(session_id)
            self._send_json({"ok": True, "session_id": session_id, "history_len": 0})
            return

        if self.path.startswith("/api/memory/rebuild"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
                return
            rep = mg.rebuild_index(write_index_file=True)
            self._send_json({"ok": True, "validation": rep.to_dict()})
            return

        if self.path.startswith("/api/memory/stage"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
                return
            try:
                from k3_memory_graph import MemoryProvenance
                prov_dict = payload.get("provenance") or {}
                prov = MemoryProvenance(
                    source_type=str(prov_dict.get("source_type") or "conversation"),
                    source_ref=str(prov_dict.get("source_ref") or "api:memory:stage"),
                    actor=str(prov_dict.get("actor") or "user"),
                )
                staged = mg.propose_cycle_memory_update(
                    cycle_kind=payload.get("cycle_kind") or "verified_observation",
                    entry_id=payload.get("id") or None,
                    title=str(payload.get("title") or "").strip(),
                    summary=str(payload.get("summary") or "").strip(),
                    content=str(payload.get("content") or payload.get("summary") or "").strip(),
                    source_evidence=str(payload.get("source_evidence") or "").strip(),
                    provenance=prov,
                    concepts=list(payload.get("concepts") or []),
                    tags=list(payload.get("tags") or []),
                    corrects_id=payload.get("corrects_id") or None,
                    refines_id=payload.get("refines_id") or None,
                    auto_promote_if_verified=bool(payload.get("auto_promote", False)),
                )
                self._send_json({
                    "ok": True,
                    "staged": {
                        "id": staged.id,
                        "title": staged.title,
                        "rel_path": staged.rel_path,
                        "status": staged.status,
                        "epistemic_status": staged.epistemic_status,
                        "relationships": [r.to_dict(include_source=True) for r in staged.relationships],
                    },
                })
            except Exception as e:
                self._send_json({"error": str(e)}, code=400)
            return

        if self.path.startswith("/api/memory/promote"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
                return
            try:
                eid = str(payload.get("entry_id") or payload.get("id") or "").strip()
                actor = str(payload.get("actor") or "human_reviewer").strip()
                note = str(payload.get("verification_note") or "Promoted via /api/memory/promote").strip()
                promoted = mg.promote_staged_entry(entry_id=eid, actor=actor, verification_note=note)
                self._send_json({
                    "ok": True,
                    "promoted": {
                        "id": promoted.id,
                        "title": promoted.title,
                        "rel_path": promoted.rel_path,
                        "status": promoted.status,
                        "epistemic_status": promoted.epistemic_status,
                    },
                    "validation": mg.last_validation.to_dict() if mg.last_validation else {},
                })
            except Exception as e:
                self._send_json({"error": str(e)}, code=400)
            return

        if self.path.startswith("/api/memory/discard"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
                return
            eid = str(payload.get("entry_id") or payload.get("id") or "").strip()
            removed = mg.discard_staged_entry(eid)
            self._send_json({"ok": removed, "entry_id": eid})
            return

        if self.path.startswith("/api/memory/query"):
            eng0 = STATE.text_workers[0]
            mg = getattr(eng0, "memory_graph", None)
            if mg is None:
                self._send_json({"error": "Memory graph not initialized"}, code=500)
                return
            q_str = str(payload.get("query") or payload.get("prompt") or "").strip()
            max_depth = int(payload.get("max_depth", 2))
            max_entries = int(payload.get("max_entries", 6))
            token_budget = int(payload.get("token_budget", 1100))
            pkg = mg.retrieve_context_package(
                query=q_str,
                max_depth=max_depth,
                max_entries=max_entries,
                token_budget=token_budget,
            )
            self._send_json({
                "query": q_str,
                "items": [it.to_dict() for it in pkg.items],
                "evolution_chains": [
                    [asdict(node) for node in ch]
                    for ch in pkg.evolution_chains
                ],
                "active_contradictions": pkg.active_contradictions,
                "estimated_tokens_used": pkg.estimated_tokens_used,
                "token_budget": pkg.token_budget,
                "rendered_context": pkg.render_for_k3(),
            })
            return

        if self.path.startswith("/api/chat"):
            prompt = str(payload.get("prompt", "")).strip()
            image_b64 = payload.get("image_base64")
            attachments = payload.get("attachments") or []
            custom_max_ctx = payload.get("max_context_tokens")
            if custom_max_ctx is not None:
                try:
                    with STATE.sess_lock:
                        meta_c = STATE._ensure_session_meta(session_id)
                        meta_c["max_context_tokens"] = max(256, min(MAX_SESSION_CONTEXT_TOKENS, int(custom_max_ctx)))
                        if meta_c["context_tokens"] > meta_c["max_context_tokens"]:
                            STATE._compact_session_when_full(meta_c)
                except Exception:
                    pass
            client_history = payload.get("history") or payload.get("supplied_history")
            if client_history and isinstance(client_history, list):
                STATE.hydrate_session_history_if_needed(session_id, client_history)
            if not isinstance(attachments, list):
                attachments = []
            # If any attachment is an image/video with data_b64 and image_b64 wasn't set separately, promote it
            non_img_attachments = []
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                m_type = str(att.get("type") or "").lower()
                d_b64 = str(att.get("data_b64") or "")
                if not image_b64 and d_b64 and m_type.startswith("image/"):
                    image_b64 = d_b64
                else:
                    non_img_attachments.append(att)

            try:
                cands = max(1, min(16, int(payload.get("cands", 8))))
            except Exception:
                cands = 8
            remember = bool(payload.get("remember", True))
            if not prompt and not image_b64 and not non_img_attachments:
                self._send_json({"error": "Empty prompt"}, code=400)
                return
            try:
                if image_b64:
                    r = STATE.submit_image_request(
                        session_id,
                        prompt or "Describe this image clearly.",
                        image_b64,
                        cands=cands,
                        remember=remember,
                        attachments=non_img_attachments,
                    )
                else:
                    r = STATE.submit_text_request(
                        session_id,
                        prompt,
                        cands=cands,
                        remember=remember,
                        attachments=non_img_attachments,
                    )
                self._send_json(r)
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
            return

        if self.path.startswith("/api/batch"):
            prompts = [str(p).strip() for p in payload.get("prompts", []) if str(p).strip()]
            cands = int(payload.get("cands", 1))
            if not prompts:
                self._send_json({"error": "Empty prompts list"}, code=400)
                return
            w_idx = STATE._acquire_text_worker()
            eng = STATE.text_workers[w_idx]
            try:
                cuda_sync(eng.dev, eng.dev_ha)
                t0 = time.perf_counter()
                b_plans = plan_batch_with_latent(eng, prompts, lat_steps=14, max_new=100)
                cuda_sync(eng.dev_ha)
                t_plan_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                outs, ids_list = render_ncm_batched_multi_prompt(eng, b_plans, prompts, cands=cands)
                cuda_sync(eng.dev, eng.dev_ha)
                t_ncm_ms = (time.perf_counter() - t1) * 1000.0
            finally:
                STATE.worker_locks[w_idx].release()

            n_tok = sum(len(x) for x in ids_list)
            tot_s = (t_plan_ms + t_ncm_ms) / 1000.0
            self._send_json({
                "n_prompts": len(prompts),
                "n_windows": sum(len(x) for x in b_plans),
                "total_k3_tokens": n_tok,
                "plan_ms": round(t_plan_ms, 1),
                "ncm_ms": round(t_ncm_ms, 1),
                "total_wall_s": round(tot_s, 2),
                "effective_s_per_prompt": round(tot_s / len(prompts), 2),
                "e2e_tok_per_s": round(n_tok / max(tot_s, 1e-6), 2),
                "ncm_stage_tok_per_s": round(n_tok / max(t_ncm_ms / 1000.0, 1e-6), 1),
                "results": [
                    {"prompt": prompts[i], "plans": b_plans[i], "reply": outs[i], "n_k3_tokens": len(ids_list[i])}
                    for i in range(len(prompts))
                ],
            })
            return

        if self.path.startswith(("/v1/chat/completions", "/chat/completions")):
            messages = payload.get("messages", [])
            stream = bool(payload.get("stream", False))
            req_model = str(payload.get("model") or "clover-kimi-k3")
            cands = int(payload.get("cands", 8))
            available_tools = payload.get("tools") or []
            tool_choice = payload.get("tool_choice")
            req_id_hdr = (
                self.headers.get("x-interaction-id")
                or self.headers.get("x-request-id")
                or payload.get("session_id")
                or "vscode_default"
            )
            session_id = "vsc_" + re.sub(r"[^a-zA-Z0-9_\-]", "_", str(req_id_hdr))[:48]

            def _extract_msg_content(m_obj):
                c = m_obj.get("content", "")
                img = None
                if isinstance(c, list):
                    t_parts = []
                    for part in c:
                        if isinstance(part, dict):
                            ptype = str(part.get("type") or "")
                            if ptype in ("text", "input_text"):
                                t_parts.append(str(part.get("text", "")))
                            elif ptype in ("image_url", "input_image"):
                                iu = part.get("image_url") or part.get("image") or {}
                                img = iu.get("url") if isinstance(iu, dict) else str(iu)
                        elif isinstance(part, str):
                            t_parts.append(part)
                    return "\n".join(p for p in t_parts if p).strip(), img
                return str(c or "").strip(), None

            workspace_info_chunks = []

            def _strip_vscode_envelope(u_text: str) -> str:
                """Extracts the clean user request, any <workspace_info>/<environment_info> for agent tool routing,
                and any editor <attachment> blocks (excluding noisy Browser Pages UUIDs)."""
                if not u_text:
                    return ""
                for m_ws in re.findall(r"<(?:workspace_info|environment_info)>\s*([\s\S]*?)\s*</(?:workspace_info|environment_info)>", u_text, flags=re.IGNORECASE):
                    if m_ws.strip():
                        workspace_info_chunks.append(m_ws.strip()[:2200])
                m_ur = re.search(r"<userRequest>\s*([\s\S]*?)\s*</userRequest>", u_text, flags=re.IGNORECASE)
                if m_ur:
                    core_q = m_ur.group(1).strip()
                    att_blocks = []
                    for m_att in re.finditer(r"<attachment\b([^>]*)>\s*([\s\S]*?)\s*</attachment>", u_text, flags=re.IGNORECASE):
                        attrs, body = m_att.group(1), m_att.group(2).strip()
                        if not body or "Browser Pages" in attrs or "Browser Pages" in body[:80] or "browser pages are currently shared" in body[:120]:
                            continue
                        att_blocks.append(body[:2400])
                    if att_blocks:
                        return (
                            "---\n".join(att_blocks[:2])
                            + "\n\nUsing the attached content / session context above, answer the user's request clearly and thoroughly:\n"
                            + core_q
                        )
                    return core_q
                cleaned = re.sub(
                    r"<(?:context|environment_info|workspace_info|reminderInstructions|instructions|toolUseInstructions|editFileInstructions|outputFormatting|memoryInstructions|userMemory|sessionMemory|repoMemory|conversation-summary)>[\s\S]*?</(?:context|environment_info|workspace_info|reminderInstructions|instructions|toolUseInstructions|editFileInstructions|outputFormatting|memoryInstructions|userMemory|sessionMemory|repoMemory|conversation-summary)>",
                    "",
                    u_text,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    r"<attachment\b[^>]*>[\s\S]*?</attachment>",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    r"Relevant context from VS Code editor:\s*(?:The following browser pages[^\r\n]*\s*(?:-\s*[^\r\n]*\s*)*)?",
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                ).strip()
                return cleaned

            user_prompt = ""
            image_b64 = None
            prior_turns = []
            tool_rounds = []
            call_id_to_meta = {}
            pending_u = ""

            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").lower()
                txt_c, img_c = _extract_msg_content(m)
                if img_c:
                    image_b64 = img_c
                if role == "user":
                    cleaned_u = _strip_vscode_envelope(txt_c)
                    if cleaned_u:
                        if not pending_u:
                            # Starting a brand-new user turn after prior turn completed
                            tool_rounds.clear()
                            pending_u = cleaned_u
                            user_prompt = cleaned_u
                        elif not tool_rounds:
                            # Consecutive user messages before any tool call in this turn
                            pending_u = cleaned_u
                            user_prompt = cleaned_u
                elif role == "assistant":
                    t_calls = m.get("tool_calls") or []
                    if isinstance(t_calls, list) and t_calls:
                        for tc in t_calls:
                            if isinstance(tc, dict):
                                cid_tc = str(tc.get("id") or "")
                                fn_tc = tc.get("function") or {}
                                if cid_tc and isinstance(fn_tc, dict):
                                    call_id_to_meta[cid_tc] = {
                                        "name": str(fn_tc.get("name") or ""),
                                        "arguments": str(fn_tc.get("arguments") or "{}"),
                                    }
                    elif pending_u and txt_c:
                        prior_turns.append({"user": pending_u, "reply": txt_c, "plans": []})
                        pending_u = ""
                        tool_rounds.clear()
                elif role == "tool":
                    tc_id = str(m.get("tool_call_id") or "")
                    meta_tc = call_id_to_meta.get(tc_id, {"name": str(m.get("name") or "tool"), "arguments": "{}"})
                    if txt_c:
                        tool_rounds.append({
                            "id": tc_id,
                            "name": meta_tc["name"],
                            "arguments": meta_tc["arguments"],
                            "output": txt_c[:3200],
                        })

            if not user_prompt and messages:
                last_txt, _ = _extract_msg_content(messages[-1])
                user_prompt = _strip_vscode_envelope(last_txt) or "Hello"

            workspace_info_str = "\n\n".join(workspace_info_chunks[-2:]) if workspace_info_chunks else ""

            # Step 1: If VS Code Agent mode supplied `tools` (and `tool_choice != "none"`), run `decide_vscode_agent_step`
            # so Clover Kimi K3 can iteratively call `read_file`, `list_dir`, `file_search`, `grep_search`, `run_in_terminal`, etc.!
            next_tool_call = None
            if available_tools and tool_choice != "none" and not image_b64:
                w_idx = STATE._acquire_text_worker()
                eng = STATE.text_workers[w_idx]
                try:
                    next_tool_call = eng.decide_vscode_agent_step(
                        user_prompt=user_prompt,
                        workspace_info=workspace_info_str,
                        history=prior_turns[-6:],
                        tool_rounds=tool_rounds,
                        available_tools=available_tools,
                    )
                    cuda_sync(eng.dev_ha)
                finally:
                    STATE.worker_locks[w_idx].release()

            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            created_ts = int(time.time())

            if next_tool_call is not None:
                tc_id = "call_" + uuid.uuid4().hex[:16]
                tc_name = next_tool_call["name"]
                tc_args = next_tool_call["arguments"]
                usage_tc = {"prompt_tokens": max(1, len(user_prompt.split())), "completion_tokens": 24, "total_tokens": max(1, len(user_prompt.split())) + 24}
                if stream:
                    role_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
                    }
                    tc_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tc_id,
                                            "type": "function",
                                            "function": {"name": tc_name, "arguments": tc_args},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    stop_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        "usage": usage_tc,
                    }
                    sse_payload = (
                        f"data: {json.dumps(role_chunk)}\n\n"
                        f"data: {json.dumps(tc_chunk)}\n\n"
                        f"data: {json.dumps(stop_chunk)}\n\n"
                        "data: [DONE]\n\n"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(sse_payload)))
                    self.end_headers()
                    self.wfile.write(sse_payload)
                    self.wfile.flush()
                else:
                    self._send_json({
                        "id": cid,
                        "object": "chat.completion",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": tc_id,
                                            "type": "function",
                                            "function": {"name": tc_name, "arguments": tc_args},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": usage_tc,
                    })
                return

            # Step 2: All needed tool rounds have completed (or no tool call needed) -> synthesize final response via 93-Layer K3 + NCM!
            if tool_rounds:
                obs_text = "\n\n".join(
                    f"[Step {idx} — {rd['name']}({rd['arguments']})]:\n{rd['output']}"
                    for idx, rd in enumerate(tool_rounds[-4:], 1)
                )
                effective_user_prompt = (
                    "Verified Workspace Tool Outputs:\n"
                    + obs_text
                    + "\n\nUsing theattached content / session context above, answer the user's request clearly and thoroughly:\n".replace("theattached", "the attached")
                    + user_prompt
                )
            else:
                effective_user_prompt = user_prompt

            if req_id_hdr == "vscode_default":
                first_u = prior_turns[0]["user"] if prior_turns else user_prompt
                session_id = "vsc_" + uuid.uuid5(uuid.NAMESPACE_DNS, first_u).hex[:16]

            with STATE.sess_lock:
                if not prior_turns:
                    STATE.session_meta.pop(session_id, None)
                STATE.sessions[session_id] = prior_turns[-6:]

            if image_b64:
                r = STATE.submit_image_request(session_id, effective_user_prompt or "Describe this image.", image_b64, cands=cands, remember=True)
            else:
                r = STATE.submit_text_request(session_id, effective_user_prompt, cands=cands, remember=True)

            reply_text = r.get("reply") or r.get("reply_mbr8") or ""
            # Append inline markdown image if Clover generated a chart or SVG illustration
            for m_item in (r.get("media") or []):
                if isinstance(m_item, dict) and m_item.get("data_url"):
                    alt_t = m_item.get("alt") or m_item.get("caption") or "Clover Visual"
                    reply_text += f"\n\n![{alt_t}]({m_item['data_url']})"

            n_tok = int(r.get("n_k3_tokens_mbr8") or max(1, len(reply_text.split())))
            p_tok = max(1, len(user_prompt.split()))
            usage_obj = {
                "prompt_tokens": p_tok,
                "completion_tokens": n_tok,
                "total_tokens": p_tok + n_tok,
            }

            if stream:
                role_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                content_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"content": reply_text}, "finish_reason": None}],
                }
                stop_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage_obj,
                }
                sse_payload = (
                    f"data: {json.dumps(role_chunk)}\n\n"
                    f"data: {json.dumps(content_chunk)}\n\n"
                    f"data: {json.dumps(stop_chunk)}\n\n"
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(sse_payload)))
                self.end_headers()
                self.wfile.write(sse_payload)
                self.wfile.flush()
            else:
                self._send_json({
                    "id": cid,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage_obj,
                })
            return

        self._send_json({"error": "Unknown endpoint"}, code=404)


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch-pause", type=float, default=BATCH_PAUSE_S)
    a = ap.parse_args()

    STATE = MultiGPUCloverCoordinator(batch_pause_s=a.batch_pause)
    srv = ThreadingHTTPServer((a.host, a.port), ChatRequestHandler)
    print(f"K3ChatServer (Clover AI 8-GPU Pool) listening on http://{a.host}:{a.port} (setup {STATE.setup_s}s; batch_pause={STATE.batch_pause_s}s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
