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
import argparse, base64, collections, concurrent.futures, io, json, os, re, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf/k3_vision")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from PIL import Image
from k3_fast_chat import K3FastChatEngine, CHAT_PLAN_PROMPT
from k3_agent_engine import CloverK3AgentEngine, K3GPUExpertsRunner, run_agent_batch_with_k3_experts
from k3_multimodal_chat import K3MultimodalChatEngine
from bench_batch_clubbing import plan_batch_with_latent, render_ncm_batched_multi_prompt


BATCH_PAUSE_S = 1.0


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clover AI</title>
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
  .actions { display: flex; align-items: center; gap: 0.5rem; }
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
    Clover <span>AI</span>
  </div>
  <div class="actions">
    <button type="button" class="btn" onclick="newSession()">New Session</button>
  </div>
</header>

<div id="chat" aria-live="polite">
  <div class="msg msg--bot"><p>Hi I am Clover AI, How can I help you today?</p></div>
</div>

<div class="preview-bar" id="img-preview-bar" hidden>
  <div class="preview-bar__info">
    <img id="img-preview-thumb" src="" alt="Attached image">
    <span id="img-preview-name">Image attached</span>
  </div>
  <button type="button" class="btn" onclick="clearImage()">Remove</button>
</div>

<footer>
  <input type="file" id="img-input" accept="image/*,video/*" hidden onchange="onImageSelected(event)">
  <button type="button" class="btn" onclick="document.getElementById('img-input').click()" title="Upload an image or video">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
      <circle cx="8.5" cy="8.5" r="1.5"></circle>
      <polyline points="21 15 16 10 5 21"></polyline>
    </svg>
    Image
  </button>
  <textarea id="prompt-in" rows="1" placeholder="Message Clover AI..."></textarea>
  <button type="button" class="btn btn--primary" id="send-btn" onclick="sendPrompt()">Send</button>
</footer>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('prompt-in');
const sendBtn = document.getElementById('send-btn');
const imgInput = document.getElementById('img-input');
const previewBar = document.getElementById('img-preview-bar');
const previewThumb = document.getElementById('img-preview-thumb');
const previewName = document.getElementById('img-preview-name');

let sessionId = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
let attachedImageDataUrl = null;

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

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg msg--' + role;
  d.innerHTML = html;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

function onImageSelected(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function (ev) {
    attachedImageDataUrl = ev.target.result;
    previewThumb.src = attachedImageDataUrl;
    previewName.textContent = file.name;
    previewBar.hidden = false;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  attachedImageDataUrl = null;
  imgInput.value = '';
  previewBar.hidden = true;
}

async function newSession() {
  sessionId = 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  clearImage();
  chatEl.innerHTML = '<div class="msg msg--bot"><p>Hi I am Clover AI, How can I help you today?</p></div>';
}

function formatBotReplyHtml(text) {
  const rawLines = String(text || '').split(/\\n+/).map(s => s.trim()).filter(Boolean);
  if (!rawLines.length) return '<p>Sorry, no response was generated.</p>';
  const htmlParts = [];
  let inList = false;
  const closeList = () => { if (inList) { htmlParts.push('</ul>'); inList = false; } };
  const formatLeadIn = (s) => {
    const m = s.match(/^([^:]{2,75}):\\s+(.+)$/);
    if (m && m[1].split(/\\s+/).length <= 12 && !/^(https?|here is|here's|note)/i.test(m[1])) {
      return '<strong>' + escHtml(m[1]) + ':</strong> ' + escHtml(m[2]);
    }
    return escHtml(s);
  };
  rawLines.forEach((line, j) => {
    const isBullet = /^[•\\-*]\\s+/.test(line);
    const cleanLine = line.replace(/^[•\\-*]\\s+/, '').trim();
    const hasLeadIn = /^([^:]{2,75}):\\s+(.+)$/.test(cleanLine) && (j > 0 || rawLines.length > 1);
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
            htmlParts.push('<p>' + escHtml(sents.slice(k, k + 2).join(' ')) + '</p>');
          }
        } else {
          htmlParts.push('<p>' + escHtml(cleanLine) + '</p>');
        }
      }
    }
  });
  closeList();
  return htmlParts.join('');
}

async function sendPrompt() {
  const text = inputEl.value.trim();
  if (!text && !attachedImageDataUrl) return;
  const imgData = attachedImageDataUrl;
  inputEl.value = '';
  clearImage();
  sendBtn.disabled = true;

  let userHtml = '';
  if (imgData) {
    userHtml += `<img class="chat-thumb" src="${imgData}" alt="Uploaded image">`;
  }
  if (text) {
    userHtml += '<p>' + escHtml(text).replace(/\\n/g, '<br>') + '</p>';
  }
  addMsg('user', userHtml);
  const pending = addMsg('bot', CLOVER_LOADER_HTML);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        prompt: text || 'Describe this image clearly.',
        image_base64: imgData,
        remember: true
      })
    });
    const r = await res.json();
    const reply = r.reply || r.reply_mbr8 || r.reply_1cand || 'Sorry, no response was generated.';
    pending.innerHTML = formatBotReplyHtml(reply);
  } catch (err) {
    pending.innerHTML = 'Unable to reach Clover AI right now. Please try again.';
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}
</script>
</body>
</html>
"""


class PendingTextRequest:
    def __init__(self, session_id, prompt, hist_snapshot, cands, remember):
        self.session_id = session_id
        self.prompt = prompt
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

        # 1. Load ALL 93 Kimi K3 Trunks (L0..L92) + all 92 MoE Expert Layers + Layer 93 Tail across GPUs 0..5
        self.k3_experts = K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=36)
        print(
            f"[Clover AI] Loaded ALL 93 Kimi K3 Trunks + 92 MoE Expert Layers across GPUs {self.k3_experts.cards} in {self.k3_experts.setup_s}s",
            flush=True,
        )

        # 2. Load Agent Mode (`HybridAgent-8B-Agentic + LiveWebSearch + PythonTool + 93-Layer K3 + NCM`) on `cuda:6`
        self.text_workers = []
        self.worker_locks = []
        eng = CloverK3AgentEngine(dev_ncm="cuda:6", dev_ha="cuda:6", k3_experts_runner=self.k3_experts)
        self.text_workers.append(eng)
        self.worker_locks.append(threading.Lock())
        print(f"[Clover AI] Loaded Agent+93LayerK3+NCM worker on cuda:6 in {eng.setup_s}s", flush=True)

        # 3. Load Multimodal Vision Agent (`MoonViT3d + Qwen3-VL-8B-Instruct + HybridLink + 93-Layer K3 + NCM`) on `cuda:7`
        self.mm_lock = threading.Lock()
        self.mm_worker = K3MultimodalChatEngine(
            dev_ncm="cuda:7", dev_vl="cuda:7", enable_option_a=True, k3_experts_runner=self.k3_experts
        )
        print(f"[Clover AI] Loaded Multimodal Vision+93LayerK3+NCM worker on cuda:7 in {self.mm_worker.setup_s}s", flush=True)

        self.setup_s = round(time.perf_counter() - t0, 2)
        self.rr_idx = 0
        self.q_lock = threading.Lock()
        self.q_cond = threading.Condition(self.q_lock)
        self.pending_queue = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.text_workers))

        self.batch_thread = threading.Thread(target=self._batch_scheduler_loop, daemon=True)
        self.batch_thread.start()

    def get_session_history(self, session_id):
        with self.sess_lock:
            return list(self.sessions.get(session_id, []))

    def reset_session(self, session_id):
        with self.sess_lock:
            self.sessions.pop(session_id, None)

    def append_session_turn(self, session_id, user_prompt, reply_txt, plans):
        with self.sess_lock:
            self.sessions[session_id].append({
                "user": user_prompt.strip(),
                "reply": reply_txt,
                "plans": plans,
            })
            if len(self.sessions[session_id]) > 20:
                self.sessions[session_id] = self.sessions[session_id][-20:]
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

    def _batch_scheduler_loop(self):
        while True:
            with self.q_cond:
                while not self.pending_queue:
                    self.q_cond.wait()
                t_deadline = time.perf_counter() + self.batch_pause_s
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
                plans, lat_steps, exit_reason, tool_calls = eng.run_agent_planning(req.prompt, history=req.hist_snapshot)
                torch.cuda.synchronize(eng.dev_ha)
                t_plan_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                text_mbr, ids_mbr, k3_expert_meta = eng.render_ncm_with_k3_experts(plans, cands=req.cands)
                torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
                t_ncm_mbr_ms = (time.perf_counter() - t1) * 1000.0

                h_len = len(req.hist_snapshot)
                if req.remember:
                    h_len = self.append_session_turn(req.session_id, req.prompt, text_mbr, plans)

                req.result = {
                    "mode": "agent_k3_experts_ncm",
                    "session_id": req.session_id,
                    "prompt": req.prompt,
                    "reply": text_mbr,
                    "reply_1cand": text_mbr,
                    "reply_mbr8": text_mbr,
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
                torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
                t_tot_ms = (time.perf_counter() - t0) * 1000.0

                for i, req in enumerate(batch):
                    h_len = len(req.hist_snapshot)
                    if req.remember:
                        h_len = self.append_session_turn(req.session_id, req.prompt, outs[i], b_plans[i])
                    req.result = {
                        "mode": "agent_k3_experts_ncm",
                        "session_id": req.session_id,
                        "prompt": req.prompt,
                        "reply": outs[i],
                        "reply_1cand": outs[i],
                        "reply_mbr8": outs[i],
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

    def submit_text_request(self, session_id, prompt, cands=8, remember=True):
        hist_snap = self.get_session_history(session_id) if remember else []
        req = PendingTextRequest(session_id, prompt, hist_snap, cands, remember)
        with self.q_cond:
            self.pending_queue.append(req)
            self.q_cond.notify_all()
        req.event.wait()
        if req.error:
            raise RuntimeError(req.error)
        return req.result

    def submit_image_request(self, session_id, prompt, image_b64, cands=8, remember=True):
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        hist_snap = self.get_session_history(session_id) if remember else []
        with self.mm_lock:
            r = self.mm_worker.multimodal_reply(pil_img, prompt.strip(), use_option_a=True, cands=cands)

        reply_txt = r["reply_mbr8"] if cands > 1 else r["reply_1cand"]
        h_len = len(hist_snap)
        if remember:
            h_len = self.append_session_turn(session_id, prompt, reply_txt, r["plans"])

        r["session_id"] = session_id
        r["reply"] = reply_txt
        r["history_len"] = h_len
        r["gpu_pair"] = "cuda:7 + K3-93Layers(cuda:0..5)"
        return r


STATE = None


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/health"):
            self._send_json({
                "status": "ready",
                "mode": "agent_full93_k3_experts_ncm",
                "setup_s": STATE.setup_s,
                "active_sessions": len(STATE.sessions),
                "text_worker_pairs": len(STATE.text_workers),
                "k3_trunks_resident": 93,
                "k3_moe_layers_resident": 92,
                "k3_experts_resident": 92 * STATE.k3_experts.k_res,
                "k3_expert_cards": STATE.k3_experts.cards,
                "multimodal_worker": True,
                "batch_pause_s": STATE.batch_pause_s,
                "model": "clover-ai-agent (HybridAgent + All 93 K3 Trunks & 92 MoE Layers + MoonViT3d + NCM on 8x RTX 3090)",
            })
        elif self.path.startswith("/v1/models"):
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": "clover-ai",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "clover-framework",
                    }
                ],
            })
        else:
            self._send_json({"error": "Not found"}, code=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

        session_id = str(payload.get("session_id") or "default")

        if self.path.startswith("/api/reset"):
            STATE.reset_session(session_id)
            self._send_json({"ok": True, "session_id": session_id, "history_len": 0})
            return

        if self.path.startswith("/api/chat"):
            prompt = str(payload.get("prompt", "")).strip()
            image_b64 = payload.get("image_base64")
            cands = int(payload.get("cands", 8))
            remember = bool(payload.get("remember", True))
            if not prompt and not image_b64:
                self._send_json({"error": "Empty prompt"}, code=400)
                return
            try:
                if image_b64:
                    r = STATE.submit_image_request(session_id, prompt or "Describe this image.", image_b64, cands=cands, remember=remember)
                else:
                    r = STATE.submit_text_request(session_id, prompt, cands=cands, remember=remember)
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
                torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
                t0 = time.perf_counter()
                b_plans = plan_batch_with_latent(eng, prompts, lat_steps=14, max_new=100)
                torch.cuda.synchronize(eng.dev_ha)
                t_plan_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                outs, ids_list = render_ncm_batched_multi_prompt(eng, b_plans, prompts, cands=cands)
                torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
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

        if self.path.startswith("/v1/chat/completions"):
            messages = payload.get("messages", [])
            stream = bool(payload.get("stream", False))
            cands = int(payload.get("cands", 8))
            user_prompt = ""
            image_b64 = None
            prior_turns = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, list):
                    txt_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                txt_parts.append(str(part.get("text", "")))
                            elif part.get("type") == "image_url":
                                iu = part.get("image_url", {})
                                image_b64 = iu.get("url") if isinstance(iu, dict) else str(iu)
                    content = " ".join(txt_parts)
                if role == "user":
                    user_prompt = str(content)
                elif role == "assistant" and user_prompt:
                    prior_turns.append({"user": user_prompt, "reply": str(content), "plans": []})
                    user_prompt = ""

            if not user_prompt and messages:
                user_prompt = str(messages[-1].get("content", ""))

            with STATE.sess_lock:
                STATE.sessions[session_id] = prior_turns[-6:]

            if image_b64:
                r = STATE.submit_image_request(session_id, user_prompt or "Describe this image.", image_b64, cands=cands, remember=True)
            else:
                r = STATE.submit_text_request(session_id, user_prompt, cands=cands, remember=True)

            reply_text = r.get("reply") or r.get("reply_mbr8") or ""
            n_tok = r.get("n_k3_tokens_mbr8", 0)
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "clover-ai",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
            else:
                self._send_json({
                    "id": cid,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "clover-ai",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": len(user_prompt.split()), "completion_tokens": n_tok, "total_tokens": len(user_prompt.split()) + n_tok},
                })
            return

        self._send_json({"error": "Unknown endpoint"}, code=404)


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch-pause", type=float, default=BATCH_PAUSE_S)
    a = ap.parse_args()

    STATE = MultiGPUCloverCoordinator(batch_pause_s=a.batch_pause)
    srv = ThreadingHTTPServer((a.host, a.port), ChatRequestHandler)
    print(f"K3ChatServer (Clover AI 8-GPU Pool) listening on http://{a.host}:{a.port} (setup {STATE.setup_s}s; batch_pause={STATE.batch_pause_s}s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
