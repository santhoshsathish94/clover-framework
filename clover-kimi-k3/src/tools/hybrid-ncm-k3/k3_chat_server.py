"""Persistent Multi-Turn & OpenAI-Compatible Server + Web UI for Clover Kimi K3 (`k3_chat_server.py`).
Keeps `K3FastChatEngine` (and optional `K3OptionAMoonViT`) warm in GPU VRAM on `gpu` (`127.0.0.1:8765`)
so local VS Code (via SSH port-forward `127.0.0.1:8765`) gets:
  1. Zero 13.6s model-load wait on every turn (models stay resident on `cuda:0` + `cuda:2` NVLink pair).
  2. Built-in Interactive Web Chat UI (`GET /`) that opens directly inside VS Code's Simple Browser,
     showing multi-turn conversation memory, `<latent>` steps, `W1/W2` plans, `1`-cand vs `MBR8` toggle,
     clubbed multi-prompt batching, and live `tok/s` metrics.
  3. Standard OpenAI-compatible API (`GET /v1/models`, `POST /v1/chat/completions` with streaming & non-streaming)
     so any VS Code AI extension (Copilot BYOK, Continue, Cline, Roo Code) can connect to `http://127.0.0.1:8765/v1`.
  4. Fast JSON endpoints (`POST /api/chat`, `POST /api/batch`, `POST /api/reset`, `GET /api/health`) for the
     local VS Code terminal client (`k3chat.ps1`).
"""
import argparse, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from k3_fast_chat import K3FastChatEngine
from bench_batch_clubbing import plan_batch_with_latent, render_ncm_batched_multi_prompt


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clover Kimi K3 — VS Code Interactive Chat</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #161f27;
    --border: #263442;
    --text: #e6edf3;
    --muted: #8b9eb0;
    --accent: #12967f;
    --accent-dim: #0a6a5f;
    --user-bg: #1c2c3b;
    --bot-bg: #15222b;
    --code-bg: #0b0f14;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh;
  }
  header {
    padding: 12px 18px; background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  }
  .title { font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 8px; }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 999px; background: rgba(18,150,127,0.18);
    color: #3fe0c5; border: 1px solid rgba(18,150,127,0.4); font-family: ui-monospace, monospace;
  }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
  select, button {
    background: #1d2935; color: var(--text); border: 1px solid var(--border); border-radius: 6px;
    padding: 5px 10px; font-size: 12px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent-dim); border-color: var(--accent); font-weight: 600; }
  button.primary:hover { background: var(--accent); }
  #chat {
    flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 14px;
  }
  .msg {
    max-width: 92%; padding: 12px 15px; border-radius: 10px; border: 1px solid var(--border);
    line-height: 1.5; font-size: 14px;
  }
  .msg.user { align-self: flex-end; background: var(--user-bg); border-color: #2d455c; }
  .msg.bot { align-self: flex-start; background: var(--bot-bg); }
  .meta {
    margin-top: 8px; padding-top: 7px; border-top: 1px dashed var(--border);
    font-size: 11.5px; color: var(--muted); font-family: ui-monospace, monospace;
    display: flex; flex-direction: column; gap: 4px;
  }
  .pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .pill {
    background: var(--code-bg); padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border);
    color: #7ee7d2;
  }
  footer {
    padding: 12px 18px; background: var(--panel); border-top: 1px solid var(--border);
    display: flex; gap: 10px; align-items: flex-end;
  }
  textarea {
    flex: 1; background: var(--code-bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; font-size: 13.5px; font-family: inherit;
    resize: none; min-height: 44px; max-height: 140px;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <div class="title">
    <span>Clover Kimi K3</span>
    <span class="badge">HybridAgent &lt;latent&gt; + e8_k3_direct NCM (163,840 vocab)</span>
    <span class="badge" id="turn-badge">Memory: 0 turns</span>
  </div>
  <div class="controls">
    <label>Decode:
      <select id="mode-select">
        <option value="mbr8" selected>MBR8 (8 candidates)</option>
        <option value="1cand">1-Candidate (Fastest ~155ms NCM)</option>
        <option value="batch">Clubbed Multi-Prompt (1 per line)</option>
      </select>
    </label>
    <label><input type="checkbox" id="remember-chk" checked> Context Memory</label>
    <button type="button" onclick="resetChat()">Reset Memory</button>
  </div>
</header>

<div id="chat">
  <div class="msg bot">
    <strong>Clover Kimi K3 Interactive Chat Ready</strong> (models hot on <code>cuda:0 &harr; cuda:2</code> NVLink pair; 93-layer verification OFF; DSpark OFF).<br>
    Type any prompt below (or multiple prompts, one per line, in <em>Clubbed Multi-Prompt</em> mode). Follow-up questions automatically use multi-turn context memory.
  </div>
</div>

<footer>
  <textarea id="prompt-in" rows="2" placeholder="Ask a question, paste facts for a report, or follow up on the previous turn... (Enter to send, Shift+Enter for newline)"></textarea>
  <button type="button" class="primary" id="send-btn" onclick="sendPrompt()">Send</button>
</footer>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('prompt-in');
const sendBtn = document.getElementById('send-btn');
const turnBadge = document.getElementById('turn-badge');

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

async function resetChat() {
  await fetch('/api/reset', { method: 'POST' });
  turnBadge.textContent = 'Memory: 0 turns';
  addMsg('bot', '<em>Conversation memory cleared.</em>');
}

async function sendPrompt() {
  const text = inputEl.value.trim();
  if (!text) return;
  const mode = document.getElementById('mode-select').value;
  const remember = document.getElementById('remember-chk').checked;
  inputEl.value = '';
  sendBtn.disabled = true;

  addMsg('user', text.replace(/\\n/g, '<br>'));
  const pending = addMsg('bot', '<em>Running &lt;latent&gt; rollout + 11-pass NCM...</em>');

  try {
    if (mode === 'batch') {
      const prompts = text.split('\\n').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompts, cands: 1 })
      });
      const data = await res.json();
      let outHtml = `<strong>Clubbed Batch (${data.n_prompts} prompts, ${data.n_windows} windows):</strong><br><br>`;
      data.results.forEach((r, idx) => {
        outHtml += `<div><strong>[${idx+1}]</strong> ${r.reply} <span style="color:#8b9eb0;font-size:11px">(${r.n_k3_tokens} K3 tok)</span></div><br>`;
      });
      outHtml += `<div class="meta"><div class="pills">
        <span class="pill">Total: ${data.total_wall_s}s (${data.effective_s_per_prompt}s/prompt)</span>
        <span class="pill">E2E: ${data.e2e_tok_per_s} tok/s</span>
        <span class="pill">NCM Stage: ${data.ncm_stage_tok_per_s} tok/s (${data.ncm_ms} ms)</span>
      </div></div>`;
      pending.innerHTML = outHtml;
    } else {
      const cands = (mode === '1cand') ? 1 : 8;
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, cands, remember })
      });
      const r = await res.json();
      turnBadge.textContent = 'Memory: ' + r.history_len + ' turns';
      const replyTxt = (cands === 1) ? r.reply_1cand : r.reply_mbr8;
      const nTok = (cands === 1) ? r.n_k3_tokens_1cand : r.n_k3_tokens_mbr8;
      const ncmMs = (cands === 1) ? r.ncm_1cand_ms : r.ncm_mbr8_ms;
      const totMs = (cands === 1) ? r.total_1cand_ms : r.total_mbr8_ms;
      const e2eTps = (nTok / (totMs / 1000.0)).toFixed(2);
      const ncmTps = (nTok / (ncmMs / 1000.0)).toFixed(1);
      pending.innerHTML = `<div>${replyTxt}</div>
        <div class="meta">
          <div class="pills">
            <span class="pill">Total: ${(totMs/1000).toFixed(2)}s</span>
            <span class="pill">E2E: ${e2eTps} tok/s</span>
            <span class="pill">NCM (${mode.toUpperCase()}): ${ncmMs} ms (${ncmTps} tok/s)</span>
            <span class="pill">Tokens: ${nTok} K3 tok</span>
            <span class="pill">Latent: ${r.latent_steps} steps</span>
          </div>
          <div>Plan: ${(r.plans || []).join(' || ')}</div>
        </div>`;
    }
  } catch (err) {
    pending.innerHTML = '<span style="color:#f85149">Error: ' + err + '</span>';
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}
</script>
</body>
</html>
"""


class ServerState:
    def __init__(self, dev_ncm="cuda:0", dev_ha="cuda:2"):
        self.lock = threading.Lock()
        self.eng = K3FastChatEngine(dev_ncm=dev_ncm, dev_ha=dev_ha)


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
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/health"):
            self._send_json({
                "status": "ready",
                "setup_s": STATE.eng.setup_s,
                "history_len": len(STATE.eng.history),
                "model": "clover-kimi-k3-ncm (e8_k3_direct, 163840 vocab)",
            })
        elif self.path.startswith("/v1/models"):
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": "clover-kimi-k3-ncm",
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

        if self.path.startswith("/api/reset"):
            with STATE.lock:
                STATE.eng.clear_history()
            self._send_json({"ok": True, "history_len": 0})
            return

        if self.path.startswith("/api/chat"):
            prompt = str(payload.get("prompt", "")).strip()
            cands = int(payload.get("cands", 8))
            remember = bool(payload.get("remember", True))
            if not prompt:
                self._send_json({"error": "Empty prompt"}, code=400)
                return
            with STATE.lock:
                r = STATE.eng.chat_reply(prompt, cands=cands, remember=remember)
                r["history_len"] = len(STATE.eng.history)
            self._send_json(r)
            return

        if self.path.startswith("/api/batch"):
            prompts = [str(p).strip() for p in payload.get("prompts", []) if str(p).strip()]
            cands = int(payload.get("cands", 1))
            if not prompts:
                self._send_json({"error": "Empty prompts list"}, code=400)
                return
            with STATE.lock:
                torch.cuda.synchronize(STATE.eng.dev); torch.cuda.synchronize(STATE.eng.dev_ha)
                t0 = time.perf_counter()
                b_plans = plan_batch_with_latent(STATE.eng, prompts, lat_steps=14, max_new=100)
                torch.cuda.synchronize(STATE.eng.dev_ha)
                t_plan_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                outs, ids_list = render_ncm_batched_multi_prompt(STATE.eng, b_plans, prompts, cands=cands)
                torch.cuda.synchronize(STATE.eng.dev); torch.cuda.synchronize(STATE.eng.dev_ha)
                t_ncm_ms = (time.perf_counter() - t1) * 1000.0

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
            prior_turns = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
                if role == "user":
                    user_prompt = str(content)
                elif role == "assistant" and user_prompt:
                    prior_turns.append({"user": user_prompt, "reply": str(content), "plans": []})
                    user_prompt = ""

            if not user_prompt and messages:
                user_prompt = str(messages[-1].get("content", ""))

            with STATE.lock:
                STATE.eng.history = prior_turns[-6:]
                r = STATE.eng.chat_reply(user_prompt, cands=cands, remember=True)
            reply_text = r["reply_mbr8"] if cands > 1 else r["reply_1cand"]
            n_tok = r["n_k3_tokens_mbr8"] if cands > 1 else r["n_k3_tokens_1cand"]
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
                    "model": "clover-kimi-k3-ncm",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
            else:
                self._send_json({
                    "id": cid,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "clover-kimi-k3-ncm",
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dev-ncm", default="cuda:0")
    ap.add_argument("--dev-ha", default="cuda:2")
    a = ap.parse_args()

    STATE = ServerState(dev_ncm=a.dev_ncm, dev_ha=a.dev_ha)
    srv = ThreadingHTTPServer((a.host, a.port), ChatRequestHandler)
    print(f"K3ChatServer listening on http://{a.host}:{a.port} (setup {STATE.eng.setup_s}s; OpenAI /v1/chat/completions + Web UI ready)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
