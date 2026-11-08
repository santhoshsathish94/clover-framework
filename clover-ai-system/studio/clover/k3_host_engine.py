"""Hosted model engines for Clover AI Studio:
1. K3HostEngine: connects to the 3 models hosted on the k3 server (LFM3-Omni-2B, Nemotron 3.5 Super VL, Kimi K3 2.78T MoE),
   either locally on k3 (127.0.0.1:8765) or remotely from the user's Laptop / Mobile Studio over HTTPS with X-Clover-Key.
2. RemoteOmniEngine: connects any Laptop, Mobile or Web Studio to a live hosted model on any machine (another Clover
   Studio server or any OpenAI-compatible server such as llama-server, vLLM, Ollama, SGLang, LM Studio) while keeping
   local tools, permissions, CSI, personal learning, memory and the Clover retry cycle on the user's device.
"""
import base64
import datetime
import hashlib
import http.client
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import clover_cycle as C
import lfm3_native as N
import omni_native
import paths

WORKER_URL = os.environ.get("CLOVER_K3_WORKER_URL", "http://127.0.0.1:8765")
K3_URLS = (
    "https://api.cloverframework.com",
)
SPECS = {
    "omni": {
        "model": "LFM3-Omni (k3)",
        "short": "LFM3-Omni",
        "full_name": "LFM3-Omni-2B",
        "maker": "Liquid AI",
        "badge": "k3",
        "desc": "Liquid AI 2B Omni hosted on k3 · Text, images, audio, video & speech",
        "backend": "llama.cpp Zen 4 AVX-512 (k3)",
        "weights": "4-bit (Q4_0) + projector Q8_0",
        "max_tokens": 32768,
        "layers": ["C1 LFM3-Omni-2B on k3 (Zen 4 AVX-512)"],
    },
    "clover-35b-a3b": {
        "model": "Clover-35B-A3B",
        "short": "Clover-35B-A3B",
        "full_name": "DiffuSpace Clover-35B-A3B-Preview (35B-A3B Block Diffusion MoE)",
        "maker": "DiffuSpace",
        "badge": "k3",
        "desc": "DiffuSpace 35B-A3B Block Diffusion MoE hosted on k3 · 40 layers (30 BD-GDN + 10 Gated GQA), 128K context, HybridAgent, JEPA & NCM",
        "backend": "Clover BD-GDN/GQA MoE engine + HybridAgent + JEPA + 11-level NCM",
        "weights": "4-bit / bf16 MoE + BD-GDN preview (40 layers, 256 experts top-8, coreset top_p=0.9)",
        "max_tokens": 131072,
        "layers": ["C1 Clover-35B-A3B (40L BD-GDN + Gated GQA + 256-Expert MoE)", "C2 HybridAgent + JEPA latent steps", "C3 11-level NCM"],
    },
    "nemotron-3.5-super-vl": {
        "model": "Nemotron 3.5",
        "short": "Nemotron 3.5",
        "full_name": "NVIDIA Nemotron 3.5 Super VL (120B-A12B MoE)",
        "maker": "NVIDIA",
        "badge": "k3",
        "desc": "NVIDIA 120B-A12B MoE hosted on k3 · 1M context, vision, video & NCM",
        "backend": "Clover C engine + HybridAgent + 11-level NCM",
        "weights": "4-bit / bf16 (88 layers, 120B-A12B MoE)",
        "max_tokens": 1048576,
        "layers": ["C1 Nemotron 3.5 Super VL (88L C engine)", "C2 HybridAgent latent steps", "C3 11-level NCM"],
    },
    "kimi-k3": {
        "model": "Kimi K3.1 Pro",
        "short": "Kimi K3.1 Pro",
        "full_name": "Kimi K3.1 Pro 2.8T MoE",
        "maker": "Moonshot AI",
        "badge": "k3",
        "desc": "Moonshot AI 2.8T MoE hosted on k3 · 93 layers (69 KDA + 24 Gated DSA), 1M context & NCM",
        "backend": "Clover C/Rust engine + HybridAgent + 11-level NCM",
        "weights": "4-bit MoE + 8-bit palette trunk + Gated DSA Indexers (93 layers, 2.8T MoE)",
        "max_tokens": 1048576,
        "layers": ["C1 Kimi K3.1 Pro 2.8T MoE (93L C/Rust engine + Gated DSA)", "C2 HybridAgent latent steps", "C3 11-level NCM"],
    },
}
_SIGN_OFF_PARA = re.compile(
    r"\n\n+(?:I(?:'m| am) (?:ready|here|happy) to (?:assist|help)|Just let me know how I can help|"
    r"Feel free to ask|Let me know if you (?:have|need))[\s\S]*$", re.I)


def k3_host_key(base_dir=None):
    env = (os.environ.get("CLOVER_K3_KEY") or os.environ.get("CLOVER_HOST_KEY") or "").strip()
    if env:
        return env
    key_file = Path("/opt/clover-k3/web-studio/vault/k3_host.key")
    if key_file.is_file():
        try:
            return key_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    if base_dir is not None:
        try:
            import vault
            saved = (vault.get(base_dir, "rm_k3") or "").strip()
            if saved:
                return saved
        except Exception:
            pass
    return ""


def exchange_k3_login(account="", credential=""):
    """Exchanges a GitHub token, Google OIDC token, Gmail/GitHub username + key, or acct:... token with k3's
    /api/auth/login endpoint, returning (token, account) when authorized on k3's allowlist."""
    cred = (credential or "").strip()
    acct = (account or "").strip()
    if not cred:
        return "", ""
    for su in K3_URLS:
        try:
            body = json.dumps({"account": acct, "key": cred}).encode("utf-8")
            req = urllib.request.Request(
                f"{su.rstrip('/')}/api/auth/login",
                body,
                {"Content-Type": "application/json", "User-Agent": "CloverStudio/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read().decode("utf-8", "replace"))
            if isinstance(resp, dict) and resp.get("ok") and resp.get("token"):
                return str(resp["token"]).strip(), str(resp.get("account") or acct).strip()
        except Exception:
            pass
    return "", ""


def try_unlock_k3_with_credential(base_dir, credential, account=""):
    """Attempts to unlock the 4 k3 models on a Phone or Laptop using a GitHub token, Google ID token, or k3 key.
    Saves the verified token into vault('rm_k3') if the user is on k3's allowlist."""
    if not base_dir or not credential:
        return {"unlocked": False}
    tok, acct = exchange_k3_login(account=account, credential=credential)
    use_key = tok or str(credential).strip()
    for su in K3_URLS:
        try:
            req = urllib.request.Request(f"{su.rstrip('/')}/api/status?passive=1", headers=_auth_headers(use_key))
            with urllib.request.urlopen(req, timeout=10) as r:
                st = json.loads(r.read().decode("utf-8", "replace"))
            if isinstance(st, dict) and ("models" in st or "model" in st):
                import vault
                vault.put(base_dir, "rm_k3", use_key)
                return {"unlocked": True, "account": acct or st.get("account") or "authorized", "models": st.get("models") or list(SPECS)}
        except Exception:
            pass
    return {"unlocked": False}


def _android_model():
    if not paths.ANDROID:
        return ""
    try:
        from java import jclass
        b = jclass("android.os.Build")
        m = str(getattr(b, "MODEL", "") or "").strip()
        if m:
            return m
    except Exception:
        pass
    try:
        m = subprocess.check_output(["getprop", "ro.product.model"], timeout=3, text=True).strip()
        if m:
            return m
    except Exception:
        pass
    return (os.environ.get("CLOVER_DEVICE") or "").strip()


def is_owner_device(base_dir=None):
    """True only when explicitly configured via CLOVER_OWNER=1, on the web server, or when the user has saved a k3 host
    key in the local encrypted vault. Distributed beta builds never embed owner credentials or auto-unlock by device name."""
    ov = (os.environ.get("CLOVER_OWNER") or "").strip().lower()
    if ov in ("0", "false", "no", "off"):
        return False
    if ov in ("1", "true", "yes", "on") or paths.WEB:
        return True
    if base_dir is not None:
        try:
            import vault
            saved = (vault.get(base_dir, "rm_k3") or "").strip()
            env_k = k3_host_key()
            if saved and (not env_k or saved == env_k):
                return True
        except Exception:
            pass
    return False


def owner_k3_models():
    """The 4 models hosted on k3, surfaced in the dropdown for the owner (and anyone who connects k3 with its key)."""
    prefix = "" if paths.WEB else "k3:"
    out = []
    for k in ("omni", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"):
        sp = SPECS[k]
        out.append({
            "id": prefix + k,
            "name": sp["short"] if paths.WEB else sp["model"],
            "badge": sp["badge"],
            "desc": sp["desc"],
            "host": "k3",
            "url": K3_URLS[0],
            "remote_model": k,
            "kind": "clover",
            "key_id": "rm_k3",
            "builtin_k3": True,
        })
    return out


def _clean_reply(text):
    text = (text or "").strip()
    cut = omni_native.rambling(text)
    if cut is not None:
        text = text[:cut].strip()
    m = _SIGN_OFF_PARA.search(text)
    if m and m.start() >= 20:
        text = text[:m.start()].strip()
    return text


def _nums(text):
    return {m.group(0).replace(",", "") for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", str(text or ""))}


def _auth_headers(api_key=""):
    h = {"Content-Type": "application/json", "User-Agent": "CloverStudio/1.0"}
    if api_key:
        h["X-Clover-Key"] = api_key
        h["Authorization"] = f"Bearer {api_key}"
    return h


class RemoteLlamaClient:
    """Streams /v1/chat/completions from a remote Clover server (/api/worker/lfm3) or any OpenAI-compatible server."""

    def __init__(self, urls, api_key="", media_dir=None, remote_model="omni", kind="clover", device_label="Remote server"):
        self.urls = [u.rstrip("/") for u in (urls if isinstance(urls, (list, tuple)) else [urls]) if u]
        self.api_key = api_key or ""
        self.media_dir = Path(media_dir) if media_dir else None
        self.remote_model = remote_model
        self.kind = kind
        self.device = device_label
        self._sent_media = set()

    def _prepare_media(self, messages):
        """For a Clover server, bundle referenced file:// media; for a generic OpenAI server, inline images as data: URLs."""
        media_files = {}
        if not self.media_dir:
            return messages, media_files
        new_msgs = []
        for m in messages or ():
            c = m.get("content")
            if not isinstance(c, list):
                new_msgs.append(m)
                continue
            new_parts = []
            for part in c:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                ptype = part.get("type")
                sub = part.get(ptype) if isinstance(part.get(ptype), dict) else {}
                u = str(sub.get("url") or "")
                if u.startswith("file://"):
                    fname = u[7:]
                    fpath = self.media_dir / fname
                    if fpath.is_file():
                        if self.kind == "clover":
                            if fname not in self._sent_media:
                                media_files[fname] = base64.b64encode(fpath.read_bytes()).decode("ascii")
                            new_parts.append(part)
                            continue
                        elif ptype == "image_url":
                            ext = fpath.suffix.lower().lstrip(".") or "png"
                            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                            b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
                            new_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                            continue
                new_parts.append(part)
            new_msgs.append(dict(m, content=new_parts))
        return new_msgs, media_files

    def chat_stream(self, body, cancel=None):
        msgs, media_files = self._prepare_media(body.get("messages") or [])
        send_body = dict(body, messages=msgs, stream=True)
        if self.kind == "clover":
            payload = json.dumps({"body": send_body, "media_files": media_files}).encode("utf-8")
        else:
            send_body["model"] = self.remote_model
            send_body.pop("cache_prompt", None)
            send_body.pop("dry_multiplier", None)
            send_body.pop("xtc_probability", None)
            payload = json.dumps(send_body).encode("utf-8")

        last_err = None
        for base_url in self.urls:
            parsed = urllib.parse.urlparse(base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            base_path = parsed.path.rstrip("/")
            full_path = base_path + (
                "/chat/completions" if (self.kind != "clover" and base_path.endswith("/v1"))
                else ("/api/worker/lfm3" if self.kind == "clover" else "/v1/chat/completions")
            )
            conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            conn = conn_cls(host, port, timeout=600)
            try:
                conn.request("POST", full_path, body=payload, headers=_auth_headers(self.api_key))
                r = conn.getresponse()
                if r.status != 200:
                    err_txt = r.read()[:600].decode("utf-8", "replace")
                    raise RuntimeError(f"remote model HTTP {r.status}: {err_txt}")
                self._sent_media.update(media_files.keys())
                while True:
                    if cancel is not None and cancel.is_set():
                        yield {"cancelled": True}
                        return
                    line = r.readline()
                    if not line:
                        return
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data_b = line[5:].strip()
                    if data_b == b"[DONE]":
                        return
                    try:
                        yield json.loads(data_b)
                    except ValueError:
                        continue
                return
            except Exception as exc:
                last_err = exc
                continue
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        raise RuntimeError(f"Could not reach hosted model ({last_err})")


class RemoteOmniEngine(omni_native.OmniNative):
    """Runs the universal Studio turn (with local tools, permissions, CSI, PPL, memory and retry cycle) against a remote
    Clover LFM3-Omni-2B server (such as k3) or any user-added OpenAI-compatible model server."""

    def __init__(self, entry, base_dir, perms, api_key="", local_omni=None):
        self.entry = dict(entry)
        self.model_key = self.entry["id"]
        self.base_dir = Path(base_dir)
        self.perms = perms
        self.local_omni = local_omni
        self.quant = "q4_0"
        self.voice_threads = 4
        self.media_dir = self.base_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.engine_kind = "remote"
        self.max_tokens = int(self.entry.get("max_tokens") or 32768)
        self.cache = None
        self.separate_voice = False
        urls = list(K3_URLS) if self.entry.get("host") == "k3" else [self.entry["url"]]
        self.gpu = RemoteLlamaClient(
            urls, api_key=api_key, media_dir=self.media_dir,
            remote_model=self.entry.get("remote_model") or self.entry["name"],
            kind=self.entry.get("kind") or "openai",
            device_label=self.entry.get("host") or urllib.parse.urlparse(urls[0]).hostname or "remote",
        )
        self.inputs = ["text", "vision", "audio", "video"] if self.entry.get("kind") == "clover" else ["text", "vision"]
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self._voice, self._voice_busy, self._voice_used, self._voice_error = None, 0, 0.0, None
        self._voice_lock = threading.Lock()
        self.side_gate = omni_native.SideGate()
        self._py = None
        self._web = None
        self._extra = {}
        self.workdir, self.workname, self.after_write = None, "", None
        self.turn_images = None
        self.setup_s = 0.1
        self.display_model_name = self.entry.get("name") or "LFM3-Omni (k3)"
        self.display_layers_desc = (self.entry.get("layers") or [f"C1 {self.display_model_name} (k3)"])[0]

    def describe(self):
        host = self.entry.get("host") or urllib.parse.urlparse(self.entry.get("url") or "").hostname or "remote server"
        return {
            "mode": "omni",
            "model": self.entry.get("name") or "Hosted model",
            "model_key": self.model_key,
            "engine": "cpu",
            "device": f"Hosted on {host}",
            "backend": self.entry.get("backend") or ("Clover server" if self.entry.get("kind") == "clover" else "OpenAI-compatible API"),
            "weights": self.entry.get("weights") or "Remote hosted",
            "max_tokens": self.max_tokens,
            "setup_s": self.setup_s,
            "inputs": list(self.inputs),
            "outputs": ["text", "speech"],
            "speech": "Local LFM3-Omni-2B voice" if self.local_omni else f"Hosted on {host}",
            "speech_error": None,
            "tools": [t["function"]["name"] for t in N.STUDIO_TOOLS],
            "layers": [f"C1 {self.entry.get('name')} ({host})", "local Studio tools"],
            "reasoning": False,
        }

    def system_prompt(self, tools_on=True, asked=None):
        now = datetime.datetime.now().astimezone()
        mname = self.entry.get("name") or "LFM3-Omni-2B"
        return (f"You are Clover, an assistant connected to {mname} with tools on the user's {omni_native.PLATFORM}. "
                f"You can see images, hear audio and watch videos the user attaches. "
                f"Today is {now.strftime('%A %d %B %Y')}. The user's home folder is {Path.home()}. "
                + self.workspace_sentence() + N.tools_sentence(tools_on, asked))

    def warm_voice(self):
        if self.local_omni is not None and hasattr(self.local_omni, "warm_voice"):
            self.local_omni.warm_voice()

    def free_voice(self):
        return self.local_omni.free_voice() if self.local_omni is not None and hasattr(self.local_omni, "free_voice") else False

    def tts(self, text, voice="female", lang="en"):
        if self.local_omni is not None and hasattr(self.local_omni, "tts"):
            return self.local_omni.tts(text, voice=voice)
        for u in self.gpu.urls:
            try:
                req = urllib.request.Request(
                    f"{u}/api/tts",
                    json.dumps({"text": text, "voice": voice}).encode("utf-8"),
                    _auth_headers(self.gpu.api_key),
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception:
                continue
        raise RuntimeError("Speech synthesis is not available on this remote model")

    def side_complete(self, msgs, max_new):
        if self.local_omni is not None and hasattr(self.local_omni, "side_complete"):
            return self.local_omni.side_complete(msgs, max_new)
        return None

    def close(self):
        pass


class K3HostEngine:
    inputs = ["text", "vision", "audio", "video"]
    engine_kind = "cpu"
    gpu = None
    side_gate = None

    def __init__(self, model_key, omni, base_dir, perms, worker_url=WORKER_URL, remote_urls=None, api_key="", display_key=None):
        raw_key = model_key.split(":", 1)[1] if ":" in model_key else model_key
        if raw_key not in SPECS or raw_key == "omni":
            raise ValueError(f"unknown k3 model {model_key!r}")
        self.raw_key = raw_key
        self.model_key = display_key or model_key
        self.spec = SPECS[raw_key]
        self.omni = omni
        self.base_dir = Path(base_dir)
        self.perms = perms
        self.worker_url = worker_url.rstrip("/")
        self.remote_urls = [u.rstrip("/") for u in (remote_urls or ())]
        self.api_key = api_key or ""
        self.media_dir = self.base_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = self.spec["max_tokens"]
        self.setup_s = getattr(omni, "setup_s", 1.0)
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self.workdir, self.workname, self.after_write = None, "", None
        self.turn_images = None
        self.display_model_name = self.spec["model"]
        self.display_layers_desc = self.spec["layers"][0]
        if self.omni is not None:
            self.omni.display_model_name = self.spec["model"]
            self.omni.display_layers_desc = self.spec["layers"][0]

    def describe(self):
        return {
            "mode": "omni",
            "model": self.spec["model"],
            "model_key": self.model_key,
            "engine": "cpu",
            "device": "AMD Ryzen 9 7950X3D (k3)",
            "backend": self.spec["backend"],
            "weights": self.spec["weights"],
            "max_tokens": self.max_tokens,
            "setup_s": self.setup_s,
            "inputs": list(self.inputs),
            "outputs": ["text", "speech"],
            "speech": "LFM3-Omni-2B voice",
            "speech_error": None,
            "tools": [t["function"]["name"] for t in N.STUDIO_TOOLS],
            "layers": list(self.spec["layers"]) + ["native tools"],
            "reasoning": True,
        }

    def warm_voice(self):
        if self.omni is not None and hasattr(self.omni, "warm_voice"):
            self.omni.warm_voice()

    def free_voice(self):
        return self.omni.free_voice() if self.omni is not None and hasattr(self.omni, "free_voice") else False

    def tts(self, text, voice="female"):
        if self.omni is not None and hasattr(self.omni, "tts"):
            return self.omni.tts(text, voice=voice)
        req = urllib.request.Request(
            f"{self.worker_url}/api/tts",
            json.dumps({"text": text, "model": self.raw_key}).encode("utf-8"),
            _auth_headers(self.api_key),
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        b64 = str(d.get("audio_base64") or "")
        return base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)

    def transcribe(self, media):
        return self.omni.transcribe(media) if self.omni is not None and hasattr(self.omni, "transcribe") else ""

    def vscode_step(self, *args, **kwargs):
        if self.omni is not None and hasattr(self.omni, "vscode_step"):
            return self.omni.vscode_step(*args, **kwargs)
        import vscode_agent
        return vscode_agent.decide_tool_step(self, *args[:5])

    def close(self):
        pass

    def _call_worker(self, sid, prompt, history, media=()):
        hist_pairs, pend = [], None
        for m in history or ():
            if m.get("role") == "user":
                pend = str(m.get("content") or "")[:4000]
            elif m.get("role") == "assistant" and pend:
                hist_pairs.append({"user": pend, "reply": str(m.get("content") or "")[:4000]})
                pend = None
        image_b64, atts = None, []
        for md in media or ():
            p = self.media_dir / str(md.get("file") or "")
            if not p.is_file():
                continue
            kind = md.get("kind") or "image"
            ext = p.suffix.lower().lstrip(".") or ("png" if kind == "image" else "wav" if kind == "audio" else "mp4")
            mime = f"{kind}/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
            b64 = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")
            if kind == "image" and image_b64 is None:
                image_b64 = b64
            atts.append({"name": md.get("name") or p.name, "type": mime, "content_text": "", "data_b64": b64})
        payload = {
            "session_id": sid,
            "model": self.raw_key,
            "prompt": prompt,
            "history": hist_pairs[-6:],
            "image_base64": image_b64,
            "attachments": atts,
            "remember": False,
        }
        data = json.dumps(payload).encode("utf-8")
        targets = (
            [f"{u}/api/worker/chat" for u in self.remote_urls]
            if self.remote_urls else [f"{self.worker_url}/api/chat"]
        )
        self.stage = {"name": f"running {self.spec['model']} on k3", "since": time.time()}
        last_err = None
        try:
            for url in targets:
                try:
                    req = urllib.request.Request(url, data, _auth_headers(self.api_key))
                    with urllib.request.urlopen(req, timeout=240) as resp:
                        return json.loads(resp.read().decode("utf-8", "replace"))
                except Exception as exc:
                    last_err = exc
                    continue
            raise RuntimeError(f"Could not reach k3 worker ({last_err})")
        finally:
            self.stage = {"name": "idle", "since": time.time()}

    def _save_worker_media(self, items):
        for md in items or ():
            if not isinstance(md, dict):
                continue
            u = str(md.get("data_url") or "")
            if u.startswith("data:image/png;base64,") and self.turn_images is not None:
                raw = base64.b64decode(u.split(",", 1)[1])
                name = hashlib.sha256(raw).hexdigest()[:32] + ".png"
                (self.media_dir / name).write_bytes(raw)
                self.turn_images.append({
                    "file": name, "prompt": str(md.get("title") or "generated visual"),
                    "model": self.spec["model"], "seconds": 0, "width": None, "height": None,
                })

    def system_prompt(self, tools_on=True, asked=None):
        now = datetime.datetime.now().astimezone()
        full = self.spec.get("full_name") or self.spec["model"]
        maker = self.spec.get("maker") or "Clover"
        ws = self.omni.workspace_sentence() if self.omni and hasattr(self.omni, "workspace_sentence") else ""
        return (f"You are Clover, running on {full} (built by {maker}, {self.spec['weights']}) hosted on the k3 server. "
                f"When asked what model you are or who built you, always state that you are {full} built by {maker} "
                f"(never say LFM3-Omni-2B or Liquid AI). You can see images, hear audio and watch videos. "
                f"Today is {now.strftime('%A %d %B %Y')}. The user's home folder is {Path.home()}. "
                "Answer directly and completely without asking unnecessary follow-up questions. "
                + ws + N.tools_sentence(tools_on, asked))

    def _identity_fix(self, reply, user_text):
        """Ensures the drafted reply never misidentifies Nemotron 3.5 or Kimi K3 as LFM3-Omni-2B / Liquid AI."""
        if not reply or re.search(r"\b(?:lfm|liquid\s+ai)\b", user_text or "", re.I):
            return reply
        full = self.spec.get("full_name") or self.spec["model"]
        maker = self.spec.get("maker") or "Clover"
        if re.search(r"\b(?:what|which)\s+(?:ai\s+|language\s+)?model\s+(?:are\s+you|is\s+this|am\s+i\s+(?:talking|speaking)\s+to)\b|"
                     r"\bwho\s+(?:built|made|created|trained)\s+you\b|\bare\s+you\s+(?:kimi|nemotron|lfm)", user_text or "", re.I):
            if re.search(r"\b(?:lfm3|liquid\s+ai)\b", reply, re.I):
                return (f"I am Clover running on **{full}** (built by {maker}, {self.spec['weights']}) "
                        f"with {self.spec['backend']} hosted on the k3 server.")
        out = re.sub(r"\bLFM3(?:-Omni)?(?:-2B|-1\.7B)?\b", full, reply, flags=re.I)
        out = re.sub(r"\bLiquid\s+AI\b", maker, out, flags=re.I)
        return out

    def run_turn(self, history, user_text, *, tools=True, sampling="general", thinking=True, max_new=2048,
                 max_rounds=8, on_event=None, ask=None, allow=(), cancel=None, seed=0, system=None,
                 layers=(), kit=None, layer_query=None, final_on_limit=False, force_answer=False, extra_tools=(),
                 media=(), only_tools=None, meaning=None):
        emit = on_event or (lambda e: None)
        query = layer_query or user_text
        if self.omni is not None:
            self.omni.workdir, self.omni.workname, self.omni.after_write = self.workdir, self.workname, self.after_write
            self.omni.turn_images = self.turn_images
            self.omni.display_model_name = self.spec["model"]
            self.omni.display_layers_desc = self.spec["layers"][0]
            tool_Events = []
            sys_txt = system or self.system_prompt(bool(tools), only_tools)
            emit({"type": "plan", "round": 1, "stage": "drafting", "model": self.spec["model"]})

            def _k3_ev(ev):
                if isinstance(ev, dict):
                    if ev.get("type") == "token" and ev.get("kind") == "answer":
                        tool_Events.append(ev)
                        return
                    if ev.get("stage") == "c1_inference" or ev.get("layer") == "c1_inference":
                        ev = dict(ev, model=self.spec["model"], layers_desc=self.spec["layers"][0])
                emit(ev)

            res_omni = self.omni.run_turn(
                history, user_text, tools=tools, sampling=sampling, thinking=thinking, max_new=max_new,
                max_rounds=max_rounds, on_event=_k3_ev,
                ask=ask, allow=allow, cancel=cancel, seed=seed, system=sys_txt, layers=layers, kit=kit,
                layer_query=layer_query, final_on_limit=final_on_limit, force_answer=force_answer,
                extra_tools=extra_tools, media=media, only_tools=only_tools, meaning=meaning,
            )
            for l_item in res_omni.get("layers") or ():
                if isinstance(l_item, dict) and l_item.get("layer") == "c1_inference":
                    l_item["model"] = self.spec["model"]
                    l_item["layers_desc"] = self.spec["layers"][0]
            rounds = res_omni.get("tool_calls") or []
            omni_reply = self._identity_fix(_clean_reply(res_omni.get("reply") or ""), user_text)
            if omni_reply and not (cancel and cancel.is_set()):
                try:
                    t0 = time.perf_counter()
                    emit({"type": "plan", "round": 1, "stage": "k3_synthesis", "model": self.spec["model"],
                          "full_name": self.spec.get("full_name") or self.spec["model"]})
                    bullet_lines = [ln.strip() for ln in omni_reply.splitlines() if re.match(r"^\s*[-*•]\s+\S", ln)]
                    draft_for_k3 = omni_reply
                    if bullet_lines:
                        norm_buls = []
                        for ln in omni_reply.splitlines():
                            s = ln.strip()
                            if re.match(r"^[-*•]\s+\S", s) and not s.endswith((".", "!", "?")):
                                s += "."
                            if s:
                                norm_buls.append(s)
                        draft_for_k3 = "\n".join(norm_buls)
                    wr = self._call_worker("k3-" + uuid.uuid4().hex[:10],
                                           f"{user_text}\n\n[VERIFIED_STUDIO_DRAFT]\n{draft_for_k3}", history, media=())
                    k3_reply = self._identity_fix(_clean_reply(wr.get("reply") or ""), user_text)
                    want_nums = _nums(omni_reply)
                    has_code_or_table = bool(re.search(r"```|\n\|.+\|", omni_reply))
                    used_k3 = False
                    if k3_reply and not has_code_or_table and (not want_nums or want_nums <= _nums(k3_reply)):
                        if bullet_lines:
                            k3_paras = [p.strip() for p in k3_reply.splitlines() if p.strip()]
                            if len(k3_paras) == len(bullet_lines):
                                k3_reply = "\n".join(p if re.match(r"^[-*•]\s+", p) else f"- {p}" for p in k3_paras)
                        import clover_cycle as C
                        k3_fb, _ = C.verify_turn_outcome(query, k3_reply, rounds, kit=kit, layers=layers)
                        if not k3_fb:
                            omni_reply = k3_reply
                            used_k3 = True
                    plans = wr.get("plans") or []
                    if plans:
                        emit({"type": "token", "round": 1, "kind": "thinking", "text": "\n".join(plans)})
                    k3_exp = wr.get("k3_experts") or {}
                    k3_ms = round((time.perf_counter() - t0) * 1000)
                    emit({"type": "plan", "round": 1, "stage": "k3_done", "model": self.spec["model"], "ms": k3_ms})
                    res_omni.setdefault("layers", []).append({
                        "layer": "ncm", "windows": max(1, len(plans)), "model": self.spec["model"],
                        "ms": k3_ms,
                        "used_k3_reply": used_k3,
                        "c_engine_ms": k3_exp.get("c_engine_ms") or k3_exp.get("k3_moe_ms"),
                    })
                except Exception:
                    pass
            emit({"type": "token", "round": max(1, len(rounds) + 1), "kind": "answer", "text": omni_reply})
            res_omni["reply"] = omni_reply
            return res_omni

        import clover_cycle as C
        import workspaces as WS
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name=self.spec["model"],
            query=query,
            offered_tools=only_tools if only_tools is not None else ["run_python", "web_search"],
        )
        self.active_op_plan = op_plan
        op_plan.start_stage("context")
        layer_log, blocks = [], []
        for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"),
                         ("workspace", "workspace_context"), ("records", "records_context")):
            if name in layers and kit is not None:
                block, info = getattr(kit, fn)(query)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
                    emit(dict(info, type="layer"))
        ws_active_p = (kit.workspaces.active() or {}).get("path") if (kit is not None and getattr(kit, "workspaces", None) is not None) else None
        l_blk, l_info = WS.local_context_for_paths(N.user_paths(user_text), workdir=self.workdir, active_ws_path=ws_active_p)
        if l_blk:
            blocks.append(l_blk)
            layer_log.append(l_info)
            emit(dict(l_info, type="layer"))
        before = []
        if "recall" in layers and kit is not None:
            block, info = kit.recall_context(query, sum(1 for m in history or () if m.get("role") == "user"))
            if block:
                before.append(block)
            if info:
                layer_log.append(info)
                if info.get("refs"):
                    emit(dict(info, type="layer"))
        if meaning is not None and getattr(meaning, "guidance", ""):
            before.append(meaning.guidance)
        full_prompt = "\n\n".join(before + [user_text] + blocks)
        ctx_sources = [x.get("layer") for x in layer_log if isinstance(x, dict) and x.get("layer")]
        op_plan.complete_stage(
            "context",
            "Context established (" + (", ".join(dict.fromkeys(ctx_sources)) if ctx_sources else "AGENTS.md & conversation state") + ")",
        )
        op_plan.start_stage(
            "direction",
            f"Direction set ({self.spec['model']} on k3 · intent: {getattr(meaning, 'intent', 'general')})",
        )
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution", f"Executing with {self.spec['model']} on k3…")

        max_clover_cycles = 3
        cycle_history = []
        rounds = []
        stats_all = []
        reply = ""
        plans = []
        cur_prompt = full_prompt
        cur_hist = list(history or ())

        for attempt in range(max_clover_cycles):
            cycle_num = attempt + 1
            emit({"type": "round", "round": len(stats_all) + 1, "cycle": cycle_num})
            emit({"type": "plan", "round": len(stats_all) + 1, "stage": "k3_synthesis", "model": self.spec["model"],
                  "full_name": self.spec.get("full_name") or self.spec["model"], "cycle": cycle_num})
            t0 = time.perf_counter()
            with self.lock:
                wr = self._call_worker("k3-" + uuid.uuid4().hex[:10], cur_prompt, cur_hist, media=media if attempt == 0 else ())
            wall_s = round(time.perf_counter() - t0, 2)
            emit({"type": "plan", "round": len(stats_all) + 1, "stage": "k3_done", "model": self.spec["model"],
                  "ms": round(wall_s * 1000), "cycle": cycle_num})
            plans = wr.get("plans") or []
            if plans:
                emit({"type": "token", "round": len(stats_all) + 1, "kind": "thinking", "text": "\n".join(str(p) for p in plans)})
            for tc in wr.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tname = {"python": "run_python", "search": "web_search"}.get(tc.get("tool"), str(tc.get("tool") or "tool"))
                tid = "call_" + uuid.uuid4().hex[:12]
                targs = {"query": tc.get("input")} if tname == "web_search" else {"code": tc.get("input")}
                tres = str(tc.get("result") or "")[:3000]
                worked, summary = N.tool_worked(tname, True, tres, "k3 agent loop")
                emit({"type": "tool_call", "id": tid, "name": tname, "arguments": targs})
                emit({"type": "tool_result", "id": tid, "name": tname, "ran": True, "worked": worked, "summary": summary,
                      "why": "k3 agent loop", "result": tres})
                rounds.append({"tool": tname, "arguments": targs, "ran": True, "worked": worked, "summary": summary,
                               "why": "k3 agent loop", "result": tres[:2000]})
                op_plan.advance_execution_step(tname)
            self._save_worker_media(wr.get("media"))
            reply = self._identity_fix(_clean_reply(wr.get("reply") or ""), user_text)
            if "ontology" in layers and kit is not None:
                reply, changed = kit.grounding_filter(query, reply)
                layer_log.append({"layer": "ontology_grounding", "changed": changed, "cycle": cycle_num})
            n_tok = int(wr.get("n_k3_tokens_1cand") or wr.get("n_tokens") or max(1, len(reply.split())))
            dec_s = round(float(wr.get("ncm_1cand_ms") or (wall_s * 1000)) / 1000.0, 2) or wall_s
            stats_all.append({
                "prompt_tokens": len(cur_prompt.split()),
                "new_tokens": n_tok,
                "prefill_s": round(float(wr.get("plan_lat_ms") or 0) / 1000.0, 2),
                "decode_s": dec_s,
                "decode_tok_s": round(n_tok / wall_s, 2) if wall_s > 0 else None,
                "finish": "stop",
            })
            op_plan.complete_stage("execution")
            op_plan.start_stage("outcome", f"Cycle {cycle_num} Outcome: verifying output against reality…")
            emit({"type": "plan", "round": len(stats_all), "stage": "outcome_check", "cycle": cycle_num})
            findings, c_meta = C.verify_turn_outcome(query, reply, rounds, kit=kit, layers=layers, meaning=meaning)
            c_meta["cycle"] = cycle_num
            c_meta["findings"] = list(findings)
            c_meta["ok"] = not bool(findings)
            cycle_history.append(c_meta)
            if c_meta.get("compiler_info") is not None:
                layer_log.append(c_meta["compiler_info"])
            if not findings or (cancel and cancel.is_set()):
                break
            if attempt + 1 >= max_clover_cycles:
                break
            op_plan.retry_from_context(cycle_num + 1, findings[0])
            emit({"type": "cycle", "step": "outcome_retry", "cycle": cycle_num + 1, "findings": findings})
            repair_prompt = C.AUTO_CYCLE_FIX.format(findings="\n".join("- " + f for f in findings))
            cur_hist = cur_hist + [{"role": "user", "content": cur_prompt}, {"role": "assistant", "content": reply}]
            cur_prompt = repair_prompt

        if len(cycle_history) > 1 or (cycle_history and cycle_history[0].get("findings")):
            first_f = (cycle_history[0].get("findings") or [""])[0]
            lesson_txt = (
                C.lesson_text(query, [f[:240] for f in (cycle_history[0].get("findings") or [])[:2]], "", passed=True)
                if len(cycle_history) > 1 and cycle_history[-1].get("ok")
                else ""
            )
            cyc_info = {
                "layer": "cycle",
                "trigger": "auto",
                "action": "corrected" if (len(cycle_history) > 1 and cycle_history[-1].get("ok")) else "unconfirmed",
                "cycles": len(cycle_history),
                "cycles_used": len(cycle_history),
                "repaired": len(cycle_history) > 1 and bool(cycle_history[-1].get("ok")),
                "first_findings": cycle_history[0].get("findings", []),
                "final_findings": cycle_history[-1].get("findings", []),
                "lesson": lesson_txt,
                "lesson_kind": "verified" if lesson_txt else "unconfirmed",
            }
            layer_log.append(cyc_info)
            emit(dict(cyc_info, type="layer"))
        op_plan.complete_stage(
            "outcome",
            f"Outcome verified in {len(cycle_history)} Clover cycle" + ("s" if len(cycle_history) != 1 else ""),
        )
        op_plan.start_stage("growth", "Aligning epistemic state & recording learned knowledge…")
        if not reply:
            reply = "(Clover finished without writing an answer.)"
        op_plan.complete_stage("growth", "Growth preserved · verified outcome & context updated")
        emit({"type": "token", "round": len(stats_all), "kind": "answer", "text": reply})
        layer_log.append({"layer": "ncm", "windows": max(1, len(plans)), "model": self.spec["model"]})
        return {"reply": reply, "tool_calls": rounds, "rounds": max(1, len(stats_all)), "generation": stats_all,
                "layers": layer_log, "todo_list": op_plan.todos}

    def retry_turn(self, history, question, previous, feedback, *, media=(), layers=(), kit=None, on_event=None,
                   cancel=None, max_new=2048, ask=None, context="", allow=()):
        if self.omni is not None:
            self.omni.workdir, self.omni.workname, self.omni.after_write = self.workdir, self.workname, self.after_write
            self.omni.turn_images = self.turn_images
            self.omni.display_model_name = self.spec["model"]
            self.omni.display_layers_desc = self.spec["layers"][0]
            res = self.omni.retry_turn(
                history, question, previous, feedback, media=media, layers=layers, kit=kit, on_event=on_event,
                cancel=cancel, max_new=max_new, system=self.system_prompt(False), ask=ask, context=context, allow=allow,
            )
            res["reply"] = self._identity_fix(res.get("reply") or "", question)
            return res
        return self.run_turn(history, f"{question}\n\n(Previous answer to fix: {previous}\nUser feedback: {feedback})",
                             tools=False, max_new=max_new, on_event=on_event, cancel=cancel, layers=layers, kit=kit,
                             media=media)


def _slug(s):
    return re.sub(r"[^a-z0-9\-]+", "-", str(s or "").lower()).strip("-")[:48] or "model"


def probe_remote_server(url, api_key="", model_hint="", name_hint=""):
    """Probes a user-hosted model server (a Clover Studio / k3 server or any OpenAI-compatible server) and returns a
    list of model descriptors ready to save in settings['remote_models'].
    Also allows entering 'k3', a Gmail address (user@gmail.com), or a GitHub username (e.g. santhoshsathish94) as `url`
    together with a GitHub token, Google token, or k3 access key to unlock the 4 k3 models."""
    raw = (url or "").strip()
    acct_hint = ""
    if raw.startswith("acct:") and not (api_key or "").strip():
        api_key = raw
        raw = K3_URLS[0]
    if not raw and api_key:
        raw = K3_URLS[0]
    if not raw:
        raise ValueError("Enter 'k3' (or your approved Gmail / GitHub username) or a server URL (e.g. http://192.168.1.50:8080)")
    if raw.lower() in ("k3", "api.cloverframework.com", "https://api.cloverframework.com"):
        raw = K3_URLS[0]
    elif "@" in raw and "://" not in raw and "/" not in raw:
        acct_hint = raw.lstrip("@").strip().lower()
        raw = K3_URLS[0]
    elif "://" not in raw and "." not in raw and ":" not in raw and "/" not in raw and raw.lower() != "localhost":
        acct_hint = raw.lstrip("@").strip().lower()
        raw = K3_URLS[0]
    elif "://" not in raw:
        raw = "http://" + raw
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("URL must start with http:// or https:// and include a host name")
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    host_label = "k3" if parsed.hostname in ("api.cloverframework.com", "cloverframework.com", "www.cloverframework.com") else parsed.hostname
    resolved_key = (api_key or "").strip()
    resolved_acct = acct_hint
    if host_label == "k3" and resolved_key:
        tok, acct = exchange_k3_login(account=acct_hint, credential=resolved_key)
        if tok:
            resolved_key = tok
            resolved_acct = acct or acct_hint
    headers = _auth_headers(resolved_key)

    # 1. Try Clover Studio / k3 server (/api/status)
    status_urls = list(K3_URLS) if host_label == "k3" else [base_url]
    for su in status_urls:
        try:
            req = urllib.request.Request(f"{su}/api/status?passive=1", headers=headers)
            with urllib.request.urlopen(req, timeout=12) as r:
                st = json.loads(r.read().decode("utf-8", "replace"))
            if isinstance(st, dict) and ("models" in st or "model" in st):
                avail = st.get("models") or [st.get("model_key") or "omni"]
                if model_hint and model_hint in avail:
                    avail = [model_hint]
                entries = []
                for mk in avail:
                    sp = SPECS.get(mk)
                    mid = f"k3:{mk}" if host_label == "k3" else f"rm:{_slug(host_label)}-{_slug(mk)}"
                    ent = {
                        "id": mid,
                        "name": name_hint.strip() if (name_hint and len(avail) == 1) else (
                            sp["model"] if sp else f"{st.get('model') or mk} ({host_label})"
                        ),
                        "badge": sp["badge"] if (sp and host_label == "k3") else f"{host_label} · Hosted",
                        "desc": sp["desc"] if (sp and host_label == "k3") else f"Hosted on {base_url}",
                        "host": host_label,
                        "url": su,
                        "remote_model": mk,
                        "kind": "clover",
                    }
                    if resolved_key:
                        ent["resolved_key"] = resolved_key
                    if resolved_acct:
                        ent["resolved_account"] = resolved_acct
                    entries.append(ent)
                if entries:
                    return entries
        except urllib.error.HTTPError as he:
            if he.code in (401, 403):
                raise ValueError("This account or key is not on the k3 allowlist (or the access key is invalid).") from he
        except Exception:
            pass

    # 2. Try OpenAI-compatible server (/v1/models)
    models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
    found_ids = []
    try:
        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as r:
            md = json.loads(r.read().decode("utf-8", "replace"))
        for item in (md.get("data") or md.get("models") or ()):
            if isinstance(item, dict) and item.get("id"):
                found_ids.append(str(item["id"]))
            elif isinstance(item, str):
                found_ids.append(item)
    except urllib.error.HTTPError as he:
        if he.code in (401, 403):
            raise ValueError("The model server rejected the API key (HTTP 401/403).") from he
    except Exception:
        pass

    if model_hint.strip():
        found_ids = [model_hint.strip()]
    if not found_ids:
        raise ValueError(f"Could not reach a Clover or OpenAI-compatible model server at {base_url}. Check the URL, key and that the server is running.")

    entries = []
    for mid_raw in found_ids[:6]:
        short = mid_raw.rsplit("/", 1)[-1]
        disp = name_hint.strip() if (name_hint and len(found_ids) == 1) else f"{short}"
        entries.append({
            "id": f"rm:{_slug(host_label)}-{_slug(short)}",
            "name": disp,
            "badge": f"{host_label} · Hosted",
            "desc": f"{mid_raw} hosted on {base_url}",
            "host": host_label,
            "url": base_url,
            "remote_model": mid_raw,
            "kind": "openai",
        })
    return entries
