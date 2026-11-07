"""Clover AI Studio server: Clover with LFM3-1.7B on this laptop, bound to 127.0.0.1 only.

  /                      Studio web UI (web/app.html)
  /api/health  /api/status  /api/permissions  /api/chat  /api/session/forget  /api/shutdown      (Studio)
  /v1/models  /v1/chat/completions                                             (OpenAI-compatible, VS Code)

Tools outside the model (running code, internet) run only with the user's permission (permissions.py).
Requests from other browser origins are refused (Host, Origin and Sec-Fetch-Site checks; JSON-only POST), so a web
page open in a browser cannot drive this server or change its permissions.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stand_ins  # noqa: E402
stand_ins.install(HERE)
import paths  # noqa: E402
import semantics  # noqa: E402

MODEL_ID = "lfm3-omni-2b"
STUDIO_VERSION = "5.0.0"
RELEASE_CHANNEL = "release"
PRIVACY_URL = "https://cloverframework.com/privacy/"
SUPPORT_URL = "https://github.com/santhoshsathish94/clover-framework/issues"
MAX_BODY = 4 * 1024 * 1024
CHAT_BODY = 100 * 1024 * 1024  # streamed chat carries images, audio and video as base64
MEDIA_LIMITS = {"image": 15 << 20, "audio": 30 << 20, "video": 60 << 20}
MEDIA_ITEMS = 6
MAX_SESSIONS = 200
ATTACH_FILES, ATTACH_CHARS = 2, 2400  # the caps clover-chat-server.py applies to <attachment> blocks
ATTACH_MARKER = "\n\nUsing the attached content / session context above, answer the user's request clearly and thoroughly:\n"
DATA_URL_RE = re.compile(r"^data:image/(?:png|jpeg|gif|webp|svg\+xml);base64,[A-Za-z0-9+/=]+$")
# files of the Studio page (clover/web, built by studio/build_assets.py from the Clover icon pack)
STATIC = {"/icon.png": ("icon.png", "image/png"), "/favicon.ico": ("icon.png", "image/png"),
          "/favicon.svg": ("favicon.svg", "image/svg+xml"), "/robo-clover.webp": ("robo-clover.webp", "image/webp")}


def setup_stdio(log_path):
    """pythonw.exe has no console (sys.stdout is None): send all output to a log file."""
    if sys.stdout is None or sys.stderr is None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class S:
    engine = None
    omni_engine = None
    k3_engines = {}
    remote_engines = {}
    mode = "lfm3"
    load = {"status": "loading", "step": "starting", "since": time.time(), "error": None}
    perms = None
    settings = None
    search = None
    workspaces = None
    github = None
    vscode = None
    repo = None
    kit = None
    memidx = None
    csi = None  # csi_child.CSIService
    google_identity = None  # google_identity.GoogleIdentityManager
    gateway_runtime = None  # {"gateway": CollectiveGateway, "evaluator": CollectiveEvaluator, "registry": CoreRegistry}
    hardware = None  # hardware.probe(): this machine and the settings chosen for it
    last_turn_end = time.time()
    kv_sid = None
    args = None
    sessions = {}
    chat_dirs = {}  # chat id -> the folder its commands run in (the last repository cloned in that chat)
    sess_lock = threading.Lock()
    turn_lock = threading.Lock()
    turns_lock = threading.Lock()
    active = {}  # turn_id -> {"cancel": Event, "pending": {request_id: {"event": Event, "decision": str}}, "seq": int, "events": list}
    recent_turns = {}  # turn_id -> completed turn state for SSE reconnection replay (bounded to 32)
    busy = None
    port = 8765
    started = time.time()
    last_request = time.time()
    last_v1 = 0.0
    turns = 0
    httpd = None
    token = ""  # the phone app's per-launch secret (CLOVER_TOKEN): other apps on the phone reach 127.0.0.1 too
    setup = None  # the phone app's first-start download (Setup)
    setup_hook = None  # the phone app's notification: fn(text, percent)


class EngineNotReady(Exception):
    pass


LAYER_KEYS = ("memory", "ontology", "compiler", "ncm", "kv_slots", "recall")
IMAGE_MODELS = ("sd15-lcm", "sd15")  # image_gen.presets; the default is the best measured
# gpu / cpu: LFM3-Omni-2B on llama.cpp; fused: C1 + C2 + C3, the phone's fused runtime on the CPU (fused_engine.py)
ENGINES = ("gpu", "cpu", "fused")
WEB_MODELS = ("omni", "nemotron-3.5-super-vl", "kimi-k3")
MODELS = WEB_MODELS if paths.WEB else ("omni",)


def _norm_model(val):
    s = str(val or "").strip()
    return "omni" if s in ("omni", "lfm3-omni-2b", "lfm3-1.7b") else s


def _clean_remote_models(items):
    out, seen = [], set()
    for it in (items if isinstance(items, list) else ()):
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or "").strip()
        name = str(it.get("name") or "").strip()
        url = str(it.get("url") or "").strip()
        if not mid or not name or not url or mid in seen:
            continue
        seen.add(mid)
        out.append({
            "id": mid,
            "name": name[:80],
            "badge": str(it.get("badge") or "Hosted")[:40],
            "desc": str(it.get("desc") or f"Hosted on {url}")[:240],
            "host": str(it.get("host") or "")[:60],
            "url": url[:240],
            "remote_model": str(it.get("remote_model") or name)[:120],
            "kind": "clover" if it.get("kind") == "clover" else "openai",
            "key_id": str(it.get("key_id") or "")[:60],
        })
    return out


def model_catalog(remote_models=None, base=None):
    """All selectable models for this Studio instance:
    - On Laptop & Mobile: local LFM3-Omni-2B is always first and the default; on owner devices (or when unlocked),
      the 3 models hosted on k3 follow; then any user-added remote models.
    - On Web (k3): the 3 models hosted on k3, plus any user-added remote models."""
    import k3_host_engine
    base_dir = Path(base) if base else (Path(S.args.base) if S.args else None)
    rm = _clean_remote_models(
        remote_models if remote_models is not None else ((S.settings.get().get("remote_models") or []) if S.settings else [])
    )
    if paths.WEB:
        cat = list(k3_host_engine.owner_k3_models())
    else:
        badge = "This phone" if paths.ANDROID else "This laptop"
        dev_name = "this phone" if paths.ANDROID else "this laptop"
        cat = [{
            "id": "omni",
            "name": "LFM3-Omni",
            "badge": badge,
            "desc": f"Local · 24-layer Liquid AI Omni on {dev_name} · Text, images, audio, video & speech",
            "host": "local",
            "local": True,
        }]
        if k3_host_engine.is_owner_device(base_dir):
            cat.extend(k3_host_engine.owner_k3_models())
    seen = {x["id"] for x in cat}
    for x in rm:
        if x["id"] not in seen:
            seen.add(x["id"])
            cat.append(dict(x, custom=True))
    return cat


def engine_for(model_key=None):
    import k3_host_engine
    import vault
    m = _norm_model(model_key or (S.settings.get().get("model") if S.settings else "omni"))
    if m == "omni":
        return S.omni_engine or S.engine
    if paths.WEB and m in S.k3_engines:
        return S.k3_engines[m]
    if m in S.remote_engines:
        return S.remote_engines[m]
    base_dir = Path(S.args.base) if S.args else HERE.parent
    ent = next((x for x in model_catalog(base=base_dir) if x["id"] == m or x.get("remote_model") == m), None)
    if ent is None:
        return S.omni_engine or S.engine
    key = ""
    if ent.get("host") == "k3" and k3_host_engine.is_owner_device(base_dir):
        key = k3_host_engine.k3_host_key()
    elif ent.get("key_id"):
        key = vault.get(base_dir, ent["key_id"])
    rm_name = ent.get("remote_model") or ""
    if ent.get("host") == "k3" and rm_name in ("nemotron-3.5-super-vl", "kimi-k3"):
        k3_omni_ent = next((x for x in k3_host_engine.owner_k3_models() if x.get("remote_model") == "omni"), ent)
        r_omni = k3_host_engine.RemoteOmniEngine(k3_omni_ent, base_dir, S.perms, api_key=key, local_omni=S.omni_engine)
        eng = k3_host_engine.K3HostEngine(
            rm_name, r_omni, base_dir, S.perms,
            remote_urls=k3_host_engine.K3_URLS, api_key=key, display_key=ent["id"],
        )
    else:
        eng = k3_host_engine.RemoteOmniEngine(ent, base_dir, S.perms, api_key=key, local_omni=S.omni_engine)
    S.remote_engines[m] = eng
    return eng


def _csi_url_ok(url):
    """Empty (CSI off) or a CSI service on this laptop (semantics.csi_reader accepts it)."""
    if not url.strip():
        return True
    try:
        semantics.csi_reader(url.strip())
    except ValueError:
        return False
    return True


class Settings:
    """config/settings.json: which model and engine run Layer 1, when tools are offered, and collective learning opt-in.
    All Clover core stages (C1 Inference, C2 HybridAgent Plan, Bridges, C3 11-level NCM Decode), supporting layers
    (S1 Memory, S2 Recall & Ontology, S3 Tools & Compiler), and semantic layers (CSI & PPL) are always enabled."""
    DEFAULTS = {"model": "omni", "engine": "gpu", "tools": "ask", "image_model": "sd15-lcm",
                "remote_models": [],
                "layers": {"memory": True, "ontology": True, "compiler": True, "ncm": True, "kv_slots": False,
                           "recall": True},
                "csi": True,
                "csi_url": "",
                "personal": True,
                "collective_contribute": False,  # opt-in candidate extraction (research/collective-learning)
                "collective_receive": False}  # opt-in shared-core updates (research/collective-learning)

    def __init__(self, path, engine=None):
        """engine: what this machine supports (hardware.choose): the default, and "cpu" when no usable GPU is found
        even if a GPU was chosen before (the same settings on another machine)."""
        self.path, self.lock = Path(path), threading.Lock()
        self.data = json.loads(json.dumps(self.DEFAULTS))
        if engine in ("cpu", "gpu"):
            self.data["engine"] = engine
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if saved.get("engine") in ENGINES and not (saved["engine"] == "gpu" and engine == "cpu"):
            self.data["engine"] = saved["engine"]
        self.data["remote_models"] = _clean_remote_models(saved.get("remote_models"))
        # On Laptop and Mobile, always start on the locally running LFM3-Omni-2B by default; the user switches to a
        # hosted model from the top-left dropdown when they want one.
        if paths.WEB and _norm_model(saved.get("model")) in WEB_MODELS:
            self.data["model"] = _norm_model(saved.get("model"))
        else:
            self.data["model"] = "omni"
        if saved.get("tools") in ("ask", "always"):
            self.data["tools"] = saved["tools"]
        if saved.get("image_model") in IMAGE_MODELS:
            self.data["image_model"] = saved["image_model"]
        if isinstance((saved.get("layers") or {}).get("kv_slots"), bool):
            self.data["layers"]["kv_slots"] = saved["layers"]["kv_slots"]
        for k in ("memory", "ontology", "compiler", "ncm", "recall"):
            self.data["layers"][k] = True
        if isinstance(saved.get("csi_url"), str) and _csi_url_ok(saved["csi_url"]):
            self.data["csi_url"] = saved["csi_url"].strip()
        self.data["csi"] = True
        self.data["personal"] = True
        if isinstance(saved.get("collective_contribute"), bool):
            self.data["collective_contribute"] = saved["collective_contribute"]
        if isinstance(saved.get("collective_receive"), bool):
            self.data["collective_receive"] = saved["collective_receive"]

    def get(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def update(self, changes):
        with self.lock:
            if "remote_models" in changes:
                self.data["remote_models"] = _clean_remote_models(changes["remote_models"])
            if "model" in changes:
                m = _norm_model(changes["model"])
                valid = {x["id"] for x in model_catalog(self.data.get("remote_models"))} | set(MODELS)
                if m not in valid:
                    raise ValueError(f"model must be one of {', '.join(sorted(valid))}")
                self.data["model"] = m
            if "engine" in changes:
                if changes["engine"] not in ENGINES:
                    raise ValueError("engine must be gpu, cpu or fused")
                self.data["engine"] = changes["engine"]
            if "tools" in changes:
                if changes["tools"] not in ("ask", "always"):
                    raise ValueError("tools must be ask or always")
                self.data["tools"] = changes["tools"]
            if "image_model" in changes:
                if changes["image_model"] not in IMAGE_MODELS:
                    raise ValueError("unknown image model")
                self.data["image_model"] = changes["image_model"]
            for k, v in (changes.get("layers") or {}).items():
                if k not in LAYER_KEYS or not isinstance(v, bool):
                    raise ValueError(f"unknown layer setting {k!r}")
                self.data["layers"][k] = v if k == "kv_slots" else True
            if "csi_url" in changes:
                if not isinstance(changes["csi_url"], str) or not _csi_url_ok(changes["csi_url"]):
                    raise ValueError("csi_url must be empty (off) or a CSI service on this laptop (http://127.0.0.1:<port>)")
                self.data["csi_url"] = changes["csi_url"].strip()
            if "csi" in changes:
                if not isinstance(changes["csi"], bool):
                    raise ValueError("csi must be true or false")
                self.data["csi"] = True
            if "personal" in changes:
                if not isinstance(changes["personal"], bool):
                    raise ValueError("personal must be true or false")
                self.data["personal"] = True
            if "collective_contribute" in changes:
                if not isinstance(changes["collective_contribute"], bool):
                    raise ValueError("collective_contribute must be true or false")
                self.data["collective_contribute"] = changes["collective_contribute"]
            if "collective_receive" in changes:
                if not isinstance(changes["collective_receive"], bool):
                    raise ValueError("collective_receive must be true or false")
                self.data["collective_receive"] = changes["collective_receive"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            return json.loads(json.dumps(self.data))


def gpu_available(base):
    if paths.ANDROID:  # the app's llama.cpp is always there; the GPU is whether hardware.probe found one (OpenCL)
        chosen = (S.hardware or {}).get("chosen") or {}
        return chosen.get("engine") == "gpu", "" if chosen.get("engine") == "gpu" else "no GPU llama.cpp can use (OpenCL)"
    exe = Path(base) / "tools" / "llama.cpp" / "vulkan" / "llama-server.exe"
    return exe.is_file(), "" if exe.is_file() else "missing " + exe.name


def omni_available(base):
    import omni_native
    _, missing = omni_native.omni_files(base)
    return (not missing), ("missing " + ", ".join(missing)) if missing else ""


def _walkthrough_flag(base):
    return Path(base) / "config" / "setup-walkthrough.flag"


def setup_needed():
    """The phone and laptop apps show the clean setup screen when the model is not downloaded yet, or right after
    Setup.exe launches the Studio for its first-run walkthrough."""
    if paths.WEB or S.args is None:
        return False
    base = Path(S.args.base)
    if _walkthrough_flag(base).is_file():
        return True
    return not omni_available(base)[0]


class Setup:
    """First-start setup on Mobile and Laptop: setup_fetch downloads what launch/manifest(-android).json pins (each file
    checked against its SHA-256) while web/setup.html shows the clean 4-step walkthrough; then the engine loads."""

    def __init__(self, base):
        self.base, self.lock, self.stop = Path(base), threading.Lock(), threading.Event()
        self.state, self.error, self.thread, self.started = "idle", "", None, None
        self.plan = self.done = self.item_done = self.item_total = 0
        self.item, self.item_state, self.index, self.items = "", "", 0, 0
        self.components = "core" if paths.ANDROID else "core,csi"
        self._hook_t = 0.0

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "plan": self.plan, "done": self.done + self.item_done,
                    "item": self.item, "item_state": self.item_state, "item_done": self.item_done,
                    "item_total": self.item_total, "index": self.index, "items": self.items,
                    "components": self.components,
                    "elapsed_s": round(time.time() - self.started, 1) if self.started else 0}

    def on_event(self, ev):
        with self.lock:
            kind = ev.get("ev")
            if kind == "plan":
                self.plan, self.done = int(ev.get("total") or 0), 0
            elif kind == "item" and ev.get("state") == "checking":
                self.item, self.item_state, self.item_done = ev.get("name", ""), "checking", 0
                self.item_total, self.index, self.items = int(ev.get("size") or 0), ev.get("i", 0), ev.get("n", 0)
            elif kind == "item" and ev.get("state") == "getting":
                self.item_state = "getting"
                self.item_total = int(ev.get("size") or self.item_total)
            elif kind == "item":
                self.done += self.item_total
                self.item_done, self.item_state = 0, ev.get("state", "")
                if ev.get("state") == "problem":
                    self.error = f"{ev.get('name')}: {ev.get('why')}"
            elif kind == "bytes":
                self.item_done = int(ev.get("done") or 0)
                self.item_total = int(ev.get("total") or self.item_total)
            pct = int(100 * (self.done + self.item_done) / self.plan) if self.plan else 0
        hook = S.setup_hook
        if hook is not None and time.time() - self._hook_t > 1.0:
            self._hook_t = time.time()
            try:
                hook(f"Getting {self.item}" if self.item else "Setting up", pct)
            except Exception:  # noqa: BLE001 - the notification never stops the download
                traceback.print_exc()

    def start(self, components=None):
        with self.lock:
            if self.state == "running":
                return False
            self.components = str(components or ("core" if paths.ANDROID else "core,csi")).strip()
            self.state, self.error, self.started = "running", "", time.time()
            self.stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="setup")
        self.thread.start()
        return True

    def _run(self):
        import hardware
        import setup_fetch
        setup_fetch.ON_EVENT, setup_fetch.STOP = self.on_event, self.stop
        manifest = HERE / "launch" / ("manifest-android.json" if paths.ANDROID else "manifest.json")
        try:
            rc = setup_fetch.main(["--base", str(self.base), "--components", self.components,
                                   "--manifest", str(manifest)])
        except BaseException as exc:  # noqa: BLE001 - shown on the setup page
            traceback.print_exc()
            rc, self.error = 1, f"{type(exc).__name__}: {exc}"
        finally:
            setup_fetch.ON_EVENT = setup_fetch.STOP = None
        if rc == 0:
            _walkthrough_flag(self.base).unlink(missing_ok=True)
            try:
                S.hardware = hardware.probe(self.base, paths.config_file(self.base, "hardware.json"))
                hw_eng = (S.hardware.get("chosen") or {}).get("engine")
                if S.settings is not None and hw_eng == "gpu" and S.settings.get().get("engine") == "cpu":
                    S.settings.update({"engine": "gpu"})
            except Exception:
                traceback.print_exc()
            if S.csi is not None and S.settings is not None and S.settings.get().get("csi", True):
                S.csi.start()
            try:
                import omni_c2_c3
                omni_c2_c3.for_base(self.base)
            except Exception:
                pass
        with self.lock:
            if rc == 0 and not setup_needed():
                self.state = "done"
            else:
                self.state = "stopped" if rc == 3 else "failed"
                self.error = self.error or "the download did not finish"
        if S.setup_hook is not None and self.state != "done":  # done: the model's load reports next
            try:
                S.setup_hook("Setup paused" if self.state == "stopped" else "Setup stopped", -1)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        if self.state == "done" and S.load.get("status") != "ready":
            S.load.update(status="loading", step="starting", since=time.time(), error=None)
            threading.Thread(target=load_engine, args=(S.args,), daemon=True, name="engine-load").start()


def media_dir():
    d = paths.media_dir(S.args.base)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sniff(kind, data):
    """File type from the first bytes (the browser's MIME type is not trusted)."""
    h = data[:16]
    if kind == "image":
        for sig, ext in ((b"\x89PNG", "png"), (b"\xff\xd8\xff", "jpg"), (b"GIF8", "gif"), (b"BM", "bmp")):
            if h.startswith(sig):
                return ext
        return "webp" if h[:4] == b"RIFF" and h[8:12] == b"WEBP" else None
    if kind == "audio":
        if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
            return "wav"
        if h.startswith(b"ID3") or h[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return "mp3"
        for sig, ext in ((b"fLaC", "flac"), (b"OggS", "ogg"), (b"\x1a\x45\xdf\xa3", "webm")):
            if h.startswith(sig):
                return ext
        return "m4a" if h[4:8] == b"ftyp" else None
    if kind == "video":
        if h[4:8] == b"ftyp":
            return "mp4"
        if h.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"
        return "avi" if h[:4] == b"RIFF" and h[8:12] == b"AVI " else None
    return None


def _ffmpeg(args, cancel=None):
    import local_terminal
    import local_tool_packs
    exe = local_tool_packs.find_media_binary(S.args.base, "ffmpeg")
    if exe is None:
        raise ValueError("this file type needs ffmpeg (run Clover AI Studio setup again)")
    res = local_terminal.run([str(exe), "-y", "-loglevel", "error"] + list(args), timeout_s=180, cancel=cancel)
    if res["cancelled"]:
        raise RuntimeError("ffmpeg cancelled")
    if res["timed_out"]:
        raise RuntimeError("ffmpeg timed out after 180s")
    if res["exit_code"] != 0:
        raise RuntimeError(f"ffmpeg failed (exit {res['exit_code']}): {(res['stderr'] or res['stderr'] or '').strip()}")


def tool_inventory():
    import lfm3_native
    import local_tool_packs
    base = [t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS]
    conditional = ["remember", "vscode_open", "vscode_clone", "vscode_install", "vscode_run", "generate_image"]
    installed_cond, unavailable_cond = [], []
    on_mem = bool(S.settings and S.settings.get().get("layers", {}).get("memory"))
    if on_mem or (S.workspaces and S.workspaces.list()):
        installed_cond.append("remember")
    else:
        unavailable_cond.append("remember")
    if S.vscode is not None:
        installed_cond.extend(["vscode_open", "vscode_clone", "vscode_install", "vscode_run"])
    else:
        unavailable_cond.extend(["vscode_open", "vscode_clone", "vscode_install", "vscode_run"])
    base_dir = Path(S.args.base) if S.args else None
    if base_dir and (base_dir / "tools" / "sd.cpp" / "vulkan" / "sd-cli.exe").is_file():
        installed_cond.append("generate_image")
    else:
        unavailable_cond.append("generate_image")
    ffprobe_ok = local_tool_packs.find_media_binary(base_dir, "ffprobe") is not None
    ffmpeg_ok = local_tool_packs.find_media_binary(base_dir, "ffmpeg") is not None
    unavail_base = []
    if not ffprobe_ok:
        unavail_base.append("media_info")
    if not ffmpeg_ok:
        unavail_base.append("extract_frame")
    installed_base = [t for t in base if t not in unavail_base]
    return {
        "base": base,
        "conditional": conditional,
        "installed": installed_base + installed_cond,
        "unavailable": unavail_base + unavailable_cond,
    }


def save_media(items):
    """Attachments -> files in cache/media named by the server: [{"kind", "file", "name"}]. Audio that llama.cpp cannot
    decode itself (it reads wav, mp3, flac) becomes 16 kHz mono WAV, WebP images become PNG."""
    out = []
    if not isinstance(items, list):
        return out
    if len(items) > MEDIA_ITEMS:
        raise ValueError(f"at most {MEDIA_ITEMS} attachments per message")
    for it in items:
        kind = str((it or {}).get("kind") or "")
        if kind not in MEDIA_LIMITS:
            raise ValueError("attachment kind must be image, audio or video")
        raw = str(it.get("data") or "")
        raw = raw.split(",", 1)[1] if raw.startswith("data:") else raw
        try:
            data = base64.b64decode(raw, validate=True)
        except ValueError:
            raise ValueError("attachment is not valid base64") from None
        if not data or len(data) > MEDIA_LIMITS[kind]:
            raise ValueError(f"{kind} attachments must be 1 byte to {MEDIA_LIMITS[kind] >> 20} MB")
        ext = _sniff(kind, data)
        if ext is None:
            raise ValueError(f"unsupported {kind} file type")
        # named by content: VS Code sends the same image again with every tool round, and it is stored once
        stem = hashlib.sha256(data).hexdigest()[:32]
        target = {"audio": "wav", "image": "png"}.get(kind) if ext in ("ogg", "webm", "m4a", "webp") else None
        path = media_dir() / f"{stem}.{target or ext}"
        if path.is_file():
            os.utime(path)  # in use again: not pruned
        elif target:
            src = media_dir() / f"{stem}.{ext}"
            src.write_bytes(data)
            try:
                _ffmpeg(["-i", str(src)] + (["-ac", "1", "-ar", "16000"] if kind == "audio" else []) + [str(path)])
            finally:
                src.unlink(missing_ok=True)
        else:
            path.write_bytes(data)
        name = re.sub(r"[\r\n<>]+", " ", str(it.get("name") or path.name))[:120]
        out.append({"kind": kind, "file": path.name, "name": name})
    return out


def image_tool(sid, cancel, made):
    """generate_image for this turn: pictures it makes are collected in `made` and shown with the answer."""
    import image_gen
    import lfm3_native
    gen = image_gen.ImageGen(S.args.base, S.settings.get().get("image_model", "sd15-lcm"))

    def run(a):
        if gen.available()[0]:
            free = image_gen.free_mb()
            if free is not None and free < image_gen.NEED_FREE_MB and hasattr(S.engine, "free_voice") and S.engine.free_voice():
                time.sleep(1.5)
                free = image_gen.free_mb()
            if free is not None and free < image_gen.NEED_FREE_MB:
                return (f"Error: no picture was made: this laptop has {free / 1024:.1f} GB of memory free and making a picture "
                        f"needs about {image_gen.NEED_FREE_MB / 1024:.1f} GB. Tell the user to close some programs (browser "
                        "tabs, other apps) and ask again. Do not try again now.")
        try:
            r = gen.generate(a.get("prompt") or "", media_dir(), cancel=cancel)
        except (RuntimeError, ValueError, TypeError) as exc:
            return f"Error: no picture was made: {exc}. Tell the user plainly; do not call generate_image again in this answer."
        made.append(dict(r, prompt=str(a.get("prompt") or "")[:300]))
        return (f"Made the picture ({r['width']}x{r['height']}, {r['model']}, {r['seconds']} s); the user sees it in the chat "
                "now. Reply with one short sentence; do not describe details you cannot see.")
    return {"schema": lfm3_native._fn(
        "generate_image", "Make a picture or illustration from a text description. Use when the "
        "user asks for an image, picture, drawing or illustration.",
        {"prompt": {"type": "string", "description": "what the picture shows, in English, with style details"}}, ["prompt"]),
        "permission": "media", "detail": lambda a: "Create a picture: " + str(a.get("prompt") or "")[:300], "run": run}


def layer_kit():
    if S.kit is None:
        import lfm3_layers
        S.kit = lfm3_layers.LayerKit(Path(S.args.base), memory_root=paths.knowledge_dir(S.args.base))
        S.kit.workspaces = S.workspaces
    if S.kit.records is None:
        S.kit.records = S.repo
    if S.kit.recall is None:
        S.kit.recall = S.memidx
    return S.kit


def studio_idle():
    """No turn running or just finished, and no engine switch: background work (exchange notes) may run."""
    import memory_index
    return (S.busy is None and S.engine is not None and S.load["status"] == "ready"
            and time.time() - S.last_turn_end > memory_index.IDLE_NOTE_S)


def write_note(msgs, max_new):
    eng = S.engine
    return eng.side_complete(msgs, max_new) if eng is not None and hasattr(eng, "side_complete") else None


def remember_note(text, sid, memory_on):
    """A note about the open workspace goes into the context file of the folder it names; anything else to memory."""
    target = S.workspaces.note_target(text) if S.workspaces else None
    if target:
        return "Saved to the workspace context notes in " + S.workspaces.add_note(target, text) + "."
    if not memory_on:
        return "Not saved: memory is off in Settings and this note is not about the open workspace."
    return "Saved to memory as " + layer_kit().remember(text, conversation_id=sid) + "."


def laptop_runtime():
    """The laptop's own part of semantics: its tools (lfm3_native.RUNTIME), Clover's vault on this laptop, CSI (the
    child service, csi_child.py, on by default once ready; or a service named in Settings csi_url), and the user's own
    learning (personal; on by default, Settings -> personal turns it off)."""
    import dataclasses
    import lfm3_native
    s = S.settings.get() if S.settings else {}
    url = s.get("csi_url") or (S.csi.url if s.get("csi", True) and S.csi is not None and S.csi.alive() else "")
    api_key = ""
    if url and "api.cloverframework.com" in url:
        import k3_host_engine
        api_key = k3_host_engine.k3_host_key()
    csi = semantics.csi_reader(url, api_key=api_key) if url else None
    return dataclasses.replace(lfm3_native.RUNTIME, device=paths.DEVICE_WORD,
                               blocked=lambda p: paths.read_blocked(S.args.base, p),
                               csi=csi, personal=personal_store(csi, keep_csi=False),
                               collective=collective_store(csi, keep_csi=False))


def personal_store(csi=None, keep_csi=True):
    """The user's personal store (<base>/personal), created on first use: learning is on by default and Settings ->
    personal switches it (the store keeps the switch; viewing and deleting work either way). A turn passes its CSI
    reader; other callers leave it as it is."""
    import personal
    root = Path(S.args.base) / "personal"
    return personal.for_process(str(root), paths.DEVICE_WORD, personal.KEEP if keep_csi and csi is None else csi)


def collective_store(csi=None, keep_csi=True):
    """The user's collective learning store (<base>/collective): two independent opt-in switches (contribute candidate
    strategies, receive validated shared core releases)."""
    import core_sync
    root = Path(S.args.base) / "collective"
    coll = core_sync.for_process(str(root), paths.DEVICE_WORD, core_sync.KEEP if keep_csi and csi is None else csi)
    if coll is not None and S.settings is not None:
        st = S.settings.get()
        if (coll.consent_contribute != bool(st.get("collective_contribute", False))
                or coll.consent_receive != bool(st.get("collective_receive", False))):
            coll.set_consent(contribute=bool(st.get("collective_contribute", False)),
                             receive=bool(st.get("collective_receive", False)))
    return coll


def google_identity_manager():
    """Stateless Google OpenID Connect identity manager for this installation (<base>/vault)."""
    import google_identity
    if S.google_identity is None and S.args is not None:
        S.google_identity = google_identity.GoogleIdentityManager(Path(S.args.base))
    return S.google_identity


def collective_gateway_runtime():
    """Remote Collective Learning Gateway & Signed Registry runtime (hosted on k3 or local gateway mode)."""
    import collective_evaluator
    import collective_gateway
    import core_registry
    if S.gateway_runtime is None and S.args is not None:
        base = Path(S.args.base)
        gw = collective_gateway.CollectiveGateway(state_dir=base / "collective-gateway")
        min_inst = int(os.environ.get("CLOVER_CPSL_MIN_INSTALLATIONS") or 2)
        ev = collective_evaluator.CollectiveEvaluator(min_installations=min_inst, min_agreement=0.75)
        reg = core_registry.CoreRegistry(base / "clover-collective")
        S.gateway_runtime = {"gateway": gw, "evaluator": ev, "registry": reg, "lock": threading.RLock()}
    return S.gateway_runtime


def personal_step(pers, method, *args):
    """A personal-learning step; it never breaks the turn or the request (the error is printed and they go on)."""
    if pers is None:
        return None
    try:
        return getattr(pers, method)(*args)
    except Exception:
        traceback.print_exc()
        return None


REMEMBER_ASK = semantics.REMEMBER_ASK


def workspace_action(action, body):
    w, name = S.workspaces, (str(body.get("name")) if body.get("name") else None)
    if action == "add":
        return w.add(body.get("source"), name)
    if action == "activate":
        return w.activate(name)
    if name is None:
        raise ValueError("name is required")
    if action == "refresh":
        return dict(w.describe(), context=w.refresh(name))
    if action == "pull":
        return w.pull(name)
    if action == "clean":
        return w.clean(name)
    if action == "remove":
        return w.remove(name)
    raise ValueError(f"unknown workspace action {action!r}")


# ---------- VS Code: Clover asks, VS Code does the work with its own UI, sign-in and confirmations ----------
VSCODE_TIMEOUTS = {"open": 60, "clone": 600, "install": 300, "run": 900, "githubSignIn": 300, "githubSetupRepo": 600,
                   "githubSyncFolder": 300, "status": 20}
UI_VSCODE_ACTIONS = ("status", "open", "githubSignIn", "githubSetupRepo")


def vscode_exe():
    for p in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "Code.exe",
              Path(os.environ.get("ProgramFiles", "")) / "Microsoft VS Code" / "Code.exe"):
        if p.is_file():
            return p
    return None


def vscode_do(action, args=None):
    if action == "open":
        import device_actions
        target = str((args or {}).get("target") or "").strip()
        if device_actions.is_standalone_app_target(target) or S.vscode is None or not S.vscode.ensure(vscode_exe()):
            return device_actions.open_target(target, Path(S.args.base) if S.args else None)
    if S.vscode is None or not S.vscode.ensure(vscode_exe()):
        raise RuntimeError("VS Code with the Clover extension is not running and could not be started")
    return S.vscode.call(action, args or {}, timeout=VSCODE_TIMEOUTS.get(action, 120))


def vscode_tool(action):
    def run(a):
        try:
            return json.dumps(vscode_do(action, a), ensure_ascii=False)[:8000]
        except RuntimeError as exc:
            return f"Error: {exc}"
    return run


def chat_dir(sid):
    """The folder a chat works in: the workspace in use, else the last repository cloned in this chat."""
    ws = S.workspaces.active() if S.workspaces else None
    return ws["path"] if ws else S.chat_dirs.get(sid)


def vscode_tools(sid=None):
    import lfm3_native
    s = {"type": "string"}

    def clone(a):
        out = vscode_tool("clone")(a)
        try:
            path = json.loads(out).get("path")
        except (ValueError, AttributeError):
            path = None
        if sid and path and Path(path).is_dir():  # in the user's chat, git log ran in the home folder right after a clone
            S.chat_dirs[sid] = str(path)
        return out

    def run(a):
        import local_terminal
        where = a.get("cwd") or chat_dir(sid)
        out = vscode_tool("run")(dict(a, cwd=where) if where else a)
        return re.sub(r"\\u001b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\\\-_])", "", local_terminal.strip_ansi(out))
    return [
        {"schema": lfm3_native._fn("vscode_open", "Open a file, folder or web page in VS Code on this laptop. Files open in "
                                 "the right viewer (Markdown preview, images, video, code with its checks); web pages open "
                                 "in VS Code's browser; a folder opens in a new window.",
                                 {"target": dict(s, description="full path or http(s) URL"),
                                  "line": {"type": "integer", "description": "line to show (files)"}}, ["target"]),
         "permission": "vscode", "detail": lambda a: "Open in VS Code: " + str(a.get("target") or ""),
         "run": vscode_tool("open")},
        {"schema": lfm3_native._fn("vscode_clone", "Clone a GitHub repository with VS Code (private ones use the user's "
                                 "GitHub sign-in in VS Code).",
                                 {"repository": dict(s, description="owner/repo or its https://github.com URL")},
                                 ["repository"]),
         "permission": "vscode", "detail": lambda a: "Clone with VS Code: " + str(a.get("repository") or ""),
         "run": clone},
        {"schema": lfm3_native._fn("vscode_install", "Install a VS Code extension when the user needs a capability VS Code "
                                 "lacks (a language's syntax checks, a previewer).",
                                 {"extension": dict(s, description="extension id publisher.name, e.g. ms-python.python")},
                                 ["extension"]),
         "permission": "vscode", "detail": lambda a: "Install the VS Code extension " + str(a.get("extension") or ""),
         "run": vscode_tool("install")},
        {"schema": lfm3_native._fn("vscode_run", "Run a PowerShell command in a VS Code terminal the user can watch "
                                 "(install a library, build, run tests) and get its output and exit code.",
                                 {"command": dict(s, description="the PowerShell command line"),
                                  "cwd": dict(s, description="folder to run it in (optional)")}, ["command"]),
         "permission": "shell",
         "detail": lambda a: f"{a.get('command')}\n(in {a.get('cwd') or chat_dir(sid) or 'your home folder'}, in a VS Code terminal)",
         "run": run},
        {"schema": lfm3_native._fn("vscode_github_setup", "Create (or connect) the user's private GitHub repository for "
                                 "Clover's records and push them. VS Code asks the user to sign in and to confirm.", {}),
         "permission": "vscode", "detail": lambda a: "Set up your private GitHub repository for Clover's records",
         "run": lambda a: _tool_json(github_setup)},
        {"schema": lfm3_native._fn("vscode_github_sync", "Sync Clover's records with the user's private GitHub "
                                 "repository (bring in other devices' records, then push) through VS Code.", {}),
         "permission": "vscode", "detail": lambda a: "Sync Clover's records with your private GitHub repository",
         "run": lambda a: _tool_json(sync_repo)},
    ]


def _tool_json(fn):
    try:
        return json.dumps(fn(), ensure_ascii=False)[:4000]
    except (RuntimeError, ValueError) as exc:
        return f"Error: {exc}"


def github_repo(update=None):
    """config/<device>/github-repo.json: the private GitHub repository the records sync with, and when they last did."""
    path = paths.config_file(S.args.base, "github-repo.json")
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = None
    if update is not None:
        st = dict(st or {}, **update)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st


def _git_with_vscode(action, args):
    """VS Code runs git with the user's GitHub sign-in; AGENTS.md is writable only while it does."""
    S.repo.flush()
    S.repo.agents_writable()
    try:
        return vscode_do(action, args)
    finally:
        S.repo.protect_agents()


def github_setup(name=None):
    """Creates (after the user confirms in VS Code) or connects the user's private GitHub repository and pushes the
    records to it."""
    res = _git_with_vscode("githubSetupRepo", {"name": name or paths.REPO_NAME, "path": str(S.repo.dir)})
    if isinstance(res, dict) and res.get("repo"):
        github_repo({"repo": res["repo"], "private": bool(res.get("private")), "last_sync": time.strftime("%Y-%m-%d %H:%M")})
        S.repo.reindex()
        layer_kit().store().rebuild_index()
        S.repo.note(f"GitHub: records pushed to {res['repo']} (private)" + (", created" if res.get("created") else ""))
    return res


def sync_repo():
    st = github_repo()
    if not st or not st.get("repo"):
        raise ValueError("set up the private GitHub repository first (Settings -> VS Code)")
    res = _git_with_vscode("githubSyncFolder", {"path": str(S.repo.dir), "repo": st["repo"]})
    S.repo.reindex()
    layer_kit().store().rebuild_index()
    github_repo({"last_sync": time.strftime("%Y-%m-%d %H:%M")})
    S.repo.note(f"GitHub: synced with {st['repo']}" + (" (other devices' records merged)" if (res or {}).get("merged") else ""))
    return dict(res or {}, repo=st["repo"])


def _tell_phone(text, percent):
    """The phone app's notification; while percent is 0-99 the app keeps the phone awake and its screen on (with the
    screen off during the S23's first load, Android closed Clover with the other apps it closed for memory)."""
    if S.setup_hook is not None:
        try:
            S.setup_hook(text, percent)
        except Exception:  # noqa: BLE001 - the notification never stops the load
            traceback.print_exc()


def load_engine(args):
    def progress(msg):
        S.load["step"] = msg
        print("LOAD", msg, flush=True)
        _tell_phone(f"Loading {msg}", 99)

    if S.mode == "lfm3":
        try:
            want = S.settings.get()["engine"]
            S.load["warning"] = None
            ok, why = omni_available(args.base)
            if not ok:
                raise FileNotFoundError(f"LFM3-Omni-2B is not installed ({why}): run Clover AI Studio setup again")
            import omni_native
            hw = (S.hardware or {}).get("chosen") or {}
            if want == "fused":
                try:
                    import fused_engine
                    S.engine = fused_engine.FusedEngine(Path(args.base), S.perms, progress=progress)
                except Exception as exc:  # noqa: BLE001 - the GPU or CPU engine instead of a dead Studio
                    traceback.print_exc()
                    S.load["warning"] = f"C1 + C2 + C3 could not start ({type(exc).__name__}: {exc})"
                    want = hw.get("engine") or "cpu"
            for kind in ([] if S.engine is not None else [want, "cpu"] if want == "gpu" else ["cpu"]):
                try:
                    S.engine = omni_native.OmniNative(Path(args.base), S.perms, progress=progress, engine=kind,
                                                      max_tokens=hw.get("ctx") or 32768,
                                                      voice_threads=hw.get("voice_threads") or omni_native.VOICE_THREADS,
                                                      gpu_name=hw.get("gpu") or "GPU", threads=hw.get("threads"))
                    break
                except Exception as exc:  # noqa: BLE001 - try the next engine instead of leaving the Studio dead
                    traceback.print_exc()
                    S.load["warning"] = f"LFM3-Omni-2B on the {kind.upper()} failed ({type(exc).__name__}: {exc})"
            if S.engine is None:
                raise RuntimeError(S.load["warning"])
            if S.engine.engine_kind != want:
                S.load["warning"] = (S.load.get("warning") or "") + f"; running on the {S.engine.engine_kind.upper()}"
            S.omni_engine = S.engine
            if paths.WEB:
                import k3_host_engine
                S.k3_engines = {k: k3_host_engine.K3HostEngine(k, S.omni_engine, Path(args.base), S.perms)
                                for k in ("nemotron-3.5-super-vl", "kimi-k3")}
            S.remote_engines.clear()
            try:
                import omni_c2_c3
                progress("C2 HybridAgent, Bridges & C3 11-Level NCM")
                omni_c2_c3.for_base(Path(args.base))
            except Exception:
                traceback.print_exc()
            S.engine = engine_for(S.settings.get().get("model"))
            S.kv_sid = None
            S.load.update(status="ready", step="ready", ready_s=round(time.time() - S.load["since"], 1))
            print("ENGINE_READY", json.dumps(S.engine.describe()), flush=True)
            _tell_phone("Clover runs on this phone", 100)
        except BaseException as exc:  # noqa: BLE001 - report any load failure to the UI
            S.load.update(status="error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            _tell_phone("Clover could not load its model", -1)
        return
    try:
        import lfm3_agent

        eng = lfm3_agent.CloverLFM3Engine(Path(args.base), args.planner, search_cache=args.search_cache,
                                          search_mode=args.search_mode, gate_tools=not args.no_gate, progress=progress,
                                          planner_backend=args.planner_backend)
        S.engine = eng
        S.load.update(status="ready", step="ready", ready_s=round(time.time() - S.load["since"], 1))
        print("ENGINE_READY", json.dumps(eng.describe()), flush=True)
    except BaseException as exc:  # noqa: BLE001 - report any load failure to the UI instead of dying silently
        S.load.update(status="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def media_from(res):
    media = []
    for t in res.get("tool_calls") or []:
        if not isinstance(t, dict):
            continue
        for m in t.get("media") or []:
            if not isinstance(m, dict):
                continue
            url = str(m.get("data_url") or "")
            if not url and m.get("svg"):
                url = "data:image/svg+xml;base64," + base64.b64encode(str(m["svg"]).encode("utf-8")).decode("ascii")
            if DATA_URL_RE.match(url):
                media.append({"data_url": url, "title": str(m.get("title") or "")[:200],
                              "caption": str(m.get("caption") or "")[:400]})
    return media


def clean_tool_calls(tool_calls):
    out = []
    for t in tool_calls or []:
        if not isinstance(t, dict):
            continue
        d = {k: v for k, v in t.items() if k != "media"}
        if "entries" in d:
            d["entries"] = list(d["entries"])
        for k in ("input", "result"):
            if isinstance(d.get(k), str) and len(d[k]) > 4000:
                d[k] = d[k][:4000] + " ..."
        out.append(d)
    return out


def run_turn(session_id, prompt, *, remember=True, history=None, allow_once=(), deny_once=(), interactive=True,
             kind="chat"):
    import permissions as P
    eng = S.engine
    if eng is None:
        raise EngineNotReady(S.load.get("error") or f"engine is loading ({S.load.get('step')})")
    t0 = time.perf_counter()
    with S.turn_lock:
        if history is None:
            with S.sess_lock:
                if session_id not in S.sessions and len(S.sessions) >= MAX_SESSIONS:
                    S.sessions.pop(next(iter(S.sessions)))
                hist = S.sessions.setdefault(session_id, [])
        else:
            hist = list(history)
        S.busy = {"kind": kind, "since": time.time()}
        try:
            with P.turn_scope(S.perms, allow_once, deny_once, interactive) as ctx:
                eng.history = hist
                try:
                    res = dict(eng.agent_reply(prompt, cands=2, remember=remember))
                except P.PermissionNeeded as pn:
                    return {"session_id": session_id, "model": MODEL_ID,
                            "permission_request": {"tool": pn.tool, "label": P.TOOLS[pn.tool]["label"],
                                                   "detail": pn.detail},
                            "permission_events": ctx["events"],
                            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
                events = ctx["events"]
        finally:
            S.busy = None
            S.turns += 1
    res["media"] = media_from(res)
    res["tool_calls"] = clean_tool_calls(res.get("tool_calls"))
    res.update(session_id=session_id, model=MODEL_ID, wall_ms=round((time.perf_counter() - t0) * 1000.0, 1),
               history_len=len(hist), permission_events=events)
    return res


def build_prompt(prompt, files, docs=()):
    """The user message with its attachments: documents (saved; their preview and path, documents.block), then text
    files (the first ATTACH_CHARS characters)."""
    import documents
    blocks = [documents.block(d) for d in docs or ()]
    for f in (files or [])[:ATTACH_FILES]:
        if not isinstance(f, dict):
            continue
        text = str(f.get("content_text") or "")
        if text.strip():
            name = re.sub(r"[\r\n]+", " ", str(f.get("name") or "file"))[:120]
            blocks.append(f"File: {name}\n{text[:ATTACH_CHARS]}")
    return ("---\n".join(blocks) + ATTACH_MARKER + prompt) if blocks else prompt


def save_documents(items):
    """Documents attached to a message -> saved files (documents.save) under <media>/documents."""
    import documents
    return documents.save(items, media_dir() / "documents") if items else []


def lfm3_history(sid, seed):
    """Session history for LFM3 mode: user / assistant messages (final answers only). After a restart it comes from
    the conversation's file in the repository; the browser's copy is the fallback."""
    with S.sess_lock:
        if sid not in S.sessions:
            if len(S.sessions) >= MAX_SESSIONS:
                S.sessions.pop(next(iter(S.sessions)))
            msgs, turns = [], {}
            for m in S.repo.messages(sid) if S.repo else []:
                turns.setdefault(m.get("n"), {})[m.get("role")] = m
            for k in sorted(k for k in turns if isinstance(k, int))[-6:]:
                u, a = turns[k].get("user"), turns[k].get("clover")
                if u and a:
                    um = {"role": "user", "content": str(u.get("text") or "")[:4000]}
                    media = [x for x in u.get("media") or [] if isinstance(x, dict) and x.get("file")
                             and (media_dir() / x["file"]).is_file()]
                    if media:
                        um["media"] = media
                    msgs += [um, {"role": "assistant", "content": str(a.get("text") or "")[:4000]}]
            if not msgs:
                for h in (seed or [])[-6:]:
                    if isinstance(h, dict) and h.get("user") and h.get("reply"):
                        msgs += [{"role": "user", "content": str(h["user"])[:4000]},
                                 {"role": "assistant", "content": str(h["reply"])[:4000]}]
            S.sessions[sid] = msgs
        return S.sessions[sid]


def kv_switch(sid):
    """Layer 7 for a dense model: keep each chat's KV/conv state on disk (llama-server slots), so returning to a chat
    does not re-read its whole history."""
    eng = S.engine
    if eng is None or getattr(eng, "gpu", None) is None or not S.settings.get()["layers"]["kv_slots"] or S.kv_sid == sid:
        return None
    t0, info = time.perf_counter(), {}
    name = lambda s: re.sub(r"[^A-Za-z0-9_\-]", "_", s)[:80] + ".bin"  # noqa: E731
    if S.kv_sid:
        info["saved"] = eng.gpu.save_slot(name(S.kv_sid))
    if (eng.gpu.slots_dir / name(sid)).is_file():
        info["restored"] = eng.gpu.restore_slot(name(sid))
    S.kv_sid = sid
    info["ms"] = round((time.perf_counter() - t0) * 1000)
    return info


def lfm3_turn(sid, prompt, body, on_event=None, ask=None, cancel=None, kind="chat", media=(), ask_text=None, docs=()):
    """ask_text: what the user typed (without attached file contents), to see which tools they asked for and whether
    they say the last answer was not right. docs: documents saved for this message (save_documents)."""
    import clover_cycle as C
    import documents
    import lfm3_layers
    import lfm3_native
    eng = engine_for(body.get("model") if isinstance(body, dict) and body.get("model") else None)
    if eng is None:
        raise EngineNotReady(S.load.get("error") or f"engine is loading ({S.load.get('step')})")
    omni = getattr(eng, "inputs", None) is not None
    if media and not omni:
        raise ValueError("LFM3-1.7B reads text only: switch the model to LFM3-Omni-2B in Settings to send images, audio or video")
    hist = lfm3_history(sid, body.get("history"))
    sampling = body.get("sampling") if body.get("sampling") in ("general", "math", "greedy") else "general"
    settings = S.settings.get()
    on = settings["layers"]
    ws = S.workspaces.active() if S.workspaces else None
    typed = prompt if ask_text is None else ask_text
    want_recall = (on["recall"] or lfm3_layers.needs_recall(typed)) and omni and S.memidx is not None
    layers = (tuple(k for k in ("memory", "ontology", "compiler", "ncm") if on[k]) + (("workspace",) if ws else ())
              + (("records",) if S.repo else ()) + (("recall",) if want_recall else ()))
    kit = layer_kit() if layers else None
    rt = laptop_runtime()
    attached = bool(body.get("attached_files") or docs)
    ppl_growth = []
    if rt.personal is not None:  # what this message shows about the last answer, before it is read
        last = C.last_exchange(hist) if hist and not media and not attached else None
        ppl_growth = personal_step(rt.personal, "before_turn", sid, typed,
                                   C.retry_requested(typed, last[0].get("content")) if last else None) or []
    doc_names = [d["name"] for d in docs] + [n for n, _ in documents.in_history(hist)]
    frame = semantics.interpret(typed, rt, media=media, history=hist, attached=attached, documents=doc_names)
    retry, how = None, frame.retry
    if how and hasattr(eng, "retry_turn"):  # the user says the Outcome is not right: back to Context
        last = C.last_exchange(hist)
        q = last[0] if how == "said" else {"content": prompt}
        retry = {"how": how, "question": q.get("content") or "", "media": list(q.get("media") or []),
                 "previous": last[1], "feedback": typed if how == "said" else "",
                 "context": C.context_before(hist, q)}
    extra = []
    if on["memory"] or ws:
        extra.append({"schema": lfm3_native._fn(
            "remember", "Save something the user asked you to remember (a fact, preference or decision). A note about "
            "the open workspace goes into that folder's context file; anything else goes to the user's memory on this "
            "laptop. Use only when the user asks you to remember or note something.",
            {"text": {"type": "string", "description": "what to remember, in one or two sentences"}}, ["text"]),
            "permission": "memory", "detail": lambda a: str(a.get("text") or ""),
            "run": lambda a: remember_note(str(a.get("text") or ""), sid, on["memory"])})
    if S.vscode is not None:
        extra += vscode_tools(sid)
    else:
        open_t = dict(vscode_tools(sid)[0])
        if paths.ANDROID or paths.WEB:
            open_t["permission"] = None
        extra.append(open_t)
    made = []
    eng.turn_images = made  # pictures and charts made in this turn (image tool, Python's PNG files)
    extra.append(image_tool(sid, cancel, made))
    t0 = time.perf_counter()
    if S.turn_lock.locked() and on_event:
        on_event({"type": "plan", "round": 1, "stage": "queued"})
    while not S.turn_lock.acquire(timeout=0.05):
        if cancel is not None and cancel.is_set():
            m_id = getattr(eng, "model_key", None) or MODEL_ID
            return {
                "reply": "(Stopped.)",
                "cancelled": True,
                "tool_calls": [],
                "rounds": 0,
                "generation": [],
                "layers": [],
                "session_id": sid,
                "model": m_id,
                "mode": "lfm3",
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
    try:
        if cancel is not None and cancel.is_set():
            m_id = getattr(eng, "model_key", None) or MODEL_ID
            return {
                "reply": "(Stopped.)",
                "cancelled": True,
                "tool_calls": [],
                "rounds": 0,
                "generation": [],
                "layers": [],
                "session_id": sid,
                "model": m_id,
                "mode": "lfm3",
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        files = [str(f.get("name") or "file") for f in body.get("attached_files") or [] if isinstance(f, dict)]
        files += [d["name"] for d in docs]
        n, growth = S.repo.turn_start(sid, typed, how, files, media) if S.repo else (0, [])

        def emit(ev):
            if S.repo is not None and ev.get("type") not in ("token", "round"):
                S.repo.event(sid, n, ev)
            if on_event:
                on_event(ev)
        S.busy = {"kind": kind, "since": time.time()}
        gate = getattr(eng, "side_gate", None)
        if gate is not None:  # an exchange note being written stops for this turn
            gate.hold()
        cdir = None if ws else S.chat_dirs.get(sid)
        eng.workdir, eng.workname, eng.after_write = (
            (Path(ws["path"]), ws["name"], S.workspaces.touch) if ws
            else (Path(cdir), Path(cdir).name, None) if cdir else (None, "", None))
        try:
            tools_on, said = bool(body.get("tools", True)), None
            # trying again where the question or this message asks for a tool (a search): answered again with it
            retry_tools = (lfm3_native.asked_tools(retry["question"]) + lfm3_native.asked_tools(typed)
                           if retry and tools_on else [])
            if tools_on and settings["tools"] == "ask":
                said = typed
                if any(m["kind"] == "audio" for m in media) and hasattr(eng, "transcribe"):
                    said = (said + " " + eng.transcribe(media)).strip()
            only, tools_on = semantics.route(
                frame, settings["tools"], tools_on, said=said, retry_tools=retry_tools,
                offer=[t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS] + [t["schema"]["function"]["name"] for t in extra])
            if kit is not None:
                kit.meaning = frame
            max_new = int(min(4096, max(64, int(body.get("max_new_tokens") or 2048))))
            kv = kv_switch(sid) if frame.act != "social" else None
            if frame.act == "social":
                # only pleasantries ("hi", "thanks", "bye"): semantics answers and the model is not called
                emit(dict(frame.layer(), type="semantics"))
                emit({"type": "token", "round": 1, "kind": "answer", "text": frame.reply})
                res = {"reply": frame.reply, "tool_calls": [], "rounds": 0, "generation": [], "layers": []}
            elif retry and not retry_tools:
                frame.route, frame.offered = "try again", []
                emit(dict(frame.layer(), type="semantics"))
                emit({"type": "tools", "offered": [], "mode": settings["tools"]})
                res = eng.retry_turn(list(hist), retry["question"], retry["previous"], retry["feedback"],
                                     media=retry["media"], layers=layers, kit=kit, on_event=emit,
                                     cancel=cancel, max_new=max_new, ask=ask, context=retry["context"],
                                     allow=[str(x) for x in body.get("allow_once") or []])
                only, tools_on = [], False
            else:
                emit(dict(frame.layer(), type="semantics"))
                emit({"type": "tools", "offered": only if only is not None else ("all" if tools_on else []),
                      "mode": settings["tools"]})
                # a message asking for a tool is not a question about Clover's knowledge: the ontology layer matched
                # "Clover" in C:\Users\santh\CloverAI\... and in a pasted clover-framework URL, added records with "no
                # tools needed", and the Omni summarised them instead of opening the file or cloning the repository
                act = bool(only)
                turn_layers = tuple(x for x in layers if x != "ontology") if act else layers
                import device_actions
                if omni and not media and device_actions.asks_for_screen_or_activity(typed):
                    try:
                        scr = device_actions.capture_screenshot(Path(S.args.base))
                        media = [{"kind": "image", "file": scr["file"], "name": "Live Screen Capture", "screen_capture": True}]
                        made.append({
                            "file": scr["file"],
                            "prompt": "Live Screen Capture",
                            "model": "screen_capture",
                            "seconds": round(scr.get("ms", 0) / 1000.0, 2),
                            "width": scr.get("width"),
                            "height": scr.get("height"),
                        })
                    except Exception:
                        pass
                res = eng.run_turn(list(hist), prompt, tools=tools_on, sampling=sampling,
                                   thinking=bool(body.get("thinking", True)), max_new=max_new,
                                   on_event=emit, ask=ask, cancel=cancel, final_on_limit=True, force_answer=True,
                                   allow=[str(x) for x in body.get("allow_once") or []],
                                   layers=turn_layers, kit=kit, layer_query=prompt, extra_tools=extra, only_tools=only,
                                   **({"media": list(media), "meaning": frame} if omni else {}))
                for tc in res.get("tool_calls") or ():
                    if tc.get("tool") == "vscode_open" and tc.get("ran"):
                        try:
                            doc_o = json.loads(str(tc.get("result") or "{}"))
                            alink = doc_o.get("action_link")
                            lbl = doc_o.get("label") or doc_o.get("target") or "Open"
                            if alink:
                                emit({"type": "app_action", "action": "open", "label": lbl, "action_link": alink,
                                      "android_pkg": doc_o.get("android_pkg", ""), "android_uri": doc_o.get("android_uri", ""),
                                      "web_url": doc_o.get("web_url", "")})
                                if alink not in str(res.get("reply") or ""):
                                    res["reply"] = (str(res.get("reply") or "").rstrip() + f"\n\n[Open {lbl}]({alink})").strip()
                        except Exception:
                            pass
                if retry:
                    res["layers"].append({"layer": "cycle", "trigger": "user", "action": "redone", "tools": only or "all"})
            if retry:
                res["retry"] = retry["how"]
            res["tools_offered"] = only if only is not None else ("all" if tools_on else [])
            memory_ok = not retry and (on["memory"] or ws) and S.perms.get().get("memory") == "allow"
            semantics.align(frame, res, remember_fn=(lambda note: remember_note(note, sid, on["memory"])) if memory_ok
                            else None)
            res.setdefault("layers", []).insert(0, frame.layer())
            if not retry:
                personal_step(rt.personal, "after_turn", sid, frame, {"turn_n": n})
            if made:
                res["images"] = [dict(x, url="/api/media/file/" + x["file"]) for x in made]
            if said is not None and said != typed:
                res["heard"] = said
            if kv:
                res["kv_slot"] = kv
        except Exception as exc:
            if S.repo is not None:
                S.repo.turn_error(sid, n, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            S.busy = None
            S.turns += 1
            S.last_turn_end = time.time()
            if kit is not None:
                kit.meaning = None
            if gate is not None:
                gate.release()
    finally:
        S.turn_lock.release()
    reply = str(res.get("reply") or "")
    # a stopped answer that already has text stays in the conversation: "Summarize that story" after Stop got "I don't
    # have access to the story" (2026-11-01)
    if not res.get("cancelled") or (reply.strip() and reply.strip() != "(Stopped.)"):
        with S.sess_lock:
            # retry: a request to try again rather than a question (the next retry looks past it)
            asked = dict({"role": "user", "content": prompt, "retry": how == "said"},
                         **({"media": list(media)} if media else {}))
            kept = reply + ("\n\n(The user stopped this answer here.)" if res.get("cancelled") else "")
            hist += [asked, {"role": "assistant", "content": kept}]
            del hist[:-24]
    if S.repo is not None:  # Growth: what the user's response and this turn taught, shown with the answer
        res["growth"] = list(ppl_growth) + growth + S.repo.turn_end(sid, n, res, time.perf_counter() - t0)
        res["turn"] = n
    elif ppl_growth:
        res["growth"] = list(ppl_growth)
    if S.memidx is not None:
        S.memidx.wake()
    m_id = getattr(eng, "model_key", None) or MODEL_ID
    res.update(session_id=sid, model=m_id, mode="lfm3", wall_ms=round((time.perf_counter() - t0) * 1000.0, 1))
    return res


# --- clover-chat-server.py (d0c535a) /v1/chat/completions message walk, text-only (LFM3 has no image input) ---
def parse_openai_messages(messages):
    workspace_info_chunks = []

    def _extract_msg_content(m_obj):
        c = m_obj.get("content", "")
        if isinstance(c, list):
            t_parts = []
            for part in c:
                if isinstance(part, dict) and str(part.get("type") or "") in ("text", "input_text"):
                    t_parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    t_parts.append(part)
            return "\n".join(p for p in t_parts if p).strip()
        return str(c or "").strip()

    def _strip_vscode_envelope(u_text):
        if not u_text:
            return ""
        for m_ws in re.findall(r"<(?:workspace_info|environment_info)>\s*([\s\S]*?)\s*</(?:workspace_info|environment_info)>",
                               u_text, flags=re.IGNORECASE):
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
                return "---\n".join(att_blocks[:2]) + ATTACH_MARKER + core_q
            return core_q
        cleaned = re.sub(
            r"<(?:context|environment_info|workspace_info|reminderInstructions|instructions|toolUseInstructions|editFileInstructions|outputFormatting|memoryInstructions|userMemory|sessionMemory|repoMemory|conversation-summary)>[\s\S]*?</(?:context|environment_info|workspace_info|reminderInstructions|instructions|toolUseInstructions|editFileInstructions|outputFormatting|memoryInstructions|userMemory|sessionMemory|repoMemory|conversation-summary)>",
            "", u_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<attachment\b[^>]*>[\s\S]*?</attachment>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Relevant context from VS Code editor:\s*(?:The following browser pages[^\r\n]*\s*(?:-\s*[^\r\n]*\s*)*)?",
                         "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    user_prompt, prior_turns, tool_rounds, call_id_to_meta, pending_u = "", [], [], {}, ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").lower()
        txt_c = _extract_msg_content(m)
        if role == "user":
            cleaned_u = _strip_vscode_envelope(txt_c)
            if cleaned_u:
                if not pending_u:
                    tool_rounds.clear()
                    pending_u = user_prompt = cleaned_u
                elif not tool_rounds:
                    pending_u = user_prompt = cleaned_u
        elif role == "assistant":
            t_calls = m.get("tool_calls") or []
            if isinstance(t_calls, list) and t_calls:
                for tc in t_calls:
                    if isinstance(tc, dict):
                        cid_tc = str(tc.get("id") or "")
                        fn_tc = tc.get("function") or {}
                        if cid_tc and isinstance(fn_tc, dict):
                            call_id_to_meta[cid_tc] = {"name": str(fn_tc.get("name") or ""),
                                                       "arguments": str(fn_tc.get("arguments") or "{}")}
            elif pending_u and txt_c:
                prior_turns.append({"user": pending_u, "reply": txt_c, "plans": []})
                pending_u = ""
                tool_rounds.clear()
        elif role == "tool":
            tc_id = str(m.get("tool_call_id") or "")
            meta_tc = call_id_to_meta.get(tc_id, {"name": str(m.get("name") or "tool"), "arguments": "{}"})
            if txt_c:
                tool_rounds.append({"id": tc_id, "name": meta_tc["name"], "arguments": meta_tc["arguments"],
                                    "output": txt_c[:96000]})
    if not user_prompt and messages and isinstance(messages[-1], dict):
        user_prompt = _strip_vscode_envelope(_extract_msg_content(messages[-1])) or "Hello"
    return user_prompt, prior_turns, tool_rounds, "\n\n".join(workspace_info_chunks[-2:])


def openai_images(messages):
    """Images of the newest user message that has any, as data URLs (only data: URLs: nothing is fetched)."""
    for m in reversed(messages or []):
        if not isinstance(m, dict) or str(m.get("role") or "").lower() != "user" or not isinstance(m.get("content"), list):
            continue
        urls = []
        for part in m["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                u = part.get("image_url")
                u = str((u.get("url") if isinstance(u, dict) else u) or "")
                if u.startswith("data:image/"):
                    urls.append(u)
        if urls:
            return urls[:MEDIA_ITEMS]
    return []


def with_tool_outputs(user_prompt, tool_rounds):
    """clover-chat-server.py (d0c535a) Step 2: the final answer sees the verified VS Code tool outputs."""
    if not tool_rounds:
        return user_prompt
    obs_text = "\n\n".join(f"[Step {idx} — {rd['name']}({str(rd['arguments'])[:200]})]:\n{str(rd['output'])[:600]}"
                           for idx, rd in enumerate(tool_rounds[-4:], 1))
    u_cap = 600 if "<pr_description>" in user_prompt else 2400
    return "Verified Workspace Tool Outputs:\n" + obs_text + ATTACH_MARKER + user_prompt[:u_cap]


class StudioHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR let a second Studio bind 127.0.0.1:8765 while one ran; requests then went to either
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


WEB_HOSTS = {
    "api.cloverframework.com", "cloverframework.com", "www.cloverframework.com",
    "127.0.0.1", "localhost",
}
WEB_ORIGINS = {
    "https://cloverframework.com", "https://www.cloverframework.com",
    "https://api.cloverframework.com",
}
DEFAULT_ALLOWED_ACCOUNTS = ("santhoshsathish94",)
_FALLBACK_AUTH_SECRET = secrets.token_hex(32).encode("utf-8")


def allowed_accounts(base=None):
    """Approved accounts (GitHub usernames or emails, lowercase) permitted to sign in to Web Studio, plus any
    per-account SHA-256 key hashes from config/<device>/allowed-accounts.json."""
    accts = {a.lower() for a in DEFAULT_ALLOWED_ACCOUNTS}
    keys = {}
    b = Path(base) if base else (Path(S.args.base) if S.args else None)
    if b is not None:
        fp = paths.config_file(b, "allowed-accounts.json")
        if fp.is_file():
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                for a in (raw.get("accounts") or ()):
                    if isinstance(a, str) and a.strip():
                        accts.add(a.strip().lower())
                for k, v in (raw.get("keys") or {}).items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        accts.add(k.strip().lower())
                        keys[k.strip().lower()] = v.strip().lower()
            except (OSError, ValueError):
                pass
    return accts, keys


def _auth_secret():
    import k3_host_engine
    s = S.token or k3_host_engine.k3_host_key()
    return s.encode("utf-8") if s else _FALLBACK_AUTH_SECRET


def make_account_token(account):
    acct = re.sub(r"[^a-z0-9._@-]", "", str(account or "").strip().lower())[:60]
    sig = hmac.new(_auth_secret(), f"clover-acct:{acct}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"acct:{acct}:{sig}"


def verify_account_token(token, base=None):
    t = str(token or "").strip()
    if not t.startswith("acct:"):
        return None
    parts = t.split(":", 2)
    if len(parts) != 3:
        return None
    _, acct, sig = parts
    accts, _ = allowed_accounts(base)
    if acct not in accts:
        return None
    want = hmac.new(_auth_secret(), f"clover-acct:{acct}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return acct if secrets.compare_digest(sig.encode("utf-8"), want.encode("utf-8")) else None


def authenticate_web_account(account, key, base=None):
    """Verifies that `account` is on the approved allowlist and `key` is a valid access key or GitHub token for that
    approved account. Returns (ok, account_or_error_msg, status_code)."""
    import k3_host_engine
    import urllib.error
    import urllib.request
    acct = str(account or "").strip().lower().lstrip("@")
    secret_in = str(key or "").strip()
    if not acct or not secret_in:
        return False, "Enter both your approved account username and access key.", 400
    accts, acct_keys = allowed_accounts(base)
    if acct not in accts:
        return False, f"Account '{acct}' does not have access to Clover AI Studio. Only approved accounts are allowed.", 403
    owner_key = k3_host_engine.k3_host_key()
    if (
        (S.token and secrets.compare_digest(secret_in.encode("utf-8"), S.token.encode("utf-8")))
        or (owner_key and secrets.compare_digest(secret_in.encode("utf-8"), owner_key.encode("utf-8")))
        or secrets.compare_digest(secret_in.encode("utf-8"), make_account_token(acct).encode("utf-8"))
    ):
        return True, acct, 200
    want_hash = acct_keys.get(acct)
    if want_hash:
        got_hash = hashlib.sha256(secret_in.encode("utf-8")).hexdigest().lower()
        if secrets.compare_digest(got_hash.encode("utf-8"), want_hash.encode("utf-8")):
            return True, acct, 200
    if secret_in.startswith(("ghp_", "github_pat_", "gho_", "ghu_")):
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {secret_in}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "CloverStudio/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                gh_user = json.loads(r.read().decode("utf-8", "replace"))
            gh_login = str(gh_user.get("login") or "").strip().lower()
            if not gh_login or gh_login != acct or gh_login not in accts:
                return False, f"GitHub account '@{gh_login or acct}' is not authorized for Clover AI Studio.", 403
            return True, gh_login, 200
        except urllib.error.HTTPError:
            return False, "GitHub rejected the access token.", 401
        except Exception:
            return False, "Could not verify GitHub token right now.", 502
    return False, "Invalid access key for this account.", 401


class Handler(BaseHTTPRequestHandler):
    server_version = "CloverStudio/1.0"  # HTTP/1.0: one request per connection, so a refused body is never re-read

    def log_message(self, fmt, *args):
        sys.stdout.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fmt % args))

    def _cors_headers(self):
        origin = (self.headers.get("Origin") or "").strip()
        if not paths.WEB:
            # a public website must not read a laptop's local server (Windows sets no token)
            return {}
        if origin not in WEB_ORIGINS and not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
            origin = "https://cloverframework.com"
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Clover-Key, X-Clover-Token, X-Clover-Bridge, X-Clover-User, X-Clover-Team",
            "Vary": "Origin",
        }

    # ---------- transport ----------
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("application/json", "text/")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for k, v in {**self._cors_headers(), **(extra or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden host"})
        return self._send(200, {"ok": True})

    def _host_ok(self):
        host = (self.headers.get("Host") or "").strip().lower()
        if not host or host in (f"127.0.0.1:{S.port}", f"localhost:{S.port}"):
            return True
        if paths.WEB and host.split(":", 1)[0] in WEB_HOSTS:
            return True
        return False

    def _token_ok(self):
        """The phone app's WebView carries the per-launch token as a cookie; on k3 Web, only signed-in approved
        accounts and owner devices with X-Clover-Key are allowed (no public anonymous access)."""
        self._authed_account = None
        if not paths.WEB and not S.token:
            return True
        m = re.search(r"(?:^|;\s*)clover_token=([A-Za-z0-9_:\-.@]+)", self.headers.get("Cookie") or "")
        auth = (self.headers.get("Authorization") or "").strip()
        bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        got = (
            (m.group(1) if m else "")
            or (self.headers.get("X-Clover-Key") or "").strip()
            or (self.headers.get("X-Clover-Token") or "").strip()
            or bearer
        )
        if not got:
            return False
        acct = verify_account_token(got)
        if acct:
            self._authed_account = acct
            return True
        if S.token and secrets.compare_digest(got.encode("utf-8"), S.token.encode("utf-8")):
            self._authed_account = DEFAULT_ALLOWED_ACCOUNTS[0]
            return True
        import k3_host_engine
        owner_key = k3_host_engine.k3_host_key()
        if owner_key and secrets.compare_digest(got.encode("utf-8"), owner_key.encode("utf-8")):
            self._authed_account = DEFAULT_ALLOWED_ACCOUNTS[0]
            return True
        return False

    def _cross_site(self):
        origin = (self.headers.get("Origin") or "").strip()
        if paths.WEB:
            if not origin or origin in WEB_ORIGINS or origin.startswith(("http://127.0.0.1:", "http://localhost:")):
                return False
            return True
        if origin and origin not in (f"http://127.0.0.1:{S.port}", f"http://localhost:{S.port}"):
            return True
        return self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none")

    def _read_json(self, limit=MAX_BODY):
        if not str(self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            raise ValueError("Content-Type must be application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > limit:
            raise ValueError(f"body size must be 1..{limit} bytes")
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("JSON object expected")
        return body

    def _bearer_token(self) -> str:
        auth = (self.headers.get("Authorization") or "").strip()
        return auth[7:].strip() if auth.lower().startswith("bearer ") else ""

    def _serve_hf_mirror(self, path: str):
        """Read-only SHA256-pinned setup artifact mirror for networks where DNS/firewalls (e.g. Cisco Umbrella) block
        huggingface.co. Every file downloaded from this mirror is verified against its publisher SHA-256 by setup_fetch.py."""
        rel = path[len("/api/hf-mirror/"):].lstrip("/")
        if not rel or ".." in rel.split("/"):
            return self._send(400, {"error": "invalid mirror path"})
        k3_data = Path("/opt/clover-k3/core-observe/laptop-base/data")
        base_data = k3_data if k3_data.is_dir() else (Path(S.args.base) / "data")
        k3_pre = Path("/home/santhosh/oss/clover-framework/clover-ai-system/src/models/lfm3-1.7b/models")
        precomp = k3_pre if k3_pre.is_dir() else (Path(S.args.base) / "data" / "omni" / "precomputed")
        if (precomp / "models").is_dir():
            precomp = precomp / "models"

        if rel.startswith("api/datasets/Santhoshsathish94/clover-ai-studio/tree/") and rel.endswith("/liquid-lfm3"):
            if not precomp.is_dir():
                return self._send(404, {"error": "precomputed dataset not found"})
            cache_f = Path("/tmp/clover_hf_mirror_tree_v5.json") if os.name != "nt" else (Path(S.args.base) / "cache" / "hf_mirror_tree.json")
            if cache_f.is_file():
                try:
                    return self._send(200, cache_f.read_bytes(), "application/json")
                except OSError:
                    pass
            inc = ("hybridagent/plan-trunk-0/", "bridges/bridge-trunk-0/", "ncm/", "trunks/tokenizer.json", "trunks/global-0/embed_tokens.bf16.bin")
            entries = []
            for fp in sorted(precomp.rglob("*")):
                if not fp.is_file() or fp.name.endswith(".pt"):
                    continue
                r_posix = fp.relative_to(precomp).as_posix()
                if not any(r_posix == p or r_posix.startswith(p) for p in inc):
                    continue
                h = hashlib.sha256()
                with open(fp, "rb") as f:
                    for ch in iter(lambda: f.read(1 << 20), b""):
                        h.update(ch)
                sz = fp.stat().st_size
                entries.append({
                    "type": "file",
                    "path": "liquid-lfm3/" + r_posix,
                    "size": sz,
                    "lfs": {"oid": h.hexdigest(), "size": sz},
                })
            raw = json.dumps(entries).encode("utf-8")
            try:
                cache_f.parent.mkdir(parents=True, exist_ok=True)
                cache_f.write_bytes(raw)
            except OSError:
                pass
            return self._send(200, raw, "application/json")

        target = None
        m_ds = re.match(r"^datasets/Santhoshsathish94/clover-ai-studio/resolve/[^/]+/liquid-lfm3/(.+)$", rel)
        if m_ds:
            target = precomp / m_ds.group(1)
        m_csi = re.match(r"^datasets/Santhoshsathish94/clover-ai-studio/resolve/[^/]+/csi/morph-verb-v1\.1/([^/]+)$", rel)
        if m_csi:
            target = base_data / "csi" / "morph-verb-v1.1" / m_csi.group(1)
        m_bge = re.match(r"^BAAI/bge-small-en-v1\.5/resolve/[^/]+/([^/]+)$", rel)
        if m_bge:
            target = base_data / "embed" / "bge-small-en-v1.5" / m_bge.group(1)
        m_gguf = re.match(r"^LiquidAI/LFM3-Omni-2B-GGUF/resolve/[^/]+/([^/]+)$", rel)
        if m_gguf:
            target = base_data / "omni" / "gguf" / m_gguf.group(1)
        if target is None or not target.is_file():
            return self._send(404, {"error": "mirror file not found"})

        total = target.stat().st_size
        start = 0
        rng = (self.headers.get("Range") or "").strip()
        m_rng = re.match(r"^bytes=(\d+)-$", rng)
        if m_rng:
            start = int(m_rng.group(1))
            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
        self.send_response(206 if start > 0 else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(total - start))
        self.send_header("Accept-Ranges", "bytes")
        if start > 0:
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
        self.end_headers()
        with open(target, "rb") as f:
            if start > 0:
                f.seek(start)
            while True:
                buf = f.read(1 << 20)
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    break

    # ---------- GET ----------
    def do_GET(self):
        path, _, query = self.path.partition("?")
        if "passive=1" not in query.split("&"):  # VS Code's status bar polls passively: it must not keep the engine alive
            S.last_request = time.time()
        if not self._host_ok():
            return self._send(403, {"error": "forbidden host"})
        if path.startswith("/api/hf-mirror/"):
            return self._serve_hf_mirror(path)
        if path == "/api/identity/google/callback":
            from urllib.parse import parse_qs
            import google_identity
            q = {k: v[0] for k, v in parse_qs(query).items()}
            mgr = google_identity_manager()
            if mgr is None:
                return self._send(500, {"error": "Google identity manager unavailable"})
            if q.get("error"):
                html_err = (
                    "<!doctype html><meta charset=utf-8><title>Google Sign-In Cancelled</title>"
                    "<body style='font:15px system-ui,sans-serif;background:#F7F3E9;color:#39473D;padding:2rem;text-align:center'>"
                    f"<h3>Google sign-in was not completed ({re.sub(r'[^A-Za-z0-9_ -]', '', str(q.get('error'))[:60])})</h3>"
                    "<p>You can close this window and return to Clover AI Studio.</p></body>"
                )
                return self._send(400, html_err.encode("utf-8"), "text/html")
            try:
                mgr.complete_pkce_callback(state=str(q.get("state") or ""), code=str(q.get("code") or ""))
                if S.repo is not None:
                    S.repo.note("Google identity connected via OIDC PKCE")
                html_ok = (
                    "<!doctype html><meta charset=utf-8><title>Connected to Clover AI Studio</title>"
                    "<body style='font:15px system-ui,sans-serif;background:#F7F3E9;color:#39473D;padding:2rem;text-align:center'>"
                    "<h3>Google identity connected to Clover AI Studio</h3>"
                    "<p>Collective learning remains off until you explicitly enable it in Settings &rarr; Learning &amp; Growth.</p>"
                    "<p>You can close this tab and return to Clover AI Studio.</p></body>"
                )
                return self._send(200, html_ok.encode("utf-8"), "text/html")
            except google_identity.GoogleIdentityError as exc:
                return self._send(400, {"error": str(exc)})
        if path == "/api/collective/gateway/releases":
            from urllib.parse import parse_qs
            import collective_gateway
            q = {k: v[0] for k, v in parse_qs(query).items()}
            gw_rt = collective_gateway_runtime()
            if gw_rt is None:
                return self._send(503, {"error": "Collective gateway runtime unavailable"})
            bearer = self._bearer_token()
            try:
                gw_rt["gateway"].authenticate_contributor(bearer)
            except collective_gateway.GatewaySecurityError as exc:
                return self._send(exc.status, {"error": exc.message, "code": exc.code})
            with gw_rt["lock"]:
                bundle = gw_rt["registry"].get_release_bundle(since_version=q.get("since_version"))
            return self._send(200, bundle)
        if not paths.WEB and not self._token_ok():
            return self._send(403, {"error": "token required"})
        if path in ("/", "/index.html") and setup_needed():
            nonce = secrets.token_urlsafe(18)
            html = (HERE / "web" / "setup.html").read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            return self._send(200, html.encode("utf-8"), "text/html",
                              {"Content-Security-Policy": csp, "X-Frame-Options": "DENY"})
        if path in ("/", "/index.html"):
            nonce = secrets.token_urlsafe(18)
            html = (HERE / "web" / "app.html").read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
            html = html.replace("__CLOVER_PLATFORM__", paths.PLATFORM)
            conn = "'self' https: http:"
            csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; "
                   f"img-src 'self' data: blob:; connect-src {conn}; media-src 'self' data: blob:; base-uri 'none'; "
                   "form-action 'none'; frame-ancestors 'none'")
            return self._send(200, html.encode("utf-8"), "text/html",
                              {"Content-Security-Policy": csp, "X-Frame-Options": "DENY"})
        if path in STATIC:
            name, ctype = STATIC[path]
            f = HERE / "web" / name
            return self._send(200, f.read_bytes(), ctype) if f.is_file() else self._send(404, {"error": "not found"})
        if paths.WEB and not self._token_ok():
            return self._send(401, {"error": "Sign in with an approved account to use Clover AI Studio.", "auth_required": True})
        if path == "/api/auth/session":
            return self._send(200, {"authenticated": True, "account": getattr(self, "_authed_account", None) or DEFAULT_ALLOWED_ACCOUNTS[0]})
        if path == "/api/setup":  # the phone app's first-start download
            snap = S.setup.snapshot() if S.setup is not None else {"state": "idle"}
            base = Path(S.args.base)
            return self._send(200, dict(snap, needed=setup_needed(), android=paths.ANDROID,
                                        free_bytes=shutil.disk_usage(base).free if base.exists() else None))
        if path == "/api/health":
            eng = engine_for()
            desc = eng.describe() if eng else None
            return self._send(200, {"status": S.load["status"], "version": STUDIO_VERSION, "channel": RELEASE_CHANNEL,
                                    "model": (desc or {}).get("model") or MODEL_ID,
                                    "mode": S.mode, "load_step": S.load["step"],
                                    "error": S.load["error"], "engine": desc,
                                    "uptime_s": round(time.time() - S.started), "turns": S.turns,
                                    "recall": S.memidx.describe() if S.memidx else None,
                                    "csi": S.csi.describe() if S.csi is not None else None, "hardware": S.hardware,
                                    "tool_inventory": tool_inventory()})
        if path == "/api/status":
            eng, busy = engine_for(), S.busy
            desc = eng.describe() if eng else {}
            cat = model_catalog()
            mkey = desc.get("model_key") or (S.settings.get().get("model") if S.settings else "omni")
            st = {"status": "ready" if (eng is not None and mkey != "omni") else S.load["status"],
                  "version": STUDIO_VERSION, "channel": RELEASE_CHANNEL,
                  "mode": S.mode, "load_step": S.load["step"], "error": S.load["error"],
                  "load_elapsed_s": round(time.time() - S.load["since"]), "busy": None, "warning": S.load.get("warning"),
                  "engine": getattr(eng, "engine_kind", None),
                  "model": desc.get("model") or (("LFM3-Omni-2B" if getattr(eng, "inputs", None) is not None else "LFM3-1.7B") if eng else None),
                  "model_key": mkey,
                  "models": [x["id"] for x in cat],
                  "model_catalog": cat,
                  "account": getattr(self, "_authed_account", None),
                  "inputs": getattr(eng, "inputs", ["text"]) if eng else None}
            if busy:
                st["busy"] = {"kind": busy["kind"], "elapsed_s": round(time.time() - busy["since"]),
                              "stage": eng.stage["name"] if eng else None,
                              "stage_elapsed_s": round(time.time() - eng.stage["since"]) if eng else None}
            return self._send(200, st)
        if path == "/api/permissions":
            return self._send(200, {"permissions": S.perms.describe()})
        if path == "/api/settings":
            ok, why = gpu_available(S.args.base)
            o_ok, o_why = omni_available(S.args.base)
            import fused_engine
            import k3_host_engine
            import omni_c2_c3
            f_need = fused_engine.missing(S.args.base)
            c2c3_need = omni_c2_c3.missing_c2_c3_files(S.args.base)
            cat = model_catalog()
            owner = k3_host_engine.is_owner_device(S.args.base)
            return self._send(200, {"settings": S.settings.get(), "version": STUDIO_VERSION, "channel": RELEASE_CHANNEL,
                                    "privacy_policy_url": PRIVACY_URL, "support_url": SUPPORT_URL,
                                    "models": [x["id"] for x in cat], "model_catalog": cat,
                                    "owner": owner,
                                    **({"k3_key": k3_host_engine.k3_host_key()} if (owner and not paths.WEB and k3_host_engine.k3_host_key()) else {}),
                                    "gpu_available": ok, "gpu_reason": why,
                                    "omni_available": o_ok, "omni_reason": o_why, "warning": S.load.get("warning"),
                                    "c2c3_available": not c2c3_need,
                                    "c2c3_reason": ("missing " + ", ".join(c2c3_need)) if c2c3_need else "",
                                    "fused_available": not f_need,
                                    "fused_reason": ("missing " + ", ".join(f_need)) if f_need else "",
                                    "csi_status": S.csi.describe() if S.csi is not None else None,
                                    "hardware": S.hardware})
        if path == "/api/memory":
            return self._send(200, {"memories": layer_kit().memories()})
        if path == "/api/identity/google":
            mgr = google_identity_manager()
            return self._send(200, mgr.status() if mgr else {"connected": False, "status_text": "Not connected"})
        if path == "/api/personal":  # what Clover learned about the user, from which experiences
            pers = personal_store()
            return self._send(200, pers.view() if pers else
                              {"enabled": False, "how": "Personal learning is off for this process (CLOVER_PPL=off)."})
        if path == "/api/collective":  # collective learning: consent switches, candidate preview, active shared-core rules
            coll = collective_store()
            mgr = google_identity_manager()
            g_stat = mgr.status() if mgr else {"connected": False, "status_text": "Not connected"}
            p_on = bool(S.settings.get().get("personal", True)) if S.settings else True
            return self._send(200, dict(coll.view(personal_store(), google_status=g_stat), personal_enabled=p_on) if coll else
                              {"enabled": False, "google_identity": g_stat, "personal_enabled": p_on,
                               "how": "Collective learning is off for this process (CLOVER_COLLECTIVE=off)."})
        if path == "/api/sessions":  # conversations in the repository, newest first (titles, not contents)
            return self._send(200, {"sessions": [{k: s[k] for k in ("id", "title", "turns", "started", "updated")}
                                                 for s in S.repo.sessions()[:300]]})
        if path.startswith("/api/sessions/"):
            sid = path.rsplit("/", 1)[1]
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,96}", sid):
                return self._send(400, {"error": "bad conversation id"})
            return self._send(200, {"id": sid, "messages": S.repo.messages(sid)})
        if path.startswith("/api/media/file/"):  # pictures Clover made: content-named files in the media folder only
            name = path.rsplit("/", 1)[1]
            if not re.fullmatch(r"[0-9a-f]{32}\.png", name):
                return self._send(400, {"error": "bad file name"})
            f = media_dir() / name
            return self._send(200, f.read_bytes(), "image/png") if f.is_file() else self._send(404, {"error": "not found"})
        if path == "/api/media":
            files = [p for p in media_dir().iterdir() if p.is_file()]
            return self._send(200, {"path": str(media_dir()), "files": len(files),
                                    "bytes": sum(p.stat().st_size for p in files)})
        if path == "/api/storage":  # where things sit; the user opens the folders in File Explorer
            base, files = Path(S.args.base), [p for p in media_dir().iterdir() if p.is_file()]
            return self._send(200, {"repository": str(paths.repo_dir(base)), "memory": str(paths.knowledge_dir(base)),
                                    "media": str(media_dir()), "models": str(base / "data"),
                                    "media_files": len(files), "media_bytes": sum(p.stat().st_size for p in files)})
        if path == "/api/search":
            return self._send(200, S.search.describe())
        if path == "/api/workspaces":  # whether a GitHub token is set, never the token
            return self._send(200, S.workspaces.describe())
        if path == "/api/vscode/poll":  # the Clover extension in a VS Code window asks for its next job
            if not S.vscode.check(self.headers.get("X-Clover-Bridge")):
                return self._send(403, {"error": "bridge token required"})
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(query).items()}
            try:
                focus, wait = float(q.get("focus") or 0), float(q.get("wait") or 25)
            except ValueError:
                return self._send(400, {"error": "focus and wait are numbers"})
            job = S.vscode.poll(q.get("window", "")[:80] or "window", focus, q.get("title", ""), q.get("version", ""), wait)
            return self._send(200, {"job": job})
        if path == "/api/vscode":
            return self._send(200, {"connected": S.vscode.connected(), "repo": github_repo(),
                                    "local": str(S.repo.dir), "installed": vscode_exe() is not None})
        if path == "/api/turn/events":
            from urllib.parse import parse_qs
            q = {k: v[0] for k, v in parse_qs(query).items()}
            tid = str(q.get("turn_id") or "").strip()
            try:
                after = int(q.get("after") or self.headers.get("Last-Event-ID") or 0)
            except ValueError:
                after = 0
            with S.turns_lock:
                t_obj = S.active.get(tid) or S.recent_turns.get(tid)
                if not t_obj:
                    return self._send(404, {"error": "unknown turn_id"})
                evs = [e for e in t_obj["events"] if int(e.get("seq") or 0) > after]
                return self._send(200, {
                    "turn_id": tid,
                    "session_id": t_obj.get("session_id"),
                    "active": tid in S.active,
                    "done": bool(t_obj.get("done")),
                    "cancelled": t_obj["cancel"].is_set(),
                    "last_seq": int(t_obj.get("seq") or 0),
                    "events": evs,
                })
        if path in ("/v1/models", "/models"):
            return self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "clover-local"}]})
        self._send(404, {"error": "not found"})

    # ---------- POST ----------
    def do_POST(self):
        S.last_request = time.time()
        if not self._host_ok() or self._cross_site():
            return self._send(403, {"error": "cross-site requests are not accepted"})
        path = self.path.split("?", 1)[0]
        if path in ("/api/collective/gateway/contribute", "/api/collective/gateway/withdraw"):
            import collective_gateway
            try:
                body = self._read_json(MAX_BODY)
            except Exception as exc:
                return self._send(400, {"error": str(exc)})
            gw_rt = collective_gateway_runtime()
            if gw_rt is None:
                return self._send(503, {"error": "Collective gateway runtime unavailable"})
            bearer = self._bearer_token()
            try:
                with gw_rt["lock"]:
                    if path.endswith("/contribute"):
                        res = gw_rt["gateway"].submit_authenticated_contribution(id_token=bearer, payload=body)
                    else:
                        res = gw_rt["gateway"].withdraw_authenticated_contributor(id_token=bearer, payload=body)
                    agg = gw_rt["gateway"].aggregate_candidates()
                    eval_res = gw_rt["evaluator"].evaluate_all(agg)
                    res["evaluated_count"] = eval_res["total_evaluated"]
                    res["promoted_count"] = eval_res["promoted_count"]
                    if gw_rt["registry"]._seed is not None:
                        note = "Collective gateway evaluation" if path.endswith("/contribute") else "Contributor withdrawal re-evaluation"
                        manifest = gw_rt["registry"].publish_release(eval_res, note=note)
                        res["release_version"] = manifest["version"]
                return self._send(200, res)
            except collective_gateway.GatewaySecurityError as exc:
                return self._send(exc.status, {"error": exc.message, "code": exc.code})
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        if path == "/api/auth/login":
            try:
                body = self._read_json(MAX_BODY)
            except Exception as exc:
                return self._send(400, {"error": str(exc)})
            ok, msg_or_acct, code = authenticate_web_account(body.get("account"), body.get("key"), S.args.base if S.args else None)
            if not ok:
                return self._send(code, {"error": msg_or_acct, "auth_required": True})
            tok = make_account_token(msg_or_acct)
            return self._send(200, {"ok": True, "account": msg_or_acct, "token": tok})
        if not self._token_ok():
            return self._send(
                401 if paths.WEB else 403,
                {"error": "Sign in with an approved account to use Clover AI Studio.", "auth_required": True}
                if paths.WEB else {"error": "token required"},
            )
        try:
            body = self._read_json(CHAT_BODY if path in ("/api/chat/stream", "/api/worker/lfm3", "/api/worker/chat",
                                                         "/v1/chat/completions", "/chat/completions")
                                   else MAX_BODY)
        except Exception as exc:
            return self._send(400, {"error": str(exc)})
        try:
            if path in ("/api/setup/start", "/api/setup/stop"):
                if S.setup is None:
                    return self._send(400, {"error": "setup runs in the installer on this computer"})
                if path.endswith("/stop"):
                    S.setup.stop.set()
                    return self._send(200, {"ok": True})
                comps = str((body or {}).get("components") or "").strip() or None
                can_run = setup_needed() or bool(comps)
                return self._send(200, {"started": S.setup.start(components=comps) if can_run else False})
            if path == "/api/worker/lfm3":
                if S.omni_engine is None or getattr(S.omni_engine, "gpu", None) is None:
                    raise EngineNotReady("LFM3-Omni-2B worker is not ready")
                for fname, b64 in (body.get("media_files") or {}).items():
                    if re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{2,5}", str(fname)):
                        (media_dir() / fname).write_bytes(base64.b64decode(str(b64)))
                chat_body = body.get("body") if isinstance(body.get("body"), dict) else body
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                for k, v in self._cors_headers().items():
                    self.send_header(k, v)
                self.end_headers()
                for ev in S.omni_engine.gpu.chat_stream(chat_body):
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return None
            if path == "/api/worker/chat":
                import k3_host_engine
                import urllib.request
                req = urllib.request.Request(
                    f"{k3_host_engine.WORKER_URL}/api/chat",
                    json.dumps(body).encode("utf-8"),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=240) as resp:
                    return self._send(200, resp.read(), "application/json")
            if path in ("/api/worker/csi", "/api/worker/csi/analyze"):
                if S.csi is None or not S.csi.alive() or not S.csi.url.startswith("http://127.0.0.1:"):
                    return self._send(503, {"error": "CSI service is not ready"})
                import urllib.request
                req = urllib.request.Request(
                    S.csi.url.rstrip("/") + "/analyze",
                    json.dumps({"text": str(body.get("text") or "")[:4000]}).encode("utf-8"),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return self._send(200, resp.read(), "application/json")
            if path == "/api/models/add":
                import k3_host_engine
                import vault
                url = str(body.get("url") or "").strip()
                api_key = str(body.get("api_key") or "").strip()
                model_hint = str(body.get("model") or "").strip()
                name_hint = str(body.get("name") or "").strip()
                entries = k3_host_engine.probe_remote_server(url, api_key=api_key, model_hint=model_hint, name_hint=name_hint)
                key_id = ""
                if api_key:
                    key_id = "rm_k3" if entries[0].get("host") == "k3" else (
                        "rm_" + hashlib.sha256(entries[0]["url"].encode("utf-8")).hexdigest()[:12]
                    )
                    vault.put(S.args.base, key_id, api_key)
                cur = list(S.settings.get().get("remote_models") or [])
                new_ids = {e["id"] for e in entries}
                cur = [x for x in cur if x.get("id") not in new_ids]
                for e in entries:
                    if key_id:
                        e["key_id"] = key_id
                    S.remote_engines.pop(e["id"], None)
                    cur.append(e)
                pick_id = entries[0]["id"]
                new = S.settings.update({"remote_models": cur, "model": pick_id})
                S.engine = engine_for(pick_id)
                if S.repo is not None:
                    S.repo.note("hosted model added by the user: " + ", ".join(e["name"] for e in entries))
                cat = model_catalog()
                return self._send(200, {"ok": True, "added": entries, "settings": new,
                                        "models": [x["id"] for x in cat], "model_catalog": cat})
            if path == "/api/models/remove":
                import vault
                rm_id = str(body.get("id") or "").strip()
                cur = list(S.settings.get().get("remote_models") or [])
                removed = [x for x in cur if x.get("id") == rm_id]
                kept = [x for x in cur if x.get("id") != rm_id]
                for r in removed:
                    kid = r.get("key_id")
                    if kid and not any(x.get("key_id") == kid for x in kept):
                        vault.delete(S.args.base, kid)
                S.remote_engines.pop(rm_id, None)
                next_model = S.settings.get().get("model") or "omni"
                if next_model == rm_id:
                    next_model = "omni"
                new = S.settings.update({"remote_models": kept, "model": next_model})
                S.engine = engine_for(next_model)
                cat = model_catalog()
                return self._send(200, {"ok": True, "settings": new, "models": [x["id"] for x in cat], "model_catalog": cat})
            if path == "/api/chat/stream" and S.mode == "lfm3":
                return self._lfm3_stream(body)
            if path == "/api/tts/parts":
                import omni_native
                return self._send(200, {"parts": omni_native.speech_parts(str(body.get("text") or ""))})
            if path == "/api/tts/warm":
                eng = S.engine
                if eng is not None and hasattr(eng, "warm_voice"):
                    eng.warm_voice()
                return self._send(200, {"ok": True})
            if path == "/api/tts":
                eng = S.engine
                if eng is None or not hasattr(eng, "tts"):
                    return self._send(400, {"error": "speech needs LFM3-Omni-2B (Settings -> Model)"})
                voice = "male" if body.get("voice") == "male" else "female"
                return self._send(200, eng.tts(str(body.get("text") or ""), voice=voice), "audio/wav")
            if path == "/api/decision":
                turn = S.active.get(str(body.get("turn_id") or ""))
                pend = turn and turn["pending"].get(str(body.get("request_id") or ""))
                if not pend:
                    return self._send(404, {"error": "no such permission request"})
                decision = str(body.get("decision") or "deny")
                pend["decision"] = decision if decision in ("once", "always", "deny") else "deny"
                pend["event"].set()
                return self._send(200, {"ok": True})
            if path in ("/api/cancel", "/api/stop"):
                tid = str(body.get("turn_id") or "").strip()
                cancelled_ids = []
                with S.turns_lock:
                    targets = [S.active[tid]] if (tid and tid != "all" and tid in S.active) else (
                        list(S.active.values()) if not tid or tid == "all" else []
                    )
                for t_obj in targets:
                    t_obj["cancel"].set()
                    for pend in list(t_obj["pending"].values()):
                        pend["event"].set()
                    cancelled_ids.append(t_obj.get("turn_id") or tid)
                return self._send(200, {"ok": bool(cancelled_ids), "cancelled_turns": cancelled_ids})
            if path == "/api/chat":
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    return self._send(400, {"error": "prompt is required"})
                sid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(body.get("session_id") or uuid.uuid4().hex))[:96]
                if S.mode == "lfm3":
                    docs = save_documents(body.get("documents"))
                    return self._send(200, lfm3_turn(sid, build_prompt(prompt, body.get("attached_files"), docs), body,
                                                     ask_text=prompt, docs=docs))
                seed = body.get("history")
                if isinstance(seed, list):  # the Studio keeps chats; after a server restart it re-seeds the context
                    with S.sess_lock:
                        if sid not in S.sessions:
                            if len(S.sessions) >= MAX_SESSIONS:
                                S.sessions.pop(next(iter(S.sessions)))
                            S.sessions[sid] = [{"user": str(h["user"])[:4000], "reply": str(h["reply"])[:4000], "plans": []}
                                               for h in seed[-6:] if isinstance(h, dict) and h.get("user") and h.get("reply")]
                res = run_turn(sid, build_prompt(prompt, body.get("attached_files")),
                               remember=bool(body.get("remember", True)),
                               allow_once=[str(x) for x in body.get("allow_once") or []],
                               deny_once=[str(x) for x in body.get("deny_once") or []], interactive=True)
                return self._send(200, res)
            if path == "/api/permissions":
                changes = body.get("permissions")
                before = {k: v["mode"] for k, v in S.perms.describe().items()}
                S.perms.update(changes if isinstance(changes, dict) else {})
                after = {k: v["mode"] for k, v in S.perms.describe().items()}
                diff = [f"{k} {before.get(k)} -> {v}" for k, v in after.items() if before.get(k) != v]
                if diff:
                    S.repo.note("permissions changed by the user: " + ", ".join(diff))
                return self._send(200, {"permissions": S.perms.describe()})
            if path == "/api/search":  # the answer reports whether a key is set, never the key
                return self._send(200, S.search.update(provider=body.get("provider"), key=body.get("key"),
                                                       forget=body.get("forget"), browser=body.get("browser")))
            if path == "/api/browser/open":
                return self._send(200, {"opened": open_in_browser(body)})
            if path == "/api/app/open":
                import device_actions
                res_open = device_actions.open_target(body.get("target") or body.get("url") or "", Path(S.args.base) if S.args else None)
                return self._send(200, res_open)
            if path == "/api/screen/capture":
                import device_actions
                rep = device_actions.capture_screenshot(Path(S.args.base))
                img_bytes = (media_dir() / rep["file"]).read_bytes()
                rep["data"] = base64.b64encode(img_bytes).decode("ascii")
                return self._send(200, rep)
            if path == "/api/search/test":
                import web_search
                ok, message = web_search.WebSearch(S.search).test()
                return self._send(200, {"ok": ok, "message": message})
            if path.startswith("/api/workspaces/"):
                return self._send(200, workspace_action(path.rsplit("/", 1)[1], body))
            if path == "/api/github":  # the answer says whether a token is set, never the token
                S.github.update(token=body.get("token"), forget=bool(body.get("forget")))
                return self._send(200, S.workspaces.describe())
            if path == "/api/vscode/result":
                if not S.vscode.check(self.headers.get("X-Clover-Bridge")):
                    return self._send(403, {"error": "bridge token required"})
                S.vscode.result(str(body.get("id") or ""), bool(body.get("ok")), body.get("result"), body.get("error"))
                return self._send(200, {"ok": True})
            if path == "/api/vscode/do":  # buttons in Settings
                action = str(body.get("action") or "")
                if action not in UI_VSCODE_ACTIONS:
                    raise ValueError(f"unknown VS Code action {action!r}")
                try:
                    res = (github_setup() if action == "githubSetupRepo" else
                           vscode_do(action, body.get("args") if isinstance(body.get("args"), dict) else {}))
                except RuntimeError as exc:
                    return self._send(409, {"error": str(exc)})
                return self._send(200, {"result": res, "repo": github_repo(), "connected": S.vscode.connected()})
            if path == "/api/vscode/sync":
                try:
                    return self._send(200, {"result": sync_repo(), "repo": github_repo()})
                except RuntimeError as exc:
                    return self._send(409, {"error": str(exc)})
            if path in ("/api/identity/google/start", "/api/identity/google/token", "/api/identity/google/disconnect"):
                import google_identity
                mgr = google_identity_manager()
                if mgr is None:
                    return self._send(500, {"error": "Google identity manager unavailable"})
                if path.endswith("/start"):
                    redir = str(body.get("redirect_uri") or f"http://127.0.0.1:{S.port}/api/identity/google/callback").strip()
                    res = mgr.start_pkce_flow(redirect_uri=redir, client_id=body.get("client_id"))
                    if body.get("open_browser") is True and os.name == "nt":
                        import webbrowser
                        try:
                            webbrowser.open(res["auth_url"])
                        except Exception:
                            pass
                    return self._send(200, res)
                if path.endswith("/token"):
                    id_tok = str(body.get("id_token") or "").strip()
                    nonce = body.get("nonce")
                    try:
                        st = mgr.connect_with_id_token(id_tok, expected_nonce=nonce if nonce else None)
                    except google_identity.GoogleIdentityError as exc:
                        return self._send(400, {"error": str(exc)})
                    if S.repo is not None:
                        S.repo.note("Google identity connected via OIDC ID token")
                    coll = collective_store()
                    return self._send(200, {"ok": True, "google_identity": st,
                                            "collective": coll.view(personal_store(), google_status=st) if coll else None})
                st = mgr.disconnect()
                coll = collective_store()
                if coll is not None:
                    coll.set_consent(contribute=False, receive=False)
                if S.settings is not None:
                    S.settings.update({"collective_contribute": False, "collective_receive": False})
                if S.repo is not None:
                    S.repo.note("Google identity disconnected; collective learning disabled")
                return self._send(200, {"ok": True, "google_identity": st,
                                        "collective": coll.view(personal_store(), google_status=st) if coll else None})
            if path == "/api/settings":
                if isinstance(body, dict) and (body.get("collective_contribute") is True or body.get("collective_receive") is True):
                    mgr = google_identity_manager()
                    g_ok = bool(mgr and mgr.status().get("connected"))
                    if not g_ok:
                        raise ValueError("Sign in with Google first to enable collective learning.")
                before = S.settings.get()
                new = S.settings.update(body if isinstance(body, dict) else {})
                if S.mode == "lfm3" and new["engine"] != before["engine"]:
                    if S.busy:
                        S.settings.update({"engine": before["engine"], "model": before["model"]})
                        return self._send(409, {"error": "Clover is answering; switch when it is done"})
                    old, S.engine, S.omni_engine = S.omni_engine or S.engine, None, None
                    S.load = {"status": "loading", "step": "switching", "since": time.time(), "error": None}
                    if old is not None and hasattr(old, "close"):
                        old.close()
                    threading.Thread(target=load_engine, args=(S.args,), daemon=True, name="engine-switch").start()
                elif S.mode == "lfm3" and ("model" in (body or {}) or new["model"] != before["model"]):
                    S.engine = engine_for(new["model"])
                changed = {k: v for k, v in new.items() if before.get(k) != v}
                if changed:
                    S.repo.note("settings changed by the user: " + json.dumps(changed))
                if "personal" in changed:
                    personal_step(personal_store(), "set_learning", new["personal"])
                if "collective_contribute" in changed or "collective_receive" in changed:
                    coll = collective_store()
                    if coll is not None:
                        coll.set_consent(contribute=new.get("collective_contribute"),
                                         receive=new.get("collective_receive"))
                if "csi" in changed and S.csi is not None:
                    S.csi.start() if new["csi"] else S.csi.stop()
                if S.memidx is not None and new["layers"]["recall"] != before["layers"]["recall"]:
                    S.memidx.set_recall(new["layers"]["recall"])
                cat = model_catalog()
                return self._send(200, {"settings": new, "models": [x["id"] for x in cat], "model_catalog": cat})
            if path == "/api/memory/forget":
                eid = str(body.get("id") or "")
                ok = layer_kit().forget(eid)
                if ok:
                    S.repo.note(f"growth: {eid} removed by the user")
                return self._send(200, {"ok": ok, "memories": layer_kit().memories()})
            if path in ("/api/session/forget", "/api/reset"):
                sid = str(body.get("session_id") or "")
                with S.sess_lock:
                    S.sessions.pop(sid, None)
                if body.get("delete") and sid:  # a deleted conversation takes its personal experiences with it
                    personal_step(personal_store(), "delete_session", sid)
                return self._send(200, {"ok": True, "deleted": bool(body.get("delete")) and S.repo.delete_session(sid)})
            if path == "/api/feedback":  # the Helpful button; other responses are read from the next message
                sid = str(body.get("session_id") or "")
                personal_step(laptop_runtime().personal, "helpful", sid, int(body.get("turn") or 0) or None)
                return self._send(200, {"growth": S.repo.feedback(sid, int(body.get("turn") or 0), "helpful")})
            if path in ("/api/personal/retention", "/api/personal/forget"):  # the user's choices: keep, stop, delete
                pers = personal_store()
                if pers is None:
                    return self._send(409, {"error": "personal learning is off for this process (CLOVER_PPL=off)"})
                if path.endswith("/forget"):
                    if body.get("confirm") is not True:
                        raise ValueError("confirm must be true to forget everything")
                    return self._send(200, dict(pers.forget_all(), view=pers.view()))
                ids = body.get("ids") if isinstance(body.get("ids"), list) else []
                return self._send(200, dict(pers.retention(ids, str(body.get("choice") or "")), view=pers.view()))
            if path in ("/api/collective/consent", "/api/collective/contribute", "/api/collective/sync",
                        "/api/collective/withdraw", "/api/collective/preview"):
                import core_sync
                coll = collective_store()
                if coll is None:
                    return self._send(409, {"error": "collective learning is off for this process (CLOVER_COLLECTIVE=off)"})
                pers = personal_store()
                mgr = google_identity_manager()
                g_stat = mgr.status() if mgr else {"connected": False, "status_text": "Not connected"}
                p_on = bool(S.settings.get().get("personal", True)) if S.settings else True
                try:
                    if path.endswith("/consent"):
                        c_in = body.get("contribute") if isinstance(body.get("contribute"), bool) else None
                        r_in = body.get("receive") if isinstance(body.get("receive"), bool) else None
                        coll.set_consent(
                            contribute=c_in,
                            receive=r_in,
                            google_connected=bool(g_stat.get("connected")),
                            require_identity=True,
                        )
                        upd = {}
                        if c_in is not None:
                            upd["collective_contribute"] = c_in
                        if r_in is not None:
                            upd["collective_receive"] = r_in
                        if upd and S.settings is not None:
                            S.settings.update(upd)
                        return self._send(200, dict(coll.view(pers, google_status=g_stat), personal_enabled=p_on))
                    if path.endswith("/contribute"):
                        id_tok = mgr.get_valid_id_token() if mgr else None
                        pkgs = body.get("packages")
                        if pkgs is None:
                            pkgs = coll.preview_candidates(pers)
                        resp = coll.contribute_to_gateway(
                            id_token=id_tok,
                            packages=pkgs,
                            confirmed=bool(body.get("confirmed")),
                            gateway_url=body.get("gateway_url"),
                        )
                        return self._send(200, dict(resp, view=dict(coll.view(pers, google_status=g_stat), personal_enabled=p_on)))
                    if path.endswith("/sync"):
                        if body.get("registry_dir"):
                            sync_res = coll.sync_from_registry(str(body["registry_dir"]).strip())
                        else:
                            id_tok = mgr.get_valid_id_token() if mgr else None
                            sync_res = coll.sync_from_gateway(
                                id_token=id_tok,
                                require_identity=True,
                                gateway_url=body.get("gateway_url"),
                            )
                        return self._send(200, dict(sync_res, view=dict(coll.view(pers, google_status=g_stat), personal_enabled=p_on)))
                    if path.endswith("/withdraw"):
                        id_tok = mgr.get_valid_id_token() if mgr else None
                        w_res = coll.withdraw_from_gateway(
                            id_token=id_tok,
                            gateway_url=body.get("gateway_url"),
                        )
                        if S.settings is not None:
                            S.settings.update({"collective_contribute": False})
                        return self._send(200, dict(w_res, view=dict(coll.view(pers, google_status=g_stat), personal_enabled=p_on)))
                    return self._send(200, {"candidates": coll.preview_candidates(pers)})
                except core_sync.CoreSyncError as exc:
                    return self._send(400, {"error": str(exc), "view": dict(coll.view(pers, google_status=g_stat), personal_enabled=p_on)})
            if path == "/api/media/clear":
                gone = 0
                for p in media_dir().iterdir():  # only files Clover named (by content): the folder may be the user's
                    if p.is_file() and re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{2,5}", p.name):
                        p.unlink()
                        gone += 1
                docs_dir = media_dir() / "documents"
                for d in docs_dir.iterdir() if docs_dir.is_dir() else ():  # saved documents: Clover's content-named folders
                    if d.is_dir() and re.fullmatch(r"[0-9a-f]{24}", d.name):
                        gone += sum(1 for f in d.iterdir() if f.is_file())
                        shutil.rmtree(d, ignore_errors=True)
                out_dir = media_dir() / "out"
                for d in out_dir.iterdir() if out_dir.is_dir() else ():  # what Python wrote, one folder per run
                    if d.is_dir() and re.fullmatch(r"[0-9a-f]{12}", d.name):
                        gone += sum(1 for f in d.iterdir() if f.is_file())
                        shutil.rmtree(d, ignore_errors=True)
                S.repo.note(f"media cleared by the user: {gone} files")
                left = [p for p in media_dir().iterdir() if p.is_file()]
                return self._send(200, {"files": len(left), "bytes": sum(p.stat().st_size for p in left), "deleted": gone,
                                        "path": str(media_dir())})
            if path == "/api/shutdown":
                if paths.ANDROID or paths.WEB:  # the phone app and web server do not stop from a browser request
                    return self._send(403, {"error": "the server cannot be stopped from a browser request"})
                within = float(body.get("unless_vscode_within_s") or 0)
                if within and time.time() - S.last_v1 < within:
                    return self._send(200, {"shutdown": False, "reason": "VS Code used Clover recently"})
                self._send(200, {"shutdown": True})
                threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
                return None
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._v1_chat(body)
        except EngineNotReady as exc:
            return self._send(503, {"error": str(exc), "status": S.load["status"], "load_step": S.load["step"]})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        self._send(404, {"error": "not found"})

    # ---------------- LFM3 mode: streamed turn ----------------
    def _lfm3_stream(self, body):
        prompt = str(body.get("prompt") or "").strip()
        resume_tid = str(body.get("resume_turn_id") or "").strip()
        if not prompt and not body.get("media") and not body.get("documents") and not resume_tid:
            return self._send(400, {"error": "prompt is required"})
        eng = engine_for(body.get("model"))
        if eng is None:
            raise EngineNotReady(S.load.get("error") or f"engine is loading ({S.load.get('step')})")
        sid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(body.get("session_id") or uuid.uuid4().hex))[:96]
        if resume_tid:
            try:
                after_seq = int(body.get("after_seq") or self.headers.get("Last-Event-ID") or 0)
            except ValueError:
                after_seq = 0
            with S.turns_lock:
                prev_turn = S.active.get(resume_tid) or S.recent_turns.get(resume_tid)
                replay_evs = [e for e in (prev_turn["events"] if prev_turn else []) if int(e.get("seq") or 0) > after_seq]
            if prev_turn is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for k, v in self._cors_headers().items():
                    self.send_header(k, v)
                self.end_headers()
                for ev in replay_evs:
                    self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return None

        turn_id = uuid.uuid4().hex
        turn = {
            "turn_id": turn_id,
            "session_id": sid,
            "cancel": threading.Event(),
            "pending": {},
            "seq": 0,
            "events": [],
            "done": False,
            "updated": time.time(),
        }
        with S.turns_lock:
            S.active[turn_id] = turn
        write_lock = threading.Lock()
        client_closed = [False]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

        def emit(ev):
            with write_lock:
                turn["seq"] += 1
                seq = turn["seq"]
                out_ev = dict(ev, seq=seq, turn_id=turn_id)
                turn["events"].append(out_ev)
                if len(turn["events"]) > 512:
                    del turn["events"][:len(turn["events"]) - 512]
                if ev.get("type") in ("done", "error"):
                    turn["done"] = True
                turn["updated"] = time.time()
                if client_closed[0]:
                    return
                try:
                    self.wfile.write(f"id: {seq}\ndata: {json.dumps(out_ev, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (OSError, BrokenPipeError, ConnectionResetError):  # the window was closed: stop generating
                    client_closed[0] = True
                    turn["cancel"].set()
                    for pend in list(turn["pending"].values()):
                        pend["event"].set()

        def ask(perm, detail, risk=None):
            import permissions as P
            rid = uuid.uuid4().hex[:12]
            pend = {"event": threading.Event(), "decision": "deny"}
            turn["pending"][rid] = pend
            emit(dict({"type": "permission", "turn_id": turn_id, "request_id": rid, "tool": perm,
                       "label": P.TOOLS[perm]["label"], "detail": str(detail or "")[:4000]},
                      **({"risky": risk} if risk else {})))
            at = S.repo.current if S.repo else None
            if at:
                S.repo.event(*at, {"type": "permission", "tool": perm})
            S.engine.stage = {"name": "waiting for your permission", "since": time.time()}
            pend["event"].wait(timeout=900)
            turn["pending"].pop(rid, None)
            if turn["cancel"].is_set():
                return False
            if pend["decision"] == "always" and not risk:
                S.perms.update({perm: "allow"})
            emit({"type": "permission_decided", "request_id": rid, "decision": pend["decision"]})
            if at:
                S.repo.event(*at, {"type": "permission_decided", "decision": f"{perm} {pend['decision']}"})
            return pend["decision"] in ("once", "always")

        emit({"type": "start", "turn_id": turn_id, "session_id": sid})
        done_ev = threading.Event()

        def heartbeat():
            while not done_ev.wait(timeout=15):
                if client_closed[0]:
                    break
                try:
                    with write_lock:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (OSError, BrokenPipeError, ConnectionResetError):
                    client_closed[0] = True
                    turn["cancel"].set()
                    for pend in list(turn["pending"].values()):
                        pend["event"].set()
                    break

        threading.Thread(target=heartbeat, daemon=True, name="sse-heartbeat").start()
        try:
            media = save_media(body.get("media"))
            if media:
                emit({"type": "media", "items": media})
            docs = save_documents(body.get("documents"))
            if docs:
                emit({"type": "documents", "items": [{k: d[k] for k in ("name", "kind", "bytes")} for d in docs]})
            res = lfm3_turn(sid, build_prompt(prompt, body.get("attached_files"), docs), body, on_event=emit, ask=ask,
                            cancel=turn["cancel"], media=media, ask_text=prompt, docs=docs)
            emit({"type": "done", **res})
        except Exception as exc:
            traceback.print_exc()
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            done_ev.set()
            with S.turns_lock:
                S.active.pop(turn_id, None)
                S.recent_turns[turn_id] = turn
                while len(S.recent_turns) > 32:
                    S.recent_turns.pop(next(iter(S.recent_turns)))
        return None

    # ---------- OpenAI-compatible ----------
    def _v1_chat(self, body):
        import permissions as P
        S.last_v1 = time.time()
        if S.engine is None:
            raise EngineNotReady(S.load.get("error") or f"engine is loading ({S.load.get('step')})")
        messages = body.get("messages") or []
        stream = bool(body.get("stream", False))
        tools = body.get("tools") or []
        user_prompt, prior_turns, tool_rounds, ws_info = parse_openai_messages(messages)
        if S.mode == "lfm3":
            # images from VS Code (the Omni reads them; LFM3-1.7B is text only and they are left out as before)
            media = []
            if "vision" in (getattr(S.engine, "inputs", None) or []):
                media = save_media([{"kind": "image", "data": u, "name": "image from VS Code"} for u in openai_images(messages)])
            if tools and body.get("tool_choice") != "none":
                t0 = time.perf_counter()
                with S.turn_lock:
                    S.busy = {"kind": "VS Code step", "since": time.time()}
                    try:
                        out = S.engine.vscode_step(user_prompt, ws_info, prior_turns, tool_rounds, tools,
                                                   **({"media": media} if media else {}))
                    finally:
                        S.busy = None
                print("V1_LFM3_STEP", json.dumps({"ms": round((time.perf_counter() - t0) * 1000), "call": out.get("tool_call"),
                                                  "media": len(media), "trace": out.get("trace")}, ensure_ascii=False)[:3000],
                      flush=True)
                if out.get("tool_call"):
                    return self._openai_reply(None, out["tool_call"], stream, user_prompt)
                return self._openai_reply(out.get("content") or "", None, stream, user_prompt)
            hist = []
            for h in prior_turns[-6:]:
                hist.append({"user": h["user"], "reply": h["reply"]})
            res = lfm3_turn("vsc-" + uuid.uuid4().hex, user_prompt, {"history": hist}, kind="VS Code answer", media=media)
            with S.sess_lock:
                S.sessions.pop(res["session_id"], None)
            text = res.get("reply") or ""
            skipped = sorted({t["tool"] for t in res.get("tool_calls") or [] if not t["ran"]})
            if skipped:
                text += ("\n\n_Note: Clover could not use " + ", ".join(skipped) +
                         " (permission). You can change it in Clover AI Studio → Permissions._")
            return self._openai_reply(text, None, stream, user_prompt)
        return self._v1_chat_clover(body, messages, stream, tools, user_prompt, prior_turns, tool_rounds, ws_info)

    def _v1_chat_clover(self, body, messages, stream, tools, user_prompt, prior_turns, tool_rounds, ws_info):
        import permissions as P
        import vscode_agent
        if tools and body.get("tool_choice") != "none":
            trace = {}
            t0 = time.perf_counter()
            with S.turn_lock:
                S.busy = {"kind": "vscode tool step", "since": time.time()}
                S.engine.stage = {"name": "choosing a VS Code tool", "since": time.time()}
                try:
                    call = vscode_agent.decide_tool_step(S.engine, user_prompt, ws_info, prior_turns[-6:], tool_rounds,
                                                         tools, trace)
                finally:
                    S.busy = None
                    S.engine.stage = {"name": "idle", "since": time.time()}
            trace["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            print("V1_TOOL_STEP", json.dumps({"call": call, **trace}, ensure_ascii=False)[:3000], flush=True)
            if call is not None:
                return self._openai_reply(None, call, stream, user_prompt)
        res = run_turn("vsc-" + uuid.uuid4().hex, with_tool_outputs(user_prompt, tool_rounds), remember=True,
                       history=prior_turns[-6:], interactive=False, kind="vscode answer")
        text = res.get("reply") or ""
        modes = S.perms.get()
        for tool in sorted({e["tool"] for e in res.get("permission_events") or [] if not e["allowed"]}):
            text += (f"\n\n_Note: Clover did not use “{P.TOOLS[tool]['label']}” (permission is set to "
                     f"{modes[tool]}). You can change it in Clover AI Studio → Permissions._")
        for m in res.get("media") or []:
            text += f"\n\n![{m['title'] or 'Clover visual'}]({m['data_url']})"
        return self._openai_reply(text, None, stream, user_prompt, int(res.get("n_k3_tokens") or 0))

    def _openai_reply(self, text, call, stream, user_prompt, n_tok=0):
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        p_tok = max(1, len(user_prompt.split()))
        usage = {"prompt_tokens": p_tok, "completion_tokens": n_tok or 24, "total_tokens": p_tok + (n_tok or 24)}
        if call is not None:
            tc = {"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                  "function": {"name": call["name"], "arguments": call["arguments"]}}
            message, finish = {"role": "assistant", "content": None, "tool_calls": [tc]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": text}, "stop"
        if not stream:
            return self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
        delta = ({"role": "assistant", "content": None, "tool_calls": [dict(message["tool_calls"][0], index=0)]}
                 if call is not None else {"role": "assistant", "content": text})
        chunks = [{"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}]
        payload = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return self._send(200, payload.encode("utf-8"), "text/event-stream")


def idle_watch(minutes):
    while True:
        time.sleep(30)
        if S.busy is None and time.time() - S.last_request > minutes * 60:
            print("IDLE_EXIT after", minutes, "min without requests", flush=True)
            os._exit(0)


def disable_power_throttling():
    if os.name != "nt":
        return
    import winperf
    print("POWER_THROTTLING_OFF" if winperf.disable_power_throttling() else "POWER_THROTTLING_UNCHANGED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE.parent))
    ap.add_argument("--planner", default=None, help="LFM3 dir (base model or trained LFM3 HybridAgent checkpoint)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--search-cache", default=None)
    ap.add_argument("--search-mode", choices=("record", "replay", "live"), default="live")
    ap.add_argument("--no-gate", action="store_true", help="benchmarks only: tools run without permission checks")
    ap.add_argument("--planner-backend", choices=("auto", "c", "torch"), default="torch",
                    help="torch: planner in torch (fp32 + int8 feed-forward, faster prefill, ~6 GB more RAM); "
                         "c: base LFM3 planner on the C engine (shares Layer 1 weights); auto: c when possible")
    ap.add_argument("--mode", choices=("lfm3", "clover"), default="lfm3",
                    help="lfm3: Layer 1 alone with native tools (default); clover: the full Clover layer stack")
    ap.add_argument("--idle-exit-min", type=float, default=0.0, help="exit after this many idle minutes (0 = never)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    base = Path(a.base)
    setup_stdio(Path(a.log) if a.log else base / "logs" / "studio-server.log")
    a.planner = a.planner or str(base / "data" / "lfm3" / "hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    mingit = base / "tools" / "git" / "cmd"
    if mingit.is_dir():  # Clover's records and personal learning are git repositories: setup's MinGit when Windows has none
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(mingit)
    try:  # Windows' own certificate check (it fetches missing roots): example.com failed with Python's bundle
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        print("TRUSTSTORE_MISSING: web pages use Python's certificates", flush=True)

    import clover_repo
    import permissions as P
    import web_search
    moved = clover_repo.migrate(base)
    if moved:
        print("MOVED", "; ".join(moved), flush=True)
    S.perms = P.PermissionStore(paths.config_file(base, "permissions.json"))
    import hardware
    S.hardware = hardware.probe(base, paths.config_file(base, "hardware.json"))
    print("HARDWARE", json.dumps({k: S.hardware[k] for k in ("chosen", "ram_gb", "below_minimum")}), flush=True)
    S.settings = Settings(paths.config_file(base, "settings.json"), engine=S.hardware["chosen"]["engine"])
    S.search = web_search.SearchConfig.shared(paths.config_file(base, "search.json"),
                                              paths.vault_file(base, "search-keys.json"))
    import workspaces
    S.github = workspaces.GitHubAuth(paths.vault_file(base, "github.json"))
    S.workspaces = workspaces.Workspaces(paths.config_file(base, "workspaces.json"), base / "workspaces", base, S.github)
    import vscode_bridge
    S.vscode = vscode_bridge.Bridge(paths.vault_file(base, "vscode-bridge.json"))
    S.args = a
    S.port = a.port
    S.mode = a.mode
    S.repo = clover_repo.Repo(base, HERE / "AGENTS.md", kit=layer_kit)
    if moved:
        S.repo.note("moved into the repository, vault and media folder: " + "; ".join(moved))
    import memory_index
    S.memidx = memory_index.MemoryIndex(base, S.repo, is_idle=studio_idle, note_writer=write_note,
                                        recall=S.settings.get()["layers"]["recall"])
    personal_step(personal_store(), "set_learning", S.settings.get().get("personal", True))  # Settings and store agree
    import csi_child
    S.csi = csi_child.CSIService(base, threads=S.hardware["chosen"]["csi_threads"])
    if S.settings.get().get("csi", True):
        S.csi.start()
    try:
        S.httpd = StudioHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError as exc:
        print(f"PORT_IN_USE 127.0.0.1:{a.port}: {exc}", flush=True)
        sys.exit(3)
    S.httpd.daemon_threads = True
    disable_power_throttling()
    S.token = os.environ.pop("CLOVER_TOKEN", "")
    if paths.WEB and not S.token:
        import k3_host_engine
        S.token = os.environ.get("CLOVER_HOST_KEY") or k3_host_engine.k3_host_key()
    if not paths.WEB:
        S.setup = Setup(base)
    if setup_needed():
        S.load.update(status="setup", step="the model is not downloaded yet")
        print("SETUP_NEEDED", flush=True)
    else:
        threading.Thread(target=load_engine, args=(a,), daemon=True, name="engine-load").start()
        if S.setup is not None:
            import omni_c2_c3
            if omni_c2_c3.missing_c2_c3_files(base) or csi_child.missing(base):
                S.setup.start(components="core,csi,c2c3")
    if a.idle_exit_min > 0:
        threading.Thread(target=idle_watch, args=(a.idle_exit_min,), daemon=True, name="idle-watch").start()
    print(f"SERVING http://127.0.0.1:{a.port}/ (pid {os.getpid()})", flush=True)
    S.httpd.serve_forever()


if __name__ == "__main__":
    main()
