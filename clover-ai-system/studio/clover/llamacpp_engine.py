"""LFM3 on the laptop GPU (Intel Iris Xe, Vulkan) through llama.cpp's llama-server and LiquidAI's QAD Q4_0 GGUF.

The Studio keeps its own chat template, tools and permission loop: prompts go in as token ids from the same HF
tokenizer the C engine uses, and generated token ids come back (`return_tokens`), so parsing is identical on both
engines. llama-server listens on 127.0.0.1 only, without its web UI, inside a job object that ends it with this process.
Measured on the laptop (AC): prompt reading 436 tok/s, writing 26.7 tok/s (C engine bf16: 53 / 8.1).
"""
import ctypes
import http.client
import json
import os
import re
import secrets
import socket
import subprocess
import time
from pathlib import Path

import paths

try:
    from ctypes import wintypes
except (ImportError, ValueError):  # not on Windows
    wintypes = None

GGUF = "LFM3-1.7B-QAD-Q4_0.gguf"
TOOL_RESPONSE_ID = 125025  # the model starting to invent a tool result: stop there, as the C engine path does


def server_exe(base_dir, kind):
    """llama-server: the laptop's build per backend under tools/llama.cpp (vulkan, cpu), on Linux llama-server in the
    same folder, or on Android the app's one build (llama.cpp's Snapdragon release: CPU, OpenCL for Adreno, Hexagon),
    packaged as a library: Android runs programs only from the app's library folder."""
    if paths.ANDROID:
        return Path(os.environ.get("CLOVER_NATIVE_LIB_DIR", "/nonexistent")) / "libllama_server.so"
    d = Path(base_dir) / "tools" / "llama.cpp" / kind
    if os.name != "nt" and (d / "llama-server").is_file():
        return d / "llama-server"
    return d / "llama-server.exe"


def server_env(env, exe_dir=None):
    """On Android its libraries sit beside it, OpenCL is the phone maker's (vendor) library and Hexagon's DSP code is
    found through ADSP_LIBRARY_PATH. On Linux its .so files sit beside llama-server."""
    if paths.ANDROID:
        lib = os.environ.get("CLOVER_NATIVE_LIB_DIR", "")
        env = dict(env, LD_LIBRARY_PATH=":".join([lib, "/vendor/lib64", "/system/vendor/lib64"]), ADSP_LIBRARY_PATH=lib)
    elif os.name != "nt" and exe_dir:
        cur = env.get("LD_LIBRARY_PATH", "")
        env = dict(env, LD_LIBRARY_PATH=f"{exe_dir}:{cur}" if cur else str(exe_dir))
    return env


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_with_parent(proc):
    """Job object with KILL_ON_JOB_CLOSE: the server dies with this process even if it crashes."""
    if os.name != "nt":
        return None

    class Basic(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("e", wintypes.DWORD), ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", ctypes.c_ulonglong * 6), ("m", ctypes.c_size_t * 4)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.LimitFlags = 0x2000
    if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.AssignProcessToJobObject(job, int(proc._handle))
    return job


def _no_power_throttling(proc):
    if os.name != "nt":
        return

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    st = PPTS(1, 0x1, 0x0)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation(int(proc._handle), 4, ctypes.byref(st), ctypes.sizeof(st))


class LlamaServer:
    # The conv-state checkpoint lands about ubatch + 4 tokens before the prompt's end. Measured reuse on a 1,416-token
    # chat: ubatch 512 -> 884 tokens, 256 -> 1,140, 128 -> 0.
    def __init__(self, base_dir, kind="vulkan", ctx=8192, timeout_s=180, ubatch=256, model=None, mmproj=None,
                 extra_args=(), name=None, threads=None):
        base_dir = Path(base_dir)
        self.exe = server_exe(base_dir, kind)
        self.model = Path(model) if model else base_dir / "data" / "lfm3" / "gguf" / GGUF
        for p in (self.exe, self.model) + ((Path(mmproj),) if mmproj else ()):
            if not p.is_file():
                raise FileNotFoundError(f"{p} is missing (run Clover AI Studio setup again)")
        self.kind, self.ctx, self.port = kind, int(ctx), _free_port()
        self.slots_dir = base_dir / "cache" / "kv-slots"
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = base_dir / "logs" / f"llama-server-{name or kind}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # LFM3's conv layers can only resume from a saved state; the default spacing (8,192 tokens) left tool round 2
        # reusing 93 of 681 prompt tokens
        args = [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.ctx),
                "-np", "1", "--no-webui", "-ngl", "99" if kind == "vulkan" else "0", "--slot-save-path", str(self.slots_dir),
                "--checkpoint-min-step", "0", "-ub", str(int(ubatch))]
        if mmproj:
            args += ["--mmproj", str(mmproj)]
        args += [str(a) for a in extra_args]
        # llama-server allows any web origin; a per-launch key keeps web pages in the user's browser from using it
        self._key = secrets.token_urlsafe(24)
        env = server_env(dict(os.environ, LLAMA_API_KEY=self._key), self.exe.parent)
        self.device, dev_id = self._list_device(env)
        if paths.ANDROID:  # one build with every backend: name the GPU, or none (no offload) for the CPU
            if kind == "vulkan" and not dev_id:
                raise RuntimeError("llama.cpp found no GPU it can use on this phone (OpenCL)")
            args += ["--device", dev_id if kind == "vulkan" else "none"]
            if threads:  # the big cores; llama.cpp's default counts the small ones too
                args += ["-t", str(int(threads))]
        elif os.name != "nt" and threads:
            args += ["-t", str(int(threads))]
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                     cwd=str(self.exe.parent), creationflags=0x08000000 if os.name == "nt" else 0)
        self._job = _kill_with_parent(self.proc)
        _no_power_throttling(self.proc)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if self._request("GET", "/health", timeout=2)[0] == 200:
                    break
            except OSError:
                pass
            if time.time() - t0 > timeout_s:
                self.close()
                raise TimeoutError(f"llama-server did not become ready in {timeout_s} s; see {self.log_path}")
            time.sleep(0.4)
        # the first prompt otherwise pays the GPU kernel set-up (4.6 s for 12 tokens in the first test)
        self._request("POST", "/completion", {"prompt": "Hello. " * 160, "n_predict": 2, "cache_prompt": False}, timeout=180)

    def _list_device(self, env):
        """-> (description, llama.cpp device name) of the GPU this server uses; ("CPU", None) for the CPU."""
        if self.kind == "cpu":
            return "CPU", None
        try:
            out = subprocess.run([str(self.exe), "--list-devices"], capture_output=True, text=True, timeout=30, env=env,
                                 creationflags=0x08000000 if os.name == "nt" else 0)
        except (OSError, subprocess.SubprocessError):
            return "GPU (Vulkan)", None
        m = re.search(r"^\s*(Vulkan\d+|GPUOpenCL\w*|OpenCL\w*):\s*(.+)$", out.stdout + out.stderr, re.M)
        return (m.group(2).split(" (")[0].strip(), m.group(1)) if m else ("GPU (Vulkan)", None)

    def _request(self, method, path, body=None, timeout=30):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            headers = {"Authorization": "Bearer " + self._key}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def stream(self, ids, params, cancel=None):
        """Yields (token_ids, final_or_None) while llama-server writes; closing the connection stops it."""
        body = dict(params, prompt=list(map(int, ids)), stream=True, return_tokens=True, cache_prompt=True)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        try:
            conn.request("POST", "/completion", body=json.dumps(body),
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
            r = conn.getresponse()
            if r.status != 200:
                raise RuntimeError(f"llama-server HTTP {r.status}: {r.read()[:400]!r}")
            while True:
                if cancel is not None and cancel.is_set():
                    yield [], {"stop_type": "cancelled"}
                    return
                line = r.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                ev = json.loads(line[6:])
                yield ev.get("tokens") or [], (ev if ev.get("stop") else None)
                if ev.get("stop"):
                    return
        finally:
            conn.close()

    def save_slot(self, name):
        return self._request("POST", "/slots/0?action=save", {"filename": name}, timeout=120)[0] == 200

    def restore_slot(self, name):
        return self._request("POST", "/slots/0?action=restore", {"filename": name}, timeout=120)[0] == 200

    def props(self):
        status, data = self._request("GET", "/props", timeout=30)
        return json.loads(data) if status == 200 else {}

    def chat_stream(self, body, cancel=None):
        """Yields the OpenAI chunks of /v1/chat/completions; {"cancelled": True} last when `cancel` is set."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True)),
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
            r = conn.getresponse()
            if r.status != 200:
                raise RuntimeError(f"llama-server HTTP {r.status}: {r.read()[:600]!r}")
            while True:
                if cancel is not None and cancel.is_set():
                    yield {"cancelled": True}
                    return
                line = r.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    return
                yield json.loads(line[6:])
        finally:
            conn.close()

    def tts(self, text, voice="female", lang="en"):
        """Speech for `text` as WAV bytes (llama-server /tts; LFM3-Omni's projector carries the speech decoder)."""
        status, data = self._request("POST", "/tts", {"input": text, "voice": voice, "lang": lang, "response_format": "wav"},
                                     timeout=600)
        if status != 200:
            raise RuntimeError(f"llama-server /tts HTTP {status}: {data[:300]!r}")
        return data

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


def sampling_params(cfg, max_new, seed, ctx=8192):
    """The model card's samplers on llama.cpp; everything llama.cpp adds by default (min_p, DRY, ...) switched off.
    The repetition penalty covers the whole context, as in transformers."""
    return {"n_predict": int(max_new), "temperature": float(cfg["temperature"]), "top_k": int(cfg["top_k"] or 0),
            "top_p": float(cfg["top_p"]), "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": float(cfg["repetition_penalty"]),
            "repeat_last_n": int(ctx), "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
            "xtc_probability": 0.0, "top_n_sigma": -1.0, "mirostat": 0, "seed": int(seed), "n_probs": 0}
