#!/usr/bin/env python3
"""
S3 · Tools — Local Terminal (`sh` / `PowerShell`) & SSH Host Runner (`local_terminal.py`).

Ported from `laptop/clover/local_terminal.py` (`Step 20.1 & 20.2`) and extended for Android Termux / Linux:
  1. **Process-Tree Termination on Timeout or Stop**:
     - On Windows: `taskkill /T /F /PID`.
     - On Android Termux / Linux: spawns in a new process group (`start_new_session=True`) and kills the
       entire process group (`os.killpg(p.pid, signal.SIGKILL)`) when `timeout_s` elapses or `cancel.is_set()`.
  2. **1 MB Output Cap & UTF-8 Decoding**:
     - Reads stdout+stderr in 8 KB chunks up to `OUT_CAP_BYTES = 1_000_000` so a runaway command can never
       exhaust device RAM.
  3. **Working Folder (`resolve_cwd`) & Timeout Clamping (`1..600s`)**:
     - Expands `~` and `$VARS`, resolves relative paths against the active workspace (`workdir`), and formats
       the final status (`[exit code N]`, `[stopped after T seconds: the time limit]`, or `[stopped: the user stopped the turn]`).
  4. **SSH Host Management (`SSHHosts`)**:
     - Generates an Ed25519 key pair on the device (`ssh-keygen -t ed25519`), exposes ONLY the public key
       (`.pub`) for the user to add on their server, verifies host fingerprints (`ssh-keyscan` + `ssh-keygen -lf`),
       and NEVER reads, logs, or sends private keys.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

MAX_TIMEOUT_S = 600
DEFAULT_TIMEOUT_S = 60
OUT_CAP_BYTES = 1_000_000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PS_HEAD = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8; "
    "$ProgressPreference = 'SilentlyContinue'\n"
)
PS_TAIL = "\nif (-not $?) { exit $(if ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }) }"


def clamp_timeout(t: Any) -> int:
    try:
        val = int(t)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return max(1, min(MAX_TIMEOUT_S, val))


def run(
    args: list[str],
    cwd: str | Path | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    cancel: Any = None,
    env: dict[str, str] | None = None,
) -> tuple[str, int | None, str]:
    """Runs `args` and returns `(output_text, exit_code_or_None, how_ended: 'exit' | 'timeout' | 'stopped')`."""
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True

    p = subprocess.Popen(args, **popen_kwargs)
    chunks: list[bytes] = []
    size = [0]

    def reader() -> None:
        assert p.stdout is not None
        for b in iter(lambda: p.stdout.read(8192), b""):
            if size[0] < OUT_CAP_BYTES:
                chunks.append(b)
            size[0] += len(b)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t0, how = time.time(), "exit"
    while p.poll() is None:
        if cancel is not None and cancel.is_set():
            how = "stopped"
            break
        if time.time() - t0 > timeout_s:
            how = "timeout"
            break
        time.sleep(0.1)

    if how != "exit":
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=NO_WINDOW)
        else:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()

    t.join(3)
    out = b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    if size[0] > OUT_CAP_BYTES:
        out += f"\n[output cut: {size[0]:,} bytes printed, the first {OUT_CAP_BYTES:,} kept]"
    return out, (p.returncode if how == "exit" else None), how


def result_text(out: str, code: int | None, how: str, timeout_s: int) -> str:
    out = (out or "").strip()
    end = {
        "exit": f"[exit code {code}]",
        "timeout": f"[stopped after {timeout_s} seconds: the time limit]",
        "stopped": "[stopped: the user stopped the turn]",
    }[how]
    return (out + "\n" if out else "(no output)\n") + end


def resolve_cwd(raw: str | Path | None, workdir: str | Path | None = None) -> tuple[Path | None, str | None]:
    s = str(raw or "").strip()
    base = Path(workdir) if workdir else Path.home()
    p = Path(os.path.expandvars(os.path.expanduser(s))) if s else base
    if not p.is_absolute():
        p = base / p
    if not p.is_dir():
        return None, f"Error: the folder {p} does not exist."
    return p.resolve(), None


def powershell(command: str, cwd: str | Path | None = None, timeout_s: int = DEFAULT_TIMEOUT_S, cancel: Any = None) -> str:
    t = clamp_timeout(timeout_s)
    out, code, how = run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", PS_HEAD + command + PS_TAIL],
        cwd=cwd,
        timeout_s=t,
        cancel=cancel,
    )
    return result_text(out, code, how, t)


def shell(command: str, cwd: str | Path | None = None, timeout_s: int = DEFAULT_TIMEOUT_S, cancel: Any = None) -> str:
    """Runs a shell command (`sh -c` on Android/Linux, `powershell.exe` on Windows) with full process-tree cleanup."""
    if os.name == "nt":
        return powershell(command, cwd=cwd, timeout_s=timeout_s, cancel=cancel)
    t = clamp_timeout(timeout_s)
    sh_bin = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
    out, code, how = run([sh_bin, "-c", str(command or "")], cwd=cwd, timeout_s=t, cancel=cancel)
    return result_text(out, code, how, t)


class SSHHosts:
    """
    Manages user-configured SSH hosts (`config/<device>/ssh-hosts.json`) and a dedicated Ed25519 key pair.
    Clover never reads, displays, or transmits the private key — only the `.pub` public key and host fingerprint.
    """

    def __init__(self, config_path: Path, ssh_dir: Path | None = None):
        self.path = Path(config_path)
        self.ssh_dir = Path(ssh_dir) if ssh_dir else (Path.home() / ".ssh")
        self.key_path = self.ssh_dir / "id_ed25519_clover"
        self.pub_path = self.ssh_dir / "id_ed25519_clover.pub"
        self.known_hosts = self.ssh_dir / "clover_known_hosts"
        self.lock = threading.RLock()
        self.hosts: list[dict[str, Any]] = []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.hosts = [h for h in (raw.get("hosts") or []) if isinstance(h, dict) and h.get("name") and h.get("host")]
        except (OSError, ValueError):
            self.hosts = []

    def ensure_keypair(self) -> str:
        """Generates `id_ed25519_clover` if missing and returns the `.pub` public key line (never the private key)."""
        self.ssh_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.ssh_dir, 0o700)
        except OSError:
            pass
        if not self.pub_path.is_file():
            keygen = shutil.which("ssh-keygen")
            if keygen:
                subprocess.run(
                    [keygen, "-t", "ed25519", "-N", "", "-C", "clover-ai-studio", "-f", str(self.key_path)],
                    capture_output=True,
                    timeout=20,
                )
        return self.pub_path.read_text(encoding="utf-8").strip() if self.pub_path.is_file() else ""

    def describe(self) -> dict[str, Any]:
        with self.lock:
            return {
                "public_key": self.ensure_keypair(),
                "hosts": list(self.hosts),
            }

    def add_host(self, name: str, host: str, user: str = "root", port: int = 22) -> dict[str, Any]:
        clean_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name or host).strip()).strip("-")[:40] or "host"
        clean_host = str(host or "").strip()
        clean_user = str(user or "root").strip()
        clean_port = max(1, min(65535, int(port or 22)))
        if not clean_host:
            raise ValueError("Host address is required")
        with self.lock:
            self.hosts = [h for h in self.hosts if h["name"] != clean_name]
            self.hosts.append({"name": clean_name, "host": clean_host, "user": clean_user, "port": clean_port})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"hosts": self.hosts}, indent=2), encoding="utf-8")
        return self.describe()

    def run_ssh(self, target: str, command: str, port: int | str = 22, timeout_s: int = DEFAULT_TIMEOUT_S, cancel: Any = None) -> str:
        t = clamp_timeout(timeout_s)
        tgt = str(target or "").strip()
        with self.lock:
            matched = next((h for h in self.hosts if h["name"] == tgt or h["host"] == tgt), None)
        if matched:
            user_host = f"{matched['user']}@{matched['host']}"
            p_str = str(matched.get("port") or port or 22)
        else:
            user_host = tgt
            p_str = str(port or 22)
        ssh_args = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", p_str]
        if self.key_path.is_file():
            ssh_args.extend(["-i", str(self.key_path)])
        ssh_args.extend([user_host, str(command or "")])
        out, code, how = run(ssh_args, timeout_s=t, cancel=cancel)
        return result_text(out, code, how, t)
