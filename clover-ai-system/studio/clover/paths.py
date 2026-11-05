"""Where Clover AI Studio keeps things under CloverAI (the --base folder).

clover-ai-studio/  the local git repository: AGENTS.md, CONTEXT.md, sessions/, execution/, outcomes/, growth/, config/
vault/             secrets (API keys, tokens); never in the repository
media/             attachments; the user clears or backs them up; never in the repository
"""
import os
import re
import socket
import sys
from pathlib import Path

REPO_NAME = "clover-ai-studio"
# The same app runs on the laptop (Windows), in the Android app (CPython embedded by Chaquopy; its python has
# sys.getandroidapilevel), and on the web server (k3, CLOVER_PLATFORM=web).
ANDROID = hasattr(sys, "getandroidapilevel")
WEB = os.environ.get("CLOVER_PLATFORM") == "web"
PLATFORM = "android" if ANDROID else ("web" if WEB else "windows")
DEVICE_WORD = "phone" if ANDROID else ("server" if WEB else "laptop")


def device():
    name = os.environ.get("CLOVER_DEVICE") if (ANDROID or WEB) else None
    return re.sub(r"[^a-z0-9-]+", "-", (name or socket.gethostname()).lower()).strip("-") or "device"


def repo_dir(base):
    return Path(base) / REPO_NAME


def knowledge_dir(base):
    return repo_dir(base) / "growth" / "knowledge"


def config_dir(base):
    return repo_dir(base) / "config" / device()


def config_file(base, name):
    return config_dir(base) / name


def vault_dir(base):
    return Path(base) / "vault"


def vault_file(base, name):
    return vault_dir(base) / name


def media_dir(base):
    return Path(base) / "media"


def _inside(p, root):
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def write_blocked(base, p):
    """The model's file tools never write Clover's records, settings or AGENTS.md, nor the vault: they change only
    through Clover's own code. On the public web server, file tools stay inside media/ and workspaces/."""
    if WEB and not (_inside(p, media_dir(base)) or _inside(p, Path(base) / "workspaces")):
        return True
    return _inside(p, repo_dir(base)) or _inside(p, vault_dir(base))


def read_blocked(base, p):
    if WEB and not (_inside(p, media_dir(base)) or _inside(p, Path(base) / "workspaces")):
        return True
    return _inside(p, vault_dir(base))
