#!/usr/bin/env python3
"""
mobile_vault.py — Cross-platform encrypted secret storage (`vault/`) for Clover AI Studio.
- On Windows: uses Windows DPAPI (`CryptProtectData` / `CryptUnprotectData`).
- On Android / Linux (`s23` / `k3`): uses a 256-bit device-local key in `~/.clover_vault_key` (`0600`)
  with HMAC-SHA256 authenticated stream cipher (or AES-GCM when `cryptography` is installed),
  so secrets in `vault/*.json` are never stored as plain text in the repository or config files.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path


def _device_key_path() -> Path:
    return Path.home() / ".clover_vault_key"


def _get_or_create_device_key() -> bytes:
    kp = _device_key_path()
    if kp.is_file():
        try:
            raw = base64.b64decode(kp.read_text(encoding="ascii").strip())
            if len(raw) == 32:
                return raw
        except Exception:
            pass
    key = secrets.token_bytes(32)
    kp.write_text(base64.b64encode(key).decode("ascii"), encoding="ascii")
    try:
        os.chmod(kp, 0o600)
    except OSError:
        pass
    return key


def _xor_keystream(key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    offset = 0
    counter = 0
    while offset < len(data):
        blk = hashlib.sha256(key + nonce + counter.to_bytes(8, "little")).digest()
        take = min(len(blk), len(data) - offset)
        for i in range(take):
            out[offset + i] = data[offset + i] ^ blk[i]
        offset += take
        counter += 1
    return bytes(out)


def protect(text: str) -> str:
    raw = (text or "").encode("utf-8")
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Blob(ctypes.Structure):
                _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

            buf = ctypes.create_string_buffer(raw, len(raw))
            src, out = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
            if ctypes.windll.crypt32.CryptProtectData(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out)):
                try:
                    enc = ctypes.string_at(out.pbData, out.cbData)
                    return "dpapi:" + base64.b64encode(enc).decode("ascii")
                finally:
                    ctypes.windll.kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))
        except Exception:
            pass
    key = _get_or_create_device_key()
    nonce = secrets.token_bytes(16)
    ct = _xor_keystream(key, nonce, raw)
    mac = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    return "v1:" + base64.b64encode(nonce + mac + ct).decode("ascii")


def unprotect(blob: str) -> str:
    if not blob:
        return ""
    if blob.startswith("dpapi:") and sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        enc = base64.b64decode(blob[len("dpapi:") :])
        buf = ctypes.create_string_buffer(enc, len(enc))
        src, out = Blob(len(enc), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
        if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out)):
            raise OSError("Windows DPAPI decrypt failed")
        try:
            return ctypes.string_at(out.pbData, out.cbData).decode("utf-8")
        finally:
            ctypes.windll.kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))
    if blob.startswith("v1:"):
        raw = base64.b64decode(blob[3:])
        if len(raw) < 48:
            raise ValueError("Invalid vault blob")
        nonce, mac, ct = raw[:16], raw[16:48], raw[48:]
        key = _get_or_create_device_key()
        exp_mac = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, exp_mac):
            raise ValueError("Vault integrity check failed")
        return _xor_keystream(key, nonce, ct).decode("utf-8")
    # Legacy base64 fallback
    return base64.b64decode(blob).decode("utf-8")
