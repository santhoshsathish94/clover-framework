"""Secrets for this Windows account, kept in CloverAI\\vault (never in the repository). Values the user types (API keys,
a GitHub token) are encrypted with Windows data protection: only this Windows user on this laptop can read them. In the
Android app the phone's keystore holds the key instead (ai.clover.studio.VaultCrypto: AES-GCM, the key never leaves
the keystore)."""
import base64
import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path

ANDROID = hasattr(sys, "getandroidapilevel")


def _android():
    from java import jclass  # Chaquopy's bridge to the app's Java classes
    return jclass("ai.clover.studio.VaultCrypto")


def _posix_key():
    kp = Path(os.environ.get("CLOVER_VAULT_KEY") or (Path.cwd() / "vault" / ".key"))
    kp.parent.mkdir(parents=True, exist_ok=True)
    if not kp.is_file():
        kp.write_bytes(secrets.token_bytes(32))
        try:
            os.chmod(kp, 0o600)
        except OSError:
            pass
    return kp.read_bytes()


def _posix_crypt(data, protect_mode):
    key = _posix_key()
    if protect_mode:
        nonce = secrets.token_bytes(16)
        ks = hashlib.pbkdf2_hmac("sha256", key, nonce, 10000, dklen=max(1, len(data)))[:len(data)]
        ct = bytes(a ^ b for a, b in zip(data, ks))
        tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        return nonce + tag + ct
    if len(data) < 48:
        raise ValueError("vault blob truncated")
    nonce, tag, ct = data[:16], data[16:48], data[48:]
    if not hmac.compare_digest(tag, hmac.new(key, nonce + ct, hashlib.sha256).digest()):
        raise ValueError("vault authentication failed")
    ks = hashlib.pbkdf2_hmac("sha256", key, nonce, 10000, dklen=max(1, len(ct)))[:len(ct)]
    return bytes(a ^ b for a, b in zip(ct, ks))


def dpapi(data, protect):
    if os.name != "nt":
        return _posix_crypt(data, protect)
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    buf = ctypes.create_string_buffer(data, len(data))
    src, out = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(src), None, None, None, None, 0x01, ctypes.byref(out)):  # 0x01: never show UI
        raise OSError("Windows data protection failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def protect(text):
    if ANDROID:
        return str(_android().protect(text))
    return base64.b64encode(dpapi(text.encode("utf-8"), True)).decode("ascii")


def unprotect(blob):
    if ANDROID:
        return str(_android().unprotect(blob))
    return dpapi(base64.b64decode(blob), False).decode("utf-8")


def _rm_file(base):
    import paths
    return paths.vault_file(Path(base), "remote-models.json")


def get(base, key_id):
    import json
    try:
        d = json.loads(_rm_file(base).read_text(encoding="utf-8"))
        blob = d.get(str(key_id))
        return unprotect(blob) if blob else ""
    except Exception:
        return ""


def put(base, key_id, secret):
    import json
    f = _rm_file(base)
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except Exception:
        d = {}
    if secret:
        d[str(key_id)] = protect(str(secret))
    else:
        d.pop(str(key_id), None)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=2), encoding="utf-8")


def delete(base, key_id):
    put(base, key_id, "")
