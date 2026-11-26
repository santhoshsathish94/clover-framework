"""Run the real Studio against isolated copies of installed models, with output hash tracing."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-base", type=Path, default=Path.home() / "CloverAI")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--reuse", action="store_true", help="Restart an existing isolated base after verifying its asset hashes")
    args = parser.parse_args()
    if args.reuse and args.base is None:
        parser.error("--reuse requires --base")
    base = args.base or Path(tempfile.mkdtemp(prefix="clover-studio-integrity-"))
    base.mkdir(parents=True, exist_ok=True)
    print(f"ISOLATED_BASE={base}", flush=True)
    inventory = {}
    if args.reuse:
        inventory = json.loads((base / "asset-hashes.json").read_text(encoding="utf-8"))
        for relative, expected in inventory.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in ("data", "tools"):
                raise ValueError("Invalid asset manifest path")
            if digest(base / path) != expected:
                raise RuntimeError(f"Isolated asset changed: {relative}")
    else:
        for folder in ("data", "tools"):
            source = args.installed_base / folder
            target = base / folder
            if target.exists():
                raise RuntimeError(f"Refusing to overwrite an existing test directory: {target}")
            print(f"COPYING {folder}", flush=True)
            shutil.copytree(source, target)
            for original in source.rglob("*"):
                if original.is_file():
                    relative = original.relative_to(args.installed_base)
                    original_hash = digest(original)
                    if digest(base / relative) != original_hash:
                        raise RuntimeError(f"Copied asset does not match: {relative}")
                    inventory[relative.as_posix()] = original_hash
        (base / "asset-hashes.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"ASSETS_VERIFIED={len(inventory)}", flush=True)

    clover = Path(__file__).resolve().parent.parent / "clover"
    sys.path.insert(0, str(clover))
    import omni_native
    import omni_c2_c3
    import server

    trace_lock = threading.Lock()
    trace_path = base / "model-output-trace.jsonl"

    def record(stage, text, metadata):
        entry = {
            "time": time.time(),
            "stage": stage,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "characters": len(text),
            "metadata": metadata,
        }
        with trace_lock:
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=True) + "\n")

    original_chat = omni_native.OmniNative._chat
    original_decode = omni_c2_c3.OmniC2C3Runtime.c3_decode

    def traced_chat(engine, *call_args, **call_kwargs):
        result = original_chat(engine, *call_args, **call_kwargs)
        record("c1_model_output", result[0], result[2])
        return result

    def traced_decode(runtime, *call_args, **call_kwargs):
        result = original_decode(runtime, *call_args, **call_kwargs)
        record("c3_model_output", result[0], result[1])
        return result

    omni_native.OmniNative._chat = traced_chat
    omni_c2_c3.OmniC2C3Runtime.c3_decode = traced_decode
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.argv = [str(clover / "server.py"), "--base", str(base), "--port", str(args.port)]
    print(f"REAL_STUDIO_URL=http://127.0.0.1:{args.port}/", flush=True)
    server.main()


if __name__ == "__main__":
    main()