import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import urllib.request


MODEL_ID = "LiquidAI/LFM3-Omni-2B"


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def file_hash(path, algorithm="sha256", git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "SEALED.json").exists():
        raise RuntimeError("Snapshot is sealed; downloading into it is prohibited")
    for name in ("original", "reports", "runs", "cache", "copies", "fixtures"):
        (root / name).mkdir(exist_ok=True)
    api_path = root / "upstream.json"
    if api_path.exists():
        upstream = json.loads(api_path.read_text())
    else:
        request = urllib.request.Request(
            f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true",
            headers={"User-Agent": "Astra-read-only-weightspace-study"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            upstream = json.load(response)
        if upstream["id"] != MODEL_ID or upstream.get("private"):
            raise RuntimeError("Unexpected model identity or access boundary")
        write_new_json(api_path, upstream)
    summary = {
        "model_id": upstream["id"], "revision": upstream["sha"],
        "safetensors": upstream.get("safetensors"),
        "files": [{key: entry.get(key) for key in ("rfilename", "size", "lfs")}
                  for entry in upstream["siblings"]],
    }
    print(json.dumps(summary, indent=2), flush=True)
    summary_path = root / "upstream_summary.json"
    if not summary_path.exists():
        write_new_json(summary_path, summary)
    environment_path = root / "environment.json"
    if not environment_path.exists():
        packages = {}
        for name in ("torch", "transformers", "safetensors", "numpy",
                     "huggingface_hub", "pillow", "scipy", "soundfile", "librosa"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        write_new_json(environment_path, {
            "platform": platform.platform(), "python": platform.python_version(),
            "packages": packages, "loadavg": os.getloadavg(),
            "meminfo": Path("/proc/meminfo").read_text(),
            "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
            "disk_free_bytes": shutil.disk_usage(root).free,
        })
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    filenames = [entry["rfilename"] for entry in upstream["siblings"]]
    if args.metadata_only:
        filenames = [name for name in filenames if Path(name).suffix in {
            ".json", ".py", ".md", ".jinja", ".txt",
        }]
    for name in filenames:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe upstream filename")
    snapshot_download(
        MODEL_ID, revision=upstream["sha"], local_dir=root / "original",
        allow_patterns=filenames, token=False, max_workers=2,
    )
    manifest = []
    for entry in upstream["siblings"]:
        name = entry["rfilename"]
        if name not in filenames:
            continue
        path = root / "original" / name
        digest = file_hash(path)
        lfs = entry.get("lfs") or {}
        if lfs and digest != lfs["sha256"]:
            raise RuntimeError(f"Upstream LFS digest mismatch: {name}")
        if not lfs and entry.get("blobId"):
            if file_hash(path, "sha1", git_blob=True) != entry["blobId"]:
                raise RuntimeError(f"Upstream Git blob mismatch: {name}")
        manifest.append({"file": name, "sha256": digest, "bytes": path.stat().st_size})
    if args.metadata_only:
        print("METADATA_VERIFIED; weights not fetched or executed", flush=True)
        return
    write_new_json(root / "SEALED.json", {
        "model_id": MODEL_ID, "revision": upstream["sha"], "files": manifest,
    })
    for path in (root / "original").rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    (root / "original").chmod(0o555)
    print("SNAPSHOT_SEALED; all upstream file digests verified", flush=True)


if __name__ == "__main__":
    main()