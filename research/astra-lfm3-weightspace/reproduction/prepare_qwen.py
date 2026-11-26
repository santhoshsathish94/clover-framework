import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(root):
    root = Path(root).resolve()
    models = root / "models"
    models.mkdir(exist_ok=True)
    sealed = models / "QWEN_SOURCE.json"
    if sealed.exists():
        raise FileExistsError("The Qwen source is already sealed; do not overwrite it")
    api_path = models / "qwen-upstream.json"
    if api_path.exists():
        info = json.loads(api_path.read_text())
    else:
        request = urllib.request.Request(
            f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true",
            headers={"User-Agent": "Clover-weightspace-paper-reproduction"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            info = json.load(response)
        if info["id"] != MODEL_ID or info["sha"] != REVISION or info.get("private"):
            raise ValueError("Unexpected model identity, revision or access boundary")
        with api_path.open("x") as output:
            json.dump(info, output, indent=2)
    selected = [entry for entry in info["siblings"] if entry["rfilename"].endswith((".safetensors", ".json", ".jinja", ".txt", ".py", ".md"))
                or entry["rfilename"] == "LICENSE"]
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    destination = models / "qwen-original"
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=destination,
                      allow_patterns=[entry["rfilename"] for entry in selected], token=False, max_workers=2)
    files = []
    for entry in selected:
        path = destination / entry["rfilename"]
        checksum = digest(path)
        if entry.get("lfs") and checksum != entry["lfs"]["sha256"]:
            raise ValueError(f"Upstream checkpoint digest mismatch: {entry['rfilename']}")
        if not entry.get("lfs") and entry.get("blobId"):
            git_hash = hashlib.sha1(f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest()
            if git_hash != entry["blobId"]:
                raise ValueError(f"Upstream Git blob mismatch: {entry['rfilename']}")
        files.append({"file": entry["rfilename"], "sha256": checksum, "bytes": path.stat().st_size})
    with sealed.open("x") as output:
        json.dump({"model_id": MODEL_ID, "revision": REVISION, "files": files}, output, indent=2)
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    print("QWEN_SOURCE_SEALED " + json.dumps({"revision": REVISION, "files": files}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    prepare(args.root)