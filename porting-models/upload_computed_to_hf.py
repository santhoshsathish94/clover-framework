#!/usr/bin/env python3
"""
Uploads computed Clover datasets, 8x RTX 3090 trained JEPA trunks/datasets/adapters,
and ported model families (`liquid-lfm3/`, `clover-35b/`, `kimi/`, `nemotron/`, `csi/`)
to Hugging Face dataset `Santhoshsathish94/clover-ai-studio`.

Reads the Hugging Face token from `HF_TOKEN` or `--token-stdin` so no token ever
appears in source files, command-line process tables, or git history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile

REPO_ID = os.environ.get("CLOVER_HF_DATASET_REPO") or "Santhoshsathish94/clover-ai-studio"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    h = hashlib.sha1(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def collect_local_files(
    precomp_dir: Path,
    csi_dir: Path | None = None,
    clover35b_dir: Path | None = None,
    readme_path: Path | None = None,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if precomp_dir.is_dir():
        for sub in ("jepa", "hybridagent", "bridges", "ncm", "trunks"):
            base = precomp_dir / sub
            if not base.exists():
                continue
            for root_d, _, fnames in os.walk(base, followlinks=True):
                for fn in sorted(fnames):
                    if fn.endswith(".pt") or fn.startswith(".clover-"):
                        continue
                    fp = Path(root_d) / fn
                    if fp.is_file():
                        rel = fp.relative_to(precomp_dir).as_posix()
                        mapping[f"liquid-lfm3/{rel}"] = fp
    if csi_dir and csi_dir.is_dir():
        for fp in sorted(csi_dir.iterdir()):
            if fp.is_file():
                mapping[f"csi/morph-verb-v1.1/{fp.name}"] = fp
    if clover35b_dir and clover35b_dir.is_dir():
        for root_d, _, fnames in os.walk(clover35b_dir, followlinks=True):
            for fn in sorted(fnames):
                if fn.startswith(".clover-") or fn.endswith(".pyc") or "__pycache__" in root_d:
                    continue
                fp = Path(root_d) / fn
                if fp.is_file():
                    rel = fp.relative_to(clover35b_dir).as_posix()
                    mapping[f"clover-35b/{rel}"] = fp
    if readme_path and readme_path.is_file():
        mapping["README.md"] = readme_path
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--precomp-dir", default=os.environ.get("CLOVER_PRECOMP_DIR", "/opt/clover-k3/web-studio/data/omni/precomputed"))
    ap.add_argument("--csi-dir", default=os.environ.get("CLOVER_CSI_DIR", "/opt/clover-k3/web-studio/data/csi/morph-verb-v1.1"))
    ap.add_argument("--clover35b-dir", default=os.environ.get("CLOVER_35B_STAGE_DIR", "/opt/clover-k3/clover-35b-computed/hf-stage"))
    ap.add_argument("--readme", default="")
    ap.add_argument("--only-prefix", default="", help="Optional prefix filter (e.g., liquid-lfm3/jepa/ or clover-35b/)")
    ap.add_argument("--commit-message", default="feat(jepa,clover-35b): upload computed JEPA trunks, datasets, and Clover-35B-A3B precomputed artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-stdin", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    if args.token_stdin:
        token = sys.stdin.read().strip() or token

    api = HfApi(token=token)
    who = None
    if token:
        try:
            who = api.whoami().get("name")
        except Exception as exc:
            print(f"WHOAMI_WARN: {exc}", file=sys.stderr)

    remote_tree: dict[str, tuple[str, str | None, int]] = {}
    for e in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True):
        if isinstance(e, RepoFile):
            remote_tree[e.path] = (
                str(e.blob_id or ""),
                str(e.lfs.sha256) if e.lfs else None,
                int(e.size or 0),
            )

    local_map = collect_local_files(
        Path(args.precomp_dir),
        Path(args.csi_dir) if args.csi_dir else None,
        Path(args.clover35b_dir) if args.clover35b_dir else None,
        Path(args.readme) if args.readme else None,
    )
    if args.only_prefix:
        local_map = {k: v for k, v in local_map.items() if k.startswith(args.only_prefix)}

    to_upload: list[tuple[str, Path, int, str]] = []
    for repo_path, local_fp in sorted(local_map.items()):
        sz = local_fp.stat().st_size
        sha256 = sha256_file(local_fp)
        rem = remote_tree.get(repo_path)
        if rem is not None:
            rem_blob, rem_lfs_sha256, rem_sz = rem
            if rem_sz == sz:
                if rem_lfs_sha256 and rem_lfs_sha256 == sha256:
                    continue
                if not rem_lfs_sha256 and rem_blob == git_blob_sha1(local_fp):
                    continue
        to_upload.append((repo_path, local_fp, sz, sha256))

    summary: dict[str, Any] = {
        "repo": args.repo,
        "authenticated_user": who,
        "local_checked": len(local_map),
        "remote_existing": len(remote_tree),
        "to_upload_count": len(to_upload),
        "to_upload_bytes": sum(x[2] for x in to_upload),
        "to_upload_files": [{"path": p, "bytes": sz, "sha256": sha[:16]} for p, _, sz, sha in to_upload],
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.dry_run or not to_upload:
        return 0

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(fp)) for p, fp, _, _ in to_upload]
    info = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=ops,
        commit_message=args.commit_message,
    )
    print(json.dumps({"committed": True, "commit_url": info.commit_url, "oid": info.oid}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
