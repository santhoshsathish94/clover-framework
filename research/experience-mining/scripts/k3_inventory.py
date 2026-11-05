"""Read-only schema inventory of Clover records on k3. Field names, counts, sizes, dates and identifiers only; no message
text. The live K3 server is only inspected through /proc and file metadata (never signalled, never connected to).
Usage: python3 k3_inventory.py"""
import collections
import json
import os
import re
import subprocess
from pathlib import Path


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, errors="replace").stdout


def kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict(" + ",".join(sorted(v)[:14]) + ")"
    return type(v).__name__


def jsonl_schema(paths, label, values=(), max_lines=200000):
    keys, n, bad, vals = collections.Counter(), 0, 0, collections.defaultdict(collections.Counter)
    for p in paths:
        try:
            fh = open(p, encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            n += 1
            if isinstance(r, dict):
                for k, v in r.items():
                    keys[(k, kind(v))] += 1
                    if k in values and isinstance(v, (str, bool, int)) and len(str(v)) <= 48:
                        vals[k][str(v)] += 1
    print(f"== {label}: files={len(paths)} records={n} unparsable={bad}")
    for (k, t), c in sorted(keys.items(), key=lambda x: -x[1])[:45]:
        print(f"    {k:26s} {t[:100]:100s} {c}")
    for k, c in vals.items():
        print(f"    values[{k}]: {dict(c.most_common(18))}")


def tree(root, depth=3, label=None):
    root = Path(root)
    if not root.exists():
        print(f"== {label or root}: missing")
        return
    print(f"== {label or root}")
    for d, subdirs, files in os.walk(root):
        rel = Path(d).relative_to(root)
        if ".git" in rel.parts or len(rel.parts) > depth:
            subdirs[:] = []
            continue
        size = sum((Path(d) / f).stat().st_size for f in files if (Path(d) / f).is_file())
        exts = collections.Counter(Path(f).suffix for f in files)
        if files:
            print(f"    {str(rel):60s} files={len(files):6d} KB={size // 1024:9d} {dict(exts.most_common(4))}")


def git_summary(repo, label):
    if not (Path(repo) / ".git").exists():
        print(f"== git {label}: not a repo")
        return
    log = sh("git", "-C", str(repo), "log", "--name-status", "--format=@@%ci|%s")
    commits, status, subj, area, days = 0, collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for line in log.splitlines():
        if line.startswith("@@"):
            commits += 1
            date, _, s = line[2:].partition("|")
            days[date[:10]] += 1
            subj[re.sub(r"[0-9a-f]{6,}|\d+", "#", s)[:60]] += 1
        elif "\t" in line:
            st, path = line.split("\t", 1)
            status[st[0]] += 1
            parts = path.split("/")
            area["/".join(parts[:2]) if len(parts) > 2 else parts[0]] += 1
    print(f"== git {label}: commits={commits} changes={dict(status)} days={dict(sorted(days.items()))}")
    print(f"    subjects: {dict(subj.most_common(12))}")
    print(f"    areas: {dict(area.most_common(16))}")


O = Path("/opt/clover-k3/core-observe")
print("== core-observe sandbox")
tree(O / "traces", 2, "core-observe/traces")
tree(O / "archive", 3, "core-observe/archive")
for base in ("fw-base", "fw-base-csi-off", "fw-base-csi-on"):
    tree(O / base / "clover-memory-repo", 3, f"{base}/clover-memory-repo")
    git_summary(O / base / "clover-memory-repo", f"{base}/clover-memory-repo")
    tree(O / base / "cache", 2, f"{base}/cache")
tree(O / "laptop-base", 4, "laptop-base")
git_summary(O / "laptop-base" / "clover-ai-studio", "laptop-base/clover-ai-studio")
mem = list((O / "fw-base" / "clover-memory-repo").rglob("*.jsonl"))
jsonl_schema(mem, "fw-base/clover-memory-repo *.jsonl", ("role", "kind", "type", "feedback"))
print("== tracer events (traces/after-a-final/framework.jsonl): layer, fn, phase")
ev = collections.Counter()
for line in open(O / "traces/after-a-final/framework.jsonl", encoding="utf-8", errors="replace"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    ev[(r.get("layer"), str(r.get("fn"))[:60], r.get("phase"))] += 1
for (l, f, p), c in sorted(ev.items(), key=lambda x: -x[1])[:40]:
    print(f"    {str(l):12s} {f:60s} {str(p):10s} {c}")
jsonl_schema([O / "traces/after-a-final/framework-turns.jsonl"], "drive turns (after-a-final)", ("stack",))

print("== live K3 server: process metadata only")
pid = None
for line in sh("ss", "-ltnp").splitlines():
    if "127.0.0.1:8765 " in line:
        m = re.search(r"pid=(\d+)", line)
        pid = m.group(1) if m else None
if pid:
    print("    pid", pid, "cwd", os.readlink(f"/proc/{pid}/cwd"))
    cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    print("    cmd", [c.decode(errors="replace")[:80] for c in cmd if c][:6])
    files = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            t = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if t.startswith("/") and not t.startswith(("/dev", "/proc")) and not re.search(r"\.(so|safetensors|bin|pt|npy|npz|gguf)(\.|$)", t):
            files.add(t)
    for t in sorted(files)[:40]:
        print("    open:", t)
    cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    for name in ("logs", "chat_logs", "sessions", "memory", "feedback", "traces", "data/sessions", "runtime"):
        tree(cwd / name, 2, f"live cwd/{name}")
print("== other record-like locations (metadata)")
for p in ("/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree",
          "/opt/clover-k3/s23-omni2b", "/tmp/k3_baseline_70_results.jsonl", "/opt/clover-k3/lfm3-hybridagent/runs"):
    q = Path(p)
    if q.is_file():
        print(f"    {p}: file {q.stat().st_size} bytes")
    else:
        tree(q, 1, p)
