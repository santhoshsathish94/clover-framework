"""Downloads what Clover AI Studio needs, from the publishers, pinned in clover/launch/manifest.json: the model and its
projector (Hugging Face, LiquidAI), llama.cpp and stable-diffusion.cpp builds, MinGit and ffmpeg (GitHub releases), the
embedder, CSI's dictionary and lemmatiser, the picture models, and Clover's precomputed parts (the clover-ai-studio
dataset on Hugging Face). Every file is checked against the SHA-256 its publisher lists before it is used; a file that
is already there with the right hash is kept, and an interrupted download resumes. Run by the installer (and again
to repair):  python setup_fetch.py --base <install folder> [--components core,csi,pictures,c2c3] [--reuse <folder>]

--reuse: another Clover install whose files with the same hash are copied instead of downloaded."""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "CloverAIStudio-setup/0.4"}
LOG = None
# progress for a setup window: "@@{json}" lines on stdout when CLOVER_PROGRESS=1 (the exe), or ON_EVENT(ev) in-process
PROGRESS = os.environ.get("CLOVER_PROGRESS") == "1"
ON_EVENT = None
STOP = None  # in-process (the phone app): a threading.Event; set, the run stops at the next chunk (resumable)


class Stopped(Exception):
    pass


def emit(**ev):
    if PROGRESS:
        print("@@" + json.dumps(ev), flush=True)
    if ON_EVENT is not None:
        ON_EVENT(ev)


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def mb(n):
    return f"{n / 1048576:,.0f} MB" if n >= 1048576 else f"{n / 1024:,.0f} KB"


def file_hash(path, kind="sha256", size=None):
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % (size if size is not None else path.stat().st_size))
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def good(path, sha256=None, size=None, sha1=None):
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        return False
    return file_hash(path, "sha1", size) == sha1 if sha1 else file_hash(path) == sha256


class NotFound(Exception):
    pass


def download(url, dest, label, sha256=None, size=None, sha1=None, group=None):
    """url -> dest through dest.part (resumed), checked before it is moved into place. group: (name, bytes before this
    file, group total) - a dataset reports its files as one item."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last, t0, tick = 0.0, time.time(), 0.0
    gname, gbefore, gtotal = group if group else (label, 0, None)
    if not group:
        emit(ev="item", name=label, state="getting", size=size)
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        if part.is_file() and size is not None and have == size:
            break
        req = urllib.request.Request(url, headers=dict(UA, **({"Range": f"bytes={have}-"} if have else {})))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                append = have and r.status == 206
                total = (have if append else 0) + int(r.headers.get("Content-Length") or 0)
                done = have if append else 0
                start_done, start_t = done, time.time()
                with open(part, "ab" if append else "wb") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        if STOP is not None and STOP.is_set():
                            raise Stopped("stopped")
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - tick > 0.5:
                            tick = now
                            emit(ev="bytes", name=gname, done=gbefore + done, total=gtotal or total or size,
                                 rate=int((done - start_done) / max(0.1, now - start_t)))
                        if now - last > 5:
                            last = time.time()
                            rate = (done - start_done) / max(0.1, time.time() - start_t) / 1048576
                            pct = f" ({done * 100 // total}%)" if total else ""
                            say(f"   {label}: {mb(done)} of {mb(total) if total else '?'}{pct}, {rate:.1f} MB/s")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # nothing left to fetch
                break
            if exc.code in (401, 403, 404):
                raise NotFound(f"{url} answered {exc.code}") from None
            wait = min(60, 5 * (attempt + 1))
            say(f"   {label}: HTTP {exc.code}, again in {wait} s")
            time.sleep(wait)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            wait = min(60, 5 * (attempt + 1))
            say(f"   {label}: {type(exc).__name__}: {exc}; again in {wait} s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"{label}: download kept failing ({url})")
    if not good(part, sha256, size, sha1):
        got = part.stat().st_size if part.exists() else 0
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{label}: the file does not match its published hash ({mb(got)}); it was deleted")
    os.replace(part, dest)
    got = dest.stat().st_size
    emit(ev="bytes", name=gname, done=gbefore + got, total=gtotal or got, rate=0)
    say(f"   {label}: done, {mb(dest.stat().st_size)} in {time.time() - t0:.0f} s, hash checked")


def fetch_file(base, item, reuse):
    dest = base / item["dest"]
    label, sha, size = item["name"], item.get("sha256"), item.get("size")
    if good(dest, sha, size):
        if size:
            emit(ev="bytes", name=label, done=size, total=size, rate=0)
        say(f"ok {label} (already here)")
        return
    if reuse is not None:
        src = reuse / item["dest"]
        if good(src, sha, size):
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, part)
            if good(part, sha, size):
                os.replace(part, dest)
                say(f"ok {label} (copied from {reuse})")
                return
            part.unlink(missing_ok=True)
    say(f"get {label}" + (f" ({mb(size)})" if size else ""))
    download(item["url"], dest, label, sha, size)


def github_latest(spec):
    req = urllib.request.Request(f"https://api.github.com/repos/{spec['repo']}/releases/latest", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.loads(r.read())
    for a in sorted(rel.get("assets") or [], key=lambda a: a["name"], reverse=True):
        if re.match(spec["pattern"], a["name"]) and str(a.get("digest") or "").startswith("sha256:"):
            return a["browser_download_url"], a["digest"][7:], a["size"], a["name"]
    raise RuntimeError(f"no release asset of {spec['repo']} matches {spec['pattern']} with a digest")


def found(dest, check):
    if "*" in check:
        return next((p for p in dest.rglob(check.split("/")[-1]) if fnmatch.fnmatch(p.name, check.split("/")[-1])), None)
    p = dest / check
    return p if p.is_file() else None


def fetch_archive(base, item, reuse):
    dest, label = base / item["dest"], item["name"]
    stamp = dest / ".clover-archive"
    if item.get("github_latest"):
        if found(dest, item["check"]) and stamp.is_file():
            say(f"ok {label} (already here: {stamp.read_text(encoding='utf-8').strip()})")
            return
        url, sha, size, name = github_latest(item["github_latest"])
    else:
        url, sha, size, name = item["url"], item["sha256"], item.get("size"), item["url"].rsplit("/", 1)[1]
        if found(dest, item["check"]) and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == sha:
            say(f"ok {label} (already here)")
            return
    z = base / "downloads" / name
    if not good(z, sha, size):
        old = (reuse / "downloads" / name) if reuse is not None else None
        tools_copy = (reuse / Path(item["dest"]).parent / name) if reuse is not None else None
        src = next((p for p in (old, tools_copy) if p is not None and good(p, sha, size)), None)
        if src is not None:
            z.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, z)
            say(f"ok {label} archive (copied from {src.parent})")
        else:
            say(f"get {label}" + (f" ({mb(size)})" if size else ""))
            download(url, z, label, sha, size)
    new = dest.with_name(dest.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():  # nothing may land outside the folder
            if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                raise RuntimeError(f"{label}: unsafe path {m!r} in the archive")
        zf.extractall(new)
    if item.get("flatten"):
        exe = next(new.rglob(item["flatten"]), None)
        if exe is None:
            raise RuntimeError(f"{label}: {item['flatten']} is not in the archive")
        if exe.parent != new:
            for p in list(exe.parent.iterdir()):
                shutil.move(str(p), str(new / p.name))
    if not found(new, item["check"]):
        raise RuntimeError(f"{label}: {item['check']} is not in the archive")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(new, dest)
    (dest / ".clover-archive").write_text(sha if not item.get("github_latest") else name, encoding="utf-8")
    if item.get("bin_txt"):
        exe = found(dest, item["check"])
        (dest / "BIN.txt").write_text(str(exe.parent), encoding="utf-8")
    z.unlink(missing_ok=True)  # the stamp records what was unpacked; a repair downloads again only if the folder is gone
    say(f"ok {label} (unpacked into {item['dest']})")


def fetch_dataset(base, item, reuse):
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    label, prefix = item["name"], item["prefix"]
    files = [e for e in HfApi().list_repo_tree(item["repo"], repo_type="dataset", path_in_repo=prefix.rstrip("/"),
                                               recursive=True, revision=item["revision"]) if isinstance(e, RepoFile)]
    total = sum(e.size for e in files)
    say(f"get {label}: {len(files)} files, {mb(total)} ({item['repo']}@{item['revision'][:8]})")
    emit(ev="item", name=label, state="getting", size=total)
    have, before = 0, 0
    for i, e in enumerate(files, 1):
        rel = e.path[len(prefix):]
        dest = base / item["dest"] / rel
        sha256, sha1 = (e.lfs.sha256, None) if e.lfs else (None, e.blob_id)
        if good(dest, sha256, e.size, sha1):
            have += 1
            before += e.size
            emit(ev="bytes", name=label, done=before, total=total, rate=0)
            continue
        src = (reuse / item["dest"] / rel) if reuse is not None else None
        if src is not None and good(src, sha256, e.size, sha1):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            before += e.size
            continue
        url = f"https://huggingface.co/datasets/{item['repo']}/resolve/{item['revision']}/{e.path}"
        download(url, dest, f"{label} [{i}/{len(files)}] {rel}", sha256, e.size, sha1, group=(label, before, total))
        before += e.size
    say(f"ok {label} ({have} were already here)")


def main(argv=None):
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--components", default="core,csi,pictures,c2c3")
    ap.add_argument("--reuse", default="")
    ap.add_argument("--manifest", default=str(HERE / "launch" / "manifest.json"))
    a = ap.parse_args(argv)
    base = Path(a.base).resolve()
    reuse = Path(a.reuse).resolve() if a.reuse else None
    if reuse is not None and reuse == base:
        reuse = None
    LOG = base / "logs" / "setup-fetch.log"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:  # Windows' certificate store (a company proxy's root is trusted there, not in Python's bundle)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    man = json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))  # a manifest saved by Notepad starts with a BOM
    want = {c.strip() for c in a.components.split(",") if c.strip()} | {"core"}
    say(f"Clover AI Studio setup {man['version']}: getting {', '.join(sorted(want))} into {base}")
    status = {}
    jobs = ([(fetch_archive, x) for x in man["archives"]] + [(fetch_file, x) for x in man["files"]]
            + [(fetch_dataset, x) for x in man["datasets"]])
    jobs = [(fn, item) for fn, item in jobs if item["component"] in want]
    # sizes from the manifest (a dataset's "bytes"; ffmpeg's latest build is estimated): the window's overall bar
    emit(ev="plan", total=sum(item.get("size") or item.get("bytes") or 0 for _, item in jobs), items=len(jobs))
    for i, (fn, item) in enumerate(jobs, 1):
        comp = item["component"]
        emit(ev="item", name=item["name"], state="checking", i=i, n=len(jobs), size=item.get("size") or item.get("bytes") or 0)
        try:
            fn(base, item, reuse)
            status.setdefault(comp, {"ok": True, "problems": []})
            emit(ev="item", name=item["name"], state="ok", i=i, n=len(jobs))
        except Stopped:
            say("SETUP_FETCH_STOPPED")
            emit(ev="done", ok=False, problems=["stopped"], stopped=True)
            return 3
        except Exception as exc:  # noqa: BLE001 - a part that cannot be fetched is reported, the rest continues
            say(f"PROBLEM {item['name']}: {type(exc).__name__}: {exc}")
            emit(ev="item", name=item["name"], state="problem", i=i, n=len(jobs), why=str(exc)[:300])
            status.setdefault(comp, {"ok": True, "problems": []})
            status[comp]["ok"] = False
            status[comp]["problems"].append(f"{item['name']}: {exc}")
    out = base / "config" / "installed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": man["version"], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "components": status}, indent=2), encoding="utf-8")
    bad = [c for c, s in status.items() if not s["ok"]]
    emit(ev="done", ok="core" not in bad, problems=bad)
    say("SETUP_FETCH_DONE" + (" with problems in " + ", ".join(bad) if bad else ""))
    return 1 if "core" in bad else 0


if __name__ == "__main__":
    sys.exit(main())
