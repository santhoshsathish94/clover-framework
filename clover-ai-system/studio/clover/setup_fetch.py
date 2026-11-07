"""Downloads what Clover AI Studio needs, from the publishers, pinned in clover/launch/manifest.json: the model and its
projector (Hugging Face, LiquidAI), llama.cpp and stable-diffusion.cpp builds, MinGit and ffmpeg (GitHub releases), the
embedder, CSI's dictionary and lemmatiser, the picture models, and Clover's precomputed parts (the clover-ai-studio
dataset on Hugging Face). Every file is checked against the SHA-256 its publisher lists before it is used; a file that
is already there with the right hash is kept, and an interrupted download resumes. Run by the installer (and again
to repair):  python setup_fetch.py --base <install folder> [--components core,csi,pictures,c2c3] [--reuse <folder>]

--reuse: another Clover install whose files with the same hash are copied instead of downloaded."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "CloverAIStudio-setup/0.5"}
LOG = None
_LOG_LOCK = threading.Lock()
# progress for a setup window: "@@{json}" lines on stdout when CLOVER_PROGRESS=1 (the exe), or ON_EVENT(ev) in-process
PROGRESS = os.environ.get("CLOVER_PROGRESS") == "1"
ON_EVENT = None
STOP = None  # in-process (the phone app): a threading.Event; set, the run stops at the next chunk (resumable)
_HF_DIRECT_OK = None  # None = untested, True = huggingface.co works, False = sticky fallback to HF_MIRROR


class Stopped(Exception):
    pass


def emit(**ev):
    if PROGRESS:
        with _LOG_LOCK:
            print("@@" + json.dumps(ev), flush=True)
    if ON_EVENT is not None:
        ON_EVENT(ev)


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with _LOG_LOCK:
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


HF_MIRROR = "https://api.cloverframework.com/api/hf-mirror"


def _candidate_urls(url):
    global _HF_DIRECT_OK
    if url.startswith("https://huggingface.co/"):
        mirror_url = HF_MIRROR + url[len("https://huggingface.co"):]
        if _HF_DIRECT_OK is False:
            return [mirror_url, url]
        return [url, mirror_url]
    return [url]


def _mark_url_outcome(cur_url, ok):
    global _HF_DIRECT_OK
    if cur_url.startswith("https://huggingface.co/"):
        if ok and _HF_DIRECT_OK is None:
            _HF_DIRECT_OK = True
        elif not ok:
            _HF_DIRECT_OK = False


def _download_parallel_ranges(url, part, label, size, gname, gbefore, gtotal, t0):
    """Fast multi-connection HTTP Range downloader (6 parallel streams of 16 MB slices) for large model/dataset files."""
    urls = _candidate_urls(url)
    chosen_url = None
    for cand in urls:
        req = urllib.request.Request(cand, headers=dict(UA, Range="bytes=0-0"))
        t_out = 10 if cand.startswith("https://huggingface.co/") else 25
        try:
            with urllib.request.urlopen(req, timeout=t_out) as r:
                _mark_url_outcome(cand, True)
                if r.status == 206:
                    chosen_url = cand
                    break
        except Exception:
            _mark_url_outcome(cand, False)
            continue
    if not chosen_url:
        return False

    slice_size = 16 * 1024 * 1024
    ranges = []
    pos = 0
    while pos < size:
        end = min(size - 1, pos + slice_size - 1)
        ranges.append((pos, end))
        pos = end + 1

    with open(part, "wb") as f:
        f.truncate(size)

    prog_lock = threading.Lock()
    state = {"done": 0, "tick": 0.0, "last": 0.0, "start_t": time.time()}

    def fetch_slice(rng_pair):
        s_off, e_off = rng_pair
        expected = e_off - s_off + 1
        for attempt in range(4):
            if STOP is not None and STOP.is_set():
                raise Stopped("stopped")
            cur = s_off
            req = urllib.request.Request(chosen_url, headers=dict(UA, Range=f"bytes={s_off}-{e_off}"))
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206, got {r.status}")
                    with open(part, "r+b") as wf:
                        wf.seek(s_off)
                        while cur <= e_off:
                            if STOP is not None and STOP.is_set():
                                raise Stopped("stopped")
                            chunk = r.read(min(1 << 20, e_off - cur + 1))
                            if not chunk:
                                break
                            wf.write(chunk)
                            cur += len(chunk)
                            with prog_lock:
                                state["done"] += len(chunk)
                                now = time.time()
                                if now - state["tick"] > 0.5:
                                    state["tick"] = now
                                    rate_bps = int(state["done"] / max(0.1, now - state["start_t"]))
                                    emit(ev="bytes", name=gname, done=gbefore + state["done"],
                                         total=gtotal or size, rate=rate_bps)
                                if now - state["last"] > 5:
                                    state["last"] = now
                                    rate_mbs = state["done"] / max(0.1, now - state["start_t"]) / 1048576
                                    pct = state["done"] * 100 // size
                                    say(f"   {label}: {mb(state['done'])} of {mb(size)} ({pct}%), {rate_mbs:.1f} MB/s")
                if cur - s_off == expected:
                    return True
            except Stopped:
                raise
            except Exception:
                with prog_lock:
                    state["done"] = max(0, state["done"] - (cur - s_off))
                time.sleep(1 + attempt)
        return False

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(ranges))) as pool:
            results = list(pool.map(fetch_slice, ranges))
        return all(results)
    except Stopped:
        part.unlink(missing_ok=True)
        raise
    except Exception:
        part.unlink(missing_ok=True)
        return False


def download(url, dest, label, sha256=None, size=None, sha1=None, group=None):
    """url -> dest through dest.part (resumed), checked before it is moved into place. group: (name, bytes before this
    file, group total) - a dataset reports its files as one item."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last, t0, tick = 0.0, time.time(), 0.0
    gname, gbefore, gtotal = group if group else (label, 0, None)
    if not group:
        emit(ev="item", name=label, state="getting", size=size)

    if size is not None and size >= 12 * 1024 * 1024 and not part.exists():
        if _download_parallel_ranges(url, part, label, size, gname, gbefore, gtotal, t0):
            if good(part, sha256, size, sha1):
                os.replace(part, dest)
                got = dest.stat().st_size
                emit(ev="bytes", name=gname, done=gbefore + got, total=gtotal or got, rate=0)
                say(f"   {label}: done, {mb(got)} in {time.time() - t0:.0f} s, hash checked")
                return
            part.unlink(missing_ok=True)

    for attempt in range(8):
        urls = _candidate_urls(url)
        have = part.stat().st_size if part.exists() else 0
        if part.is_file() and size is not None and have == size:
            break
        ok_attempt = False
        last_exc = None
        for cur_url in urls:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(cur_url, headers=dict(UA, **({"Range": f"bytes={have}-"} if have else {})))
            t_out = 12 if (cur_url.startswith("https://huggingface.co/") and _HF_DIRECT_OK is not True) else 45
            try:
                with urllib.request.urlopen(req, timeout=t_out) as r:
                    _mark_url_outcome(cur_url, True)
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
                ok_attempt = True
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 416:  # nothing left to fetch
                    ok_attempt = True
                    break
                _mark_url_outcome(cur_url, False)
                last_exc = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                _mark_url_outcome(cur_url, False)
                last_exc = exc
        if ok_attempt:
            break
        if isinstance(last_exc, urllib.error.HTTPError) and last_exc.code in (401, 403, 404):
            raise NotFound(f"{url} answered {last_exc.code}") from None
        wait = min(30, 3 * (attempt + 1))
        say(f"   {label}: {type(last_exc).__name__}: {last_exc}; again in {wait} s")
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


def _list_dataset_files(repo, revision, prefix):
    """List dataset files via Hugging Face REST API (`urllib.request`, zero external library required on Android)
    and `HF_MIRROR`, merging entries by path so newly computed JEPA & C2/C3 artifacts on `HF_MIRROR` are always included."""
    pfx = prefix.rstrip("/")
    api_url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{pfx}?recursive=true"
    mirror_api_url = HF_MIRROR + api_url[len("https://huggingface.co"):]
    merged = {}
    for cur_url in (mirror_api_url, api_url) if _HF_DIRECT_OK is False else (api_url, mirror_api_url):
        req = urllib.request.Request(cur_url, headers=UA)
        t_out = 12 if cur_url.startswith("https://huggingface.co/") else 30
        try:
            with urllib.request.urlopen(req, timeout=t_out) as r:
                entries = json.loads(r.read().decode("utf-8"))
            for e in entries:
                if not isinstance(e, dict) or e.get("type") != "file":
                    continue
                lfs = e.get("lfs") if isinstance(e.get("lfs"), dict) else None
                p_str = str(e["path"])
                item_meta = {
                    "path": p_str,
                    "size": int(e.get("size") or (lfs.get("size") if lfs else 0) or 0),
                    "sha256": str(lfs.get("oid")) if (lfs and lfs.get("oid")) else None,
                    "sha1": None if (lfs and lfs.get("oid")) else str(e.get("oid") or ""),
                }
                if p_str not in merged or cur_url == mirror_api_url:
                    merged[p_str] = item_meta
        except Exception:
            continue
    if merged:
        return [merged[k] for k in sorted(merged)]
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    return [
        {
            "path": e.path,
            "size": e.size,
            "sha256": e.lfs.sha256 if e.lfs else None,
            "sha1": None if e.lfs else e.blob_id,
        }
        for e in HfApi().list_repo_tree(repo, repo_type="dataset", path_in_repo=pfx, recursive=True, revision=revision)
        if isinstance(e, RepoFile)
    ]


def fetch_dataset(base, item, reuse):
    label, prefix = item["name"], item["prefix"]
    dest_root = base / item["dest"]
    inc = tuple(item.get("include_prefixes") or ())
    exc = tuple(item.get("exclude_suffixes") or ())
    stamp_path = dest_root / ".clover-dataset.json"

    if stamp_path.is_file():
        try:
            st_doc = json.loads(stamp_path.read_text(encoding="utf-8"))
            st_files = st_doc.get("files") or {}
            if (
                st_doc.get("revision") == item["revision"]
                and tuple(st_doc.get("include_prefixes") or ()) == inc
                and st_files
                and (not ("jepa/" in inc) or any(k.startswith("jepa/") for k in st_files))
            ):
                all_present = True
                total_st = 0
                for rel_p, sz_expected in st_files.items():
                    fp_c = dest_root / rel_p
                    if not fp_c.is_file() or fp_c.stat().st_size != int(sz_expected):
                        all_present = False
                        break
                    total_st += int(sz_expected)
                if all_present:
                    emit(ev="bytes", name=label, done=total_st, total=total_st, rate=0)
                    say(f"ok {label} ({len(st_files)} were already here)")
                    return
        except Exception:
            pass

    all_files = _list_dataset_files(item["repo"], item["revision"], prefix)
    files = []
    for e in all_files:
        rel = e["path"][len(prefix):] if e["path"].startswith(prefix) else e["path"]
        if inc and not any(rel == p or rel.startswith(p) for p in inc):
            continue
        if exc and any(rel.endswith(s) for s in exc):
            continue
        files.append((rel, e))
    total = sum(e["size"] for _, e in files)
    say(f"get {label}: {len(files)} files, {mb(total)} ({item['repo']}@{item['revision'][:8]})")
    emit(ev="item", name=label, state="getting", size=total)

    def _check_existing(idx_pair):
        i, (rel, e) = idx_pair
        dest = dest_root / rel
        sha256, sha1, sz = e["sha256"], e["sha1"], e["size"]
        if good(dest, sha256, sz, sha1):
            return ("have", i, rel, e)
        src = (reuse / item["dest"] / rel) if reuse is not None else None
        if src is not None and good(src, sha256, sz, sha1):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return ("reused", i, rel, e)
        return ("todo", i, rel, e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(_check_existing, enumerate(files, 1)))

    have = sum(1 for status_k, _, _, _ in checked if status_k == "have")
    done_bytes = sum(e["size"] for status_k, _, _, e in checked if status_k in ("have", "reused"))
    if done_bytes:
        emit(ev="bytes", name=label, done=done_bytes, total=total, rate=0)

    todo = [(i, rel, e) for status_k, i, rel, e in checked if status_k == "todo"]
    small_todo = [(i, rel, e) for i, rel, e in todo if e["size"] < 12 * 1024 * 1024]
    large_todo = [(i, rel, e) for i, rel, e in todo if e["size"] >= 12 * 1024 * 1024]

    if small_todo:
        prog_lock = threading.Lock()
        state = {"done": done_bytes}

        def _fetch_small(item_triple):
            i, rel, e = item_triple
            if STOP is not None and STOP.is_set():
                raise Stopped("stopped")
            dest = dest_root / rel
            sha256, sha1, sz = e["sha256"], e["sha1"], e["size"]
            url = f"https://huggingface.co/datasets/{item['repo']}/resolve/{item['revision']}/{e['path']}"
            with prog_lock:
                cur_before = state["done"]
            download(url, dest, f"{label} [{i}/{len(files)}] {rel}", sha256, sz, sha1, group=(label, cur_before, total))
            with prog_lock:
                state["done"] += sz
                emit(ev="bytes", name=label, done=state["done"], total=total, rate=0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_fetch_small, t) for t in small_todo]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        done_bytes = state["done"]

    for i, rel, e in large_todo:
        dest = dest_root / rel
        sha256, sha1, sz = e["sha256"], e["sha1"], e["size"]
        url = f"https://huggingface.co/datasets/{item['repo']}/resolve/{item['revision']}/{e['path']}"
        download(url, dest, f"{label} [{i}/{len(files)}] {rel}", sha256, sz, sha1, group=(label, done_bytes, total))
        done_bytes += sz

    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        stamp_doc = {
            "revision": item["revision"],
            "include_prefixes": list(inc),
            "files": {rel: int(e["size"]) for rel, e in files},
        }
        stamp_path.write_text(json.dumps(stamp_doc, indent=2), encoding="utf-8")
    except OSError:
        pass
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
    want = {c.strip() for c in a.components.split(",") if c.strip()} | {"core", "csi", "c2c3"}
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
