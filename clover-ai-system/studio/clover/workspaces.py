"""Folders and GitHub repositories the user works on with Clover, and a context file in each folder.

The user adds a local folder or a GitHub repository (cloned into CloverAI\\workspaces). In a workspace Clover keeps
`<folder>-context.md` in each folder (two levels deep) and `<repo>-context-index.md` at the top, the names the phone
app uses. A context record is offered, not imposed (AGENTS.md), so:
- Clover never runs git add, commit or push in a user's repository: the files stay for the user to keep or delete.
- Context files describe structure (names, sizes, Markdown headings, Python docstrings and definitions) and never copy
  other file contents; files that commonly hold secrets are listed by name only.
- The Notes section of each file is kept when Clover refreshes the lists: the place for what the human and Clover
  learn about that folder. A file of the same name that Clover did not write is left alone.
"""
import ast
import base64
import datetime
import json
import os
import re
import shutil
import stat
import subprocess
import threading
from pathlib import Path

import semantics

MARK = "<!-- clover:context v1 - Clover AI Studio keeps the lists up to date; edit the Notes freely -->"
NOTES = "## Notes"
NOTES_HINT = ("_What the human and Clover learn about this folder: decisions, constraints, what was tried. Clover keeps "
              "this section when it refreshes the lists above._")
PHONE_NOTES = "## User & Agent Notes"
PHONE_TITLE = re.compile(r"^# `[^`]*` — (?:Folder Context|Overall Repository Context Index)")
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "env", "target", "build", "dist", "out", "bin", "obj",
             "site-packages", "vendor"}
SECRETISH = re.compile(r"(?i)(?:^\.?env(?:\..*)?$|\.(?:pem|key|p12|pfx|kdbx|keystore|jks)$|^id_(?:rsa|dsa|ecdsa|ed25519)"
                       r"|credential|secret|token|passw)")
BACKUP = re.compile(r"\.clover-\d{8}-\d{6}\.bak$")
KINDS = {".py": "Python", ".md": "Markdown", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
         ".tsx": "TypeScript", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".ini": "settings",
         ".cfg": "settings", ".html": "HTML", ".css": "CSS", ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin",
         ".c": "C", ".h": "C header", ".cpp": "C++", ".hpp": "C++ header", ".cs": "C#", ".ps1": "PowerShell",
         ".sh": "shell script", ".bat": "batch script", ".sql": "SQL", ".txt": "text", ".csv": "CSV", ".ipynb": "notebook",
         ".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image", ".pdf": "PDF"}
MAX_FOLDERS, MAX_FILES, DEPTH = 80, 60, 2
PROJECT_WORDS = re.compile(r"\b(?:repo|repository|project|codebase|workspace|this code|the code)\b", re.I)
GH_URL = re.compile(r"^(?:https?://github\.com/|git@github\.com:)([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9._-]{1,100}?)"
                    r"(?:\.git)?/?$")
GH_SHORT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9._-]{1,100}?)(?:\.git)?$")
TOKEN = re.compile(r"^[\x21-\x7e]{20,255}$")
STOP = set("""the and for are was were this that with from its into about what which how does can could would should your
our their there then than them they you not but all any has have had will shall may might must just only also use""".split())


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def slug(text, n=32):
    return re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")[:n].strip("-") or "folder"


def _size(n):
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def _terms(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9_\-]{2,}", (text or "").lower()) if w not in STOP}


def describe_file(p, size):
    """What a file is, from its structure: never its other contents (a settings file's first line held a secret)."""
    kind = KINDS.get(p.suffix.lower(), "")
    if SECRETISH.search(p.name):
        return ", ".join(x for x in (kind, "not read: may hold secrets") if x)
    if p.is_symlink() or size > 1_000_000 or p.suffix.lower() not in (".md", ".py"):
        return kind
    try:
        raw = p.read_bytes()[:300_000]
    except OSError:
        return kind
    if b"\x00" in raw[:4096]:
        return "binary"
    text = raw.decode("utf-8", errors="replace")
    if p.suffix.lower() == ".md":
        head = next((ln.lstrip("#").strip() for ln in text.splitlines() if ln.startswith("#")), "")
        return f"{kind}: {head[:100]}" if head else kind
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return kind
    doc = (ast.get_docstring(tree) or "").strip().splitlines()
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    out = kind + (f": {doc[0].strip()[:100].rstrip('.')}" if doc else "")
    return out + (f". Defines {', '.join(defs[:6])}{', ...' if len(defs) > 6 else ''}" if defs else "")


def _file_lines(files):
    lines = [f"- [`{n}`](<{n}>) ({_size(s)}){' - ' + d if d else ''}" for n, s, d in files[:MAX_FILES]]
    if len(files) > MAX_FILES:
        lines.append(f"- ... and {len(files) - MAX_FILES} more")
    return lines or ["- (none)"]


def _existing(fp):
    """(ours, notes): ours = missing or written by Clover (this app or the phone app); notes = its Notes section."""
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return True, ""
    except OSError:
        return False, ""
    sep = "\n" + NOTES + "\n"
    if MARK in text:
        notes = text.split(sep, 1)[1].strip() if sep in text else ""
        return True, "" if notes == NOTES_HINT else notes
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if PHONE_TITLE.match(first):
        notes = text.split(PHONE_NOTES, 1)[1].strip() if PHONE_NOTES in text else ""
        return True, "" if notes.startswith("Add folder-specific architecture notes") else notes
    return False, ""


def _plain(text):
    """A context file as the model reads it: no marker, no Markdown links (the model copied them into its answer)."""
    text = re.sub(r"\[`([^`]+)`\]\(<[^>]*>\)", r"\1", text.replace(MARK, ""))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


EXCLUDE_START, EXCLUDE_END = "# clover:context - files Clover AI Studio keeps (not part of the project)", "# clover:context end"


def _git_exclude(root, rels):
    """Lists Clover's context files in the repository's own .git/info/exclude (local, never committed), so `git status`
    stays as the user left it; rels=[] removes the list. Exact paths only: no pattern can hide a file of the user's."""
    git = Path(root) / ".git"
    if not git.is_dir():
        return
    fp = git / "info" / "exclude"
    try:
        text = fp.read_text(encoding="utf-8")
    except OSError:
        text = ""
    text = re.sub(re.escape(EXCLUDE_START) + r"[\s\S]*?" + re.escape(EXCLUDE_END) + r"\n?", "", text)
    if rels:
        text = text.rstrip("\n") + ("\n" if text.strip() else "") + "\n".join(
            [EXCLUDE_START] + ["/" + r.replace("\\", "/") for r in sorted(rels)] + [EXCLUDE_END]) + "\n"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text, encoding="utf-8")


def _write_if_changed(fp, body):
    try:
        if fp.read_text(encoding="utf-8") == body:
            return False
    except (OSError, ValueError):
        pass
    fp.write_text(body, encoding="utf-8")
    return True


def _git_env(token=None):
    """No prompts (a hidden server cannot answer them); a token travels as a header in the environment, never in the
    command line or in .git/config."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                   GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}")
    return env


def _redact(text, token):
    if token:
        text = text.replace(token, "***").replace(base64.b64encode(f"x-access-token:{token}".encode()).decode(), "***")
    return text


def _rmtree(p):
    def unlock(fn, path, _):  # git marks its object files read-only
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    shutil.rmtree(p, onexc=unlock)


def _dpapi(data, protect):
    """Windows data protection for the current user: only this Windows account can decrypt what it encrypted."""
    import vault
    return vault.dpapi(data, protect)


class GitHubAuth:
    """CloverAI\\vault\\github.json: the user's GitHub token for private repositories, typed into Settings on this laptop
    and kept encrypted for this Windows account (DPAPI). describe() says whether one is set, never the token."""

    def __init__(self, path):
        self.path, self.lock, self.token = Path(path), threading.Lock(), None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            t = (_dpapi(base64.b64decode(raw["token_dpapi"]), False).decode("utf-8") if raw.get("token_dpapi")
                 else raw.get("token"))
            self.token = t if isinstance(t, str) and TOKEN.match(t) else None
        except (OSError, ValueError, KeyError):
            pass

    def describe(self):
        return {"token_set": self.token is not None,
                "create_url": "https://github.com/settings/personal-access-tokens/new"}

    def update(self, token=None, forget=False):
        with self.lock:
            if forget:
                self.token = None
            elif token is not None:
                token = str(token).strip()
                if not TOKEN.match(token):
                    raise ValueError("that does not look like a GitHub token (20-255 visible characters, no spaces)")
                self.token = token
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            blob = (base64.b64encode(_dpapi(self.token.encode("utf-8"), True)).decode("ascii") if self.token else None)
            tmp.write_text(json.dumps({"token_dpapi": blob}), encoding="utf-8")
            tmp.replace(self.path)
        return self.describe()


class Workspaces:
    """config/workspaces.json: the folders the user added, which one is in use, and the context files Clover wrote."""

    def __init__(self, config_path, clone_root, app_dir, auth=None):
        self.path, self.clone_root, self.app_dir = Path(config_path), Path(clone_root), Path(app_dir).resolve()
        self.auth, self.lock = auth, threading.RLock()
        self.items, self.active_name = [], None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [w for w in raw.get("workspaces") or [] if isinstance(w, dict) and w.get("name") and w.get("path")]
            self.active_name = raw.get("active") if any(w["name"] == raw.get("active") for w in self.items) else None
        except (OSError, ValueError):
            pass

    # ---------------- registry ----------------
    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"active": self.active_name, "workspaces": self.items}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _find(self, name):
        for w in self.items:
            if w["name"] == name:
                return w
        raise ValueError(f"no workspace named {name!r}")

    def describe(self):
        with self.lock:
            out = []
            for w in self.items:
                out.append(dict(w, exists=Path(w["path"]).is_dir(), active=w["name"] == self.active_name))
            return {"active": self.active_name, "workspaces": out, "clone_root": str(self.clone_root),
                    "github": self.auth.describe() if self.auth else {"token_set": False}}

    def active(self):
        with self.lock:
            if self.active_name is None:
                return None
            w = next((w for w in self.items if w["name"] == self.active_name), None)
            return dict(w) if w and Path(w["path"]).is_dir() else None

    def activate(self, name):
        with self.lock:
            if name is not None:
                self._find(name)
            self.active_name = name
            self._save()
        return self.describe()

    def remove(self, name):
        """Takes the workspace off the list; its files, and the context files in it, stay where they are."""
        with self.lock:
            self._find(name)
            self.items = [w for w in self.items if w["name"] != name]
            if self.active_name == name:
                self.active_name = None
            self._save()
        return self.describe()

    def _unique(self, name):
        base, n, names = slug(name, 40), 2, {w["name"] for w in self.items}
        out = base
        while out in names:
            out, n = f"{base}-{n}", n + 1
        return out

    def _too_broad(self, p):
        home = Path.home().resolve()
        if p == Path(p.anchor):
            return "a whole drive"
        if p == home or p in [home / d for d in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos",
                                                  "OneDrive")]:
            return "a whole personal folder; choose a project folder inside it"
        win = Path(os.environ.get("WINDIR", r"C:\Windows")).resolve()
        if p == win or win in p.parents or "program files" in str(p).lower():
            return "a Windows or program folder"
        clones = self.clone_root.resolve()
        if (p == self.app_dir or self.app_dir in p.parents) and clones not in p.parents:
            return "Clover's own folder"
        if p in self.app_dir.parents:
            return "a folder that contains Clover's own files"
        return None

    def add(self, source, name=None):
        """source: an absolute folder path (or ~\\...), a GitHub URL, or owner/repo. Returns describe() + the context
        summary of the new workspace, which becomes the active one."""
        src = str(source or "").strip().strip('"')
        if not src:
            raise ValueError("enter a folder path or a GitHub repository (owner/repo or its URL)")
        expanded = os.path.expandvars(os.path.expanduser(src))
        if os.path.isabs(expanded) or src.startswith("~"):
            p = Path(expanded).resolve()
            if not p.is_dir():
                raise ValueError(f"{p} is not a folder on this laptop")
            why = self._too_broad(p)
            if why:
                raise ValueError(f"{p} is {why}")
            entry = {"source": "folder", "repo": None}
        else:
            m = GH_URL.match(src) or GH_SHORT.match(src)
            if not m:
                raise ValueError("use a full folder path (C:\\...) or a GitHub repository (owner/repo or https://github.com/owner/repo)")
            owner, repo = m.group(1), m.group(2)
            p = self._clone(owner, repo, name or repo)
            entry = {"source": "github", "repo": f"{owner}/{repo}"}
        with self.lock:
            same = next((w for w in self.items if Path(w["path"]).resolve() == p), None)
            if same is None:
                same = dict(entry, name=self._unique(name or p.name), path=str(p), added_at=_now(), folders=[],
                            index=None, context_at=None)
                self.items.append(same)
            self.active_name = same["name"]
            self._save()
        summary = self.refresh(same["name"])
        return dict(self.describe(), context=summary)

    def _clone(self, owner, repo, name):
        dest = self.clone_root / slug(name, 40)
        token = self.auth.token if self.auth else None
        if dest.exists():
            r = subprocess.run(["git", "-C", str(dest), "remote", "get-url", "origin"], capture_output=True, text=True,
                               timeout=30)
            url = r.stdout.strip().lower().removesuffix(".git").rstrip("/")
            if r.returncode == 0 and url.endswith(f"github.com/{owner}/{repo}".lower()):
                return dest.resolve()
            raise ValueError(f"{dest} already exists and is not a clone of {owner}/{repo}")
        self.clone_root.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(["git", "clone", "--filter=blob:none", "--", f"https://github.com/{owner}/{repo}.git",
                                str(dest)], capture_output=True, text=True, timeout=600, env=_git_env(token),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            if dest.exists():
                _rmtree(dest)
            raise ValueError(f"cloning {owner}/{repo} took longer than 10 minutes and was stopped") from None
        if r.returncode != 0:
            if dest.exists():
                _rmtree(dest)
            err = (r.stderr or r.stdout or "").strip()
            if re.search(r"could not read Username|Authentication failed|terminal prompts disabled|Repository not found", err):
                raise ValueError(f"GitHub did not give access to {owner}/{repo}: it does not exist, or it is private"
                                 + (" and the saved token cannot read it." if token else
                                    ". For a private repository, add a GitHub token in Settings first."))
            why = [ln.strip() for ln in err.splitlines() if ln.strip().startswith(("fatal:", "remote:", "error:"))]
            msg = _redact(" ".join(why) or err, token)[-300:]
            hint = "" if token else " If it is a private repository, add a GitHub token in Settings first."
            raise ValueError(f"could not clone {owner}/{repo}: {msg}{hint}")
        return dest.resolve()

    def pull(self, name):
        """Updates a cloned GitHub workspace (fast-forward only: never a merge, never a push)."""
        with self.lock:
            w = dict(self._find(name))
        if w.get("source") != "github":
            raise ValueError("only workspaces cloned from GitHub are updated here")
        token = self.auth.token if self.auth else None
        r = subprocess.run(["git", "-C", w["path"], "pull", "--ff-only"], capture_output=True, text=True, timeout=600,
                           env=_git_env(token), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = _redact((r.stdout + "\n" + r.stderr).strip(), token)[-400:]
        if r.returncode != 0:
            raise ValueError(f"git pull --ff-only failed: {out}")
        return dict(self.describe(), message=out, context=self.refresh(name))

    # ---------------- context files ----------------
    def _scan(self, root, index_name):
        root_files, top, folders, limit_hit = [], [], {}, False
        for cur, dirs, files in os.walk(root):
            cur_p = Path(cur)
            rel = cur_p.relative_to(root).as_posix()
            depth = 0 if rel == "." else rel.count("/") + 1
            dirs[:] = sorted((d for d in dirs if not d.startswith(".") and d.lower() not in SKIP_DIRS), key=str.lower)
            own = index_name if depth == 0 else f"{slug(cur_p.name)}-context.md"
            entries = []
            for f in sorted(files, key=str.lower):
                if f.startswith(".") or f == own or BACKUP.search(f):
                    continue
                fp = cur_p / f
                try:
                    size = fp.stat().st_size
                except OSError:
                    continue
                entries.append((f, size, describe_file(fp, size)))
            if depth == 0:
                root_files, top = entries, list(dirs)
                continue
            if len(folders) >= MAX_FOLDERS:
                limit_hit, dirs[:] = True, []
                continue
            if entries or dirs:
                folders[rel] = {"files": entries, "subdirs": list(dirs), "depth": depth}
            if depth >= DEPTH:
                dirs[:] = []
        return root_files, top, folders, limit_hit

    def refresh(self, name):
        """Writes or updates the context files of a workspace. Returns what was written, unchanged and skipped."""
        with self.lock:
            w = self._find(name)
            root = Path(w["path"])
        if not root.is_dir():
            raise ValueError(f"{root} no longer exists")
        index_name = f"{slug(root.name, 36)}-context-index.md"
        root_files, top, folders, limit_hit = self._scan(root, index_name)
        written, unchanged, skipped, known = [], [], [], []

        def put(fp, rel, render):
            ours, notes = _existing(fp)
            if not ours:
                skipped.append(f"{rel} (a file of that name exists that Clover did not write)")
                return
            (written if _write_if_changed(fp, render(notes)) else unchanged).append(rel)
            known.append(rel)

        for rel, f in folders.items():
            ctx_rel = f"{rel}/{slug(rel.rsplit('/', 1)[-1])}-context.md"

            def render(notes, rel=rel, f=f):
                lines = [f"# `{rel}/` - folder context", MARK, "",
                         f"Part of the workspace `{w['name']}`. Clover reads this file when a question is about this "
                         "folder.", "", "## Files", *_file_lines(f["files"])]
                if f["subdirs"]:
                    lines += ["", "## Subfolders"]
                    for d in f["subdirs"]:
                        sub = f"{rel}/{d}"
                        lines.append(f"- [`{d}/`](<{d}/{slug(d)}-context.md>)" if sub in folders else f"- `{d}/`")
                return "\n".join(lines + ["", NOTES, notes or NOTES_HINT, ""])
            put(root / ctx_rel, ctx_rel, render)

        def render_index(notes):
            lines = [f"# `{root.name}` - context index", MARK, "",
                     "Clover keeps a context file in each folder (two levels deep) and this index at the top. They "
                     "describe structure only; read the files themselves for details.", "", "## Top-level files",
                     *_file_lines(root_files), "", "## Folders"]
            for d in top:
                if d not in folders:
                    continue
                lines.append(f"- [`{d}/`](<{d}/{slug(d)}-context.md>) - {len(folders[d]['files'])} files")
                for c in folders[d]["subdirs"]:
                    sub = f"{d}/{c}"
                    if sub in folders:
                        lines.append(f"  - [`{sub}/`](<{sub}/{slug(c)}-context.md>) - {len(folders[sub]['files'])} files")
            if limit_hit:
                lines.append(f"- ... more folders not indexed (limit {MAX_FOLDERS})")
            return "\n".join(lines + ["", NOTES, notes or NOTES_HINT, ""])
        put(root / index_name, index_name, render_index)
        _git_exclude(root, known)

        with self.lock:
            w = self._find(name)
            w["folders"] = [k for k in known if k != index_name]
            w["index"] = index_name if index_name in known else None
            w["context_at"] = _now()
            self._save()
        return {"workspace": name, "path": str(root), "written": written, "unchanged": len(unchanged),
                "skipped": skipped, "folders": len(folders), "limit_hit": limit_hit}

    def touch(self, path):
        """After Clover writes a file inside the active workspace, its folder's file list is refreshed."""
        w = self.active()
        if not w:
            return None
        try:
            Path(path).resolve().relative_to(Path(w["path"]).resolve())
        except (ValueError, OSError):
            return None
        return self.refresh(w["name"])

    def clean(self, name):
        """Deletes the context files Clover wrote in a workspace (the ones with its marker line)."""
        with self.lock:
            w = self._find(name)
            root = Path(w["path"]).resolve()
        removed = []
        for rel in list(w.get("folders") or []) + ([w["index"]] if w.get("index") else []):
            fp = (root / rel).resolve()
            try:
                fp.relative_to(root)
                if MARK in fp.read_text(encoding="utf-8", errors="replace"):
                    fp.unlink()
                    removed.append(rel)
            except (ValueError, OSError):
                continue
        _git_exclude(root, [])
        with self.lock:
            w = self._find(name)
            w["folders"], w["index"] = [], None
            self._save()
        return dict(self.describe(), removed=removed)

    # ---------------- Context for a question; notes where they belong ----------------
    def note_target(self, text):
        """The context file a note belongs in: the deepest folder of the active workspace that the note names, or the
        index when it is about the project as a whole. None: not about the workspace (it goes to Clover's memory)."""
        w = self.active()
        if not w:
            return None
        low, best = (text or "").lower(), None
        for ctx_rel in w.get("folders") or []:
            folder = ctx_rel.rsplit("/", 1)[0]
            last = folder.rsplit("/", 1)[-1].lower()
            if len(last) >= 3 and re.search(r"(?<![\w-])" + re.escape(last) + r"(?![\w-])", low):
                if best is None or folder.count("/") > best[0].count("/"):
                    best = (folder, ctx_rel)
        if best:
            return best[1]
        if w.get("index") and (PROJECT_WORDS.search(text or "") or w["name"].lower() in low
                               or Path(w["path"]).name.lower() in low):
            return w["index"]
        return None

    def add_note(self, ctx_rel, text):
        w = self.active()
        if not w:
            raise ValueError("no workspace is in use")
        root = Path(w["path"]).resolve()
        fp = (root / ctx_rel).resolve()
        fp.relative_to(root)
        body = fp.read_text(encoding="utf-8", errors="replace")
        ours, notes = _existing(fp)
        if not ours or MARK not in body:
            raise ValueError(f"{ctx_rel} is not a context file Clover keeps; refresh the workspace first")
        head = body.split("\n" + NOTES + "\n", 1)[0].rstrip("\n")
        clean_note = " ".join(str(text).split())[:600]
        entry = f"- {datetime.date.today().isoformat()}: {clean_note}"
        if notes and clean_note.lower() in notes.lower():
            return str(fp)
        fp.write_text(head + "\n\n" + NOTES + "\n" + (notes + "\n" + entry if notes else entry) + "\n", encoding="utf-8")
        return str(fp)

    def record_outcome_note(self, text, touched_paths=()):
        """Stage 5 (Growth): preserve what the cycle taught or what changed in the workspace's context file beside the work."""
        w = self.active()
        if not w or not text:
            return None
        root = Path(w["path"]).resolve()
        target = None
        for p in touched_paths or ():
            try:
                rel = Path(p).resolve().relative_to(root).as_posix()
            except (ValueError, OSError):
                continue
            parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
            if parent:
                for ctx_rel in w.get("folders") or []:
                    folder = ctx_rel.rsplit("/", 1)[0]
                    if parent == folder or parent.startswith(folder + "/"):
                        if target is None or folder.count("/") > target[0].count("/"):
                            target = (folder, ctx_rel)
        ctx_rel = target[1] if target else (self.note_target(text) or w.get("index"))
        if not ctx_rel:
            return None
        try:
            return self.add_note(ctx_rel, text)
        except (ValueError, OSError):
            return None

    def context_block(self, query, budget=1800):
        """(block, info) for a question about the active workspace: up to two folder context files that match it,
        the index's top-level files/folders/notes, and any CONTEXT.md or TODO.md beside the work. None when the
        question does not concern the workspace (a 2B model cites whatever context it is given, so unrelated context
        is left out)."""
        w = self.active()
        if not w or not (w.get("folders") or w.get("index")):
            return None, None
        root, q = Path(w["path"]), _terms(query)
        sep, scored = "\n" + NOTES + "\n", []
        for ctx_rel in w.get("folders") or []:
            try:
                text = (root / ctx_rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            folder = ctx_rel.rsplit("/", 1)[0]
            files = re.findall(r"^- \[`([^`]+)`\]", text, re.M)
            names = _terms(folder.replace("/", " ")) | {folder.rsplit("/", 1)[-1].lower()}
            fterms = _terms(" ".join(files)) | {f.lower() for f in files}
            notes = text.split(sep, 1)[1] if sep in text else ""
            if notes.strip() == NOTES_HINT:
                notes = ""
            s = 3 * len(q & names) + 2 * len(q & fterms) + len(q & _terms(notes))
            if s:
                scored.append((s, ctx_rel, text))
        idx_text, idx_score = "", 0
        if w.get("index"):
            try:
                idx_text = (root / w["index"]).read_text(encoding="utf-8", errors="replace")
                top_files = re.findall(r"^- \[`([^`]+)`\]", idx_text, re.M)
                top_terms = _terms(" ".join(top_files)) | {f.lower() for f in top_files}
                idx_notes = idx_text.split(sep, 1)[1] if sep in idx_text else ""
                if idx_notes.strip() == NOTES_HINT:
                    idx_notes = ""
                idx_score = 2 * len(q & top_terms) + len(q & _terms(idx_notes))
            except OSError:
                idx_text = ""
        scored.sort(key=lambda x: -x[0])
        low = (query or "").lower()
        todo_asked = bool(re.search(r"\b(?:todo|todos|task|tasks|checklist|roadmap|next\s+steps?|context)\b", low))
        asked = (
            bool(PROJECT_WORDS.search(query or ""))
            or w["name"].lower() in low
            or root.name.lower() in low
            or todo_asked
        )
        if not asked and (not scored or scored[0][0] < 2) and idx_score < 2:
            return None, None
        head = (f'The user is working in the workspace "{w["name"]}" ({root}); relative file paths refer to it. From '
                "its context files (Clover's index of folders, files, notes, and TODOs; read a file for details). Files named "
                "*-context.md and *-context-index.md are Clover's own notes, not part of the project:")
        parts, used, room = [head], [], budget - len(head)
        for extra_name in ("CONTEXT.md", "TODO.md"):
            efp = root / extra_name
            if efp.is_file() and (asked or extra_name.lower() in low or (extra_name == "TODO.md" and todo_asked)):
                try:
                    raw_ex = semantics.flag_injection(efp.read_text(encoding="utf-8", errors="replace").strip())
                    if raw_ex and room >= 250:
                        chunk = f"[{extra_name}]\n{raw_ex[:min(500, room - 40)]}"
                        parts.append(chunk)
                        used.append(extra_name)
                        room -= len(chunk)
                except OSError:
                    pass
        for _, ctx_rel, text in scored[:2]:
            clean_t = semantics.flag_injection(_plain(text.replace(NOTES_HINT, "")))
            chunk = f"[{ctx_rel}]\n" + clean_t[:min(900, max(0, room - 40))]
            parts.append(chunk)
            used.append(ctx_rel)
            room -= len(chunk)
            if room < 250:
                break
        if room >= 200 and w.get("index") and idx_text:
            clean_idx = semantics.flag_injection(_plain(idx_text.replace(NOTES_HINT, "")))
            body_idx = clean_idx.split("\n", 2)[-1].strip() if "\n" in clean_idx else clean_idx
            if body_idx:
                parts.append(f"[{w['index']}]\n{body_idx[:room - 40]}")
                used.append(w["index"])
        return "\n\n".join(parts), {"layer": "workspace", "workspace": w["name"], "files": used}

    # ---------------- VS Code-like Explorer Tree, File Viewer & Direct GitHub Client ----------------
    def _resolve_root(self, target=None, local_repo_dir=None):
        """Returns (ws_id, label, root_path, source, repo_slug) for local_repo, an added workspace, or github:owner/repo."""
        t = str(target or "").strip()
        if t.startswith("github:"):
            gh_spec = t.split("github:", 1)[1].strip().strip("/")
            m = GH_URL.match(gh_spec) or GH_SHORT.match(gh_spec)
            if not m:
                raise ValueError(f"invalid GitHub repository {gh_spec!r} (expected owner/repo)")
            slug_repo = f"{m.group(1)}/{m.group(2)}"
            return f"github:{slug_repo}", f"{slug_repo} (GitHub)", None, "github_direct", slug_repo

        if t in ("__local_repo__", "clover-memory-repo", "local"):
            lroot = Path(local_repo_dir) if local_repo_dir else (self.app_dir / "clover-ai-studio")
            return "__local_repo__", f"{lroot.name} (Local Repo)", lroot.resolve(), "local_repo", None

        with self.lock:
            if t:
                found = next((w for w in self.items if w["name"] == t), None)
                if found:
                    return found["name"], found["name"], Path(found["path"]).resolve(), found.get("source") or "folder", found.get("repo")
                m = GH_URL.match(t) or GH_SHORT.match(t)
                if m:
                    slug_repo = f"{m.group(1)}/{m.group(2)}"
                    return f"github:{slug_repo}", f"{slug_repo} (GitHub)", None, "github_direct", slug_repo
                raise ValueError(f"no workspace named {t!r}")
            act = self.active()
            if act:
                return act["name"], act["name"], Path(act["path"]).resolve(), act.get("source") or "folder", act.get("repo")
        lroot = Path(local_repo_dir) if local_repo_dir else (self.app_dir / "clover-ai-studio")
        return "__local_repo__", f"{lroot.name} (Local Repo)", lroot.resolve(), "local_repo", None

    def browse_tree(self, target=None, subpath="", local_repo_dir=None):
        """Lists directory entries (folders first, then files) for VS Code-style Explorer in Workspaces."""
        ws_id, label, root, source, repo_slug = self._resolve_root(target, local_repo_dir=local_repo_dir)
        clean_sub = str(subpath or "").replace("\\", "/").strip("/")
        if clean_sub == ".":
            clean_sub = ""
        if any(part in ("..", "") for part in clean_sub.split("/") if clean_sub):
            raise ValueError("invalid relative path")

        if source == "github_direct":
            tok = self.auth.token if self.auth else None
            api_p = f"/repos/{repo_slug}/contents/{clean_sub}" if clean_sub else f"/repos/{repo_slug}/contents"
            code, data = github_client_request(api_p, token=tok)
            if code != 200 or not isinstance(data, list):
                msg = data.get("message") if isinstance(data, dict) else f"HTTP {code}"
                raise ValueError(f"GitHub could not list {repo_slug}/{clean_sub}: {msg}")
            dirs, files = [], []
            for item in data:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                nm = str(item["name"])
                rel_p = str(item.get("path") or (f"{clean_sub}/{nm}" if clean_sub else nm))
                itype = item.get("type")
                if itype == "dir":
                    dirs.append({"name": nm, "path": rel_p, "type": "dir", "size": 0, "size_human": ""})
                elif itype == "file":
                    sz = int(item.get("size") or 0)
                    ext = Path(nm).suffix.lower()
                    files.append({
                        "name": nm,
                        "path": rel_p,
                        "type": "file",
                        "size": sz,
                        "size_human": _size(sz),
                        "ext": ext,
                        "kind": KINDS.get(ext, "file"),
                        "secret": bool(SECRETISH.search(nm)),
                    })
            dirs.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["name"].lower())
            return {
                "workspace": ws_id,
                "label": label,
                "source": source,
                "repo": repo_slug,
                "path": clean_sub,
                "entries": dirs + files,
            }

        if not root or not root.is_dir():
            raise ValueError(f"workspace folder {root} does not exist")
        cur = (root / clean_sub).resolve() if clean_sub else root
        try:
            cur.relative_to(root)
        except ValueError:
            raise ValueError("path is outside the workspace") from None
        if not cur.is_dir():
            raise ValueError(f"{clean_sub or '.'} is not a folder")

        dirs, files = [], []
        try:
            children = list(cur.iterdir())
        except OSError as exc:
            raise ValueError(f"could not read folder: {exc}") from None

        for child in children:
            nm = child.name
            if nm in (".git", "vault") or nm.endswith(".tmp") or BACKUP.search(nm):
                continue
            rel_p = child.relative_to(root).as_posix()
            try:
                is_d = child.is_dir() and not child.is_symlink()
            except OSError:
                continue
            if is_d:
                if nm.lower() in SKIP_DIRS:
                    continue
                dirs.append({"name": nm, "path": rel_p, "type": "dir", "size": 0, "size_human": ""})
            elif child.is_file():
                try:
                    sz = child.stat().st_size
                except OSError:
                    sz = 0
                ext = child.suffix.lower()
                files.append({
                    "name": nm,
                    "path": rel_p,
                    "type": "file",
                    "size": sz,
                    "size_human": _size(sz),
                    "ext": ext,
                    "kind": KINDS.get(ext, "file"),
                    "secret": bool(SECRETISH.search(nm)),
                })
        dirs.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())
        return {
            "workspace": ws_id,
            "label": label,
            "source": source,
            "repo": repo_slug,
            "root_path": str(root),
            "path": clean_sub,
            "entries": dirs + files,
        }

    def read_workspace_file(self, target=None, filepath="", local_repo_dir=None, max_bytes=200_000):
        """Reads a file for the VS Code-style File Viewer in Workspaces (from local_repo, workspace, or GitHub client)."""
        ws_id, label, root, source, repo_slug = self._resolve_root(target, local_repo_dir=local_repo_dir)
        clean_p = str(filepath or "").replace("\\", "/").strip("/")
        if not clean_p or any(part in ("..", "") for part in clean_p.split("/")):
            raise ValueError("a valid relative file path is required")
        nm = clean_p.rsplit("/", 1)[-1]
        ext = Path(nm).suffix.lower()
        kind = KINDS.get(ext, "file")
        if SECRETISH.search(nm) or "vault" in [p.lower() for p in clean_p.split("/")]:
            return {
                "workspace": ws_id,
                "label": label,
                "source": source,
                "repo": repo_slug,
                "path": clean_p,
                "name": nm,
                "size": 0,
                "size_human": "—",
                "ext": ext,
                "kind": kind,
                "lines": 0,
                "binary": False,
                "secret_blocked": True,
                "content": "",
                "message": "Protected file: Clover does not display files whose names suggest secrets, keys, or credentials.",
            }

        if source == "github_direct":
            tok = self.auth.token if self.auth else None
            code, data = github_client_request(f"/repos/{repo_slug}/contents/{clean_p}", token=tok)
            if code != 200 or not isinstance(data, dict):
                msg = data.get("message") if isinstance(data, dict) else f"HTTP {code}"
                raise ValueError(f"GitHub could not read {repo_slug}/{clean_p}: {msg}")
            sz = int(data.get("size") or 0)
            b64_content = data.get("content") or ""
            raw = base64.b64decode(b64_content) if b64_content else b""
            if b"\x00" in raw[:4096]:
                return {
                    "workspace": ws_id,
                    "label": label,
                    "source": source,
                    "repo": repo_slug,
                    "path": clean_p,
                    "name": nm,
                    "size": sz,
                    "size_human": _size(sz),
                    "ext": ext,
                    "kind": kind or "binary",
                    "lines": 0,
                    "binary": True,
                    "secret_blocked": False,
                    "content": "",
                    "html_url": data.get("html_url") or f"https://github.com/{repo_slug}/blob/HEAD/{clean_p}",
                }
            truncated = len(raw) > max_bytes
            text = raw[:max_bytes].decode("utf-8", errors="replace")
            return {
                "workspace": ws_id,
                "label": label,
                "source": source,
                "repo": repo_slug,
                "path": clean_p,
                "name": nm,
                "size": sz,
                "size_human": _size(sz),
                "ext": ext,
                "kind": kind,
                "lines": len(text.splitlines()) or (1 if text else 0),
                "binary": False,
                "truncated": truncated,
                "secret_blocked": False,
                "content": text,
                "html_url": data.get("html_url") or f"https://github.com/{repo_slug}/blob/HEAD/{clean_p}",
            }

        if not root or not root.is_dir():
            raise ValueError(f"workspace folder {root} does not exist")
        fp = (root / clean_p).resolve()
        try:
            fp.relative_to(root)
        except ValueError:
            raise ValueError("file path is outside the workspace") from None
        if not fp.is_file():
            raise ValueError(f"{clean_p} is not a file")
        try:
            sz = fp.stat().st_size
            with fp.open("rb") as f:
                raw = f.read(max_bytes + 1)
        except OSError as exc:
            raise ValueError(f"could not read {clean_p}: {exc}") from None
        if b"\x00" in raw[:4096]:
            return {
                "workspace": ws_id,
                "label": label,
                "source": source,
                "repo": repo_slug,
                "path": clean_p,
                "full_path": str(fp),
                "name": nm,
                "size": sz,
                "size_human": _size(sz),
                "ext": ext,
                "kind": kind or "binary",
                "lines": 0,
                "binary": True,
                "secret_blocked": False,
                "content": "",
            }
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        return {
            "workspace": ws_id,
            "label": label,
            "source": source,
            "repo": repo_slug,
            "path": clean_p,
            "full_path": str(fp),
            "name": nm,
            "size": sz,
            "size_human": _size(sz),
            "ext": ext,
            "kind": kind,
            "lines": len(text.splitlines()) or (1 if text else 0),
            "binary": False,
            "truncated": truncated,
            "secret_blocked": False,
            "content": text,
        }


def github_client_request(api_path: str, token: str | None = None, method: str = "GET", payload: dict | None = None):
    """Direct GitHub REST client for browsing repositories, trees, and files with or without a saved token."""
    import urllib.error
    import urllib.request
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "CloverAIStudio/5.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"https://api.github.com{api_path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            doc = json.loads(raw) if raw.strip() else {}
        except ValueError:
            doc = {"message": raw[:200]}
        return exc.code, doc
    except OSError as exc:
        return 503, {"message": f"GitHub unreachable: {exc}"}


def github_client_status(token: str | None = None, sync_repo_info: dict | None = None, local_repo_dir: Path | None = None):
    """Returns GitHub client connection status and accessible repositories, or falls back to the local repository."""
    lpath = str(local_repo_dir) if local_repo_dir else ""
    sync_r = (sync_repo_info or {}).get("repo") or ""
    if not token:
        return {
            "connected": False,
            "login": "",
            "name": "",
            "sync_repo": sync_r,
            "local_repo": lpath,
            "repos": [],
            "message": "Connected to local repository (clover-ai-studio). Add your GitHub token below to connect directly to your GitHub repositories.",
        }
    code, user_doc = github_client_request("/user", token=token)
    if code != 200 or not isinstance(user_doc, dict) or not user_doc.get("login"):
        return {
            "connected": False,
            "token_invalid": code in (401, 403),
            "login": "",
            "name": "",
            "sync_repo": sync_r,
            "local_repo": lpath,
            "repos": [],
            "message": f"GitHub token could not sign in ({user_doc.get('message') if isinstance(user_doc, dict) else code}). Using local repository.",
        }
    _, repos_doc = github_client_request("/user/repos?sort=updated&per_page=30", token=token)
    repos = []
    if isinstance(repos_doc, list):
        for r in repos_doc:
            if isinstance(r, dict) and r.get("full_name"):
                repos.append({
                    "full_name": r["full_name"],
                    "name": r.get("name") or r["full_name"].split("/")[-1],
                    "private": bool(r.get("private")),
                    "description": r.get("description") or "",
                    "default_branch": r.get("default_branch") or "main",
                    "updated_at": (r.get("updated_at") or "")[:10],
                })
    return {
        "connected": True,
        "login": user_doc.get("login") or "",
        "name": user_doc.get("name") or user_doc.get("login") or "",
        "sync_repo": sync_r,
        "local_repo": lpath,
        "repos": repos,
        "message": f"Connected to GitHub as @{user_doc.get('login')} ({len(repos)} repositories).",
    }


def local_context_for_paths(paths_list, workdir=None, active_ws_path=None, budget=1100):
    """Stage 1 (Context): read any context file (`CONTEXT.md`, `*-context.md`, `*-context-index.md`, `TODO.md`) beside
    the files/folders named in the turn (`paths_list`) or `workdir` when they lie outside the active workspace.
    Enforces WEB host isolation and vault read blocking, and flags prompt-injection strings via `semantics.flag_injection`."""
    import paths as _paths
    if getattr(_paths, "WEB", False):
        return None, None
    ws_root = None
    if active_ws_path:
        try:
            ws_root = Path(active_ws_path).resolve()
        except OSError:
            ws_root = None
    cand_dirs = []
    for raw_p in list(paths_list or []) + ([workdir] if workdir else []):
        if not raw_p:
            continue
        try:
            p = Path(str(raw_p)).resolve()
        except OSError:
            continue
        d = p if p.is_dir() else p.parent
        if not d.is_dir():
            continue
        if any(part.lower() == "vault" for part in d.parts):
            continue
        if ws_root is not None:
            try:
                d.relative_to(ws_root)
                continue
            except ValueError:
                pass
        if d not in cand_dirs:
            cand_dirs.append(d)
    if not cand_dirs:
        return None, None
    parts, used, room = [], [], budget
    for d in cand_dirs[:2]:
        names = [
            "CONTEXT.md",
            f"{slug(d.name)}-context.md",
            f"{slug(d.name)}-context-index.md",
            "TODO.md",
        ]
        for nm in names:
            fp = d / nm
            if not fp.is_file():
                continue
            try:
                raw_txt = fp.read_text(encoding="utf-8", errors="replace").replace(NOTES_HINT, "")
                txt = semantics.flag_injection(_plain(raw_txt)).strip()
            except OSError:
                continue
            if not txt or room < 180:
                continue
            rel_label = f"{d.name}/{nm}"
            chunk = f"[{rel_label}]\n{txt[:min(550, room - 30)]}"
            parts.append(chunk)
            used.append(str(fp))
            room -= len(chunk)
    if not parts:
        return None, None
    head = "Context files found beside the target folder/files:"
    return head + "\n\n" + "\n\n".join(parts), {"layer": "local_context", "files": used}

