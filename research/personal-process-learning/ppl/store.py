"""The personal experience store: one git repository per person.

experiences/<session>/<n>.json   one file per interaction (what was said, understood, revealed)
retention.json                   experience id -> "stop" (kept as history, excluded from learning)
tombstones.json                  salted hashes of deleted experiences (so a restored copy is not re-learned)
copies.json                      clones and backups made of this store (they need their own deletion)
knowledge/knowledge.json         derived knowledge with provenance; recomputed, never edited by hand

Deleting removes the experience from the working tree, from every commit (git filter-repo), from the reflog and from
the object store (gc --prune=now). Derived knowledge is recomputed from the remaining eligible experiences, and the
history of knowledge/ is rewritten too, so no earlier derived state survives. The salt never enters history."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class PersonalStore:
    def __init__(self, root, person, filter_repo):
        self.dir = Path(root) / person
        self.filter_repo = list(filter_repo)
        fresh = not (self.dir / ".git").exists()
        self.dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            self._git("init", "-q")
            self._git("config", "user.name", "ppl")
            self._git("config", "user.email", "ppl@localhost")
            self._git("config", "commit.gpgsign", "false")
            for name, value in (("retention.json", {}), ("tombstones.json", []), ("copies.json", [])):
                self._write(name, value)
            self._commit("init", ["retention.json", "tombstones.json", "copies.json"])
        salt = self.dir / ".git" / "ppl-salt"
        if not salt.exists():
            salt.write_text(os.urandom(16).hex())
        self.salt = salt.read_text().strip()

    # ---- plumbing ----
    def _git(self, *args, input=None):
        r = subprocess.run(["git", "-C", str(self.dir), *args], capture_output=True, input=input)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace')}")
        return r.stdout

    def _write(self, rel, value):
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    def _read(self, rel, default=None):
        path = self.dir / rel
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    def _commit(self, message, paths):
        self._git("add", "-A", "--", *paths)
        if self._git("status", "--porcelain", "--", *paths).strip():
            self._git("commit", "-q", "-m", message)

    @staticmethod
    def path_of(exp_id):
        session, n = exp_id.split("-")
        return f"experiences/{session}/{n}.json"

    # ---- experiences ----
    def add(self, exp):
        rel = self.path_of(exp["id"])
        self._write(rel, exp)
        self._commit(f"experience {exp['id']}", [rel])

    def all_experiences(self):
        out = [json.loads(p.read_text(encoding="utf-8")) for p in (self.dir / "experiences").glob("*/*.json")]
        return sorted(out, key=lambda e: e["order"])

    def eligible(self):
        stop = self._read("retention.json", {})
        return [e for e in self.all_experiences() if stop.get(e["id"]) != "stop"]

    def retention(self):
        return self._read("retention.json", {})

    def keep(self, ids):
        r = self._read("retention.json", {})
        for i in ids:
            r.pop(i, None)
        self._write("retention.json", r)
        self._commit(f"keep {len(ids)}", ["retention.json"])

    def stop_learning(self, ids):
        r = self._read("retention.json", {})
        for i in ids:
            r[i] = "stop"
        self._write("retention.json", r)
        self._commit(f"stop learning from {len(ids)}", ["retention.json"])

    def tombstone(self, exp):
        return hashlib.sha256((self.salt + exp["text"]).encode()).hexdigest()

    def delete(self, ids):
        """Remove experiences everywhere in this repository. Returns the strings that must no longer be found."""
        ids = sorted(set(ids))
        every = self.all_experiences()
        exps = {e["id"]: e for e in every if e["id"] in ids}
        remaining = {s for e in every if e["id"] not in exps for s in [e["text"], *e.get("reactions", [])]}
        secrets = sorted({s for e in exps.values() for s in [e["text"], *e.get("reactions", [])]
                          if len(s) >= 12 and not any(s in r for r in remaining)})
        stones = self._read("tombstones.json", [])
        stones += [self.tombstone(e) for e in exps.values()]
        self._write("tombstones.json", sorted(set(stones)))
        r = self._read("retention.json", {})
        for i in ids:
            r.pop(i, None)
        self._write("retention.json", r)
        paths = [self.path_of(i) for i in ids]
        for p in paths:
            (self.dir / p).unlink(missing_ok=True)
        self._commit(f"delete {len(ids)} experiences", ["tombstones.json", "retention.json", *paths])
        args = ["--force", "--quiet", "--invert-paths", "--path", "knowledge/"]
        for p in paths:
            args += ["--path", p]
        r = subprocess.run([*self.filter_repo, *args], cwd=str(self.dir), capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"filter-repo: {r.stderr.decode(errors='replace')}")
        self._git("reflog", "expire", "--expire=now", "--all")
        self._git("gc", "--prune=now", "--quiet")
        return secrets

    def is_tombstoned(self, exp):
        return self.tombstone(exp) in set(self._read("tombstones.json", []))

    # ---- derived knowledge ----
    def save_knowledge(self, knowledge, message="knowledge"):
        self._write("knowledge/knowledge.json", knowledge)
        self._commit(message, ["knowledge/knowledge.json"])

    # ---- verification and copies ----
    def find(self, strings, where=None):
        """Strings found in any object of the repository (reachable or not), in .git files outside the object store,
        or in the working tree."""
        root = Path(where) if where else self.dir
        blob = subprocess.run(["git", "-C", str(root), "cat-file", "--batch-all-objects", "--batch"],
                              capture_output=True).stdout
        for p in root.rglob("*"):
            if p.is_file() and "objects" not in p.relative_to(root).parts:
                blob += p.read_bytes()
        return [s for s in strings if s.encode() in blob]

    def make_copy(self, dest, kind="clone"):
        subprocess.run(["git", "clone", "-q", "--no-local", str(self.dir), str(dest)], check=True, capture_output=True)
        copies = self._read("copies.json", [])
        copies.append({"path": str(dest), "kind": kind, "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "head": self._git("rev-parse", "HEAD").decode().strip()})
        self._write("copies.json", copies)
        self._commit(f"register {kind}", ["copies.json"])

    def copies(self):
        return self._read("copies.json", [])

    def remove(self):
        shutil.rmtree(self.dir, ignore_errors=True)
