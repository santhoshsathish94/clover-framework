"""Collective Problem-Solving Learning — Git-Backed Core Knowledge Registry (`core_registry.py`).

Manages the versioned `clover-collective/` Git repository:
  clover-collective/
  ├── schema/
  │   ├── learning-event.schema.json
  │   └── core-knowledge.schema.json
  ├── candidates/
  │   └── reviewed-learning-packages/
  ├── core/
  │   ├── semantics/
  │   ├── reasoning-patterns/
  │   ├── problem-solving/
  │   └── tool-use-patterns/
  ├── evaluations/
  │   ├── regression/
  │   ├── generalisation/
  │   └── cross-domain/
  ├── releases/
  │   └── manifest.json
  └── CHANGELOG.md
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

from collective_gateway import LEARNING_EVENT_SCHEMA

CORE_KNOWLEDGE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CloverCoreKnowledgeRule",
    "type": "object",
    "required": [
        "rule_id",
        "version",
        "category",
        "trigger",
        "target",
        "independent_installations",
        "heldout_accuracy",
        "heldout_lift",
        "regressions",
    ],
    "optional": ["strategy"],
    "additionalProperties": False,
}


def _writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


class CoreRegistry:
    """Manages approved collective knowledge in a Git repository and publishes HMAC-SHA256 signed releases."""

    def __init__(self, repo_dir: str | Path, signing_key: str = "clover-collective-core-signing-key-v1"):
        self.root = Path(repo_dir)
        self.signing_key = signing_key.encode("utf-8")
        self.git = shutil.which("git")
        self.version_num = 0
        self._init_repo()

    def _git(self, *args):
        if not self.git:
            return b""
        r = subprocess.run([self.git, "-C", str(self.root), "-c", "core.hooksPath=", *args], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {args[0]}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def _write_json(self, rel: str, obj):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    def _init_repo(self):
        for sub in [
            "schema",
            "candidates/reviewed-learning-packages",
            "core/semantics",
            "core/reasoning-patterns",
            "core/problem-solving",
            "core/tool-use-patterns",
            "evaluations/regression",
            "evaluations/generalisation",
            "evaluations/cross-domain",
            "releases",
        ]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        if self.git and not (self.root / ".git").exists():
            self._git("init", "-q")
            for k, v in (("user.name", "CloverCoreRegistry"), ("user.email", "registry@cloverframework.local"),
                         ("commit.gpgsign", "false")):
                self._git("config", k, v)

        if not (self.root / "schema" / "learning-event.schema.json").exists():
            self._write_json("schema/learning-event.schema.json", LEARNING_EVENT_SCHEMA)
        if not (self.root / "schema" / "core-knowledge.schema.json").exists():
            self._write_json("schema/core-knowledge.schema.json", CORE_KNOWLEDGE_SCHEMA)
        if not (self.root / "CHANGELOG.md").exists():
            (self.root / "CHANGELOG.md").write_text("# Clover Collective Core — Changelog\n\n", encoding="utf-8")
        self._commit("Initialize clover-collective repository structure and schemas")

    def _commit(self, msg: str):
        if not self.git:
            return
        self._git("add", "-A")
        if self._git("status", "--porcelain").strip():
            self._git("commit", "-q", "-m", msg)

    def publish_release(self, eval_summary: dict, note: str = "Automated evaluation & promotion") -> dict:
        self.version_num += 1
        version = f"1.0.{self.version_num}"

        for cat in ("semantics", "reasoning-patterns", "problem-solving", "tool-use-patterns"):
            cat_dir = self.root / "core" / cat
            for old_f in cat_dir.glob("*.json"):
                old_f.unlink()

        self._write_json(
            f"candidates/reviewed-learning-packages/review-v{version}.json",
            {
                "version": version,
                "total_evaluated": eval_summary["total_evaluated"],
                "promoted_count": eval_summary["promoted_count"],
                "rejected_count": eval_summary["rejected_count"],
                "candidates": eval_summary["all_reports"],
            },
        )
        self._write_json(
            f"evaluations/regression/regression-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "regressions": r["metrics"]["regressions"],
                 "personal_hijacks": r["metrics"]["personal_hijacks"], "pass": r["gates"]["G5_zero_regression"]}
                for r in eval_summary["all_reports"]
            ],
        )
        self._write_json(
            f"evaluations/generalisation/generalisation-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "heldout_accuracy": r["metrics"]["heldout_accuracy"],
                 "heldout_lift": r["metrics"]["heldout_lift"], "pass": r["gates"]["G3_generalisation"]}
                for r in eval_summary["all_reports"]
            ],
        )
        self._write_json(
            f"evaluations/cross-domain/cross-domain-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "matched_domains": r["metrics"]["matched_domains"],
                 "pass": r["gates"]["G4_cross_domain"]}
                for r in eval_summary["all_reports"]
            ],
        )

        promoted_rules = []
        file_hashes = {}
        for r in eval_summary["promoted"]:
            rule_id = f"core_{r['candidate_id'][4:]}"
            rule_obj = {
                "rule_id": rule_id,
                "version": version,
                "category": r["category"],
                "trigger": r["trigger"],
                "target": r["target"],
                "strategy": r.get("strategy", ""),
                "independent_installations": r["independent_installations"],
                "heldout_accuracy": r["metrics"]["heldout_accuracy"],
                "heldout_lift": r["metrics"]["heldout_lift"],
                "regressions": r["metrics"]["regressions"],
            }
            rel_path = f"core/{r['category']}/{rule_id}.json"
            self._write_json(rel_path, rule_obj)
            raw_bytes = (self.root / rel_path).read_bytes()
            file_hashes[rel_path] = hashlib.sha256(raw_bytes).hexdigest()
            promoted_rules.append(rule_obj)

        payload_to_sign = json.dumps(
            {"version": version, "rules": promoted_rules, "file_hashes": file_hashes},
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.new(self.signing_key, payload_to_sign, hashlib.sha256).hexdigest()

        manifest = {
            "version": version,
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": note,
            "rules_count": len(promoted_rules),
            "rules": promoted_rules,
            "file_hashes": file_hashes,
            "signature_alg": "HMAC-SHA256",
            "signature": signature,
        }
        self._write_json("releases/manifest.json", manifest)

        changelog = self.root / "CHANGELOG.md"
        prev_log = changelog.read_text(encoding="utf-8")
        entry = (
            f"## v{version} ({manifest['published_utc']})\n"
            f"- Note: {note}\n"
            f"- Evaluated candidates: {eval_summary['total_evaluated']} "
            f"(promoted: {eval_summary['promoted_count']}, rejected: {eval_summary['rejected_count']})\n\n"
        )
        changelog.write_text(prev_log + entry, encoding="utf-8")
        self._commit(f"Release v{version}: {len(promoted_rules)} validated collective rules ({note})")
        return manifest

    def find_strings_anywhere_in_repo(self, strings: list[str]) -> list[str]:
        blob = (
            subprocess.run(
                [self.git, "-C", str(self.root), "cat-file", "--batch-all-objects", "--batch"],
                capture_output=True,
            ).stdout
            if self.git
            else b""
        )
        for p in self.root.rglob("*"):
            if p.is_file() and p.relative_to(self.root).parts[:2] != (".git", "objects"):
                blob += p.read_bytes()
        return [s for s in strings if s.encode("utf-8") in blob]
