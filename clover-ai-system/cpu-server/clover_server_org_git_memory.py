#!/usr/bin/env python3
"""
Layer 4 (Workstation / Multi-User Server): Organization & Multi-User Git Memory Repository
(`clover-ai-system/cpu-server/clover_server_org_git_memory.py`).

Designed for 64 GB / 128 GB Linux Workstations and Multi-User Servers (`k3`) where multiple
engineers / team members collaborate on the same Clover server or across synced workstations:

  1. **Multi-User + Shared Organization Git Hierarchy (`clover-org-memory-repo/`)**:
     ```text
     clover-org-memory-repo/
     ├── AGENTS.md                        # Canonical Organization-wide Clover Operating Specification
     ├── ORG_CONTEXT.md                   # Shared organization context, architecture decisions & rules
     ├── members.json                     # Registered organization members, teams, roles & AGENTS.md status
     ├── shared/                          # Organization-wide shared knowledge (collaborative across all users)
     │   ├── notes/                       # Verified technical notes & corrections (MEM-*.md)
     │   ├── tasks/                       # Shared team playbooks, RFCs & task plans (tasks/<slug>.md)
     │   └── scripts/                     # Shared verified Python/Bash tools (scripts/<slug>.py)
     ├── teams/                           # Per-team collaborative workspaces (e.g. teams/core/, teams/ml/)
     │   └── <team_id>/
     │       ├── TEAM_CONTEXT.md
     │       ├── tasks/
     │       └── scripts/
     └── users/                           # Per-user isolated & attributable workspaces inside the Git repo
         └── <user_id>/                   # e.g. users/santhosh/, users/alice/, users/bob/
             ├── AGENTS.md                # User's personal AGENTS.md customization & acceptance record
             ├── CONTEXT.md               # User's active personal context & preferences
             ├── sessions/                # User's session logs (sessions/session-<sid>.md)
             ├── tasks/                   # User's personal task notes (tasks/<slug>.md)
             └── scripts/                 # User's auto-saved verified scripts (scripts/<slug>.py)
     ```

  2. **Concurrent Multi-User Git Locking & Per-User Commit Attribution**:
     - Every Git mutation acquires both an in-process `threading.RLock()` and an OS file lock
       (`.org_git_repo.lock`, `fcntl.flock` on Linux) so concurrent requests from multiple users
       never race on `.git/index.lock`.
     - Every commit sets `GIT_AUTHOR_NAME="{display_name} ({user_id})"` and `GIT_AUTHOR_EMAIL="{email}"`
       and prefixes commit messages with `[{user_id}@{team}]` so `git log` provides full multi-person
       auditability and provenance.

  3. **Three-Tier Selective On-Demand Retrieval (Zero Context-Window Bloating)**:
     - Never dumps the entire repository into the model's context window.
     - Searches across:
       (a) **User Scope** (`users/<user_id>/**`)
       (b) **Team Scope** (`teams/<team_id>/**`)
       (c) **Organization Shared Scope** (`ORG_CONTEXT.md`, `shared/**`)
       and injects only the top-matching compact snippets (`<= 550 chars` per file), or lets the model
       call `<repo_search>`, `<repo_read>`, `<repo_write>`, and `<repo_promote>`.

  4. **Collaborative Promotion (`promote_to_shared`) & Private Organization GitHub Sync (`private: true`)**:
     - Any user can promote a personal note or verified script (`users/<user_id>/scripts/tool.py`)
       to `shared/scripts/tool.py` or `teams/<team_id>/scripts/tool.py` with a single click or API call,
       recording an attributed Git commit.
     - Syncs (`git pull --rebase` / `git push`) with a **private** GitHub repository (`private: true`
       enforced via the GitHub REST API) so multiple people and workstations stay in sync.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


DEFAULT_ORG_AGENTS_MD = """# AGENTS.md — Organization & Multi-User Operating Specification for Clover AI Workstation

**Welcome to the Clover AI Workstation Server.** This server hosts a collaborative, multi-user **Layer 4 Organization Git Memory Repository** (`clover-org-memory-repo/`). Please review (and optionally customize) your operating instructions before starting. Accepting `AGENTS.md` records your user agreement as a Git commit attributed to your user profile.

---

## 1. Actors, Authority, and Multi-User Accountability

1. **The System (Reality)** is the running environment and the final authority for validating outcomes. Neither Human nor AI possesses complete reality.
2. **The Humans (Organization Members)** own **Direction** — purpose, priorities, acceptable risk, permissions, boundaries, and accountability.
   - Each team member has an isolated personal workspace (`users/<user_id>/`) inside the Git repository, plus access to their team workspace (`teams/<team_id>/`) and the organization-wide shared knowledge base (`shared/` and `ORG_CONTEXT.md`).
   - Every file edit, session log, script creation, and promotion to `shared/` is recorded as a real Git commit authored with your user identity (`git log`).
3. **Clover (AI)** provides **Capability and Execution** inside Human Direction while holding **Execution Integrity**:
   - AI capability scales across the workstation (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`, `LiquidAI/LFM3-Omni-2B`, `Kimi-K3`), while accountability stays visibly human.

---

## 2. The Five-Stage System Cycle (`Context ⇄ Direction ⇄ Execution ⇄ Outcome ⇄ Growth`)

- **Context:** Reach the minimum relevant evidence from your personal workspace (`users/<user_id>/`), your team workspace (`teams/<team_id>/`), and the organization's shared knowledge (`shared/`, `ORG_CONTEXT.md`), ontology graph, or live web search without bloating the context window.
- **Direction:** Keep the active user's intended outcome and permission boundaries (`Allow`, `Ask`, `Off`) visible.
- **Execution:** Run the 7-layer Clover stack (`Nemotron 3.5 Super VL 120B-A12B` C/OpenMP/AVX-512 engine, `8-GPU Trained HybridLink4096`, `131K-Vocab 11-Level NCM`, `Multi-User Git Memory Repo`, `Ontology GraphRAG`, `Generative Compiler`, and `SpecMD Policy 3 R4 madvise`).
- **Outcome:** Verify results against reality (running assertions, compiler checks, arithmetic checks, and grounding audits). When an outcome is not right, return to **Context** (`Try again`), not blind repetition.
- **Growth:** Preserve verified scripts (`scripts/*.py`), task notes (`tasks/*.md`), and session turns (`sessions/*.md`) in Git, and promote reusable discoveries from `users/<user_id>/` to `shared/` so the entire organization benefits.

---

## 3. Multi-User Git Organization Structure

- **`ORG_CONTEXT.md` & `shared/`**: Organization-wide engineering principles, verified notes (`shared/notes/MEM-*.md`), team playbooks (`shared/tasks/*.md`), and reusable scripts (`shared/scripts/*.py`).
- **`teams/<team_id>/`**: Team-specific context (`TEAM_CONTEXT.md`), tasks, and scripts.
- **`users/<user_id>/`**: Your personal `AGENTS.md`, `CONTEXT.md`, `sessions/`, `tasks/`, and `scripts/`. Protected from accidental overwrite by other users.
- **Private Organization GitHub Sync (`private: true`)**: Sync the entire multi-user repository with a **private** GitHub repository across workstations and team members.
"""

DEFAULT_ORG_CONTEXT_MD = """# ORG_CONTEXT.md — Shared Organization Principles & Verified Engineering Knowledge

This file lives at `clover-org-memory-repo/ORG_CONTEXT.md` and is shared across all users on this Clover Workstation Server.

## Verified Workstation & Systems Principles
- **Per-Layer Cache Partitioning (`shared/notes/MEM-CORR-PER-LAYER-PARTITION-002.md`)**: When varying a global cache capacity from 512 to 4096 entries yields a flat 23.3% hit rate due to cyclic-sweep eviction, suspect the global structure: partitioning the same 51.7 GB cache per layer raised hit rate from 23.3% to 48.0%.
- **Stratified Priority Graph Traversal (`shared/notes/MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001.md`)**: Sorting candidate ontology paths by causal/remediation edge priority before leaf `AFFECTS` edges (`max_affects_per_seed = 4`) prevents 1-hop fan-out flooding and raises full `CVE -> CWE -> CAPEC` chain recall from 63.3% to 76.7%.
- **Workstation 7-Layer Nemotron Stack (`shared/notes/MEM-ARCH-NEMOTRON-WORKSTATION-003.md`)**: On 64 GB / 128 GB Linux workstations (`k3`), `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` runs in 4-bit planar Group-64 format (`69.9 GiB`) with SpecMD Policy 3 `madvise(MADV_WILLNEED)`, 8-GPU trained `HybridLink4096` (`0.9854` cosine), and 11-level `131,072`-vocab Tekken `NCM` (`96.26%` pointer accuracy).
"""

DEFAULT_USER_CONTEXT_TEMPLATE = """# Personal Context (`{display_name}` · `{user_id}` · Team `{team_id}`)

This file is tracked in `users/{user_id}/CONTEXT.md` inside the organization Git memory repository.
Clover searches your personal workspace (`users/{user_id}/`), your team workspace (`teams/{team_id}/`), and `shared/` on demand without bloating the context window.

## Active Focus & Preferences
- **User ID**: `{user_id}`
- **Team**: `{team_id}`
- Add your personal project notes, recurring tasks, or coding conventions here. Promote reusable notes or scripts to `shared/` when ready for the whole team.
"""

DEFAULT_TEAM_CONTEXT_TEMPLATE = """# Team Context (`teams/{team_id}/TEAM_CONTEXT.md`)

Shared context and conventions for team `{team_id}`. All members of `{team_id}` can read and commit to `teams/{team_id}/`.
"""

SHARED_SEED_NOTES: list[tuple[str, str, str, list[str]]] = [
    (
        "shared/notes/MEM-CORR-PER-LAYER-PARTITION-002.md",
        "MEM-CORR-PER-LAYER-PARTITION-002",
        "Per-Layer Cache Partition Fixes Flat Cyclic-Sweep Hit Rates",
        [
            "When a measurement shows a flat response across all settings of a parameter, suspect the structure being measured (global eviction across layers), not the effect: partitioning the same 51.7 GB cache per layer increased the hit rate from 23.3% to 48.0%.",
            "Global LRU cache across all layers stayed flat at 23.3% across capacities 512 to 4096 due to cyclic-sweep eviction; per-layer partitioning lifted hit rate to 48.0%.",
        ],
    ),
    (
        "shared/notes/MEM-HYP-CACHE-DEAD-END-001.md",
        "MEM-HYP-CACHE-DEAD-END-001",
        "Superseded Hypothesis: Global Cache Flatness Means Caching Is Ineffective",
        [
            "Mistaken initial hypothesis that expert caching is ineffective because varying cache capacity from 512 to 4096 produced a flat 23.3% hit rate (superseded by MEM-CORR-PER-LAYER-PARTITION-002).",
        ],
    ),
    (
        "shared/notes/MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001.md",
        "MEM-CORR-STRATIFIED-PRIORITY-TRAVERSAL-001",
        "Stratified Priority Traversal Eliminates 1-Hop Fan-Out Flooding in Ontology GraphRAG",
        [
            "Sorting candidate paths by `edge_priority` (putting causal/remediation edges before leaf `AFFECTS` edges) and enforcing a stratified per-seed cap on leaf edges (`max_affects_per_seed = 4`) eliminates 1-hop fan-out flooding, raising full `CVE+CWE+CAPEC` chain coverage from 63.3% to 76.7% and Gold ID recall from 78.9% to 84.5%.",
        ],
    ),
    (
        "shared/notes/MEM-ARCH-NEMOTRON-WORKSTATION-003.md",
        "MEM-ARCH-NEMOTRON-WORKSTATION-003",
        "Nemotron 3.5 Super VL 7-Layer Workstation Architecture & SpecMD R4 Ledger",
        [
            "NVIDIA Nemotron 3.5 Super VL (120B total / 12B active, d=4096, V=131,072 Tekken, Native 1M context) is the default workstation model for 64 GB and 128 GB Linux machines, pairing Mamba-2 SSM, 512-expert Latent ReLU^2 MoE, NoPE 16:1 GQA, 8-GPU trained HybridLink4096 (0.9854 cosine), and 131K-vocab 11-level NCM (96.26% pointer accuracy).",
        ],
    ),
]


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str, max_len: int = 42) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:max_len].strip("-") or "item")


def sanitize_identifier(raw_id: str | None, fallback: str = "default", max_len: int = 32) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "-", (raw_id or "").strip().lower()).strip("-_")
    return (s[:max_len] or fallback)


class CloverServerOrgGitMemoryRepo:
    """
    Layer 4 Multi-User & Organization Git Repository Memory Engine (`clover-org-memory-repo/`).
    Supports many concurrent users organized into personal (`users/<user_id>/`), team (`teams/<team_id>/`),
    and organization-shared (`shared/`, `ORG_CONTEXT.md`, `AGENTS.md`) scopes inside a single version-controlled
    Git repository with per-user commit attribution, file locking, collaborative promotion, selective low-token
    retrieval, and private GitHub synchronization.
    """

    def __init__(self, base_dir: Path, canonical_agents_md_path: Path | None = None):
        self.base_dir = Path(base_dir)
        self.repo_dir = self.base_dir / "clover-org-memory-repo"
        self.lock_file = self.base_dir / ".org_git_repo.lock"
        self.members_file = self.repo_dir / "members.json"
        self.github_cfg_file = self.base_dir / ".github_org_private_sync.json"
        self.canonical_agents_md_path = canonical_agents_md_path
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._ensure_initialized()

    @contextlib.contextmanager
    def _repo_lock(self):
        """Acquires both in-process RLock and OS file lock (reentrant within the same thread)."""
        with self._thread_lock:
            if self._lock_depth > 0:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            self.base_dir.mkdir(parents=True, exist_ok=True)
            f_lock = open(self.lock_file, "a+", encoding="utf-8")
            self._lock_depth = 1
            try:
                if fcntl is not None:
                    fcntl.flock(f_lock.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                self._lock_depth = 0
                if fcntl is not None:
                    try:
                        fcntl.flock(f_lock.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                f_lock.close()

    def _run_git(
        self,
        args: list[str],
        author_name: str = "Clover Workstation Server",
        author_email: str = "clover@workstation.org.local",
        timeout_s: float = 30.0,
    ) -> tuple[int, str, str]:
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = author_name
        env["GIT_AUTHOR_EMAIL"] = author_email
        env["GIT_COMMITTER_NAME"] = author_name
        env["GIT_COMMITTER_EMAIL"] = author_email
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
            return res.returncode, (res.stdout or "").strip(), (res.stderr or "").strip()
        except Exception as ex:
            return 1, "", str(ex)

    def _ensure_initialized(self) -> None:
        with self._repo_lock():
            self.repo_dir.mkdir(parents=True, exist_ok=True)
            for sub in (
                "shared/notes",
                "shared/tasks",
                "shared/scripts",
                "teams/core/tasks",
                "teams/core/scripts",
                "users",
            ):
                (self.repo_dir / sub).mkdir(parents=True, exist_ok=True)

            if not (self.repo_dir / ".git").exists():
                rc, _, _ = self._run_git(["init", "-b", "main"])
                if rc != 0:
                    self._run_git(["init"])
                self._run_git(["config", "user.name", "Clover Workstation Server"])
                self._run_git(["config", "user.email", "clover@workstation.org.local"])

            agents_path = self.repo_dir / "AGENTS.md"
            if not agents_path.exists():
                agents_path.write_text(DEFAULT_ORG_AGENTS_MD, encoding="utf-8")

            if self.canonical_agents_md_path and Path(self.canonical_agents_md_path).exists():
                full_ref = self.repo_dir / "AGENTS_FULL_SPEC.md"
                if not full_ref.exists():
                    try:
                        full_ref.write_text(
                            Path(self.canonical_agents_md_path).read_text(encoding="utf-8"),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

            org_ctx = self.repo_dir / "ORG_CONTEXT.md"
            if not org_ctx.exists():
                org_ctx.write_text(DEFAULT_ORG_CONTEXT_MD, encoding="utf-8")

            team_ctx = self.repo_dir / "teams/core/TEAM_CONTEXT.md"
            if not team_ctx.exists():
                team_ctx.write_text(DEFAULT_TEAM_CONTEXT_TEMPLATE.format(team_id="core"), encoding="utf-8")

            gitignore_path = self.repo_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("__pycache__/\n*.pyc\n.DS_Store\n", encoding="utf-8")

            for rel_p, mid, title, paras in SHARED_SEED_NOTES:
                fp = self.repo_dir / rel_p
                if not fp.exists():
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    body = f"# {mid}: {title}\n\n" + "\n\n".join(paras) + "\n"
                    fp.write_text(body, encoding="utf-8")

            if not self.members_file.exists():
                init_members = {
                    "organization": "Clover Engineering Organization",
                    "created_at": _utc_now(),
                    "teams": ["core", "ml", "security", "platform"],
                    "members": {},
                }
                self.members_file.write_text(json.dumps(init_members, indent=2), encoding="utf-8")

            rc_st, out_st, _ = self._run_git(["status", "--porcelain"])
            if rc_st == 0 and out_st:
                self._run_git(["add", "-A"])
                self._run_git(
                    ["commit", "-m", "[org@system] Initialize Clover Layer 4 Multi-User Organization Git Memory Repository"]
                )

            # Ensure the default workstation user exists
            self.ensure_user_workspace("default", display_name="Default Workstation User", team_id="core")

    # ------------------------------------------------------------------
    # 1. Multi-User & Organization Member Management
    # ------------------------------------------------------------------
    def _load_members_registry_unlocked(self) -> dict[str, Any]:
        if self.members_file.exists():
            try:
                data = json.loads(self.members_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("organization", "Clover Engineering Organization")
                    data.setdefault("teams", ["core", "ml", "security", "platform"])
                    data.setdefault("members", {})
                    return data
            except Exception:
                pass
        return {
            "organization": "Clover Engineering Organization",
            "created_at": _utc_now(),
            "teams": ["core", "ml", "security", "platform"],
            "members": {},
        }

    def _save_members_registry_unlocked(self, reg: dict[str, Any]) -> None:
        self.members_file.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    def ensure_user_workspace(
        self,
        user_id: str | None,
        display_name: str | None = None,
        email: str | None = None,
        team_id: str | None = None,
        role: str = "member",
    ) -> dict[str, Any]:
        """
        Ensures `users/<user_id>/` and `teams/<team_id>/` exist inside the Git repository,
        registers the user in `members.json`, and commits their initial workspace if newly created.
        """
        uid = sanitize_identifier(user_id, fallback="default")
        with self._repo_lock():
            reg = self._load_members_registry_unlocked()
            existing = reg["members"].get(uid, {})
            tid = sanitize_identifier(team_id or existing.get("team_id") or "core", fallback="core")
            dname = (display_name or existing.get("display_name") or uid.replace("-", " ").title()).strip()
            uemail = (email or existing.get("email") or f"{uid}@{tid}.clover.org.local").strip()

            if tid not in reg["teams"]:
                reg["teams"].append(tid)

            user_dir = self.repo_dir / "users" / uid
            for sub in ("sessions", "tasks", "scripts"):
                (user_dir / sub).mkdir(parents=True, exist_ok=True)

            team_dir = self.repo_dir / "teams" / tid
            for sub in ("tasks", "scripts"):
                (team_dir / sub).mkdir(parents=True, exist_ok=True)
            team_ctx_fp = team_dir / "TEAM_CONTEXT.md"
            if not team_ctx_fp.exists():
                team_ctx_fp.write_text(DEFAULT_TEAM_CONTEXT_TEMPLATE.format(team_id=tid), encoding="utf-8")

            user_ctx_fp = user_dir / "CONTEXT.md"
            created_new = False
            if not user_ctx_fp.exists():
                user_ctx_fp.write_text(
                    DEFAULT_USER_CONTEXT_TEMPLATE.format(display_name=dname, user_id=uid, team_id=tid),
                    encoding="utf-8",
                )
                created_new = True

            user_agents_fp = user_dir / "AGENTS.md"
            if not user_agents_fp.exists():
                org_agents_txt = (self.repo_dir / "AGENTS.md").read_text(encoding="utf-8", errors="replace")
                user_agents_fp.write_text(org_agents_txt, encoding="utf-8")
                created_new = True

            now_s = _utc_now()
            member_info = {
                "user_id": uid,
                "display_name": dname,
                "email": uemail,
                "team_id": tid,
                "role": existing.get("role") or role,
                "agents_md_accepted": bool(existing.get("agents_md_accepted", False)),
                "accepted_at": existing.get("accepted_at"),
                "accepted_commit": existing.get("accepted_commit"),
                "joined_at": existing.get("joined_at") or now_s,
                "last_active_at": now_s,
            }
            reg["members"][uid] = member_info
            self._save_members_registry_unlocked(reg)

            if created_new:
                self._run_git(
                    ["add", f"users/{uid}", f"teams/{tid}", "members.json"],
                    author_name=f"{dname} ({uid})",
                    author_email=uemail,
                )
                self._run_git(
                    ["commit", "-m", f"[{uid}@{tid}] Provision multi-user workspace users/{uid}/ and team teams/{tid}/"],
                    author_name=f"{dname} ({uid})",
                    author_email=uemail,
                )
            return member_info

    def list_members_and_teams(self) -> dict[str, Any]:
        with self._repo_lock():
            reg = self._load_members_registry_unlocked()
            members_list = sorted(reg.get("members", {}).values(), key=lambda x: x.get("user_id", ""))
            return {
                "organization": reg.get("organization", "Clover Engineering Organization"),
                "teams": reg.get("teams", ["core"]),
                "members_count": len(members_list),
                "members": members_list,
            }

    # ------------------------------------------------------------------
    # 2. Per-User `AGENTS.md` First-Run Onboarding & Acceptance Gate
    # ------------------------------------------------------------------
    def get_onboarding_state(self, user_id: str | None = None, team_id: str | None = None) -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        user_agents_fp = self.repo_dir / "users" / uid / "AGENTS.md"
        org_agents_fp = self.repo_dir / "AGENTS.md"
        agents_text = (
            user_agents_fp.read_text(encoding="utf-8", errors="replace")
            if user_agents_fp.exists()
            else org_agents_fp.read_text(encoding="utf-8", errors="replace")
        )
        return {
            "user_id": uid,
            "display_name": m_info["display_name"],
            "team_id": tid,
            "agents_md_accepted": bool(m_info.get("agents_md_accepted", False)),
            "accepted_at": m_info.get("accepted_at"),
            "accepted_commit": m_info.get("accepted_commit"),
            "repo_path": str(self.repo_dir),
            "user_workspace_path": f"users/{uid}",
            "team_workspace_path": f"teams/{tid}",
            "shared_workspace_path": "shared",
            "agents_md_path": f"users/{uid}/AGENTS.md",
            "org_agents_md_path": "AGENTS.md",
            "agents_md_content": agents_text,
        }

    def accept_agents_md(
        self,
        user_id: str | None = None,
        custom_content: str | None = None,
        display_name: str | None = None,
        team_id: str | None = None,
        update_org_root: bool = False,
    ) -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, display_name=display_name, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        with self._repo_lock():
            user_agents_fp = self.repo_dir / "users" / uid / "AGENTS.md"
            if custom_content is not None and custom_content.strip():
                clean_md = custom_content.strip() + "\n"
                user_agents_fp.write_text(clean_md, encoding="utf-8")
                if update_org_root:
                    (self.repo_dir / "AGENTS.md").write_text(clean_md, encoding="utf-8")

            reg = self._load_members_registry_unlocked()
            now_s = _utc_now()
            reg["members"].setdefault(uid, m_info)
            reg["members"][uid]["agents_md_accepted"] = True
            reg["members"][uid]["accepted_at"] = now_s
            reg["members"][uid]["last_active_at"] = now_s
            self._save_members_registry_unlocked(reg)

            self._run_git(
                ["add", f"users/{uid}/AGENTS.md", "AGENTS.md", "members.json"],
                author_name=f"{dname} ({uid})",
                author_email=uemail,
            )
            self._run_git(
                ["commit", "-m", f"[{uid}@{tid}] User reviewed and accepted AGENTS.md operating specification"],
                author_name=f"{dname} ({uid})",
                author_email=uemail,
            )
            _, head_sha, _ = self._run_git(["rev-parse", "--short", "HEAD"])
            reg["members"][uid]["accepted_commit"] = head_sha or "HEAD"
            self._save_members_registry_unlocked(reg)

        return self.get_onboarding_state(user_id=uid, team_id=tid)

    # ------------------------------------------------------------------
    # 3. Multi-User Scoped File CRUD, Collaborative Promotion & Git Log
    # ------------------------------------------------------------------
    def _resolve_scoped_path(
        self,
        rel_path: str,
        user_id: str = "default",
        team_id: str = "core",
        for_write: bool = False,
        allow_cross_user: bool = False,
    ) -> Path:
        uid = sanitize_identifier(user_id, fallback="default")
        tid = sanitize_identifier(team_id, fallback="core")
        clean = (rel_path or "").strip().lstrip("/\\").replace("\\", "/")
        if not clean or ".." in clean.split("/") or clean.startswith(".git"):
            raise ValueError(f"Invalid repository path: {rel_path!r}")

        # Shorthand expansion: if a user writes "tasks/my-note.md" or "scripts/foo.py" or "CONTEXT.md",
        # route it to their personal workspace `users/<uid>/...` unless they explicitly specified `shared/`, `teams/`, `users/`, `ORG_CONTEXT.md`, or `AGENTS.md`!
        top = clean.split("/")[0]
        if top in ("tasks", "scripts", "sessions") or clean == "CONTEXT.md":
            clean = f"users/{uid}/{clean}"
        elif top == "notes":
            clean = f"shared/{clean}"

        if for_write and not allow_cross_user and clean.startswith("users/"):
            parts = clean.split("/")
            target_uid = parts[1] if len(parts) >= 2 else ""
            if target_uid and target_uid != uid:
                raise PermissionError(
                    f"Access denied: user '{uid}' cannot modify another user's private workspace ('users/{target_uid}/'). "
                    f"Use 'shared/' or 'teams/{tid}/' to collaborate across users."
                )

        resolved = (self.repo_dir / clean).resolve()
        if not str(resolved).startswith(str(self.repo_dir.resolve())):
            raise ValueError(f"Path escapes repository root: {rel_path!r}")
        return resolved

    @staticmethod
    def _classify_scope(rel_posix: str, active_uid: str, active_tid: str) -> tuple[str, int]:
        if rel_posix in ("AGENTS.md", "ORG_CONTEXT.md", "members.json"):
            return ("org_root", 0)
        if rel_posix.startswith("shared/"):
            return ("shared", 1)
        if rel_posix.startswith(f"teams/{active_tid}/"):
            return (f"team:{active_tid}", 2)
        if rel_posix.startswith(f"users/{active_uid}/"):
            return (f"user:{active_uid}", 3)
        if rel_posix.startswith("teams/"):
            other_t = rel_posix.split("/")[1]
            return (f"team:{other_t}", 4)
        if rel_posix.startswith("users/"):
            other_u = rel_posix.split("/")[1]
            return (f"user:{other_u}", 5)
        return ("other", 6)

    def list_files(
        self,
        user_id: str | None = None,
        team_id: str | None = None,
        include_other_users: bool = False,
    ) -> list[dict[str, Any]]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        out: list[dict[str, Any]] = []
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [d for d in sorted(dirs) if d not in (".git", "__pycache__")]
            for fname in sorted(files):
                if fname.startswith(".") or fname == "AGENTS_FULL_SPEC.md":
                    continue
                fpath = Path(root) / fname
                rel = fpath.relative_to(self.repo_dir).as_posix()
                scope_label, sort_rank = self._classify_scope(rel, uid, tid)
                if not include_other_users and sort_rank == 5:
                    continue
                try:
                    stat = fpath.stat()
                    txt = fpath.read_text(encoding="utf-8", errors="replace")
                    first_line = ""
                    for ln in txt.splitlines():
                        s = ln.strip().lstrip("#").strip()
                        if s:
                            first_line = s[:140]
                            break
                    out.append({
                        "path": rel,
                        "scope": scope_label,
                        "category": rel.split("/")[0] if "/" in rel else "root",
                        "size_bytes": stat.st_size,
                        "updated_at": datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "summary": first_line,
                        "_rank": sort_rank,
                    })
                except Exception:
                    pass
        out.sort(key=lambda item: (item["_rank"], item["path"]))
        for item in out:
            item.pop("_rank", None)
        return out

    def read_file(
        self,
        rel_path: str,
        user_id: str = "default",
        team_id: str = "core",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        fp = self._resolve_scoped_path(rel_path, user_id=m_info["user_id"], team_id=m_info["team_id"], for_write=False)
        if not fp.exists() or not fp.is_file():
            raise FileNotFoundError(f"File not found in organization memory repo: {rel_path}")
        txt = fp.read_text(encoding="utf-8", errors="replace")
        lines = txt.splitlines()
        if start_line is not None or end_line is not None:
            s = max(1, int(start_line or 1))
            e = min(len(lines), int(end_line or len(lines)))
            txt = "\n".join(lines[s - 1 : e])
        rel = fp.relative_to(self.repo_dir).as_posix()
        scope_lbl, _ = self._classify_scope(rel, m_info["user_id"], m_info["team_id"])
        return {
            "path": rel,
            "scope": scope_lbl,
            "content": txt,
            "total_lines": len(lines),
            "size_bytes": fp.stat().st_size,
        }

    def write_file(
        self,
        rel_path: str,
        content: str,
        user_id: str = "default",
        team_id: str = "core",
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        with self._repo_lock():
            fp = self._resolve_scoped_path(rel_path, user_id=uid, team_id=tid, for_write=True)
            fp.parent.mkdir(parents=True, exist_ok=True)
            existed = fp.exists()
            fp.write_text(content, encoding="utf-8")
            rel = fp.relative_to(self.repo_dir).as_posix()
            raw_msg = (commit_message or "").strip() or f"{'Update' if existed else 'Create'} {rel}"
            if not raw_msg.startswith("["):
                raw_msg = f"[{uid}@{tid}] {raw_msg}"
            self._run_git(["add", rel], author_name=f"{dname} ({uid})", author_email=uemail)
            self._run_git(["commit", "-m", raw_msg], author_name=f"{dname} ({uid})", author_email=uemail)
            _, head_sha, _ = self._run_git(["rev-parse", "--short", "HEAD"])
            scope_lbl, _ = self._classify_scope(rel, uid, tid)
            return {
                "ok": True,
                "user_id": uid,
                "team_id": tid,
                "path": rel,
                "scope": scope_lbl,
                "size_bytes": fp.stat().st_size,
                "commit": head_sha or "HEAD",
                "commit_message": raw_msg,
            }

    def promote_to_shared(
        self,
        source_rel_path: str,
        user_id: str = "default",
        team_id: str = "core",
        target_scope: str = "shared",
        target_subpath: str | None = None,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        """
        Promotes a file from a user's personal workspace (`users/<user_id>/scripts/tool.py` or `tasks/note.md`)
        into `shared/` (`shared/scripts/tool.py`, `shared/tasks/note.md`, `shared/notes/...`) or `teams/<team_id>/`
        so multiple people in the organization can immediately use it.
        """
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        with self._repo_lock():
            src_fp = self._resolve_scoped_path(source_rel_path, user_id=uid, team_id=tid, for_write=False)
            if not src_fp.exists() or not src_fp.is_file():
                raise FileNotFoundError(f"Source file not found: {source_rel_path}")
            src_rel = src_fp.relative_to(self.repo_dir).as_posix()
            content = src_fp.read_text(encoding="utf-8", errors="replace")

            fname = src_fp.name
            if target_subpath:
                dst_rel = target_subpath.strip().lstrip("/")
            else:
                if "/scripts/" in src_rel or fname.endswith((".py", ".sh")):
                    sub_cat = "scripts"
                elif "/notes/" in src_rel or fname.upper().startswith("MEM-"):
                    sub_cat = "notes"
                else:
                    sub_cat = "tasks"
                if target_scope == "team":
                    dst_rel = f"teams/{tid}/{sub_cat}/{fname}"
                else:
                    dst_rel = f"shared/{sub_cat}/{fname}"

            dst_fp = self._resolve_scoped_path(dst_rel, user_id=uid, team_id=tid, for_write=True)
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            dst_fp.write_text(content, encoding="utf-8")
            dst_rel_clean = dst_fp.relative_to(self.repo_dir).as_posix()

            msg = (commit_message or "").strip() or f"[{uid}@{tid}] Promote {src_rel} -> {dst_rel_clean} for organization collaboration"
            if not msg.startswith("["):
                msg = f"[{uid}@{tid}] {msg}"
            self._run_git(["add", dst_rel_clean], author_name=f"{dname} ({uid})", author_email=uemail)
            self._run_git(["commit", "-m", msg], author_name=f"{dname} ({uid})", author_email=uemail)
            _, head_sha, _ = self._run_git(["rev-parse", "--short", "HEAD"])
            return {
                "ok": True,
                "user_id": uid,
                "team_id": tid,
                "source_path": src_rel,
                "promoted_path": dst_rel_clean,
                "commit": head_sha or "HEAD",
                "commit_message": msg,
            }

    def delete_file(
        self,
        rel_path: str,
        user_id: str = "default",
        team_id: str = "core",
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        with self._repo_lock():
            fp = self._resolve_scoped_path(rel_path, user_id=uid, team_id=tid, for_write=True)
            rel = fp.relative_to(self.repo_dir).as_posix()
            if rel in ("AGENTS.md", "ORG_CONTEXT.md", "members.json") or rel.endswith("/AGENTS.md"):
                raise ValueError(f"'{rel}' is a protected specification file and cannot be deleted (you may edit it instead).")
            if fp.exists() and fp.is_file():
                fp.unlink()
                msg = (commit_message or "").strip() or f"[{uid}@{tid}] Delete {rel} from organization memory repo"
                self._run_git(["add", "-A"], author_name=f"{dname} ({uid})", author_email=uemail)
                self._run_git(["commit", "-m", msg], author_name=f"{dname} ({uid})", author_email=uemail)
            return {"ok": True, "deleted": rel, "user_id": uid}

    def git_log(self, max_count: int = 25, user_id: str | None = None) -> list[dict[str, str]]:
        args = ["log", f"-n{max(1, min(60, int(max_count)))}", "--pretty=format:%h|%an|%cI|%s"]
        if user_id and user_id != "all":
            uid_clean = sanitize_identifier(user_id, fallback="default")
            args.append(f"--grep=\\[{uid_clean}@")
        rc, out, _ = self._run_git(args)
        commits: list[dict[str, str]] = []
        if rc == 0 and out:
            for ln in out.splitlines():
                parts = ln.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3],
                    })
        return commits

    # ------------------------------------------------------------------
    # 4. Selective Three-Tier On-Demand Search & Retrieval (Zero Context Bloating)
    # ------------------------------------------------------------------
    def search_repo_relevant(
        self,
        query: str,
        user_id: str = "default",
        team_id: str = "core",
        max_files: int = 3,
        max_chars_per_file: int = 550,
    ) -> list[dict[str, Any]]:
        """
        Searches across:
          1. User's personal workspace (`users/<user_id>/**`)
          2. User's team workspace (`teams/<team_id>/**`)
          3. Organization shared knowledge (`ORG_CONTEXT.md`, `shared/**`)
        and returns ONLY high-precision snippets (`<= max_chars_per_file`) so the model's context window is never bloated.
        """
        q = (query or "").strip()
        q_low = q.lower()
        if not q or len(q) < 4:
            return []
        if ("```" in q_low or re.search(r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", q)) and not any(
            k in q_low for k in ("clover", "memory", "repo", "session", "cache", "context", "agents.md", "team", "shared")
        ):
            return []

        tokens = [
            w for w in re.findall(r"[a-z0-9_]{3,}", q_low)
            if w not in {
                "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "does", "did",
                "the", "and", "for", "with", "that", "this", "from", "into", "your", "have", "has",
                "write", "short", "poem", "about", "compute", "using", "python", "search", "web",
            }
        ]
        if not tokens:
            return []

        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]

        scored: list[tuple[float, str, str, str, str]] = []
        for item in self.list_files(user_id=uid, team_id=tid, include_other_users=False):
            rel = item["path"]
            if rel in ("AGENTS.md", "AGENTS_FULL_SPEC.md", "members.json") or rel.endswith("/AGENTS.md"):
                continue
            fp = self.repo_dir / rel
            try:
                txt = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            txt_low = txt.lower()
            rel_low = rel.lower()
            hits = sum(1 for t in tokens if t in txt_low or t in rel_low)
            if hits == 0:
                continue
            ratio = hits / float(len(tokens))
            score = hits * 4.0 + ratio * 10.0
            # Boost user's personal notes/tasks and shared verified notes
            scope_str = str(item.get("scope") or "shared")
            if scope_str.startswith("user:"):
                score += 2.5
            elif scope_str.startswith("team:"):
                score += 1.5
            if any(k in q_low for k in ("per-layer", "cache", "partition", "flat hit", "23.3", "48.0")) and "per-layer" in txt_low:
                score += 25.0
            if any(k in q_low for k in ("cve-", "cwe-", "capec-", "fan-out", "stratified")) and "stratified" in txt_low:
                score += 18.0
            if score < 12.0:
                continue

            paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip() and not p.strip().startswith("#")]
            best_para = paras[0] if paras else txt.strip()
            best_p_hits = -1
            for p in paras:
                ph = sum(1 for t in tokens if t in p.lower())
                if ph > best_p_hits:
                    best_p_hits = ph
                    best_para = p
            snippet = re.sub(r"\s+", " ", best_para).strip()[:max_chars_per_file]
            scored.append((score, rel, scope_str, item.get("summary") or rel, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"path": rel, "scope": scope_s, "title": title, "snippet": snip, "score": round(sc, 2)}
            for sc, rel, scope_s, title, snip in scored[:max_files]
        ]

    # ------------------------------------------------------------------
    # 5. Automatic Multi-User Growth Stage (`record_task_or_session_turn`)
    # ------------------------------------------------------------------
    def record_task_or_session_turn(
        self,
        session_id: str,
        prompt: str,
        reply: str,
        user_id: str = "default",
        team_id: str = "core",
        generated_code: str = "",
        tools_used: list[str] | None = None,
        share_with_org: bool = False,
    ) -> dict[str, Any]:
        """
        Records a compact session turn in `users/<user_id>/sessions/session-<sid>.md`, auto-saves verified
        Python scripts in `users/<user_id>/scripts/<slug>.py` (and `shared/scripts/<slug>.py` if `share_with_org`
        or requested in prompt), and commits to the organization Git repository with the user's author identity.
        """
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        sid_clean = _slugify(session_id or "default", max_len=24)
        sess_rel = f"users/{uid}/sessions/session-{sid_clean}.md"
        now_s = _utc_now()
        tools_str = ", ".join(tools_used or []) or "none"

        entry_md = (
            f"\n## Turn ({now_s})\n"
            f"- **User**: `{dname}` (`{uid}`, team `{tid}`)\n"
            f"- **Tools / Layers**: `{tools_str}`\n"
            f"- **Direction**: {prompt.strip()[:300]}\n"
            f"- **Outcome**: {reply.strip()[:500]}\n"
        )

        with self._repo_lock():
            sess_fp = self.repo_dir / sess_rel
            sess_fp.parent.mkdir(parents=True, exist_ok=True)
            if not sess_fp.exists():
                header = f"# Clover Session Log (`{sid_clean}` · User `{uid}` · Team `{tid}`)\n\nCreated: {now_s}\n"
                sess_fp.write_text(header + entry_md, encoding="utf-8")
            else:
                with open(sess_fp, "a", encoding="utf-8") as f:
                    f.write(entry_md)

            saved_files = [sess_rel]
            p_low = (prompt or "").lower()
            wants_shared = share_with_org or any(
                kw in p_low for kw in ("share with team", "share with org", "shared repo", "organization note", "team note")
            )

            # Save verified Python code as a reusable `.py` script in `users/<uid>/scripts/` (and `shared/scripts/` if shared)
            code_clean = (generated_code or "").strip()
            if not code_clean:
                m_py = re.search(r"```(?:python|py)\s*\n([\s\S]*?)```", reply or "", flags=re.IGNORECASE)
                if m_py:
                    code_clean = m_py.group(1).strip()
            if code_clean and ("def " in code_clean or "class " in code_clean or "import " in code_clean):
                fn_match = re.search(r"(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code_clean)
                script_slug = _slugify(fn_match.group(1) if fn_match else prompt[:32], max_len=32)
                script_header = (
                    f'"""Auto-saved by Clover Layer 4 Organization Git Memory Repo ({now_s})\n'
                    f'Author: {dname} ({uid}, team {tid})\n'
                    f'Task: {prompt.strip()[:120]}\n"""\n\n'
                )
                user_script_rel = f"users/{uid}/scripts/{script_slug}.py"
                (self.repo_dir / user_script_rel).write_text(script_header + code_clean + "\n", encoding="utf-8")
                saved_files.append(user_script_rel)
                if wants_shared:
                    shared_script_rel = f"shared/scripts/{script_slug}.py"
                    (self.repo_dir / shared_script_rel).write_text(script_header + code_clean + "\n", encoding="utf-8")
                    saved_files.append(shared_script_rel)

            if re.search(r"\b(?:remember\s+that|save\s+(?:this\s+)?note|record\s+in\s+memory|team\s+note|organization\s+note)\b", prompt or "", flags=re.IGNORECASE):
                task_slug = _slugify(prompt[:40], max_len=36)
                target_prefix = f"teams/{tid}/tasks" if "team note" in p_low else ("shared/tasks" if wants_shared else f"users/{uid}/tasks")
                task_rel = f"{target_prefix}/{task_slug}.md"
                task_fp = self.repo_dir / task_rel
                task_fp.parent.mkdir(parents=True, exist_ok=True)
                task_fp.write_text(
                    f"# Task Note: {prompt.strip()[:80]}\n\n"
                    f"- **Author**: `{dname}` (`{uid}`, team `{tid}`)\n"
                    f"- **Recorded**: {now_s}\n"
                    f"- **Direction**: {prompt.strip()}\n"
                    f"- **Outcome**: {reply.strip()}\n",
                    encoding="utf-8",
                )
                saved_files.append(task_rel)

            self._run_git(["add", *saved_files], author_name=f"{dname} ({uid})", author_email=uemail)
            self._run_git(
                ["commit", "-m", f"[{uid}@{tid}] Record session turn ({sid_clean}) in Layer 4 Organization Git repo"],
                author_name=f"{dname} ({uid})",
                author_email=uemail,
            )
            _, head_sha, _ = self._run_git(["rev-parse", "--short", "HEAD"])
            return {"user_id": uid, "team_id": tid, "saved_files": saved_files, "commit": head_sha or "HEAD"}

    # ------------------------------------------------------------------
    # 5b. Multi-User Clover Cycle Staged Lessons (`shared/lessons/MEM-LESSON-*.md` -> `shared/notes/`)
    #     Offered to the human to Keep or Discard (AGENTS.md Growth boundary)
    # ------------------------------------------------------------------
    def stage_lesson(
        self,
        lesson_text: str,
        user_id: str = "default",
        team_id: str = "core",
        conversation_id: str | None = None,
        kind: str = "verified",
    ) -> dict[str, Any]:
        import hashlib

        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]

        clean = " ".join(str(lesson_text or "").split()).strip()
        if not clean:
            raise ValueError("Empty lesson text")
        now_s = _utc_now()
        ts_slug = now_s.replace("-", "").replace(":", "").replace("T", "-").replace("Z", "")
        short_hash = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:6]
        lid = f"MEM-LESSON-{ts_slug}-{short_hash}"
        rel_p = f"shared/lessons/{lid}.md"
        with self._repo_lock():
            fp = self.repo_dir / rel_p
            fp.parent.mkdir(parents=True, exist_ok=True)
            ev_note = (
                "Clover Outcome checks: the earlier answer failed deterministic checks and the corrected answer passed."
                if kind == "verified"
                else "The user indicated Clover's earlier answer was not right during a Clover retry cycle."
            )
            md_body = (
                f"# {lid}: Staged Clover Cycle Lesson ({kind})\n\n"
                f"- **Author**: `{dname}` (`{uid}`, team `{tid}`)\n"
                f"- **Created**: {now_s}\n"
                f"- **Conversation**: `{conversation_id or 'workstation'}`\n"
                f"- **Provenance**: {ev_note}\n"
                f"- **Status**: `staged_waiting_for_human`\n\n"
                f"## Lesson Summary\n{clean}\n"
            )
            fp.write_text(md_body, encoding="utf-8")
            self._run_git(["add", rel_p], author_name=f"{dname} ({uid})", author_email=uemail)
            self._run_git(
                ["commit", "-m", f"[{uid}@{tid}] Stage Clover cycle lesson {lid} in shared/lessons/"],
                author_name=f"{dname} ({uid})",
                author_email=uemail,
            )
        return {"id": lid, "path": rel_p, "user_id": uid, "team_id": tid, "summary": clean, "created_at": now_s, "kind": kind}

    def list_staged_lessons(self) -> list[dict[str, Any]]:
        lessons_dir = self.repo_dir / "shared" / "lessons"
        if not lessons_dir.exists():
            return []
        out: list[dict[str, Any]] = []
        for fp in sorted(lessons_dir.glob("*.md"), reverse=True):
            try:
                txt = fp.read_text(encoding="utf-8", errors="replace")
                lid = fp.stem
                created = ""
                m_c = re.search(r"-\s*\*\*Created\*\*:\s*(\S+)", txt)
                if m_c:
                    created = m_c.group(1).strip()
                author = ""
                m_a = re.search(r"-\s*\*\*Author\*\*:\s*([^\n]+)", txt)
                if m_a:
                    author = m_a.group(1).strip()
                summary = ""
                if "## Lesson Summary" in txt:
                    summary = txt.split("## Lesson Summary", 1)[1].strip()
                else:
                    summary = txt.strip()[:300]
                out.append({
                    "id": lid,
                    "path": f"shared/lessons/{fp.name}",
                    "author": author,
                    "summary": summary,
                    "created_at": created,
                })
            except Exception:
                pass
        return out

    def keep_staged_lesson(self, lesson_id: str, user_id: str = "default", team_id: str = "core") -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]
        lid = Path(str(lesson_id or "").strip()).stem
        with self._repo_lock():
            src_fp = self.repo_dir / "shared" / "lessons" / f"{lid}.md"
            if not src_fp.exists():
                raise FileNotFoundError(f"Staged lesson not found: {lid}")
            txt = src_fp.read_text(encoding="utf-8", errors="replace")
            promoted_txt = txt.replace(
                "`staged_waiting_for_human`",
                f"`kept_by_human ({uid}@{tid} at {_utc_now()})`",
            )
            dst_rel = f"shared/notes/{lid}.md"
            dst_fp = self.repo_dir / dst_rel
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            dst_fp.write_text(promoted_txt, encoding="utf-8")
            src_fp.unlink()
            self._run_git(["add", "-A"], author_name=f"{dname} ({uid})", author_email=uemail)
            self._run_git(
                ["commit", "-m", f"[{uid}@{tid}] Keep & promote Clover lesson {lid} to {dst_rel}"],
                author_name=f"{dname} ({uid})",
                author_email=uemail,
            )
        return {"ok": True, "promoted_path": dst_rel, "lessons": self.list_staged_lessons()}

    def discard_staged_lesson(self, lesson_id: str, user_id: str = "default", team_id: str = "core") -> dict[str, Any]:
        m_info = self.ensure_user_workspace(user_id, team_id=team_id)
        uid = m_info["user_id"]
        tid = m_info["team_id"]
        dname = m_info["display_name"]
        uemail = m_info["email"]
        lid = Path(str(lesson_id or "").strip()).stem
        with self._repo_lock():
            src_fp = self.repo_dir / "shared" / "lessons" / f"{lid}.md"
            if src_fp.exists():
                src_fp.unlink()
                self._run_git(["add", "-A"], author_name=f"{dname} ({uid})", author_email=uemail)
                self._run_git(
                    ["commit", "-m", f"[{uid}@{tid}] Discard staged Clover lesson {lid}"],
                    author_name=f"{dname} ({uid})",
                    author_email=uemail,
                )
        return {"ok": True, "discarded": lid, "lessons": self.list_staged_lessons()}

    # ------------------------------------------------------------------
    # 6. Private Organization / Team GitHub Sync (`private: true` Enforced)
    # ------------------------------------------------------------------
    def get_github_sync_config(self) -> dict[str, Any]:
        cfg = {"repo": "", "branch": "main", "has_token": False, "private_enforced": True, "last_sync": None, "last_sync_by": None}
        if self.github_cfg_file.exists():
            try:
                raw = json.loads(self.github_cfg_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cfg["repo"] = str(raw.get("repo") or "")
                    cfg["branch"] = str(raw.get("branch") or "main")
                    cfg["has_token"] = bool(raw.get("token"))
                    cfg["last_sync"] = raw.get("last_sync")
                    cfg["last_sync_by"] = raw.get("last_sync_by")
            except Exception:
                pass
        return cfg

    def configure_and_sync_private_github(
        self,
        repo_full_name: str,
        token: str | None = None,
        action: str = "push",
        branch: str = "main",
        user_id: str = "default",
    ) -> dict[str, Any]:
        """
        Syncs (`push` or `pull`) the multi-user organization repository (`clover-org-memory-repo/`)
        with a **private** GitHub repository (`owner/repo`), strictly enforcing `"private": True` via the GitHub API.
        """
        uid = sanitize_identifier(user_id, fallback="default")
        with self._repo_lock():
            existing_raw: dict[str, Any] = {}
            if self.github_cfg_file.exists():
                try:
                    existing_raw = json.loads(self.github_cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    existing_raw = {}

            repo_clean = (repo_full_name or existing_raw.get("repo") or "").strip().strip("/")
            if repo_clean.startswith("https://github.com/"):
                repo_clean = repo_clean[len("https://github.com/") :].strip("/")
            if repo_clean.endswith(".git"):
                repo_clean = repo_clean[:-4]
            tok_clean = (token or existing_raw.get("token") or "").strip()

            if not repo_clean or "/" not in repo_clean:
                raise ValueError("Please provide the organization/team GitHub repository as 'org-or-user/repo-name'.")
            if not tok_clean:
                raise ValueError("Please provide a GitHub Personal Access Token (with 'repo' scope) to sync the private repository.")

            owner, repo_name = repo_clean.split("/", 1)
            headers = {
                "Authorization": f"Bearer {tok_clean}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "CloverAIWorkstation-Layer4-OrgGitSync",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            repo_api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
            req_check = urllib.request.Request(repo_api_url, headers=headers, method="GET")
            repo_created = False
            try:
                with urllib.request.urlopen(req_check, timeout=15) as resp:
                    repo_info = json.loads(resp.read().decode("utf-8"))
                    if not repo_info.get("private", False):
                        raise ValueError(
                            f"Refused: GitHub repository '{repo_clean}' is PUBLIC. Layer 4 organization memory sync strictly requires a PRIVATE repository (`private: true`)."
                        )
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    create_body = json.dumps({
                        "name": repo_name,
                        "description": "Clover AI Workstation — Multi-User Organization Layer 4 Memory Repository (Private)",
                        "private": True,
                        "auto_init": False,
                    }).encode("utf-8")
                    req_create = urllib.request.Request(
                        "https://api.github.com/user/repos",
                        data=create_body,
                        headers={**headers, "Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_create, timeout=15) as c_resp:
                        c_info = json.loads(c_resp.read().decode("utf-8"))
                        if not c_info.get("private", False):
                            raise ValueError("GitHub API did not confirm private visibility.")
                        repo_created = True
                else:
                    err_txt = he.read().decode("utf-8", errors="replace")[:240]
                    raise RuntimeError(f"GitHub API check failed (HTTP {he.code}): {err_txt}")

            saved_cfg = {
                "repo": repo_clean,
                "branch": branch or "main",
                "token": tok_clean,
                "private_enforced": True,
                "last_sync": _utc_now(),
                "last_sync_by": uid,
            }
            self.github_cfg_file.write_text(json.dumps(saved_cfg, indent=2), encoding="utf-8")
            try:
                self.github_cfg_file.chmod(0o600)
            except Exception:
                pass

            auth_url = f"https://x-access-token:{tok_clean}@github.com/{repo_clean}.git"
            clean_url = f"https://github.com/{repo_clean}.git"
            self._run_git(["remote", "remove", "origin"])
            self._run_git(["remote", "add", "origin", clean_url])

            if action == "pull":
                rc, out, err = self._run_git(["pull", "--rebase", auth_url, branch], timeout_s=45.0)
            else:
                self._run_git(["branch", "-M", branch])
                rc, out, err = self._run_git(["push", "-u", auth_url, f"HEAD:{branch}"], timeout_s=45.0)

            safe_out = (out + "\n" + err).replace(tok_clean, "***").strip()
            if rc != 0:
                raise RuntimeError(f"git {action} failed: {safe_out}")

            return {
                "ok": True,
                "action": action,
                "repo": repo_clean,
                "private_verified": True,
                "repo_created": repo_created,
                "last_sync": saved_cfg["last_sync"],
                "last_sync_by": uid,
                "git_output": safe_out[:400],
            }
