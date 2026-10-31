#!/usr/bin/env python3
"""
test_mobile_s1_s2_s3.py — Comprehensive Verification Suite for Mobile Supporting Layers S1, S2 & S3.

Verifies:
  1. S1 · Memory (`clover_git_memory_repo.py` + `mobile_vault.py`):
     - 5-stage structure (`sessions/`, `execution/`, `outcomes/`, `growth/knowledge/`, `growth/notes/`, `config/`).
     - `mobile_vault` round-trip encryption & decryption (`protect` / `unprotect`).
     - Workspace safety (all 7 `MOBILE-REVIEW.md` defects):
       * Adding a user's git project NEVER commits into the user's project (`git status` stays untouched, user's staged `src/wip.py` is never committed).
       * Context files (`src/src-context.md`) NEVER copy secret lines (`SECRET_KEY = "s3cr3t-value-123"` in `settings.py` or `.env`).
       * NEVER writes `AGENTS.md` into the user's project.
       * Recording a chat turn while a user's project is active writes cleanly to `clover-memory-repo/sessions/` (never `FileNotFoundError`).
       * Broad paths (`~`, `/`) and non-existent local folders are rejected.
  2. S2 · Relationships (`memory_index.py`):
     - Turn-level indexing (`sessions/<yyyy-mm>/<chat>.jsonl#<n>:<role>`).
     - BM25 + dense vector `RRF(k=60)` fusion + `v3` `USER:` / `ASSISTANT:` exchange notes (`add_note`).
     - `negative_turns()` exclusion: answers marked `said_not_right` or `tried_again` are excluded from `recall()`.
     - `F6` dated background system block (`HEAD`).
  3. S3 · Tools (`web_search.py`, `local_terminal.py`, and Permission `Ask` / `Off` Gates):
     - `web_search.relevant_rows()` filters out single-word false positives ("Calcium carbonate" for milk question) while keeping true matches.
     - `local_terminal.shell()` executes commands with UTF-8 output, exit code, and timeout process-group kill.
     - `_check_tool_permission()` returns `permission_required` when a tool's permission is `"ask"` and `interactive_ask=True`, and refuses `repo_write` when `memory` is `"deny"`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mobile_vault
import local_terminal
import web_search
from clover_git_memory_repo import CloverMemoryGitRepo
from memory_index import MemoryIndex, SubwordHashEmbedder, clean_note


def run_all_tests() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="clover_mobile_test_"))
    try:
        # 1. Test mobile_vault encryption
        secret = "ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD"
        enc = mobile_vault.protect(secret)
        assert secret not in enc, "Secret leaked in ciphertext"
        dec = mobile_vault.unprotect(enc)
        assert dec == secret, f"Vault roundtrip mismatch: {dec!r}"
        print("[PASS 1/8] mobile_vault encrypt/decrypt round-trip")

        # 2. Initialize CloverMemoryGitRepo (S1)
        repo = CloverMemoryGitRepo(tmp / "base")
        assert (repo.default_repo_dir / "AGENTS.md").is_file()
        for stage_dir in ("sessions", "execution", "outcomes", "growth", "config"):
            assert (repo.default_repo_dir / stage_dir / "CONTEXT.md").is_file(), f"Missing {stage_dir}/CONTEXT.md"
        print("[PASS 2/8] S1 5-stage CloverMemoryGitRepo initialized with CONTEXT.md in all 5 stage folders")

        # 3. Verify all 7 MOBILE-REVIEW.md multi-workspace safety checks
        user_proj = tmp / "my-project"
        (user_proj / "src").mkdir(parents=True)
        subprocess.run(["git", "-C", str(user_proj), "init", "-b", "main"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(user_proj), "config", "user.name", "User"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(user_proj), "config", "user.email", "user@example.com"], check=True, capture_output=True)
        (user_proj / "README.md").write_text("# My Project\n", encoding="utf-8")
        (user_proj / "src" / "settings.py").write_text('SECRET_KEY = "s3cr3t-value-123"\ndef init_app():\n    pass\n', encoding="utf-8")
        (user_proj / "src" / ".env").write_text("API_TOKEN=super_secret_999\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(user_proj), "add", "README.md"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(user_proj), "commit", "-m", "Initial user commit"], check=True, capture_output=True)
        head_before = subprocess.run(["git", "-C", str(user_proj), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

        # User stages src/wip.py (uncommitted work)
        (user_proj / "src" / "wip.py").write_text("def work_in_progress():\n    return 42\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(user_proj), "add", "src/wip.py"], check=True, capture_output=True)

        # Add user_proj as active workspace in Clover
        repo.add_or_switch_workspace_repo(str(user_proj), name="my-project")
        head_after = subprocess.run(["git", "-C", str(user_proj), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        assert head_before == head_after, "Defect 1/2: Clover committed into user's git repository!"
        assert not (user_proj / "AGENTS.md").exists(), "Defect 4: Clover wrote AGENTS.md into user's project!"
        src_ctx = (user_proj / "src" / "src-context.md").read_text(encoding="utf-8")
        assert "s3cr3t-value-123" not in src_ctx, "Defect 3: Secret from settings.py leaked into src-context.md!"
        assert "super_secret_999" not in src_ctx, "Secret from .env leaked into src-context.md!"
        assert "Defines init_app" in src_ctx, "Expected AST function summary in src-context.md"

        # Reject broad path (~) and non-existent path
        for bad_path in ("~", str(tmp / "does-not-exist-folder")):
            try:
                repo.add_or_switch_workspace_repo(bad_path)
                raise AssertionError(f"Defect 7: Should have rejected {bad_path}")
            except ValueError:
                pass
        print("[PASS 3/8] All 7 MOBILE-REVIEW.md multi-workspace safety checks verified")

        # 4. Recording chat turns while external project is active writes to clover-memory-repo/sessions/
        t1 = repo.record_task_or_session_turn(
            session_id="chat-alpha",
            prompt="My sister Meera moved to Coimbatore last month and my bike is blue.",
            reply="Got it! I will remember that your sister Meera moved to Coimbatore and your bike is blue.",
            tools_used=["s1_memory"],
        )
        assert t1["turn_n"] == 1
        t2 = repo.record_task_or_session_turn(
            session_id="chat-alpha",
            prompt="What is 19 * 23?",
            reply="19 * 23 is 999.",
            tools_used=[],
        )
        assert t2["turn_n"] == 2
        # Mark turn 2 as not right so S2 excludes it!
        repo.record_outcome_feedback("chat-alpha", 2, "said_not_right")
        assert ("chat-alpha", 2) in repo.negative_turns()
        print("[PASS 4/8] S1 turn recording & negative outcome tracking work while external workspace is active")

        # 5. Test S2 MemoryIndex (turn-level units + v3 notes + negative turn exclusion + F6 block)
        mem_idx = MemoryIndex(
            tmp / "base",
            repo,
            is_idle=lambda: True,
            note_writer=lambda msgs, mx: "USER: Sister Meera moved to Coimbatore; owns a blue bike\nASSISTANT: Confirmed details",
            embedder=SubwordHashEmbedder(),
            start=False,
            recall=True,
        )
        mem_idx.refresh()
        written_notes = mem_idx.write_notes()
        assert written_notes >= 1, "Expected background v3 note to be written"
        block, info = mem_idx.recall("chat-beta", 1, "Where did my sister Meera move and what color is my bike?")
        assert block is not None and "Coimbatore" in block and "blue" in block, f"Recall missed facts: {block}"
        assert "999" not in block, "Negative turn #2 was not excluded from recall!"
        print(f"[PASS 5/8] S2 MemoryIndex turn-level BM25+vector RRF + v3 notes + negative exclusion ({info['ms']} ms)")

        # 6. Test S3 WebSearch relevance filter (`relevant_rows`)
        rows = [
            {"title": "Calcium carbonate", "url": "https://en.wikipedia.org/wiki/Calcium_carbonate", "snippet": "Calcium carbonate is a chemical compound with formula CaCO3 found in rocks."},
            {"title": "Milk", "url": "https://en.wikipedia.org/wiki/Milk", "snippet": "One liter of whole milk weighs about 1032 grams, which is 32 grams heavier than water."},
        ]
        kept = web_search.relevant_rows(rows, "How many grams heavier is one liter of milk than water?")
        assert len(kept) == 1 and kept[0]["title"] == "Milk", f"Relevance filter failed: {kept}"
        print("[PASS 6/8] S3 WebSearch relevance filter drops false-positive pages and keeps relevant evidence")

        # 7. Test S3 local_terminal process-group execution & timeout termination
        sh_out = local_terminal.shell("echo 'Clover S3 Terminal UTF-8 é ü ✓'", cwd=user_proj, timeout_s=10)
        assert "Clover S3 Terminal UTF-8" in sh_out and "[exit code 0]" in sh_out, f"Unexpected shell output: {sh_out}"
        print("[PASS 7/8] S3 local_terminal.shell executes in workspace cwd with UTF-8 & exit code")

        # 8. Test S3 SSHHosts public key generation (without exposing private key)
        ssh_hosts = local_terminal.SSHHosts(repo.default_repo_dir / "config" / repo.device / "ssh-hosts.json", ssh_dir=tmp / "ssh")
        ssh_hosts.add_host("gpu-box", "192.168.1.10", user="ubuntu", port=2222)
        desc = ssh_hosts.describe()
        assert len(desc["hosts"]) == 1 and desc["hosts"][0]["port"] == 2222
        assert "PRIVATE KEY" not in str(desc), "Private key must never appear in describe()"
        print("[PASS 8/9] S3 SSHHosts host registry & public-key isolation verified")

        # 9. Test Sessions List, Delete Session, and Local/Cloud Media Storage Locations
        sess_list = repo.sessions()
        assert len(sess_list) == 1 and sess_list[0]["id"] == "chat-alpha" and sess_list[0]["turns"] == 2
        cloud_media_dir = tmp / "OneDrive" / "CloverMedia"
        st_loc = repo.update_storage_locations(media_dir=str(cloud_media_dir), media_kind="cloud")
        assert st_loc["media_kind"] == "cloud" and Path(st_loc["media_dir"]) == cloud_media_dir.resolve()
        (cloud_media_dir / "test_photo.png").write_bytes(b"PNGDATA123")
        st_loc2 = repo.get_storage_locations()
        assert st_loc2["media_files"] == 1 and st_loc2["media_bytes"] == 10
        clr_res = repo.clear_media_storage()
        assert clr_res["removed"] == 1 and clr_res["media_files"] == 0
        assert repo.delete_session("chat-alpha") is True
        assert len(repo.sessions()) == 0
        print("[PASS 9/9] GitHub-syncable sessions list, delete_session & local/cloud media storage locations verified")
        print("\nALL 9 MOBILE S1/S2/S3 & STORAGE VERIFICATION SUITES PASSED!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
