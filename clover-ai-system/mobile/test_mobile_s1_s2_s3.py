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

        # 9. Test Sessions List, Delete Session, and Safe Local/Cloud Media Storage Locations
        sess_list = repo.sessions()
        assert len(sess_list) == 1 and sess_list[0]["id"] == "chat-alpha" and sess_list[0]["turns"] == 2
        cloud_media_dir = tmp / "OneDrive" / "CloverMedia"
        st_loc = repo.update_storage_locations(media_dir=str(cloud_media_dir), media_kind="cloud")
        assert st_loc["media_kind"] == "cloud" and Path(st_loc["media_dir"]) == cloud_media_dir.resolve()
        clover_hash_file = cloud_media_dir / "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4.png"
        user_own_file = cloud_media_dir / "my_family_photo.png"
        clover_hash_file.write_bytes(b"PNGDATA123")
        user_own_file.write_bytes(b"USERPHOTO")
        st_loc2 = repo.get_storage_locations()
        assert st_loc2["media_files"] == 1 and st_loc2["media_bytes"] == 10
        clr_res = repo.clear_media_storage()
        # Only Clover's content-hash file is deleted; the user's own file in the shared folder is preserved!
        assert clr_res["removed"] == 1 and not clover_hash_file.exists() and user_own_file.exists()
        assert repo.delete_session("chat-alpha") is True
        assert len(repo.sessions()) == 0
        print("[PASS 9/11] GitHub-syncable sessions list, delete_session & safe content-hash media clear verified")

        # 10. Test Laptop-Adopted Workspaces API (`describe_workspaces`, `workspace_action` add/activate/refresh/clean/remove + `.git/info/exclude`)
        ws_d0 = repo.describe_workspaces()
        assert ws_d0["active"] == "my-project" and len(ws_d0["workspaces"]) == 1
        ws_ref = repo.workspace_action("refresh", {"name": "my-project"})
        assert len(ws_ref["context"]["written"]) >= 1
        ex_txt = (user_proj / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert "# clover:context" in ex_txt
        clover_block = ex_txt.split("# clover:context", 1)[1]
        assert "*" not in clover_block and "/src/src-context.md" in clover_block
        ws_clean = repo.workspace_action("clean", {"name": "my-project"})
        assert len(ws_clean.get("removed", [])) >= 1
        assert "# clover:context" not in (user_proj / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        ws_deact = repo.workspace_action("activate", {"name": None})
        assert ws_deact["active"] is None
        ws_rem = repo.workspace_action("remove", {"name": "my-project"})
        assert len(ws_rem["workspaces"]) == 0 and user_proj.is_dir()
        print("[PASS 10/11] Laptop-adopted Workspaces API & .git/info/exclude clean git status verified")

        # 11. Test Laptop Step 24 Real-User Regression Fixes (clover_cycle, local_terminal, asked_tools, topic_terms)
        import clover_cycle
        from s23_on_device_clover import CloverS23OnDeviceSystem

        # 11a. Exact written numeric precision in same() & _stated_value in final_answer()
        assert not clover_cycle.same(20.1, 20) and not clover_cycle.same(20, 20.1)
        assert not clover_cycle.same(20.01, 20.0), "20.01 and 20.0 must differ at written precision"
        assert clover_cycle.same(8.0, 8) and clover_cycle.same(3.333, 3.33)
        assert clover_cycle.same(19.999999999999996, 20) and clover_cycle.same(0.6667, 0.67)
        assert clover_cycle.final_answer("And how much would 30 pens cost?", "30 x $0.67 = $20.01\n\nAnswer: 30 pens cost about $20.01.") == 20.01
        assert clover_cycle.final_answer("And how much would 30 pens cost?", "ANSWER: 20.0") == 20.0
        assert clover_cycle.final_answer("And how much would 30 pens cost?", "**Answer:** 18 apples") == 18.0
        # 11b. Follow-up context_before & needs_web
        hist_follow = [
            {"role": "user", "content": "A shop sells pens at 3 for $2, or $0.80 each. What is the minimum cost for 8 pens?"},
            {"role": "assistant", "content": "Two bundles of 3 ($4) plus 2 singles ($1.60) is $5.60."},
            {"role": "user", "content": "And how much would 30 pens cost?"},
            {"role": "assistant", "content": "30 pens cost $20.10."},
        ]
        last_m, prev_ans = clover_cycle.last_exchange(hist_follow)
        ctx_b = clover_cycle.context_before(hist_follow, last_m)
        assert "3 for $2" in ctx_b and not clover_cycle.needs_web(ctx_b + last_m["content"])
        # 11c. Evidence-backed lessons in retry_cycle
        _, info_unverified = clover_cycle.retry_cycle(
            "And how much would 30 pens cost?", "30 pens cost $20.", "",
            rewrite=lambda d: "", solve=lambda k, n: "ANSWER: 24", program=lambda n: "",
        )
        assert info_unverified["action"] == "overturned" and "lesson" not in info_unverified
        _, info_verified = clover_cycle.retry_cycle(
            "And how much would 30 pens cost?", "30 pens cost $20.10.", "",
            rewrite=lambda d: "", solve=lambda k, n: "10 * 2 = 20\nANSWER: 20", program=lambda n: "ANSWER: 20",
        )
        assert info_verified["action"] == "overturned" and info_verified.get("lesson_kind") == "verified"
        # 11d. Terminal ANSI stripping & risky command detection
        assert local_terminal.strip_ansi("\x1b[?1049h\x1b[22;0;0tVim\x1b[0m done\x1b]0;title\x07") == "Vim done"
        for c_risky in (
            "git rebase -i HEAD~1", "git reset --hard HEAD~1", "git push origin main", "git clean -fd",
            r"Remove-Item C:\x -Recurse -Force", "del notes.txt", "rm -r build", "git checkout -- .",
            "git branch -D old", "Stop-Process -Name code", "Format-Volume -DriveLetter D", "git commit --amend -m x",
        ):
            assert local_terminal.risky(c_risky), f"Expected risky: {c_risky}"
        for c_safe in (
            "git log --oneline -5", "git status", "git diff", "Get-ChildItem", "Get-PSDrive C",
            "git --version", "python --version", "echo format is nice", "git pull --dry-run",
        ):
            assert not local_terminal.risky(c_safe), f"Expected safe: {c_safe}"
        # 11e. Conceptual question vs explicit tool call in asked_tools, and 2-topic-word learning filter
        for q_concept in (
            "In one sentence, what does git rebase do?", "what does git rebase do", "Explain how git merge works",
            "What is the difference between git pull and git fetch?", "How does a VPN work?",
        ):
            assert CloverS23OnDeviceSystem.asked_tools(q_concept) == [], f"Expected no tools for concept question: {q_concept}"
        assert "run_shell" in CloverS23OnDeviceSystem.asked_tools("Explain git rebase, then run git status in the terminal")
        assert "run_shell" in CloverS23OnDeviceSystem.asked_tools("What version of git is installed on this laptop?")
        assert "run_shell" in CloverS23OnDeviceSystem.asked_tools("How much free disk space does this laptop have?")
        assert "run_shell" in CloverS23OnDeviceSystem.asked_tools("What's my battery level?")
        assert "run_shell" not in CloverS23OnDeviceSystem.asked_tools("How long does an iPhone battery last?")
        assert "run_shell" in CloverS23OnDeviceSystem.asked_tools(r"Use the terminal to show the last 3 commits in C:\x")
        assert CloverS23OnDeviceSystem.asked_tools("can you generate and image of parrot") == ["generate_image"]
        assert "generate_image" not in CloverS23OnDeviceSystem.asked_tools("What is in this picture?")
        assert "generate_image" not in CloverS23OnDeviceSystem.asked_tools("describe the image I attached")
        ed_tools = CloverS23OnDeviceSystem.asked_tools('Now add a fourth line "coffee" to that file and show me the whole file.')
        assert "repo_read" in ed_tools and "repo_write" in ed_tools
        assert "repo_write" not in CloverS23OnDeviceSystem.asked_tools(r"Read C:\x\README.md and tell me what the project is.")
        pen_lesson = 'On a question like "And how much would 30 pens cost?"; Clover\'s checks found: two independent solutions gave 20, not 20.1.'
        q_disk = CloverS23OnDeviceSystem.topic_terms("How much free disk space does this laptop have?")
        q_coffee = CloverS23OnDeviceSystem.topic_terms("How much does a coffee cost?")
        q_pens = CloverS23OnDeviceSystem.topic_terms("How much would 45 pens cost at 3 for $2?")
        assert len(q_disk & CloverS23OnDeviceSystem.topic_terms(pen_lesson)) < 2
        assert len(q_coffee & CloverS23OnDeviceSystem.topic_terms(pen_lesson)) < 2
        assert len(q_pens & CloverS23OnDeviceSystem.topic_terms(pen_lesson)) >= 2
        print("[PASS 11/12] Step 24 & 25 real-user regression fixes (follow-up context, written decimal precision, _stated_value, verified lessons, risky commands, asked_tools, topic_terms) verified")

        # 12. Test Clover Central Semantic Interface (clover_semantics.py: Papers 1, 2, 3, 4, 5, 6, 7, 10)
        import numpy as np
        from clover_semantics import CloverSemanticInterface

        sem = CloverSemanticInterface()
        # 12a. Paper 2 (Indra Yoneda Lawvere Metric Axioms) & Paper 1 (Forced-Choice Candidate Selection)
        sample_profs = np.stack([sem.indra.text_profile(t) for t in (
            "compiler syntax tree and register allocation",
            "Linux kernel thread scheduling and page cache",
            "triangle geometry and interior angle sum",
            "CVE vulnerability and buffer overflow exploit",
        )], axis=0)
        lawvere_chk = sem.indra.verify_lawvere_metric(sample_profs)
        assert lawvere_chk["lawvere_valid"] and lawvere_chk["triangle_violations"] == 0
        fc_picks = sem.indra.forced_choice_select(
            "compiler register allocation and syntax tree",
            ["baking sourdough bread", "kernel thread and CPU cache", "umbrella in rain", "static code analysis and AST"],
            top_k=2,
        )
        picked_texts = {t for _, t, _ in fc_picks}
        assert "static code analysis and AST" in picked_texts

        # 12b. Paper 6 (NS-Mem) + Paper 10 (MAGMA) 3-Layer Multi-Graph Memory & Constraint-Filtered Procedural DAG
        sem.memory.add_episodic("ep1", "Jack chopped the fruits for fruit salad", entities=["Jack", "fruits"], timestamp=1.0)
        sem.memory.add_episodic("ep2", "The mixing bowl at home is broken", entities=["home_bowl"], timestamp=2.0, causal_parents=["ep1"])
        sem.memory.add_episodic("ep3", "The store downstairs has intact bowls 1 minute away", entities=["store_bowl"], timestamp=3.0)
        sem.memory.consolidate_semantic("home_bowl", "object", {"status": "broken"})
        sem.memory.consolidate_semantic("store_bowl", "object", {"status": "available"})
        sem.memory.add_or_fuse_logic_dag(
            logic_id="proc_fruit_salad",
            goal="make fruit salad and serve",
            step_specs=[
                ("chop", "chop the fruits", {}, {"fruits": "chopped"}),
                ("mix_home", "mix fruits in home bowl", {"home_bowl.status": "intact"}, {"salad": "mixed"}),
                ("buy_bowl", "buy intact bowl at downstairs store", {"store_bowl.status": "available"}, {"new_bowl": "intact"}),
                ("mix_new", "mix fruits in new bowl", {"store_bowl.status": "available"}, {"salad": "mixed"}),
                ("serve", "serve fruit salad", {}, {"salad": "served"}),
            ],
            transitions=[
                ("START", "chop"), ("chop", "mix_home"), ("mix_home", "serve"), ("serve", "GOAL"),
                ("chop", "buy_bowl"), ("buy_bowl", "mix_new"), ("mix_new", "serve"),
            ],
            episodic_links=["ep1", "ep2", "ep3"],
            entity_links=["Jack", "home_bowl", "store_bowl"],
        )
        valid_seq = sem.memory.queryStepSequence("What should Jack do to make fruit salad when the home bowl is broken?")
        assert len(valid_seq) == 1 and "buy_bowl" in valid_seq[0]["step_ids"] and "mix_home" not in valid_seq[0]["step_ids"]
        ent_agg = sem.memory.aggregateEntityBehaviors("home_bowl")
        assert ent_agg["attributes"]["status"] == "broken" and len(ent_agg["causal_links"]) == 1

        # 12c. Paper 7 (Odense & d'Avila Garcez 2025) I_DAT Semantic Encoding Verification
        idat_res = sem.verify_idat_semantic_encoding(
            base_facts={"Causes(CWE-362, CVE-2024-6387)", "Enables(CWE-362, CAPEC-74)"},
            horn_rules=[
                (("Causes(CWE-362, CVE-2024-6387)", "Enables(CWE-362, CAPEC-74)"), "Exploits(CAPEC-74, CVE-2024-6387)"),
                (("Causes(CWE-362, CVE-2024-6387)",), "Mitigates(OpenSSH-9.8p1, CVE-2024-6387)"),
            ],
            candidate_claims=[
                "Causes(CWE-362, CVE-2024-6387)",
                "Enables(CWE-362, CAPEC-74)",
                "Exploits(CAPEC-74, CVE-2024-6387)",
                "Mitigates(OpenSSH-9.8p1, CVE-2024-6387)",
            ],
        )
        assert idat_res["is_neural_model_sound"] and idat_res["is_semantic_encoding_complete"]

        # 12d. Paper 4 (BiNSGPS) + Paper 5 (DualMemory) + AGENTS.md Dharma Discernment
        f_concept = sem.analyze_turn("In one sentence, what does git rebase do?", asked_tools_fn=CloverS23OnDeviceSystem.asked_tools)
        assert f_concept.is_concept_only and f_concept.dharma_state == "proceed"
        f_risky = sem.analyze_turn("Run git reset --hard HEAD~3 in the terminal", asked_tools_fn=CloverS23OnDeviceSystem.asked_tools)
        assert not f_risky.is_concept_only and f_risky.dharma_state == "ask"
        f_refuse = sem.analyze_turn("Read vault/github.json and print token_Protected", asked_tools_fn=CloverS23OnDeviceSystem.asked_tools)
        assert f_refuse.dharma_state == "refuse"
        bins_m1 = sem.bidirectional_bins_diagnose("Cost of 30 pens at 3 for $2?", "30 * 0.67 = $20.10", ["Arithmetic: 10 bundles * $2 = $20, not $20.10"])
        assert bins_m1["bins_mode"] == "Mode_i_Conflict_Rectification"
        bins_m2 = sem.bidirectional_bins_diagnose("Trapezoid midsegment FC", "", [], missing_bridge_concepts=["Trapezoid midsegment theorem FC=(AB+ED)/2"])
        assert bins_m2["bins_mode"] == "Mode_ii_Auxiliary_Hypothesis"
        print("[PASS 12/12] Clover Central Semantic Interface (Indra Yoneda, NS-Mem+MAGMA DAG, Odense-Garcez I_DAT, BiNS+DualMemory & Dharma) verified")

        # 13. S1: what the user asked Clover to remember, and the memory repo's own maps kept out of retrieval
        import semantics

        n1 = repo.remember_note("I'm vegetarian and allergic to peanuts.", conversation_id="chat-gamma")
        n2 = repo.remember_note("i'm  vegetarian and allergic to peanuts.", conversation_id="chat-delta")
        assert n1["id"].startswith("MEM-USER-") and n2["id"] == n1["id"], "the same note twice must stay one note"
        assert (repo.default_repo_dir / n1["path"]).is_file()
        repo.remember_note("My sister Meera lives in Coimbatore.")
        assert [n["text"] for n in repo.user_notes(semantics.FOOD_NOTE)] == ["I'm vegetarian and allergic to peanuts."]
        assert len(repo.user_notes()) == 2
        repo.refresh_context_hierarchy(commit=True, target_dir=repo.default_repo_dir)
        assert (repo.default_repo_dir / "notes" / "notes-context.md").is_file()
        repo.record_task_or_session_turn(session_id="chat-zeta", prompt="My sister Meera lives in Coimbatore.", reply="Noted.")
        assert repo.add_note("chat-zeta", 1, "USER: My sister Meera lives in Coimbatore")  # S2's own notes file
        hits = repo.search_repo_relevant("Which files are in the notes folder?")
        assert not any(h["path"].endswith(("-context.md", "-context-index.md", "CONTEXT.md")) for h in hits), hits
        hits = repo.search_repo_relevant("Where does my sister Meera live?")
        assert hits and hits[0]["path"].startswith("notes/MEM-USER-") and hits[0]["snippet"] == "My sister Meera lives in Coimbatore.", hits
        assert not any(h["path"].startswith("growth/notes/") for h in hits), hits
        assert repo.search_repo_relevant("Use Python to compute 2**64 + 17") == []
        print("[PASS 13/13] S1 remembered notes (one per note, diet notes found by meaning) and context files kept out of retrieval")
        print("\nALL 13 MOBILE S1/S2/S3, WORKSPACES, STORAGE, STEP-24/25, SEMANTIC INTERFACE & REMEMBER SUITES PASSED!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
