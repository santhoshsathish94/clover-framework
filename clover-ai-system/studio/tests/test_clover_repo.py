"""Offline checks of Clover's repository (no model, no engine): moving the old layout in, the records of a turn in
sessions/, execution/ and outcomes/, feedback read from the next message, growth (learn, score, contest, correct),
retrieval of only what a message needs, secret redaction, AGENTS.md protection, the summaries and the commits.
Usage: python tests/test_clover_repo.py"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import clover_repo as R  # noqa: E402
import k3_memory_graph as M  # noqa: E402
import lfm3_layers  # noqa: E402
import paths  # noqa: E402
import vault  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
base = Path(tempfile.mkdtemp())
old = base / "config"
old.mkdir()
for name, data in (("settings.json", {"model": "omni", "engine": "gpu", "tools": "always"}),
                   ("permissions.json", {"shell": "allow"}), ("workspaces.json", {"workspaces": [], "active": None}),
                   ("search.json", {"provider": "tavily", "keys": {"tavily": "tvly-" + "k" * 24}}),
                   ("github.json", {"token_dpapi": None}), ("vscode-bridge.json", {"token": "b" * 64})):
    (old / name).write_text(json.dumps(data), encoding="utf-8")
(base / "cache" / "media").mkdir(parents=True)
(base / "cache" / "media" / "photo.png").write_bytes(b"png")
moved = R.migrate(base)
cfg, vdir = paths.config_dir(base), paths.vault_dir(base)
ok &= check("settings, permissions, workspaces moved into the repository config",
            all((cfg / n).is_file() for n in ("settings.json", "permissions.json", "workspaces.json")))
ok &= check("tokens moved to the vault", (vdir / "github.json").is_file() and (vdir / "vscode-bridge.json").is_file())
ok &= check("search key moved to the vault, encrypted; provider stays in config",
            json.loads((cfg / "search.json").read_text()) == {"provider": "tavily"}
            and json.loads(vault.unprotect(json.loads((vdir / "search-keys.json").read_text())["keys_dpapi"]))["tavily"].startswith("tvly-"))
ok &= check("media moved out of the cache", (base / "media" / "photo.png").is_file() and not old.exists())
ok &= check("migration reports what moved", len(moved) >= 7)

agents = base / "AGENTS-source.md"
agents.write_text("# AGENTS.md\nThe rules.\n", encoding="utf-8")
kit = lfm3_layers.LayerKit(base, memory_root=paths.knowledge_dir(base))
kit._memory = M.CloverMemoryGraphStore(paths.knowledge_dir(base), auto_load=True)  # no clover_loop (torch) here
repo = R.Repo(base, agents, kit=lambda: kit)
repo.flush()
rd = repo.dir
ok &= check("git repository with protected AGENTS.md", (rd / ".git").is_dir()
            and (rd / "AGENTS.md").read_bytes() == agents.read_bytes() and not os.stat(rd / "AGENTS.md").st_mode & stat.S_IWRITE)
ok &= check("CONTEXT.md at the top and in every folder", all((rd / f / "CONTEXT.md").is_file()
            for f in ("", "sessions", "execution", "outcomes", "growth", "config")))

n, ch = repo.turn_start("s1", "What is 17 times 23?")
repo.event("s1", n, {"type": "tool_call", "name": "run_python", "arguments": {"code": "print(17*23)"}})
repo.event("s1", n, {"type": "tool_result", "name": "run_python", "ran": True, "why": "allowed in settings", "result": "391"})
repo.turn_end("s1", n, {"reply": "17 x 23 = 391", "rounds": 2, "layers": [{"layer": "compiler", "checked": False}],
                        "tool_calls": [{"tool": "run_python", "ran": True, "result": "391"}]}, 3.2)
msgs = repo.messages("s1")
ok &= check("session holds the input and the answer only",
            [(m["n"], m["role"], m["text"]) for m in msgs] == [(1, "user", "What is 17 times 23?"), (1, "clover", "17 x 23 = 391")]
            and set(msgs[0]) == {"n", "role", "at", "text"})

lesson = 'On a question like "What is 17 times 23?"; the user said the first answer was not right: "check it again".'
n, ch = repo.turn_start("s1", "That is not right, check it again", retry="said")
repo.turn_end("s1", n, {"reply": "Checked again: 391.", "layers": [{"layer": "cycle", "action": "kept", "lesson": lesson,
                                                                   "lesson_kind": "user"}]}, 9.0)
recs = [json.loads(x) for f in (rd / "outcomes").rglob("*.jsonl") for x in f.read_text().splitlines()]
fb1 = [r for r in recs if r.get("n") == 1 and "feedback" in r]
learn = [e for e in kit.store().entries.values() if "learning" in e.tags]
ok &= check("'not right' recorded as feedback that did not work", len(fb1) == 1 and fb1[0]["feedback"] == "said_not_right"
            and fb1[0]["worked"] is False and fb1[0]["by"] == 2)
ok &= check("the correction became a learning, linked from the outcome", len(learn) == 1
            and any(r.get("n") == 2 and r.get("lesson") == learn[0].id for r in recs))

n, ch = repo.turn_start("s1", "Thanks. Now what is 12 squared?")
ok &= check("continuing counts as worked for the learning", any("worked (worked 1, did not work 0)" in c for c in ch)
            and kit.tally(kit.store().entries[learn[0].id]) == (1, 0))
repo.turn_end("s1", n, {"reply": "144"}, 1.0)
ok &= check("Helpful recorded once per turn", repo.feedback("s1", 3, "helpful") == [] and
            sum(1 for f in (rd / "outcomes").rglob("*.jsonl") for x in f.read_text().splitlines()
                if '"helpful"' in x) == 1 and repo.feedback("s1", 3, "helpful") == [])

lid = learn[0].id
kit.score_learning(lid, False, "test")
c2 = kit.score_learning(lid, False, "test")
ok &= check("a learning that fails more than it works becomes contested", "now contested" in c2
            and kit.store().entries[lid].status == "contested")
new_id, ch = kit.learn(lesson.replace("check it again", "check the multiplication again"), "s9", 1)
ok &= check("a later correction corrects the contested learning (ontology)",
            any(r.type == "corrects" and r.target == lid for r in kit.store().entries[new_id].relationships)
            and kit.store().entries[lid].status == "superseded")

repo.turn_start("s2", "Something new")
blk, info = repo.context_block("What did we talk about last time?")
ok &= check("an earlier conversation is retrieved when the message refers to it",
            blk and "17 times 23" in blk and info["parts"] == ["sessions"])
blk, info = repo.context_block("What have you learned so far?")
ok &= check("the growth summary when asked", blk and new_id in blk and info["parts"] == ["growth"])
ok &= check("nothing for an unrelated question", repo.context_block("What is 2 + 2?") == (None, None))
repo.turn_end("s2", 1, {"reply": "ok"}, 0.5)

token = "ghp_" + "A" * 36
repo.turn_start("s3", f"my token is {token}")
log = "".join(f.read_text() for f in (rd / "execution").rglob("*.log"))
ok &= check("secrets never reach the repository", token not in (rd / "sessions").joinpath(
    repo._index["s3"].relative_to(rd / "sessions")).read_text() and "redacted before saving: GitHub token" in log)
repo.turn_error("s3", 1, "RuntimeError: engine stopped")
n, ch = repo.turn_start("s3", "hello again")
recs = [json.loads(x) for f in (rd / "outcomes").rglob("*.jsonl") for x in f.read_text().splitlines()]
ok &= check("an error turn is recorded and not judged",
            any(r.get("session") == "s3" and r.get("result") == "error" for r in recs)
            and not any(r.get("session") == "s3" and "feedback" in r for r in recs))

os.chmod(rd / "AGENTS.md", stat.S_IWRITE | stat.S_IREAD)
(rd / "AGENTS.md").write_text("changed", encoding="utf-8")
ok &= check("a changed AGENTS.md is restored", repo.protect_agents() == "restored"
            and (rd / "AGENTS.md").read_bytes() == agents.read_bytes())
ok &= check("the model's file tools cannot write the repository or read the vault",
            paths.write_blocked(base, rd / "AGENTS.md") and paths.write_blocked(base, rd / "outcomes" / "x")
            and paths.read_blocked(base, vdir / "github.json") and not paths.write_blocked(base, base / "notes.txt"))
ok &= check("deleting a conversation removes its file", repo.delete_session("s2") and "s2" not in repo._index)
held = open(repo._index["s3"], "rb")  # Windows: an open handle blocks deletion, as a scanner's does
threading.Timer(0.4, held.close).start()
ok &= check("a conversation held open for a moment is still deleted", repo.delete_session("s3") and "s3" not in repo._index)
repo.turn_start("s4", "kept while locked")
held = open(repo._index["s4"], "rb")
try:
    repo.delete_session("s4")
    locked = False
except PermissionError:
    locked = True
held.close()
ok &= check("a conversation that stays locked stays listed and deletes later",
            locked and "s4" in repo._index and repo.delete_session("s4") and "s4" not in repo._index)

repo.flush()
git = lambda *a: subprocess.run(["git", *a], cwd=rd, capture_output=True, text=True).stdout  # noqa: E731
summary = {f: (rd / f / "CONTEXT.md").read_text(encoding="utf-8") for f in ("sessions", "outcomes", "growth", "execution")}
ok &= check("summaries reflect the records", "17 times 23" in summary["sessions"] and "did not work" in summary["outcomes"]
            and new_id in summary["growth"] and "corrects" in summary["growth"] and "answers" in summary["execution"])
ok &= check("everything committed; the graph index stays local", int(git("rev-list", "--count", "HEAD").strip() or 0) >= 2
            and git("status", "--porcelain").strip() == "" and "_index" not in git("ls-files"))
ok &= check("no secret anywhere in the repository", token not in "".join(
    p.read_text(encoding="utf-8", errors="replace") for p in rd.rglob("*") if p.is_file() and ".git" not in p.parts))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
