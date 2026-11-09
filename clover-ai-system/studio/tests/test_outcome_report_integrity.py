"""The Outcome stage reports what was checked, and nothing it did not check (consciousness-papers integration, step 1).
Offline: no model, no engine. Usage: python tests/test_outcome_report_integrity.py"""
import json
import sys
import tempfile
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import clover_cycle as C  # noqa: E402
import clover_repo as R  # noqa: E402
import curiosity_foundation as CF  # noqa: E402
import k3_memory_graph as M  # noqa: E402
import lfm3_layers  # noqa: E402
import paths  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


class Kit:
    """compiler_check as LayerKit does it: a reply without code is not checked."""

    def compiler_check(self, q, reply):
        if not lfm3_layers.extract_python(reply).strip():
            return None, {"layer": "compiler", "checked": False}
        return None, {"layer": "compiler", "checked": True, "ok": True}


def outcome(q, a):
    findings, meta = C.verify_turn_outcome(q, a, [], kit=Kit(), layers=("compiler",))
    return findings, meta["checks"]


ok = True
f, ch = outcome("What is the capital of Australia?", "The capital of Australia is Sydney.")
ok &= check("prose: nothing found and nothing checked", f == [] and ch == [])
ok &= check("prose: the stage says no automatic check applies",
            C.outcome_label(ch) == "No automatic check applies to this answer")
f, ch = outcome("What is 17 * 23?", "17 * 23 = 391")
ok &= check("arithmetic that is right: checked and passed", f == [] and ch == ["arithmetic"]
            and C.outcome_label(ch) == "Checked: arithmetic (passed)")
f, ch = outcome("What is 17 * 23?", "17 * 23 = 401")
ok &= check("arithmetic that is wrong: a finding (the check can fail)", f and "arithmetic" in ch)
f, ch = outcome("What is the capital of Australia? Answer in one sentence.", "The capital is Canberra.")
ok &= check("one sentence asked and given: form checked", f == [] and ch == ["requested form"])
f, ch = outcome("What is the capital of Australia? Answer in one sentence.", "It is Canberra. It is in the ACT.")
ok &= check("one sentence asked, two given: a finding", f and ch == ["requested form"])
f, ch = outcome("Write a function that doubles a number.", "```python\ndef double(x):\n    return 2 * x\n```")
ok &= check("code in the answer: compiled", "code compiles" in ch)
f, ch = outcome("Run it.", "")
ok &= check("empty answer: a finding, and no check claimed", f and ch == [])
ok &= check("repaired answer: the label says in how many cycles",
            C.outcome_label(["arithmetic"], 2) == "Checked: arithmetic (passed in 2 Clover cycles)")

d = CF.derive_real_turn_outcome("What is 17 * 23?", "17 * 23 = 391",
                                layers=[{"layer": "outcome", "checks": ["arithmetic"], "passed": True}])
ok &= check("a passed check counts as verified in the curiosity record", d["verified_in_turn"] and d["comp_error"] == 0.0)
d = CF.derive_real_turn_outcome("What is the capital of Australia?", "Sydney.",
                                layers=[{"layer": "outcome", "checks": [], "passed": True}])
ok &= check("no check: not verified in the curiosity record", not d["verified_in_turn"] and d["comp_error"] is None)

base = Path(tempfile.mkdtemp())
agents = base / "AGENTS-source.md"
agents.write_text("# AGENTS.md\nThe rules.\n", encoding="utf-8")
kit = lfm3_layers.LayerKit(base, memory_root=paths.knowledge_dir(base))
kit._memory = M.CloverMemoryGraphStore(paths.knowledge_dir(base), auto_load=True)
repo = R.Repo(base, agents, kit=lambda: kit)
n, _ = repo.turn_start("s1", "What is the capital of Australia?")
repo.turn_end("s1", n, {"reply": "Canberra.", "layers": [{"layer": "outcome", "checks": [], "passed": True}]}, 1.0)
n, _ = repo.turn_start("s1", "What is 17 * 23?")
repo.turn_end("s1", n, {"reply": "17 * 23 = 391", "layers": [{"layer": "outcome", "checks": ["arithmetic"],
                                                            "passed": True}]}, 1.0)
recs = [json.loads(x) for fp in (repo.dir / "outcomes").rglob("*.jsonl") for x in fp.read_text().splitlines()]
by_n = {r["n"]: r for r in recs if r.get("session") == "s1" and "result" in r}
ok &= check("outcomes/ records that nothing was checked", {"check": "outcome", "applied": []} in by_n[1].get("checks", []))
ok &= check("outcomes/ records the check that passed",
            {"check": "outcome", "applied": ["arithmetic"], "passed": True} in by_n[2].get("checks", []))
repo.flush()

texts = {f: (CLOVER / f).read_text(encoding="utf-8")
         for f in ("omni_native.py", "lfm3_native.py", "k3_host_engine.py", "fused_engine.py")}
ok &= check("no engine writes 'Outcome verified' or 'verified outcome' any more",
            not any(("Outcome verified" in t or "verified outcome" in t) for t in texts.values()))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
