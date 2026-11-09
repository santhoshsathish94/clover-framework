"""Item 2 (consciousness integration, step 4 follow-up): a question's literal calls of a function an externally verified
memory note recorded are computed by running the verified copy in IsolatedPythonSandbox, and the note's wording no
longer says "answer from them" or calls verified notes unverified. Usage: python tests/test_verified_function_results.py"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import k3_memory_graph as M  # noqa: E402
import lfm3_layers  # noqa: E402

RULE = ("def effective_throughput(total_slots: int, stages: int, partitioned: bool) -> float:\n"
        "    if total_slots <= 0 or stages <= 0:\n        return 0.0\n"
        "    return float(total_slots // stages) if partitioned else 1.0\n")
SUMMARY = ("Verified workspace function effective_throughput(total_slots, stages, partitioned): if total_slots <= 0 or "
           "stages <= 0: return 0.0; return float(total_slots // stages) if partitioned else 1.0 | Verified behavioral & "
           "boundary cases: effective_throughput(512, 4, True) == 128.0; effective_throughput(256, 8, False) == 1.0")


def content(code):
    return (f"Verified in IsolatedPythonSandbox (4 checks).\nSummary: {SUMMARY}\n```python\n{code}\n"
            "assert round(effective_throughput(512, 4, True), 6) == 128.0\n"
            "RESULT = {'passed': True, 'function': 'effective_throughput', 'verified_cases': 4}\n```")


def entry(code=RULE, status="externally_verified"):
    return SimpleNamespace(id="MEM-HYP-WS-TEST", summary=SUMMARY, content=content(code), epistemic_status=status)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


class NoPython:
    def run(self, code):
        raise OSError("no runnable Python")


Q_NEW = ("In our ring_buffer module, what does `effective_throughput(100, 3, True)` return, and what does "
         "`effective_throughput(5, 0, True)` return? State both exact numbers.")
ok = True
got = lfm3_layers.run_verified_calls(entry(), Q_NEW)
ok &= check("literal calls are run on the verified copy (33.0 is not among the note's cases)",
            got == [("effective_throughput(100, 3, True)", "= 33.0"), ("effective_throughput(5, 0, True)", "= 0.0")])
ok &= check("liveness: a different stored body gives a different result (the value comes from running it)",
            lfm3_layers.run_verified_calls(entry(RULE.replace("//", "/")), Q_NEW)[0][1] == "= 33.33333333")
ok &= check("an unverified note is not run", lfm3_layers.run_verified_calls(entry(status="hypothesis"), Q_NEW) == [])
ok &= check("arguments that are not literals are not run",
            lfm3_layers.run_verified_calls(entry(), "What does effective_throughput(n, 3, True) return?") == []
            and lfm3_layers.run_verified_calls(entry(), "effective_throughput(open, 3, True)?") == []
            and lfm3_layers.run_verified_calls(entry(), "effective_throughput(len('x'), 3, True)?") == [])
raised = lfm3_layers.run_verified_calls(entry(), "What does effective_throughput('a', 3, True) return?")
ok &= check("a call that raises is reported as raising", len(raised) == 1 and raised[0][1].startswith("raises TypeError"))
ok &= check("no runnable Python for the child: nothing is added",
            lfm3_layers.run_verified_calls(entry(), Q_NEW, sandbox=NoPython()) == [])
helper = RULE.replace("return 0.0", "return floor_value()")
ok &= check("a NameError (the copy lacks a helper the module has): nothing is added",
            lfm3_layers.run_verified_calls(entry(helper), "effective_throughput(0, 3, True)?") == [])
ok &= check("a question without a call adds nothing",
            lfm3_layers.run_verified_calls(entry(), "How does effective_throughput behave at the boundaries?") == [])

base = Path(tempfile.mkdtemp(prefix="verified-calls-"))
kit = lfm3_layers.LayerKit(base, memory_root=base / "growth" / "knowledge")
kit.store().create_entry(M.MemoryEntry(
    id="MEM-HYP-WS-TEST", title="Workspace Hypothesis: ring_buffer.py:effective_throughput", entry_type="hypothesis",
    epistemic_status="externally_verified", summary=SUMMARY, content=content(RULE),
    source_evidence="Inspected 1 workspace file: found ring_buffer.py:effective_throughput",
    provenance=M.MemoryProvenance(source_type="observation", source_ref="workspace:ws_a", actor="clover-curiosity"),
    concepts=["ring_buffer", "effective_throughput", "module"], tags=["ring_buffer", "effective_throughput", "module"]))
block, info = kit.memory_context(Q_NEW)
ok &= check("memory turn: the note carries the computed results",
            block is not None and "Computed now by running that verified function: effective_throughput(100, 3, True) = 33.0; "
            "effective_throughput(5, 0, True) = 0.0" in block)
ok &= check("memory turn: the layer info records what was computed", [c["call"] for c in (info or {}).get("computed", [])]
            == ["effective_throughput(100, 3, True)", "effective_throughput(5, 0, True)"])
ok &= check("wording: still 'no tools needed', no longer 'answer from them'",
            block is not None and "(no tools needed)" in block and "answer from them" not in block)
ok &= check("wording: verified notes are not called unverified",
            "not verified facts" not in lfm3_layers.APP_MEMORY_INSTRUCTION
            and "only a note marked externally verified was checked" in lfm3_layers.APP_MEMORY_INSTRUCTION)
block2, info2 = kit.memory_context("In our ring_buffer module, how does effective_throughput behave when stages is zero?")
ok &= check("memory turn without a call: no computed line", block2 is not None and "Computed now" not in block2
            and "computed" not in (info2 or {}))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
