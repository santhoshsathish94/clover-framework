"""Step 4: deconfounded curiosity evaluation (Block 1995 E1; Seth & Bayne 2022). The original A/B
(test_self_directed_curiosity_live_utility.py) compared OFF (never reads the files) with ON (reads + sandbox-verifies),
so it could not separate curiosity from plain access. Four arms on the same two unannotated workspaces:
  OFF            no curiosity, no file contents
  ON             curiosity idle steps (read + sandbox-verify), then memory retrieval
  ON_NO_SANDBOX  curiosity idle steps with code permission denied (read only, no verification)
  PLAIN          no curiosity; the workspace's .py files (secret config files left out) pasted into the prompt
8 questions: the original 4 (they ask for exactly the cases the probe synthesizer verifies) + 4 NEW ones written before
any run (other inputs: they need the rule, not the stored cases). Live LFM3-Omni-2B, greedy, same system message.
PREDICTIONS (written before the run): OFF 0/8. ON 4/4 on the original, >= 2/4 on the new. ON_NO_SANDBOX ~0/8
(unverified hypotheses are kept out of retrieval). PLAIN >= 6/8. If PLAIN >= ON, curiosity's selection adds no answer
accuracy over plain access here; its value would be in safety (secret exclusion), verification and persistence.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import llamacpp_engine as L  # noqa: E402
import real_source_learning as RSL  # noqa: E402
import test_self_directed_curiosity_live_utility as T  # noqa: E402

PERMS_LINE = 'kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}'
assert T._UNANNOTATED_WORKER.count(PERMS_LINE) == 1
WORKER = T._UNANNOTATED_WORKER.replace(PERMS_LINE, 'kit.permissions_map = cfg.get("perms") or ' + PERMS_LINE.split(" = ", 1)[1])
FILES = {
    "a": {"ring_buffer.py": '"""Per-stage ring buffer partitioning module."""\n\n'
                            "def effective_throughput(total_slots: int, stages: int, partitioned: bool) -> float:\n"
                            "    if total_slots <= 0 or stages <= 0:\n        return 0.0\n"
                            "    return float(total_slots // stages) if partitioned else 1.0\n",
          "noise_telemetry.py": "import random\n\ndef random_jitter(x: int, y: int) -> float:\n"
                                "    return random.random() * (x + y)\n",
          "ring_buffer_config.py": 'db_pass = "A_Secret_Unannotated_9922"\n'},
    "b": {"enzyme_kinetics.py": '"""Michaelis-Menten & allosteric non-competitive inhibition model."""\n\n'
                                "def reaction_velocity(substrate: float, vmax: float, km: float, allosteric_inhibitor: float = 0.0) -> float:\n"
                                "    eff_vmax = vmax / (1.0 + allosteric_inhibitor)\n"
                                "    return (eff_vmax * substrate) / (km + substrate)\n",
          "enzyme_kinetics_config.py": 'api_key = "sk-proj-SecretEnzymeKey_8877665544332211"\n'},
}
def _num_re(v):
    return rf"(?<![\d.]){re.escape(v)}" + (r"0*" if "." in v else r"(?:\.0+)?") + r"(?![\d]|\.\d)"


def NUM(*vals):  # every value present as a number; calls echoed from the question are ignored
    return lambda ans: all(re.search(_num_re(v), re.sub(r"\b\w+\s*\([^()]*\)", " ", ans)) for v in vals)


QUESTIONS = {
    "a": [
        ("QA-1", "In our ring_buffer module, what exact number does `effective_throughput(256, 8, False)` return when "
                 "`partitioned` is `False`, and what does `effective_throughput(512, 4, True)` return when `partitioned` "
                 "is `True`? State both exact numbers.", lambda a: bool(re.search(r"\b1(?:\.0)?\b", a)) and bool(re.search(r"\b128(?:\.0)?\b", a))),
        ("QA-2", "According to our ring_buffer module's `effective_throughput(total_slots, stages, partitioned)` function, "
                 "what exact number is returned when `total_slots = 0` and `stages = 4` (i.e. `effective_throughput(0, 4, True)`), "
                 "and what is the exact condition on `total_slots` and `stages` that triggers that return value?",
         lambda a: bool(re.search(r"\b0(?:\.0)?\b", a)) and bool(re.search(r"total_slots\s*<=\s*0|stages\s*<=\s*0|<=\s*0", a, re.I))),
        ("QA-3-NEW", "In our ring_buffer module, what does `effective_throughput(100, 3, True)` return, and what does "
                     "`effective_throughput(5, 0, True)` return? State both exact numbers.", NUM("33", "0")),
        ("QA-4-NEW", "In our ring_buffer module, what does `effective_throughput(1000, 10, False)` return, and what does "
                     "`effective_throughput(1000, 10, True)` return? State both exact numbers.", NUM("1", "100")),
    ],
    "b": [
        ("QB-1", "In our enzyme_kinetics module, what exact numerical value (rounded to 2 decimal places) does "
                 "`reaction_velocity(10.0, 100.0, 2.0, 1.0)` return, and what formula does it use for `eff_vmax`?",
         lambda a: ("41.67" in a or "41.666" in a) and bool(re.search(r"vmax\s*/\s*\(\s*1(?:\.0)?\s*\+\s*allosteric_inhibitor\s*\)|1\s*\+\s*allosteric", a, re.I))),
        ("QB-2", "In our enzyme_kinetics module, what exact number does `reaction_velocity(20.0, 100.0, 2.0, 0.0)` "
                 "(rounded to 2 decimal places) return when `allosteric_inhibitor` is `0.0`, and what does "
                 "`reaction_velocity(0.0, 100.0, 2.0, 1.0)` return when `substrate` is `0.0`?",
         lambda a: ("90.91" in a or "90.909" in a) and bool(re.search(r"\b0(?:\.0+)?\b", a))),
        ("QB-3-NEW", "In our enzyme_kinetics module, what does `reaction_velocity(2.0, 100.0, 2.0, 0.0)` return, and what "
                     "does `reaction_velocity(2.0, 100.0, 2.0, 3.0)` return? Round both to 2 decimal places.", NUM("50", "12.5")),
        ("QB-4-NEW", "In our enzyme_kinetics module, what does `reaction_velocity(8.0, 60.0, 4.0, 1.0)` return? "
                     "Round to 2 decimal places.", NUM("20")),
    ],
}
SYS = ("You are Clover, a precise engineering and scientific assistant. "
       "Answer the user's question directly with exact numbers and formulas.")


def run_worker(cfg):
    p = subprocess.run([sys.executable, "-c", WORKER, str(CLOVER), json.dumps(cfg)], capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    line = next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("__CLOVER_AB_JSON__:"))
    return json.loads(line.split(":", 1)[1])


def main():
    liveness = {"QA-3-NEW right": NUM("33", "0")("It returns 33.0 and 0.0."),
                "QA-3-NEW wrong": NUM("33", "0")("effective_throughput(5, 0, True) gives 33.33."),
                "QB-3-NEW right": NUM("50", "12.5")("50.00 and 12.50"), "QB-3-NEW wrong": NUM("50", "12.5")("50.5 and 12.55"),
                "QB-4-NEW right": NUM("20")("20.00"), "QB-4-NEW wrong": NUM("20")("200.0")}
    assert liveness == {"QA-3-NEW right": True, "QA-3-NEW wrong": False, "QB-3-NEW right": True, "QB-3-NEW wrong": False,
                        "QB-4-NEW right": True, "QB-4-NEW wrong": False}, liveness
    tmp = Path(tempfile.mkdtemp(prefix="consc-step4-"))
    ws = {}
    for k, files in FILES.items():
        ws[k] = tmp / f"ws_{k}"
        ws[k].mkdir()
        for name, body in files.items():
            (ws[k] / name).write_text(body, encoding="utf-8")
    contexts = {}
    arms = {"OFF": (False, None), "ON": (True, None),
            "ON_NO_SANDBOX": (True, {"memory": "allow", "files": "allow", "code": "deny", "internet": "deny"})}
    idle = {}
    for arm, (cur_on, perms) in arms.items():
        contexts[arm] = {}
        for k in ("a", "b"):
            inst = tmp / f"inst_{arm}_{k}"
            p1 = run_worker({"inst_dir": str(inst), "ws_dir": str(ws[k]), "ws_name": f"ws_{k}", "curiosity_enabled": cur_on,
                             "perms": perms, "phase": "session1_and_idle",
                             "initial_question": "How does this module behave across its normal and boundary inputs?"})
            idle[f"{arm}_{k}"] = p1["idle_actions"]
            p2 = run_worker({"inst_dir": str(inst), "ws_dir": str(ws[k]), "ws_name": f"ws_{k}", "curiosity_enabled": cur_on,
                             "perms": perms, "phase": "session2_retrieve_after_restart",
                             "eval_questions": [{"id": i, "question": q} for i, q, _ in QUESTIONS[k]]})
            for qid, c in p2["retrieved_contexts"].items():
                contexts[arm][qid] = c["memory_context"]
    contexts["PLAIN"] = {}
    for k in ("a", "b"):
        code = "\n\n".join(f"# file: {n}\n{b}" for n, b in FILES[k].items() if not n.endswith("_config.py"))
        for qid, _, _ in QUESTIONS[k]:
            contexts["PLAIN"][qid] = "The workspace's Python files:\n" + code
    sandbox = Path(os.environ["TEMP"]) / "clover-consc" / "base"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in ("tools", "data"):
        if not (sandbox / name).exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(sandbox / name), str(Path.home() / "CloverAI" / name)],
                           check=True, capture_output=True)
    srv = None
    rows, summary = [], {}
    try:
        srv = L.LlamaServer(sandbox, kind="vulkan", ctx=16384, model=sandbox / "data" / "omni" / "gguf" / "LFM3-Omni-2B-Q4_0.gguf",
                            extra_args=("--reasoning", "off"))
        client = RSL.LiveModelClient(endpoint_url=f"http://127.0.0.1:{srv.port}", api_key=srv._key)
        for arm in ("OFF", "ON", "ON_NO_SANDBOX", "PLAIN"):
            for k in ("a", "b"):
                for qid, q, chk in QUESTIONS[k]:
                    ctx = contexts[arm].get(qid) or ""
                    user = f"{ctx}\n\nUser Question:\n{q}" if ctx else q
                    t0 = time.perf_counter()
                    ans = client.generate([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                          max_tokens=320, temperature=0.0, seed=20261109)["content"]
                    ok = bool(chk(ans))
                    rows.append({"arm": arm, "id": qid, "had_context": bool(ctx), "passed": ok,
                                 "s": round(time.perf_counter() - t0, 1), "answer": ans[:800]})
                    print(f"{arm:14} {qid:9} context={bool(ctx)!s:5} passed={ok}", flush=True)
            got = [r for r in rows if r["arm"] == arm]
            summary[arm] = {"original": f"{sum(r['passed'] for r in got if 'NEW' not in r['id'])}/4",
                            "new": f"{sum(r['passed'] for r in got if 'NEW' in r['id'])}/4",
                            "with_context": sum(r["had_context"] for r in got)}
    finally:
        if srv is not None:
            srv.close()
        for name in ("tools", "data"):
            subprocess.run(["cmd", "/c", "rmdir", str(sandbox / name)], capture_output=True)
    out = {"summary": summary, "idle_actions": idle, "rows": rows}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "step4_deconfounded_curiosity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
