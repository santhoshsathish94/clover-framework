"""Held-out re-test of the superagent-jev-dory routing claim (review of 9b12f274).

Imports the study script unchanged and runs its own backend class on:
  - the original 12 tasks (reproduction gate G0 against the recorded results), and
  - 24 tasks written and pre-registered before this run (heldout_tasks.json).
See PREREGISTRATION.md for the decision rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("STUDY_DIR", "/opt/clover-k3/superagent-jev-dory-study"))
sys.path.insert(0, str(STUDY))

import torch  # noqa: E402

if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

import pydantic  # noqa: E402
import transformers  # noqa: E402
import run_openjev_live_experiments as S  # noqa: E402  (unchanged study script)
from openjev.backends.hf import HFBackend  # noqa: E402
from openjev.core.primitives import Choice, ChoiceAnswer, SystemOneRequest  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

INSTRUCTIONS = "Given the state, what should the agent do next?"
ROUTING_CRITERIA = {
    "act": "Directly perform the task or answer now because all inputs are present",
    "gather": "Read files, logs, or system state to gather missing evidence first",
    "clarify": "Ask the user a clarifying question because the request is ambiguous or high-risk",
    "verify": "Run a check or compare against raw evidence to verify an unconfirmed claim",
}
# (name, kind, prior_alpha, chat_template, pre_registered)
ARMS = [
    ("A0_raw_hfbackend", "raw", None, None, True),
    ("A1_markdown_a0.8", "upg", 0.8, False, True),
    ("A2_chat_a0.55", "upg", 0.55, True, True),
    ("X_chat_a0.0", "upg", 0.0, True, False),
    ("X_chat_a0.3", "upg", 0.3, True, False),
    ("X_chat_a0.8", "upg", 0.8, True, False),
    ("X_chat_a1.0", "upg", 1.0, True, False),
    ("X_markdown_a0.55", "upg", 0.55, False, False),
]
MODELS = [
    ("LFM3-1.7B", "/opt/clover-k3/models/LFM3-1.7B"),
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
]
RECORDED_KEY = {"A0_raw_hfbackend": "raw_pred", "A1_markdown_a0.8": "upg_pred", "A2_chat_a0.55": "struct_pred"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leak_words(state: str) -> list[str]:
    text = state.lower()
    for ch in ".,'\"?!:;()":
        text = text.replace(ch, " ")
    return [w for w in text.split() if w == "act" or w.startswith(("gather", "clarif", "verif"))]


def summarize(recs: list[dict]) -> dict:
    m = S.compute_ece_and_brier(recs)
    m["correct"] = sum(1 for r in recs if r["predicted"] == r["gold"])
    m["n"] = len(recs)
    m["predicted_counts"] = {k: sum(1 for r in recs if r["predicted"] == k) for k in ROUTING_CRITERIA}
    return m


def run_model(short: str, path: str, tasks: list[dict]) -> dict:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).eval()
    raw = HFBackend(model_id=path, device="cpu")
    raw._torch = torch
    raw._tok = tok
    raw._model = model
    raw.device = "cpu"
    arms_out = {}
    for name, kind, alpha, chat, prereg in ARMS:
        if kind == "raw":
            backend = raw
        else:
            backend = S.CloverUpgradedOpenJevBackend(
                model=model, tokenizer=tok, model_name=short, device="cpu",
                prior_alpha=alpha, temperature=0.7,
            )
            backend.use_chat_template = chat
        recs = []
        for item in tasks:
            req = SystemOneRequest(
                state=item["state"],
                questions={"direction": Choice(instructions=INSTRUCTIONS, criteria=ROUTING_CRITERIA)},
            )
            ans = backend.decide(req).answers["direction"]
            assert isinstance(ans, ChoiceAnswer)
            recs.append({
                "id": item["id"], "set": item["set"], "gold": item["gold"],
                "predicted": ans.choice, "confidence": ans.confidence,
                "probabilities": ans.probabilities,
            })
        held = [r for r in recs if r["set"] in ("H1", "H2")]
        arms_out[name] = {
            "pre_registered": prereg,
            "prior_alpha": alpha,
            "chat_template": chat,
            "original": summarize([r for r in recs if r["set"] == "original"]),
            "H1": summarize([r for r in recs if r["set"] == "H1"]),
            "H2": summarize([r for r in recs if r["set"] == "H2"]),
            "heldout": summarize(held),
            "records": recs,
        }
        a = arms_out[name]
        print(f"{short} {name}: original {a['original']['correct']}/12 "
              f"H1 {a['H1']['correct']}/12 H2 {a['H2']['correct']}/12 "
              f"heldout {a['heldout']['correct']}/24 ece {a['heldout']['ece']}", flush=True)
    del model
    return {"model_path": path, "seconds": round(time.time() - t0, 1), "arms": arms_out}


def gate_g0(short: str, model_out: dict, recorded: dict) -> dict:
    rec_model = recorded["models_evaluated"].get(short)
    if rec_model is None:
        return {"checked": 0, "matched": 0, "mismatches": ["model not in recorded results"]}
    rows = {r["id"]: r for r in rec_model["defect3_routing_calibration_benchmark"]["per_task_comparison"]}
    checked = matched = 0
    mismatches = []
    max_conf_diff = 0.0
    for arm, key in RECORDED_KEY.items():
        for r in model_out["arms"][arm]["records"]:
            if r["set"] != "original":
                continue
            checked += 1
            want = rows[r["id"]][key]
            if r["predicted"] == want:
                matched += 1
            else:
                mismatches.append(f"{arm} {r['id']}: got {r['predicted']}, recorded {want}")
            if arm == "A2_chat_a0.55":
                max_conf_diff = max(max_conf_diff, abs(r["confidence"] - rows[r["id"]]["struct_conf"]))
    return {"checked": checked, "matched": matched, "mismatches": mismatches,
            "a2_max_abs_confidence_diff_vs_recorded": round(max_conf_diff, 6)}


def main() -> None:
    src = (STUDY / "run_openjev_live_experiments.py").read_text(encoding="utf-8")
    for k, v in ROUTING_CRITERIA.items():
        assert f'"{k}": "{v}"' in src, f"criterion drift: {k}"
    assert INSTRUCTIONS in src, "instruction drift"

    spec = json.loads((HERE / "heldout_tasks.json").read_text(encoding="utf-8"))
    held = spec["tasks"]
    assert len(held) == 24 and len({t["id"] for t in held}) == 24
    for label in ROUTING_CRITERIA:
        for s in ("H1", "H2"):
            assert sum(1 for t in held if t["gold"] == label and t["set"] == s) == 3, (label, s)
    for t in held:
        assert not leak_words(t["state"]), (t["id"], leak_words(t["state"]))
    original = [{"id": t["id"], "set": "original", "gold": t["gold"], "state": t["state"]}
                for t in S.CLOVER_ROUTING_BENCHMARK]
    tasks = original + held

    recorded_path = STUDY / "openjev_live_results.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    out = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pydantic": pydantic.__version__,
        "torch_threads": torch.get_num_threads(),
        "sha256": {
            "study_script": sha256(STUDY / "run_openjev_live_experiments.py"),
            "recorded_results": sha256(recorded_path),
            "heldout_tasks": sha256(HERE / "heldout_tasks.json"),
            "preregistration": sha256(HERE / "PREREGISTRATION.md"),
            "this_harness": sha256(Path(__file__).resolve()),
        },
        "models": {},
        "gate_g0": {},
    }
    only = os.environ.get("ONLY_MODEL")
    for short, path in MODELS:
        if only and short != only:
            continue
        print(f"=== {short} ===", flush=True)
        out["models"][short] = run_model(short, path, tasks)
        out["gate_g0"][short] = gate_g0(short, out["models"][short], recorded)
        print(f"G0 {short}: {out['gate_g0'][short]}", flush=True)
        (HERE / "heldout_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "heldout_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("HELDOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
