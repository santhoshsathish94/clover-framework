"""Does developmental continuity survive replacing the worker?

The architecture's central claim is that the model is a replaceable component
while the persistent state carries the developmental identity. This runs one
state through two different workers and checks whether the second continues the
first's trajectory instead of restarting it.

It also checks the gate in both directions: a worker that asserts its own
success is rejected, and one that supplies independent evidence is accepted.
Without the second half, "nothing was accepted" would prove only that the gate
is shut, not that it discriminates.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "model_swap_evidence.json"
sys.path.insert(0, str(HERE))

class ScriptedBackend:
    """A deterministic stand-in for a model. Distinguishable, and optionally evidenced."""

    def __init__(self, name, question, stop_after=2, independent_evidence=False):
        self.name = name
        self.question = question
        self.stop_after = stop_after
        self.independent_evidence = independent_evidence
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        result = {
            "status": "PAUSE" if self.calls >= self.stop_after else "CONTINUE",
            "inherited_question": state.get("next_question"),
            "learning": f"{self.name} advanced the investigation",
            "next_question": f"{self.question} (after {self.name} call {self.calls})",
        }
        if self.independent_evidence:
            result["independent_evidence"] = {
                "source": "environment",
                "observation": "child process returned exit code 0",
            }
        return result

class CycleLimited:
    """Caps a worker's cycles. CLOVER_MAX_CYCLES cannot do this: the supervisor reads it at import."""

    def __init__(self, inner, limit):
        self.inner = inner
        self.name = inner.name
        self.limit = limit
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        result = self.inner.invoke(state)
        if self.calls >= self.limit:
            result["status"] = "PAUSE"
        return result

def build_workers():
    """Real local models when CLOVER_SWAP_MODELS names at least two, scripted otherwise."""
    names = [m.strip() for m in os.getenv("CLOVER_SWAP_MODELS", "").split(",") if m.strip()]
    if len(names) < 2:
        return None
    from supervisor.local_chat_backend import ChatCompletionsBackend
    workers = [CycleLimited(ChatCompletionsBackend(n), 2) for n in names]

    url = urlparse(workers[0].inner.base_url)
    with socket.socket() as probe:
        probe.settimeout(2)
        if probe.connect_ex((url.hostname, url.port or 80)) != 0:
            raise SystemExit(
                f"No inference server listening at {workers[0].inner.base_url}.\n"
                f"Start one (for example `ollama serve`), or unset CLOVER_SWAP_MODELS "
                f"to run the scripted arm instead."
            )
    return workers

def main():
    real = build_workers()
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "swap_state.json"
        os.environ["CLOVER_STATE"] = str(state_file)
        os.environ["CLOVER_MAX_CYCLES"] = "20"
        # The supervisor reads both of those at import time.
        from supervisor.supervisor import run

        if real:
            worker_kind = f"local language models via chat completions ({len(real)})"
            states = []
            for worker in real:
                run(worker)
                states.append(json.loads(state_file.read_text()))
            after_a, after_b, after_c = states[0], states[1], states[-1]
        else:
            worker_kind = "scripted stand-ins, not language models"
            worker_a = ScriptedBackend("worker-a", "what does the runtime permit")
            run(worker_a)
            after_a = json.loads(state_file.read_text())

            worker_b = ScriptedBackend("worker-b", "what does the evaluator reject")
            run(worker_b)
            after_b = json.loads(state_file.read_text())

            worker_c = ScriptedBackend("worker-c", "closing", independent_evidence=True)
            run(worker_c)
            after_c = json.loads(state_file.read_text())

    backends = [h["backend"] for h in after_c["history"]]
    cycles = [h["cycle"] for h in after_c["history"]]
    first_worker = backends[0]
    second_worker = next((b for b in backends if b != first_worker), None)
    b_first = next((h for h in after_b["history"] if h["backend"] == second_worker), None)
    a_last_question = after_a.get("next_question")

    checks.append(("state survived the swap",
                   len(after_b["history"]) > len(after_a["history"])))
    checks.append(("cycle numbering continued, not restarted",
                   cycles == list(range(1, len(cycles) + 1))))
    checks.append(("both workers appear in one history",
                   len(set(backends)) > 1))
    checks.append((f"{second_worker} inherited {first_worker}'s open question",
                   bool(a_last_question) and b_first is not None
                   and b_first["result"].get("inherited_question") == a_last_question))
    if not real:
        checks.append(("self-certified claims stayed out of validated_knowledge",
                       all("worker-a" not in k and "worker-b" not in k
                           for k in after_c.get("validated_knowledge", []))))
        checks.append(("evidenced claim was accepted",
                       any("worker-c" in k for k in after_c.get("validated_knowledge", []))))

    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\ncycles run           : {len(cycles)}")
    print(f"workers used         : {', '.join(sorted(set(backends)))}")
    print(f"validated_knowledge  : {len(after_c.get('validated_knowledge', []))}")
    print(f"unverified_claims    : {len(after_c.get('unverified_claims', []))}")

    failed = [label for label, ok in checks if not ok]

    EVIDENCE.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "worker_type": worker_kind,
        "workers": sorted(set(backends)),
        "checks": [{"claim": label, "passed": ok} for label, ok in checks],
        "cycles": [{"cycle": h["cycle"], "worker": h["backend"],
                    "status": h["result"].get("status"),
                    "claim_accepted": h["assessment"]["accepted"],
                    "inherited_question": h["result"].get("inherited_question")}
                   for h in after_c["history"]],
        "handover": {
            "question_left_by_worker_a": a_last_question,
            "question_received_by_worker_b": b_first["result"]["inherited_question"],
        },
        "validated_knowledge": after_c.get("validated_knowledge", []),
        "unverified_claims": after_c.get("unverified_claims", []),
        "result": "pass" if not failed else "fail",
        "failed_checks": failed,
        "does_not_establish": ([
            "that the developmental state is meaningful, only that it is carried",
            "that the workers reasoned well, only that the handover held",
            "open-ended development, general capability, or subjective experience",
        ] if real else [
            "that a language model would continue the trajectory usefully",
            "that the developmental state is meaningful, only that it is carried",
            "anything about worker quality; the workers are deterministic scripts",
            "open-ended development, general capability, or subjective experience",
        ]),
    }, indent=2))

    print(f"\nevidence written to {EVIDENCE.name}")
    print(f"verdict: {'continuity survived worker replacement' if not failed else 'FAILED: ' + '; '.join(failed)}")
    return 0 if not failed else 1

if __name__ == "__main__":
    raise SystemExit(main())
