"""Regenerates the memory contexts the step-4 curiosity arms handed the model (step 4 saved the answers, not the
contexts). No model: the curiosity worker runs on stand-ins. Cross-check that it is the same memory: the idle actions
(strategy, competence before/after, promoted id) must equal the ones step 4 recorded."""
import json
import re
import tempfile
from pathlib import Path

import step4_deconfounded_curiosity as S4


def main():
    tmp = Path(tempfile.mkdtemp(prefix="consc-step4ctx-"))
    ws = {}
    for k, files in S4.FILES.items():
        ws[k] = tmp / f"ws_{k}"
        ws[k].mkdir()
        for name, body in files.items():
            (ws[k] / name).write_text(body, encoding="utf-8")
    arms = {"ON": None, "ON_NO_SANDBOX": {"memory": "allow", "files": "allow", "code": "deny", "internet": "deny"}}
    idle, contexts = {}, {}
    for arm, perms in arms.items():
        contexts[arm] = {}
        for k in ("a", "b"):
            base = {"inst_dir": str(tmp / f"inst_{arm}_{k}"), "ws_dir": str(ws[k]), "ws_name": f"ws_{k}",
                    "curiosity_enabled": True, "perms": perms}
            p1 = S4.run_worker({**base, "phase": "session1_and_idle",
                                "initial_question": "How does this module behave across its normal and boundary inputs?"})
            idle[f"{arm}_{k}"] = p1["idle_actions"]
            p2 = S4.run_worker({**base, "phase": "session2_retrieve_after_restart",
                                "eval_questions": [{"id": i, "question": q} for i, q, _ in S4.QUESTIONS[k]]})
            for qid, c in p2["retrieved_contexts"].items():
                contexts[arm][qid] = c["memory_context"]
    recorded = json.loads((S4.HERE / "results" / "step4_deconfounded_curiosity.json").read_text(encoding="utf-8"))
    secrets = [re.search(r'"([^"]+)"', body).group(1)
               for files in S4.FILES.values() for name, body in files.items() if name.endswith("_config.py")]
    out = {"idle_actions_match_step4": {key: idle[key] == recorded["idle_actions"][key] for key in idle},
           "secret_value_in_any_context": {arm: any(s in c for c in ctx.values() for s in secrets)
                                           for arm, ctx in contexts.items()},
           "idle_actions": idle, "contexts": contexts}
    (S4.HERE / "results" / "step4_contexts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("idle_actions_match_step4", "secret_value_in_any_context")}, indent=2))


if __name__ == "__main__":
    main()
