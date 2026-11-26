from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from raw_runtime import initialize_paths, load_c2


def run(root):
    experiment = initialize_paths(root)
    sys.path.insert(0, str(experiment / "planner_source"))
    import torch
    from hybrid_agent.engine import HybridGenerationEngine
    from hybrid_agent.model import HybridAgentModel
    from hybrid_agent.tools import ToolExecutor

    sys.path.insert(0, str(root / "composite"))
    from prepare import file_hash, write_json

    output = experiment / "planner_probe"
    output.mkdir(exist_ok=False)
    selection = json.loads((experiment / "results" / "C2_SELECTION.json").read_text())
    source = experiment / "planner_source" / "hybrid_agent"
    protocol = {
        "questions": ["What is 17 plus 26?", "Which city is the capital of Italy?"],
        "system": "Answer the question carefully and finish with a concise final answer.",
        "max_generation_steps": 96,
        "temperature": 0.0,
        "latent_limits": [4, 32],
        "tool_calls_allowed": 0,
        "interpretation": "Adapter usage/readiness only; not a new improvement gate.",
        "source_hashes": {str(path.relative_to(experiment)): file_hash(path) for path in source.rglob("*.py")},
        "script_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "PROTOCOL.json", protocol)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    backbone, link, tokenizer = load_c2(root)
    model = HybridAgentModel(backbone, link).eval()

    class TrainingPromptEngine(HybridGenerationEngine):
        def _build_prompt(self):
            prompt = super()._build_prompt()
            if prompt.endswith("<think>"):
                prompt += "\n"
            self.initial_prompt = prompt
            return prompt

    results = []
    for name in ("original", selection["selected"]):
        checkpoint = experiment / "copies" / ("c2-" + name) / "1-adapter.pt"
        model.hybridlink.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
        for index, question in enumerate(protocol["questions"]):
            torch.manual_seed(20261112)
            engine = TrainingPromptEngine(
                model, tokenizer, torch.device("cpu"), ToolExecutor(), protocol["system"], question,
                batch_size=1, max_response_tokens=protocol["max_generation_steps"],
                max_tool_calls=0, max_python_calls=0, max_search_calls=0,
                terminate_on_tool_cap=True, temperature=0.0, min_latent_steps=4,
                max_latent_steps=32, record_latent_readout=False,
            )
            engine.run()
            result = engine.row_result(engine.rows[0])
            if result["python_calls"] or result["search_calls"]:
                raise ValueError("a tool was executed")
            result.update(variant=name, question=question, seconds=engine.total_time, checkpoint_sha256=file_hash(checkpoint))
            results.append(result)
            write_json(output / f"{name}-{index}.json", result)
            print(json.dumps({"variant": name, "question": index, "latent_steps": result["latent_steps"], "end_reason": result["end_reason"]}), flush=True)
    comparisons = []
    for question in protocol["questions"]:
        records = [result for result in results if result["question"] == question]
        comparisons.append({"question": question, "outputs_equal": records[0]["output"] == records[1]["output"], "latent_steps": [record["latent_steps"] for record in records], "end_reasons": [record["end_reason"] for record in records]})
    write_json(output / "SUMMARY.json", {"comparisons": comparisons, "gate_reopened": False})
    print("PLANNER_USAGE_COMPLETE " + json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve(strict=True))