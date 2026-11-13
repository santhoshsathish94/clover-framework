from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from artifacts import physical_copy, sha256, tensor_hash, write_json


POLICIES = {
    "original": (1.0, 1.0),
    "standard_092": (0.92, 1.0),
    "standard_108": (1.08, 1.0),
    "native_092": (1.0, 0.92),
    "native_108": (1.0, 1.08),
}


def configure(root):
    root = Path(root).resolve(strict=True)
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(root / "cache" / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    sys.path.insert(0, str(root / "vendor"))
    return root


def gate(candidate, baseline, require_significance=False):
    if len(candidate) != len(baseline) or not baseline:
        raise ValueError("invalid paired gate inputs")
    guarded = all(
        new["ntp"] <= old["ntp"] + 1e-6
        and new["text_ntp"] <= old["text_ntp"] + 1e-6
        and new["cos_std"] >= old["cos_std"] - 1e-6
        and new["cos_nat"] >= old["cos_nat"] - 1e-6
        and not set(old["correct_positions"]) - set(new["correct_positions"])
        for new, old in zip(candidate, baseline)
    )
    improved = sum(new["ntp"] for new in candidate) < sum(old["ntp"] for old in baseline) - 1e-4
    better = sum(new["ntp"] < old["ntp"] - 1e-6 for new, old in zip(candidate, baseline))
    significant = not require_significance or 2.0 ** (-better) <= 0.05
    return guarded and improved and significant


def freeze(root):
    root = configure(root)
    from transformers import AutoTokenizer
    from build_nemotron_hybridsft_dataset import tokenize_trajectory_for_nemotron

    tokenizer = AutoTokenizer.from_pretrained(root / "inputs" / "tokenizer", local_files_only=True, trust_remote_code=False)
    cases = []
    for index in range(24):
        boxes = 3 + index
        per_box = 4 + index % 7
        total = boxes * per_box
        messages = [
            {"role": "user", "content": f"There are {boxes} sealed boxes with {per_box} labels each. How many labels are there?"},
            {"role": "assistant", "content": f"<think><latent>{boxes} times {per_box} equals {total}</latent></think>The answer is {total}."},
        ]
        case = tokenize_trajectory_for_nemotron(tokenizer, messages, max_seq_len=128, max_latent_steps=8)
        if case is None:
            raise ValueError("diagnostic trajectory was not accepted by saved data builder")
        case.update(id=f"labels-{index:02d}", split="discovery" if index < 8 else "validation", messages=messages)
        cases.append(case)
    frozen = root / "frozen"
    frozen.mkdir(exist_ok=False)
    write_json(frozen / "cases.json", cases)
    paths = [Path(__file__), Path(__file__).parent / "artifacts.py", root / "OBSERVATION.json", root / "INPUTS.json", frozen / "cases.json"]
    paths += [path for path in (root / "vendor").rglob("*.py")]
    write_json(frozen / "PROTOCOL.json", {
        "policies": POLICIES,
        "gate": "lower mean NTP with no per-example NTP/text loss, alignment, or correct-token regression; float tolerance 1e-6",
        "selection": "discovery only, then frozen candidate on reserved validation",
        "validation_significance": "paired one-sided sign p<=0.05, in addition to zero meaningful per-example regressions",
        "scope": "new arithmetic diagnostic trajectories on the saved 3-stage backbone, not full-model validation",
        "deployment_eligible": False,
    })
    write_json(frozen / "MANIFEST.json", {str(path.resolve()): sha256(path) for path in paths + [frozen / "PROTOCOL.json"]})
    print(json.dumps({"discovery": 8, "validation": 16, "policies": list(POLICIES)}), flush=True)


def evaluate(model, cases):
    import torch
    import torch.nn.functional as functional
    from train_nemotron_hybridagent_gpu import collate_hybridsft

    records = []
    for case in cases:
        batch = collate_hybridsft([case])
        embeddings = model.get_adapted_embeddings(batch["input_ids"])
        _, teacher = model.run_backbone(embeddings)
        previous = torch.zeros_like(teacher, dtype=torch.float32)
        previous[:, 1:] = teacher[:, :-1].float()
        standard = model.hybrid_link(previous)
        native = model.nemo_native_hybrid_link(previous)
        mask = batch["latent_mask"].clone()
        mask[:, 0] = False
        feedback = 0.5 * (standard + native)
        inputs = torch.where(mask[:, :, None], feedback, embeddings)
        logits, _ = model.run_backbone(inputs)
        labels = batch["labels"][:, 1:]
        valid = labels != -100
        losses = functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(labels)
        predictions = logits[:, :-1].argmax(-1)
        text_mask = valid & ~torch.isin(labels, torch.tensor([12, 13, 30, 31]))
        records.append({
            "id": case["id"], "ntp": float(losses[valid].mean()), "text_ntp": float(losses[text_mask].mean()),
            "cos_std": float(functional.cosine_similarity(standard[mask], embeddings[mask], dim=-1).mean()),
            "cos_nat": float(functional.cosine_similarity(native[mask], embeddings[mask], dim=-1).mean()),
            "correct_positions": torch.nonzero((predictions == labels) & valid, as_tuple=False)[:, 1].tolist(),
            "predictions": predictions[valid].tolist(), "labels": labels[valid].tolist(),
        })
    return records


def trial(root):
    root = configure(root)
    import copy
    import torch
    from artifacts import tensor_leaves

    for path, digest in json.loads((root / "frozen" / "MANIFEST.json").read_text()).items():
        if sha256(path) != digest:
            raise ValueError("frozen Nemotron source/cases changed")
    verify_inputs(root)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261112)
    model, state = load_model(root)
    base = {name: tensor_hash(value) for name, value in tensor_leaves(state)}
    cases = json.loads((root / "frozen" / "cases.json").read_text())
    discovery = [case for case in cases if case["split"] == "discovery"]
    validation = [case for case in cases if case["split"] == "validation"]
    results = root / "results"
    results.mkdir(exist_ok=False)
    summaries = {}
    paths = {}
    for name, (standard_gain, native_gain) in POLICIES.items():
        target = root / "trials" / name / "checkpoint.pt"
        record = physical_copy(root / "baseline" / "nemotron_hybridagent_final.pt", target)
        altered = copy.deepcopy(state)
        for key in ("post_ln.weight", "post_ln.bias"):
            altered["hybrid_link"][key] = altered["hybrid_link"][key] * standard_gain
        altered["nemo_native_hybrid_link"]["out_scale"] = altered["nemo_native_hybrid_link"]["out_scale"] * native_gain
        if name != "original":
            temporary = target.with_name("new.pt")
            torch.save(altered, temporary)
            temporary.replace(target)
        loaded = torch.load(target, weights_only=True, map_location="cpu", mmap=True)
        after = {key: tensor_hash(value) for key, value in tensor_leaves(loaded)}
        changed = [key for key in base if base[key] != after.get(key)]
        expected = [] if name == "original" else ["hybrid_link.post_ln.weight", "hybrid_link.post_ln.bias"] if standard_gain != 1 else ["nemo_native_hybrid_link.out_scale"]
        if set(base) != set(after) or set(changed) != set(expected):
            raise ValueError("Nemotron trial changed unexpected tensors")
        write_json(target.parent / "PATCH_AUDIT.json", {"copy": record, "changed_tensors": changed, "before": base, "after": after, "sha256": sha256(target)})
        model.hybrid_link.load_state_dict(loaded["hybrid_link"], strict=True)
        model.nemo_native_hybrid_link.load_state_dict(loaded["nemo_native_hybrid_link"], strict=True)
        rows = evaluate(model, discovery)
        write_json(results / f"discovery-{name}.json", rows)
        summaries[name] = rows
        paths[name] = target
        print(json.dumps({"policy": name, "mean_ntp": sum(row["ntp"] for row in rows) / len(rows), "qualifies": name != "original" and gate(rows, summaries["original"])}), flush=True)
    eligible = [name for name in POLICIES if name != "original" and gate(summaries[name], summaries["original"])]
    selected = min(eligible, key=lambda name: sum(row["ntp"] for row in summaries[name])) if eligible else None
    write_json(results / "SELECTION.json", {"selected": selected, "validation_opened": bool(selected)})
    accepted = False
    if selected:
        reserved = {}
        for name in ("original", selected):
            loaded = torch.load(paths[name], weights_only=True, map_location="cpu", mmap=True)
            model.hybrid_link.load_state_dict(loaded["hybrid_link"], strict=True)
            model.nemo_native_hybrid_link.load_state_dict(loaded["nemo_native_hybrid_link"], strict=True)
            reserved[name] = evaluate(model, validation)
            write_json(results / f"validation-{name}.json", reserved[name])
        accepted = gate(reserved[selected], reserved["original"], require_significance=True)
    for name, path in paths.items():
        if not (accepted and name == selected):
            temporary = path.with_name("restored.pt")
            with temporary.open("xb") as target, (root / "baseline" / "nemotron_hybridagent_final.pt").open("rb") as source:
                import shutil
                shutil.copyfileobj(source, target)
            temporary.replace(path)
            if sha256(path) != sha256(root / "baseline" / "nemotron_hybridagent_final.pt"):
                raise ValueError("Nemotron reversal failed")
            write_json(path.parent / "REVERSAL.json", {"restored": True, "sha256": sha256(path)})
    verify_inputs(root)
    write_json(results / "OUTCOME.json", {"selected": selected, "stage_gate_passed": accepted, "deployment_eligible": False, "baseline": str(root / "baseline" / "nemotron_hybridagent_final.pt"), "candidate": str(paths[selected]) if accepted else None, "scope": "reduced-stage diagnostic only"})
    print(json.dumps({"nemotron_complete": True, "selected": selected, "stage_gate_passed": accepted, "deployment_eligible": False}), flush=True)


def snapshot(root, containers, tokenizer):
    root = configure(root)
    records = []
    target = root / "inputs"
    target.mkdir(exist_ok=False)
    required = {
        "global-0": ["embeddings.bf16.bin", "lm_head.bf16.bin", "norm_f.bf16.bin"],
        "trunk-0": ["index.bin", "common.bin", "operator.bin"],
        "trunk-1": ["index.bin", "common.bin", "routed-shared.bin"],
        "trunk-7": ["index.bin", "common.bin", "operator.bin"],
        "expert-1": ["experts.direct"],
    }
    for directory, filenames in required.items():
        for filename in filenames:
            source = containers / directory / filename
            records.append(physical_copy(source, target / "pruned" / directory / filename))
    for filename in ("tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja"):
        records.append(physical_copy(tokenizer / filename, target / "tokenizer" / filename))
    write_json(root / "INPUTS.json", {"files": records, "scope": "three-stage training backbone, not full 88-layer model"})
    print(json.dumps({"copied_files": len(records), "copied_bytes": sum(record["bytes"] for record in records)}), flush=True)


def verify_inputs(root):
    observation = json.loads((root / "OBSERVATION.json").read_text(encoding="utf-8"))
    records = [observation["copy"]] + json.loads((root / "INPUTS.json").read_text(encoding="utf-8"))["files"]
    for record in records:
        if sha256(record["copy"]) != record["sha256"]:
            raise ValueError(f"baseline copy changed: {record['copy']}")
    return records


def load_model(root, checkpoint=None):
    import torch
    from nemotron_gpu_backbone import NemotronGPUStageModel
    from train_nemotron_hybridagent_gpu import NemotronHybridSFTTrainableModule

    root = configure(root)
    checkpoint = checkpoint or root / "baseline" / "nemotron_hybridagent_final.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if state["hidden_size"] != 4096 or state["vocab_size"] != 131072:
        raise ValueError("unexpected Nemotron checkpoint dimensions")
    backbone = NemotronGPUStageModel(root / "inputs" / "pruned", experts_on_gpu=False, init_latent_special_tokens=True)
    model = NemotronHybridSFTTrainableModule(backbone)
    model.hybrid_link.load_state_dict(state["hybrid_link"], strict=True)
    model.nemo_native_hybrid_link.load_state_dict(state["nemo_native_hybrid_link"], strict=True)
    with torch.no_grad():
        for name in ("special_emb_delta", "special_lm_delta", "norm_f_delta"):
            getattr(model, name).copy_(state[name])
    return model.eval().requires_grad_(False), state


def smoke(root):
    root = configure(root)
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model, _ = load_model(root)
    ids = torch.tensor([[1, 10, 12, 30, 100, 200, 31, 13, 11]], dtype=torch.long)
    labels = ids.clone()
    labels[:, :2] = -100
    labels[:, 4:6] = -100
    latent = torch.tensor([[False, False, False, True, True, True, False, False, False]])
    with torch.inference_mode():
        result = model(ids, labels, latent, alpha=1.0, beta=0.1)
    values = {name: float(value) for name, value in result.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite stage-model smoke")
    write_json(root / "SMOKE.json", {"metrics": values, "scope": "structural token smoke only; no natural-language quality claim"})
    print(json.dumps(values), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--containers", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--trial", action="store_true")
    args = parser.parse_args()
    if args.snapshot and args.containers and args.tokenizer:
        snapshot(args.root, args.containers, args.tokenizer)
    elif args.smoke:
        smoke(args.root)
    elif args.freeze:
        freeze(args.root)
    elif args.trial:
        trial(args.root)
    else:
        parser.error("choose --snapshot with input paths, or --smoke")