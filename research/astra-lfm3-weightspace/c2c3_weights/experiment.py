from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import unittest
from pathlib import Path

from raw_runtime import initialize_paths, least_squares_gain, load_c2, load_c3


TEXTS = [
    "The delivery contains twelve blue folders and seven green notebooks.",
    "The west entrance opens at eight and closes at six.",
    "Mira placed four clean cups beside the silver kettle.",
    "The first shipment weighs thirty kilograms before packing.",
    "A spare battery is stored in the lower drawer.",
    "The report lists three missing labels on the second page.",
    "Each tray holds six glasses and there are five trays.",
    "The red cable connects the monitor to the left socket.",
    "The team approved the first draft after two corrections.",
    "The invoice includes a ten percent discount on paper.",
    "The parcel left the station on Tuesday morning.",
    "The square has four equal sides and four right angles.",
    "The sensor measures temperature once every twenty seconds.",
    "The north room contains eight chairs and two desks.",
    "The recipe requires one cup of rice and two cups of water.",
    "The public library is closed on the first Monday.",
    "A yellow sticker marks the equipment awaiting inspection.",
    "The second container holds fifteen empty bottles.",
    "The account balance increased from forty to fifty dollars.",
    "The final paragraph describes the required safety inspection.",
    "The package contains nine black pens and fourteen orange cards.",
    "The east gate opens at seven and closes at five.",
    "Arun placed three plates beside the wooden cutting board.",
    "The next shipment weighs forty kilograms before packing.",
    "The replacement key is kept in the upper cabinet.",
    "The checklist identifies four loose screws on the back panel.",
    "Each carton holds eight jars and there are seven cartons.",
    "The white cable connects the printer to the right socket.",
    "The committee accepted the revised plan after one correction.",
    "The receipt includes a five percent discount on envelopes.",
    "The order reached the depot on Thursday afternoon.",
    "A triangle has three sides and three interior angles.",
    "The instrument records pressure once every thirty seconds.",
    "The south office contains six tables and eleven chairs.",
    "The mixture uses two cups of flour and one cup of milk.",
    "The local museum remains closed on the last Wednesday.",
    "A purple tag identifies the tools scheduled for repair.",
    "The third box contains seventeen folded towels.",
    "The payment increased the balance from sixty to eighty dollars.",
    "The opening section explains the monthly maintenance schedule.",
]


def helpers(root):
    sys.path.insert(0, str(root / "composite"))
    from prepare import file_hash, physical_copy, tensor_digest, write_json

    return file_hash, physical_copy, tensor_digest, write_json


def freeze(root):
    experiment = root / "c2c3_weights"
    file_hash, _, _, write_json = helpers(root)
    frozen = experiment / "frozen"
    frozen.mkdir(exist_ok=False)
    cases = [{"id": f"text-{index:02d}", "split": "discovery" if index < 20 else "heldout", "text": text} for index, text in enumerate(TEXTS)]
    write_json(frozen / "cases.json", cases)
    write_json(frozen / "PROTOCOL.json", {
        "c2_gains": ["original", "least_squares", "norm_match"],
        "c2_controls": ["true_token", "zero_input"],
        "c2_gate": "lower held-out per-text feedback NLL than original and zero; one-sided paired sign p<=0.05 against both",
        "c3_scales": [0.5, 1.0, 2.0],
        "c3_gate": "lower held-out per-text code NLL with paired sign p<=0.05, plus no loss in generated nonpad token accuracy",
        "c2_objective": "predict token t+2 after one HybridLink feedback input for token t+1",
        "c3_objective": "teacher code NLL and free NCM/VQ reconstruction of the supplied text without pointer copy bonuses",
        "split_limit": "held out from this calibration only; no claim of disjointness from original pretraining",
        "selection": "discovery only; do not retune using heldout outcomes",
        "gradient_training": False,
        "data_fitting": "closed-form scalar C2 calibration using discovery texts only",
    })
    paths = [experiment / filename for filename in ("experiment.py", "raw_runtime.py", "observe.py", "OBSERVATION.json", "RUNTIME_INPUTS.json")]
    paths += [path for path in (experiment / "vendor").rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    paths += [frozen / "cases.json", frozen / "PROTOCOL.json"]
    write_json(frozen / "MANIFEST.json", {str(path.relative_to(root)): file_hash(path) for path in paths})
    write_json(frozen / "HYPOTHESIS.json", {"text": (experiment / "CONTEXT.md").read_bytes().decode("utf-8")})
    print(json.dumps({"discovery_texts": 20, "heldout_texts": 20, "c2_policies": 3, "c3_scales": 3}), flush=True)


def verify_frozen(root):
    file_hash, _, _, _ = helpers(root)
    frozen = root / "c2c3_weights" / "frozen"
    for relative, expected in json.loads((frozen / "MANIFEST.json").read_text()).items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    return json.loads((frozen / "cases.json").read_text())


def sign_test(changed, original):
    better = sum(new < old - 1e-6 for new, old in zip(changed, original))
    worse = sum(new > old + 1e-6 for new, old in zip(changed, original))
    count = better + worse
    probability = sum(math.comb(count, index) for index in range(better, count + 1)) / (2 ** count) if count else 1.0
    return {"better": better, "worse": worse, "ties": len(changed) - count, "one_sided_p": probability}


def patch_checkpoint(root, source, label, component, factor):
    import torch
    from observe import tensor_leaves

    file_hash, physical_copy, tensor_digest, write_json = helpers(root)
    experiment = root / "c2c3_weights"
    directory = experiment / "copies" / label
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / source.name
    copy_record = physical_copy(source, destination, experiment)
    before = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    before_hashes = {name: tensor_digest(tensor) for name, tensor in tensor_leaves(before)}
    after = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    if component == "c2":
        keys = ["post_ln.weight", "post_ln.bias"]
        for key in keys:
            after[key] = (after[key].float() * factor).to(after[key].dtype)
    else:
        keys = ["lfm3_combo_bridge.scale"]
        after["lfm3_combo_bridge"]["scale"] = after["lfm3_combo_bridge"]["scale"] * factor
    if factor != 1.0:
        temporary = directory / "changed.pt"
        torch.save(after, temporary)
        temporary.replace(destination)
    stored = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    after_hashes = {name: tensor_digest(tensor) for name, tensor in tensor_leaves(stored)}
    changed = [name for name in before_hashes if before_hashes[name] != after_hashes.get(name)]
    if set(before_hashes) != set(after_hashes) or (factor != 1.0 and set(changed) != set(keys)) or (factor == 1.0 and changed):
        raise ValueError("unexpected changed tensor set")
    write_json(directory / "PATCH_AUDIT.json", {
        "component": component, "factor": factor, "copy": copy_record,
        "changed_tensors": changed, "before": before_hashes, "after": after_hashes,
        "checkpoint_sha256": file_hash(destination),
    })
    return destination


def c2_samples(backbone, link, tokenizer, cases):
    import torch

    samples = []
    for case in cases:
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "Continue the reasoning carefully."},
            {"role": "user", "content": "State the facts in this note: " + case["text"]},
        ], tokenize=False, add_generation_prompt=True)
        if prompt.endswith("<think>"):
            prompt += "\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = tokenizer.encode(case["text"], add_special_tokens=False)
        positions = sorted({max(1, len(text_ids) // 3), max(1, 2 * len(text_ids) // 3)})
        for position in positions:
            if position + 2 >= len(text_ids):
                continue
            prefix = prompt_ids + text_ids[:position + 1]
            result = backbone(torch.tensor([prefix]), output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = result.hidden_states[-1][:, -1].detach()
            predicted = link(hidden).detach()
            target = backbone.get_input_embeddings()(torch.tensor([text_ids[position + 1]])).detach()
            samples.append({"id": case["id"], "prefix": prefix, "hidden": hidden, "predicted": predicted, "embedding": target, "next_id": text_ids[position + 2]})
    return samples


def c2_evaluate(backbone, link, samples, mode="link"):
    import torch
    import torch.nn.functional as functional

    records = []
    for sample in samples:
        latent = link(sample["hidden"]) if mode == "link" else sample["embedding"] if mode == "true_token" else torch.zeros_like(sample["embedding"])
        prefix = backbone.get_input_embeddings()(torch.tensor([sample["prefix"]]))
        inputs = torch.cat([prefix, latent[:, None, :]], dim=1)
        result = backbone(inputs_embeds=inputs, use_cache=False, return_dict=True, logits_to_keep=1)
        nll = functional.cross_entropy(result.logits[:, -1].float(), torch.tensor([sample["next_id"]])).item()
        records.append({"id": sample["id"], "next_id": sample["next_id"], "nll": nll, "prediction": int(result.logits[0, -1].argmax()), "mse": float((latent.float() - sample["embedding"].float()).square().mean()), "latent_rms": float(latent.float().square().mean().sqrt())})
    return records


def c3_example(modules, backbone, tokenizer, case):
    import torch
    from ncm_runtime import pack_combo_prefix

    token_ids = tokenizer.encode(case["text"], add_special_tokens=False)
    if len(token_ids) > 38:
        raise ValueError("text exceeds declared renderer window")
    padded = torch.tensor([token_ids + [124893] * (48 - len(token_ids))])
    mask = padded == 124893
    hidden = backbone(torch.tensor([token_ids]), output_hidden_states=True, use_cache=False).hidden_states[-1].float()
    full_hidden = torch.zeros(1, 48, 2048)
    full_hidden[:, :len(token_ids)] = hidden
    latent = modules["native"](full_hidden)
    bpe = modules["vqvae"].encoder.wte(padded) + modules["expert"](full_hidden).masked_fill(mask[:, :, None], 0.0)
    target_ids = token_ids + [124900]
    targets = torch.tensor([target_ids + [124893] * (64 - len(target_ids))])
    codes = modules["vqvae"].encode(targets)
    inputs = modules["vqvae"].vq.idx_to_ncm_input(codes)
    return {"id": case["id"], "text": case["text"], "latent": latent, "bpe": bpe, "mask": mask, "target": targets, "target_count": len(target_ids), "codes": codes, "ncm_inputs": inputs}


def c3_evaluate(modules, examples, generate=True):
    import torch
    import torch.nn.functional as functional
    from ncm_runtime import generate_from_zpref, pack_combo_prefix
    from types import SimpleNamespace

    records = []
    for sample in examples:
        rh = modules["combo"](sample["latent"]).float().masked_fill(sample["mask"][:, :, None], 0.0)
        prefix, mask = pack_combo_prefix(rh, sample["mask"], sample["bpe"], sample["mask"])
        logits = modules["net"](torch.cat([prefix, sample["ncm_inputs"]], dim=1), prefix_pad_mask=mask)
        if logits.shape[:2] != sample["codes"].shape:
            raise ValueError("teacher code layout mismatch")
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), sample["codes"].reshape(-1)).item()
        record = {"id": sample["id"], "code_nll": loss, "rh_rms": float(rh[:, :int((~sample['mask']).sum())].square().mean().sqrt())}
        if generate:
            codes = generate_from_zpref(modules["net"], SimpleNamespace(vqvae=modules["vqvae"]), prefix, mask, [0.0] * 11)
            decoded = modules["vqvae"].decode(codes).argmax(-1)
            count = sample["target_count"]
            correct = int((decoded[:, :count] == sample["target"][:, :count]).sum())
            oracle = modules["vqvae"].decode(sample["codes"]).argmax(-1)
            record.update(generated_ids=decoded[0].tolist(), generated_correct=correct, target_count=count, oracle_correct=int((oracle[:, :count] == sample["target"][:, :count]).sum()))
        records.append(record)
    return records


def means_by_text(rows, metric):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row[metric])
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def run(root):
    experiment = initialize_paths(root)
    cases = verify_frozen(root)
    _, _, _, write_json = helpers(root)
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261112)
    results = experiment / "results"
    results.mkdir(exist_ok=False)
    backbone, link, tokenizer = load_c2(root)
    modules = load_c3(root)
    discovery = [case for case in cases if case["split"] == "discovery"]
    validation = [case for case in cases if case["split"] == "heldout"]
    samples = c2_samples(backbone, link, tokenizer, discovery)
    predicted = torch.cat([sample["predicted"].float() for sample in samples]).numpy()
    targets = torch.cat([sample["embedding"].float() for sample in samples]).numpy()
    gains = {"original": 1.0, "least_squares": least_squares_gain(predicted, targets), "norm_match": float(np.sqrt(np.mean(targets ** 2) / np.mean(predicted ** 2)))}
    c2_paths = {}
    c2_discovery = {}
    for name, gain in gains.items():
        path = patch_checkpoint(root, experiment / "inputs" / "1-adapter.pt", "c2-" + name, "c2", gain)
        c2_paths[name] = path
        link.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        records = c2_evaluate(backbone, link, samples)
        c2_discovery[name] = records
        write_json(results / f"c2-discovery-{name}.json", records)
    c2_selected = min(("least_squares", "norm_match"), key=lambda name: sum(row["nll"] for row in c2_discovery[name]))
    write_json(results / "C2_SELECTION.json", {"gains": gains, "selected": c2_selected})
    print("C2_DISCOVERY " + json.dumps({"gains": gains, "selected": c2_selected}), flush=True)

    link.load_state_dict(torch.load(c2_paths["original"], weights_only=True, map_location="cpu"), strict=True)
    heldout_samples = c2_samples(backbone, link, tokenizer, validation)
    c2_heldout = {}
    for name in ("original", c2_selected, "zero_input", "true_token"):
        if name in c2_paths:
            link.load_state_dict(torch.load(c2_paths[name], weights_only=True, map_location="cpu"), strict=True)
        records = c2_evaluate(backbone, link, heldout_samples, mode=name if name in ("zero_input", "true_token") else "link")
        c2_heldout[name] = records
        write_json(results / f"c2-heldout-{name}.json", records)
    c2_means = {name: means_by_text(records, "nll") for name, records in c2_heldout.items()}
    c2_tests = {name: sign_test(list(c2_means[c2_selected].values()), list(c2_means[name].values())) for name in ("original", "zero_input")}
    c2_pass = all(test["one_sided_p"] <= 0.05 for test in c2_tests.values()) and all(sum(c2_means[c2_selected].values()) < sum(c2_means[name].values()) for name in ("original", "zero_input"))

    examples = [c3_example(modules, backbone, tokenizer, case) for case in discovery]
    c3_discovery = {}
    c3_paths = {}
    for factor in (0.5, 1.0, 2.0):
        name = str(factor)
        path = patch_checkpoint(root, experiment / "inputs" / "2-lfm3_ncm_core_bf16.pt", "c3-" + name, "c3", factor)
        c3_paths[name] = path
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        modules["combo"].load_state_dict(state["lfm3_combo_bridge"], strict=True)
        records = c3_evaluate(modules, examples)
        c3_discovery[name] = records
        write_json(results / f"c3-discovery-{name}.json", records)
    c3_selected = min(("0.5", "2.0"), key=lambda name: sum(row["code_nll"] for row in c3_discovery[name]))
    write_json(results / "C3_SELECTION.json", {"selected": c3_selected})
    print("C3_DISCOVERY " + json.dumps({"selected": c3_selected}), flush=True)
    heldout_examples = [c3_example(modules, backbone, tokenizer, case) for case in validation]
    c3_heldout = {}
    for name in ("1.0", c3_selected):
        state = torch.load(c3_paths[name], map_location="cpu", weights_only=True, mmap=True)
        modules["combo"].load_state_dict(state["lfm3_combo_bridge"], strict=True)
        records = c3_evaluate(modules, heldout_examples)
        c3_heldout[name] = records
        write_json(results / f"c3-heldout-{name}.json", records)
    c3_test = sign_test([row["code_nll"] for row in c3_heldout[c3_selected]], [row["code_nll"] for row in c3_heldout["1.0"]])
    counts = {name: {"correct": sum(row["generated_correct"] for row in records), "total": sum(row["target_count"] for row in records), "oracle_correct": sum(row["oracle_correct"] for row in records)} for name, records in c3_heldout.items()}
    c3_pass = c3_test["one_sided_p"] <= 0.05 and counts[c3_selected]["correct"] >= counts["1.0"]["correct"] and sum(row["code_nll"] for row in c3_heldout[c3_selected]) < sum(row["code_nll"] for row in c3_heldout["1.0"])
    summary = {
        "c2": {"gains": gains, "selected": c2_selected, "heldout_mean_nll": {name: sum(values.values()) / len(values) for name, values in c2_means.items()}, "paired_tests": c2_tests, "passes_feedback_gate": c2_pass},
        "c3": {"selected_bridge_factor": c3_selected, "heldout_mean_code_nll": {name: sum(row["code_nll"] for row in records) / len(records) for name, records in c3_heldout.items()}, "generated_tokens": counts, "paired_test": c3_test, "passes_gate": c3_pass},
        "qualification": "Held out from this calibration, not an end-to-end planning or broad language-quality validation. No deployment.",
    }
    verify_frozen(root)
    write_json(results / "SUMMARY.json", summary)
    print("WEIGHT_EXPERIMENT_COMPLETE " + json.dumps(summary), flush=True)


class ProtocolTests(unittest.TestCase):
    def test_disjoint_texts(self):
        self.assertEqual(len(TEXTS), 40)
        self.assertEqual(len(set(TEXTS)), 40)
        self.assertFalse(set(TEXTS[:20]) & set(TEXTS[20:]))

    def test_sign_test_controls(self):
        self.assertEqual(sign_test([0] * 6, [1] * 6)["one_sided_p"], 1 / 64)
        self.assertEqual(sign_test([1] * 6, [1] * 6)["one_sided_p"], 1.0)
        self.assertEqual(sign_test([2] * 6, [1] * 6)["one_sided_p"], 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke-case", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.smoke_case:
        initialize_paths(args.root)
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        backbone, link, tokenizer = load_c2(args.root)
        modules = load_c3(args.root)
        case = {"id": "discovery-smoke", "text": TEXTS[0]}
        samples = c2_samples(backbone, link, tokenizer, [case])
        c2_result = c2_evaluate(backbone, link, samples)
        example = c3_example(modules, backbone, tokenizer, case)
        c3_result = c3_evaluate(modules, [example])
        print(json.dumps({"c2": c2_result, "c3": c3_result}), flush=True)
    elif args.root and args.freeze:
        freeze(args.root.resolve(strict=True))
    elif args.root and args.run:
        run(args.root.resolve(strict=True))
    else:
        parser.error("choose --self-test or --root with --freeze/--run")


if __name__ == "__main__":
    main()