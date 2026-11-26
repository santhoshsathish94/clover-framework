import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np
import torch
from safetensors import safe_open

from interventions import read_header, sha256, sparse_directional_edit, tensor_audit, validate_spec, verify_original, verify_other_files, distinct_files
from run_trials import child, choose_candidate


DOSES = (4.0, 6.0)


def make_copy(root, dose, name):
    root = Path(root)
    if dose not in DOSES or Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid follow-up dose or copy name")
    trials = json.loads((root / "improvement" / "TRIALS.json").read_text())
    if not trials["selected"] or trials["selected"]["layer"] != 22:
        raise ValueError("Conditional layer22 follow-up is not eligible")
    parent_path = root / "improvement" / "proposals" / "layer22-a2p0.json"
    parent = validate_spec(root, parent_path)
    if parent["layer"] != 22:
        raise ValueError("Wrong parent hypothesis")
    manifest = verify_original(root)
    original = root / "original"
    destination = root / "improvement" / "copies" / name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(original, destination, copy_function=shutil.copy2, ignore=shutil.ignore_patterns(".cache"))
    destination.chmod(0o755)
    model_file = destination / "model.safetensors"
    distinct_files(original / "model.safetensors", model_file)
    base_file_hash = sha256(model_file)
    if base_file_hash != sha256(original / "model.safetensors"):
        raise ValueError("Follow-up copy is not exact")
    with safe_open(original / "model.safetensors", framework="pt", device="cpu") as source:
        weight = source.get_tensor(parent["target"])
    before = weight.view(torch.uint8).numpy().tobytes()
    if hashlib.sha256(before).hexdigest() != parent["base_tensor_sha256"]:
        raise ValueError("Parent tensor identity changed")
    with np.load(root / "improvement" / "proposals" / "directions.npz", allow_pickle=False) as data:
        direction = torch.from_numpy(data["direction_22"])
        columns = data["columns_22"]
    if columns.tolist() != parent["selected_columns"]:
        raise ValueError("Follow-up changed column selection")
    replacement = sparse_directional_edit(weight, direction, columns, dose)
    if not torch.isfinite(replacement).all():
        raise ValueError("Nonfinite follow-up tensor")
    after = replacement.view(torch.uint8).numpy().tobytes()
    with model_file.open("rb") as stream:
        entries, _ = read_header(stream, model_file.stat().st_size)
    entry = next(item for item in entries if item["name"] == parent["target"])
    model_file.chmod(0o644)
    with model_file.open("r+b") as stream:
        stream.seek(entry["file_offsets"][0])
        if stream.read(len(before)) != before:
            raise ValueError("Copied target changed")
        stream.seek(entry["file_offsets"][0])
        stream.write(after)
        stream.flush()
        os.fsync(stream.fileno())
    audit = tensor_audit(original / "model.safetensors", model_file, [parent["target"]])
    verify_other_files(destination, manifest)
    audit.update({"base_checkpoint_sha256": base_file_hash, "edited_checkpoint_sha256": sha256(model_file),
                  "target": parent["target"], "file_offsets": entry["file_offsets"],
                  "changed_entries": int((weight != replacement).sum()),
                  "changed_columns": torch.where((weight != replacement).any(0))[0].tolist(),
                  "parent_spec_sha256": sha256(parent_path), "dose": dose,
                  "hypothesis_sha256": sha256(root / "improvement" / "DOSE_FOLLOWUP.md")})
    (destination / "PATCH_AUDIT.json").write_text(json.dumps(audit, indent=2))
    (destination / "FOLLOWUP_SPEC.json").write_text(json.dumps({"parent": parent, "dose": dose, "name": name,
                                                               "hypothesis_sha256": audit["hypothesis_sha256"]}, indent=2))
    model_file.chmod(0o444)
    verify_original(root)
    print("FOLLOWUP_COPY_AUDITED " + json.dumps({"name": name, "dose": dose, "changed_entries": audit["changed_entries"]}), flush=True)


def run(root):
    root = Path(root)
    initial = json.loads((root / "improvement" / "TRIALS.json").read_text())
    if not initial["selected"] or initial["selected"]["layer"] != 22:
        raise ValueError("Conditional follow-up not eligible")
    output = root / "improvement" / "FOLLOWUP_TRIALS.json"
    if output.exists():
        raise FileExistsError(output)
    trials = []
    for dose in DOSES:
        name = f"layer22-followup-a{int(dose)}"
        make_copy(root, dose, name)
        try:
            child(root, "evaluate.py", "--split", "tuning", "--name", name,
                  "--checkpoint", root / "improvement" / "copies" / name)
            result = json.loads((root / "improvement" / "runs" / name / "summary.json").read_text())
            trials.append({"name": name, "layer": 22, "strength": dose, "totals": result["totals"],
                           "results_sha256": result["results_sha256"], "followup": True,
                           "hypothesis_sha256": sha256(root / "improvement" / "DOSE_FOLLOWUP.md")})
            print("FOLLOWUP_TUNING " + json.dumps(trials[-1]), flush=True)
        finally:
            child(root, "interventions.py", "--revert", name)
    selected = choose_candidate(initial["baseline"], initial["trials"] + trials)
    record = {"baseline": initial["baseline"], "trials": initial["trials"] + trials,
              "selected": selected, "initial_trials_sha256": sha256(root / "improvement" / "TRIALS.json"),
              "final_validation_not_run": True, "all_trial_copies_reversed": True,
              "adaptive_followup_disclosed": True}
    output.write_text(json.dumps(record, indent=2))
    print("FOLLOWUP_COMPLETE " + json.dumps({"selected": selected, "new_trials": trials}), flush=True)


class DoseTests(unittest.TestCase):
    def test_alpha_six_is_not_whole_tensor_scaling(self):
        weight = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        edited = sparse_directional_edit(weight, torch.tensor([1.0, 0.0]), [0], 6)
        self.assertEqual(float(edited[0, 0]), -10)
        self.assertTrue(torch.equal(edited[:, 1], weight[:, 1]))
        self.assertTrue(torch.equal(edited[1], weight[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--copy-name")
    parser.add_argument("--dose", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if args.self_test:
        unittest.main(argv=["dose_followup.py"], verbosity=2)
    elif not args.root:
        parser.error("--root required")
    elif args.copy_name:
        make_copy(args.root, args.dose, args.copy_name)
    else:
        run(args.root)