"""Score saved PropSegmEnt encodings with the released evaluate_propsegment on CPU. The released main() calls it
without passing args.cuda and its default is cuda=True, so CPU runs crash after encoding; the encodings are saved
first (--query_output_path/--target_output_path), and this scores them with the released function, cuda=False.
Usage (PYTHONPATH=<sub-sentence-encoder repo>): python sse_score.py <repo> <queries.jsonl> <targets.jsonl> <name> <dir>
Reads <dir>/<name>_queries.pt and <dir>/<name>_targets.pt; prints the released metrics."""
import importlib.util
import os
import sys

import torch

repo, qpath, tpath, name, d = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("eval_retrieval", os.path.join(repo, "scripts/evaluation/eval_retrieval.py"))
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)  # released module (its __main__ block does not run)

queries, targets = er.read_jsonl(qpath), er.read_jsonl(tpath)
qe = torch.load(os.path.join(d, f"{name}_queries.pt"))
te = torch.load(os.path.join(d, f"{name}_targets.pt"))
print(f"{name}: queries {tuple(qe.shape)}, targets {tuple(te.shape)}")
er.evaluate_propsegment(qe, te, [q["id"] for q in queries], [q["label"] for q in queries], [t["id"] for t in targets],
                        cuda=False)
