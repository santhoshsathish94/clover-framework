"""Print the pre-registered 50-error sample of Experiment 1N (PAWS) compactly for categorisation."""
import json
import sys

r = json.load(open(sys.argv[1]))
for i, e in enumerate(r["error_sample_seed7"]):
    d = "; ".join(f"{x['type']}:{x['a']}/{x['b']}" for x in e["differences"])
    print(f"[{i}] label={e['label']} csi_equivalent={e['equivalent']} diffs={d or '-'}")
    print("   A:", e["a"][:220])
    print("   B:", e["b"][:220])
