"""Print a compact summary of a run_cogs.py result file (per split and per gen case)."""
import json
import sys

r = json.load(open(sys.argv[1]))
for split, s in r["splits"].items():
    print(split, "n", s["n"], "coverage", s["coverage"], "predicate", s["predicate_match"], "exact", s["exact_match"],
          "roles", {k: (v["p"], v["r"]) for k, v in s["roles"].items()})
    for k, v in list(s["error_patterns"].items())[:8]:
        print("   ", v, k)
    for case, c in s["by_case"].items():
        print(f"  {case:58s} n={c['n']:5d} cov={c['coverage']:.3f} pred={c['predicate_match']:.3f} "
              f"exact={c['exact_match']:.3f} top_error={c['top_error']}")
