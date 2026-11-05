"""Which of the 70 tasks each Clover layer acts on (no LFM3 run): memory/ontology gates, compiler verdict on a base run."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import lfm3_layers  # noqa: E402

kit = lfm3_layers.LayerKit(HERE.parent)
tasks = json.loads((HERE / "tasks70.json").read_text(encoding="utf-8"))
base = {}
if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    for ln in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            base[r["task_id"]] = r
out = []
for t in tasks:
    q = t["prompt"]
    mb, mi = kit.memory_context(q)
    ob, oi = kit.ontology_context(q)
    row = {"task_id": t["task_id"], "category": t["category"], "memory": mi, "ontology": oi}
    if t["task_id"] in base:
        r = base[t["task_id"]]
        row["compiler"] = kit.compiler_check(q, r["raw_output"])[1]
        g, ch = kit.grounding_filter(q, r["raw_output"])
        row["grounding_changed"] = ch
    out.append(row)
    flags = [k for k in ("memory", "ontology") if row[k]]
    if row.get("compiler", {}).get("checked") and not row["compiler"]["ok"]:
        flags.append("compiler-reject")
    if row.get("grounding_changed"):
        flags.append("grounding")
    print(f"{t['task_id']:36s} {t['category']:22s} {flags}")
    if mb:
        print("   MEMORY:", mb[:600].replace("\n", " | "))
    if ob:
        print("   ONTOLOGY:", ob[:600].replace("\n", " | "))
    if row.get("compiler", {}).get("diagnostics"):
        print("   COMPILER:", row["compiler"]["diagnostics"])
(HERE.parent / "verify" / "layers").mkdir(parents=True, exist_ok=True)
(HERE.parent / "verify" / "layers" / "gates.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("memory fires:", sum(1 for r in out if r["memory"]), " ontology fires:", sum(1 for r in out if r["ontology"]),
      " compiler rejects:", sum(1 for r in out if r.get("compiler", {}).get("ok") is False),
      " grounding changes:", sum(1 for r in out if r.get("grounding_changed")))
