"""Per live-probe turn: which deterministic Outcome checks applied to the actual reply, against the stage text shown.
Reads results/probe_live.json (probe_live_self_reports.py). Liveness: the form check must fire on a bad reply."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parents[2] / "clover-ai-system" / "studio" / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import clover_cycle as C  # noqa: E402
import lfm3_layers as LL  # noqa: E402

FORM_ASK = re.compile(r"\bone\s+sentence\b|\b(?:in|with|as|exactly)\s+(?:\d+|one|two|three|four|five)\s+sentences\b|"
                      r"\bbullet|\bwords\b|\bjson\b|####|\bround", re.I)
rows = []
for r in json.loads((HERE / "results" / "probe_live.json").read_text(encoding="utf-8")):
    q, a = r["question"], r["reply"]
    applied = (["code"] if LL.extract_python(a).strip() else []) + (
        ["arithmetic"] if C.direct_arithmetic_value(q) is not None or re.search(r"\d\s*[+\-*/x\u00d7]\s*\d.*=", a) else []
    ) + (["form"] if FORM_ASK.search(q) else [])
    rows.append({"key": r["key"], "checks_applied": applied, "findings": C.outcome_findings(q, a),
                 "stage_shown": r["outcome_stage_detail_shown"]})
summary = {
    "turns": len(rows),
    "shown_verified": sum(1 for x in rows if "verified" in (x["stage_shown"] or "")),
    "shown_verified_with_no_check": sum(1 for x in rows if not x["checks_applied"] and "verified" in (x["stage_shown"] or "")),
    "liveness_form_check": C.format_findings("Answer in one sentence.", "Canberra is the capital. It is in the ACT."),
}
(HERE / "results" / "outcome_checks_per_turn.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2),
                                                               encoding="utf-8")
print(json.dumps({"rows": rows, "summary": summary}, indent=2))
