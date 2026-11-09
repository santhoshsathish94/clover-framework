import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
print("sha256:", {k: v[:12] for k, v in r["sha256"].items()})
print("gate:", json.dumps(r["gate"])[:600])
for name, d in r["omni_diagnostics"].items():
    if name == "server":
        print("server:", d)
        continue
    print(name, "label tokens", d["label_token_counts"], "bos", d["bos"], "floors", d["floors"], "consistency", d["two_pass_consistency"])
for group in ("omni", "hf_reference"):
    print(f"\n== {group}: per arm at tau 0.6 and 0.9 (all 200: false caution /50, recall /150, 4-way /200, per label)")
    for name, d in r[group].items():
        for tau in ("0.6", "0.9"):
            a = d["by_tau"][tau]["all"]
            c = d["by_tau"][tau]["calibration"]
            t = d["by_tau"][tau]["test"]
            print(f"{name:20} tau {tau}: FC {a['false_caution']:2}/50 recall {a['caution_recall']:3}/150 4way {a['correct_4way']:3}/200 "
                  f"per {a['per_label_correct']} pred {a['predicted']} | cal FC {c['false_caution']}/25 rec {c['caution_recall']}/75 | "
                  f"test FC {t['false_caution']}/25 rec {t['caution_recall']}/75")
