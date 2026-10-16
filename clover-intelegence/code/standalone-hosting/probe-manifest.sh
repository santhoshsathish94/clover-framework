#!/bin/sh
# Where the 54 GB actually sits, by slot, from the static manifests.
# No engine run needed - this is on-disk structure.
D=/opt/clover-k3/clover-intelegence/dataset

echo "===== one KDA layer manifest (trunk-46) ====="
head -c 3000 $D/trunk-46/manifest.json
echo
echo "..."

echo
echo "===== one MLA layer manifest (trunk-59), keys only ====="
python3 - <<'PY'
import json
for n in (46, 59, 0):
    p = "/opt/clover-k3/clover-intelegence/dataset/trunk-%d/manifest.json" % n
    try:
        m = json.load(open(p))
    except Exception as e:
        print(n, "unreadable", e); continue
    print("--- trunk-%d: top-level keys: %s" % (n, list(m)[:12]))
PY
