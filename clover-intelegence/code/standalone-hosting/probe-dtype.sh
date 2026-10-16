#!/bin/sh
# Is the trunk int8 because the model is int8, or because our preparation made
# it int8? The model card says MXFP4 weights from SFT onward, and the engine
# already runs an mxfp4 kernel for experts. If the source tensors are mxfp4 and
# we upconverted the trunk, we are streaming twice the bytes the model holds.
D=/opt/clover-k3/clover-intelegence/dataset

echo "===== build-values.json for a KDA layer: what does it record? ====="
head -c 2500 $D/trunk-46/build-values.json
echo
echo "..."

echo
echo "===== record kinds present, with shapes ====="
python3 - <<'PY'
import json, collections
m = json.load(open("/opt/clover-k3/clover-intelegence/dataset/trunk-46/manifest.json"))
k = collections.defaultdict(list)
for r in m["records"]:
    k[r.get("kind")].append((r.get("id"), r.get("rows"), r.get("columns"), r.get("bytes")))
for kind in sorted(k, key=lambda x: (x is None, x)):
    v = k[kind]
    print("kind %s: %d records" % (kind, len(v)))
    for id_, rows, cols, b in v[:3]:
        per = b / float(rows * cols) if rows and cols else 0
        print("    id %-3s %6sx%-6s %12s bytes = %.4f bytes/value" % (id_, rows, cols, b, per))
PY

echo
echo "===== how an expert is stored, for comparison (mxfp4 is known there) ====="
find $D -maxdepth 2 -name 'expert*' -o -maxdepth 2 -name '*expert*' 2>/dev/null | head -5
ls $D | head -20

echo
echo "===== any original model checkpoint on the box? ====="
find /opt/clover-k3 -maxdepth 3 \( -name '*.safetensors' -o -name '*.index.json' \) 2>/dev/null | head -5
