#!/usr/bin/env bash
set -euo pipefail

REPO="/root/dm-study/flashattention/repo/flash-attention"
W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e2"
PY="/root/ncm-cpu/venv/bin/python"
export PYTHONPATH="$W2/pylib:${PYTHONPATH:-}"

rm -rf "$WORK"
mkdir -p "$WORK"

echo "=== 0. Repository Triton attention inventory ==="
git -C "$REPO" rev-parse HEAD
wc -l "$REPO/flash_attn/flash_attn_triton.py" "$REPO/flash_attn/flash_attn_triton_og.py"
find "$REPO" -maxdepth 3 -name "*triton*" | sort

# Copy pristine files into WORK
cp "$REPO/flash_attn/flash_attn_triton.py" "$WORK/orig_flash_attn_triton.py"
cp "$REPO/flash_attn/flash_attn_triton_og.py" "$WORK/orig_flash_attn_triton_og.py"

# Create host-only patched copy (to show the Triton 2.0 vs 3.8 tl.dot(trans_b=True) error)
# and the minimally patched copy for Triton 3.8 in WORK
"$PY" - "$WORK/orig_flash_attn_triton.py" "$WORK/host_patched_flash_attn_triton.py" "$WORK/patched_flash_attn_triton.py" "$WORK/orig_flash_attn_triton_og.py" "$WORK/patched_flash_attn_triton_og.py" << 'EOF'
import sys
src, host_dst, dst, og_src, og_dst = sys.argv[1:6]
text = open(src).read()
# Host-only patch: allow float32 and remove .is_cuda assertions
host_text = text.replace(
    'assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"',
    'assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], "Support fp16, bf16, fp32"'
)
host_text = host_text.replace('assert q.is_cuda and k.is_cuda and v.is_cuda', '# assert q.is_cuda (removed for CPU interpreter)')
host_text = host_text.replace('assert bias.is_cuda', '# assert bias.is_cuda (removed for CPU interpreter)')
open(host_dst, 'w').write(host_text)

# Kernel patch for Triton 3.8: replace tl.dot(..., trans_b=True) and trans_a=True with tl.trans(...)
k_text = host_text.replace('tl.dot(q, k, trans_b=True)', 'tl.dot(q, tl.trans(k))')
k_text = k_text.replace('tl.dot(do, v, trans_b=True)', 'tl.dot(do, tl.trans(v))')
k_text = k_text.replace('tl.dot(p.to(do.dtype), do, trans_a=True)', 'tl.dot(tl.trans(p.to(do.dtype)), do)')
k_text = k_text.replace('tl.dot(ds, q, trans_a=True)', 'tl.dot(tl.trans(ds), q)')
open(dst, 'w').write(k_text)

og_text = open(og_src).read()
og_text = og_text.replace('import pytest', '# import pytest')
og_text = og_text.replace('tl.dot(q, k, trans_b=True)', 'tl.dot(q, tl.trans(k))')
og_text = og_text.replace('tl.dot(do, v, trans_b=True)', 'tl.dot(do, tl.trans(v))')
og_text = og_text.replace('tl.dot(p.to(do.dtype), do, trans_a=True)', 'tl.dot(tl.trans(p.to(do.dtype)), do)')
og_text = og_text.replace('tl.dot(ds.to(q.dtype), q, trans_a=True)', 'tl.dot(tl.trans(ds.to(q.dtype)), q)')
open(og_dst, 'w').write(og_text)
EOF

echo "=== Diff between orig_flash_attn_triton.py and patched_flash_attn_triton.py ==="
diff -u "$WORK/orig_flash_attn_triton.py" "$WORK/patched_flash_attn_triton.py" || true

cp "$W2/e2_triton_fwd.py" "$WORK/e2_triton_fwd.py"
TRITON_INTERPRET=1 "$PY" "$WORK/e2_triton_fwd.py" "$WORK"
