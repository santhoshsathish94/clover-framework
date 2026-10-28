#!/usr/bin/env bash
set -euo pipefail

REPO="/root/dm-study/flashattention/repo/flash-attention"
W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e3"
PY="/root/ncm-cpu/venv/bin/python"
export PYTHONPATH="$W2/pylib:${PYTHONPATH:-}"

rm -rf "$WORK"
mkdir -p "$WORK"

cp "$REPO/flash_attn/flash_attn_triton.py" "$WORK/orig_flash_attn_triton.py"

"$PY" - "$WORK/orig_flash_attn_triton.py" "$WORK/patched_flash_attn_triton.py" "$WORK/bwd_patched_flash_attn_triton.py" << 'EOF'
import sys
src, e2_dst, bwd_dst = sys.argv[1:4]
text = open(src).read()
text = text.replace(
    'assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"',
    'assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], "Support fp16, bf16, fp32"'
)
text = text.replace('assert q.is_cuda and k.is_cuda and v.is_cuda', '# assert q.is_cuda')
text = text.replace('assert bias.is_cuda', '# assert bias.is_cuda')
text = text.replace('tl.dot(q, k, trans_b=True)', 'tl.dot(q, tl.trans(k))')
text = text.replace('tl.dot(do, v, trans_b=True)', 'tl.dot(do, tl.trans(v))')
text = text.replace('tl.dot(p.to(do.dtype), do, trans_a=True)', 'tl.dot(tl.trans(p.to(do.dtype)), do)')
text = text.replace('tl.dot(ds, q, trans_a=True)', 'tl.dot(tl.trans(ds), q)')
open(e2_dst, 'w').write(text)

# For bwd_patched_flash_attn_triton.py: disable @triton.autotune on _bwd_kernel (just like _fwd_kernel)
# and add thin wrapper functions run_fwd_With_blocks and run_bwd_with_blocks that accept block_m, block_n
bwd_text = text.replace('@triton.autotune(\n    configs=[', '# @triton.autotune(\n#   configs=[')
bwd_text = bwd_text.replace(
    '    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],\n)',
    '# )'
)
# Comment out the interior of configs=[...] in _bwd_kernel
lines = bwd_text.splitlines()
out_lines = []
in_autotune = False
for line in lines:
    if line.startswith('# @triton.autotune('):
        in_autotune = True
        out_lines.append(line)
        continue
    if in_autotune:
        out_lines.append('# ' + line if not line.startswith('#') else line)
        if line.strip() == '# )':
            in_autotune = False
        continue
    out_lines.append(line)

helper_code = """

def run_fwd_With_blocks(q, k, v, causal=False, block_m=128, block_n=128):
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    softmax_scale = 1.0 / math.sqrt(d)
    seqlen_q_rounded = math.ceil(seqlen_q / block_m) * block_m
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = (triton.cdiv(seqlen_q, block_m), batch * nheads)
    _fwd_kernel[grid](
        q, k, v, None, o, lse, tmp, softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        0, 0, 0,
        o.stride(0), o.stride(2), o.stride(1),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
        seqlen_q // 32, seqlen_k // 32,
        "none", causal, BLOCK_HEADDIM,
        BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=1
    )
    return o, lse, softmax_scale

def run_bwd_with_blocks(do, q, k, v, o, lse, causal=False, block_m=128, block_n=128, seq_par=False, zero_delta=False):
    do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    seqlen_q_rounded = math.ceil(seqlen_q / block_m) * block_m
    softmax_scale = 1.0 / math.sqrt(d)
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid_pre = (triton.cdiv(seqlen_q, block_m), batch * nheads)
    _bwd_preprocess_do_o_dot[grid_pre](
        o, do, delta,
        o.stride(0), o.stride(2), o.stride(1),
        do.stride(0), do.stride(2), do.stride(1),
        nheads, seqlen_q, seqlen_q_rounded, d,
        BLOCK_M=block_m, BLOCK_HEADDIM=BLOCK_HEADDIM
    )
    if zero_delta:
        delta.zero_()
    grid_bwd = (triton.cdiv(seqlen_k, block_n) if seq_par else 1, batch * nheads)
    _bwd_kernel[grid_bwd](
        q, k, v, None, do, dq_accum, dk, dv, lse, delta, softmax_scale,
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1),
        0, 0, 0,
        do.stride(0), do.stride(2), do.stride(1),
        dq_accum.stride(0), dq_accum.stride(2), dq_accum.stride(1),
        dk.stride(0), dk.stride(2), dk.stride(1),
        dv.stride(0), dv.stride(2), dv.stride(1),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
        seqlen_q // 32, seqlen_k // 32,
        "none", causal, BLOCK_HEADDIM,
        SEQUENCE_PARALLEL=seq_par, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=1
    )
    return dq_accum.to(q.dtype), dk, dv
"""
open(bwd_dst, 'w').write('\n'.join(out_lines) + helper_code)
EOF

cp "$W2/e3_triton_bwd_and_counts.py" "$WORK/e3_triton_bwd_and_counts.py"
TRITON_INTERPRET=1 "$PY" "$WORK/e3_triton_bwd_and_counts.py" "$WORK"
