#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/vaswani-2017

echo "=== 1. Run jadore801120/attention-is-all-you-need-pytorch Transformer ==="
cd "$BASE/repo/jadore801120-attention-is-all-you-need-pytorch"
$PY - << 'EOF'
import sys, torch
sys.path.insert(0, ".")
from transformer.Models import Transformer
m = Transformer(
    n_src_vocab=100, n_trg_vocab=100, src_pad_idx=0, trg_pad_idx=0,
    trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
    d_k=64, d_v=64, d_model=512, d_word_vec=512, d_inner=2048,
    n_layers=6, n_head=8, dropout=0.1, scale_emb_or_prj='emb'
)
src = torch.randint(1, 100, (2, 10))
trg = torch.randint(1, 100, (2, 8))
out = m(src, trg)
print("  jadore801120 Transformer output shape:", tuple(out.shape), "total params:", sum(p.numel() for p in m.parameters()))
EOF

echo "=== 2. Run hyunwoongko/transformer Transformer ==="
cd "$BASE/repo/hyunwoongko-transformer"
$PY - << 'EOF'
import sys, torch
sys.path.insert(0, ".")
from models.model.transformer import Transformer
m = Transformer(
    src_pad_idx=0, trg_pad_idx=0, trg_sos_idx=1,
    enc_voc_size=100, dec_voc_size=100, d_model=512, n_head=8,
    max_len=100, ffn_hidden=2048, n_layers=6, drop_prob=0.1, device="cpu"
)
src = torch.randint(1, 100, (2, 10))
trg = torch.randint(1, 100, (2, 8))
out = m(src, trg)
print("  hyunwoongko Transformer output shape:", tuple(out.shape), "total params:", sum(p.numel() for p in m.parameters()))
EOF

echo "=== 3. Run our verified vaswani2017_experiments.py (E2, E3, E4) ==="
cd "$BASE/work"
$PY vaswani2017_experiments.py
echo "FINISHED"
