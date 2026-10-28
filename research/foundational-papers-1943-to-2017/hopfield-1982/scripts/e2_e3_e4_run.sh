#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/hopfield-1982"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"

echo "=== 1. Run pauljmello/Hopfield-Networks in a work copy (with plot_results=False & check torch.sign(0)==0) ==="
rm -rf "$BASE/work/pauljmello_run"
cp -r "$BASE/repo/pauljmello-Hopfield-Networks" "$BASE/work/pauljmello_run"
cd "$BASE/work/pauljmello_run"
sed -i 's/plot_results = True/plot_results = False/g' main.py
sed -i 's/from utils import/# from utils import/g' main.py
$PY -c "
import torch
from hopfield import HopfieldNetwork
from modernhopfield import ModernHopfieldNetwork

# 1. Test traditional & modern Hopfield on 3 orthogonal-ish patterns of size 100:
torch.manual_seed(1982)
pats = torch.sign(torch.randn(3, 100))
net = HopfieldNetwork(100).store(pats)
noisy = pats[0].clone()
noisy[:15] *= -1
rec, energies = net.update(noisy, steps=300)
print('pauljmello Traditional Hopfield: initial overlap=', net.get_overlap(noisy).tolist(),
      '-> recovered overlap=', net.get_overlap(rec).tolist(), 'dE=', energies[-1] - energies[0])

mnet = ModernHopfieldNetwork(100, beta=8.0).store(pats)
mrec, menergies = mnet.update(noisy, steps=5)
print('pauljmello Modern Hopfield:      recovered overlap=', [round(x, 4) for x in mnet.get_overlap(torch.sign(mrec)).tolist()])

# 2. Check what happens in pauljmello HopfieldNetwork when n=2 patterns have local_field == 0:
p2 = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])
net2 = HopfieldNetwork(4).store(p2)
# For neuron 0, weights[0] = [0, 0.5, 0, 0]. If state = [1, 0, 1, -1], local_field = 0 -> torch.sign(0.0) == 0.0!
s_tie = torch.tensor([1.0, -1.0, 1.0, 1.0])  # local_field at i=2 is w[2] @ s_tie = 0*1 + 0*(-1) + 0*1 + 0.5*1 = 0.5; at i=0 with s=[1,0,1,-1]:
h_at_0 = torch.einsum('j,j->', net2.weights[0], torch.tensor([1.0, -1.0, 1.0, -1.0]))
print('pauljmello torch.sign tie check on n=2 patterns: local_field =', float(net2.weights[2] @ torch.tensor([1.0, -1.0, 1.0, 0.0])), '-> torch.sign(0.0) =', float(torch.sign(torch.tensor(0.0))))
"

echo "=== 2. Run E2, E3, and E4 (hopfield1982_experiments.py) ==="
cd "$BASE/work"
$PY hopfield1982_experiments.py
