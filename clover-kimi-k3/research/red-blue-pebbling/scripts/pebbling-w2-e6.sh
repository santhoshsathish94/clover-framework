#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
mkdir -p "$W2"
cd "$W2"

gcc -O3 -Wall cache_sim.c -lm -o cache_sim
./cache_sim | tee e6_raw.txt

echo ""
echo "=== 5. LOG-LINEAR POWER-LAW FIT OF EXPONENTS IN n AND S ==="
/root/ncm-cpu/venv/bin/python - << 'EOF'
import re
import math
import numpy as np

rows = []
with open("e6_raw.txt") as f:
    for line in f:
        m = re.match(r"n=\s*(\d+)\s+S=\s*(\d+)\s+b=\s*(\d+)\s+\|\s+NaiveLRU=\s*(\d+)\s+NaiveOPT=\s*(\d+)\s+\|\s+BlkLRU=\s*(\d+).*BlkOPT=\s*(\d+)", line)
        if m:
            n, S, b, nlru, nopt, blru, bopt = map(int, m.groups())
            rows.append((n, S, b, nlru, nopt, blru, bopt))

# Fit log(IO) = log(c) + alpha * log(n) + beta * log(S)
def fit_power_law(pts, col_idx, max_n=512):
    sub = [r for r in pts if r[0] <= max_n and r[col_idx] > 0]
    X = np.array([[1.0, math.log(r[0]), math.log(r[1])] for r in sub])
    y = np.array([math.log(r[col_idx]) for r in sub])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    r2 = 1.0 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    return math.exp(coef[0]), coef[1], coef[2], r2

c_lru_all, a_lru_all, b_lru_all, r2_lru_all = fit_power_law(rows, 5, max_n=512)
c_lru_256, a_lru_256, b_lru_256, r2_lru_256 = fit_power_law(rows, 5, max_n=256)
c_opt_256, a_opt_256, b_opt_256, r2_opt_256 = fit_power_law(rows, 6, max_n=256)

print(f"Blocked LRU (n=64..512, S=48..3072, 16 pts): IO = {c_lru_all:.3f} * n^{a_lru_all:.4f} * S^{b_lru_all:.4f}  (R^2={r2_lru_all:.6f})")
print(f"Blocked LRU (n=64..256, S=48..3072, 12 pts): IO = {c_lru_256:.3f} * n^{a_lru_256:.4f} * S^{b_lru_256:.4f}  (R^2={r2_lru_256:.6f})")
print(f"Blocked OPT (n=64..256, S=48..3072, 12 pts): IO = {c_opt_256:.3f} * n^{a_opt_256:.4f} * S^{b_opt_256:.4f}  (R^2={r2_opt_256:.6f})")
EOF
