#!/usr/bin/env python3
"""Compute exact mathematical golden answers in Python and verify RTL testbench outputs in E1."""
import pathlib, re
import numpy as np

logdir = pathlib.Path("/root/dm-study/eyeriss-v2/w2/build_fixed")

def conv2d_valid(act: np.ndarray, ker: np.ndarray) -> np.ndarray:
    H, W = act.shape
    R, S = ker.shape
    out = np.zeros((H - R + 1, W - S + 1), dtype=np.int64)
    for r in range(H - R + 1):
        for c in range(W - S + 1):
            out[r, c] = int(np.sum(act[r:r+R, c:c+S] * ker))
    return out

# 1. Check PE_tb: 1D row dot products of w=[1,2,3] with rows of act 1..25 (5x5)
# On iter 4, start=1 is held for #25ns (spanning 2 rising clock edges of 20ns period):
# the 1st posedge sees iter==3 (act_size-kernel_size+1) and resets iter<=0 in IDLE;
# the 2nd posedge (still inside #25ns with start=1) launches iter=0 (14), and iter 5 computes iter=1 (44).
pe_log = (logdir / "PE_tb.run.log").read_text()
pe_psums = [int(x) for x in re.findall(r"Final PSUM of Iteration \d+:\s*(\d+)", pe_log)]
w_row0 = np.array([1, 2, 3])
act_5x5 = np.arange(1, 26).reshape(5, 5)
gold_pe_3 = [int(np.dot(w_row0, act_5x5[i, 0:3])) for i in range(3)]
gold_pe_5 = gold_pe_3 + [gold_pe_3[0], gold_pe_3[1]]
print(f"[PE_tb] RTL={pe_psums} | Golden(iters 1..5 with iter wrap)={gold_pe_5} | match={pe_psums == gold_pe_5}")
assert pe_psums == gold_pe_5

# 2. Check PE_cluster_tb: 5x5 act (1..25) * 3x3 kernel (1..9)
pec_log = (logdir / "PE_cluster_tb.run.log").read_text()
pec_vals = [int(x) for x in re.findall(r"psum from column\s+\d+\s+is:\s*(\d+)", pec_log)][:9]
ker_3x3 = np.arange(1, 10).reshape(3, 3)
gold_pec = conv2d_valid(act_5x5, ker_3x3)
# Note: PE_cluster binds [0:2] to [2:0] in PE_cluster_tb, reversing column order per row
gold_pec_rev = gold_pec[:, ::-1].flatten().tolist()
print(f"[PE_cluster_tb] RTL={pec_vals} | Golden_2D={gold_pec.flatten().tolist()} | Golden_col_reversed={gold_pec_rev} | match_reversed={pec_vals == gold_pec_rev}")
assert pec_vals == gold_pec_rev

# 3. Check router_cluster_pe_cluster_tb: 5x5 act (1..25) * 3x3 all-ones kernel
rcp_log = (logdir / "router_cluster_pe_cluster_tb.run.log").read_text()
rcp_vals = [int(x) for x in re.findall(r"psum from column\s+\d+\s+is:\s*(\d+)", rcp_log)]
rcp_cyc = int(re.search(r"Total #cycles taken:\s*(\d+)", rcp_log).group(1))
gold_rcp = conv2d_valid(act_5x5, np.ones((3, 3), dtype=int)).flatten().tolist()
print(f"[router_cluster_pe_cluster_tb] RTL={rcp_vals} (cycles={rcp_cyc}) | Golden={gold_rcp} | exact_match={rcp_vals == gold_rcp}")
assert rcp_vals == gold_rcp

# 4. Check router_cluster_tb (Phase 1): why does it print [44,44,44, 44,44,44, 20,27,27]?
rct_log = (logdir / "router_cluster_tb.run.log").read_text()
rct_vals = [int(x) for x in re.findall(r"psum from column\s+\d+\s+is:\s*(\d+)", rct_log)]
rct_cyc = int(re.search(r"Total Cycles taken\(including data transfer\):\s*(\d+)", rct_log).group(1))
rct_pe_cyc = int(re.search(r"Total Cycle taken for computation:\s*(\d+)", rct_log).group(1))
# Phase 1 router_weight skips index 0 -> loads w_shifted = [2,3,4, 5,6,7, 8,9,0]
# Phase 1 act_5x5.txt has 20 ones, router_iact skips index 0 -> 19 ones then 6 zeros
w_p1 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0]).reshape(3, 3)
a_p1 = np.array([1]*19 + [0]*6).reshape(5, 5)
gold_p1_true = conv2d_valid(np.ones((5, 5), dtype=int), ker_3x3).flatten().tolist()
gold_p1_bug = conv2d_valid(a_p1, w_p1)[:, ::-1].flatten().tolist()
print(f"[router_cluster_tb Phase 1] RTL={rct_vals} (total_cycles={rct_cyc}, compute_cycles={rct_pe_cyc}) | TrueGolden={gold_p1_true} | BugModel={gold_p1_bug} | matches_bug_model={rct_vals == gold_p1_bug}")
assert rct_vals == gold_p1_bug

# 5. Check router_pe_4_clusters_tb: 4 parallel convolutions
r4_log = (logdir / "router_pe_4_clusters_tb.run.log").read_text()
r4_cyc = int(re.search(r"Total #cycles taken:\s*(\d+)", r4_log).group(1))
m4 = re.search(r"ALL4_LAST_ITER: west0=\((\d+),(\d+),(\d+)\) west1=\((\d+),(\d+),(\d+)\) east0=\((\d+),(\d+),(\d+)\) east1=\((\d+),(\d+),(\d+)\)", r4_log)
rtl_last = {
    "west0": [int(m4.group(1)), int(m4.group(2)), int(m4.group(3))],
    "west1": [int(m4.group(4)), int(m4.group(5)), int(m4.group(6))],
    "east0": [int(m4.group(7)), int(m4.group(8)), int(m4.group(9))],
    "east1": [int(m4.group(10)), int(m4.group(11)), int(m4.group(12))],
}
gold_last = {
    "west0": conv2d_valid(act_5x5 * 5, ker_3x3)[2, ::-1].tolist(),
    "west1": conv2d_valid(act_5x5 + 3, ker_3x3)[2, ::-1].tolist(),
    "east0": conv2d_valid(act_5x5, ker_3x3)[2, ::-1].tolist(),
    "east1": conv2d_valid(act_5x5 + 2, ker_3x3)[2, ::-1].tolist(),
}
print(f"[router_pe_4_clusters_tb] cycles={r4_cyc} | RTL_row2={rtl_last} | Golden_row2={gold_last} | match={rtl_last == gold_last}")
assert rtl_last == gold_last
print("ALL E1 PYTHON GOLDEN CHECKS PASSED.")
