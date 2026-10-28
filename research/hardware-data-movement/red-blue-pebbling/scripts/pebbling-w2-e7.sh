#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
mkdir -p "$W2"
cd "$W2"

/root/ncm-cpu/venv/bin/python - << 'EOF'
import math
from dag_lib import fft_butterfly_dag
from exact_solver import solve_dijkstra
from fft_schedules import (
    solve_dijkstra_bounded,
    build_recursive_blocked_fft_schedule,
    build_fft8_handover_schedule,
    verify_schedule_and_peak_red,
)

print("=== 1. 4-POINT FFT BUTTERFLY (12 VERTICES: 4 INPUTS, 4 STAGE-1, 4 OUTPUTS) EXACT Q*(S) ===")
fft4 = fft_butterfly_dag(4)
for S in range(1, 8):
    cost, sched, states = solve_dijkstra(fft4, S, return_schedule=True, dead_pebble_prune=True)
    if cost is None:
        print(f"  FFT_4 S={S}: Q*=None (infeasible, states_explored={states})")
    else:
        ok, vcost, peak_r, msg = verify_schedule_and_peak_red(fft4, S, sched)
        assert ok and vcost == cost
        r1_in = sum(1 for m in sched if m.rule == "R1" and m.node_id in fft4.inputs)
        r1_mid = sum(1 for m in sched if m.rule == "R1" and m.node_id not in fft4.inputs)
        r2_mid = sum(1 for m in sched if m.rule == "R2" and m.node_id not in fft4.outputs)
        r2_out = sum(1 for m in sched if m.rule == "R2" and m.node_id in fft4.outputs)
        print(f"  FFT_4 S={S}: Q*={cost:2d} (R1_in={r1_in}, R1_mid={r1_mid}, R2_mid={r2_mid}, R2_out={r2_out} | verified={ok}, peak_red={peak_r}, states={states})")

print("\n=== 2. 8-POINT FFT BUTTERFLY (32 VERTICES): EXACT STATE-SPACE BOUND & CONSTRUCTIVE SCHEDULES ===")
fft8 = fft_butterfly_dag(8)
for S in [1, 2, 3, 4, 6]:
    cost, states, hit_cap = solve_dijkstra_bounded(fft8, S, max_states=200000)
    print(f"  FFT_8 S={S:2d} exact 0-1 BFS: cost={str(cost):>4s}, states_explored={states:6d}, hit_200k_cap={hit_cap}")

print("\n  Constructive verified schedules on FFT_8 (32 vertices):")
for m_blk, S_req, label in [(1, 3, "radix-2 pass-by-pass"), (2, 6, "radix-4 + radix-2 pass-by-pass"), (3, 10, "radix-8 full in-register")]:
    mvs = build_recursive_blocked_fft_schedule(8, m_blk)
    ok, io_c, peak_r, msg = verify_schedule_and_peak_red(fft8, S_req, mvs)
    print(f"    {label:32s} (S={S_req:2d}): verified={ok}, IO={io_c:2d}, peak_red={peak_r:2d}, total_moves={len(mvs)}")

for S_ho, label in [(3, "radix-2 with 1-pebble handover"), (6, "radix-4 with 4-pebble handover")]:
    mvs = build_fft8_handover_schedule(S_ho)
    ok, io_c, peak_r, msg = verify_schedule_and_peak_red(fft8, S_ho, mvs)
    print(f"    {label:32s} (S={S_ho:2d}): verified={ok}, IO={io_c:2d}, peak_red={peak_r:2d}, total_moves={len(mvs)} ({msg})")

print("\n=== 3. RECURSIVE RADIX-B BLOCKED FFT SCHEDULES FOR n = 4..256 (VERIFIED BY PAPER ENGINE) ===")
print(f"{'n':>4s} {'N_verts':>7s} {'B':>3s} {'m=log2B':>7s} {'S_req':>5s} {'Passes':>6s} | {'IO_count':>8s} {'PeakRed':>7s} {'Verified':>8s} | {'IO/(n*log2n/log2B)':>18s} {'Naive/Blocked':>13s}")
for n in [4, 8, 16, 32, 64, 128, 256]:
    dag = fft_butterfly_dag(n)
    k = int(math.log2(n))
    naive_io = 2 * n * k
    for m in [1, 2, 3, 4]:
        B = 1 << m
        if B > n:
            continue
        S_req = 3 if m == 1 else (B + 2)
        passes = (k + m - 1) // m
        mvs = build_recursive_blocked_fft_schedule(n, m)
        ok, io_c, peak_r, msg = verify_schedule_and_peak_red(dag, S_req, mvs)
        assert ok, f"Failed on n={n}, m={m}: {msg}"
        norm_c = io_c / (n * (k / m))
        ratio = naive_io / io_c
        print(f"{n:4d} {dag.num_nodes:7d} {B:3d} {m:7d} {S_req:5d} {passes:6d} | {io_c:8d} {peak_r:7d} {str(ok):>8s} | {norm_c:18.3f} {ratio:12.2f}x")
EOF
