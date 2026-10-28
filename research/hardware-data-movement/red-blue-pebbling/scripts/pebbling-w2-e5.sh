#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
cd "$W2"

/root/ncm-cpu/venv/bin/python - << 'EOF'
from dag_lib import chain_dag, diamond_dag, binary_tree_dag, pyramid_dag
from exact_solver import solve_dijkstra
from s_partition import (
    schedule_to_2s_partition, check_s_partition, exact_min_partitions_P
)

graphs = [
    chain_dag(4),
    chain_dag(6),
    diamond_dag(),
    binary_tree_dag(3),
    pyramid_dag(3),
    pyramid_dag(4),
]

print("=== (a) THEOREM 3.1 VERIFICATION: CUTTING E3 OPTIMAL SCHEDULES INTO S-I/O SEGMENTS ===")
print(f"{'Graph':16s} {'S':>2s} {'Q*':>3s} {'h':>2s} | {'2S-Valid':>8s} {'max|D_i|':>8s} {'max|VRuVBR|':>11s} {'max|M_i|':>8s} {'2S':>4s} | {'S(h-1)<Q*<=Sh':>14s} | Partition V_i")
for g in graphs:
    for S in [2, 3, 4]:
        q, sched, _ = solve_dijkstra(g, S, return_schedule=True, dead_pebble_prune=False)
        if q is None:
            continue
        V_list, seg_info = schedule_to_2s_partition(g, S, sched)
        ok, info = check_s_partition(g, V_list, 2 * S)
        assert ok, f"2S-partition check failed on {g.name} S={S}: {info}"
        h = info["h"]
        max_d = max(info["dom_sizes"])
        max_vr = max(s["V_R_union_V_BR_size"] for s in seg_info)
        max_m = max(info["min_sizes"])
        ineq_ok = (S * (h - 1) <= q <= S * h)
        parts_str = ", ".join(str(sorted(p)) for p in info["parts"])
        print(f"{g.name:16s} {S:2d} {q:3d} {h:2d} | {str(ok):>8s} {max_d:8d} {max_vr:11d} {max_m:8d} {2*S:4d} | {str(ineq_ok):>14s} | {parts_str}")

print("\n=== NEGATIVE CONTROL FOR 2S-PARTITION CHECKER ===")
g_bt = binary_tree_dag(3)
# 1. Invalid partition with a cycle: V_1 = {0, 1, 2, 3, 6}, V_2 = {4, 5} (edges 0->4 in V1->V2 and 4->6 in V2->V1!)
ok_cyc, err_cyc = check_s_partition(g_bt, [{0, 1, 2, 3, 6}, {4, 5}], K=6)
print(f"  Cyclic partition on bintree_7 (V1={{0,1,2,3,6}}, V2={{4,5}}): ok={ok_cyc}, err={err_cyc['error']}")
# 2. Invalid K-partition where |D_1| > K: V_1 = {0,1,2,3,4,5,6} with K=2 (needs |D_1|=4 > 2)
ok_dom, err_dom = check_s_partition(g_bt, [{0, 1, 2, 3, 4, 5, 6}], K=2)
print(f"  Oversized dominator on bintree_7 (V1=V with K=2): ok={ok_dom}, err={err_dom['error']}")

print("\n=== (b) EXACT P(K) FOR K=1..6 AND TIGHTNESS OF LEMMA 3.1: Q* >= S * (P(2S) - 1) ===")
print("First, exact P(K) table across K=1..6:")
for g in graphs:
    p_vals = []
    for K in [1, 2, 3, 4, 5, 6]:
        pk, parts = exact_min_partitions_P(g, K)
        ok, _ = check_s_partition(g, parts, K)
        assert ok
        p_vals.append(f"P({K})={pk}")
    print(f"  {g.name:16s} (n={g.num_nodes:2d}, |I|={len(g.inputs)}, |O|={len(g.outputs)}): " + "  ".join(p_vals))

print("\nComparison of exact Q*(G, S) against Lemma 3.1 bound S*(P(2S)-1) and against S*(P(S)-1):")
print(f"{'Graph':16s} {'S':>2s} | {'Q*':>4s} | {'P(2S)':>5s} {'S*(P(2S)-1)':>12s} | {'P(S)':>5s} {'S*(P(S)-1)':>11s} | {'Q* >= S(P(2S)-1)':>16s}")
for g in graphs:
    for S in [1, 2, 3, 4]:
        q, _, _ = solve_dijkstra(g, S, return_schedule=False)
        p2s, _ = exact_min_partitions_P(g, 2 * S)
        ps, _ = exact_min_partitions_P(g, S)
        lb_2s = S * (p2s - 1)
        lb_s = S * (ps - 1)
        q_str = str(q) if q is not None else "INF"
        holds = True if q is None else (q >= lb_2s)
        print(f"{g.name:16s} {S:2d} | {q_str:>4s} | {p2s:5d} {lb_2s:12d} | {ps:5d} {lb_s:11d} | {str(holds):>16s}")
EOF
