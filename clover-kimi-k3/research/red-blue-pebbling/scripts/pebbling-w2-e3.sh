#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
mkdir -p "$W2"
cd "$W2"

/root/ncm-cpu/venv/bin/python - << 'EOF'
from paper_engine import PaperPebbleEngine, PaperMove
from dag_lib import (
    chain_dag, v_tree_dag, diamond_dag, binary_tree_dag,
    pyramid_dag, matmul_2x2_dag
)
from exact_solver import solve_dijkstra, solve_raw_bfs


def verify_schedule(dag, S, moves):
    eng = PaperPebbleEngine(dag.num_nodes, dag.preds, S=S, inputs=dag.inputs, outputs=dag.outputs)
    for i, mv in enumerate(moves):
        ok, _, err = eng.step(mv)
        if not ok:
            return False, eng.io_cost, f"Step {i+1} ({mv}) failed: {err}"
    if not eng.is_terminal():
        return False, eng.io_cost, f"Not in terminal state: red={eng.red}, blue={eng.blue}, expected blue={dag.outputs}"
    return True, eng.io_cost, "OK"


print("=== (a) & (b) HAND-DERIVED TINY GRAPHS VS DIJKSTRA VS RAW BFS + SCHEDULE VERIFIER ===")
tiny_graphs = [
    chain_dag(3),
    v_tree_dag(),
    diamond_dag(),
    binary_tree_dag(3),
]
print(f"{'Graph':14s} {'S':>2s} | {'Dijkstra':>8s} {'Pruned':>8s} {'RawBFS':>8s} | {'Verified':>8s} {'States':>8s}")
for g in tiny_graphs:
    for S in [1, 2, 3, 4, 5]:
        q_dijk, sched, n_states = solve_dijkstra(g, S, return_schedule=True, dead_pebble_prune=False)
        q_prun, sched_p, _ = solve_dijkstra(g, S, return_schedule=True, dead_pebble_prune=True)
        q_raw = solve_raw_bfs(g, S)
        assert q_dijk == q_prun == q_raw, f"Mismatch on {g.name} S={S}: {q_dijk} vs {q_prun} vs {q_raw}"
        if sched is not None:
            ok, v_cost, msg = verify_schedule(g, S, sched)
            assert ok and v_cost == q_dijk, f"Verify failed: {msg}"
            ver_str = f"OK({v_cost})"
        else:
            ver_str = "INFEAS"
        print(f"{g.name:14s} {S:2d} | {str(q_dijk):>8s} {str(q_prun):>8s} {str(q_raw):>8s} | {ver_str:>8s} {n_states:8d}")

print("\n=== (c) NEGATIVE CONTROL 1: DELIBERATELY BROKEN RULE (R3 WITH >=1 RED PREDECESSOR) ===")
for g in [diamond_dag(), binary_tree_dag(3)]:
    for S in [1, 2, 3, 4]:
        q_exact, _, _ = solve_dijkstra(g, S, broken_any_pred=False)
        q_broken, _, _ = solve_dijkstra(g, S, broken_any_pred=True)
        print(f"  {g.name:12s} S={S}: exact Q*={str(q_exact):>4s} vs broken_any_pred Q={str(q_broken):>4s}")

print("\n=== (d) NEGATIVE CONTROL 2: ILLEGAL SCHEDULES REJECTED BY VERIFIER ===")
g_dia = diamond_dag()
_, valid_sched, _ = solve_dijkstra(g_dia, 3, return_schedule=True)
ok0, c0, m0 = verify_schedule(g_dia, 3, valid_sched)
print(f"  Valid diamond S=3 schedule ({len(valid_sched)} moves): ok={ok0}, cost={c0}, msg={m0}")
# Mutate 1: drop the first move R1(0)
bad1 = valid_sched[1:]
ok1, _, m1 = verify_schedule(g_dia, 3, bad1)
print(f"  Mutated schedule (missing R1(0)): ok={ok1}, msg={m1}")
# Mutate 2: drop the last move (leaving a non-output or red pebble at end)
bad2 = valid_sched[:-1]
ok2, _, m2 = verify_schedule(g_dia, 3, bad2)
print(f"  Mutated schedule (missing final R4 cleanup): ok={ok2}, msg={m2}")
# Mutate 3: run valid S=3 schedule under S=2
ok3, _, m3 = verify_schedule(g_dia, 2, valid_sched)
print(f"  Valid S=3 schedule run with S=2 limit: ok={ok3}, msg={m3}")

print("\n=== (e) CANONICAL DAGS SWEEP (CHAIN, TREE, DIAMOND, PYRAMIDS) ===")
canon_graphs = [
    chain_dag(4),
    chain_dag(6),
    diamond_dag(),
    binary_tree_dag(3),
    pyramid_dag(3),
    pyramid_dag(4),
]
for g in canon_graphs:
    row = []
    for S in [1, 2, 3, 4, 5, 6]:
        q, sched, st = solve_dijkstra(g, S, return_schedule=True, dead_pebble_prune=True)
        if g.num_nodes <= 6:
            q_unp, _, _ = solve_dijkstra(g, S, return_schedule=False, dead_pebble_prune=False)
            assert q == q_unp
        if sched is not None:
            ok, vc, msg = verify_schedule(g, S, sched)
            assert ok and vc == q, msg
        row.append(f"S={S}:{q if q is not None else 'INF'}")
    print(f"  {g.name:16s} (n={g.num_nodes:2d}, |I|={len(g.inputs)}, |O|={len(g.outputs)}, d={g.max_fan_in}): " + "  ".join(row))
EOF
