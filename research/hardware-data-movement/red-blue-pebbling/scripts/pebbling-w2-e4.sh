#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
cd "$W2"

/root/ncm-cpu/venv/bin/python - << 'EOF'
import itertools
import random
from dag_lib import make_dag, DAGSpec
from exact_solver import solve_dijkstra
from paper_engine import PaperPebbleEngine


def verify_sched(dag: DAGSpec, S: int, moves, allow_recompute: bool):
    eng = PaperPebbleEngine(dag.num_nodes, dag.preds, S=S, inputs=dag.inputs, outputs=dag.outputs)
    r3_counts = {}
    for i, mv in enumerate(moves):
        if mv.rule == "R3":
            r3_counts[mv.node_id] = r3_counts.get(mv.node_id, 0) + 1
            if not allow_recompute and r3_counts[mv.node_id] > 1:
                return False, f"Node {mv.node_id} computed {r3_counts[mv.node_id]} times under allow_recompute=False"
        ok, _, err = eng.step(mv)
        if not ok:
            return False, f"Step {i+1} ({mv}) failed: {err}"
    if not eng.is_terminal():
        return False, f"Not terminal: red={eng.red}, blue={eng.blue}"
    return True, f"OK (io_cost={eng.io_cost}, max_r3={max(r3_counts.values(), default=0)})"


print("=== 1. HAND-CONSTRUCTED SMALLEST EXAMPLES (G5 AT S=3, G6 AT S=3 AND S=4) ===")
g5 = make_dag("G5_recomp", 5, [(0, 1), (1, 2), (0, 3), (2, 3), (1, 4), (3, 4)])
g6 = make_dag("G6_recomp", 6, [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (4, 5)])

for g, s_list in [(g5, [3, 4]), (g6, [3, 4])]:
    for S in s_list:
        q_rec, sched_rec, st_rec = solve_dijkstra(g, S, allow_recompute=True, return_schedule=True, dead_pebble_prune=False)
        q_nor, sched_nor, st_nor = solve_dijkstra(g, S, allow_recompute=False, return_schedule=True, dead_pebble_prune=False)
        ok_r, msg_r = verify_sched(g, S, sched_rec, True) if sched_rec else (True, "INFEAS")
        ok_n, msg_n = verify_sched(g, S, sched_nor, False) if sched_nor else (True, "INFEAS")
        print(f"{g.name} (n={g.num_nodes}) S={S}: Q*_recomp={q_rec} ({msg_r}) vs Q*_no_recomp={q_nor} ({msg_n})")
        if g.name == "G5_recomp" and S == 3:
            print("  --- Move-by-move optimal schedule on G5 (S=3) WITH recomputation (Q*=2) ---")
            eng = PaperPebbleEngine(g.num_nodes, g.preds, S=S, inputs=g.inputs, outputs=g.outputs)
            for idx, mv in enumerate(sched_rec, 1):
                eng.step(mv)
                print(f"    {idx:2d}. {mv.rule:2s}(v={mv.node_id}{','+mv.color if mv.color else ''}) -> red={sorted(eng.red)}, blue={sorted(eng.blue)}, io={eng.io_cost}")
            print("  --- Move-by-move optimal schedule on G5 (S=3) WITHOUT recomputation (Q*=4) ---")
            eng = PaperPebbleEngine(g.num_nodes, g.preds, S=S, inputs=g.inputs, outputs=g.outputs)
            for idx, mv in enumerate(sched_nor, 1):
                eng.step(mv)
                print(f"    {idx:2d}. {mv.rule:2s}(v={mv.node_id}{','+mv.color if mv.color else ''}) -> red={sorted(eng.red)}, blue={sorted(eng.blue)}, io={eng.io_cost}")


def enum_topological_dags(n: int):
    """All topological DAGs on 0..n-1 where 0 is input and each v>=1 has 1 or 2 preds in 0..v-1."""
    choices_per_v = []
    for v in range(1, n):
        opts = [(u,) for u in range(v)] + list(itertools.combinations(range(v), 2))
        choices_per_v.append(opts)
    for combo in itertools.product(*choices_per_v):
        edges = []
        for v_idx, preds in enumerate(combo, start=1):
            for p in preds:
                edges.append((p, v_idx))
        yield make_dag(f"enum_n{n}", n, edges)


print("\n=== 2. EXHAUSTIVE SEARCH OVER ALL TOPOLOGICAL DAGS (n=4, 5, 6; fan-in 1..2) ===")
for n in [4, 5, 6]:
    for S in [2, 3, 4]:
        total = 0
        both_feas = 0
        recomp_strictly_better = 0
        only_recomp_feas = 0
        max_gap = 0
        example_edges = None
        for dag in enum_topological_dags(n):
            total += 1
            qr, _, _ = solve_dijkstra(dag, S, allow_recompute=True, dead_pebble_prune=False)
            qn, _, _ = solve_dijkstra(dag, S, allow_recompute=False, dead_pebble_prune=False)
            if qr is not None and qn is not None:
                both_feas += 1
                assert qr <= qn, f"Impossible: qr={qr} > qn={qn}"
                if qr < qn:
                    recomp_strictly_better += 1
                    if qn - qr > max_gap:
                        max_gap = qn - qr
                        example_edges = [(u, v) for v in range(n) for u in dag.preds[v]]
            elif qr is not None and qn is None:
                only_recomp_feas += 1
        print(f"  n={n} (total={total:4d}), S={S}: both_feas={both_feas:4d}, Q*_rec < Q*_norec: {recomp_strictly_better:4d} ({100.0*recomp_strictly_better/max(1,both_feas):5.2f}%), only_rec_feas={only_recomp_feas}, max_gap={max_gap}")


print("\n=== 3. RANDOM DAG SEARCH FOR n=7, 8 (500 RANDOM DAGS EACH, S=3, 4, SEED=42) ===")
rng = random.Random(42)
for n in [7, 8]:
    choices_per_v = []
    for v in range(1, n):
        opts = [(u,) for u in range(v)] + list(itertools.combinations(range(v), 2))
        choices_per_v.append(opts)
    samples = []
    for i in range(500):
        edges = []
        for v_idx, opts in enumerate(choices_per_v, start=1):
            preds = rng.choice(opts)
            for p in preds:
                edges.append((p, v_idx))
        samples.append(make_dag(f"rand_n{n}_{i}", n, edges))

    for S in [3, 4]:
        both_feas = 0
        strictly_better = 0
        only_rec_feas = 0
        max_gap = 0
        for dag in samples:
            qr, _, _ = solve_dijkstra(dag, S, allow_recompute=True, dead_pebble_prune=False)
            qn, _, _ = solve_dijkstra(dag, S, allow_recompute=False, dead_pebble_prune=False)
            if qr is not None and qn is not None:
                both_feas += 1
                assert qr <= qn
                if qr < qn:
                    strictly_better += 1
                    if qn - qr > max_gap:
                        max_gap = qn - qr
            elif qr is not None and qn is None:
                only_rec_feas += 1
        print(f"  n={n} (samples=500), S={S}: both_feas={both_feas:3d}, Q*_rec < Q*_norec: {strictly_better:3d} ({100.0*strictly_better/max(1,both_feas):5.2f}%), only_rec_feas={only_rec_feas}, max_gap={max_gap}")
EOF
