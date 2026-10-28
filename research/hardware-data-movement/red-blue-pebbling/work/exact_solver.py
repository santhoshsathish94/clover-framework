"""
Exact minimum-I/O solvers for Hong & Kung (1981) Red-Blue Pebble Game (E3, E4, E5).
Includes:
1. solve_dijkstra: 0-1 BFS / Dijkstra over bitmask states (supports allow_recompute=True/False,
   broken_any_pred=True/False negative control, and full schedule reconstruction).
2. solve_raw_bfs: slower unreduced 0-1 BFS over full (red_mask, blue_mask) states with explicit
   mid-game R4(blue) transitions, reaching exact terminal state (red=0, blue=outputs_mask).
"""

from __future__ import annotations
from collections import deque
from dag_lib import DAGSpec
from paper_engine import PaperPebbleEngine, PaperMove


def solve_dijkstra(
    dag: DAGSpec,
    S: int,
    allow_recompute: bool = True,
    broken_any_pred: bool = False,
    return_schedule: bool = False,
    dead_pebble_prune: bool = False,
) -> tuple[int | None, list[PaperMove] | None, int]:
    """
    Exact minimum-I/O solver using 0-1 BFS over bitmask states.
    Returns (min_io_cost_or_None, schedule_or_None, states_explored).
    """
    n = dag.num_nodes
    if S <= 0:
        return None, None, 0

    pred_masks = [0] * n
    succ_masks = [0] * n
    for v in range(n):
        pm = 0
        for p in dag.preds[v]:
            pm |= (1 << p)
        pred_masks[v] = pm
        for p in dag.preds[v]:
            succ_masks[p] |= (1 << v)

    inputs_mask = sum(1 << i for i in dag.inputs)
    outputs_mask = sum(1 << i for i in dag.outputs)
    non_inputs = [v for v in range(n) if v not in dag.inputs]

    def clean_dead(red: int, blue: int) -> tuple[int, int, list[PaperMove]]:
        """Apply free R4 moves on dead pebbles (outputs already in blue, and nodes whose succs are all dead)."""
        if not dead_pebble_prune:
            return red, blue, []
        extra: list[PaperMove] = []
        dead = 0
        for o in dag.outputs:
            if succ_masks[o] == 0 and (blue & (1 << o)):
                dead |= (1 << o)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                bit = 1 << v
                if not (dead & bit) and succ_masks[v] != 0 and (succ_masks[v] & dead) == succ_masks[v]:
                    dead |= bit
                    changed = True
        for v in range(n):
            bit = 1 << v
            if dead & bit:
                if red & bit:
                    red ^= bit
                    extra.append(PaperMove("R4", v, "red"))
                if (blue & bit) and (v not in dag.outputs) and (v not in dag.inputs):
                    blue ^= bit
                    extra.append(PaperMove("R4", v, "blue"))
        return red, blue, extra

    # Initial state: red = 0, blue = inputs_mask, comp = inputs_mask (if not allow_recompute)
    init_state = (0, inputs_mask, 0 if allow_recompute else inputs_mask)
    dist: dict[tuple[int, int, int], int] = {init_state: 0}
    parent: dict[tuple[int, int, int], tuple[tuple[int, int, int], list[PaperMove]]] = {}
    dq = deque([init_state])

    goal_state = None
    goal_cost = None

    while dq:
        state = dq.popleft()
        d = dist[state]
        red, blue, comp = state

        if (blue & outputs_mask) == outputs_mask:
            goal_state = state
            goal_cost = d
            break

        red_count = red.bit_count()

        # Cost-0 transitions (pushleft):
        # 1. R4(red): delete any red pebble
        r_tmp = red
        while r_tmp:
            lsb = r_tmp & -r_tmp
            v = lsb.bit_length() - 1
            r_tmp ^= lsb
            ns = (red ^ lsb, blue, comp)
            if d < dist.get(ns, 10**9):
                dist[ns] = d
                if return_schedule:
                    parent[ns] = (state, [PaperMove("R4", v, "red")])
                dq.appendleft(ns)

        # 2. R3 (Compute): place red on non-input v if preds in red and red_count < S
        if red_count < S:
            for v in non_inputs:
                bit = 1 << v
                if red & bit:
                    continue
                if not allow_recompute and (comp & bit):
                    continue
                pm = pred_masks[v]
                ok = ((red & pm) != 0) if broken_any_pred else ((red & pm) == pm)
                if ok:
                    n_red, n_blue, extra = clean_dead(red | bit, blue)
                    n_comp = comp if allow_recompute else (comp | bit)
                    ns = (n_red, n_blue, n_comp)
                    if d < dist.get(ns, 10**9):
                        dist[ns] = d
                        if return_schedule:
                            parent[ns] = (state, [PaperMove("R3", v)] + extra)
                        dq.appendleft(ns)

        # Cost-1 transitions (push right):
        # 3. R1 (Input): place red on v in blue if v not in red and red_count < S
        if red_count < S:
            b_tmp = blue & ~red
            while b_tmp:
                lsb = b_tmp & -b_tmp
                v = lsb.bit_length() - 1
                b_tmp ^= lsb
                ns = (red | lsb, blue, comp)
                nd = d + 1
                if nd < dist.get(ns, 10**9):
                    dist[ns] = nd
                    if return_schedule:
                        parent[ns] = (state, [PaperMove("R1", v)])
                    dq.append(ns)

        # 4. R2 (Output): place blue on v in red if v not in blue
        r_tmp = red & ~blue
        while r_tmp:
            lsb = r_tmp & -r_tmp
            v = lsb.bit_length() - 1
            r_tmp ^= lsb
            n_red, n_blue, extra = clean_dead(red, blue | lsb)
            ns = (n_red, n_blue, comp)
            nd = d + 1
            if nd < dist.get(ns, 10**9):
                dist[ns] = nd
                if return_schedule:
                    parent[ns] = (state, [PaperMove("R2", v)] + extra)
                dq.append(ns)

    if goal_state is None:
        return None, None, len(dist)

    if not return_schedule:
        return goal_cost, None, len(dist)

    # Reconstruct schedule and append final R4 cleanup to reach exact terminal state (red=0, blue=outputs)
    chunks: list[list[PaperMove]] = []
    cur = goal_state
    while cur != init_state:
        prev, mvs = parent[cur]
        chunks.append(mvs)
        cur = prev
    chunks.reverse()
    moves: list[PaperMove] = [mv for ch in chunks for mv in ch]

    final_red, final_blue, _ = goal_state
    for v in range(n):
        if final_red & (1 << v):
            moves.append(PaperMove("R4", v, "red"))
    for v in range(n):
        if (final_blue & (1 << v)) and (v not in dag.outputs):
            moves.append(PaperMove("R4", v, "blue"))

    return goal_cost, moves, len(dist)


def solve_raw_bfs(dag: DAGSpec, S: int) -> int | None:
    """
    Slower, unreduced 0-1 BFS over (red_mask, blue_mask) that includes explicit
    R4(blue) transitions during the search and terminates ONLY at (red=0, blue=outputs_mask).
    """
    n = dag.num_nodes
    if S <= 0:
        return None
    pred_masks = [sum(1 << p for p in dag.preds[v]) for v in range(n)]
    inputs_mask = sum(1 << i for i in dag.inputs)
    outputs_mask = sum(1 << i for i in dag.outputs)
    non_inputs = [v for v in range(n) if v not in dag.inputs]

    init = (0, inputs_mask)
    target = (0, outputs_mask)
    dist = {init: 0}
    dq = deque([init])

    while dq:
        red, blue = dq.popleft()
        d = dist[(red, blue)]
        if (red, blue) == target:
            return d
        rc = red.bit_count()

        # R4(red) and R4(blue)
        for v in range(n):
            bit = 1 << v
            if red & bit:
                ns = (red ^ bit, blue)
                if d < dist.get(ns, 10**9):
                    dist[ns] = d
                    dq.appendleft(ns)
            if blue & bit:
                ns = (red, blue ^ bit)
                if d < dist.get(ns, 10**9):
                    dist[ns] = d
                    dq.appendleft(ns)
        # R3
        if rc < S:
            for v in non_inputs:
                bit = 1 << v
                if not (red & bit) and (red & pred_masks[v]) == pred_masks[v]:
                    ns = (red | bit, blue)
                    if d < dist.get(ns, 10**9):
                        dist[ns] = d
                        dq.appendleft(ns)
        # R1 and R2
        if rc < S:
            for v in range(n):
                bit = 1 << v
                if (blue & bit) and not (red & bit):
                    ns = (red | bit, blue)
                    if d + 1 < dist.get(ns, 10**9):
                        dist[ns] = d + 1
                        dq.append(ns)
        for v in range(n):
            bit = 1 << v
            if (red & bit) and not (blue & bit):
                ns = (red, blue | bit)
                if d + 1 < dist.get(ns, 10**9):
                    dist[ns] = d + 1
                    dq.append(ns)
    return None
