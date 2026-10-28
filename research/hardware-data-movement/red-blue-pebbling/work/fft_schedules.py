"""
FFT butterfly graph exact/bounded-state solver and recursive radix-B blocked
schedule generator & verifier (E7).
"""

from __future__ import annotations
from collections import deque
from dag_lib import DAGSpec, fft_butterfly_dag
from paper_engine import PaperPebbleEngine, PaperMove
from exact_solver import solve_dijkstra


def solve_dijkstra_bounded(
    dag: DAGSpec,
    S: int,
    max_states: int = 200000,
) -> tuple[int | None, int, bool]:
    """
    Run 0-1 BFS with dead-pebble pruning up to max_states.
    Returns (min_io_or_None, states_explored, hit_cap).
    """
    n = dag.num_nodes
    pred_masks = [sum(1 << p for p in dag.preds[v]) for v in range(n)]
    succ_masks = [0] * n
    for v in range(n):
        for p in dag.preds[v]:
            succ_masks[p] |= (1 << v)

    inputs_mask = sum(1 << i for i in dag.inputs)
    outputs_mask = sum(1 << i for i in dag.outputs)
    non_inputs = [v for v in range(n) if v not in dag.inputs]

    def clean_dead(red: int, blue: int) -> tuple[int, int]:
        dead = 0
        for o in dag.outputs:
            if blue & (1 << o):
                dead |= (1 << o)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                bit = 1 << v
                if not (dead & bit) and succ_masks[v] != 0 and (succ_masks[v] & dead) == succ_masks[v]:
                    dead |= bit
                    changed = True
        red &= ~dead
        for v in range(n):
            bit = 1 << v
            if (dead & bit) and (v not in dag.outputs) and (v not in dag.inputs):
                blue &= ~bit
        return red, blue

    init = (0, inputs_mask)
    dist = {init: 0}
    dq = deque([init])

    while dq:
        if len(dist) >= max_states:
            return None, len(dist), True
        red, blue = dq.popleft()
        d = dist[(red, blue)]
        if (blue & outputs_mask) == outputs_mask:
            return d, len(dist), False

        rc = red.bit_count()
        # R4(red)
        r_tmp = red
        while r_tmp:
            lsb = r_tmp & -r_tmp
            r_tmp ^= lsb
            ns = (red ^ lsb, blue)
            if d < dist.get(ns, 10**9):
                dist[ns] = d
                dq.appendleft(ns)
        # R3
        if rc < S:
            for v in non_inputs:
                bit = 1 << v
                if not (red & bit) and (red & pred_masks[v]) == pred_masks[v]:
                    ns = clean_dead(red | bit, blue)
                    if d < dist.get(ns, 10**9):
                        dist[ns] = d
                        dq.appendleft(ns)
        # R1
        if rc < S:
            b_tmp = blue & ~red
            while b_tmp:
                lsb = b_tmp & -b_tmp
                b_tmp ^= lsb
                ns = (red | lsb, blue)
                if d + 1 < dist.get(ns, 10**9):
                    dist[ns] = d + 1
                    dq.append(ns)
        # R2
        r_tmp = red & ~blue
        while r_tmp:
            lsb = r_tmp & -r_tmp
            r_tmp ^= lsb
            ns = clean_dead(red, blue | lsb)
            if d + 1 < dist.get(ns, 10**9):
                dist[ns] = d + 1
                dq.append(ns)

    return None, len(dist), False


def build_recursive_blocked_fft_schedule(n: int, m_block: int) -> list[PaperMove]:
    """
    Construct a complete legal Red-Blue Pebble Game schedule for the n-point FFT
    butterfly graph using recursive radix-B (B = 2^m_block) sub-FFTs.
    Peak red pebbles required:
      - m_block == 1 (B=2): S = 3
      - m_block >= 2 (B=2^m_block): S = B + 2 (e.g. S=6 for B=4, S=10 for B=8, S=18 for B=16)
    Total I/O moves: 2 * n * ceil(log2(n) / m_block).
    """
    import math
    k_total = int(math.log2(n))
    assert (1 << k_total) == n
    moves: list[PaperMove] = []
    blue_sim: set[int] = set(range(n))

    def emit_subfft_keep_in_red(s0: int, m: int, indices: list[int]):
        """
        Given indices (length 2^m) whose stage-s0 vertices are in blue_sim,
        load them and compute up to stage s0 + m, leaving ONLY stage-(s0+m)
        vertices of `indices` in red (peak red = 2^m + 2 for m >= 1).
        """
        if m == 1:
            i0, i1 = indices[0], indices[1]
            u0, u1 = s0 * n + i0, s0 * n + i1
            v0, v1 = (s0 + 1) * n + i0, (s0 + 1) * n + i1
            moves.append(PaperMove("R1", u0))
            moves.append(PaperMove("R1", u1))
            moves.append(PaperMove("R3", v0))
            moves.append(PaperMove("R3", v1))
            moves.append(PaperMove("R4", u0, "red"))
            moves.append(PaperMove("R4", u1, "red"))
            return
        half = len(indices) // 2
        emit_subfft_keep_in_red(s0, m - 1, indices[:half])
        emit_subfft_keep_in_red(s0, m - 1, indices[half:])
        # Update pairs in-place from stage s0+m-1 to s0+m (+2 extra red slots)
        st_prev = s0 + m - 1
        st_cur = s0 + m
        for j in range(half):
            i0, i1 = indices[j], indices[j + half]
            u0, u1 = st_prev * n + i0, st_prev * n + i1
            v0, v1 = st_cur * n + i0, st_cur * n + i1
            moves.append(PaperMove("R3", v0))
            moves.append(PaperMove("R3", v1))
            moves.append(PaperMove("R4", u0, "red"))
            moves.append(PaperMove("R4", u1, "red"))

    def emit_subfft_to_blue(s0: int, m: int, indices: list[int]):
        """
        Evaluate an m-stage sub-FFT starting at stage s0 on `indices` (size 2^m),
        writing stage-(s0+m) outputs to blue and freeing all red pebbles.
        """
        if m == 1:
            i0, i1 = indices[0], indices[1]
            u0, u1 = s0 * n + i0, s0 * n + i1
            v0, v1 = (s0 + 1) * n + i0, (s0 + 1) * n + i1
            moves.append(PaperMove("R1", u0))
            moves.append(PaperMove("R1", u1))
            moves.append(PaperMove("R3", v0))
            moves.append(PaperMove("R2", v0))
            blue_sim.add(v0)
            moves.append(PaperMove("R4", v0, "red"))
            moves.append(PaperMove("R3", v1))
            moves.append(PaperMove("R2", v1))
            blue_sim.add(v1)
            moves.append(PaperMove("R4", v1, "red"))
            moves.append(PaperMove("R4", u0, "red"))
            moves.append(PaperMove("R4", u1, "red"))
            return
        half = len(indices) // 2
        emit_subfft_keep_in_red(s0, m - 1, indices[:half])
        emit_subfft_keep_in_red(s0, m - 1, indices[half:])
        st_prev = s0 + m - 1
        st_cur = s0 + m
        for j in range(half):
            i0, i1 = indices[j], indices[j + half]
            u0, u1 = st_prev * n + i0, st_prev * n + i1
            v0, v1 = st_cur * n + i0, st_cur * n + i1
            moves.append(PaperMove("R3", v0))
            moves.append(PaperMove("R2", v0))
            blue_sim.add(v0)
            moves.append(PaperMove("R4", v0, "red"))
            moves.append(PaperMove("R3", v1))
            moves.append(PaperMove("R2", v1))
            blue_sim.add(v1)
            moves.append(PaperMove("R4", v1, "red"))
            moves.append(PaperMove("R4", u0, "red"))
            moves.append(PaperMove("R4", u1, "red"))

    s0 = 0
    while s0 < k_total:
        m = min(m_block, k_total - s0)
        B = 1 << m
        # Partition 0..n-1 into blocks of size B varying bits s0..s0+m-1
        low_mask = (1 << s0) - 1
        for g in range(n // B):
            base = ((g >> s0) << (s0 + m)) | (g & low_mask)
            indices = [base | (t << s0) for t in range(B)]
            emit_subfft_to_blue(s0, m, indices)
        # Clean up stage-s0 blue pebbles
        for i in range(n):
            u = s0 * n + i
            if u in blue_sim:
                moves.append(PaperMove("R4", u, "blue"))
                blue_sim.remove(u)
        s0 += m

    return moves


def verify_schedule_and_peak_red(dag: DAGSpec, S: int, moves: list[PaperMove]) -> tuple[bool, int, int, str]:
    """
    Verify schedule on PaperPebbleEngine with capacity S.
    Returns (ok, io_cost, peak_red, msg).
    """
    eng = PaperPebbleEngine(dag.num_nodes, dag.preds, S=S, inputs=dag.inputs, outputs=dag.outputs)
    peak_red = 0
    for idx, mv in enumerate(moves):
        ok, _, err = eng.step(mv)
        if not ok:
            return False, eng.io_cost, peak_red, f"Step {idx+1} ({mv}) failed: {err}"
        if len(eng.red) > peak_red:
            peak_red = len(eng.red)
    if not eng.is_terminal():
        return False, eng.io_cost, peak_red, f"Non-terminal end: red={eng.red}, blue={eng.blue}"
    return True, eng.io_cost, peak_red, "OK"


def build_fft8_handover_schedule(S: int) -> list[PaperMove]:
    """
    Constructive schedule for 8-point FFT (32 nodes) with inter-pass red handover:
    - S=3: radix-2 stage-by-stage keeping 1 red pebble across stage 1->2 and stage 2->3 (44 I/Os).
    - S=6: pass 1 runs two radix-4 sub-FFTs on {0..3} and {4..7}, keeping 3 outputs of {4..7}
      in red (nodes 21, 22, 23) while streaming stage-3 pairs (26 I/Os).
    """
    n = 8
    moves: list[PaperMove] = []
    if S == 3:
        # Stage 1: 4 butterflies (0,1), (2,3), (4,5), (6,7). Keep node 15 (stage 1, idx 7) in red!
        for i0, i1 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            u0, u1 = i0, i1
            v0, v1 = 8 + i0, 8 + i1
            moves.extend([PaperMove("R1", u0), PaperMove("R1", u1), PaperMove("R3", v0),
                          PaperMove("R2", v0), PaperMove("R4", v0, "red"), PaperMove("R3", v1),
                          PaperMove("R4", u0, "red"), PaperMove("R4", u1, "red"),
                          PaperMove("R4", u0, "blue"), PaperMove("R4", u1, "blue")])
            if i1 != 7:
                moves.extend([PaperMove("R2", v1), PaperMove("R4", v1, "red")])
        # Stage 2: start with (5,7) since 15=(1,7) is already in red! Keep last node 20=(2,4) in red!
        for i0, i1 in [(5, 7), (1, 3), (0, 2), (4, 6)]:
            u0, u1 = 8 + i0, 8 + i1
            v0, v1 = 16 + i0, 16 + i1
            moves.append(PaperMove("R1", u0))
            if (i0, i1) != (5, 7):
                moves.append(PaperMove("R1", u1))
            moves.extend([PaperMove("R3", v0), PaperMove("R2", v0), PaperMove("R4", v0, "red"),
                          PaperMove("R3", v1), PaperMove("R4", u0, "red"), PaperMove("R4", u1, "red"),
                          PaperMove("R4", u0, "blue")])
            if (i0, i1) != (5, 7):
                moves.append(PaperMove("R4", u1, "blue"))
            if (i0, i1) != (4, 6):
                moves.extend([PaperMove("R2", v1), PaperMove("R4", v1, "red")])
        # Stage 3: node 22=(2,6) is in red! Run (2,6) first, then (0,4), (1,5), (3,7)
        for i0, i1 in [(2, 6), (0, 4), (1, 5), (3, 7)]:
            u0, u1 = 16 + i0, 16 + i1
            v0, v1 = 24 + i0, 24 + i1
            moves.append(PaperMove("R1", u0))
            if (i0, i1) != (2, 6):
                moves.append(PaperMove("R1", u1))
            moves.extend([PaperMove("R3", v0), PaperMove("R2", v0), PaperMove("R4", v0, "red"),
                          PaperMove("R3", v1), PaperMove("R2", v1), PaperMove("R4", v1, "red"),
                          PaperMove("R4", u0, "red"), PaperMove("R4", u1, "red"),
                          PaperMove("R4", u0, "blue")])
            if (i0, i1) != (2, 6):
                moves.append(PaperMove("R4", u1, "blue"))
        return moves

    # S == 6 handover schedule (26 I/Os):
    # 1. First 4-point sub-FFT on {0,1,2,3} -> write 16,17,18,19 to blue (8 I/Os)
    moves.extend([
        PaperMove("R1", 0), PaperMove("R1", 1), PaperMove("R3", 8), PaperMove("R3", 9),
        PaperMove("R4", 0, "red"), PaperMove("R4", 1, "red"),
        PaperMove("R4", 0, "blue"), PaperMove("R4", 1, "blue"),
        PaperMove("R1", 2), PaperMove("R1", 3), PaperMove("R3", 10), PaperMove("R3", 11),
        PaperMove("R4", 2, "red"), PaperMove("R4", 3, "red"),
        PaperMove("R4", 2, "blue"), PaperMove("R4", 3, "blue"),
        PaperMove("R3", 16), PaperMove("R2", 16), PaperMove("R4", 16, "red"),
        PaperMove("R3", 18), PaperMove("R2", 18), PaperMove("R4", 18, "red"),
        PaperMove("R4", 8, "red"), PaperMove("R4", 10, "red"),
        PaperMove("R3", 17), PaperMove("R2", 17), PaperMove("R4", 17, "red"),
        PaperMove("R3", 19), PaperMove("R2", 19), PaperMove("R4", 19, "red"),
        PaperMove("R4", 9, "red"), PaperMove("R4", 11, "red"),
    ])
    # 2. Second 4-point sub-FFT on {4,5,6,7} -> keep 20,22,21 in red, spill 23 to blue?
    # Wait: in stage 2 of {4,5,6,7}, we have {12,13,14,15} in red (4 slots).
    # Compute 20, 22 from 12, 14 (6 slots), delete 12, 14 (4 slots: {13,15,20,22}).
    # Compute 21, 23 from 13, 15 (6 slots), delete 13, 15 (4 slots: {20,21,22,23} in red!).
    # Now {20,21,22,23} are ALL 4 in red (4 slots used out of 6)!
    # Can we complete stage 3 with ZERO spills of {20,21,22,23}?!
    # YES! Look: {20,21,22,23} uses 4 slots! Load 16 (R1(16), 5 slots), compute 24 (6 slots),
    # R2(24), R4(24,red) (5 slots), compute 28 (6 slots), R2(28), R4(28,red), R4(16,red), R4(20,red) (3 slots)!
    # Then load 17, 18, 19 and combine with 21, 22, 23! That saves ALL 4 R2s and ALL 4 R1s (-8 I/Os -> 24 I/Os)!
    moves.extend([
        PaperMove("R1", 4), PaperMove("R1", 5), PaperMove("R3", 12), PaperMove("R3", 13),
        PaperMove("R4", 4, "red"), PaperMove("R4", 5, "red"),
        PaperMove("R4", 4, "blue"), PaperMove("R4", 5, "blue"),
        PaperMove("R1", 6), PaperMove("R1", 7), PaperMove("R3", 14), PaperMove("R3", 15),
        PaperMove("R4", 6, "red"), PaperMove("R4", 7, "red"),
        PaperMove("R4", 6, "blue"), PaperMove("R4", 7, "blue"),
        PaperMove("R3", 20), PaperMove("R3", 22), PaperMove("R4", 12, "red"), PaperMove("R4", 14, "red"),
        PaperMove("R3", 21), PaperMove("R3", 23), PaperMove("R4", 13, "red"), PaperMove("R4", 15, "red"),
    ])
    for i0, i1 in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        u0, u1 = 16 + i0, 16 + i1
        v0, v1 = 24 + i0, 24 + i1
        moves.extend([
            PaperMove("R1", u0),
            PaperMove("R3", v0), PaperMove("R2", v0), PaperMove("R4", v0, "red"),
            PaperMove("R3", v1), PaperMove("R2", v1), PaperMove("R4", v1, "red"),
            PaperMove("R4", u0, "red"), PaperMove("R4", u1, "red"),
            PaperMove("R4", u0, "blue"),
        ])
    return moves

