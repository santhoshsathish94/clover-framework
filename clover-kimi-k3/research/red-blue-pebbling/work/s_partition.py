"""
Hong & Kung (1981) Section 3 S-partition checker, Theorem 3.1 schedule-to-2S-partition
constructor, and exact brute-force P(K) solver (E5).
"""

from __future__ import annotations
import itertools
from collections import deque
from dag_lib import DAGSpec
from paper_engine import PaperPebbleEngine, PaperMove


def is_dominator_set(dag: DAGSpec, U: set[int], D: set[int]) -> bool:
    """
    Check if D <= V is a dominator set for U <= V:
    Every path from an input of G to a vertex in U contains some vertex in D.
    Equivalently, in G \\ D, no vertex in U is reachable from inputs \\ D.
    """
    if not U:
        return True
    # Any input in U has a 0-length path to itself, so it must be in D
    for u in U:
        if u in dag.inputs and u not in D:
            return False
    succs = {i: [] for i in range(dag.num_nodes)}
    for v in range(dag.num_nodes):
        for p in dag.preds[v]:
            succs[p].append(v)

    visited = set()
    dq = deque([inp for inp in dag.inputs if inp not in D])
    while dq:
        x = dq.popleft()
        if x in visited:
            continue
        visited.add(x)
        if x in U:
            return False
        for y in succs[x]:
            if y not in D and y not in visited:
                dq.append(y)
    return True


def min_dominator_set(dag: DAGSpec, U: set[int]) -> set[int]:
    """Return a minimum-size dominator set D <= V for U by brute force."""
    if not U:
        return set()
    n = dag.num_nodes
    for k in range(1, n + 1):
        for comb in itertools.combinations(range(n), k):
            D = set(comb)
            if is_dominator_set(dag, U, D):
                return D
    return set(range(n))


def minimum_set(dag: DAGSpec, U: set[int]) -> set[int]:
    """
    Return the minimum set M of U: vertices in U that do not have any sons belonging to U.
    """
    succs_in_U = {u: False for u in U}
    for v in U:
        for p in dag.preds[v]:
            if p in U:
                succs_in_U[p] = True
    return {u for u in U if not succs_in_U[u]}


def check_s_partition(dag: DAGSpec, partition: list[set[int]], K: int) -> tuple[bool, dict]:
    """
    Verify Properties P1, P2, P3, P4 of a K-partition {V_1, ..., V_h}.
    """
    n = dag.num_nodes
    # Remove empty sets if any
    parts = [set(V_i) for V_i in partition if len(V_i) > 0]
    h = len(parts)
    # P1: disjoint and union = V
    union_all = set()
    for V_i in parts:
        if union_all & V_i:
            return False, {"error": "P1 failed: sets not disjoint"}
        union_all |= V_i
    if union_all != set(range(n)):
        return False, {"error": f"P1 failed: union={union_all} != V"}

    # P2 & P3
    dom_sizes = []
    min_sizes = []
    for i, V_i in enumerate(parts):
        D_i = min_dominator_set(dag, V_i)
        M_i = minimum_set(dag, V_i)
        dom_sizes.append(len(D_i))
        min_sizes.append(len(M_i))
        if len(D_i) > K:
            return False, {"error": f"P2 failed on V_{i+1}: |D|={len(D_i)} > {K}"}
        if len(M_i) > K:
            return False, {"error": f"P3 failed on V_{i+1}: |M|={len(M_i)} > {K}"}

    # P4: no cyclic dependence among {V_1, ..., V_h}
    node_to_part = {}
    for idx, V_i in enumerate(parts):
        for u in V_i:
            node_to_part[u] = idx
    dep_adj = {i: set() for i in range(h)}
    for v in range(n):
        pv = node_to_part[v]
        for u in dag.preds[v]:
            pu = node_to_part[u]
            if pu != pv:
                dep_adj[pu].add(pv)
    # Cycle check via Kahn's algorithm
    in_deg = {i: 0 for i in range(h)}
    for u in range(h):
        for v in dep_adj[u]:
            in_deg[v] += 1
    q = deque([i for i in range(h) if in_deg[i] == 0])
    visited_parts = 0
    while q:
        u = q.popleft()
        visited_parts += 1
        for v in dep_adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    if visited_parts != h:
        return False, {"error": "P4 failed: cyclic dependence among V_i"}

    return True, {"h": h, "dom_sizes": dom_sizes, "min_sizes": min_sizes, "parts": parts}


def schedule_to_2s_partition(dag: DAGSpec, S: int, moves: list[PaperMove]) -> tuple[list[set[int]], list[dict]]:
    """
    Construct {V_1, ..., V_h} from a complete calculation C using Theorem 3.1 (p. 327):
    - Divide C into h subcalculations C_1, ..., C_h where C_1..C_{h-1} each have exactly S
      R1/R2 transitions and C_h has <= S R1/R2 transitions (ending when the last R1/R2 finishes
      and any free R3s complete, before the final R4 terminal cleanup).
    - For i = 1..h, V_i is the largest vertex set satisfying (i), (ii), (iii).
    """
    # Trim trailing R4 moves after the last R1/R2/R3 move so C_h ends with outputs in blue
    last_non_r4 = -1
    for idx, mv in enumerate(moves):
        if mv.rule in ("R1", "R2", "R3"):
            last_non_r4 = idx
    active_moves = moves[: last_non_r4 + 1]

    # Cut active_moves into segments where C_1..C_{h-1} have S I/O moves and C_h has the rest (1..S)
    total_io = sum(1 for mv in active_moves if mv.rule in ("R1", "R2"))
    segments: list[list[PaperMove]] = []
    cur_seg: list[PaperMove] = []
    cur_io = 0
    io_seen = 0
    for mv in active_moves:
        cur_seg.append(mv)
        if mv.rule in ("R1", "R2"):
            cur_io += 1
            io_seen += 1
            if cur_io == S and io_seen < total_io:
                segments.append(cur_seg)
                cur_seg = []
                cur_io = 0
    if cur_seg:
        segments.append(cur_seg)

    eng = PaperPebbleEngine(dag.num_nodes, dag.preds, S=S, inputs=dag.inputs, outputs=dag.outputs)
    assigned = set()
    V_list: list[set[int]] = []
    seg_info: list[dict] = []

    for seg in segments:
        V_R = set(eng.red)
        blue_before = set(eng.blue)
        placed_red_r1_r3 = set()
        placed_blue_r2 = set()
        V_BR = set()

        for mv in seg:
            if mv.rule == "R1":
                placed_red_r1_r3.add(mv.node_id)
                if mv.node_id in blue_before:
                    V_BR.add(mv.node_id)
            elif mv.rule == "R3":
                placed_red_r1_r3.add(mv.node_id)
            elif mv.rule == "R2":
                placed_blue_r2.add(mv.node_id)
            eng.step(mv)

        red_end = set(eng.red)
        # Property (i) and (iii) candidates:
        cand = placed_red_r1_r3 - assigned
        # Property (ii): largest subset of cand where each v is in red_end, or in placed_blue_r2, or has a son in V_i
        V_i = {v for v in cand if (v in red_end) or (v in placed_blue_r2)}
        changed = True
        while changed:
            changed = False
            for u in cand - V_i:
                if any(w in V_i for w in eng.succs[u]):
                    V_i.add(u)
                    changed = True
        assigned |= V_i
        V_list.append(V_i)
        seg_info.append({
            "io_count": sum(1 for m in seg if m.rule in ("R1", "R2")),
            "V_R_union_V_BR_size": len(V_R | V_BR),
            "V_i": sorted(V_i),
        })

    return V_list, seg_info


def exact_min_partitions_P(dag: DAGSpec, K: int) -> tuple[int, list[set[int]]]:
    """
    Compute P(K) = minimum number of vertex sets in any K-partition of dag,
    by exact BFS over ancestral subsets W <= V.
    """
    n = dag.num_nodes
    pred_masks = [sum(1 << p for p in dag.preds[v]) for v in range(n)]

    # Enumerate all ancestral (downward-closed) bitmasks
    ancestral = []
    for mask in range(1 << n):
        ok = True
        for v in range(n):
            if (mask & (1 << v)) and ((mask & pred_masks[v]) != pred_masks[v]):
                ok = False
                break
        if ok:
            ancestral.append(mask)

    # Precheck which subsets U (represented as bitmask) satisfy P2 (|D|<=K) and P3 (|M|<=K)
    valid_U_cache: dict[int, bool] = {}

    def is_valid_block(u_mask: int) -> bool:
        if u_mask in valid_U_cache:
            return valid_U_cache[u_mask]
        U = {v for v in range(n) if u_mask & (1 << v)}
        M = minimum_set(dag, U)
        if len(M) > K:
            valid_U_cache[u_mask] = False
            return False
        D = min_dominator_set(dag, U)
        res = len(D) <= K
        valid_U_cache[u_mask] = res
        return res

    full_mask = (1 << n) - 1
    dist = {0: 0}
    parent = {}
    dq = deque([0])

    while dq:
        w = dq.popleft()
        d = dist[w]
        if w == full_mask:
            break
        for w_next in ancestral:
            if w_next != w and (w_next & w) == w:
                u_mask = w_next ^ w
                if w_next not in dist and is_valid_block(u_mask):
                    dist[w_next] = d + 1
                    parent[w_next] = (w, u_mask)
                    dq.append(w_next)

    if full_mask not in dist:
        return -1, []
    parts = []
    cur = full_mask
    while cur != 0:
        prev, u_mask = parent[cur]
        parts.append({v for v in range(n) if u_mask & (1 << v)})
        cur = prev
    parts.reverse()
    return dist[full_mask], parts
