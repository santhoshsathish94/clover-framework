"""
Canonical DAG definitions for Red-Blue Pebble Game experiments (E3, E4, E5, E7).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DAGSpec:
    name: str
    num_nodes: int
    preds: dict[int, list[int]]
    inputs: set[int]
    outputs: set[int]

    @property
    def max_fan_in(self) -> int:
        return max((len(p) for p in self.preds.values()), default=0)


def make_dag(name: str, num_nodes: int, edges: list[tuple[int, int]]) -> DAGSpec:
    preds = {i: [] for i in range(num_nodes)}
    succs = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        if u not in preds[v]:
            preds[v].append(u)
        if v not in succs[u]:
            succs[u].append(v)
    inputs = {i for i in range(num_nodes) if len(preds[i]) == 0}
    outputs = {i for i in range(num_nodes) if len(succs[i]) == 0}
    return DAGSpec(name=name, num_nodes=num_nodes, preds=preds, inputs=inputs, outputs=outputs)


def chain_dag(n: int = 4) -> DAGSpec:
    edges = [(i, i + 1) for i in range(n - 1)]
    return make_dag(f"chain_{n}", n, edges)


def v_tree_dag() -> DAGSpec:
    return make_dag("v_tree_3", 3, [(0, 2), (1, 2)])


def diamond_dag() -> DAGSpec:
    return make_dag("diamond_4", 4, [(0, 1), (0, 2), (1, 3), (2, 3)])


def binary_tree_dag(levels: int = 3) -> DAGSpec:
    """Complete binary tree with 2^(levels-1) leaves at bottom, 1 root at top."""
    # Level 0: leaves 0..2^(L-1)-1, ..., root at 2^L - 2
    edges = []
    cur_start = 0
    width = 1 << (levels - 1)
    nid = width
    while width > 1:
        next_width = width // 2
        for i in range(next_width):
            left = cur_start + 2 * i
            right = cur_start + 2 * i + 1
            parent = nid + i
            edges.append((left, parent))
            edges.append((right, parent))
        cur_start = nid
        nid += next_width
        width = next_width
    return make_dag(f"bintree_{nid}", nid, edges)


def pyramid_dag(base: int = 3) -> DAGSpec:
    """2D pyramid of given base width: level d has (base - d) vertices."""
    edges = []
    levels = []
    nid = 0
    for w in range(base, 0, -1):
        row = list(range(nid, nid + w))
        levels.append(row)
        nid += w
    for d in range(len(levels) - 1):
        for i, parent in enumerate(levels[d + 1]):
            edges.append((levels[d][i], parent))
            edges.append((levels[d][i + 1], parent))
    return make_dag(f"pyramid_b{base}_n{nid}", nid, edges)


def matmul_2x2_dag() -> DAGSpec:
    """
    Ordinary 2x2 matrix multiplication C = A * B (Section 6, p. 330):
    - 8 inputs: A00(0), A01(1), A10(2), A11(3), B00(4), B01(5), B10(6), B11(7)
    - 8 products: p_{i,j,k} = A_{i,k} * B_{k,j} (nodes 8..15)
    - 4 outputs: C_{i,j} = p_{i,j,0} + p_{i,j,1} (nodes 16..19) -> total 20 nodes!
    Wait: let's check vertex count!
      4 entries of A + 4 entries of B = 8 input vertices
      8 product vertices (each has fan-in 2 from A_{ik}, B_{kj})
      4 addition/output vertices C_{ij} (each has fan-in 2 from the two products)
      Total = 8 + 8 + 4 = 20 vertices.
    """
    edges = []
    # A indices: (i,k) -> 2*i + k in 0..3
    # B indices: (k,j) -> 4 + 2*k + j in 4..7
    prod_id = 8
    sum_id = 16
    for i in range(2):
        for j in range(2):
            p0 = prod_id
            p1 = prod_id + 1
            prod_id += 2
            a_i0, b_0j = 2 * i + 0, 4 + 2 * 0 + j
            a_i1, b_1j = 2 * i + 1, 4 + 2 * 1 + j
            edges.extend([(a_i0, p0), (b_0j, p0), (a_i1, p1), (b_1j, p1)])
            c_ij = sum_id
            sum_id += 1
            edges.extend([(p0, c_ij), (p1, c_ij)])
    return make_dag("matmul_2x2_n20", 20, edges)


def fft_butterfly_dag(n: int = 4) -> DAGSpec:
    """
    n-point FFT butterfly graph (Figure 2-1, p. 327):
    log2(n) + 1 levels of n vertices each, total n * (log2(n) + 1) vertices.
    At stage s (1..log2(n)), vertex (s, i) depends on (s-1, i) and (s-1, i ^ (1 << (s-1))).
    """
    import math
    k = int(math.log2(n))
    assert (1 << k) == n
    total = n * (k + 1)
    edges = []
    for s in range(1, k + 1):
        span = 1 << (s - 1)
        for i in range(n):
            u1 = (s - 1) * n + i
            u2 = (s - 1) * n + (i ^ span)
            v = s * n + i
            edges.append((u1, v))
            edges.append((u2, v))
    return make_dag(f"fft_{n}pt_n{total}", total, edges)
