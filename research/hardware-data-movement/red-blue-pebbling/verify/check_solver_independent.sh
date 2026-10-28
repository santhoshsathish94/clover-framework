cd /root/dm-study/pebbling
ls w2/dag_lib.py > /dev/null && echo "dag_lib found"
/root/ncm-cpu/venv/bin/python - <<'EOF'
# Coordinator's independent solver (own code, not the study's solver); uses only the study's DAG definitions.
import heapq, sys, time
sys.path.insert(0, "/root/dm-study/pebbling/w2")
import dag_lib

def min_io(spec, S, allow_recompute=True, cap=3_000_000):
    n = spec.num_nodes
    pm = [sum(1 << p for p in spec.preds[v]) for v in range(n)]
    src = sum(1 << v for v in spec.inputs)
    snk = sum(1 << v for v in spec.outputs)
    start = (0, src, src)
    best = {start: 0}
    pq = [(0, 0, start)]
    tie = 0
    while pq:
        cost, _, st = heapq.heappop(pq)
        if best.get(st, 1 << 60) < cost:
            continue
        red, blue, done = st
        if blue & snk == snk:
            return cost, len(best)
        if len(best) > cap:
            return "CAP", len(best)
        rc = red.bit_count()
        for v in range(n):
            b = 1 << v
            nxt = []
            if blue & b and not red & b and rc < S:
                nxt.append((cost + 1, (red | b, blue, done)))
            if red & b and not blue & b:
                nxt.append((cost + 1, (red, blue | b, done)))
            if not src & b and not red & b and rc < S and (red & pm[v]) == pm[v]:
                if allow_recompute:
                    nxt.append((cost, (red | b, blue, done)))
                elif not done & b:
                    nxt.append((cost, (red | b, blue, done | b)))
            if red & b:
                nxt.append((cost, (red & ~b, blue, done)))
            for c, s in nxt:
                if c < best.get(s, 1 << 60):
                    best[s] = c
                    tie += 1
                    heapq.heappush(pq, (c, tie, s))
    return None, len(best)

def row(spec, S_list, expect, recompute=True):
    out = []
    for S in S_list:
        t0 = time.time()
        q, ns = min_io(spec, S, recompute)
        out.append("S=%d:%s" % (S, q))
        if expect is not None:
            exp = expect.get(S)
    print("%-18s %s  (expected from study: %s)" % (spec.name, "  ".join(out), expect))
    sys.stdout.flush()

row(dag_lib.chain_dag(4), [1, 2, 3], {1: None, 2: 2})
row(dag_lib.diamond_dag(), [2, 3, 4], {2: None, 3: 2})
row(dag_lib.binary_tree_dag(3), [2, 3, 4, 5], {3: 7, 4: 5, 5: 5})
row(dag_lib.pyramid_dag(3), [2, 3, 4], {3: 6, 4: 4})
row(dag_lib.pyramid_dag(4), [3, 4, 5, 6], {3: 11, 4: 7, 5: 5, 6: 5})
row(dag_lib.matmul_2x2_dag(), [3, 4, 5, 6, 7], {3: 25, 4: 17, 5: 14, 6: 13, 7: 12})
row(dag_lib.fft_butterfly_dag(4), [3, 4, 5, 6], {3: 14, 4: 11, 5: 9, 6: 8})
# E4 smallest recompute example from the study (n=5, S=3): 2 with recompute, 4 without
g5 = dag_lib.make_dag("G5_recomp", 5, [(0, 1), (1, 2), (0, 3), (2, 3), (1, 4), (3, 4)])
print("G5_recomp S=3  with recompute:", min_io(g5, 3, True)[0], " without:", min_io(g5, 3, False)[0], " (study: 2 and 4)")
EOF
echo DONE-PEB
