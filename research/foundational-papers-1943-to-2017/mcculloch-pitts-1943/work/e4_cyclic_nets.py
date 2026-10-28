"""
E4: Nets with Circles in McCulloch & Pitts (1943):
  - Theorem 7 (p. 108, Fig. 1i): Alterable (learning) synapses replaced by a circle.
  - Theorem 8 (pp. 109-110, Eq. 5-9): Solution of cyclic nets of order p via 2^p state conjunctions X_1..X_{2^p}.
  - Theorem 10 (pp. 112-113): Existential (Ez2 < z1), Universal (z2 < z1), and Modulo-n (C_{mn}) cyclic nets.
  - Section 4 (pp. 113-114): Irreciprocity of causality (forward determinism vs backward many-to-one collapse).
"""
import itertools
from mp1943_calculus import MPNet1943


def test_theorem7_fig1i():
    print("=== E4.1: Theorem 7 (Fig. 1i, p. 108) Alterable Synapse vs Fixed Circle ===")
    # Alterable synapse specification (p. 108 & Fig. 1i caption on p. 105):
    # N_3(t) .=. N_2(t-1) .v. N_1(t-1) . (Ex)_{x < t-1} (N_1(x) . N_2(x))
    # Why? At time x, c1 and c2 both fire (so at x+1, c3 fires while c1's axonal termination was excited at x),
    # permanently activating the c1 -> c3 synapse for all subsequent steps t-1 > x!
    # In a fixed net with a circle (order p=1):
    #   ca: N_a(t) = N_1(t-1) . N_2(t-1)   (coincidence detector, theta=2, exc={c1:1, c2:1})
    #   cm (cyclic neuron): N_m(t) = N_a(t-1) v N_m(t-1) (or directly N_m(t) = (N_1(t-1).N_2(t-1)) v N_m(t-1)
    #     by giving cm theta=2 with exc={c1:1, c2:1, cm:2}!)
    #   Wait: if cm has theta=2 and exc={c1:1, c2:1, cm:2}, then:
    #     N_m(t) = 1 iff (c1(t-1) + c2(t-1) + 2*cm(t-1) >= 2) <=> (c1(t-1)&c2(t-1)) v cm(t-1) <=> (Ex < t)(N_1(x).N_2(x))!
    #   Then at c3:
    #     If c3 has theta=2 and exc={c2:2, c1:1, cm:1}, then:
    #     N_3(t) = 1 iff (2*c2(t-1) + c1(t-1) + cm(t-1) >= 2) <=> N_2(t-1) v (N_1(t-1) . N_m(t-1))
    #            <=> N_2(t-1) v (N_1(t-1) . (Ex < t-1)(N_1(x) . N_2(x)))!
    net_circle = MPNet1943(["c1", "c2"])
    net_circle.add_neuron("cm", theta=2, excitatory={"c1": 1, "c2": 1, "cm": 2}, inhibitory=[])
    net_circle.add_neuron("c3", theta=2, excitatory={"c2": 2, "c1": 1, "cm": 1}, inhibitory=[])

    # Compare against stateful alterable-synapse neuron across all 4^6 = 4,096 input histories of length 6
    matched = 0
    total = 0
    for seq in itertools.product([(0, 0), (0, 1), (1, 0), (1, 1)], repeat=6):
        sched = [{"c1": s[0], "c2": s[1]} for s in seq]
        tr = net_circle.simulate(sched)

        # Alterable synapse simulation:
        synapse_learned = False
        alt_c3 = [0]
        for t in range(1, len(seq)):
            c1_prev, c2_prev = seq[t-1]
            # c3 fires at t if c2(t-1)==1 or (synapse_learned and c1(t-1)==1)
            fired = 1 if (c2_prev == 1 or (synapse_learned and c1_prev == 1)) else 0
            alt_c3.append(fired)
            # Learning rule: if c1 and c2 co-occurred at t-1, synapse is now permanently active for future steps
            if c1_prev == 1 and c2_prev == 1:
                synapse_learned = True

        cir_c3 = [st["c3"] for st in tr]
        # Also check exact Fig. 1i formula on p. 105:
        form_c3 = [0]
        for t in range(1, len(seq)):
            ex = any(seq[x][0] == 1 and seq[x][1] == 1 for x in range(t - 1))
            fval = 1 if (seq[t-1][1] == 1 or (seq[t-1][0] == 1 and ex)) else 0
            form_c3.append(fval)

        if alt_c3 == cir_c3 == form_c3:
            matched += 1
        total += 1

    print(f"Theorem 7 (Fig. 1i) alterable synapse == order-1 circle == p.105 formula across {matched}/{total} 6-step histories")
    assert matched == total == 4096


def test_theorem8_cyclic_order2():
    print("\n=== E4.2: Theorem 8 (Eq. 5-9, pp. 109-110) State-Transition Solution for Order p=2 Cyclic Net ===")
    # Build an order p=2 cyclic net with cyclic set {c1, c2} (2^p = 4 states X_1..X_4) and peripheral afferents {u1, u2}:
    #   c1(t): fires iff u1(t-1) v (c2(t-1) . ~u2(t-1))  [theta=1, exc={u1:1, c2:1}, inh=[u2]]
    #   c2(t): fires iff (u1(t-1) . c1(t-1)) v (c1(t-1) . c2(t-1)) [theta=2, exc={u1:1, c1:1, c2:1}, inh=[]]
    net = MPNet1943(["u1", "u2"])
    net.add_neuron("c1", theta=1, excitatory={"u1": 1, "c2": 1}, inhibitory=["u2"])
    net.add_neuron("c2", theta=2, excitatory={"u1": 1, "c1": 1, "c2": 1}, inhibitory=[])

    # The 2^p = 4 cyclic state conjunctions X_1..X_4 at time t (p. 109):
    #   X_0: ~N1(t) . ~N2(t)   (state (0,0))
    #   X_1: ~N1(t) .  N2(t)   (state (0,1))
    #   X_2:  N1(t) . ~N2(t)   (state (1,0))
    #   X_3:  N1(t) .  N2(t)   (state (1,1))
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def psi_matrix(u1_val: int, u2_val: int, broken: bool = False):
        """
        Computes Psi_{i,j}(t) in Eq. (5): X_i(t) .=. V_{j=0}^3 Psi_{i,j}(t) . X_j(t-1),
        where Psi_{i,j}(t) is a TPE of the peripheral afferents at t-1.
        """
        mat = [[0] * 4 for _ in range(4)]
        for j, (c1_prev, c2_prev) in enumerate(states):
            n1_next = 1 if (not u2_val and (u1_val + c2_prev >= 1)) else 0
            n2_next = 1 if (u1_val + c1_prev + c2_prev >= 2) else 0
            i = states.index((n1_next, n2_next))
            if broken and j == 2:
                i = (i + 1) % 4
            mat[i][j] = 1
        return mat

    # Verify Eq. (5) and Eq. (8) path-existential formula across all 4^5 = 1,024 afferent sequences
    exact_matches = 0
    broken_matches = 0
    for seq in itertools.product([(0, 0), (0, 1), (1, 0), (1, 1)], repeat=5):
        sched = [{"u1": s[0], "u2": s[1]} for s in seq]
        tr = net.simulate(sched)

        # Track state vector X(t) via Eq. (5)
        X = [1, 0, 0, 0]  # at t=0, c1(0)=0, c2(0)=0 -> X_0(0)=1
        Xb = [1, 0, 0, 0]
        ok_exact = True
        ok_broken = True
        for t in range(1, len(seq)):
            u1_prev, u2_prev = seq[t-1]
            psi = psi_matrix(u1_prev, u2_prev, broken=False)
            psib = psi_matrix(u1_prev, u2_prev, broken=True)
            X = [int(any(psi[i][j] and X[j] for j in range(4))) for i in range(4)]
            Xb = [int(any(psib[i][j] and Xb[j] for j in range(4))) for i in range(4)]
            sim_idx = states.index((tr[t]["c1"], tr[t]["c2"]))
            if X[sim_idx] != 1 or sum(X) != 1:
                ok_exact = False
            if Xb[sim_idx] != 1:
                ok_broken = False
        if ok_exact:
            exact_matches += 1
        if ok_broken:
            broken_matches += 1

    print(f"Theorem 8 Eq. (5)-(8) over 2^p=4 state conjunctions matched Eq. (1) simulation on {exact_matches}/1024 sequences")
    print(f"Negative control (swapped transition predicate Psi_{{i,2}}): matched on {broken_matches}/1024 sequences")
    assert exact_matches == 1024 and broken_matches < 1024


def test_theorem10_and_section4_irreciprocity():
    print("\n=== E4.3: Theorem 10 (pp. 112-113) Quantified & Modulo-n Cyclic Nets ===")
    # 1. Existential net: N_i(z1) <=> Pr_1(z1-1) v N_i(z1-1) realizes (Ez2 < z1) Pr_1(z2)
    net_ex = MPNet1943(["p1"])
    net_ex.add_neuron("ci", theta=1, excitatory={"p1": 1, "ci": 1}, inhibitory=[])

    # 2. Universal net: N_i(z1) <=> Pr_1(z1-1) . N_i(z1-1) realizes (z2 < z1) Pr_1(z2) when ci(0)=1
    # Note: In our simulator non-afferents start at 0 unless initialized by a seed afferent `init` at t=0:
    # Let ci have theta=2, exc={"p1": 1, "ci": 1, "init": 1}, so at t=0 with init(0)=1 and p1(0), ci(1) = p1(0),
    # and for t >= 1 (with init(t)=0), ci(t+1) = p1(t) . ci(t) = /\_{z2 <= t} p1(z2)!
    net_all = MPNet1943(["p1", "init"])
    net_all.add_neuron("ci", theta=2, excitatory={"p1": 1, "ci": 1, "init": 1}, inhibitory=[])

    # 3. Modulo-n ring (p. 113): simple circuit c1 -> c2 -> c3 -> c1 of n=3 links, seeded at t=0 by afferent `seed` on c1
    net_mod3 = MPNet1943(["seed"])
    net_mod3.add_neuron("c1", theta=1, excitatory={"seed": 1, "c3": 1}, inhibitory=[])
    net_mod3.add_neuron("c2", theta=1, excitatory={"c1": 1}, inhibitory=[])
    net_mod3.add_neuron("c3", theta=1, excitatory={"c2": 1}, inhibitory=[])

    ok_ex = 0
    ok_all = 0
    for bits in itertools.product([0, 1], repeat=6):
        sched_ex = [{"p1": b} for b in bits]
        tr_ex = net_ex.simulate(sched_ex)
        if all(tr_ex[t]["ci"] == int(any(bits[:t])) for t in range(1, 6)):
            ok_ex += 1

        sched_all = [{"p1": b, "init": 1 if idx == 0 else 0} for idx, b in enumerate(bits)]
        tr_all = net_all.simulate(sched_all)
        if all(tr_all[t]["ci"] == int(all(bits[:t])) for t in range(1, 6)):
            ok_all += 1

    sched_mod = [{"seed": 1 if t == 0 else 0} for t in range(10)]
    tr_mod = net_mod3.simulate(sched_mod)
    c1_times = [t for t in range(1, 10) if tr_mod[t]["c1"] == 1]
    c2_times = [t for t in range(1, 10) if tr_mod[t]["c2"] == 1]
    c3_times = [t for t in range(1, 10) if tr_mod[t]["c3"] == 1]
    print(f"Theorem 10 Existential (Ez2 < z1) Pr_1(z2): verified on {ok_ex}/64 6-step streams")
    print(f"Theorem 10 Universal   (z2 < z1) Pr_1(z2): verified on {ok_all}/64 6-step streams")
    print(f"Theorem 10 Modulo-3 ring firing times (t=1..9): c1={c1_times} (t==1 mod 3), c2={c2_times} (t==2 mod 3), c3={c3_times} (t==0 mod 3)")
    assert ok_ex == 64 and ok_all == 64 and c1_times == [1, 4, 7] and c2_times == [2, 5, 8] and c3_times == [3, 6, 9]

    print("\n=== E4.4: Section 4 (pp. 113-114) Irreciprocity of Causality ===")
    # "Specification of the nervous net provides the law of necessary connection whereby one can compute from the
    #  description of any state that of the succeeding state, but the inclusion of disjunctive relations prevents
    #  complete determination of the one before. Moreover, the regenerative activity of constituent circles renders
    #  reference indefinite as to time past."
    # Quantify on Fig. 1e (2 afferents c1,c2 + 4 internal neurons ca,cb,c3,c4 -> 2^6 = 64 total states (u(t), s(t)))
    # mapping to internal state s(t+1) in {0,1}^4 (16 possible internal states).
    from e3_illusions_and_equivalences import build_fig1e_cutaneous_net
    net_1e = build_fig1e_cutaneous_net()
    internal_nodes = ["ca", "cb", "c3", "c4"]
    preimages = {}
    for bits in itertools.product([0, 1], repeat=6):
        st = {"c1": bits[0], "c2": bits[1], "ca": bits[2], "cb": bits[3], "c3": bits[4], "c4": bits[5]}
        next_s = tuple(net_1e.neurons[n].step_eq1(st) for n in internal_nodes)
        preimages.setdefault(next_s, []).append(bits)

    reachable = len(preimages)
    counts = sorted([len(v) for v in preimages.values()], reverse=True)
    print(f"Fig. 1e transition map: 64 predecessor configurations (c1,c2,ca,cb,c3,c4) -> {reachable}/16 reachable next internal states")
    print(f"Preimage sizes per reachable next state (backward ambiguity): {counts} (min={min(counts)}, max={max(counts)})")
    assert reachable < 16 and min(counts) >= 4


if __name__ == "__main__":
    test_theorem7_fig1i()
    test_theorem8_cyclic_order2()
    test_theorem10_and_section4_irreciprocity()
