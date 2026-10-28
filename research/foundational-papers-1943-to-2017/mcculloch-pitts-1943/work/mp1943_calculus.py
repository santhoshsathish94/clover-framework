"""
Exact McCulloch & Pitts (1943) Calculus Engine & Theorem 1-3 TPE Compiler/Verifier.
Implements:
  - Section 2 physical assumptions (1)-(5) and Equation (1) (p. 103):
      N_i(t) <=> S [ \prod_{m=1}^q ~N_{j_m}(t) & \sum_{s} n_{is} N_{i_s}(t) >= \theta_i ]
    Note: The paper writes "exceeds \theta_i" in prose on p. 103 while using integer thresholds
    (e.g. Fig. 1a-i where \theta=1 or \theta=2 means >= \theta active synapses). With integer
    synapse counts, "> \theta - 1" and ">= \theta" for positive integer \theta >= 1 are identical.
  - Fundamental elements of Figure 1a-d (p. 104-105):
      1a (delay S):      N_2(t) = N_1(t-1)                      [1 exc, theta=1]
      1b (disjunction):  N_3(t) = N_1(t-1) v N_2(t-1)           [1 exc from c1, 1 exc from c2, theta=1]
      1c (conjunction):  N_3(t) = N_1(t-1) . N_2(t-1)           [1 exc from c1, 1 exc from c2, theta=2]
      1d (conj neg):     N_3(t) = N_1(t-1) . ~N_2(t-1)          [1 exc from c1, 1 inh from c2, theta=1]
  - Theorem 2 & 3 constructive synthesis (pp. 104-106):
    Any Boolean truth table over k peripheral afferents (N_1, ..., N_k) that is FALSE on
    (0, ..., 0) has a Hilbert disjunctive normal form in which every minterm has m >= 1
    positive literals (S_1 . ... . S_m) and r >= 0 negative literals (~S_{m+1} . ... . ~S_k),
    rewritten via p. 106 line 2 as:
      (S_1 . S_2 . ... . S_m) . ~(S_{m+1} v S_{m+2} v ... v S_k)
    and realized at a uniform synaptic delay d.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class MPNeuron1943:
    name: str
    theta: int  # strictly positive integer >= 1 in 1943 paper (Theorem 1, p. 103)
    excitatory: Dict[str, int] = field(default_factory=dict)  # source_name -> synapse count n_{is} >= 1
    inhibitory: List[str] = field(default_factory=list)       # list of inhibitory afferent names

    def __post_init__(self):
        if self.theta < 1:
            raise ValueError(f"1943 Theorem 1 requires theta_i > 0 (got theta={self.theta} on {self.name})")
        for src, cnt in self.excitatory.items():
            if cnt < 0:
                raise ValueError(f"Excitatory synapse count must be non-negative (got {cnt} on {src}->{self.name})")

    def step_eq1(self, prev_state: Dict[str, int]) -> int:
        # Assumption (4) & Eq (1): any active inhibitory synapse absolutely prevents excitation
        for inh in self.inhibitory:
            if prev_state.get(inh, 0) != 0:
                return 0
        exc_sum = sum(cnt * prev_state.get(src, 0) for src, cnt in self.excitatory.items())
        return 1 if exc_sum >= self.theta else 0


class MPNet1943:
    def __init__(self, afferents: List[str]):
        self.afferents = list(afferents)
        self.neurons: Dict[str, MPNeuron1943] = {}

    def add_neuron(self, name: str, theta: int, excitatory: Dict[str, int], inhibitory: List[str] = ()):
        assert name not in self.afferents and name not in self.neurons, f"Duplicate node {name}"
        self.neurons[name] = MPNeuron1943(name=name, theta=theta, excitatory=dict(excitatory), inhibitory=list(inhibitory))
        return name

    # Fundamental building blocks of Fig. 1a-d (pp. 104-105)
    def fig1a_delay(self, name: str, src: str) -> str:
        return self.add_neuron(name, theta=1, excitatory={src: 1}, inhibitory=[])

    def fig1b_or(self, name: str, srcs: List[str]) -> str:
        exc = {}
        for s in srcs:
            exc[s] = exc.get(s, 0) + 1
        return self.add_neuron(name, theta=1, excitatory=exc, inhibitory=[])

    def fig1c_and(self, name: str, srcs: List[str]) -> str:
        exc = {}
        for s in srcs:
            exc[s] = exc.get(s, 0) + 1
        return self.add_neuron(name, theta=len(srcs), excitatory=exc, inhibitory=[])

    def fig1d_and_not(self, name: str, pos_srcs: List[str], neg_srcs: List[str]) -> str:
        exc = {}
        for s in pos_srcs:
            exc[s] = exc.get(s, 0) + 1
        return self.add_neuron(name, theta=len(pos_srcs), excitatory=exc, inhibitory=list(neg_srcs))

    def simulate(self, afferent_schedule: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        Runs discrete time steps t = 0, 1, ..., T-1.
        At t = 0, peripheral afferents take afferent_schedule[0] and non-afferent neurons start at 0.
        At t -> t+1, every non-afferent neuron updates by Eq. (1) from state at t,
        and peripheral afferents take afferent_schedule[t+1].
        """
        state = {a: afferent_schedule[0].get(a, 0) for a in self.afferents}
        for n in self.neurons:
            state[n] = 0
        trace = [dict(state)]
        for t in range(1, len(afferent_schedule)):
            next_state = {a: afferent_schedule[t].get(a, 0) for a in self.afferents}
            for name, neuron in self.neurons.items():
                next_state[name] = neuron.step_eq1(state)
            state = next_state
            trace.append(dict(state))
        return trace


def enumerate_single_neuron_2input() -> Dict[Tuple[int, int, int, int], Tuple[int, Dict[str, int], List[str]]]:
    """
    Exhaustively tests all single 1943 neurons on 2 inputs (x1, x2) with:
      theta in 1..4, n1 in 0..3, n2 in 0..3, inhibitory subset of {x1, x2}.
    Returns map: truth_table_tuple -> (theta, exc_dict, inh_list).
    """
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    realized = {}
    for theta in range(1, 5):
        for n1 in range(0, 4):
            for n2 in range(0, 4):
                for inh_mask in range(4):
                    inh = []
                    if inh_mask & 1:
                        inh.append("x1")
                    if inh_mask & 2:
                        inh.append("x2")
                    exc = {}
                    if n1 > 0:
                        exc["x1"] = n1
                    if n2 > 0:
                        exc["x2"] = n2
                    neuron = MPNeuron1943("out", theta=theta, excitatory=exc, inhibitory=inh)
                    tt = tuple(neuron.step_eq1({"x1": x1, "x2": x2}) for x1, x2 in inputs)
                    if tt not in realized:
                        realized[tt] = (theta, exc, inh)
    return realized


def compile_tpe_theorem2(k: int, truth_table: Dict[Tuple[int, ...], int]) -> Tuple[MPNet1943, str, int]:
    """
    Constructs an order-0 1943 net realizing `truth_table` over afferents x1..xk
    using Theorem 2 & Theorem 3 (pp. 104-106).
    Raises ValueError if Theorem 3 condition f(0,...,0) == 0 is violated.
    Returns (net, output_neuron_name, synaptic_delay).
    """
    zeros = tuple([0] * k)
    if truth_table[zeros] != 0:
        raise ValueError("Theorem 3 violation: truth_table(0,...,0) == 1 has an all-negated minterm; not a TPE.")

    afferents = [f"x{i+1}" for i in range(k)]
    net = MPNet1943(afferents)

    active_rows = [bits for bits, val in truth_table.items() if val == 1]
    if not active_rows:
        # Constant FALSE (0 everywhere): single neuron with theta=1 and no excitatory synapses
        out = net.add_neuron("out", theta=1, excitatory={}, inhibitory=[])
        return net, out, 1

    # Pure Fig. 1a-d decomposition (p. 106 top):
    # Each active row has m >= 1 positive afferents P and r >= 0 negative afferents Q.
    # Level 1 (t=1):
    #   - Delayed copies of each afferent via Fig. 1a: d_xi(t) = xi(t-1)
    #   - For each minterm j with r >= 1 negative afferents, disjunction of negative afferents via Fig. 1b:
    #     neg_or_j(t) = V_{q in Q_j} q(t-1)
    # Level 2 (t=2):
    #   - For each minterm j, combine delayed positive afferents P_j (excitatory, theta=len(P_j))
    #     and inhibit by neg_or_j (if r >= 1) via Fig. 1c/1d:
    #     term_j(t) = (/\_{p in P_j} d_p(t-1)) & ~neg_or_j(t-1)
    # Level 3 (t=3):
    #   - Disjunction across all minterms term_j via Fig. 1b:
    #     out(t) = V_j term_j(t-1)
    for a in afferents:
        net.fig1a_delay(f"d_{a}", a)

    term_nodes = []
    for idx, bits in enumerate(active_rows):
        pos_delayed = [f"d_x{i+1}" for i, b in enumerate(bits) if b == 1]
        neg_raw = [f"x{i+1}" for i, b in enumerate(bits) if b == 0]
        assert len(pos_delayed) >= 1
        if neg_raw:
            neg_or = net.fig1b_or(f"neg_or_{idx}", neg_raw)
            t_node = net.fig1d_and_not(f"term_{idx}", pos_delayed, [neg_or])
        else:
            t_node = net.fig1c_and(f"term_{idx}", pos_delayed)
        term_nodes.append(t_node)

    out = net.fig1b_or("out", term_nodes)
    return net, out, 3


def verify_compiled_net(net: MPNet1943, out_node: str, delay: int, k: int, truth_table: Dict[Tuple[int, ...], int]) -> bool:
    afferents = [f"x{i+1}" for i in range(k)]
    for bits, expected in truth_table.items():
        inp = {afferents[i]: bits[i] for i in range(k)}
        zero_inp = {afferents[i]: 0 for i in range(k)}
        # Present input at t=0, then zeros for t=1..delay; check out_node at t=delay
        sched = [inp] + [zero_inp] * delay
        trace = net.simulate(sched)
        actual = trace[delay][out_node]
        if actual != expected:
            return False
    return True


def main():
    print("=== E2.1: Exhaustive Single-Neuron (1943 Eq. 1, theta >= 1) on 2 Inputs ===")
    single_2 = enumerate_single_neuron_2input()
    inputs_2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    names_2 = {
        (0, 0, 0, 0): "FALSE (0)",
        (0, 0, 0, 1): "AND (x1 . x2)",
        (0, 0, 1, 0): "x1 . ~x2 (Fig 1d)",
        (0, 0, 1, 1): "PROJ x1 (Fig 1a)",
        (0, 1, 0, 0): "~x1 . x2",
        (0, 1, 0, 1): "PROJ x2",
        (0, 1, 1, 0): "XOR (x1 ^ x2)",
        (0, 1, 1, 1): "OR (x1 v x2, Fig 1b)",
    }
    print(f"Total 2-input truth tables realized by 1 neuron: {len(single_2)} / 16")
    for tt_int in range(16):
        tt = tuple((tt_int >> (3 - i)) & 1 for i in range(4))
        status = "REALIZED" if tt in single_2 else "IMPOSSIBLE"
        detail = f"theta={single_2[tt][0]}, exc={single_2[tt][1]}, inh={single_2[tt][2]}" if tt in single_2 else "-"
        label = names_2.get(tt, f"f(0,0)=1 table #{tt_int}")
        print(f"  tt={tt} ({label:<22}): {status:<10} | {detail}")

    print("\n=== E2.2: Theorem 2 & 3 Compiler across all 16 2-input Boolean functions ===")
    compiled_2 = 0
    rejected_2 = 0
    for tt_int in range(16):
        tt_bits = tuple((tt_int >> (3 - i)) & 1 for i in range(4))
        tt_map = {inputs_2[i]: tt_bits[i] for i in range(4)}
        try:
            net, out_node, delay = compile_tpe_theorem2(2, tt_map)
            ok = verify_compiled_net(net, out_node, delay, 2, tt_map)
            assert ok, f"Verification failed on 2-input table {tt_bits}"
            compiled_2 += 1
        except ValueError:
            rejected_2 += 1
    print(f"2-input functions compiled & verified: {compiled_2}/16; rejected by Theorem 3 (f(0,0)=1): {rejected_2}/16")

    print("\n=== E2.3: Theorem 2 & 3 Compiler across all 256 3-input Boolean functions ===")
    inputs_3 = list(itertools.product([0, 1], repeat=3))
    compiled_3 = 0
    rejected_3 = 0
    neuron_counts = []
    for tt_int in range(256):
        tt_bits = tuple((tt_int >> (7 - i)) & 1 for i in range(8))
        tt_map = {inputs_3[i]: tt_bits[i] for i in range(8)}
        if tt_map[(0, 0, 0)] == 0:
            net, out_node, delay = compile_tpe_theorem2(3, tt_map)
            ok = verify_compiled_net(net, out_node, delay, 3, tt_map)
            assert ok, f"Verification failed on 3-input table #{tt_int}"
            compiled_3 += 1
            neuron_counts.append(len(net.neurons))
        else:
            try:
                compile_tpe_theorem2(3, tt_map)
                raise AssertionError("Should have rejected f(0,0,0)=1")
            except ValueError:
                rejected_3 += 1
    print(f"3-input functions compiled & 100% verified under Eq. (1): {compiled_3}/256")
    print(f"3-input functions rejected by Theorem 3 (f(0,0,0)=1):    {rejected_3}/256")
    print(f"Constructed net sizes (non-afferent neurons): min={min(neuron_counts)}, median={sorted(neuron_counts)[len(neuron_counts)//2]}, max={max(neuron_counts)}")

    print("\n=== E2.4: Negative Controls ===")
    # Negative Control 1: Can ANY acyclic net of 1943 neurons (theta >= 1) output 1 when afferents are 0?
    # Test 1,000 random order-0 nets of up to 8 neurons with theta >= 1 on all-zero afferent schedule.
    import random
    rng = random.Random(1943)
    any_fired_on_zero = 0
    for _ in range(1000):
        net = MPNet1943(["x1", "x2", "x3"])
        names = ["x1", "x2", "x3"]
        for idx in range(6):
            nname = f"c{idx+1}"
            exc = {s: rng.randint(1, 3) for s in names if rng.random() < 0.6}
            inh = [s for s in names if rng.random() < 0.3]
            theta = rng.randint(1, 4)
            net.add_neuron(nname, theta=theta, excitatory=exc, inhibitory=inh)
            names.append(nname)
        trace = net.simulate([{"x1": 0, "x2": 0, "x3": 0}] * 7)
        if any(trace[t][f"c{idx+1}"] != 0 for t in range(7) for idx in range(6)):
            any_fired_on_zero += 1
    print(f"Negative Control 1 (1,000 random order-0 nets on all-zero input): {any_fired_on_zero}/1000 ever fired (Theorem 3 necessity)")

    # Negative Control 2: Mutate threshold of the compiled 3-input parity (x1 ^ x2 ^ x3) net and confirm verifier fails
    parity_map = {b: (b[0] ^ b[1] ^ b[2]) for b in inputs_3}
    pnet, pout, pdelay = compile_tpe_theorem2(3, parity_map)
    assert verify_compiled_net(pnet, pout, pdelay, 3, parity_map) is True
    pnet.neurons["term_0"].theta += 1  # corrupt one minterm threshold
    corrupted_ok = verify_compiled_net(pnet, pout, pdelay, 3, parity_map)
    print(f"Negative Control 2 (corrupted threshold on 3-input parity net): verify_compiled_net={corrupted_ok} (expected False)")
    assert corrupted_ok is False


if __name__ == "__main__":
    main()
