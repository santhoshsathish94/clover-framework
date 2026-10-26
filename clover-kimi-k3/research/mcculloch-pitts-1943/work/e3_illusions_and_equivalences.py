"""
E3: Cutaneous heat-from-transient-cooling illusion (Fig. 1e, pp. 105-107)
and Equivalence Theorems 4, 5, 6 (Fig. 1f-h, pp. 105-108) in McCulloch & Pitts (1943).
"""
import itertools
from mp1943_calculus import MPNet1943


def build_fig1e_cutaneous_net() -> MPNet1943:
    """
    Constructs the exact net of pages 106-107 (Fig. 1e):
      Peripheral afferents:
        c1: cutaneous heat receptor
        c2: cutaneous cold receptor
      Internal neurons:
        ca: Fig. 1a delay from c2 -> Na(t) = S N2(t) = N2(t-1)
        cb: Fig. 1d from ca (exc) and c2 (inh) -> Nb(t) = S[Na(t) . ~N2(t)] = N2(t-2) . ~N2(t-1)
      Output sensation neurons:
        c4 (cold sensation): Fig. 1c from ca and c2 -> N4(t) = S[Na(t) . N2(t)] = N2(t-2) . N2(t-1)
        c3 (heat sensation): Fig. 1b from c1 and cb -> N3(t) = S[N1(t) v Nb(t)] = N1(t-1) v (N2(t-3) . ~N2(t-2))
    """
    net = MPNet1943(["c1", "c2"])
    net.fig1a_delay("ca", "c2")
    net.fig1d_and_not("cb", ["ca"], ["c2"])
    net.fig1c_and("c4", ["ca", "c2"])
    net.fig1b_or("c3", ["c1", "cb"])
    return net


def run_fig1e_experiments():
    print("=== E3.1: Cutaneous Heat-from-Transient-Cooling Illusion (Fig. 1e, pp. 106-107) ===")
    net = build_fig1e_cutaneous_net()

    scenarios = [
        ("Transient cooling (1-step cold pulse at t=0)",
         [{"c1": 0, "c2": 1}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}]),
        ("Sustained cooling (3-step cold at t=0,1,2)",
         [{"c1": 0, "c2": 1}, {"c1": 0, "c2": 1}, {"c1": 0, "c2": 1}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}]),
        ("Direct heat (1-step heat pulse at t=0)",
         [{"c1": 1, "c2": 0}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}, {"c1": 0, "c2": 0}]),
    ]

    for title, sched in scenarios:
        trace = net.simulate(sched)
        print(f"\nScenario: {title}")
        print("  t | c1(heat_in) c2(cold_in) | ca(S c2) cb(trans_cold) | c3(HEAT_SENS) c4(COLD_SENS)")
        for t, st in enumerate(trace):
            print(f"  {t} |      {st['c1']}           {st['c2']}      |    {st['ca']}           {st['cb']}        |       {st['c3']}             {st['c4']}")
            if t >= 3:
                # Verify analytical formulas on p. 106:
                exp_c3 = 1 if (trace[t-1]["c1"] == 1 or (trace[t-3]["c2"] == 1 and trace[t-2]["c2"] == 0)) else 0
                exp_c4 = 1 if (trace[t-2]["c2"] == 1 and trace[t-1]["c2"] == 1) else 0
                assert st["c3"] == exp_c3 and st["c4"] == exp_c4

    # Check key physiological predictions on p. 106:
    tr_transient = net.simulate(scenarios[0][1])
    assert [st["c3"] for st in tr_transient] == [0, 0, 0, 1, 0], "Transient cooling must produce heat at t=3"
    assert [st["c4"] for st in tr_transient] == [0, 0, 0, 0, 0], "Transient cooling must produce zero cold"

    tr_sustained = net.simulate(scenarios[1][1])
    # Note: During sustained cooling t=0,1,2, cold is felt at t=2,3 (with zero preliminary warmth at t=1,2,3!).
    # What happens at t=4 (2 steps after removal of sustained cooling at t=3)?
    # Because c2(2)=1 and c2(3)=0, the trailing edge of sustained cooling ALSO satisfies N2(t-3) & ~N2(t-2) at t=5!
    print("\nExtended trace of sustained cooling (t=0,1,2 cold on, t=3,4,5,6 cold off):")
    sched_long = [{"c1": 0, "c2": 1}] * 3 + [{"c1": 0, "c2": 0}] * 4
    tr_long = net.simulate(sched_long)
    for t, st in enumerate(tr_long):
        print(f"  t={t}: c2={st['c2']} -> ca={st['ca']}, cb={st['cb']} -> HEAT(c3)={st['c3']}, COLD(c4)={st['c4']}")


def run_theorem4_fig1f():
    """
    Theorem 4 (p. 107) & Fig. 1f (p. 105): Relative vs Absolute Inhibition.
    In Fig. 1f:
      Relative inhibition neuron c4 has base threshold theta = 1.
      - c1 has 1 relative-inhibitory synapse on c4 (raises threshold by +1 when c1=1, to 2).
      - c2 has 1 excitatory synapse on c4.
      - c3 has 1 excitatory synapse on c4.
      Wait: if c1=0, threshold is 1, so c2 v c3 fires c4: ~N1(t-1) . (N2(t-1) v N3(t-1)).
            if c1=1, threshold is 2, so BOTH c2 and c3 must fire: N1(t-1) . N2(t-1) . N3(t-1).
      Let's check Fig. 1f caption on p. 105!
      Caption (f) says:
        N_4(t) .=. ~N_1(t-1) . N_2(t-1) v N_3(t-1) .v. N_1(t-1) . N_2(t-1) . N_3(t-1)
      Using Principia dot conventions (p. 102): `.v.` is the main disjunction, and `.` binds tighter than `v`,
      so `~N_1(t-1) . (N_2(t-1) v N_3(t-1))  v  (N_1(t-1) . N_2(t-1) . N_3(t-1))`!
      And the second line of (f) for the absolute-inhibition net is:
        N_4(t) .=. ~N_1(t-2) . (N_2(t-2) v N_3(t-2))  v  (N_1(t-2) . N_2(t-2) . N_3(t-2)),
      which is S^1 applied to the relative-inhibition neuron!
    """
    print("\n=== E3.2: Theorem 4 (Fig. 1f) Relative vs Absolute Inhibition Equivalence ===")
    # Absolute inhibition net realizing S^1 of the relative-inhibition neuron:
    # Intermediates at t-1:
    #   ca: ~c1 . c2   (exc: c2, inh: c1, theta=1)
    #   cb: ~c1 . c3   (exc: c3, inh: c1, theta=1)
    #   cc: c1 . c2 . c3 (exc: c1, c2, c3, theta=3; or since c2.c3 suffices regardless of c1, exc: c2, c3, theta=2)
    # Output c4 at t: disjunction of ca, cb, cc (theta=1).
    net_abs = MPNet1943(["c1", "c2", "c3"])
    net_abs.fig1d_and_not("ca", ["c2"], ["c1"])
    net_abs.fig1d_and_not("cb", ["c3"], ["c1"])
    net_abs.fig1c_and("cc", ["c1", "c2", "c3"])
    net_abs.fig1b_or("c4", ["ca", "cb", "cc"])

    all_match = True
    print("  c1 c2 c3 | RelInh N4(t=1) | AbsInh N4(t=2) | Formula Fig.1f")
    for c1, c2, c3 in itertools.product([0, 1], repeat=3):
        # Relative inhibition rule: base theta=1, c1 adds +1 to theta, c2 and c3 each add +1 excitation
        eff_theta = 1 + c1
        exc_sum = c2 + c3
        rel_out = 1 if exc_sum >= eff_theta else 0

        # Fig. 1f formula with Principia dots: (~c1 & (c2 | c3)) | (c1 & c2 & c3)
        form_out = int(((not c1) and (c2 or c3)) or (c1 and c2 and c3))

        # Simulate absolute inhibition net for 2 steps
        tr = net_abs.simulate([{"c1": c1, "c2": c2, "c3": c3}, {"c1": 0, "c2": 0, "c3": 0}, {"c1": 0, "c2": 0, "c3": 0}])
        abs_out = tr[2]["c4"]
        if not (rel_out == form_out == abs_out):
            all_match = False
        print(f"   {c1}  {c2}  {c3} |       {rel_out}        |       {abs_out}        |       {form_out}")
    assert all_match, "Theorem 4 equivalence failed"
    print(f"Theorem 4 extended equivalence (S^1 shift, t=1 vs t=2) verified across all 8 inputs: {all_match}")


def run_theorem5_and_6_fig1gh():
    """
    Theorem 5 (p. 107-108, Fig. 1g): Extinction vs Absolute Inhibition.
      Fig. 1g caption: N_3(t) .=. N_1(t-2) . ~N_2(t-3) (or self-extinction after firing: N_3(t) = N_1(t-2) . ~N_1(t-3)).
      Specifically, suppose neuron c3 has base threshold theta=1 from c1, and after c1 fires (or c3 fires),
      threshold rises by +1 for 1 step (extinction). Then c3 fires on the first pulse of c1 but is extinguished
      on the immediately following step: N_3(t) = N_1(t-2) . ~N_1(t-3).
    Theorem 6 (p. 108, Fig. 1h): Facilitation / Temporal Summation vs Spatial Summation.
      Fig. 1h caption: N_3(t) .=. N_1(t-1) . N_1(t-2).
      c1 synapses directly on c3 AND on delay neuron c2 (N_2(t) = N_1(t-1)), which also synapses on c3 (theta=2).
    """
    print("\n=== E3.3: Theorem 5 (Fig. 1g Extinction) & Theorem 6 (Fig. 1h Temporal->Spatial Summation) ===")
    # Fig. 1g net: c2 delays c1 by 1 step (N2(t) = N1(t-1));
    # wait: to align N1(t-2) and ~N1(t-3) at c3(t), let ca delay c1 (Na(t)=N1(t-1)), cb delay ca (Nb(t)=N1(t-2)),
    # and c3 have excitatory from ca and inhibitory from cb -> N3(t) = Na(t-1) . ~Nb(t-1) = N1(t-2) . ~N1(t-3)!
    net_1g = MPNet1943(["c1"])
    net_1g.fig1a_delay("ca", "c1")
    net_1g.fig1a_delay("cb", "ca")
    net_1g.fig1d_and_not("c3", ["ca"], ["cb"])

    # Fig. 1h net: c2 delays c1 by 1 step (N2(t) = N1(t-1)); c3 has exc from c1 and c2 with theta=2
    # -> N3(t) = N1(t-1) . N2(t-1) = N1(t-1) . N1(t-2).
    net_1h = MPNet1943(["c1"])
    net_1h.fig1a_delay("c2", "c1")
    net_1h.fig1c_and("c3", ["c1", "c2"])

    # Verify across all 2^6 = 64 binary input streams of length 6
    ok_1g = 0
    ok_1h = 0
    for bits in itertools.product([0, 1], repeat=6):
        sched = [{"c1": b} for b in bits] + [{"c1": 0}] * 3
        tr_g = net_1g.simulate(sched)
        tr_h = net_1h.simulate(sched)
        match_g = all(tr_g[t]["c3"] == (1 if (bits[t-2] == 1 and (t < 3 or bits[t-3] == 0)) else 0) for t in range(2, 6))
        match_h = all(tr_h[t]["c3"] == (1 if (bits[t-1] == 1 and bits[t-2] == 1) else 0) for t in range(2, 6))
        if match_g:
            ok_1g += 1
        if match_h:
            ok_1h += 1
    print(f"Theorem 5 (Fig. 1g extinction N3(t) = N1(t-2) . ~N1(t-3)) verified on {ok_1g}/64 6-step streams")
    print(f"Theorem 6 (Fig. 1h temporal summation N3(t) = N1(t-1) . N1(t-2)) verified on {ok_1h}/64 6-step streams")
    assert ok_1g == 64 and ok_1h == 64


if __name__ == "__main__":
    run_fig1e_experiments()
    run_theorem4_fig1f()
    run_theorem5_and_6_fig1gh()
