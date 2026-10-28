"""
E3 & E4 (Turing 1950, Sections 1, 2, 6, 7):
  - E3.1: Imitation Game (Sections 1, 2, 6.5, pp. 1-3, 8, 13):
          Interrogator C distinguishing Human B vs Machine A using arithmetic latency and
          "errors of conclusion" (Turing's 34957 + 70764 = 105621 after 30s pause).
          Compares Honest Machine A1 (instant, 0% carry slip) vs Spoofing Machine A2
          (Turing p. 13: "deliberately introduce mistakes in a manner calculated to confuse
          the interrogator") vs Faulty Machine A3 ("errors of functioning": mechanical fault
          making implausible non-carry errors).
  - E3.2: The Mathematical Objection (Section 6.3, pp. 10-11):
          Diagonal critical question Q(M_i): "Will M_i answer 'No' to Q(M_i)?"
          Demonstrates that every machine M_i fails (wrong answer or non-termination) on its own
          critical question Q(M_i), while another machine M_{i+1} answers Q(M_i) 100% correctly
          ("questions which cannot be answered by one machine may be satisfactorily answered by another").
  - E3.3: Continuity in the Nervous System (Section 6.7, pp. 15-16):
          Digital computer with stochastic output table ([3.12, 3.13, 3.14, 3.15, 3.16] with
          Turing's exact probabilities [0.05, 0.15, 0.55, 0.19, 0.06] on p. 16) mimicking a
          continuous differential analyser with Gaussian measurement noise around pi.
  - E4.1: Reward/Punishment Information Bound vs "Unemotional" Symbolic Channel (Section 7, p. 20):
          Teaching the first line of "Casabianca" ("The boy stood on the burning deck", 33 chars)
          by pure binary reward/punishment ("Twenty Questions" / blow per wrong guess) vs
          teaching a small symbolic alphabet map by R/P and transmitting the text over the
          unemotional channel.
  - E4.2: Child Machine with Built-in Logical Inference, Epistemic Statuses, and Imperatives (pp. 20-21):
          Implements Turing's 5 proposition statuses + imperatives ("Do your homework now",
          "When Socrates is mentioned, use the syllogism in Barbara", "If one method has been
          proved to be quicker than another, do not use the slower method").
  - E4.3: Random vs Systematic Search in Learning Machines (Section 7, p. 22):
          Turing's exact problem: "find a number between 50 and 200 which was equal to the
          square of the sum of its digits" (n = 81 = (8+1)^2), comparing systematic orderings vs
          memoryless random sampling.
"""
import math
import random
from typing import Dict, List, Optional, Tuple


def run_e3_imitation_and_objections():
    print("=== E3.1: Imitation Game — Arithmetic Speed & Error Spoofing (Sections 2 & 6.5, pp. 2, 13) ===")
    # Human B profile on 5-digit additions (like 34957 + 70764):
    #   - Latency: N(mu=30.0s, sigma=6.0s)
    #   - Error of conclusion: 18% chance of single-column carry slip (off by +/- 10^k, k in 1..4), 82% exact.
    # Interrogator C asks K=5 5-digit addition problems and uses a log-likelihood ratio (Bayes optimal)
    # classifier between Human B and the suspected Machine model, or anomaly detector against Human B's distribution:
    # Specifically, C checks whether the responder's latency and error pattern match Human B's distribution:
    # between two witnesses X and Y (one Human B, one Machine A), C picks as "Human" whichever witness has
    # higher log-likelihood under the true Human B model!
    rng = random.Random(1950)

    def sample_response(agent_type: str) -> Tuple[float, int]:
        # Returns (latency_sec, abs_error)
        if agent_type == "human_B":
            lat = max(2.0, rng.gauss(30.0, 6.0))
            err = 10 ** rng.randint(1, 4) if rng.random() < 0.18 else 0
            return lat, err
        elif agent_type == "honest_machine_A1":
            # Fast (0.001s) and deadly accurate (0 error) -> unmasked!
            return 0.001, 0
        elif agent_type == "spoofing_machine_A2":
            # Turing's strategy (pp. 2, 13): pause ~30s and deliberately introduce human-like carry mistakes
            lat = max(2.0, rng.gauss(30.0, 6.0))
            err = 10 ** rng.randint(1, 4) if rng.random() < 0.18 else 0
            return lat, err
        elif agent_type == "faulty_machine_A3":
            # Turing p. 13: "A mechanical fault would probably show itself through an unsuitable decision
            # as to what sort of a mistake to make in the arithmetic" (e.g. pause 30s, but when making an
            # error, outputs a random non-power-of-10 error like 37192)
            lat = max(2.0, rng.gauss(30.0, 6.0))
            err = rng.randint(3, 99999) if rng.random() < 0.18 else 0
            return lat, err
        raise ValueError(agent_type)

    def human_log_likelihood(trials: List[Tuple[float, int]]) -> float:
        ll = 0.0
        for lat, err in trials:
            # Gaussian log-density for latency around 30s (sigma=6s)
            ll += -0.5 * ((lat - 30.0) / 6.0) ** 2
            # Error type log-prob: 0 -> 0.82, 10^k (k=1..4) -> 0.18/4, other -> 1e-6 (unsuitable mistake!)
            if err == 0:
                ll += math.log(0.82)
            elif err in (10, 100, 1000, 10000):
                ll += math.log(0.18 / 4.0)
            else:
                ll += math.log(1e-6)
        return ll

    n_games = 2000
    k_questions = 5
    for m_type, desc in [
        ("honest_machine_A1", "Honest Machine (0.001s, 0% error)"),
        ("spoofing_machine_A2", "Turing Spoofing Machine (~30s pause, 18% carry slip, like 105621)"),
        ("faulty_machine_A3", "Unsuitable-Error Machine (~30s pause, non-carry mechanical mistakes)"),
    ]:
        right_id = 0
        for _ in range(n_games):
            b_trials = [sample_response("human_B") for _ in range(k_questions)]
            a_trials = [sample_response(m_type) for _ in range(k_questions)]
            if human_log_likelihood(b_trials) > human_log_likelihood(a_trials):
                right_id += 1
        acc = right_id / n_games
        print(f"  {desc:<68}: Interrogator right ID = {acc*100:5.1f}% (Turing 50-yr target <= 70.0%)")

    print("\n=== E3.2: Section 6.3 The Mathematical Objection (Diagonal Question Sequence, pp. 10-11) ===")
    # Consider a sequence of machines M_0, M_1, M_2, M_3 with increasing step budgets B_i = 10 * (i + 1).
    # For each machine M_k, its critical diagonal question Q_k asks:
    #   "Does M_k(Q_k) return 'No' (or fail to halt within B_k steps)?"
    # When M_k tries to answer Q_k by simulating M_k(Q_k) for up to B_k steps, each recursive call
    # consumes at least 1 step of budget, so M_k exhausts its B_k budget and fails to answer Q_k;
    # whereas M_{k+1} (with larger budget B_{k+1} > B_k) simulates M_k(Q_k) to completion (seeing that
    # M_k timed out after B_k steps) and returns the TRUE answer 'Yes' in B_k + 1 steps!
    budgets = [10, 20, 30, 40]

    def run_machine(m_idx: int, q_idx: int, remaining_budget: Optional[int] = None) -> Tuple[str, int]:
        if remaining_budget is None:
            remaining_budget = budgets[m_idx]
        if remaining_budget <= 0:
            return "TIMEOUT", 0
        # To answer Q_{q_idx} ("Does M_{q_idx}(Q_{q_idx}) fail to say 'Yes'?"),
        # machine M_{m_idx} simulates M_{q_idx} on Q_{q_idx} for min(remaining_budget - 1, budgets[q_idx]) steps:
        sim_budget = min(remaining_budget - 1, budgets[q_idx])
        sub_ans, sub_steps = run_machine(q_idx, q_idx, remaining_budget=sim_budget)
        total_steps = sub_steps + 1
        if sub_ans == "TIMEOUT" and sim_budget < budgets[q_idx]:
            # M_{m_idx} ran out of its OWN budget before M_{q_idx}'s budget B_{q_idx} expired!
            return "TIMEOUT", total_steps
        # Otherwise M_{m_idx} observed the complete outcome of M_{q_idx}(Q_{q_idx}):
        ans = "NO" if sub_ans == "YES" else "YES"
        return ans, total_steps

    print("  Machine \\ Critical Question |   Q_0 (for M_0)   |   Q_1 (for M_1)   |   Q_2 (for M_2)   |   Q_3 (for M_3)")
    for m_idx in range(4):
        cells = []
        for q_idx in range(4):
            ans, _ = run_machine(m_idx, q_idx)
            # Ground truth of Q_{q_idx}: since M_{q_idx}(Q_{q_idx}) always times out, the true answer to Q_{q_idx} is "YES"!
            correct = (ans == "YES")
            cells.append(f"{ans:<7} ({'OK' if correct else 'FAIL':<4})")
            if m_idx == q_idx:
                assert not correct, "Machine cannot answer its own diagonal question!"
            elif m_idx > q_idx:
                assert correct, "Higher machine must answer lower machine's diagonal question!"
        print(f"  M_{m_idx} (budget={budgets[m_idx]:2d})            | " + " | ".join(cells))

    print("\n=== E3.3: Section 6.7 Continuity in the Nervous System (Differential Analyser on pi, p. 16) ===")
    # Turing's discrete stochastic table for pi (p. 16):
    vals = [3.12, 3.13, 3.14, 3.15, 3.16]
    probs = [0.05, 0.15, 0.55, 0.19, 0.06]
    assert abs(sum(probs) - 1.0) < 1e-9
    mean_turing = sum(v * p for v, p in zip(vals, probs))
    var_turing = sum(((v - mean_turing) ** 2) * p for v, p in zip(vals, probs))
    sd_turing = math.sqrt(var_turing)
    # Continuous differential analyser with normal error N(3.14159, sd_turing^2) rounded to 2 decimal places:
    cont_counts = {v: 0 for v in vals}
    n_samples = 100_000
    for _ in range(n_samples):
        x = round(rng.gauss(math.pi, sd_turing), 2)
        x = min(3.16, max(3.12, x))
        cont_counts[x] += 1
    cont_probs = [cont_counts[v] / n_samples for v in vals]
    tv_dist = 0.5 * sum(abs(p - q) for p, q in zip(probs, cont_probs))
    print(f"  Turing p. 16 discrete distribution: vals={vals}, probs={probs}, mean={mean_turing:.5f}, sd={sd_turing:.5f}")
    print(f"  Rounded continuous analyser probs:  {[round(q, 4) for q in cont_probs]}, TV distance = {tv_dist:.4f}")


def run_e4_learning_machines():
    print("\n=== E4.1: Section 7 Reward/Punishment Bound vs Unemotional Symbolic Channel (p. 20) ===")
    # Target text: "CASABIANCA" (10 letters) and first line "THE BOY STOOD ON THE BURNING DECK" (33 chars, alphabet V=27: A-Z and space).
    # Method 1 (Pure Reward/Punishment "Twenty Questions" / blow per wrong symbol, no symbolic channel):
    #   For each position, without an unemotional channel, the pupil must try symbols from the 27-symbol alphabet
    #   until rewarded (expected (V+1)/2 = 14.0 blows/rewards per character if tried without replacement, or V/2).
    # Method 2 (Teach the V=27 symbol-action mapping once via R/P, then transmit any text of length L over the
    #   unemotional symbolic channel with 0 further blows, plus 1 final reward):
    V = 27
    # Expected R/P signals to learn V symbol mappings by trial-and-error without replacement:
    # For the k-th symbol (k=0..V-1), expected trials among V (or remaining V-k) actions:
    rp_to_learn_language = sum((V - k + 1) / 2.0 for k in range(V))
    for label, text in [("Word 'CASABIANCA' (L=10)", "CASABIANCA"), ("Line 1 'THE BOY STOOD ON THE BURNING DECK' (L=33)", "THE BOY STOOD ON THE BURNING DECK"), ("Full poem 'Casabianca' (~1,600 chars)", "A" * 1600)]:
        L = len(text)
        rp_only_expected = L * ((V + 1) / 2.0)
        unemotional_channel_AfterLang = 1  # 1 reward after obeying symbolic stream
        print(f"  {label:<52}: Pure R/P = {rp_only_expected:8.1f} blows/rewards | Symbolic channel (after {rp_to_learn_language:.0f}-signal language setup) = {unemotional_channel_AfterLang} signal")

    print("\n=== E4.2: Section 7 Child Machine with Epistemic Statuses & Imperatives (pp. 20-21) ===")
    # Propositions have status in:
    #   {"well_established", "conjecture", "proved_theorem", "given_by_authority", "no_belief_value"}
    # Imperative execution rule (p. 20):
    #   "as soon as an imperative is classed as 'well established' the appropriate action automatically takes place."
    store = {
        "Teacher says 'Do your homework now'": "well_established",
        "Everything that teacher says is true": "well_established",
        "Do your homework now": "no_belief_value",  # expression having logical form of imperative, not yet established
        "Socrates is a man": "well_established",
        "All men are mortal": "well_established",
    }
    imperatives = {"Do your homework now", "Use syllogism in Barbara", "Use exhaustive truth-table search"}
    executed_actions = []

    # Inference step 1 (p. 20): Teacher authority rule
    if (store.get("Teacher says 'Do your homework now'") == "well_established"
            and store.get("Everything that teacher says is true") == "well_established"):
        store["Do your homework now"] = "well_established"

    for prop, status in store.items():
        if prop in imperatives and status == "well_established":
            executed_actions.append(prop)

    # Inference step 2 (p. 21): Search-order imperatives ("These choices make the difference between a brilliant
    # and a footling reasoner, not the difference between a sound and a fallacious one.")
    # Compare proving "Socrates is mortal" from 10 axioms (2 relevant, 8 irrelevant distractors):
    #   - Footling reasoner (no search-order imperative): tries all pairs of axioms in order (10 * 9 = 90 pairs,
    #     putting Socrates at index 8,9 -> 72 steps).
    #   - Brilliant reasoner (obeys imperative "When Socrates is mentioned, use the syllogism in Barbara" and
    #     "If one method has been proved to be quicker than another, do not use the slower method"):
    #     filters immediately to propositions mentioning "Socrates" and applies Barbara in 1 step!
    axioms = [f"Distractor_{i} is P_{i}" for i in range(8)] + ["Socrates is a man", "All men are mortal"]
    footling_steps = 0
    for a in axioms:
        for b in axioms:
            footling_steps += 1
            if a == "Socrates is a man" and b == "All men are mortal":
                break
        else:
            continue
        break

    brilliant_steps = 0
    for a in [x for x in axioms if "Socrates" in x]:
        for b in [x for x in axioms if "is a man" in a and "All men" in x]:
            brilliant_steps += 1

    print(f"  Status of 'Do your homework now' after inference: {store['Do your homework now']} -> executed_actions={executed_actions}")
    print(f"  Steps to prove 'Socrates is mortal' among 10 axioms: without search imperative = {footling_steps} steps vs with Barbara imperative = {brilliant_steps} step")
    assert executed_actions == ["Do your homework now"] and brilliant_steps == 1 and footling_steps > 50

    print("\n=== E4.3: Section 7 Random vs Systematic Search on [50, 200] Digit-Sum Squared (p. 22) ===")
    # "Suppose for instance we wanted to find a number between 50 and 200 which was equal to the square
    #  of the sum of its digits, we might start at 51 then try 52 and go on until we got a number that worked.
    #  Alternatively we might choose numbers at random until we got a good one."
    candidates = list(range(50, 201))  # 151 numbers
    solutions = [n for n in candidates if n == sum(int(d) for d in str(n)) ** 2]
    sys_up_steps = candidates.index(solutions[0]) + 1          # starting at 50 (or 51: 31 steps)
    sys_down_steps = list(reversed(candidates)).index(solutions[0]) + 1  # starting at 200 across the dead block 82..200

    # Also test Turing's general claim on p. 22: "The systematic method has the disadvantage that there may be
    # an enormous block without any solutions in the region which has to be investigated first... Since there is
    # probably a very large number of satisfactory solutions the random method seems to be better than the systematic."
    # Suppose out of N=151 candidates, the first B=120 are an empty dead block and the last k=15 are all solutions:
    rng = random.Random(1950)
    rand_single_trials = []
    rand_multi_trials = []
    for _ in range(5000):
        t = 1
        while rng.choice(candidates) not in solutions:
            t += 1
        rand_single_trials.append(t)
        t2 = 1
        while rng.randrange(151) < 120:  # first 120 are dead block, last 31 are solutions
            t2 += 1
        rand_multi_trials.append(t2)

    print(f"  Solutions to n == (sum of digits of n)^2 in [50, 200]: {solutions} (81 = (8+1)^2)")
    print(f"  Single solution (n=81): Systematic from 51 = {81 - 51 + 1} steps | Systematic from 200 (dead block first) = {sys_down_steps} steps | Random mean = {sum(rand_single_trials)/len(rand_single_trials):.1f} steps")
    print(f"  Dead-block-first with 31/151 solutions (Turing p. 22 regime): Systematic = 121 steps | Random mean = {sum(rand_multi_trials)/len(rand_multi_trials):.2f} steps (theoretical 151/31 = {151/31:.2f})")
    assert solutions == [81]


if __name__ == "__main__":
    run_e3_imitation_and_objections()
    run_e4_learning_machines()
