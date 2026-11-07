# Running "The Exponential Time Hypothesis is False": what it teaches

Direction (human, 2026-11-07): new branch and worktree; run the experiment and see what it can teach, because only
running will help us understand, not reading. Branch `research/eth-subexponential-counting` from `main` 87dce4c.

## Sources (cloned, not vendored)

- Paper and Lean proof: `openai/math` @ `4eaee87`,
  `preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/` (LaTeX `main_full_version.tex`, sha256
  `BDFD52A5...`). Claim: exact #k-SAT in `(n+m)^{O_k(1)} 2^{n/(log n)^{1/2+o(1)}}` time.
- Independent kernel check and mechanism test: `unaltuzun/exponential-time-hypothesis-verify` @ `a775e22`
  (their build: 0 errors, axioms `propext, Classical.choice, Quot.sound`, Comparator accepted; we read their logs, we
  did not rebuild the Lean proof).
- Limits of the method: `Mehrshadr/eth-method-limits` @ `ca7f933` (read only).

## What ran here (i7-1355U, Windows, Python 3.14.7, numpy 2.5.4)

| # | Run | Result |
|---|---|---|
| 1 | The core mechanism (exact arithmetic, `mechanism_test.py` from the verify repo, seeds 1-3; 350/180/454 s) | ALL CHECKS PASSED. Section 4.2 lemmas hold; counts recovered exactly by Hermite interpolation for n = 5, 6, 7 (19/32/65, 19/29/55, 16/27/50). At these sizes the Taylor order q is 5-10, close to or above n, so that step alone costs nearly 3^n. |
| 2 | The paper's own parameter checks (`experiments/paper_parameters.py`, 1 s) | Every n up to 2^1,000,000 fails a check and the paper falls back to all 2^n assignments (L3152, L3185-3197). The engine first runs at n ~ 10^3,284,052,030; there it costs 2^1.52n (worse than exhaustive). Below 2^n only from n ~ 10^(1.85x10^10); below 2^0.5n from n ~ 10^(1.57x10^11). |
| 3 | Best case with tuned parameters (`experiments/method_optimum.py`, 5 s) | Keeping only the construction's own requirements and choosing r, Q, R freely (polynomial factors ignored, which favours the engine): beats exhaustive search only from n ~ 10^48,175. For every n up to 2^16,000 the best design costs at least 2^1.6n (Taylor step = 3^n; the allowed designs only grow with n). Without the schedule-balance requirement (auxiliary matrix allowed 2^(n/2) work): from n ~ 1.37 million (r = 5, Q = 27,361, R = 961, a 2^127,061-dimensional auxiliary space). |
| 4 | Practical exact counters on the paper's input class and two others (`experiments/practical_counters.py`, 5 seeds, ~25 min) | Exponent per variable (slope of log2 work vs n): see the table below. 250 cross-checks between the counters passed. |

Run 4, largest sizes reached within a 90 s budget per size (median of 5 seeds):

| Instances | Brute force (all 2^n) | Search + components + cache | Tensor contraction (min-degree order) |
|---|---|---|---|
| bounded: 3-CNF, every variable in <= 3 clauses (the paper's engine input) | 1.11; n = 22 in 4.4 s | 0.33; n = 60: 6.6 M nodes, 116 s | 0.33; n = 60: 24 M table entries, 0.6 s (width 23; n = 70 wider than 24) |
| random2: random 3-CNF, m = 2n | 1.10; n = 22 in 4.5 s | 0.29; n = 70: 6.3 M nodes, 101 s | 0.45; n = 40 (width 21; n = 50 wider than 24) |
| chain: clauses on (i, i+1, i+2), treewidth 2 | 1.12; n = 22 in 1.6 s | 0.02; n = 300 in 0.2 s | 0.01; n = 300 in 0.03 s (width 3) |

Brute-force slopes are above 1 because the work counts clause checks (2^n x m). Slopes are fitted at small n and
include polynomial factors.

## What running showed that reading did not

1. **The theorem is about the limit; at any size that exists, the algorithm is exhaustive search.** The paper says
   the fallback happens for "finitely many n". Running its own checks shows those n are every n below about
   10^3,284,052,030, and it is slower than 2^n until about 10^(1.85x10^10). Even with parameters tuned for speed the
   crossover is about 10^48,175 variables. "Subexponential" here names a growth class, not a speed anyone will see.
2. **Representation changed the measured exponent at feasible sizes, and it was old structure, not the new idea.**
   On the paper's own input class, splitting into independent components with a cache, or contracting the clause
   network in a good order, took the exponent from 1.0 to 0.33 per variable; on structured inputs (low treewidth)
   the cost became polynomial. That is the user's "representation before acceleration" and "exploit structure",
   measured: a 3x smaller exponent at n = 60 is about 2^40 times less work than brute force.
3. **Structure decides practical cost.** The same counter is polynomial on a chain and exponential on an expander-like
   formula. The worst-case class says nothing about which of the two our inputs are; only measuring our inputs does.
4. **A growth claim needs its crossover point.** An exponent without constants did not predict the regime: here
   the asymptotically better method costs 2^1.5n where it starts, while on the same inputs counters measured at
   2^0.33n (for n <= 60) already exist.

## What this does not show

- Nothing here disputes the theorem. The mechanism held exactly on every instance run; the independent group's
  kernel check accepted the Lean proof (we did not rebuild it: about 40 GB and Mathlib cache needed).
- The cost model is the paper's eq. (overview-three-costs) for the bounded-occurrence engine (Sections 2-4), up to
  polynomial factors. The Appendix D/E tensor-contraction route behind Theorem 1.1's `n/sqrt(log n)` bound was not
  modelled; its constants were not derived. The Lean proof uses its own (similar) parameters and its fallback cutoff
  comes from an "eventually" statement, so it has no explicit value; not computed here.
- Practical counters are Python prototypes on one machine, small n, 5 seeds per size; contraction widths capped at 24.

## For Clover

- Clover's runtime (`clover-ai-system/studio/clover`, `mobile`) has no code that enumerates combinations (searched for
  itertools products/combinations); its heavy costs are model inference, whose growth Clover already measures (K3:
  KDA constant vs MLA linear per token). So the paper's technique has no direct target in Clover today.
- What carries over is the method: fit how cost grows with input size on Clover's own inputs before calling
  something exponential or fast; look for independent parts, reuse (caches) and contraction order before buying
  hardware; ask for the crossover size of any asymptotic gain.
- If a Clover layer ever needs an exact sum over combinations (planning over tool sequences, inference over the
  memory or ontology graph), the measured tools are component caching and tree-decomposition contraction.

## Rerun

```
python experiments/paper_parameters.py
python experiments/method_optimum.py            # add --dim free for the sensitivity run
python experiments/practical_counters.py        # --quick for a short check
# mechanism (external): python -I scripts/mechanism_test.py <seed>  in unaltuzun/exponential-time-hypothesis-verify
```

Results: `results/*.json`, `results/practical_counters.log`, `results/mechanism_test_seed{1,2,3}.log`.
