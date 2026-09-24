# v4 — code

**Probes:** a different domain. Python source instead of prose.

```
prompt   "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
         (60 bytes, sha1 154cc3deb3cc, 18 ids)
gen      4
trace    /root/k3flow/v4.jsonl        109.0 s wall, I/O 33.3%
```

## Outcome

Generated `' fibonacci(n-'` — 21490, 96445, 2145, 12. A correct continuation of the
recursive call.

| step | seconds | expert GB read | cache hit |
|---:|---:|---:|---:|
| 0 (prefill, T=18) | 86.05 | 210.06 | 49.2% |
| 1 | 9.13 | 25.51 | 99.9% |
| 2 | 7.13 | 16.56 | 100% |
| 3 | 6.73 | 16.55 | 100% |

Prefill read 210 GB of experts — more than double v1's 99.7 GB — because 18 positions
route more widely than 5. In-layer diversity 45.2%, between v1's 77.2% (5 positions) and
v3's 37.0% (12 identical positions). Mean adjacent-step reuse 48.0%.

## The finding: the last two MLA layers can amplify enormously

The residual stream ends this prefill at **L2 20494**. Every other run and every other
step ends between 67 and 271.

Measuring the ratio of layer 91's attention output to layer 90's residual, across every
run and step captured so far:

| run | step | L90 out | L91 attn out | ratio |
|---|---:|---:|---:|---:|
| v1 factual | 0 | 109.63 | 15.61 | 0.1× |
| v2 factual | 0–3 | 39–110 | 6.7–15.6 | 0.1–0.2× |
| v3 repetitive | 0–3 | 39–251 | 6.3–87.6 | 0.2–0.3× |
| **v4 code** | **0** | **272.35** | **5207.62** | **19.1×** |
| v4 code | 1–3 | 45–54 | 10.4–11.3 | 0.2× |

Everything sits at 0.1–0.3× except this one prefill, at 19.1×. v4's own decode steps on
the same content are normal.

### It is not a uniform scale-up, and the input is not unusual

| | v1 prefill | v4 prefill |
|---|---:|---:|
| L91 norm.pre_attn — the *input* to the layer | L2 67.7, max 4.29 | L2 129.9, max **4.37** |
| L91 attn.out — the *output* | L2 15.6, max **1.34** | L2 5207.6, max **3457.84** |
| L92 layer.out | L2 171.0, max 33.89 | L2 20494.0, max **7177.29** |

The input to layer 91's attention is ordinary. The output has a maximum element **2585×
larger** than the same layer on the factual prompt, while the mean stays at ~0 (−0.02).
A handful of extreme elements, not a shifted distribution. The amplification happens
inside MLA at layer 91, and layer 92 — the only other consecutive MLA layer in the model
— compounds it.

### Why the model still works

RMSNorm is scale-invariant. Layer 92's `norm.pre_attn` reads L2 78.2 from a residual of
10169, and the final norm feeds logits with max 23.5 and L2 1398 — in line with every
other run. The magnitude never reaches the output. No non-finite value appeared anywhere
in 4968 tensors.

### Why it is still worth knowing

Activations here are fp32, so 7177 is harmless. This is the documented "massive
activations" behavior, caught directly rather than inferred. Anything that narrows
activation precision — not the int8 *weights* used here, which are unaffected — would
have to survive elements four orders of magnitude above the typical residual, and would
meet them only on some inputs and only during prefill.

## Named gap

The taps hash the whole `T × hidden` buffer, so this cannot say **which position** or
which channels carry the extreme values. The "one token dominates" reading is the
obvious hypothesis and is **not** established by this data. Confirming it needs a
per-position tap.

## Corrected after v6

This file first read the spike as specific to code, on the strength of v5 (7 positions,
no spike). **v6 — 225 positions of English prose — also spikes, at 7.30× with a maximum
element of 3319.9.** So it is not code-specific. Across the five prefills:

| run | T | L91 attn / L90 residual | L91 max element |
|---|---:|---:|---:|
| v1 factual | 5 | 0.14× | 1.34 |
| v5 french | 7 | 0.14× | 1.17 |
| v3 repetitive | 12 | 0.35× | 3.36 |
| v4 code | 18 | **19.12×** | **3457.84** |
| v6 long prose | 225 | **7.30×** | **3319.94** |

There is a threshold between 12 and 18 positions, and beyond it the maximum element sits
around 3300–3500 regardless of length. But **content and length vary together across
these five runs**, so neither is isolated. Separating them needs the same text truncated
to several lengths, which is the one experiment this set still owes.
