# c — ceiling and channel

**Probes:** the two questions left open by the length sweep. Is the ~3400 magnitude a
ceiling, and is the spike always in the same channel? Three cheap code variants, chosen so
that one keeps the structure and changes the names, one changes the language, and one
removes the line breaks while keeping the meaning.

```
c1-factorial    "def factorial(k):\n    if k <= 1:\n        return k\n    return"
c2-javascript   "function fib(n) {\n    if (n <= 1) {\n        return n;\n    }\n    return"
c3-no-newline   "def fibonacci(n): if n <= 1: return n; return"
gen 1, K3_TRACE_ROWS=1, otherwise identical settings to every earlier run
```

## Results

| run | T | ratio | L91 max | position | channel | above 100× median |
|---|---:|---:|---:|---:|---:|---:|
| v4 code (earlier) | 18 | 19.12× | 3457.84 | 15 | **4590** | 1 |
| **c1 factorial** | 17 | **19.31×** | **3293.84** | 14 | **4590** | 1 |
| c2 javascript | 22 | 0.34× | 4.52 | 17 | 3680 | 0 |
| c3 no newline | 15 | 0.35× | 3.21 | 9 | 537 | 0 |
| v6 prose (earlier) | 225 | 7.30× | 3319.94 | 216 | not yet known | 1 |

## The magnitude looks like a ceiling

Three spiking cases now:

```
3457.84   v4 code,  18 positions
3293.84   c1 code,  17 positions
3319.94   v6 prose, 225 positions
```

A **5% spread** across Python code and English prose, across a 13× difference in prompt
length, landing on different tokens. The two code ratios are also nearly equal — 19.12×
and 19.31×.

That is three data points, not a proof, but it is much harder to explain as an
input-driven value than as a level the mechanism saturates at.

## For code, the standalone newline is the trigger

Token inventories explain the two negatives exactly:

| run | contains token 198 `'\n'` | spikes |
|---|---|---|
| v4 | yes, at position 15 | yes, at position 15 |
| c1 | yes, at position 14 | yes, at position 14 |
| c2 javascript | **no** — newlines merge with the following indentation into other tokens | no |
| c3 no newline | **no** — the line breaks were removed | no |

Four for four. Within code prompts, a bare newline token is present exactly when the
spike is, and the spike sits on it.

**But that is not the general rule.** v6 is English prose with no newline anywhere, and it
spikes on `' read'`. So a bare newline is *sufficient* here and clearly not *necessary*.
Two routes to the same place, or one rule that neither observation has yet named.

Worth noting against the earlier correction: the delimiter reading was rejected after v6,
and it was right to reject it as a general rule. It survives as a code-specific one.

## Channel

Channel **4590** in both spiking code runs, entered at layer 90 and held through 92, while
non-spiking positions peak elsewhere — 3680, 537, 6532. Whether the prose spike uses 4590
as well is the decisive test and is running.
