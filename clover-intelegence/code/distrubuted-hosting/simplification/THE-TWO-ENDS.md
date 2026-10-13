# The two ends of the chain

The stage walk covered pods 1 to 92 — stages 121 to 11160 of 11,280. The creator pointed
out that this is not the whole chain:

> because there are stages before 92 layers total 96 layers from input to token vectors
> then vectors in layer 0 then layer 1 to 92 then normalization then vector token to
> output so look at them fully observer don't assume look at one step at a time

Both ends had never been instrumented. This is what they do, read off the running
engine rather than reasoned about.

---

## The chain, read off the source

`pipeline.c` lines 311–337, `numeric-table.h`, `server.c`, `transformer-93.c`:

| step | where | what happens |
|---|---|---|
| token id → vector | `server_input`, `numeric-table.h` | row `id` of a 163,840 × 7168 bf16 table, widened to f32. No scaling, no position term. This vector **is** the initial residual. |
| layer 0 | `server.c`, owner 0 | flat sequence, not a stage machine. Normalize → QKV → decay → attention → 2-source fold → dense. No routed experts. |
| layers 1–92 | `transformer-N.c`, owners 1–92 | the stage machine, stages 121–11160 |
| normalisation | `transformer-93.c` stages 1–9, owner 93 | 9-source fold over the 8 block snapshots and the residual, then RMS |
| vector → token | `server_output_project`, `numeric-table.h` | dot product against all 163,840 rows, argmax, ties to the lowest id. Greedy, no softmax, no sampling. |
| token → text | `id_to_word` | tokenizer lookup |

The tail runs on `tokens[count-1]` only — the last position. Normalisation and the head
are one call per generated token, not one per position.

---

## Instrumentation and its gate

Two patches, both behind `clover_vectors_active`, both idempotent:

- `port-server-trace.py` → owner 0, locals 5201 input, 5202 normalized, 5203 residual
  after attention, 5204 aggregate, 5205 dense output, 5206 outgoing residual, 5207 the
  two fold scores.
- `port-tail-trace.py` → owner 93, locals 5301 residual, 5302 nine scores, 5303 nine
  weights, 5304 folded, 5305 after RMS, 5310–5317 the eight incoming snapshots.

`pipeline/build.sh` needed `-Wno-unused-function` on the server line; the tail line
already had it. `common/trace.h` carries six statics and the server calls two of them.

**Gate, tracing off: twelve prompts through the production fleet, 12/12 unchanged.**
Paris, Tokyo, Berlin, " was", " the", " Celsius", ",", "?\n\n", " f", ".\",\n", " and",
" of". The server's own reference gate also passes: *layer-0 residual and S0 exact at 5
positions*, on both its prompts.

(`fidelity.sh` reports `question` and `french` as MISSING. That is its `grep -o
'\[.*\]'` failing on an answer containing a newline, not a failure. The raw lines read
`[?$|$|]` and `[.",$|]` under `cat -A`.)

---

## Layer 0, genesis, all 16 positions

| pos | embedding | normalized | attn residual | aggregate | dense out | out | weight on embedding |
|---|---|---|---|---|---|---|---|
| 0 | 2.0487 | 13.7306 | 0.3618 | 0.3562 | 0.8000 | 0.9303 | 0.0327 |
| 1 | 1.6927 | 13.4552 | 0.5658 | 0.5219 | 0.7455 | 1.0768 | 0.0816 |
| 2 | 1.9152 | 14.0267 | 0.5052 | 0.4965 | 0.5936 | 0.8833 | 0.0324 |
| 3 | 1.9811 | 13.9034 | 0.5940 | 0.5830 | 0.5368 | 0.8722 | 0.0283 |
| 4 | 1.9562 | 13.7977 | 0.5385 | 0.5325 | 0.5155 | 0.7990 | 0.0168 |
| 5 | 1.6927 | 13.4552 | 0.6792 | 0.6358 | 0.5141 | 0.9621 | 0.0610 |
| 6 | 1.6618 | 13.9965 | 0.5703 | 0.5642 | 0.4886 | 0.8445 | 0.0141 |
| 7 | 1.5434 | 13.4066 | 0.7149 | 0.6841 | 0.4261 | 0.9279 | 0.0418 |
| 8 | 1.6927 | 13.4552 | 0.6574 | 0.6305 | 0.4405 | 0.9108 | 0.0346 |
| 9 | 1.8655 | 14.0207 | 0.5870 | 0.5821 | 0.5918 | 0.9378 | 0.0105 |
| 10 | 1.5434 | 13.4066 | 0.7198 | 0.6905 | 0.4032 | 0.9076 | 0.0390 |
| 11 | 1.6927 | 13.4552 | 0.6597 | 0.6349 | 0.4437 | 0.9190 | 0.0314 |
| 12 | 1.8655 | 14.0207 | 0.6298 | 0.6250 | 0.7813 | 1.1275 | 0.0094 |
| 13 | 2.0064 | 13.5331 | 0.6135 | 0.5944 | 0.4163 | 0.8026 | 0.0317 |
| 14 | 2.0309 | 13.9109 | 0.5816 | 0.5686 | 0.4750 | 0.8394 | 0.0274 |
| 15 | 2.0429 | 13.9097 | 0.6362 | 0.6285 | 0.5697 | 0.9523 | 0.0144 |

### Four things the table shows

**The normalize step is nearly constant in magnitude.** 13.41 to 14.03 across sixteen
positions, a 4.5% spread, from inputs that span 1.54 to 2.05 (a 32% spread). That is
what an RMS normalisation with a fixed gain does, and it is visible rather than assumed.

**Layer 0's fold almost discards the raw embedding.** The two-source fold puts between
0.94% and 8.2% on the incoming embedding and the rest on the attention residual. Median
about 3%. The aggregate norm is below the attention-residual norm at every position, so
what the embedding contributes is partly cancellation, not reinforcement.

**The fold's first score is a function of the token alone.** Genesis repeats token 276
at positions 1, 5, 8 and 11. All four have embedding norm 1.6927, normalized 13.4552,
and score[0] exactly 0.1594. Token 316 at positions 7 and 10: 1.5434, 13.4066, 0.1843.
Token 13618 at 9 and 12: 1.8655, 14.0207, −0.4274. The *second* score moves with history
(276 scores 2.5804, 2.8926, 3.4887, 3.5888 at those four positions) and so does the
attention residual. The source being scored at index 0 is the snapshot, and at layer 0
the snapshot is the input, so this is what the arithmetic has to do — but it is now
observed, not inferred.

**The layer shrinks the vector.** In at norm 1.54–2.05, out at 0.80–1.13.

---

## The tail, genesis, the position the head reads

| source | norm | fold weight |
|---|---|---|
| snapshot 0 | **2.0429** | 0.0022 |
| snapshot 1 | 128.8138 | 0.0043 |
| snapshot 2 | 63.5395 | 0.0030 |
| snapshot 3 | 36.1914 | 0.0017 |
| snapshot 4 | 68.1819 | 0.0021 |
| snapshot 5 | 141.5318 | 0.0062 |
| snapshot 6 | 51.5442 | 0.0378 |
| snapshot 7 | 71.3653 | 0.0978 |
| **residual** | 80.9239 | **0.8448** |
| folded | 70.1919 | |
| after RMS | 48.6706 | |

### Snapshot slot 0 is the raw token embedding, bit for bit

Norm 2.0429 is the embedding of position 15 from the layer-0 table above. Compared
directly:

```
tail slot 0 vs server input, position 15:  cos 1.000000000  max abs difference 0  bit identical: yes
tail slot 0 vs server output, position 15: cos 0.011977     max abs difference 0.191  no
```

The server publishes its *input* as its snapshot (`server.c` line 542 copies
`sequence->snapshot`, which line 513 set from `input`). So the raw table row for the
last token travels the entire depth of the model untouched, arrives at the final fold,
and is given 0.22% of the weight. It is also near-orthogonal to what layer 0 produced
from it — cosine 0.012.

### The last two blocks carry the snapshot side

Slots 6 and 7 take 13.6 of the 15.5 percentage points the snapshots get between them.
Slots 0 through 5 share 1.9 points. There are seven block-write pods (12, 24, 36, 48,
60, 72, 84) filling slots 1–7, and the server filling slot 0.

---

## Vector to token

Each source projected through the real head (`head-walk`, the production
`outputs/fruit.bin`). The answer was " and", token 316.

| what the head was given | norm | token | top | second | margin | prob |
|---|---|---|---|---|---|---|
| after RMS — *what the engine uses* | 48.67 | **316** | 20.122 | 18.704 | 1.418 | 0.742 |
| folded, before RMS | 70.19 | **316** | 26.655 | 24.041 | 2.614 | 0.913 |
| residual alone, no fold | 80.92 | **316** | 30.066 | 27.344 | 2.722 | 0.925 |
| snapshot 0 (raw embedding) | 2.04 | 97759 | 0.307 | 0.288 | 0.019 | 0.000008 |
| snapshot 1 | 128.81 | 159359 | 18.470 | 16.752 | 1.718 | 0.454 |
| snapshot 2 | 63.54 | 25100 | 8.377 | 8.131 | 0.246 | 0.005 |
| snapshot 3 | 36.19 | 43570 | 5.399 | 4.680 | 0.719 | 0.0005 |
| snapshot 4 | 68.18 | 114172 | 9.452 | 9.079 | 0.372 | 0.005 |
| snapshot 5 | 141.53 | 48606 | 25.367 | 24.212 | 1.155 | 0.552 |
| snapshot 6 | 51.54 | 71 | 5.305 | 5.195 | 0.110 | 0.001 |
| snapshot 7 | 71.37 | 1521 | 12.681 | 12.158 | 0.523 | 0.396 |

**No snapshot votes for the answer.** Each of the eight picks something unrelated, most
of them with almost no confidence. They are not eight opinions being averaged; they are
corrections to a vector that already carries the decision.

On this prompt the residual, the folded vector and the vector after RMS all pick 316, so
the obvious reading is that the answer is settled before the tail runs. **That reading
is wrong, and twelve prompts show it.**

### What this does not establish

Margins and probabilities across these rows are **not** comparable as confidence. The
vectors have different norms, a dot product scales with norm, and the softmax that turns
scores into the "prob" column is not scale-invariant. The comparison that *is* sound is
the one on the token column.

Feeding a snapshot to the head is also not the counterfactual "what if the model stopped
at that block" — `normalization_apply_rms` is a per-coordinate gain, not a uniform
scale, so a source that has not been through it is not on the head's footing. The rows
are there to show the snapshots are not carrying the answer, which the near-zero
probabilities make clear whatever the scale.

---

## Twelve prompts, and the correction

The tail was captured for all twelve prompts and each one's residual, folded vector and
post-RMS vector were put through the head. Ids decoded with `decode-ids`, which calls
the engine's own `id_to_word` — tiktoken is not installed on this host and a guessed
mapping is not evidence.

| prompt | residual alone | folded | after RMS = **the engine's answer** |
|---|---|---|---|
| france | " Paris" | " Paris" | " Paris" |
| french | "." | "." | **".\",\n"** |
| genesis | " and" | " and" | " and" |
| germany | " the" | " the" | **" Berlin"** |
| hello | "," | "," | "," |
| japan | " Tokyo" | " Tokyo" | " Tokyo" |
| n2 | " of" | " of" | " of" |
| nonsense | "\n" | "\n" | **" f"** |
| once | " was" | " was" | " was" |
| question | "?" | "?" | **"?\n\n"** |
| thethe | " the" | " the" | " the" |
| water | "," | "," | **" Celsius"** |

The after-RMS column reproduces the fleet's answer on all twelve, which is a check on
the instrumentation as much as a result.

**The nine-source fold never changed the token. 12 of 12.** It changes confidence, and
sometimes a lot — `n2` goes 0.554 → 0.158 — but the argmax is the same every time. The
15.5% the snapshots get does not decide anything on this set.

**`normalization_apply_rms` changed the token in 5 of 12.** It is the last arithmetic
before the head and it is not cosmetic. Without it:

- germany would answer **" the"** instead of **" Berlin"**
- water would answer **","** instead of **" Celsius"**
- nonsense would answer **"\n"** instead of **" f"**
- french would answer **"."** instead of **".\",\n"**
- question would answer **"?"** instead of **"?\n\n"**

In every one of the five the pre-RMS vector picks something generic — a space-the, a
comma, a newline, a full stop — and the per-coordinate gain moves it to the specific
continuation. Two of the five (french, question) keep the same leading character and
gain a longer token; three change the word outright.

### What it corrects

This report said "the answer is already decided before the tail runs" on the strength of
genesis. Genesis is one of the seven where it happens to hold. Eleven more prompts cost
one pass of the fleet each and turned a clean sentence into a false one. The sound
statement is narrower: **the snapshot fold does not decide the token; the final RMS gain
sometimes does.**

### What it still does not establish

Twelve prompts, one generated token each, greedy top-1. "Changed the token in 5 of 12"
is a count on this set, not a rate. Nothing here says the fold is removable — a token
that is unchanged can sit on a margin that is not, and `n2` losing two thirds of its
probability is exactly that. No timing was measured.

### The fold weights are stable across prompts

| prompt | snap0 | snap1 | snap2 | snap3 | snap4 | snap5 | snap6 | snap7 | residual |
|---|---|---|---|---|---|---|---|---|---|
| france | 0.0026 | 0.0028 | 0.0047 | 0.0016 | 0.0052 | 0.0121 | 0.0205 | 0.1410 | 0.8097 |
| french | 0.0033 | 0.0058 | 0.0030 | 0.0010 | 0.0093 | 0.0048 | 0.0067 | 0.0826 | 0.8835 |
| genesis | 0.0022 | 0.0043 | 0.0030 | 0.0017 | 0.0021 | 0.0062 | 0.0378 | 0.0978 | 0.8448 |
| germany | 0.0027 | 0.0033 | 0.0052 | 0.0026 | 0.0058 | 0.0112 | 0.0235 | 0.1294 | 0.8163 |
| hello | 0.0211 | 0.0245 | 0.0022 | 0.0023 | 0.0013 | 0.0155 | 0.0145 | 0.1097 | 0.8089 |
| japan | 0.0027 | 0.0031 | 0.0051 | 0.0022 | 0.0060 | 0.0093 | 0.0187 | 0.1198 | 0.8331 |
| n2 | 0.0140 | 0.0162 | 0.0038 | 0.0196 | 0.0114 | 0.0190 | 0.0863 | 0.1219 | 0.7079 |
| nonsense | 0.0681 | 0.0031 | 0.0024 | 0.0008 | 0.0021 | 0.0031 | 0.0119 | 0.0976 | 0.8108 |
| once | 0.0064 | 0.0091 | 0.0038 | 0.0013 | 0.0049 | 0.0070 | 0.0401 | 0.1068 | 0.8206 |
| question | 0.0425 | 0.0064 | 0.0036 | 0.0035 | 0.0077 | 0.0133 | 0.0350 | 0.1176 | 0.7704 |
| thethe | 0.0028 | 0.0048 | 0.0044 | 0.0019 | 0.0030 | 0.0042 | 0.0080 | 0.0806 | 0.8903 |
| water | 0.0080 | 0.0052 | 0.0038 | 0.0015 | 0.0068 | 0.0062 | 0.0195 | 0.1097 | 0.8394 |

The residual takes 0.708 to 0.890 on every prompt and slot 7 takes 0.081 to 0.141 on
every prompt. Slots 1 to 6 are under 0.09 everywhere except `n2`'s slot 6. The shape of
the final fold barely responds to what was asked.

Slot 0 — the raw embedding — is the exception that is worth naming: usually 0.002 to
0.008, but 0.068 on `nonsense`, 0.043 on `question`, 0.021 on `hello` and 0.014 on `n2`.
The four highest are the two shortest prompts and the two least ordinary ones. Why is
not established here; the observation is that something about short or unusual input
raises the weight on the untouched input vector by an order of magnitude.

---

---

## Where the instrumentation is

| file | what |
|---|---|
| `port-server-trace.py` | patch owner 0 |
| `port-tail-trace.py` | patch owner 93 |
| `setup-tail-trace.sh` | backup, patch, confirm additions only, build the tail alone |
| `ends-table.py` | a dump laid out as positions × locals, with short records expanded through softmax |
| `cross-compare.py` | one record in one dump against a different local in another |
| `pull-f32.py` | records out of a dump into the flat f32 `head-walk` reads |

`server/decode-ids.c` turns a token id back into text through the engine's own
`id_to_word`. It lives in `server/` rather than here for the same reason `head-walk.c`
does: it includes `ends.h`.

Backups on the host: `/tmp/servertrace-before.tar`, `/tmp/tailtrace-before.tar`,
`/tmp/build.sh.bak`.

The tokenizer config is **not** beside the model and vocabulary:
`server/bin/configs/tokenizer_config.json`, while `tiktoken.model` and
`vocabulary.bin` are in `server/bin/dataset/`. `pipeline.c` line 218 is the only place
that says so.
