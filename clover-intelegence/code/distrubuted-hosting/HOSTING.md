# Hosting Guide

How to build, lay out the data, run and verify Clover-K3's distributed stages. Every
command here was run on AX102 (Debian, 16 cores, 124 GB RAM) on 2026-10-03. For why the
system is split this way, see [system-architecture.md](system-architecture.md).

## What you need

| | |
|---|---|
| OS | Linux x86-64 |
| CPU | AVX2 (the expert kernel uses it; there is a scalar fallback but it is much slower) |
| Compiler | `gcc` with OpenMP (`-fopenmp`) |
| Node | only to regenerate `live-root.h` or `transformer-1.c`; not needed to build or run |
| Dataset | the prepared tree — 1,430.9 GB in total, but **no single stage needs all of it** |

Memory depends on how you deploy. One machine running everything needs ~124 GB and will
still be disk-bound. One machine per layer needs **20 GB each** and is not.

## 1. Lay out the per-stage data

This is the step that matters most and the easiest to skip. Each stage must see only its
own data; otherwise every pod maps the whole 1.4 TB tree and the split buys nothing.

```bash
cd code/distrubuted-hosting
bash link-stage-datasets.sh /path/to/dataset
```

It checks every required path exists, then builds one dataset directory per stage and
prints what it made:

```
PASS: per-stage datasets linked
  server         4.432 GB
  each pod      15.512 GB
```

| stage | gets |
|---|---|
| server | `tiktoken.model`, `vocabulary.bin`, `inputs/seed.bin`, `outputs/fruit.bin`, `trunk-0`, `trunk-0-qkv`, `leaves.json` |
| layer N | `root-N`, `trunk-N`, `operators/qkv-all/layer-N` |

The script uses symlinks, which is right when every stage is on one machine. For real
pods, copy those three paths into each node's volume instead — the directory layout the
stage expects is identical either way.

`leaves.json` belongs to the **server**, not to a separate normalisation stage. The
server owns the final normalisation now. Forgetting it is the one failure that only
shows up at the very end of a request.

## 2. Build

```bash
bash pipeline/build.sh
```

That compiles the coordinator plus 94 stage modules — server, normalization, and all 92
layers — into `bin/pipeline-stage.so` in each stage directory. Expect:

```
PASS: coordinator and 94 existing stage modules compiled without source changes
```

Everything builds with `-Wall -Wextra -Werror -pedantic` and no warnings. If you see a
warning, the build has already failed.

You do **not** need Node for this. `common/live-root.h` is generated and committed.
Regenerate it only if you change the reader:

```bash
cd ../standalone-hosting && node derive-root.mjs     # writes both copies
cd ../distrubuted-hosting/tansformers/transformer-1 && node derive-transformer.mjs
```

Both refuse to overwrite a file that has been edited by hand, and
`derive-transformer.mjs` additionally pins the SHA-256 of `server.c`. If you change
`server.c`, re-pin that hash or the generator will stop rather than emit a stale pod.

## 3. Run

```bash
bash pipeline/start-fleet.sh
```

This opens every stage once and keeps it resident. It raises the memlock limit, locks
the trunk in RAM, sets the OpenMP threads, and reads prompts
from a FIFO at `/tmp/clover.in`.

```
Fleet resident: server + 92 layers in 7.712s; request costs 752 MB, budget 24.0 GB, so 32 run at once
```

Startup is 6–15 s cold and under 2 s once the page cache is warm. **Do not restart it
between prompts** — the whole point is that the weights stay mapped.

### Asking it something

One line per request: the number of tokens you want, then the prompt's token ids.

```bash
echo "1 91019 25528" > /tmp/clover.in     # "Rain falls", 1 new token
cat /tmp/clover.out
```

```
READY
REQUEST 0 OK 90.086s [ on]
END
```

Send several lines at once and they all run concurrently. There is no slot count: a
request allocates its own state on arrival and runs as long as the memory budget has
room for it.

```bash
printf '2 91019 25528\n2 29349 147043 623\n2 1008 10484 318 15383 387\n' > /tmp/clover.in
```

```
REQUEST 0 OK 171.116s [ on the]
REQUEST 1 OK 194.108s [ 100]
REQUEST 2 OK 253.965s [ Paris.",
]
END
```

Measured on AX102 by stepping one resident fleet from 1 to 32 concurrent requests,
never restarting it, same prompt each time, trunk locked, `OMP_NUM_THREADS=4`:

| concurrent | wall | per request | throughput | CPU of 3200% | disk read |
|---|---|---|---|---|---|
| 1 | 123.9 s | 123.9 s | 0.0081 req/s | — | — |
| 2 | 130.8 s | 129.5 s | 0.0153 | — | 68.3 GB |
| 4 | 138.2 s | 137.9 s | 0.0289 | 716% | 60.8 GB |
| 8 | 152.3 s | 152.1 s | 0.0525 | 1,541% | 62.9 GB |
| **16** | 186.4 s | 185.6 s | **0.0858** ← peak | 2,221% | 68.0 GB |
| 32 | 416.9 s | 291 s | 0.0767 | 2,262% | **134.5 GB** |

**Throughput peaks near sixteen and drops 11% by thirty-two.** Look at the disk column
for the reason: flat to sixteen, then doubling, while CPU barely moves. Thirty requests
holding ~875 MB each evict the page cache that was holding experts, so the machine
starts re-reading them. Running more at once costs you cache you were already using.

Latency degrades gently up to that knee — 123.9 s at one request, 185.6 s at sixteen,
a 50% rise for 16x the load.

Two caveats before planning capacity from this. Every request here used the *same*
prompt, so they shared experts and page cache perfectly; eight *distinct* prompts gave
60% efficiency where this shows 81%, and real traffic sits between. And these are
2-token generations, so most of the work is prefill.

Two things to know about the protocol. A request that arrives alone runs immediately; it
does not wait to fill a batch. And `/tmp/clover.out` is held open by the service, so read
it, don't truncate it — truncating leaves the file looking empty while the service keeps
writing at its old offset.

### Knobs

| variable | effect |
|---|---|
| `CLOVER_LOCK_TRUNK=0` | do not lock the trunk resident; use if the box is short on RAM |
| `OMP_NUM_THREADS` | threads per request. 4 is the measured knee; 16 buys only 4% more, and binding to cores is 2% *worse* |
| `CLOVER_MEMORY_BUDGET_MB` | cap on memory for in-flight requests; concurrency follows from it |

**Set `CLOVER_MEMORY_BUDGET_MB` deliberately.** The default is half of `MemAvailable`,
which on this box derives ~30 concurrent — past the throughput peak, because it spends
memory the expert cache needed. It is a safety bound that prevents the box dying, not a
performance setting. About sixteen requests' worth (~14 GB here) is where this hardware
actually runs fastest. A budget that adapts to observed throughput is not built.

Locking the trunk pins about 72 GB when all 92 layers run on one machine. That is fine
on a 124 GB box but leaves less page cache for experts, and we measured 3.2 GB of swap
under four concurrent requests. On a 20 GB-per-layer deployment the question does not
arise. Set `CLOVER_LOCK_TRUNK=0` if you see swapping.

## 4. Verify it is computing the right thing

Each of these compares against a pinned original, not against itself.

```bash
# layer 1 against its original fixtures — 10 positions, france and japan
cd tansformers/transformer-1
gcc -std=c11 -O2 -ffp-contract=off -fno-fast-math -fopenmp -I../../common -I. \
    test-reference.c -lm -o bin/test-reference
bash test-reference.sh /path/to/dataset

# server layer 0 against its original
cd ../../server
bash test-reference.sh "$D/operators/trunk-0-qkv/qkv.bin" "$D/trunk-0" \
                       "$D/eqidx.bin" /path/to/original/trunk.bin
```

Expect `PASS: position N, layer-1 residual, 16 routes and S0 exact` for each position,
and `PASS: original layer-0 residual and S0 exact at 5 positions`.

The end-to-end check is the France prompt, which must return token 17374:

```bash
echo "1 1008 10484 318 15383 387" > /tmp/clover.in
# REQUEST 0 OK ... [ Paris]
```

Six tokens from that prompt should be `17374, 20829, 10, 427, 414, 1008`, which is
exactly what the standalone engine produces.

## 5. Moving to one machine per layer

Nothing above assumes a single host except the symlinks and the single process. Per node:

| node | RAM | holds |
|---|---|---|
| server | 8 GB | 4.432 GB: tokenizer, embeddings, head, trunk-0, leaves |
| layer 1–92 | **20 GB** | 15.512 GB for that layer, plus 7.41 MB per in-flight request |

Copy the two dataset paths onto each node, build the stage there, and the layer code
is unchanged — it already opens only its own data and maps rather than copies it.

**What is missing before this can run as pods:** there is no network transport. All 93
stages are loaded into one process with `dlopen`; a search of every `.c` and `.h` for
`socket`, `bind`, `listen`, `accept`, `AF_INET` or `grpc` returns nothing. The stages'
`--stream` mode reads ASCII floats from stdin, which works over a pipe but is not a
service. Hosting on Kubernetes needs a small wrapper per stage: listen on a port, use
length-prefixed binary frames, carry a session id so a pod can select the right
sequence, and expose a readiness probe. That is one shared file, not 92 changes.

## Troubleshooting

| symptom | cause |
|---|---|
| `server tail failed` at the end of a request | `leaves.json` missing from the server's dataset |
| `stage load: ... cannot open shared object file` | `build.sh` was not run, or was run before the datasets were linked |
| request appears to hang forever | reading a truncated `/tmp/clover.out`; the service holds it open |
| `trunk not locked: Cannot allocate memory` | memlock limit too low — `start-fleet.sh` raises it; set `CLOVER_LOCK_TRUNK=0` to skip |
| build stops on a warning | intended; `-Werror` is on everywhere |
| generator refuses to run | `server.c` changed and the pinned SHA-256 in `derive-transformer.mjs` no longer matches |
