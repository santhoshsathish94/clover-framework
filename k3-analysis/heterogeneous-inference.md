# Running a very large model on the machines you can actually get

**Where this stands: a direction, based on three experiments we ran ourselves. Nothing here
has been tried on a GPU yet, and none of it is finished. The next step may get us closer.
We do not know that it will.**

This is exploratory. It is not Clover doctrine.

---

## Before anything else

Clover is not against any system. Every approach mentioned here is a sensible answer to the
situation its builders were in, and ours is a different situation, not a better one. What
they all have in common is people improving their own system.

So this is not a comparison. It is what happened to us, in the order it happened, and what
each step told us to do next.

---

## The question we started with

How much of what we assume about running AI is really about the model, and how much is just
about how we build everything around it?

We could have argued about it. Instead we ran something.

---

## Step 1 — We ran a 2.78-trillion-parameter model on one CPU

We used Fareed Khan's [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), which
streams the model off disk instead of holding it in memory. We rented a machine with a good
CPU, 124 GB of memory and fast NVMe drives, downloaded the 1.56 TB model, and generated text.

**It worked.** About **5.3 seconds per token**, steadily, run after run.

**What it told us:** this is real. A model this size does not need a datacenter. But almost
half the time went on reading expert weights off the disk, so **storage was the limit, not
cleverness.**

**Why that pointed somewhere:** a smaller model means less to read. So we tried making it
smaller.

---

## Step 2 — We halved the model, and it sped up exactly as expected

We converted the always-used part of the model from 16-bit numbers to 8-bit. Before running
it, we wrote down what we expected: about 1.1 seconds saved per word.

**We measured 1.18 seconds saved.** **5.3 down to 4.1 seconds per token**, and the memory
needed fell from 119 GB to 65 GB.

The answers changed slightly, because 8-bit is genuinely a different model. It produced
identical output for the first 21 words, then went its own way while still making sense.

**What it told us:** our understanding of the machine was correct. Memory speed was the
limit, and halving the data halved that cost.

**Why that pointed somewhere:** if halving worked once, it should work again. That
expectation is the entire reason for step 3.

---

## Step 3 — We halved it again, and the speed-up stopped

We converted the same part down to 4-bit. The file shrank exactly as predicted, to under
29 GB. We expected another half-second saved per token.

**We got a tenth of a second.** 4.1 down to **4.0 seconds per token**. And the quality cost
was far worse: it now matched the original for 3 tokens instead of 21.

**What it told us, and this is the one that matters:** 4-bit numbers have to be unpacked
before they can be used. On this CPU, that unpacking cost about as much as the reading it
saved. **We had stopped being limited by memory speed and started being limited by processor
work.**

We had already written down "shrinking further is a dead end." That was wrong. It is a dead
end *on this processor*, and says nothing about any other kind of machine.

**Why that points somewhere:** a graphics card is the opposite shape of machine. It has huge
amounts of exactly the work that slowed us down, and its memory is the scarce part. The step
that failed on a CPU is the step that should pay on a GPU. We cannot find that out here.

---

## What the three steps add up to

The model is really two problems wearing one name:

- The part used **for every single word** is now about 29 GB. It wants to sit on the fastest
  device available.
- The **expert** part is 1.45 TB. It will never fit anywhere, on any machine, and must be
  streamed from disk no matter what.

Those two want different hardware. That is not a preference; it is what the measurements
showed.

---

## Others solved this shape of problem already

**Games did it decades ago.** A game world is far bigger than the console it runs on, and
nobody waited for bigger consoles. They load scenery as you walk toward it, draw distant
things roughly, compress for fast unpacking rather than smallest size, and give every feature
a fixed time budget it must earn. That is their answer to their problem, and it is worth
learning from.

**[`llama.cpp`](https://github.com/ggml-org/llama.cpp) does the mixed-device version today.**
We read its source rather than articles about it. The rule is simple: **what is used for every
word goes on the fast device; what is used rarely goes on the slow one.** It decides this
piece by piece, as the model loads. The small set of experts consulted every time stays on the
fast device; only the rarely-picked ones are pushed out.

That is the same dividing line our three steps pointed at, reached separately. Mild evidence
the line is real.

---

## What we are doing next

Adding a **GPU-Server GEX131-1** with 96 GB of GPU VRAM, 256 GB of CPU memory, and
two 960 GB NVMe drives. The full 1.56 TB checkpoint fits across the two drives when the
model weights are **split across them**.

That changes the experiment.

The first storage setup mirrored the disks. The next setup will shard the model across the
two disks so that the storage paths can be measured independently. We do not yet know whether
the right layout is simple file placement, parallel reads, or another scheduling strategy.

The 96 GB GPU also changes the token-processing question. The measured INT8 trunk is about
54.47 GB and the MXFP4 trunk about 28.94 GB, so either can fit entirely on the GPU. The BF16
trunk is 108.81 GB, so it cannot. (That is the packed trunk itself. The bf16 run's peak
memory was 118.93 GB, but that figure also covers embeddings, recurrent state, buffers and
the KV cache, which are not what would be placed on a card.) This gives us three meaningful
placement cases rather than a simple fit/no-fit boundary.

The working hypothesis is a hierarchy:

```text
                 full model
                     ↓
             two-disk backing store
                     ↓
              CPU memory working set
                  ↙       ↘
              trunk      experts
                  ↘       ↙
               GPU working set
                     ↓
               token generation
```

The GPU may hold the most frequently used trunk components and/or selected experts while the
larger model remains backed by CPU memory and sharded NVMe. Prefetching may allow storage reads
for the next token to overlap with computation for the current token.

This is not an architecture claim. It is the next experiment.

The question is no longer simply whether a 4-bit trunk is faster on a GPU.

> **Can we discover a useful division of the model across GPU, CPU memory, and sharded storage
> that makes each token faster without requiring the whole model to fit on one device?**

---

## What we do not know

- **Nothing here has been measured on a GPU.** This is a direction, not a result.
- **Lessons from one machine do not transfer.** Step 3 is the proof of that.
- **Quality was checked loosely** — one prompt, 64 words, compared against the original.
  Enough to say the output changed, not enough to say it got worse at its job.
- **We have not reproduced anyone else's published speed figures,** so none are quoted here.

---

## Whose work is whose

The engine is Fareed Khan's
[`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Apache-2.0) and stays his.
The mixed-device approach described above is
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) and stays its authors'. The three
experiments and their numbers are ours, run on rented hardware.

The changes we made to the engine to run them are local for now. The intention is to offer
them back to the upstream project rather than keep them here.

Fuller figures, including the ones we got wrong:
[CONTEXT-kimi-k3-benchmark.md](CONTEXT-kimi-k3-benchmark.md).
