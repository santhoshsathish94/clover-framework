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

**It worked.** About **5.3 seconds per word**, steadily, run after run.

**What it told us:** this is real. A model this size does not need a datacenter. But almost
half the time went on reading expert weights off the disk, so **storage was the limit, not
cleverness.**

**Why that pointed somewhere:** a smaller model means less to read. So we tried making it
smaller.

---

## Step 2 — We halved the model, and it sped up exactly as expected

We converted the always-used part of the model from 16-bit numbers to 8-bit. Before running
it, we wrote down what we expected: about 1.1 seconds saved per word.

**We measured 1.18 seconds saved.** **5.3 down to 4.1 seconds per word**, and the memory
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
29 GB. We expected another half-second saved per word.

**We got a tenth of a second.** 4.1 down to **4.0 seconds per word**. And the quality cost
was far worse: it now matched the original for 3 words instead of 21.

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

Adding a **GPU-Server GEX45-1**: a graphics card with 24 GB of memory that handles 4-bit
numbers natively, next to a modest CPU and 64 GB of memory.

Two honest limits, stated before it arrives:

- **This model will not fit on it.** Its disks hold about 0.95 TB; the model needs 1.56 TB.
- **Even our shrunk 29 GB part does not quite fit** in 24 GB of card memory. Some will have to
  stay on the CPU.

That is fine, because the machine is not there to run this model. It is there to answer the
one question our CPU could not: **does the 4-bit step that failed on a processor succeed on a
graphics card?**

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
