#!/usr/bin/env python3
"""Make the expert prefetch actually fetch, instead of only asking.

Stage 27 calls root_prefetch for each of the sixteen selected experts, and
root_prefetch calls posix_madvise(WILLNEED) and returns. The comment above it says the
point is that "the device never sees more than one request in flight", which is the
right diagnosis, but madvise only queues a hint. This repo already measured that on
this box: WILLNEED alone was 49.77 s against 42.21 s for driving the faults and waiting,
and 53.52 s for no prefetch at all. The advisory call was the worst of the three.

So this keeps the madvise and adds the part that was missing: touch one byte per page
across the OpenMP team and wait for it. The bytes are needed either way; the only
question is whether they are fetched concurrently now or demand-faulted one at a time
inside the arithmetic later.

CLOVER_PREFETCH_TOUCH=0 restores the old behaviour so the two can be run against each
other in the same build.

Nothing else changes: no new struct field, no new call site, no change to any of the 93
generated transformer sources.

usage: port-prefetch-touch.py PATH-TO-live-root.h
"""
import sys

BEFORE = """static void root_prefetch(const Root *root, unsigned expert)
{
    if (!root || !root->direct || expert >= 896) return;
    (void)posix_madvise((void *)(root->direct + (size_t)expert * ROOT_EXPERT_RAW), ROOT_EXPERT_RAW, POSIX_MADV_WILLNEED);
}
"""

AFTER = """/* CLOVER_PREFETCH_TOUCH */
static volatile unsigned char root_touch_sink;
static int root_touch_setting = -1;

static int root_touch_enabled(void)
{
    if (root_touch_setting < 0) {
        const char *value = getenv("CLOVER_PREFETCH_TOUCH");
        root_touch_setting = (value && value[0] == '0') ? 0 : 1;
    }
    return root_touch_setting;
}

static void root_prefetch(const Root *root, unsigned expert)
{
    if (!root || !root->direct || expert >= 896) return;
    unsigned char *base = (unsigned char *)(root->direct + (size_t)expert * ROOT_EXPERT_RAW);
    (void)posix_madvise(base, ROOT_EXPERT_RAW, POSIX_MADV_WILLNEED);
    if (!root_touch_enabled()) return;
    /* madvise only queues the hint, so the faults still arrive one at a time inside the
       arithmetic. Reading a byte per page across the team issues them together and
       waits, which is what measured faster here. */
    const long pages = (long)(ROOT_EXPERT_RAW / 4096);
    unsigned char seen = 0;
#pragma omp parallel for schedule(static) reduction(|:seen)
    for (long page = 0; page < pages; page++)
        seen = (unsigned char)(seen | base[(size_t)page * 4096]);
    root_touch_sink = seen;
}
"""


def main():
    path = sys.argv[1]
    with open(path, "r") as handle:
        text = handle.read()

    if "CLOVER_PREFETCH_TOUCH" in text:
        print("already patched")
        return 0
    if text.count(BEFORE) != 1:
        print("root_prefetch not found exactly once, found %d" % text.count(BEFORE))
        return 1

    with open(path, "w") as handle:
        handle.write(text.replace(BEFORE, AFTER))
    print("patched %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
