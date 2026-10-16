#!/usr/bin/env python3
"""What acceptance rate could a cheap drafter reach on this model's own output?

Speculation is the only way to put more than one output token on a single pass,
because token N+1's input is token N. But it is far less favourable here than
on a GPU: the marginal cost of an extra position in a pass is 0.8 s, measured
(3.6 s at one position, 9.2 s at eight), so a verify pass of k+1 positions
costs 3.6 + 0.8k. Below roughly 50% acceptance the arithmetic is a loss.

So the acceptance rate decides it, and it can be measured without a draft model
and without touching the engine: take the model's own generated tokens and ask
how often a trivial predictor, built only from tokens it has already seen,
calls the next one correctly. Causal by construction - the table is updated
after each prediction, never before - so there is no leakage.

Three drafters, cheapest first:
  unigram   most frequent token so far
  bigram    what followed this token last time
  trigram   what followed this pair last time

usage: draft-acceptance.py [file ...]
"""
import collections
import glob
import json
import os
import sys


def tokens_from(path):
    out = []
    for line in open(path):
        if line.startswith("TOKEN_JSON "):
            try:
                out.append(json.loads(line[11:])["token"])
            except Exception:
                pass
        elif line.startswith("TOKEN "):
            try:
                out.append(json.loads(line[6:])["emitted"])
            except Exception:
                pass
    return out


def measure(seq):
    uni = collections.Counter()
    bi = {}
    tri = {}
    hit = {"unigram": 0, "bigram": 0, "trigram": 0}
    seen = {"unigram": 0, "bigram": 0, "trigram": 0}
    cover = {"bigram": 0, "trigram": 0}

    for i, tok in enumerate(seq):
        # predict position i from what is already known, then learn it
        if uni:
            seen["unigram"] += 1
            if uni.most_common(1)[0][0] == tok:
                hit["unigram"] += 1
        if i >= 1:
            key = seq[i - 1]
            seen["bigram"] += 1
            if key in bi:
                cover["bigram"] += 1
                if bi[key] == tok:
                    hit["bigram"] += 1
        if i >= 2:
            key2 = (seq[i - 2], seq[i - 1])
            seen["trigram"] += 1
            if key2 in tri:
                cover["trigram"] += 1
                if tri[key2] == tok:
                    hit["trigram"] += 1

        uni[tok] += 1
        if i >= 1:
            bi[seq[i - 1]] = tok
        if i >= 2:
            tri[(seq[i - 2], seq[i - 1])] = tok
    return hit, seen, cover


def speedup(p, k):
    if p >= 1.0:
        acc = k
    else:
        acc = p * (1 - p ** k) / (1 - p)
    return (1 + acc) / (1 + 0.22 * k)


def main():
    paths = sys.argv[1:] or (["/tmp/run1000/out.txt"] + sorted(glob.glob("/tmp/sweep/route-*.jsonl")))
    seqs = []
    for p in paths:
        if os.path.exists(p):
            t = tokens_from(p)
            if len(t) > 20:
                seqs.append((p, t))
    if not seqs:
        print("no generated token sequences found")
        return 1

    total_hit = collections.Counter()
    total_seen = collections.Counter()
    total_cover = collections.Counter()
    print("%-34s %7s %9s %9s %9s" % ("sequence", "tokens", "unigram", "bigram", "trigram"))
    for path, seq in seqs:
        h, s, c = measure(seq)
        print("%-34s %7d %8.1f%% %8.1f%% %8.1f%%"
              % (os.path.basename(path)[:34], len(seq),
                 100.0 * h["unigram"] / max(1, s["unigram"]),
                 100.0 * h["bigram"] / max(1, s["bigram"]),
                 100.0 * h["trigram"] / max(1, s["trigram"])))
        for k in h:
            total_hit[k] += h[k]
            total_seen[k] += s[k]
        for k in c:
            total_cover[k] += c[k]

    print()
    print("pooled over %d sequences, %d tokens" % (len(seqs), sum(len(s) for _, s in seqs)))
    best = 0.0
    for name in ("unigram", "bigram", "trigram"):
        p = total_hit[name] / max(1, total_seen[name])
        best = max(best, p)
        extra = ""
        if name in total_cover:
            extra = "   (key known for %.1f%% of positions)" % (
                100.0 * total_cover[name] / max(1, total_seen[name]))
        print("  %-9s acceptance %6.2f%%%s" % (name, 100.0 * p, extra))

    print()
    print("speedup at the measured 0.8 s marginal cost per extra position")
    print("  %-6s %9s %9s %9s" % ("k", "p=best", "p=0.7", "p=0.9"))
    for k in (1, 2, 3, 4, 6, 8):
        print("  %-6d %8.2fx %8.2fx %8.2fx"
              % (k, speedup(best, k), speedup(0.7, k), speedup(0.9, k)))
    print()
    print("Below about 1.0x speculation costs more than it saves here, because an")
    print("extra position is 0.8 s rather than nearly free as it would be on a GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
