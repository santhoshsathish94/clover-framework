#!/usr/bin/env python3
# Token ids for new prompts, without inventing any.
#
# tiktoken.model is "base64 rank" per line. Greedy longest-match is not BPE, so the
# result is only trustworthy if it reproduces ids that are already known good. The
# script refuses to emit a prompt that does not survive a round trip through the
# vocabulary, and it checks itself against the two prompts whose ids came from the
# repository's own test-reference.sh before encoding anything new.
import base64, sys

VOCAB = "/opt/clover-k3/clover-intelegence/dataset/tiktoken.model"

tokens = {}
for line in open(VOCAB):
    parts = line.split()
    if len(parts) != 2:
        continue
    try:
        tokens[base64.b64decode(parts[0])] = int(parts[1])
    except Exception:
        continue
longest = max(len(t) for t in tokens)
inverse = {v: k for k, v in tokens.items()}

def encode(text):
    raw = text.encode("utf-8")
    out, at = [], 0
    while at < len(raw):
        for size in range(min(longest, len(raw) - at), 0, -1):
            piece = raw[at:at + size]
            if piece in tokens:
                out.append(tokens[piece])
                at += size
                break
        else:
            return None
    return out

def decode(ids):
    return b"".join(inverse[i] for i in ids).decode("utf-8", "replace")

KNOWN = {
    "The capital of France is": [1008, 10484, 318, 15383, 387],
    "The capital of Japan is": [1008, 10484, 318, 10417, 387],
}

print("self-check against ids taken from the repository's test-reference.sh")
ok = True
for text, expected in KNOWN.items():
    got = encode(text)
    good = got == expected
    ok = ok and good
    print("  %-26s expected %-30s got %-30s %s"
          % (text, expected, got, "match" if good else "MISMATCH"))
if not ok:
    print("\ngreedy matching does not reproduce known ids; not emitting new prompts")
    sys.exit(1)

print("\nnew prompts, each round-tripped back to text before being accepted")
for text in [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Italy is",
    "The largest planet is",
    "Two plus two equals",
    "The colour of the sky is",
]:
    ids = encode(text)
    if ids is None:
        print("  %-30s CANNOT ENCODE" % text)
        continue
    back = decode(ids)
    print("  %-30s %-34s round trip %s"
          % (text, " ".join(str(i) for i in ids), "ok" if back == text else "FAILED: %r" % back))
