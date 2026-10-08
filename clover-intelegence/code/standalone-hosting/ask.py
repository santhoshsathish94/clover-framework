"""Encode/decode text with the shipped tiktoken vocabulary, without the tiktoken package.

The pretokeniser is the Kimi pat_str reduced to ASCII English, which is all the prompts
here need; non-Latin text would require the full \\p{...} classes and the regex module.
Self-checks against known-good ids before it will encode anything.

Usage:  python3 ask.py encode "What is gravity?"
        python3 ask.py decode 1008 10484 318
"""
import base64
import re
import sys

MODEL = "/opt/clover-k3/clover-intelegence/dataset/tiktoken.model"

PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""|[^\r\nA-Za-z0-9]?[A-Za-z]+"""
    r"""|[0-9]{1,3}"""
    r"""| ?[^\sA-Za-z0-9]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

ranks = {}
for line in open(MODEL):
    parts = line.split()
    if len(parts) == 2:
        try:
            ranks[base64.b64decode(parts[0])] = int(parts[1])
        except Exception:
            pass
inverse = {v: k for k, v in ranks.items()}

SPECIAL = {
    "[BOS]": 163584, "[EOS]": 163585, "<|end_of_msg|>": 163586, "<|open|>": 163587,
    "<|close|>": 163588, "<|sep|>": 163589, "[start_header_id]": 163590,
    "[end_header_id]": 163591, "[EOT]": 163593,
}
for name, value in SPECIAL.items():
    inverse[value] = name.encode()


def merge(piece):
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best, where = None, -1
        for index in range(len(parts) - 1):
            rank = ranks.get(parts[index] + parts[index + 1])
            if rank is not None and (best is None or rank < best):
                best, where = rank, index
        if where < 0:
            break
        parts[where:where + 2] = [parts[where] + parts[where + 1]]
    return parts


def encode(text):
    out = []
    for name, value in SPECIAL.items():
        text = text.replace(name, "\x00%d\x00" % value)
    for segment in text.split("\x00"):
        if segment.isdigit() and int(segment) in inverse and int(segment) >= 163584:
            out.append(int(segment))
            continue
        for chunk in PATTERN.findall(segment):
            raw = chunk.encode("utf-8")
            if raw in ranks:
                out.append(ranks[raw])
                continue
            for part in merge(raw):
                out.append(ranks[part])
    return out


def decode(ids):
    return b"".join(inverse.get(i, b"") for i in ids).decode("utf-8", "replace")


CHECKS = {
    "The capital of France is": [1008, 10484, 318, 15383, 387],
    "The largest planet is": [1008, 10604, 19645, 387],
    "Two plus two equals": [17285, 9620, 2069, 28542],
    "The colour of the sky is": [1008, 12955, 318, 276, 18985, 387],
}

if __name__ == "__main__":
    for text, want in CHECKS.items():
        got = encode(text)
        if got != want:
            print("ENCODER SELF-CHECK FAILED: %r -> %s, expected %s" % (text, got, want),
                  file=sys.stderr)
            sys.exit(1)
    if sys.argv[1] == "encode":
        ids = encode(sys.argv[2])
        print(",".join(str(i) for i in ids))
        print("%d tokens, roundtrip: %r" % (len(ids), decode(ids)), file=sys.stderr)
    else:
        print(decode([int(x) for x in sys.argv[2:]]))
