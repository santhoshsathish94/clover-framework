#!/usr/bin/env python3
"""Generate text with clover-k3 by reusing the prefix state between tokens.

Step 0 is a full prefill that also writes a cache covering every prompt
position. Each later step loads that cache, computes only the one new
position, and writes the cache the next step needs.

  gen.py "The capital of France is" 12
  gen.py --ids 1008,10484,318,15383,387 12
"""
import base64
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
WORK = os.environ.get("K3_GENWORK", "/tmp/k3gen")

RUN_ENV = {
    "OMP_PROC_BIND": "close", "OMP_PLACES": "cores", "OMP_NUM_THREADS": "16",
    "K3_PREFETCH": "4", "K3_TRUNKRAM": "0", "K3_NREADER": "14",
    "K3_XDEC": "2", "K3_PLGRAN": "1", "K3_HUGE": "1",
}


def config():
    """Source config.env rather than parse it; the entries use shell expansion."""
    want = ["K3_MODEL", "K3_TRUNK", "K3_TRUNKPATH"]
    script = ". %s; %s" % (os.path.join(HERE, "config.env"),
                           "; ".join('printf "%%s\\n" "$%s"' % v for v in want))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return dict(zip(want, out))


CFG = config()
VOCAB_FILE = os.path.join(CFG.get("K3_MODEL", "/root/k3model"), "tiktoken.model")

_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        _vocab = {}
        for line in open(VOCAB_FILE, "rb"):
            p = line.split()
            if len(p) == 2:
                _vocab[int(p[1])] = base64.b64decode(p[0])
    return _vocab


def show(tok):
    return vocab().get(tok, b"?").decode("utf-8", "replace")


def binary(npos):
    b = os.path.join(BUILD, "ck3_%d" % npos)
    src = os.path.join(HERE, "clover-k3.c")
    # mtime, not existence: a binary left by an older source silently lacks
    # whatever the newer source added.
    if not os.path.exists(b) or os.path.getmtime(b) < os.path.getmtime(src):
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-ffp-contract=off", "-fopenmp",
             "-DNPOS=%d" % npos, "-o", b, src, "-lm", "-lpthread"], check=True)
    return b


def step(ids, load, out):
    """One forward pass. Returns (token, seconds, positions actually computed)."""
    npos = len(ids)
    env = dict(os.environ)
    env.update(RUN_ENV)
    env["K3_TRUNKPATH"] = CFG["K3_TRUNKPATH"]
    env["K3_INDEX"] = os.path.join(BUILD, "eqidx.bin")
    env["K3_IDS"] = ",".join(str(i) for i in ids)
    env["K3_LOGITS"] = os.path.join(WORK, "logits.bin")
    computed = npos
    if load:
        env["K3_PFXLOAD"] = load
        env["K3_PFXN"] = str(npos - 1)
        computed = 1
    if out:
        env["K3_PFXOUT"] = out
    t0 = time.time()
    r = subprocess.run([binary(npos)], env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        sys.exit("run failed at %d positions:\n%s" % (npos, r.stderr[-2000:]))
    m = re.search(r"^emitted token\s*:\s*(\d+)", r.stdout, re.M)
    if not m:
        sys.exit("no token in output:\n%s" % r.stdout[-2000:])
    return int(m.group(1)), dt, computed


def run(prompt, n, reuse):
    """Generate n tokens. reuse=False recomputes every position each step."""
    a = os.path.join(WORK, "pfx_a.bin")
    b = os.path.join(WORK, "pfx_b.bin")
    out, total, load = [], 0.0, None
    for i in range(n):
        cur = prompt + out
        dst = (a if i % 2 == 0 else b) if reuse else None
        tok, dt, computed = step(cur, load if reuse else None, dst)
        total += dt
        out.append(tok)
        if reuse:
            load = dst
        print("  %2d  %7.2f s  %2d pos  %-7d %r"
              % (i + 1, dt, computed, tok, show(tok)))
        sys.stdout.flush()
    return out, total


def main():
    args = sys.argv[1:]
    if len(args) < 2 or args[0] != "--ids":
        sys.exit(__doc__)
    ids = [int(x) for x in args[1].split(",")]
    n = int(args[2]) if len(args) > 2 else 8
    mode = args[3] if len(args) > 3 else "both"

    os.makedirs(WORK, exist_ok=True)
    print("prompt : %r" % "".join(show(t) for t in ids))
    print("ids    : %s" % ",".join(str(t) for t in ids))

    res = {}
    for label, reuse in (("full recompute", False), ("prefix reuse", True)):
        if mode not in ("both", label.split()[0]):
            continue
        print("\n--- %s" % label)
        res[label] = run(ids, n, reuse)
        toks, total = res[label]
        print("  %d tokens in %.1f s   %.2f s per token" % (n, total, total / n))
        print("  text: %r" % "".join(show(t) for t in toks))

    if len(res) == 2:
        f, p = res["full recompute"], res["prefix reuse"]
        same = f[0] == p[0]
        print("\ntokens identical : %s" % ("YES" if same else "NO"))
        if not same:
            print("  full   %s" % f[0])
            print("  prefix %s" % p[0])
        print("speedup          : %.2fx  (%.1f s -> %.1f s)"
              % (f[1] / p[1], f[1], p[1]))
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
