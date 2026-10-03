# Transformer 22

Standalone layer-22 KDA implementation with separate functions for loading, aggregation, normalization, projection, attention, routing, expert computation and I/O. Main coordinates only. Standard C/math and platform 64-bit file I/O, no third-party runtime. Stored values are reused; input-dependent expert and QKV outputs are computed live.

## Runtime

From this directory, build with:

```sh
mkdir -p bin
ulimit -c 0
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math transformer-22.c -lm -o bin/transformer-22
cd bin
./transformer-22 --inspect
./transformer-22 --stream
```

All runtime data is under bin/dataset: trunk-22, root-22, operators/qkv-all/layer-22. Old model paths remain compatibility aliases. Large payloads remain on AX102; local folders carry metadata only. Runtime does not read observations or external model data.

## Interface

Each token input is 7168 residual floats followed by 2 snapshots of 7168 floats each, oldest first. Output is residual followed by 2 snapshots, one hexadecimal-float line per vector. EOF ends the sequence. Snapshot boundary updates occur after pre-attention aggregation. KDA keeps recurrent state and raw convolution history.

The API transformer_process accepts model, sequence, input, input snapshots, input snapshot count, output and output snapshots. Callers allocate enough output storage and use nonoverlapping buffers outside internal state. Each sequence owns state; all calls sharing a model must be serialized because its expert-file cursor is shared. Reset starts a new sequence; close sequences before their model. No new client/network connection is implemented.

## Verification

Generation is not numerical verification. See [campaign evidence](../remaining/README.md) for current status, tested scope and per-layer results. [generation.json](generation.json) pins source hashes and layer-specific metadata. Existing transformer1/server sources are unchanged.
