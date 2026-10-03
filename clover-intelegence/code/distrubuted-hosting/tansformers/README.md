# Transformers

[Transformer 1](transformer-1/README.md) and all transformers 2-92 are implemented.
The directory spelling follows the requested layout. Each transformer owns its
C source, decoder headers and bin runtime with its three real dataset directories
on AX102. Standard C/math and platform file I/O only; main coordinates separate
functions. Precomputed parameter values are reused, while current-input Q/K/V and
expert outputs are calculated live.

[All layer packages](remaining/LAYERS.md) lists transformers 2-92 with their KDA
or MLA type, snapshot counts and individual verification records. The
[completed campaign](remaining/README.md) documents 182 passing layer/case
comparisons, representative sanitizer/CLI checks and 273 verified directory moves.
Transformer 1 retains its earlier separate implementation and tests unchanged.

From inside transformer-N/bin, run `./transformer-N --inspect` or
`./transformer-N --stream`. Layer-specific guides define the input snapshot bundle.
Use `ulimit -c 0` first. Existing model aliases remain; local dataset folders hold
metadata only. The packages have not been connected into an end-to-end client,
server and transformer pipeline.