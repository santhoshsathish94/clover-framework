# Verified seed + fruit model

**PASS on AX102:** seed supplies input embeddings; fruit supplies the output
head. Both France/Japan five-token prefills match the unchanged reference's
full norm/logits and route trace. All thirteen prior representation gates pass.

## Frozen artifacts

- [results.json](results.json): final dataset, model and provenance verification.
- [model/candidate.c](model/candidate.c): tested combined source and headers; compiled binaries stay local/on AX102.
- [native-results.json](native-results.json): full native fruit reconstruction and boundary-row checks.
- [enabled/results.json](enabled/results.json): unchanged prior stage-check results.
- [dataset/all-n.tsv](dataset/all-n.tsv): every distinct fruit scalar with its exact bits and frequency.
- [dataset/results.json](dataset/results.json): fruit file size and complete build verification.
- [dataset/blocks.jsonl](dataset/blocks.jsonl): candidate sizes and selected layout for every block.

The large dataset remains at `/opt/clover-k3/fruit/fruit.bin`. Its reader reuses
the K3SEED1 format; the source tensor hash distinguishes fruit from seed.
The candidate is `/opt/clover-k3/fruit-integrated-20260929-a/candidate`.
See [publication scope](../README.md#publication-scope) for excluded binaries,
archive bundles and third-party dependencies; their recorded hashes are retained.

## Run the tested model

On AX102, with the existing model configuration and a fresh output directory:

```sh
set -eu
work=/opt/clover-k3/fruit-integrated-20260929-a
output=$(mktemp -d "$work/manual-XXXXXX")
. /opt/clover-k3/config.env
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
export K3_PREFETCH=4 K3_TRUNKRAM=0 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1
export K3_INDEX=/opt/clover-k3/build/eqidx.bin
export K3_GROUP_COVERAGE=2 K3_JOINT_RANK=1 K3_SCALE_ROWS=1 K3_SCALE_EXCEPTIONS=1
export K3_VECTOR_ENDPOINTS=3 K3_BASIS_ROUTER=2
export K3_SEED_INPUT=/opt/clover-k3/seed/seed.bin K3_SEED_GUARD_PROBE=1
export K3_FRUIT_HEAD=/opt/clover-k3/fruit/fruit.bin K3_FRUIT_GUARD_PROBE=1
export K3_IDS=1008,10484,318,15383,387
K3_GROUP_COVERAGE_REPORT="$output/france.jsonl" \
K3_LOGITS="$output/france.bin" K3_DUMPSEL="$output/france.routes" \
K3_ENDPOINT_REPORT="$output/france-endpoints.json" \
K3_BASIS_REPORT="$output/france-basis.bin" \
K3_REVERSE_VECTOR_REPORT="$output/france-vector.bin" \
K3_SEED_REPORT="$output/france-seed.json" \
K3_SEED_ROWS_REPORT="$output/france-seed-rows.bf16" \
K3_FRUIT_REPORT="$output/france-fruit.json" \
"$work/candidate" > "$output/france.log" 2> "$output/france.stderr"
cmp "$work/france-reference.bin" "$output/france.bin"
cmp "$work/france-reference.routes" "$output/france.routes"
printf 'Output and routes match. Artifacts: %s\n' "$output"
```

## Consumption and guards

Each fruit consumer streams all 10240 blocks, covering 163840 rows and
1174405120 weights. Each worker holds one decoded 16-row block, not the entire
head. Both the ordinary projection and rounded-source tail evaluator preserve
the original sixteen-double-lane accumulation and final reduction tree.
The duplicate projections remain for the existing correctness comparison.

Per prompt, both fruit passes read 1662367992 compressed payload bytes each;
metadata reads are additional. Each pass's ordered decoded-block hashes match
an independent full decode. Seed's actual prompt rows also match the original
input tensor when checked outside the candidate process.

Both original table mappings have 2348806144 interior bytes protected with
PROT_NONE. Child probes receive SIGSEGV with core dumps disabled. Each table
has 4096 bytes on shared boundary pages outside that guard. It protects those
mappings, not arbitrary alternative file access. Original tables and other
checkpoint parameters remain on disk and are not modified.

Missing fruit and seed supplied as fruit are explicitly rejected; no fallback
or logits are produced. The dataset identities are not interchangeable.

## Limits

Native and independent decoders verified the complete output table; model
tests cover the two five-token prefills and existing sampled expert stages.
No new generation, context-length, throughput or peak-RAM measurement is
claimed. Private OpenSSL/GMP/zlib dependencies are unchanged from seed's
integration, with hashes recorded in the result. No system package installation,
checkpoint mutation, installed-model replacement or commit/push was performed.

These are frozen verified snapshots. Continue from a new copy, not by editing
their recorded source or evidence.