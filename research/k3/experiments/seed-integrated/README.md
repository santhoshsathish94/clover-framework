# Verified seed-integrated model

This directory preserves the tested AX102 integration snapshot. The complete
combined model source is [model/candidate.c](model/candidate.c), with its headers
beside it. [results.json](results.json) records consumed-row verification,
reference comparisons, source guards, hashes and dependencies.

All thirteen previous representation stages remain active. The new input path
reads the requested embedding rows from the standalone seed dataset, including
the entry-normalization consumer. It does not reconstruct the entire embedding
table in RAM. The original checkpoint and installed reference were not changed.

AX102 experiment: `/opt/clover-k3/seed-integrated-20260929-a`.
Input dataset: `/opt/clover-k3/seed/seed.bin`.

## Run the tested candidate

On AX102, use a new output directory to preserve the existing evidence:

```sh
set -eu
work=/opt/clover-k3/seed-integrated-20260929-a
output=$(mktemp -d "$work/manual-XXXXXX")
. /opt/clover-k3/config.env
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
export K3_PREFETCH=4 K3_TRUNKRAM=0 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1
export K3_INDEX=/opt/clover-k3/build/eqidx.bin
export K3_GROUP_COVERAGE=2 K3_JOINT_RANK=1 K3_SCALE_ROWS=1 K3_SCALE_EXCEPTIONS=1
export K3_VECTOR_ENDPOINTS=3 K3_BASIS_ROUTER=2
export K3_SEED_INPUT=/opt/clover-k3/seed/seed.bin K3_SEED_GUARD_PROBE=1
export K3_IDS=1008,10484,318,15383,387
K3_GROUP_COVERAGE_REPORT="$output/france.jsonl" \
K3_LOGITS="$output/france.bin" K3_DUMPSEL="$output/france.routes" \
K3_ENDPOINT_REPORT="$output/france-endpoints.json" \
K3_BASIS_REPORT="$output/france-basis.bin" \
K3_REVERSE_VECTOR_REPORT="$output/france-vector.bin" \
K3_SEED_REPORT="$output/france-seed.json" \
K3_SEED_ROWS_REPORT="$output/france-seed-rows.bf16" \
"$work/candidate" > "$output/france.log" 2> "$output/france.stderr"
cmp "$work/france-reference.bin" "$output/france.bin"
cmp "$work/france-reference.routes" "$output/france.routes"
printf 'Output and routes match. Artifacts: %s\n' "$output"
```

Native SHA256 support uses a privately unpacked OpenSSL archive; GMP/zlib are
also private experiment dependencies. Dependency paths and hashes are in the
result. No system package was installed. The archive stores the tested binaries
and sources; private dependency packages remain on AX102.
Compiled binaries and the archive are [excluded from Git](../README.md#local-only-artifacts);
the source, reports and verification fixtures remain published.

The original embedding mapping's interior is PROT_NONE during seed-enabled
runs. A child-process SIGSEGV probe checks the guard, with core dumps disabled.
Shared boundary pages leave 4096 embedding bytes outside that page-level guard.
The guard protects this mapping, not arbitrary alternative file accesses.

The native decoder passed a full-table reconstruction check independently of
model execution. Model passes cover France/Japan five-token prefills with the
existing sampled expert stages. No new context-length, generation, throughput
or process peak-RAM result is claimed. Future edits should use another copy,
not modify these frozen tested snapshots.