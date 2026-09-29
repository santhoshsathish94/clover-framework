# Verified cumulative snapshots

The adjacent step directories are frozen AX102 experiment snapshots, ordered
step13 through step1. Each candidate was compiled and passed both prompt gates
before the next item was added. The final cumulative source is step1/candidate.c.

These are experimental consumer integrations with sampled coverage and retained
reference data. Intermediate descriptors are composed for correctness; they
are not a deployed checkpoint format or measured memory/speed improvement.

The campaign result lists each step's source/result identities, original
protected-file hashes, dependency hashes and coverage. Raw group/projection
traces remain on AX102 under /opt/clover-k3/reverse-integration-20260929-a.
Private GMP/zlib files were unpacked locally, not system-installed. Compiled
binaries, the copied deps directory and archive are
[excluded from Git](../README.md#local-only-artifacts); source and result
identities in the reports are unchanged.

Prior source and gates are preserved. Do not edit these snapshots to change a
recorded result; create a new experimental step instead.