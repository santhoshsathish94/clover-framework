# Computation and data experiments

Chronological investigations, including approaches that did not work. Interpret every
measurement in its recorded environment; do not replace an earlier observation with a later result.

- [Equation reduction](k3-equation-reduction.md)
- [Implementation optimization journal](k3-equation-solution.md)
- [Redundancy investigation](k3-redundancy.md)
- [Data-movement problem](k3-data-problem.md)
- [Vector and expert representation study](vector-equations.md)
- [Reverse-order cumulative integration](reverse-integration.md)
- [Measured integrated speed and RAM](integrated-performance.md)
- [Actual server source datasets](source-datasets.md)
- [Complete input-table value census](input-table-values/README.md)
- [Seed input dataset](seed/README.md)
- [Seed model integration](seed-integrated/README.md)
- [Fruit output dataset](fruit/README.md)
- [Seed and fruit model integration](fruit-integrated/README.md)
- [Remaining equation candidates](remaining-equation-candidates.md)

## Publication scope

Git contains research source, patches, frozen source snapshots, reports and
verification fixtures. The experiment files retain their exact bytes so that
recorded source hashes survive checkout. Historical pass/fail records are not
rewritten when later work changes the outcome or interpretation.

## Local-only artifacts

Compiled executables, copied third-party development packages/libraries and
archive bundles are excluded from Git. They remain in the original local
workspace and on AX102; the complete seed/fruit datasets remain on AX102 only.
Nothing was deleted to prepare this commit.

The reports preserve hashes for both published files and these excluded
artifacts. A clone alone is therefore not the full archived execution
environment. Rebuild using the recorded flags/dependencies or use the preserved
server experiment; do not interpret an absent binary as a failed source check.

| Evidence bundle | Preserved AX102 path |
|---|---|
| Reverse integration | `/opt/clover-k3/reverse-integration-20260929-a/verified-snapshots.tar.gz` |
| Integrated speed/RAM | `/opt/clover-k3/integrated-performance-20260929-a/evidence.tar.gz` |
| Seed integration | `/opt/clover-k3/seed-integrated-20260929-a/verified-artifacts.tar.gz` |
| Fruit integration | `/opt/clover-k3/fruit-integrated-20260929-a/verified-artifacts.tar.gz` |

The dataset paths are `/opt/clover-k3/seed/seed.bin` and
`/opt/clover-k3/fruit/fruit.bin`. Their dictionaries, value censuses, block-size
audits and verification results are published, but their full payloads are not.

[Investigation index](../README.md).