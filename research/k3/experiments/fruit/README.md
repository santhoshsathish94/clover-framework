# fruit: output-head dataset

Direction: do for the actual output-head table what was done for seed, including
model integration and exact-output tests. Original checkpoint and seed remain
unchanged. This is a distinct dataset with its own dictionary and source hash.

The output head is BF16[163840,7168], 2348810240 bytes, starting at byte 584 in
shard94. The [fruit builder](../fruit.py) reuses the unchanged seed container
and codecs. First census every output weight, then choose the smallest of six
exact layouts per 16-row block, reopen and compare every byte and frequency.

The shared K3SEED1 format supports both dataset names; filename and full source
hash distinguish fruit from seed. Reader software does not require the original
checkpoint. No numeric-domain assumption is copied from the input-table census.

Next, a separate copy of the seed-integrated model will stream decoded fruit
blocks into both ordinary and source-expression output projections. Original
projection reduction order and prior correctness gates must stay unchanged.
No full-head materialization, compression optimum or speedup is assumed.

Full output-table census passed: 1174405120 entries, 6590 distinct normal
finite BF16 values, no zeros/subnormals/infinities/NaNs. Fixed dictionary IDs
need 13 bits. The observed range is -0.578125 to 0.46484375.

Output payload SHA256:
`11c1f1c09a8e0db55547b5e68ebfd1d8e3b503bee56c4e1312ef55ecd3e5580f`

All codec controls and the full build passed. The fruit file occupies 1662954740
bytes including 128 header bytes, 13180 dictionary bytes, 573440 index bytes and
1662367992 payload bytes. It saves 685855500 bytes (29.20%) versus the original.
10239 blocks selected compressed BF16 byte planes; one selected bit planes.
Every reopened byte and frequency matched the complete source table.

The unchanged native reader also reconstructed the complete fruit payload hash
without opening the checkpoint. Row checks include the sole bit-plane block,
which starts at row 176.

Initial model integration passed on France: fresh reference, seed-only prior
binary, fruit-disabled control, and seed+fruit enabled all match full outputs
and routes. Both fruit consumers traversed all 10240 blocks and 163840 rows;
all 13 previous representation gates still pass. Both seed/head interior mappings
were protected and their child read probes faulted as expected.

Japan also passed full outputs/routes and all 13 previous checks with both
datasets enabled. Both fruit consumers again read the entire head. A repeated
France reference matches. Missing fruit and incorrectly supplying seed as fruit
are rejected, without producing logits.

## Completed outcome

**fruit is built and integrated alongside seed.** The complete file is
**1662954740 bytes (1.663 GB)** versus 2348810240 bytes originally, a **29.20%**
reduction including header, dictionary, index, checksums and block payloads.
The distinct-value dictionary is separate from seed: **6590 values**, not 6658.

A separate standard-library decoder reconstructed every fruit value without
opening the checkpoint. Its ordered block hashes match both model consumers
on both prompts. Source data, seed, installed reference and all prior checks
retain their recorded hashes. Both original table interior mappings were
guarded against reads, with the documented shared-boundary-page exception.

- [All fruit values and frequencies](../fruit-integrated/dataset/all-n.tsv)
- [Dataset size and full reconstruction](../fruit-integrated/dataset/results.json)
- [Tested model, run instructions and verification](../fruit-integrated/README.md)
- [Complete integration report](../fruit-integrated/results.json)

On AX102, read a requested output weight row with:

```sh
python3 /opt/clover-k3/fruit/fruit.py read /opt/clover-k3/fruit/fruit.bin --row 17374
```

The packaged reader uses NumPy and zlib, and needs no original checkpoint for
row decoding. This command prints the row hash and first eight values; the
FruitReader API exposes complete BF16 bytes or exact float32 values.

Fruit SHA256:
`b6a1ead4fda2928127841ffa2ec63d4f56d3de5709bfd24b4da2cc0d07b58c16`

The [local-only verified snapshot archive](../README.md#local-only-artifacts) SHA256 is
`df42a580f08b78ac0e050e165cccea5b2cdfa8b228c646222d86883fec3db534`.
The actual fruit binary stays on AX102; code, census, reports and model snapshots
are local. No speed or peak-RAM claim is made by these correctness tests.