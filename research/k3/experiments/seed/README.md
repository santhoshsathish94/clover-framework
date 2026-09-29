# seed

Direction: store the complete input-vector dataset losslessly and reconstruct
the values when needed. Source table stays unchanged. A subsequent
[isolated model integration](../seed-integrated.md) now consumes seed rows
and passes the recorded model checks; the installed reference is unchanged.

## Completed build

**PASS.** The complete input table is stored as `/opt/clover-k3/seed/seed.bin`
on AX102, with the reader at `/opt/clover-k3/seed/seed.py`.

| Component | Bytes |
|---|---:|
| Original input BF16 payload | 2348810240 |
| Seed header | 128 |
| Exact value dictionary | 13316 |
| Block index, including checksums | 573440 |
| Compressed block payloads | 1654480549 |
| Complete seed file | **1655067433** |
| Bytes saved versus original payload | **693742807** |

That is **1.655 GB instead of 2.349 GB**, or **29.54% smaller**, counting all
metadata inside the seed file. Companion reports and software are separate;
their sizes are not included. The original checkpoint remains present.

All 10240 blocks selected `zlib_bf16_byte_planes`: grouping the low and high
bytes separately before compression was smaller than the other tested layouts.
No block selected the 13-bit dictionary-ID layout. The format still includes
the exact 6658-value dictionary, but that decoder does not need it for these
blocks. This is not the earlier one-n-plus-mask-per-vector size estimate.

## Format

The dataset is one self-contained seed.bin file with a 128-byte header, the
exact BF16 value dictionary, a 56-byte entry per block, and block payloads.
Each block contains up to 16 input vectors; a row lookup decodes only its block.
Every index entry has a payload CRC32 and reconstructed-byte SHA256. Header,
dictionary and index have a shared SHA256. No original checkpoint is needed
by the reader.

Six lossless layouts are compared for each block; the smallest payload wins:
raw packed dictionary IDs, zlib-compressed packed IDs, zlib ID bitplanes,
zlib raw BF16 bytes, zlib BF16 byte planes, and zlib BF16 bitplanes.
The full-table dictionary has 6658 values, so packed IDs have 13 bits each.
This is best among those tested layouts for the fixed block size, not a proof
of the best possible encoding.

The [builder and reader](../seed.py) preserve bit patterns and support on-demand
BF16 or exactly widened float32 rows. A single n/mask pair is not assumed to
represent a complete row; sampled rows have roughly1600 distinct values.
The new layouts retain the position information needed to reconstruct them all.

## Read on demand

On AX102:

```sh
python3 /opt/clover-k3/seed/seed.py read /opt/clover-k3/seed/seed.bin 1008
```

The command reconstructs the requested row and prints its hash and first eight
values. To obtain the full vector in Python, use the packaged reader:

```python
from seed import SeedReader

with SeedReader("/opt/clover-k3/seed/seed.bin") as dataset:
	vector = dataset.read_float32(1008)
	original_bf16_bytes = dataset.read_bf16(1008)
```

`vector` contains all 7168 float32 values, exactly widened from the original
BF16 bits. The script directory must be on Python's import path; NumPy and
Python's standard zlib module are required (tested with NumPy 1.26.4 and zlib
1.3). A lookup decodes 16 rows, or 229376 raw bytes, rather than the whole
table. No benchmark of read latency, process RAM, or model throughput is claimed.

## Verification

Controls passed for all codecs, independently assembled 13-bit fields, all
65536 BF16 patterns, random-access reads, exact float32 widening and corrupted
file rejection. After building, the reopened file matched every original byte
of all 163840 rows and 1174405120 values. Counts and the reconstructed payload
hash match the previous full input-table census.

A separate standard-library decoder then reconstructed all blocks without
opening the checkpoint, checked metadata/payload checksums, reproduced the
same full-table hash and verified every chosen codec's size against the block
audit. The packaged on-demand reader also reproduced token row 1008.

- [results.json](results.json): actual size, codec comparisons and full build verification.
- [verification.json](verification.json): independent standalone decode verification.
- [blocks.jsonl](blocks.jsonl): all block boundaries, candidate sizes and chosen payloads.

Seed file SHA256:
`9f064e022771c82fb8d218a03d6e5ebdfab5e3f22d4a59ec5cd25b8498a552d3`

Reconstructed input-table SHA256:
`4a79cdabdab6826b994aff69d90a72aa35dae32e90f8acaf1ce312c7bdc4a487`

Source file size, inode and modification time were unchanged. No output head,
trunk or expert data is included. The storage verification above is distinct
from the subsequent model-integration evidence and is not a global compression
optimum. The large seed file remains on AX102; code, reports and block audit
are local here.

The [fruit output-head integration](../fruit-integrated/README.md) subsequently
passed with seed still enabled, so both tables now have a tested compressed
source path in the isolated candidate. The installed reference remains unchanged.