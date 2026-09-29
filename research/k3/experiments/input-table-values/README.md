# All n in the input embedding table

The actual AX102 input table was scanned in full on 2026-09-29:
163840 rows x 7168 BF16 values, totaling 1174405120 entries and 2348810240 bytes.

**There are 6658 distinct stored values of n. A fixed-width identifier needs
13 bits because 4096 < 6658 <= 8192.** Here n means an actual signed scalar
in the input table, not an extracted factor or a fitted vector coefficient.

## Complete log

- [all-n.tsv](all-n.tsv): every distinct value once, with its occurrence count,
  dictionary ID, BF16 and float32 bit patterns, decimal/hexadecimal rendering,
  and exact rational numerator/denominator. IDs are ordered by unsigned BF16
  bit pattern, not by numerical magnitude. All repeated occurrences are counted.
- [dictionary.bf16](dictionary.bf16): the 6658 values stored exactly as BF16,
  13316 bytes, in dictionary-ID order.
- [counts.u64le](counts.u64le): 65536 unsigned little-endian 64-bit counters,
  one for every possible BF16 bit pattern, including absent patterns.
- [scan-chunks.tsv](scan-chunks.tsv): complete sequential row coverage and
  SHA256 for each scanned chunk of the original payload.
- [results.json](results.json): counts, range, most frequent values, artifact
  hashes, source identity and verification results.

Observed range: **-0.3515625 to 0.1533203125**. All entries are normal finite
BF16 values; no zeros, subnormals, infinities or NaNs were observed.

## Verification

The [scanner](../input-table-values.py) first passed controls over every possible
BF16 bit pattern against independently counted frequencies, including special
values and chunk boundaries. It then read every input-table value, followed by
a second full pass that encoded each entry as a dictionary ID and reconstructed
the original BF16 bytes. Both payload hashes and all frequencies match.

Payload SHA256:
`4a79cdabdab6826b994aff69d90a72aa35dae32e90f8acaf1ce312c7bdc4a487`

Checkpoint file size, modification time and inode stayed unchanged. Only the
input embedding tensor was scanned, not the output table, trunk or expert pool.
Raw artifacts remain on AX102 under
`/opt/clover-k3/input-table-values-20260929-a`.

All artifacts are also preserved here and hash-verified after transfer. Local
checks independently matched every TSV value to its binary dictionary entry,
float32 bits, exact rational value and occurrence counter. All 65536 counters
sum to 1174405120 entries, and the 640 chunk records cover every input row
exactly once. The scanner's local source hash matches the executed source.

## Identifier size accounting

| Component | Bytes |
|---|---:|
| Original BF16 table | 2348810240 |
| Ideal 13-bit identifiers for every entry | 1908408320 |
| Exact BF16 dictionary | 13316 |
| Identifiers plus dictionary | 1908421636 |
| Difference below original size | 440388604 |

This is calculated fixed-width storage accounting, excluding headers and
alignment. No full bitpacked 13-bit table or model consumer has been built;
the second pass verifies the ID-to-value mapping, not a packed-table format.
These 13 bits identify one n using the shared dictionary; they do not encode
an entire vector or its position masks. Other encodings can have different costs.