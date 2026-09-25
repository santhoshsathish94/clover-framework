# w — the France example, step by step

`"The capital of France is"` → `" Paris"`, one stage at a time, at full detail.
Every number here is read from this checkpoint or this machine, not from documentation.

Run: `A-p1_tiny-tuned`, trace `/root/k3flow/v1.jsonl`.

---

## Step 1 — text becomes token ids

### The vocabulary file

`/root/k3model/tiktoken.model`. **163,584 entries**, ranks 0 to 163,583, each line a base64
byte string and its rank.

Two precise points:

- Entries are **byte sequences, not characters**. Any input encodes; there is no unknown case.
- **The rank IS the token id.** Id 10484 is not an index into some other table — it is BPE
  rank 10484.

The engine adds 16 more:

```
[TOK] 163584 ranks (max id 163583) + 16 added tokens | kimi=1 rankbpe=1
```

163,584 + 16 = **163,600 reachable ids**, against an embedding matrix of **163,840 rows**.
The 240 spare rows are covered in `m-context-free-map.md`; they are *not* inert.

### The table has two regions

Verified by reading the file:

| ranks | content | |
|---|---|---|
| 0–255 | the 256 single bytes | all 256 present, confirmed |
| 256+ | learned merges | |

So merge number *n* has rank 255 + *n*.

### The algorithm

Start from individual bytes. Repeatedly concatenate the **two** adjacent pieces whose joined
form has the **lowest rank** present in the table. Stop when no adjacent pair is in the table.

Traced on ` capital`:

```
start:    [' '] ['c'] ['a'] ['p'] ['i'] ['t'] ['a'] ['l']

merge 1:  ' '    + 'c'     -> ' c'        rank    275     1+1=2
merge 2:  'i'    + 't'     -> 'it'        rank    277     1+1=2
merge 3:  'a'    + 'l'     -> 'al'        rank    283     1+1=2
merge 4:  'a'    + 'p'     -> 'ap'        rank    447     1+1=2
merge 5:  'it'   + 'al'    -> 'ital'      rank   3090     2+2=4
merge 6:  ' c'   + 'ap'    -> ' cap'      rank   3127     2+2=4
merge 7:  ' cap' + 'ital'  -> ' capital'  rank  10484     4+4=8

final:    id 10484
```

**Always exactly two pieces, never three.** A 3-byte token is a 2-byte piece plus a 1-byte
piece (`'in'` + `'g'` → `'ing'`, rank 288). It is a binary tree and the token is its root.

The full word is never matched against the input string. At merge 7 the current pieces are
`[' cap', 'ital']`; they are concatenated and *that* is looked up. ` capital` is constructed
bottom-up, not recognized.

### Why ` c` won the first merge

All seven adjacent pairs were looked up; the minimum rank wins:

| pair | merged | rank |
|---|---|---:|
| `' '`+`'c'` | `' c'` | **275** |
| `'c'`+`'a'` | `'ca'` | 7606 |
| `'a'`+`'p'` | `'ap'` | 447 |
| `'p'`+`'i'` | `'pi'` | 7661 |
| `'i'`+`'t'` | `'it'` | 277 |
| `'t'`+`'a'` | `'ta'` | 4458 |
| `'a'`+`'l'` | `'al'` | 283 |

275, 277, 283 — three pairs within eight ranks decide the shape of the word. `'ca'` at 7606
is why it builds as `' c'`+`'ap'` and not `'ca'`+`'pital'`: `ca` mid-word is much rarer than
`c` after a space.

Rank 275 means ` c` was the **20th merge ever learned**. What beat it: `'  '`, `'    '`,
`' t'`, `'in'`, `' a'`, `'er'`, `'on'`, `'re'`, `'he'`, `'at'`, `'or'`, `'的'`, `'st'`,
`'en'`, `'   '` — the commonest English bigrams, whitespace runs from code, and Chinese byte
fragments. Approximately frequency order; not exactly, because BPE is greedy and each merge
changes the next round's counts.

### The result

```
"The capital of France is"   24 bytes
54 68 65 20 63 61 70 69 74 61 6C 20 6F 66 20 46 72 61 6E 63 65 20 69 73
```

| id | text | bytes | hex |
|---:|---|---:|---|
| 1008 | `The` | 3 | `54 68 65` |
| 10484 | ` capital` | 8 | `20 63 61 70 69 74 61 6C` |
| 318 | ` of` | 3 | `20 6F 66` |
| 15383 | ` France` | 7 | `20 46 72 61 6E 63 65` |
| 387 | ` is` | 3 | `20 69 73` |
| | | **24** | |

**Lossless** — concatenating the five byte strings reproduces the input exactly, verified
`True`. Nothing normalized, lowercased or stripped.

**The space belongs to the following token.** ` France` and `France` are different ids with
different embedding rows, learned separately.

**Id magnitude tracks frequency**, since id = rank = merge order: ` of` 318, ` is` 387,
`The` 1008, ` capital` 10484, ` France` 15383, `iffel` 142957.

**4.8 bytes per token** here (24 ÷ 5).

### Why not just split on spaces and look up whole words

Tested on the four real prompts from our runs:

| scheme | words | found as one token | would need UNK |
|---|---:|---:|---:|
| split on space | 275 | 56 (**20.4%**) | 219 (79.6%) |
| strip punctuation first | 273 | 65 (**23.8%**) | 208 (76.2%) |

Stripping punctuation buys **3.4 percentage points**. It was never the main problem — what
fails is ordinary vocabulary and code: `evict`, `1`, `2`, `10`, `int64_t`,
`chunk_bytes(void`, `getenv("CHUNK`, `1<<20`.

The vocabulary is already built the other way: only **24.6%** of entries look like whole
words; **75.3%** do not start with a space at all.

Four reasons the whole-word scheme cannot work:

1. **Finite table, infinite language.** 163,584 slots against every number, name, typo, URL
   and identifier. `int64_t` shows it — identifiers cannot be enumerated.
2. **UNK is irreversible.** Today tokenization is lossless. An unknown token discards bytes
   permanently.
3. **Not every language uses spaces.** Rank 269 is `的`, a complete Chinese word.
4. **Subwords transfer.** ` capitalisation` → `[' capital', 'isation']`, and `isation` is
   already understood from other words. A whole-word scheme learns nothing reusable.

And stripping punctuation destroys meaning: `3.14` → `314`, `C++` → `C`, `np.array` →
`nparray`, `int64_t` → `int64t`.

### The argument that settles it

**The vocabulary is also the output alphabet.** The model picks what to write from these
same 163,840. Our own run generated:

| id | token |
|---:|---|
| 20829 | `.",\n` |
| 11 | `,` |
| 13 | `.` |
| 414 | ` "` |
| 427 | `           ` (11 spaces) |

Remove punctuation from the vocabulary and the model cannot emit a full stop, cannot close a
quote, cannot indent Python. Token 427 is code indentation.

Cost of the merging: **7 hash lookups** for ` capital`, against 108.81 GB of weights
downstream. Not a cost worth optimizing.

### Not verified

tiktoken normally applies a **regex pre-split** before BPE so merges cannot cross word or
punctuation boundaries. `encoding_k3.py` was grepped and the pattern was not found in a
readable form. The merge algorithm above is traced and real; the pre-split rule for this
checkpoint is **unconfirmed**. It matters for punctuation runs and numbers, not for this
sentence.
