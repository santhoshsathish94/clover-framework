# Clover Intelegence

The separated client, server, normalization, pipeline and transformer packages live
under [code/distrubuted-hosting](code/distrubuted-hosting/), using the requested
spelling. [Standalone hosting](code/standalone-hosting/README.md) now contains the
resident implementation bound to the same current stored datasets. Central data
remains outside code; bins provide only dataset mapping links.

Standalone startup owns the prepared layer/operator mappings. Input/head/leaves
read the existing seed.bin, fruit.bin and leaves.json without recreating parameter
datasets. It now supports variable-length token-ID requests and live generation,
with separate configurable128input/128outputlimits and attention state kept inRAM.
Production expert calculations use existing root weights, not recorded results.
Computed expert outputs are now cached in process RAM on exact layer/expert/input
matches, skipping their calculations and weight reads. The default 256 MiB budget
holds 8,192 entries. In an AX102 two-input/two-output test, Rain falls took65.79s
initially and7.90s on exact repeat with identical output, reusing4,083of4,416expert
results. A changed prompt still computed its misses correctly. These are repeat
cache savings, not a claim that unseen inputs share those results or that the
whole time difference excludes head-cache warmup. No result vectors go to disk.
The output head retains verified BF16 bytes in 2.35 GB RAM. Experts now use a
72 MiB two-buffer read pipeline: the next selected expert is fetched while the
current one is evaluated. The former 16 GiB expert cache is off by default.
Faster DEFLATE and equivalent CRC calculations address the measured decoder cost.
On AX102, two-input/two-output requests retained exact outputs and fell from
156.67 to 72.03 s for Rain falls, and121.28 to55.71s for Tea tastes. Warm output
steps were17-19s, not the historical5seconds. These are combined-change individual
observations, not proof of prefetch's isolated benefit or guaranteed latency.
Bounded live prefill/continuation, JSONprotocol and controller-limit checks passed;
a full128outputmodelrun has not been performed. Before caching, observedAX102outputsteps took
67-71seconds, so the earlier recorded-result request timing does not apply. Build
and inspect with its package scripts. The older run-resident.sh below still
launches the historical binary and is not the new standalone entry point.

The current resident Clover K3 datasets were relocated on AX102 to:

```text
/opt/clover-k3/clover-intelegence/dataset
```

**Logical dataset inventory:3399files,1,387,808,622,300bytes (about1.388TB).**
All physical package datasets now reside under the top-level dataset directory.
Trunks, roots, operators, input/output tables, vocabulary and leaves occupy their
original paths there as real files/directories, not aliases. There are no links
inside dataset. Client, normalization and transformers use bin/dataset links to
the dataset root. Server has two links inside bin/dataset for trunk-0 and its
QKV operator. Linux uses relative symlinks; Windows uses directory junctions.
It excludes code/config/relocation records and documentation updates.
The redundant265008216-byte qkv-all/layer-0 payload was deleted after exact
comparison with the server's canonical QKV file. Its distinct provenance manifest
was preserved beside that file. The old duplicate folder and its one legacy
alias were explicitly retired; no current server dependency was removed.
The QKV operator physically resides under dataset/operators/trunk-0-qkv and the
complete trunk-0 under dataset/trunk-0. Their server bin entries are mapping links.
The original resident-only relocation moved373units. A second95unit relocation
added the complete92computed roots, then inputs, outputs and leaves.json, while
preserving already-moved observations. Each unit was individually verified.
These were same-filesystem renames, not downloads, conversions or regenerated
values. Old model-dataset locations retain compatibility symlinks; the client
data/config aliases were subsequently removed as described below. The C source
and compiled resident program were not changed.

The client source folder is `code/distrubuted-hosting/client`; compiled programs and tokenizer config
remain in `code/distrubuted-hosting/client/bin`. Inputs, outputs and both vocabulary datasets physically
reside at the original paths directly under `dataset`, reached through `code/distrubuted-hosting/client/bin/dataset`. The earlier
twelve retired client data/config aliases remain removed; only the requested new
bin dataset links were added. A bin directory alone is no longer self-contained.
[client.c](code/distrubuted-hosting/client/client.c) supplies
libc-only exact text/ID mapping. See [the package commands](code/distrubuted-hosting/client/README.md);
it intentionally does not split unknown words or sentences with BPE.

## Layout

```text
clover-intelegence/
  run-resident.sh
  restore-dataset-paths.sh
  restore-dataset-paths.ps1
  DATASET-ORIGINAL-PATHS-AX102/
  DATASET-ORIGINAL-PATHS-WINDOWS/
  code/
    standalone-hosting/
      clover-one.c
      numeric-table.h
      normalization-values.h
      build.sh
      run.sh
      bin/
        clover-one
        dataset -> ../../../dataset
    distrubuted-hosting/
      client/
        client.c
        bin/
          client
          configs/tokenizer_config.json
          dataset -> ../../../../dataset
      server/
        server.c
        bin/
          server
          dataset/
            trunk-0 -> ../../../../../dataset/trunk-0
            trunk-0-qkv -> ../../../../../dataset/operators/trunk-0-qkv
      normalization/
        normalization.c
        json.h
        bin/
          normalization
          dataset -> ../../../../dataset
      tansformers/
        transformer-1/ ... transformer-92/
          transformer-N.c
          root.h
          decode.h
          bin/
            transformer-N
            dataset -> ../../../../../dataset
        remaining/
      pipeline/
        pipeline.c
        run-prompt.mjs
        trace-input.c
        bin/
  configs/README.md
  dataset/
    inputs/
    outputs/
    tiktoken.model
    vocabulary.bin
    leaves.json
    trunk-0/ ... trunk-92/
    root-1/ ... root-92/
    operators/
      trunk-0-qkv/
      qkv-all/
        layer-1/ ... layer-92/
    eqidx.bin
    model/model-00094-of-000096.safetensors
    existing package documentation companions
    historical relocation inventories
```

This is the AX102 runtime layout, condensed to omit tests and documentation.
Windows has the same original central paths with bin junctions, small datasets and
metadata only; large numeric payloads and Linux executables remain on AX102.
Windows junctions contain absolute targets and need recreation if the checkout moves.

The index still contains its original absolute global-tensor path; that path
resolves through the approved compatibility link into `dataset/model`. This is
a working same-machine relocation, not a self-contained portable index.
Every package now depends on original central dataset paths through its bin mappings.
Existing relative runtime commands still work. Earlier package notes describing
real files inside bin or inside package-grouped central folders describe older
layouts, superseded here. Older package folders in dataset retain their real
documentation companions only; no compatibility links remain inside dataset.
See [the transformer functions and commands](code/distrubuted-hosting/tansformers/transformer-1/README.md).

All [remaining transformer packages](code/distrubuted-hosting/tansformers/remaining/LAYERS.md) are
also implemented and packaged. Each has its own KDA/MLA behavior, root tables
and snapshot counts. The [completed campaign](code/distrubuted-hosting/tansformers/remaining/README.md)
records 182 passing layer/case comparisons and 273 verified directory moves.
All 3,325 moved files retained their content and identities. External legacy model
paths remain aliases; original paths inside dataset are now real. The twelve
retired external client aliases stay removed.

Each layer was checked independently against original layer arithmetic with
exact-input-bound historical expert observations as the test-only oracle. This
does not claim that the standalone client, server and 92 transformers are already
connected into a validated end-to-end pipeline. No additional model data was generated.

## Scope

The [final normalization stage](code/distrubuted-hosting/normalization/README.md) follows transformer
92. It aggregates the eight snapshots and the selected position's residual, then
returns a normalized7168vector. Its only dataset is the unchanged leaves.json.
Stage93 is the original tail's diagnostic label, not another transformer layer.
eqidx.bin is a shared whole-model index; shard94 contains embeddings, LM-head
weights and the original tail tensors. Both stay in the main dataset and are
not normalization runtime dependencies. LM-head projection and token decoding
remain separate work, not part of this normalization-only implementation.

The first inventory followed the actual dependencies of
[the resident C](../research/k3/experiments/clover-k3-resident/clover-k3.c). The
human then explicitly requested all computed roots, inputs, outputs and leaves.
The logical inventory includes their existing metadata alongside payloads;
inputs, outputs and vocabulary now reside directly under dataset, with the original
code/distrubuted-hosting/client/bin/dataset path retained as a link:

| Group | Historical Move Units | Current Files | Current Bytes |
|---|---:|---:|---:|
| Binary index | 1 | 1 | 21,877,087 |
| Prepared trunks | 93 | 650 | 56,257,772,227 |
| Derived QKV operators, including retained provenance | 94 | 187 | 19,635,790,633 |
| Complete computed roots, including observations | 92 | 2,537 | 1,303,864,040,892 |
| Inputs | 1 | 7 | 1,658,872,001 |
| Outputs | 1 | 13 | 1,667,986,556 |
| Leaves index | 1 | 1 | 58,615 |
| Embedding, tail and head shard | 1 | 1 | 4,697,664,072 |
| Token vocabulary and ID-to-text table | 2 | 2 | 4,560,217 |

The QKV row accounts for one canonical layer-0 payload instead of two; both
original provenance manifests remain. Historical move-unit counts and prior
journals retain their original scope. See the
[duplicate cleanup record](code/distrubuted-hosting/server/QKV-DEDUP-JOURNAL.tsv). Commands that used
the retired qkv-all/layer-0 path must now use the canonical server QKV path.

The complete root directories now include the computed expert representation
previously left outside this folder. Inputs and outputs are the existing
clover-data tables, not newly generated inference output. The original packed
trunk, branch datasets and other experiments' files
remain outside this explicitly requested scope. In particular, the unchanged
resident uses exact-matching expert observations, not the root weight payloads.
Its existing five-input-token/one-next-token restriction and expert-result miss
behavior remain unchanged. Relocation does not add arbitrary-input generation.

Historical manifests and observation metadata moved unchanged with their folders.
No new prompt, route, activation or model-output dataset was created.

## Launch

On AX102:

```bash
bash /opt/clover-k3/clover-intelegence/run-resident.sh
```

[run-resident.sh](run-resident.sh) binds the relocated index, prepared trunks and
derived operators directly, then starts the existing resident binary. It clears
inherited diagnostic/replay options and retains the tested16thread configuration.
The binary remains in `/opt/clover-k3/clover-k3-resident-20261002-a`.
Input protocol is unchanged: an existing observation-set label and five integer
token IDs per line; EOF shuts the process down. See the
[resident program documentation](../research/k3/experiments/clover-k3-resident/README.md).

`CLOVER_DATASET` and `CLOVER_RESIDENT_BINARY` may explicitly override the paths.
The local workspace contains the launcher, C mapping source, relocation inventory
and small vocabulary/config files. The approximately1.388TBnumeric payload remains
on AX102. The3478byteconfig is under code/distrubuted-hosting/client/bin/configs, outside the dataset total
above. Local client input/output folders contain location notes, not numeric data.

## Checks

The latest correction replaced279central AX102aliases with their own real payloads
and restored4client entries at the dataset root. All283remote/284local moves passed
their frozen checks. Local server manifests were placed at canonical trunk/operator
paths. Every local payload hash/timestamp and every remote moved entry identity
matched; remote files<=1MiB were also SHA256-checked. Both dataset trees now contain
zero links, and code bins contain zero real dataset files. All95package mappings,
external aliases, unchanged code/binaries and runtime loader checks passed.
No inference ran. See the DATASET-ORIGINAL-PATHS records and [context](CONTEXT.md).

The preceding package-grouped centralization moved95dataset directories on both machines.
AX102 retained all3491file identities across1,383,089,110,645bytes and checked SHA256
for2561files at most1MiB; large files were renamed on the same filesystem, not
rehash-verified or copied. Windows retained all472file hashes across7,631,695bytes.
These counts include existing dataset companions added after the model inventory.
All bin links, original model aliases, shared data and code/binary metadata passed
the frozen move verification before documentation updates. Client lookup and all94
AX102 stage loaders passed through the new links, without inference. Local client
and normalization loaders passed; large numerical datasets are not present locally.
See [the current context](CONTEXT.md) and the separate DATASET-CENTRALIZATION records.

The following paragraphs record earlier relocation outcomes. Their client-path
compatibility claims predate the explicitly approved bin-only move below.

[relocate.py](relocate.py) planned every source and destination, checked existing
readers, refused changed sources or occupied destinations, and moved one unit at
a time. Each move was followed by per-file device/inode/size/mtime checks through
both new and old paths before its journal entry and the next move. Final checks
confirmed every old path resolves and global index targets are inside the new
dataset folder. Payload hashes were not independently recomputed across81.85GB;
same-inode rename and file-identity checks establish this move's preservation.

[relocate-roots.py](relocate-roots.py) performs the second relocation in the
requested order. It consolidates each complete source root with its previously
moved real observations, then installs one whole-root alias at the original
location. This avoids circular child aliases. It preserves original root and
file identities and uses a separate STARTED/completed journal. Ordinary merge
errors roll back that root; an interrupted transaction stops for inspection.

The second pass moved1070additional files/1,306,218,356,434bytes. Final verification
checked all3398combined dataset files, every old-path resolution and all1488prior
observation files. All92root binary headers were readable, leaves.json parsed,
and no staging directory or nested self-link remained. The original relocation
verifier recognizes aliases at either the source or its ancestor and retains
all exact target/file-identity checks. No full1.3TBrehash or new inference run
was needed for the extension; no numerical payload contents were changed.

Runtime verification launched the unchanged binary with the new bindings and
inspected its actual mappings:372prepared/operator payload mappings plus the
global shard were at the new paths. Both existing reference cases passed, with
93layers/651operator records/7360exact expert matches and zero fixed mapping churn.
Startup4.765682s; requests2.479453s and2.217874s; shutdown3.118694s. These are
relocation smoke-test observations, not a performance comparison. File identities
were rechecked afterward. The process exited; no service was left running.
That runtime check belongs to the first relocation; the extension was checked
through file identities, alias resolution and actual header/JSON reads.

[The dataset inventory](dataset/README.md) links the complete plan, journal and
verification summary. [CONTEXT.md](CONTEXT.md) records the direction and outcome.
No active readers were found during preflight; no campaign was signalled and no
unrelated data was deleted. No commit or push was made.

The text-asset move is recorded by [TOKENIZER-MOVE-PLAN.tsv](TOKENIZER-MOVE-PLAN.tsv)
and [TOKENIZER-MOVE-JOURNAL.tsv](TOKENIZER-MOVE-JOURNAL.tsv). All three asset SHA256
hashes matched before and after relocation and local transfer. The C exact mapper
passed whole-vocabulary, missing-ID, ownership and malformed-data tests on both
Windows and Linux; Linux memory/undefined-behavior sanitizers also passed.

The later client-package move is recorded separately by
[move-client.sh](move-client.sh), [CLIENT-MOVE-PLAN.tsv](CLIENT-MOVE-PLAN.tsv) and
[CLIENT-MOVE-JOURNAL.tsv](CLIENT-MOVE-JOURNAL.tsv). All 30 original server files
passed full SHA256 and device/inode/size/mtime checks after moving: seven code
files, seven input files, thirteen output files, two vocabulary files and one
config. The package contained no internal symlinks, and old data paths resolved
to the same payload inodes. Local relocation preserved 10 files by hash/size/mtime.

These plans are historical preservation snapshots, not mutable build manifests.
The client move's full verification ran before README/CONTEXT updates; rerunning
that frozen check after editing or rebuilding planned files will detect changes.
Earlier relocation scripts retain their original target-path assumptions, which
predate the new client aliases. Their journals were not rewritten.

The final runtime move is recorded by [move-client-bin.sh](move-client-bin.sh),
[CLIENT-BIN-MOVE-PLAN.tsv](CLIENT-BIN-MOVE-PLAN.tsv),
[CLIENT-BIN-ALIASES.tsv](CLIENT-BIN-ALIASES.tsv) and
[CLIENT-BIN-MOVE-JOURNAL.tsv](CLIENT-BIN-MOVE-JOURNAL.tsv). All 27 original runtime
files passed full SHA256/device/inode/size/mtime checks before documentation
updates. Existing tests and the CLI passed using only bin-internal paths, then
all twelve approved aliases were unlinked without deleting their payloads.
The removed-link record preserves exact old link text and destinations for audit.
The local eight-file move preserved hashes/size/mtime and passed the same tests.

Only the current client runtime moved. Other experiments and their archived
copies were not cleaned up or rewritten. Old commands and relocation verifiers
that require the removed aliases now need explicit new paths; historical plans
remain unchanged. No numeric data, C behavior, source/tests or model run changed.

## Layer 0 Server

[server.c](code/distrubuted-hosting/server/server.c) now implements only layer 0 with individually
scoped functions and owned model/sequence state. Standard C and standard math,
no third-party libraries or OpenMP in the server. It reads raw embedding vectors
from stdin and returns the layer-0 residual plus S0; it is not a network listener.
The current client is not yet connected to it.

The requested QKV operator folder now lives at code/distrubuted-hosting/server/bin/dataset/trunk-0-qkv,
with identical hashes and file identities and old operator paths retained as
aliases. The complete prepared trunk-0 directory also lives inside server/bin/dataset;
its six files and directory identity were preserved. The server now loads both
datasets through internal relative paths, with its source and binary unchanged.
No MLA/MoE/shared experts
or pre-attention AR are performed. See [server commands and tests](code/distrubuted-hosting/server/README.md)
and the [historical server data notes](dataset/server/README.md). Source and tests
stay outside bin; all current server runtime files are together beneath bin.

Layer-0 residual and S0 matched the original source exactly on both existing
five-token sequences. Independent/reset sequences, standalone sanitizers and CLI
framing/error checks passed. This is not a new full-model or performance result.