# Independent Input and Output C Functions

## Follow-Up: Native Character Merging

The human points to character combining and the K3 stages document. The new
pipeline/trace-input.c diagnostic starts from the exact text and invokes original
km_letters, bpe_piece and tok_encode functions unchanged. On AX102 it observed
five text chunks, then seven final IDs after BPE completes: " captial" remains
" capt" + "ial", and " france" remains " fr" + "ance". Both whole chunks are
absent from the vocabulary. The native result round-trips to the exact input.

Its actual stdout IDs fed the unchanged test-input directly, with no handwritten
ID list: all50176float32 coordinates matched original embedding rows. The stages
document's five-vector measurement explicitly uses "The capital of France is".
Executed original tokenizer headers match the inspected local sources by SHA256.
See ../pipeline/CONTEXT.md for the diagnostic scope and build-path correction.
No runtime source, original test, dataset, prompt spelling or model semantics were
changed. Full pipeline remains paused; this was input-only execution, not inference.

## Current Direction: Input Verification Only

The human pauses the full prompt pipeline and asks to verify input-to-vector
against the original K3 code, expecting five words to give five vectors. Do not
resume layers, normalization or output-head work. No matching active pipeline
process was found during the pause check.

Read original kimi-k3-in-c/src/cli/k3_run.c: --prompt uses tok_encode and forward
loads one embedding row per returned ID. k3_bind.h k3_embed_row indexes by token
row, not a whole-word dictionary. Original clover-k3.c's five-ID default is
1008,10484,318,15383,387, not a general rule that five words give five vectors.
Actual native tokenizer plus the current launcher agree: exact lowercase typo
"the captial of france is" gives2108,11989,682,318,2225,876,387 (seven tokens);
"The capital of France is" gives the five original default IDs. The seven decoded
pieces are the / space+capt / ial / space+of / space+fr / ance / space+is.
No spelling/case correction or merging is authorized by the original-code check.

The assistant's offer to average token vectors before tracing original behavior
was premature; that would change the model's input. Do not force a five-vector
count by dropping pieces, averaging, or substituting the historical prompt.
Add a focused test of only client_input, comparing all coordinates of the actual
seven token IDs and historical five IDs to original shard94 embedding rows.
The runtime mapping remains unchanged unless this check exposes a real defect.

The input-only test passed on AX102 with strict C compilation and corelimit0:
seven actual prompt token vectors, all50176float32coordinates, matched original
shard94 embedding rows; the historical five tokens matched all35840coordinates.
No layer0/transformer/normalization/output-head computation or vector persistence.
No numerical input implementation defect was demonstrated, so client_input and
the model representation were not changed to force the wrong vector count.
The corrected conclusion is prompt-specific: five ordinary words may span seven
model tokens; the original five-token example is different text. Next work remains
paused at this verified input boundary until the human gives further direction.

## Current Direction: Full Prompt Through Client Output

The human requests running exact text "the captial of france is" through the
packaged pipeline and mapping normalization back to client.c output. Existing
client has only exacttext/IDlookup, not numeric embedding/head functions. Add
owned input/output table readers and separate client_input/client_output functions,
retaining existing exactlookup API/tests. No silent correction of spelling/case,
fixed five-token substitutions, historical activation reuse or older model switch.

Input/output tables use K3SEED1 sixcodec blocks with SHA256 metadata/raw blocks
and compressed CRC. Reuse the tested project C DEFLATE decoder as a copied helper,
implement standard-C SHA256 to retain integrity checks, and compare actual decoded
rows/projections with independent original shard data before model execution.
No third-party runtime. All forward vectors, prompts and output results remain
in memory/console, corelimit0. Tokenization for the exact prompt must be matched
to existing tokenizer semantics, not longest-match or spelling normalization.

Run one next-token prediction via client embeddings -> existing server -> all92
transformers -> existing normalization -> client LMhead/argmax/text. Stage model
data may be loaded layer-by-layer to bound RAM. Preserve existing stage sources,
validation artifacts and dataset payloads. No general generation or speed claim.

## Direction

The human requests client.c with separate input(text) and output(vector)
responsibilities and no shared mutable state. After clarification, the explicit
contract is C only, no Python or third-party libraries, using only the inputs
and outputs datasets: input returns7168values per word, output returns one word
from a7168value vector. Do not substitute a Python tokenizer or silently change
the interface to numeric token IDs.

## Dataset Evidence

Read the existing seed-reader.h and fruit-head.h consumers and the actual relocated
inputs/outputs metadata on AX102. Both binary tables use K3SEED1,163840rows by
7168BF16values,16rows per compressed block. inputs/seed.bin is the embedding
matrix; outputs/fruit.bin is the output-head matrix. The input block example and
output producer use zlib byte-plane encoding. These tables do not contain UTF-8
word spellings or a word-to-row/row-to-word index.

Listed every file in the two allowed directories. blocks.jsonl identifies blocks
and first numeric rows, codecs, byte lengths and checksums. outputs/all-n.tsv
describes coefficient IDs, BF16/float bits, rational/decimal values and counts.
outputs/dictionary.bf16 contains numeric coefficients, not word definitions. The
original token vocabulary/tokenizer is a separate asset outside these datasets.
One stored row corresponds to a model token ID, not necessarily a complete word.

## Outcome and Boundary

The numeric functions are separable, but the requested text-only interface cannot
be completed from only these two numeric datasets. A pure C decompressor would
not provide the missing text mapping. Do not hash words to arbitrary rows, average
token embeddings into a newly invented word vector, return token IDs as words,
or claim input embeddings/output-head rows are inverse functions.

No client.c with fabricated behavior was created, no new dependency installed,
and no input/output dataset or model code changed. Only this context record was
created in the requested code directory. Implementation needs a specified word
mapping dataset/contract compatible with the existing matrices, or an explicit
revision to token-vector semantics and allowed vocabulary access. Existing
tokenization behavior cannot guarantee one vector or output per complete word.
No inference run or compressed-data preparation occurred.

## Existing Text Mapping Located

The human asks whether the mapping code or dataset already exists. Yes; verified
server paths and sizes without tokenization/inference:
- /opt/clover-k3/direct-equation-20261001-a/tiktoken.model:2795286bytes,
  base64 token bytes paired with BPE rank/token ID.
- /root/k3model/tokenizer_config.json:3478bytes, special-token definitions.
- /opt/clover-k3/direct-equation-20261001-a/data/vocabulary.bin:1764931bytes,
  163841little-endian offsets plus token text bytes for the C output decoder.
- /opt/clover-k3/live-generation-20261002-a/tokenizer-definition.py:16145bytes,
  existing reference tokenization definition, not permission to use Python.

Native C loader exists at oss/kimi-k3-in-c/src/tokenizer/k3_tok.h. It reads the
released model/config directly and delegates encode/decode to vendored tok.h,
with json.h for parsing. This needs no Python runtime, but does use third-party
header-only C libraries and cannot silently satisfy the explicit no-library
constraint. The current live-entry-replay/equation.c print_word function already
decodes IDs using vocabulary.bin. Its generator gives uncovered/special IDs
placeholder labels, so use tokenizer_config when exact special-token text matters.

Prior file-name search for "token" did not find k3_tok.h; content search found
the existing implementation. Do not infer code absence from a filename-only
search. The word mapping was missing only from the two allowed dataset directories,
not from the system. No client implementation, dependency installation or dataset
relocation is authorized by this location question. All mappings are token-based,
not one-row-per-arbitrary-whole-word guarantees.

## Text Assets and Exact Lookup Authorized

The human now requests moving the mapping datasets into clover-intelegence/dataset,
tokenizer_config.json into clover-intelegence/configs, and C mapping functions in
client.c. They explicitly selected "New C exact lookup only" rather than using
vendored tokenizer headers. This narrows the implementation to text already
present as a single vocabulary entry; unknown text must report a miss. No BPE,
sentence splitting, Python or third-party C libraries in the implementation.

Move tiktoken.model and vocabulary.bin, then tokenizer_config.json on AX102,
preserving same-inode identity, full SHA256 and old-path symlinks. Use a separate
three-file plan/journal so prior dataset inventories remain historical and intact.
The numeric input/output endpoints are not part of this exact-mapping increment.

Implementation hypothesis: loading the base64 rank table, checking it against the
existing binary text table, and applying explicit added-token text from the config
gives a complete exact lookup for known token spellings. A whole-vocabulary
ID/text round trip, known IDs, special IDs, missing words and malformed-file
controls can falsify it. Use explicit owned context with no mutable globals or
implicit input/output cache; expose binary text lengths so NUL bytes are safe.

## Relocation Passed and Implementation

All three moves passed original inode/size/mtime and complete SHA256 checks:
dataset/tiktoken.model,dataset/vocabulary.bin,configs/tokenizer_config.json.
Original server paths are compatibility aliases. These small text assets and
config were also copied into the workspace for C testing; large numeric tables
remain server-only. Shell tooling only, no Python process or third-party library.

client.c implements tokenizer_open/close, word_to_id and id_to_word with explicit
owned state and length-aware bytes. It checks every one of163584base entries
against the binary vocabulary, loads only configured added-token spellings from
JSON, and rejects undefined reserved IDs rather than returning placeholder labels.
JSON strings support escapes/surrogate pairs. Exact matching does not split text,
normalize case/whitespace or add BOS/EOS. No global mutable tokenizer state. The
numeric input/output vector functions remain a separate unimplemented task.

## First C Checks

Native strict C11 compilation passed with O2/Wall/Wextra/Werror/pedantic. Known
base text/ID and configured special-token lookups passed; unknown phrase rejected
with exit3. First attempts exposed terminal setup issues only: PowerShell HOME
is readonly and case-insensitive; use clientRoot. MinGW GCC needs its bin path
on PATH in this environment. No source fix or warning suppression was needed.

test-client.c now checks all defined vocabulary IDs in both directions, undefined
reserved IDs, binary tokens containing NUL, known numeric IDs, malformed config
and Base64, JSON escapes/surrogates and independent contexts. Crossed vocabulary
files must fail rather than silently produce a different mapping. Next run these
same C controls locally and on AX102; no model execution or vector data needed.

## Verified Outcome

All163600defined IDs passed bidirectional exact mapping on Windows and Linux;
240undefined IDs were rejected, binary tokens preserved and independently loaded
contexts remained separate. Parser/malformed-data controls passed. Strict Linux
build and AddressSanitizer/UndefinedBehaviorSanitizer with leak checks passed.
The original three relocated assets retained full hashes and compatibility links.
No Python execution or third-party runtime/library used by the implementation.

client.c,tests and compiled client are at /opt/clover-k3/clover-intelegence/code;
the same source and small assets exist in the workspace. Numeric input/output
functions and full BPE tokenization are not implemented or claimed. Unknown whole
words/sentences remain misses. Standard C ownership and explicit byte lengths
allow later independent endpoints without a hidden cache. No model inference,
numeric dataset change, commit or push.

## Client Package Relocation Requested

The human requests all code files inside code/client and inputs/outputs inside
its dataset subfolder, producing one self-contained folder. Include the current
mapper's actual vocabulary and tokenizer-config dependencies inside that package
as well; preserve old data/config paths as aliases on AX102. The CLI takes explicit
paths, so no C arithmetic or API change is needed. Exact-lookup-only scope remains.

On AX102, code/client is currently an executable, not a directory. Move the whole
code directory through an exclusive staging name into code/client, preserving
that executable as code/client/client. Then move inputs,outputs,vocabulary and
config by same-filesystem rename. Compare every moved file's full SHA256 and
device/inode/size/mtime. Check active users first; stop on collision or change.
This nests independent client data without moving trunk/root/model datasets.

Local code files and small mapping/config assets follow the new layout. Large
inputs/outputs remain only on AX102; local subfolders document that explicitly.
Validate building/running from inside the new package using only its own paths,
and preserve prior relocation journals as historical records. New move journal
is separate; update affected documentation after the byte-preservation check.

The server code-directory move passed all seven original file identity/hash
checks. The next input move stopped before rename because a shell read helper
overwrote caller path variables through Bash dynamic scope. Made the plan-loop
variables function-local; input payload was not moved by the failed attempt.
Repeat input source/hash validation and the same move without changing the plan.

## Client Package Relocation Passed

All five server groups completed and were journaled: code, inputs, outputs,
vocabulary, config. The unchanged plan verified all 30 original files by full
SHA256 and device/inode/size/mtime after the moves. No internal package symlinks
or staging directory remained; old data/config aliases resolve to the same
payloads, including earlier alias chains. No C source, tests or data were edited.

On Windows the newly created client subdirectory already held location notes,
so the seven original code files were moved individually into it, including
.gitignore and both executables. The three small vocabulary/config dependencies
followed. All 10 files preserved SHA256, size and mtime; old local file paths are
absent rather than duplicated. Inputs/outputs have local location notes only;
their complete original files remain on AX102 and will not be overwritten by
these notes. The full package is /opt/clover-k3/clover-intelegence/code/client.

The relocated local package passed strict C compilation, unchanged tests for
163600 defined IDs/240 undefined IDs and the known CLI lookup using only internal
relative paths. Server move plan and completed journal were downloaded as metadata
only. README/CONTEXT paths are being updated after the immutable preservation
check; their new bytes intentionally differ from the move plan's snapshot.

Growth: Bash read-loop variables must be local to the helper; dynamic scope can
otherwise overwrite caller paths. The unchanged source guards stopped the first
input attempt before rename. The corrected helper passed that same move/check
and all subsequent groups. Packaging changes file ownership paths, not the C
mapping contract or the unimplemented numeric endpoint scope. Final AX102
package-relative compilation/tests and documentation link checks remain next.

## Final Package Checks

AX102 strict C11 builds in /tmp/clover-client-check.x6LUoXmt passed, followed by
all 163600 defined/240 undefined ID tests. The original sanitized test passed
ASan/UBSan with leak detection from the new package. Known forward and reverse
CLI lookups passed; all five earlier input/output/vocabulary/config alias chains
resolve to the same files, and the package has no internal symlinks. Original
relocated executables were not rebuilt or overwritten.

Local verification passed 37 links across nine documents, the complete 30-file
plan and five completed journal entries, and all five matching local C-source/
vocabulary/config hashes against server originals. Editor diagnostics are clean;
reserved clover-one/context.md stays empty. Documentation now describes the full
AX102 package versus the local source/small-asset subset. Local numeric location
notes are not uploaded over original dataset companions. This completes the
requested packaging; no model run, numerical endpoint implementation or git
commit/push was performed.

## Runtime Bin Requested

The human requests a bin directory inside client containing compiled programs,
data, configuration and everything needed at runtime, nowhere else. Keep C source,
test source, README, CONTEXT and build ignore rules outside bin. Move the existing
three Linux executables and complete dataset/configs directories, without
regenerating data or changing the exact-mapping C API. Windows follows the same
layout with its two executables and local small assets/location notes only.

Fresh main source confirms all three runtime file paths are supplied explicitly.
Running the unchanged client/tests from bin with bin-internal relative paths
can disconfirm the packaging hypothesis. Source and test bytes must not change.

Found twelve compatibility aliases, five inside clover-intelegence and seven at
older clover-data/direct-equation/k3model/seed/fruit paths. The human explicitly
selected Remove all 12 aliases after being told older experiment paths stop
working. This supersedes the earlier requirement to retain these specific links,
not any other dataset aliases. Record their exact link text/identity before
unlinking so restoration is possible; never remove resolved payloads. Preserve
all old plans/journals/tests as historical evidence with obsolete path assumptions.

move-client-bin.sh will preflight 27 original runtime files with full SHA256 and
device/inode/size/mtime, active-user checks and absent destination, then move,
verify, run the existing vocabulary tests and remove only the twelve approved
links. Its manifests are separate from earlier relocation snapshots. No model
inference, numeric endpoint implementation, broader cleanup, commit or push.

## Runtime Bin Move Passed

Preflight checked all 27 original AX102 runtime files and all twelve aliases,
with no active users. The five groups (client,test-client,test-client-sanitized,
dataset,configs) moved into bin, and every full SHA256/device/inode/size/mtime
matched afterward. Existing tests passed all 163600 defined ID round trips and
240 undefined ID rejections from bin; the known CLI lookup passed too. Only
after those checks, all twelve recorded aliases were verified and unlinked.
Final verification passed absent old paths, no internal symlinks and all 17
journal entries. Source and test source stayed in the client directory unchanged.

Local executables plus dataset/configs moved as four groups containing eight
files. Every hash, size and mtime matched, old paths were absent, and the same
whole-vocabulary/CLI checks passed from bin. Local numeric folders still contain
location notes only. The three new server inventories were downloaded as metadata;
no large data moved between machines. Old unrelated experiments/copies were not
deleted under the bin-only direction.

Build commands now target bin/client and bin/test-client. Ignore patterns follow
those paths without ignoring runtime vocabulary/config files as a whole. Current
docs no longer promise client compatibility aliases. Runtime metadata README
updates intentionally follow, not precede, the immutable preservation checks.
Historical plans remain frozen; old experiment commands/verifiers depending on
removed links need explicit new paths. No C/API/numeric behavior was changed.

Final bin checks passed: AX102 strict source compilation, existing ASan/UBSan
with leak detection, both known CLI directions and no runtime symlinks. The
source root contains only C source/test source, README, CONTEXT and bin; runtime
executables/dataset/configs are inside bin. Local checks passed 42 documentation
links, eight ignore expectations, 27 runtime records, twelve removed-link records,
17 journal entries and five original C-source/vocabulary/config hashes. Reserved
context remains empty and editor diagnostics are clean. Sync current docs and
ignore rules only; do not upload local numeric location notes over AX102 data.
The requested bin layout and alias retirement are complete; no further runtime
change or reconstruction is needed for this packaging task.

## Embedding To Layer 0: Source Inspection

The human asks whether input-to-vector feeds layer 0 directly. Read the original
clover-k3/clover-k3.c main, rmsnorm and KDA entry. Embedding lookup widens the BF16
row into resid. Apart from timing/output and scratch allocation, execution then
enters the L=0 loop. Layer-static aggregation folds are prepared inside that loop.
For fresh input nsnap=0, so pre-attention aggregation is a copy resid -> hb, not
AR arithmetic. Layer 0 saves the original embedding as snapshot S0, then applies
its S_IN_LN weighted RMSNorm to produce x1b, which feeds KDA Q/K/V projections.
KDA state initialization (or optional prefix-state restoration) is also inside
layer 0. No separate learned input transform or expert routing runs before L=0.

Thus the proposed client/layer boundary is after raw embedding lookup; snapshot
handling and pre-attention normalization belong to layer 0. This is an inspection
of the original reference source, not a new client integration or inference run.