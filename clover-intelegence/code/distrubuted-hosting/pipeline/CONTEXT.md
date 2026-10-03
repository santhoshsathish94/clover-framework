# Packaged Prompt Run

## Location After Hosting Move

All five code packages now reside in code/distrubuted-hosting. The prompt launcher
uses /opt/clover-k3/clover-intelegence/code/distrubuted-hosting as its remote root;
relative sibling imports and existing binaries are unchanged. Dataset links still
reach the top-level central dataset groups. The folder move does not authorize
resuming inference: use explicit check mode for tokenizer-only path validation.
See the project CONTEXT.md for preservation evidence and the local junction
recovery. Historical absolute paths below refer to the earlier layout.

## Original Character-Merging Trace

The human asks to revisit the seven-vector conclusion and inspect the character
combining described in the K3 stages or other documentation. The earlier embedding
test accepted IDs, so it did not itself expose where character merging stops.

Read research/k3/model/k3-stages.md stage 1: its five integers/vectors are explicitly
for "The capital of France is", with recovered IDs1008,10484,318,15383,387. Read
the original third_party/tok.h km_letters, bpe_piece and tok_encode paths. Merging
continues while an adjacent concatenation has a vocabulary rank, not until each
text chunk becomes one token. There is no later whole-word pooling in this input
path.

Added test-only trace-input.c, using those original functions without editing or
copying their algorithms. The diagnostic accepts ASCII letter/space text, compares
every chunk's completed BPE tokens with the whole native tok_encode result, checks
exact text roundtrip, and emits those actual IDs for the unchanged embedding test.
It is not a general tokenizer or a new client runtime dependency.

AX102 exact input "the captial of france is" produced:

```text
"the"     -> 2108="the"
" captial" -> 11989=" capt" 682="ial"
" of"      -> 318=" of"
" france"  -> 2225=" fr" 876="ance"
" is"      -> 387=" is"
5 pre-tokenizer chunks -> 7 final token IDs
```

The two multi-token chunks both have whole-chunk vocabulary ID -1 (absent). The
seven IDs above are after the native BPE loop terminates, not intermediate byte
pieces. Passing stdout IDs directly to client/bin/test-input matched all50176
float32 coordinates against original shard94 embedding rows. No manually supplied
IDs, layer execution, output head or persisted vectors were involved.

First compile failed before execution because the remote checkout path was guessed
as /opt/kimi-k3-in-c. The located original is /root/kimi-k3-in-c. Strict C11 build
with -O2 -Wall -Wextra -Werror -pedantic and that checkout's src/tokenizer and
third_party include directories passed. Core dumps were disabled. SHA256 checks
matched remote k3_tok.h, tok.h and json.h against the local originals inspected.
Editor diagnostics are clean; the existing /bin/ ignore rule covers the probe.

This resolves the chunk-versus-final-token distinction on this original native
input path. The shared browser page currently displays a recorded Japan example,
not execution of this typo prompt. If a different document or implementation is
intended, identify that source before changing model semantics. Full pipeline
remains paused. Original sources, existing tests and runtime behavior are unchanged.

## Paused For Original Input Verification

The latest human direction limits work to input-to-vector and checking original
K3 source. Do not run the full pipeline or output head until redirected. A process
check found no active matching pipeline. Original tok_encode and embedding-loop
inspection confirms one row per token, not one row per whitespace-delimited word.
The exact requested lowercase typo prompt yields seven IDs; the old five-ID
example decodes a differently capitalized, correctly spelled prompt. Both were
cross-checked with the existing native K3 tokenizer in check-only mode.
No prompt normalization or token-vector averaging is approved. Prior full-run
direction below is historical and remains paused.

Input-only original-row comparison passed all50176coordinates for the exact
seven-token prompt and35840for the historical five-token example. No stage was
executed. No tokenizer/vector-count workaround was applied. Preserve the exact
prompt; do not substitute the old five-ID example or average subtoken embeddings.

Run the human's exact "the captial of france is" prompt without correcting case
or spelling. Current client exactlookup remains unchanged; a separate narrowly
scoped ASCII-letter/space rank-BPE launcher obtains IDs using the released split
rules and verifies byte roundtrip. Cross-check with existing native tokenizer as
a test-only oracle before execution. No third-party runtime added to client/stages.

The C coordinator receives explicit IDs, uses client_input for embeddings, loads
the existing server, transformers1..92 and normalization with native shared-library
adapters compiled from unchanged sources, then calls client_output on normalization's
last-position result. Each stage processes every prompt position with its own
sequence state; unload fixed model data after the stage to bound resident RAM.
This is full recomputation for one next token, not autoregressive generation.

No saved activations or reference answers feed the candidate. Intermediate bundles
remain in process memory. Corelimit0, no prompt/input/output/logit files. Only static
source/build identities and test summaries may be retained. Exact answer observed,
not forced to Paris. Existing stage source, data and historical test gates stay intact.