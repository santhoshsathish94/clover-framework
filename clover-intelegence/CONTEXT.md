# Dataset Relocation

## Current: Exact Computed Expert Results Reused In RAM

The human explicitly asks to store computedvalues inRAM rather than onlyexpertIDs.
Standalone nowmemoizesfinal3584floatdownvectors keyedbymodel-ownedcacheinstance,
layer,expertandexact3584inputfloatbytes. Hashfilterplusfullmemcmp; noID-onlyreuse.
Default256MiBbudget withinexistingstartupheadroom; actual235077632B/8192entries.
Bounded8wayLRUsets, zeroconfigdisables, allocationfailurefallsbacktolivecompute.
Cachepersistsacrossrequests, freedatshutdown; noactivationfilesorcrossmodelreuse.

Lookupsoccurbeforeprefetch,copyhitsbeforeeviction; onlymissingranksenter72MiB
doublebufferreader, resultsinsertedafterallgate/up/activation/downcomplete.
Mixstayscurrentweight/originalrankorder. Coveragechecksaccountforliveprojections
plus3*resulthits=4416/position. Observationfilesnotusedbyproduction.

RealAX102Rainfallsfirst2in/2out418,276in65.785s,0hits4416misses; exactrepeat
sameIDs7.902s,4083hits333misses/reads; changedTeatastes1517,15600in50.907s,
0hits4416misses. Coldheadfillcontributestofirsttime,sodonotclaimpurecachefactor.
Cacheunitstrict+ASanUBSanleaksPASS(exactkeys/collision/signedzero/eviction/owner
isolation),configbudget/disable/lowmemoryPASS, oldgenerationcontrolsunchanged.
Originalcanonicaldataidentitiesunchanged. DetailsinstandaloneREADME/CONTEXT.

## Current: Measured Input And Every Layer Stage

The human requested one-stage-at-a-time diagnosis using workflow/stage docs and
the existing website. Read actual workflow and shared /ai/ recorded example,
instrumented numericalcode withcompile-time-onlyhooks, and measurednativeinput
beforelayers. Tokenizerwarmmean0.000371ms,cachedembedding~0.0023ms;initialcompressed
embedding1.60-1.77ms. Allrowsmatchoriginaldata. No layercostattributedtolookup.

standalone-hosting/STAGE-PROFILE.md contains all93layerwalltables andnestedstage/
projection/worker breakdowns fromcomplete279-layercaptures. Finalcontinuation
layer0~28.913ms;totalattention1.018s,router54.333ms,expertmix15.141s,shared314.354ms,
tailnorm0.052ms,head50.909ms. Firstheadfill~15s separatefromwarmcost. Fixedprepared
folds37/38anddecaybase39alreadyreused; furtherseconds-scalegapisinexpertdecode.

TestedlargerHuffmanlookupsnogain(rejected); boundedfour-literalbatchimprovedlocal
decoderprobe. SamewholemodeloutputIDs;correctedcontinuation17.023243to16.687541s,
not5seconds. Originalaftercapturehad16microsecondmaxprofilingboundarycontamination;
fixedtimersandkeptbothrawreports. Capturestreamspreventterminaltruncation, asserts
coverage/reconciliationbeforepublishing. Originaldecoderoracles/rejections/sanitizers
andfullstoredhead/leaf/embedding/layerreaderchecksPASS. No datasetchanges, preserved
originalsources, no128modelrun. See standalonecontextforcompleteevidence/limitations.

## Current: Earlier Reads And Measured Decoder Fix

The human challengeslatencyandasksforn-1prefetchratherthanmorecache. Read historical
gen.py (14readerthreads,prefixreuse,AVX2),Xm,pl_startandnx_begin;five-secondgeneration
wasnotrecorded-output-only. Crosslayernx_beginrequiresknownprompt-keyedroutes;
gen.pydoesnotsetK3_NX. Currenthintafterroutingwasnotequivalentreaderpipeline.

Measuredcurrent16expert/48projectionlayerprobe:~0.456swall,workerread0.53/decode4.14/
CRC1.75/math0.36seconds. OptimizedexistingCdecoder9bitlookup/boundedbitreads/chunked
Adlerandidentical8byteCRC:~0.151swall;allintegritychecksretained. Newpersistentreader
uses72MiBtwobuffers,readsexpertnwhileexpert(n-1)runsafterselection. Zero default
expertcache/WILLNEED, headcacheunchanged. Doesnotpredictunseennextlayerselection.

RealAX102Rainfalls->' on the'72.029s(vsprior156.674);Tea tastes->' like tea'55.714s
(vs121.282). ExactprioroutputIDs,zeroreplay;warmoutputsteps17-19sNOT5s. Firsthead
fill15.264ssubsequent0.051s. Pipelinewait1.669/1.371s/request;~69.77GBlogicalreads
each,72MiBstaging,0expertcacheRAM. Warmone-layerpipelineprobe0.172svsdirect0.162s,
sonoisolatedpipeline-speedupclaim. Noall128runorhistoricalrawmodelrerun.

Unchanged24valid/118invaliddecoderoraclePASSincludingASan/UBSan;newstagedCRC/zlib
corruption/EOF/threadshutdownchecksPASS;fulloriginalheadbyte/score,leaf,retained
layercomparisonsPASS. Canonicaldatasetidentitiesunchanged. Rootcodegenerator
reproducesmeasuredsource,recordsindentation-anchorfix. Bothoriginalsourcesremain.
Storedexpertoutputsareinput-specific;fixedQKVoperatorvaluesareuniversalparameters
notinput-specificQKVcaptures. Detailedcurrentguideexplainsrolesandknownlatencygap.

## Current: Bounded Fixed-Data Caching

The human requests general input support withwiseRAM/diskcache/prefetch, avoiding
repeatedworkonprecomputedvalues. This supersedes the preceding unresolved request
toavoidallliveexpertarithmetic. Liveinput-dependentcalculationscontinue; noanswer
replayornewstoredobservations. Standaloneonlychanges, centraldatasetsunchanged.

HeadcachekeepsverifiedBF16bytes2348810240B, fills once thencomputesfreshscoresin
parallelforeachinput; secondchanged-vector test0.050833svs24.378498sstreaming,
all163840scorebits/argmaxexact. Expertcachekeepsverifiedcompressed-placementdecode
bytesunder16GiBsplitacross92layers, frequency-awareadmission/aging/eviction; original
CRCcheckedbeforepublish. DiskbackingremainsoriginalfilesplusOSpagecache. Bounded
64MiB/layerWILLNEEDhintsfollowactualselectedexperts, notpredictiveanswerreuse.

Configfieldshead_cache_mib2240/expert_cache_mib16384/prefetch_mib64/reserve8192;
startupheadroomsubtractsfixedmappingbytes,reserveand2GiBstateallowance, reducesor
disablescacheswheninsufficient. Lazyallocationswithinbudget; noautomaticadaptation
tolaterexternalmemorypressure. Restartifdatasetfileschange; nohostmemorysettings.

AX102livecheckcacheoffRainfalls->' on the'179.911s; sameinputcacheon156.674s,
identicaloutputIDs418,276. SameenabledprocessTea tastes->' like tea'121.282s;
headnewdecodes0and76000expertblockhits. ExpertRAM17040504960Bplushead2348810240B.
Cold/freshrequeststillcosts; secondinputnotidenticalworkload; combinedoptimization
observationsnotcontrolledspeeduporindividualprefetchproof. Cacheunitexactness,
eviction/budgetsanitizers, generationcontrols, sourcegenerationandbuildallpassed.
Canonicaldatasetentryidentitiesunchanged. Full128-outputmodelrunstilluntested.
See standaloneREADME/CONTEXT forcommands,counters,limitsandcompleteevidence.

## Current: Configurable Live Standalone Generation

The human explicitly approves live expert calculations from existingrootdata and
sets separate128input/128outputlimits. standalone-hosting now accepts JSONlines
input_ids plusoptionalmax_new_tokens, configured bybin/configs/generation.json.
No observation-set label/five-token restriction inproduction. Maxcontext256;
NPOS=1isstepwidth. Each layer retains KDA/MLAstate inRAMacrosstokens and resets
betweenrequests. StopsatEOSoroutputcap; streamsTOKEN_JSONthenDONE_JSON.

Existingrootreader generalizedvia perlayer metadata andcursor-freeparallelread,
samevalue/reduction/CRC. No datasetsreconstructed/regenerated. Fixedparameter
ownershipretained. Fullfive-tokenprefill7360expertinput/outputcomparisonsPASS,
next17374thenlivecontinuation20829; productionexpertmatches0,4416liveprojections
perprocessedtoken. RealJSONonetokenrequests/reset,129input/outputrejectionsPASS.
Controller1..128input/128output/EOS/failuretests+sanitizersPASS, not a full128real
modelworkload. All92rootsload, layer1/2/92liveprojectionsmatchreference. Original
centralpayloadidentitiesremainunchanged; nooriginalsources/testsmodified.

Importantcost: observedAX102outputsteps67-71seconds (about44sexpert,25shead in
the referencecontinuation), not old1.86srecorded-outputtime. No full128outputrun
or newcontrolledbenchmark. Data,attentionstate,input/outputvectorsremainRAMor
consoleonly,corelimit0. See standaloneREADME/CONTEXT foractualscope andcommands.

## Current: Standalone Resident Uses Current Stored Values

The human selects the startup-owned resident implementation and requires
metadata-first, stepwise binding to current datasets without regenerating or
reconstructing them. code/standalone-hosting/clover-one.c now derives from resident
source1e55e1f0794ad62e217803e029e5e2142033a85a6d1d2ef695fb7ac40074e67d,
superseding its earlier exactcopy of main0eb364c. Both original sources remain
unchanged. No distributed package runtime source or dataset payload changed.

One resolved central dataset root supplies prepared trunks/operators, exact expert
observations and eqidx metadata. Existing copied readers now consume inputs/seed.bin,
outputs/fruit.bin and leaves.json directly; old checkpoint paths embedded in eqidx
are not opened by the runtime. Full tensors are not reconstructed; decoding uses
bounded16-row input/head buffers. Final AR/RMS and layer arithmetic are preserved.
Expert outputs remain restricted to exact input/ID matches; not arbitrary inference.

Todo sequence completed metadata/source check, adoptioncompile, startupinspect,
stored-value comparison, failure controls and packaging. Validation passed six
seedrows/all21504leafvalues, everyfruitbyte/all163840headscores against original
data/arithmetic,186retainedlayerbindings/39154876416IDs/21780480vector-tapvalues,
14720existingexpertrecords and372mappingreleases. Exact expert mismatch exits3;
crossedcontainers, badtoken and missingdataset rejected. No model request ran.

Standalonebin now has only compiled programs and a dataset mapping to the real
centralroot; localjunctionmirrorsit but noWindowsresidentport. build/run scripts
do no dataset preparation; run disables core dumps. Original central dataset
identities/size/mtime/ownership rechecked unchanged and no links insidedataset.
No new timing claim: inspect skipped pagewarmup, and streamedcompressedhead differs
from oldrawBF16mmap. See code/standalone-hosting/CONTEXT.md and README.md for details.

## Current: Standalone Repository Source Copied

The human requests code/standalone-hosting/clover-one.c as a COPY of the main
repository clover-k3/clover-k3.c, not the resident experiment and not a move.
Created locally and onAX102; both fullSHA256hashes match the unchanged original:
5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65.
The source matches0eb364c12cade5697090688d56c8e224c80501ec. No dataset adaptation,
build, new runtime links or inference occurred. Original source and all dataset
paths remain untouched. code now has distrubuted-hosting and standalone-hosting;
self-hosting was never created. See code/standalone-hosting/CONTEXT.md for scope.
This supersedes the earlier pending source-location decision, not the unresolved
original-checkpoint versus prepared-dataset compatibility question.

## Current: Original Dataset Paths Must Be Real

The human corrects the mapping and explicitly selects real dataset/trunk-N,
dataset/root-N and original operator/client paths, with mappings only in code
bins. This revises the package-grouped central layout. Initial audit found all95
bin links correct for that older layout, but AX102 retained279aliases inside
dataset pointing to package-grouped real directories. Local dataset had no links.
Do not present package-grouped storage as satisfying the clarified requirement.

restore-dataset-paths.sh/.ps1 freeze separate DATASET-ORIGINAL-PATHS-AX102/WINDOWS
plans and journals. AX102283moves replace279exact destination aliases with their
own targets and move4client entries to the root. Local284moves include three
server manifest files placed at canonical trunk/operator paths. Directory renames
preserve payload identities; remote files<=1MiB getSHA256, every local file is
hashed. Larger AX102payloads use same-device/inode/size/mtime checks, not a full
rehash. Code, datasets not being moved and external model aliases are frozen too.

The95packages keep their existing bin/dataset paths:94link to the central dataset
root; server has a mapping-only directory with trunk-0 and trunk-0-qkv links to
the canonical trunk and operators/trunk-0-qkv directories. Linux symlinks are
relative, Windows junctions absolute. No central dataset links should remain.
Historical package READMEcompanions stay real files centrally; no data is deleted,
reconstructed or downloaded. Retired external client aliases and duplicateQKV0
remain retired. Self-hosting and full inference remain paused.

Both no-move plans passed. First normalization move and unchanged --inspect loader
passed locally and onAX102: leaves.json is real at dataset root, only bin mapping
remains. Continue remaining packages with original gates and per-unit journaling.

Completed all283AX102and284local planned moves across95packages. Original unit
identities, small remote hashes, every local hash/timestamp and unchanged code
checks passed. The local moved payload/metadata subset is378files/7,602,191bytes;
remaining package documentation companions stayed where they were. AX102 retains
3500real dataset files including current documentation/inventories, unchanged in
count. No real data files are inside code bins. Both dataset trees contain ZERO
links. There are96mapping links/junctions for95packages:94dataset-root links and
two server mappings inside its real mapping-only bin/dataset directory.

Existing clover-data aliases keep their exact link text and resolved identities.
The279former central aliases are now their original real target files/directories,
not removed datasets. Restored central inputs/outputs/vocabulary paths are real
files, not recreated retired external client aliases. Configuration remains inbin;
the retired duplicate qkv-all/layer-0 is still absent. No payload regeneration,
large download, source change, rebuild, inference or persisted activation occurred.

AX102 client exactlookup and all94stage --inspect loaders passed through bins.
Local client and normalization --inspect passed; the other local payloads remain
metadata-only. All local code and binaries stayed unchanged. Relocation records
are preserved separately; archive SHA256 is
0feefe24ee24f1191e2f875089bc45ed7a0ab6cff8ce6407086886a3700fd32f.
Documentation updates follow the frozen verification. A later verify will detect
the documented changes to dataset/README.md, which was part of its frozen
unchanged-file snapshot; do not weaken the original gate to conceal this.

Growth: the intended contract was original dataset paths as real entries, not
merely real files somewhere below a package-grouped dataset tree. Inspect entry
types at the paths the human means, as well as bin resolution. Preserve old
experiments as history without reintroducing their central compatibility aliases.

## Pending: Repository Version For Self-Hosting

The human first selected the resident version associated with5.819640934sstartup,
then stopped that move and requested comparing repository clover-k3.c history to
find the best version. No self-hosting folder, source move, mapping change, build
or model execution has occurred. Resident and main reference sources are intact.

Inspected all available local Git refs with log --all --follow for
clover-k3/clover-k3.c: six revisions, latest0eb364c12cade5697090688d56c8e224c80501ec.
The current working source has no diff against that revision. Compared actual
source diff from f4f7844through0eb364c and the measurement records. Recommendation
is0eb364c, retaining cached-route lookahead and adding optional prefix reuse.
The a74dfd0stage-preload experiment remains off by default; its enabled modes
showed no gain or a slowdown, so it is not a reason to choose an older source.

Recorded evidence, not rerun: f4f7844lookahead8.75to7.25s; a74dfd0off7.25/7.25/7.27s
versusL3 7.27/7.26/7.29s and slowerL1;0eb364c campaign34/34first-token agreement,
0/34logit differences between lookahead arms, total445.97to394.94s; France7.23s.
Separate six-token prefix-reuse experiment35.6svs83.2s, identical selected tokens,
5.93s/token average, about5.1s steady decode on its recorded five-token prompt.
Measurements are scoped to the recorded Ryzen7950X3D/AX102configuration; warm,
cached-route and prefix-reuse measurements are not cold or unrestricted timing
guarantees. The evidence-methods document explicitly records incomplete raw-artifact
bundling, so do not claim a fully verified commit-to-every-run provenance bundle.

Dataset compatibility remains unresolved before self-hosting execution. Main source
uses original packed trunk and raw expert tensors; prepared trunk/root/operator
files are different formats. Read-only AX102check found trunk.bin54468222976B and
trunk.json353643B under /root/k3trunk_i8; only4safetensors at /root/k3model and its
/srv/k3/model location, and1under central dataset/model. This is a location inventory,
not a complete audit of index-referenced file availability. Do not silently copy
or reconstruct retired weights, replace expert calculations with recorded outputs,
enable persistent prefix/route dumps, or claim central mappings alone solve it.
Next decision is source-only relocation versus an explicitly scoped reader/data
compatibility change. Original no-inference/no-persisted-activation boundaries hold.

## Current: Code Nested Under distrubuted-hosting

2026-10-02: the human requests code/distrubuted-hosting and all current code moved
inside it. Preserve that exact spelling. The five packages client, normalization,
pipeline, server and tansformers now live there, locally and on AX102. code has
only this one child. Datasets remain under the existing top-level dataset groups;
configuration and binaries moved with their code packages. Full inference remains
paused. No new hosting/network functionality is implied by this folder name.

move-hosting.sh/.ps1 and HOSTING-MOVE-AX102/WINDOWS inventories record the move.
AX102 hashed all1325regular code files, checked active readers, moved five packages
by same-filesystem rename and verified allfilehashes/identities. Its95relative
bin/dataset links gained one parent level;279existing dataset aliases referencing
the old code paths now point directly to the same central real datasets. No old
package compatibility directories are left under code. Older external alias chains
continue resolving to the same recorded directory/file identities.

Local preflight hashed1486code files and recorded95absolute junctions. The first
PowerShell Move-Item execution moved the packages but traversed92transformer
junctions:462metadata files temporarily left central storage for real bin/dataset
directories. The original final verification failed, correctly. All code/data
hashes still matched, with no missing data across the inspected locations. The
three other junctions remained intact. This was a transient packaging failure,
not an unchanged-link move; do not erase it from the outcome.

Recovery used the frozen original central-data inventory: require empty central
targets, hash every source file, return exact files with System.IO.File.Move,
remove only resulting empty directories, recreate92junctions and restore recorded
directory timestamps. The unchanged hosting inventory comparison then passed all
1486code hashes and95junctions. Future directory moves use System.IO.Directory.Move
instead of PowerShell Move-Item to avoid traversal. No test/acceptance data was
changed to make verification pass. Large numeric payloads never left AX102.

After the byte-preservation check, pipeline/run-prompt.mjs changes only its remote
code root to the new hosting folder. Sibling-relative package paths still work.
Historical generators, tests and relocation plans are not rewritten: commands
with hard-coded old absolute roots require explicit rebasing before rerunning.
Current layout docs supersede older code/client and code/tansformers locations.
Move snapshots intentionally precede the launcher and documentation edits.

Final checks: all472local central-data file hashes and original directory/file
timestamps match the frozen centralization plan after recovery. Relocated local
client lookup and normalization --inspect pass. AX102 moved client lookup and
all94stage --inspect loaders pass. The updated launcher finds its relocated native
tokenizer in explicit check mode; no pipeline/layer/head inference ran. Twenty
rebased top-level Markdown links and95documented relative dataset paths resolve.
Editor diagnostics are clean. Current docs and launcher are synchronized separately
after move verification; original historical inventories remain immutable.

## Current: Central Datasets With Bin Links

2026-10-02: the human explicitly requests moving all package datasets into the
top-level dataset folder and retaining links in each code bin. This supersedes
the earlier physical-bin-only requirement. Scope is the95existing package dataset
directories on AX102 and their local counterparts; unrelated experiments and
retired client aliases are not restored. Configuration files and executables stay
in bin. Existing shared eqidx/model data and old model alias chains stay intact.

Destinations preserve ownership: dataset/client, dataset/server,
dataset/normalization and dataset/tansformers/transformer-N for N1..92.
Each former code/.../bin/dataset becomes a relative symlink on Linux and a
directory junction on Windows. No source/API changes or model inference are needed.

centralize-datasets.sh and centralize-datasets.ps1 record separate immutable
DATASET-CENTRALIZATION-AX102 and DATASET-CENTRALIZATION-WINDOWS inventories and
journals. Remote verification retains every directory/file device,inode,size,
nanosecondmtime,mode/ownership and SHA256 for files at most1MiB; large payloads use
same-filesystem rename and identity checks, not a new1.388TBrehash. Local data is
small: every file is hashed. Both preserve code/binary metadata independently.

Preflight found95real datasets, no nested links, no destination collisions and no
matching running model process. AX102 device2306is shared by source/destination.
No node executable was on the remote PATH; use Bash there, PowerShell locally.
The first reader check stopped before any inventory/move: this AX102 fuser rejects
the -- separator. A direct filename check distinguished that behavior; absolute
filenames without -- preserve the same active-reader gate. Both plans then passed.

First bounded move: normalization passed on both machines, including contents and
runtime links. AX102 existing model aliases still resolve to original identities.
Continue the remaining planned directories with per-unit verification; stop on
any change or occupied target. Full model pipeline remains paused.

Completed: all95moves passed on each machine. AX102 moved3491files totaling
1,383,089,110,645bytes, including current dataset companions/documentation;
2561files at most1MiB were SHA256-checked, all files/directories retained recorded
identities. Windows moved472files totaling7,631,695bytes with every SHA256 and
creation/modification timestamp preserved. These live relocation counts include
metadata added after the older3399-file logical model inventory.

Final relocation verifiers passed all95bin links, frozen payload inventories and
unchanged code/binary metadata. AX102 old model aliases and existing shared files
retained original identities. Linux links are relative; Windows junctions are
absolute and must be recreated if the checkout moves. No large data was copied,
downloaded, regenerated or deleted. Twelve retired client aliases and the deleted
duplicateQKV path remain retired. Configuration files remain in client/bin/configs.

Runtime checks: AX102 client lookup returned2108for "the"; server, normalization
and all92transformer --inspect loaders passed through the unchanged bin paths.
Local client and normalization loaded through their junctions. Other local loaders
cannot load absent large datasets; their metadata and junctions were verified.
No model layer arithmetic, output projection or inference ran. Current root and
dataset documentation is updated only after the frozen move checks; older package
notes/journals remain historical where they claim real payloads live in bin.
The frozen shared-before snapshot includes dataset/README.md, so a later whole
verify detects this documented post-move README edit; it must not be weakened.

Growth: preserve whole dataset directory ownership under the central root, then
link the original runtime directory once. This retained every existing consumer
path and avoided reversing old per-layer aliases into loops. Future packaging
starts from the central paths above, not the older physical-bin-only records.

## Current: Final Normalization Implemented

The human requested stage93 called normalization after asking about eqidx,
leaves and shard94. Original layers are0..92;93is a diagnostic tail label, not
another transformer. eqidx is shared whole-model offset metadata. Shard94 contains
embedding/LM-head matrices and3globaltail tensors. Existing leaves.json stores
the latter exactly, so code/normalization depends only on that small file.

Separate standard-C/math functions load/decode the schema, prepare fixed fold,
score8snapshots+residual, softmax, ordered aggregate and final RMSNorm. Input is
the chosen position's layer92residual plusS0..S84; output7168normalizedfloats.
No LM-head/argmax/tokenization/network/client integration. Candidate runtime opens
no index/checkpoint. Live aggregation still computes; fixedleafvalues reused.

Both5tokenFrance/Japan reference cases matched allfive normalized positions,
71680floats total. All21504leafparameters independently matched shard94 rawbytes.
Reference original layer arithmetic uses exact-input-bound historical expert
observations test-only, as earlier tests; not freshretiredcheckpoint or pipeline
validation. StandaloneASan/UBSan/leak and15schema rejection controls passed,
as did deployedCLI framing/2bundles/EOF/malformed/incomplete/nonfinite checks.
Corelimit0 and no new runtimevectors. Initial reference generator CRLFanchor
failure fixed after rawsourcehash verification, no numericalcriteria changed.

leaves.json moved unchanged into code/normalization/bin/dataset with fullSHA256/
inode/size/mtime and botholdaliases verified. eqidx/shardidentities untouched.
Existing client/server/transformer sources not modified. Local smallleafcopy,
documentation and strict Windows executable build are complete; the same local
CLI controls passed. Three leaf payload hashes and both reference-result records
reconciled independently; helper/reference reproduction, documentation links and
reserved context checks passed. Runtime normalization requires only leaves.json;
the shared index/shard remain main-dataset assets. No commit/push.

## Current: Redundant Layer-0 QKV Copy Removed

The human approved deleting main-dataset operators/qkv-all/layer-0 to remove
confusion and explicitly selected removal of its one legacy clover-data alias.
Both payloads were265008216bytes with identical SHA256
4925ea619a4fd268b441e8008e4c1cd714f681b328b03cccd52c69b6a9b6f654 and full cmp
agreement. No active readers detected. Preserved the duplicate's distinct1863-byte
manifest by same-filesystem move to
code/server/bin/dataset/trunk-0-qkv/qkv-all-layer-0-manifest.json before unlinking
only the redundant payload, removing its empty directory and unlinking its alias.

Canonical qkv.bin and original manifest stayed byte/identity unchanged. Both
retired paths verified absent and unchanged server source/binary passed --inspect.
All other aliases and payloads retained. Updated logical inventory3399files/
1,387,808,622,300bytes (excludes generated documentation/relocation records).
QKV-DEDUP-PLAN/JOURNAL.tsv in server document the operation; old inventories are
preserved historically, not rewritten to hide the removed path. No local numeric
payload existed; only small audit/provenance copies were downloaded. No rebuild,
new inference, commit or push.

## Current: Transformers 2-92 Complete

The human requested repetition of the layer1 implementation/package process for
every remaining layer. All91function-oriented C variants now exist, with67KDA and
24MLA implementations, per-layer root constants/maps/templates and correct1..8
snapshot bundle handling. Main coordinates separate loading, arithmetic and I/O
functions. Standard C/math and platform64bitfileIO only; no third-party runtime.
Computed parameter values are reused directly, while input-dependent projections
and activations still execute. No saved expert-output fallback in candidates.

All182independent layer/case comparisons (France/Japan,5positions) passed exact
residuals,16routes and snapshot outputs. Original layer arithmetic uses historical
expert observations only in the test oracle with complete exact input matching;
retired original expert tensors were not reconstructed. Representative2/3/12/92
ASan/UBSan/leak and packagedCLI framing/error checks passed, as did strict builds
and real dataset loads for all91. Not a chained standalone pipeline validation,
new end-to-end generation result or speed benchmark.

All273dependency directory moves completed to
code/tansformers/transformer-N/bin/dataset forN2..92:3,325files and
1,363,192,610,753bytes, unchanged fullSHA256/file/directory identities. Old model
alias chains retained and final-audited; removed client aliases not restored.
No payloads downloaded locally. Per-layer plans/journals/verification.json and
remaining/results.json preserve evidence; archive SHA256
a2ddb36492b805f33842fe565b7cd095ed941ce34f47dbf66f313fcf81016e04 verified locally.
Existing client/server/transformer1 code, data, tests and prior journals untouched.
No new input/activation/output files, core dumps, commit or push. See
[campaign context](code/tansformers/remaining/CONTEXT.md) for the complete record.

## Direction

2026-10-02: create clover-intelegence/dataset; identify dependencies of the current
resident clover-k3.c first, then move datasets one at a time. Human explicitly
approved AX102 destination /opt/clover-k3/clover-intelegence/dataset with old-path
compatibility links and a matching local inventory folder. No large payload
download, regeneration, unrelated deletion, commit or push.

## Inspected Scope

Verified the current resident source/binary hashes and its actual open paths.
Runtime dependencies: prepared trunk0-92groups/indexes, derived trunk0-QKV and
all-layer operator directories, France/Japan expert input/result observations for
roots1-92, binary model index, and its one global tensor shard. Adjacent existing
manifests and observation metadata travel with their dataset directories. Uncalled
legacy QKV/shared-result readers do not establish additional active dependencies.

Preflight found373units containing81,850,713,865bytes, all on device2306. No current
process had dependency files open or mapped. The global tensor path is
/srv/k3/model/model-00094-of-000096.safetensors. Its five global records supply
embedding, tail and head. Other root weight payloads and the original trunk are
not read by this resident and remain outside this relocation.

## Local Hypothesis and Check

Same-filesystem rename preserves payload inode/size/mtime and content; creating an
old-path symlink preserves existing consumers and paths embedded in eqidx.bin.
relocate.py inventories every file, refuses changed sources or existing targets,
rechecks process/campaign use, renames a single unit, verifies identities through
both paths, and durably journals that unit before the next. On link/validation
failure it restores that just-moved unit where possible and stops. Never copy or
delete unrelated data. Initial check is script syntax/plan with zero payload moves,
then move the index alone and parse its bindings before touching larger groups.

Compatibility links intentionally remain; the unchanged index is still machine-
specific, not a portable archive. New startup configuration will use the new
root directly while old programs continue resolving the aliases. No campaign
process is paused or signalled. Final verification includes resident startup and
bounded existing-input checks using the new root, with no inference files saved.

## First Moves Verified

The relocation plan passed for all373units with no active dataset users or running
prepared-trunk campaign detected. Index moved first, trunk0second, then trunks1-92
serially. Every payload/metadata file retains its original device,inode,size,mtime;
all old paths resolve to the new files. The server journal records each successful
unit separately. No contents were copied or recalculated. Continue with operators,
then existing expert observations and the global tensor file.

## All Moves Verified

Completed all373units serially: index,93prepared trunks,94derived operator
directories,184existing expert observation directories and one global tensor
file. Final verifier checked2328files/81,850,713,865bytes against original
device,inode,size,mtime identities. Every old path resolves to its corresponding
new payload; the unchanged index's global tensor paths resolve into dataset/model.
No active readers were detected before the move groups. No process was signalled.

run-resident.sh binds index,prepared data and operators directly to the new
dataset root. It runs the existing resident binary unchanged and clears inherited
diagnostic/replay configuration. The global tensor still resolves through the
approved old-path alias embedded in the unchanged index. This is a same-machine
relocation, not a portable rewritten index. Next check launcher syntax and actual
resident requests from the new root; no new inference output files.

## Runtime and Handoff Outcome

The new launcher passed bash syntax checking. The unchanged resident source and
binary hashes matched their recorded build. Actual /proc/PID/maps inspection
confirmed all372prepared/operator payload mappings and the global tensor shard
at new paths. Existing France/Japan reference checks passed93layers/651records/
7360expert matches with zero fixed mapping opens/releases and zero expert
projections. Startup4.765682158s, requests2.479453151s/2.217874347s,
shutdown3.118694412s. These are smoke-test observations, not a speedup claim.

All moved file identities and compatibility links passed again after execution.
No new inference files; process exited. MOVE-PLAN.json, MOVE-JOURNAL.jsonl and
RELOCATION.json were downloaded into the local dataset folder, metadata only.
The full payload remains AX102; local README states this explicitly. Old absolute
references remain functional without editing historical manifests or binaries.

Growth: moving real datasets need not invalidate their consumers or copy tens
of GB. Trace actual opens, parse embedded paths, check shared use, then use a
same-filesystem rename with per-unit verification and compatibility links. This
only establishes the current resident dependency set, not all experimental
programs' datasets. Keep links until an explicit migration retires old consumers.

## Complete Roots and Tables Requested

The human now explicitly adds complete root-1throughroot-92, then inputs, outputs
and leaves.json to the same destination. This expands the earlier resident-only
inventory. Preserve old paths as compatibility links, no new payloads or inference.

Fresh read-only preflight:92original real root directories contain1049remaining
files totaling1,302,891,439,262bytes. Destination roots currently contain only the
already-moved France/Japan observations. Each old root has two observation symlinks
pointing at that destination. inputs contains7files/1,658,872,001bytes; outputs
13files/1,667,986,556bytes; leaves.json58,615bytes. All on device2306; no active
readers or protected prepared-trunk campaign detected. No other files requested.

Hypothesis: stage each partial destination root, rename the original full root
into its place, then transplant its existing real observations in place of the
two exact old aliases. Finally alias the old whole-root path to the new root.
This preserves original root and payload inodes, preserves observations, and
avoids self-referencing aliases after consolidation. Verify the merged inventory
and every old path before moving the next root. Temporary staging contains only
this root transaction; ordinary failures roll it back. An interrupted transaction
must be inspected rather than overwritten. Inputs/outputs/leaves then use simple
same-filesystem renames, in that order. No TB-scale copying or content regeneration.

relocate-roots.py records a separate plan/journal, preserving the first relocation
evidence. Initial check is syntax and zero-payload-move planning, then root1alone
as the discriminating merge check. Verify original observations through the
whole-root alias rather than requiring a redundant symlink at each child.

## First Complete Root Passed

Extension plan passed95units/1070additional files/1,306,218,356,434additional
bytes, with1488previous observation files to preserve. Root1alone was consolidated
and verified:15new files,16previous observation files; original root inode and
all payload/observation identities preserved. The original source path is now a
whole-root compatibility link; its observation paths resolve through that parent.

The original relocate.py verifier now accepts a symlink in a source ancestor,
while still requiring the exact expected target and every prior file identity.
This supports the new whole-root aliases without changing the old recorded
inventories. Run its full original verification before moving root2through92.

## All Complete Roots Passed

The original373unit relocation verifier passed after root1consolidation. Roots2-92
then completed sequentially, each checked and journaled before the next. All92
computed-root directories now contain their original remaining payloads and the
previously moved observations.1049additional files/1,302,891,439,262bytes moved;
all1488observation files preserved. Old root paths are whole-root aliases. No
self-links, payload copies, active process signals or regeneration. Proceed next
with inputs, then outputs, then leaves.json as requested.

## Root Extension Completed

Moved inputs,outputs,leaves.json in that order after all92roots. Their original
paths now alias the new locations. Final extension verification passed all95
items,1070additional files/1,306,218,356,434bytes, with1488previous observation
files unchanged. Combined current dataset3398files/1,388,069,070,299bytes, excluding
relocation journals/docs. Root directories themselves retain source identities.

All92root experts.bin headers checked as K3MAPS01through new paths; old/new payload
paths refer to the same inodes. leaves.json parses. No staging directory or nested
self-link remains. No model run, dataset reconstruction or full TB-scale payload
rehash: verified same-filesystem rename,all file identities,actual read access
and previous relocation invariants. No service/campaign signals or source changes.

ROOTS-MOVE-PLAN.json,ROOTS-MOVE-JOURNAL.jsonl,ROOTS-RELOCATION.json copied to the
local dataset folder as metadata only. Old evidence remains intact; README now
distinguishes the initial resident subset from this expanded user-requested scope.
The old relocate.py validator accepts whole-root parent aliases with the same
exact-target and file checks. Computed roots, inputs, outputs and leaves are now
inside clover-intelegence/dataset; original packed trunk and branch datasets remain
untouched because they were not requested. No commit or push.

## Token Vocabulary and Config Relocated

The human requested mapping datasets moved into dataset, a configs folder for
tokenizer_config.json and new mapping code in code/client.c. They explicitly
chose new C exact lookup only, no third-party code or Python, rather than complete
BPE sentence tokenization. move-tokenizer.sh checked same-device identity, inactive
files and full hashes, then relocated tiktoken.model,vocabulary.bin,config in order.
All three original paths are compatibility links; complete content hashes match.
TOKENIZER-MOVE-PLAN.tsv and TOKENIZER-MOVE-JOURNAL.tsv preserve the three moves.

Dataset now includes3400files/1,388,073,630,516bytes excluding relocation/docs;
config3478bytes is outside that dataset total. Small vocabulary/config files
were copied locally for native C testing, not the large numeric datasets.
C exact lookup/decoding tested all163600defined and240undefined IDs, independent
contexts and malformed data. Windows/Linux strict compile and Linux sanitizer
checks passed. Source and docs in code; no numeric input/output functions or
full tokenizer were silently substituted. No model execution or data regeneration.

## Self-Contained Client Package

2026-10-02: the human requests code/client with its own dataset/inputs and
dataset/outputs. Included the mapper's vocabulary/config dependencies in that
folder too. AX102 code/client was an executable, so the entire code directory
was staged then nested, preserving the executable as code/client/client.
Inputs, outputs, vocabulary and config followed by rename plus old-path aliases.
The first input move stopped before rename due to helper variables overwriting
caller paths; localizing the Bash plan-loop variables fixed it without changing
the plan, payloads or validation criteria.

All 30 original server files passed complete SHA256 and device/inode/size/mtime
verification; all five groups have completed CLIENT-MOVE-JOURNAL entries and no
staging/internal symlinks remain. The old clover-data and tokenizer paths still
resolve through compatibility aliases. Trunk/root/model datasets were not moved
again. Logical data totals stay unchanged; physical client data now resides
inside code/client/dataset rather than the main dataset directory.

The local seven code files and three small assets moved with all hashes, sizes
and mtimes preserved. Complete numeric input/output payloads remain AX102-only;
local subfolders hold explicit location notes. Whole-vocabulary tests and strict
compilation passed from the local package using internal relative paths. No C
behavior changes, numeric functions, data regeneration, inference or git actions.
Documentation updates follow the frozen move verification; original inventories
and scripts keep historical path assumptions. Final server package execution and
current documentation links are the remaining checks.

Final AX102 package-only strict builds, complete vocabulary tests, existing
ASan/UBSan/leak checks and both known CLI directions passed. All five original
alias chains reach the same files; no internal package symlinks exist. Builds
were isolated under /tmp, preserving the relocated binaries. Local documentation
checks passed all 37 links in nine files, the 30-file plan, five-stage journal,
five unchanged source/data hashes and reserved empty context. Editor diagnostics
are clean. Updated package/parent docs are synced separately from numeric data;
local location notes do not replace original AX102 dataset files. Packaging is
complete, with the full numeric package on AX102 and only small assets locally.

## Client Runtime Exclusively Inside Bin

The next human direction moves compiled client programs and all client runtime
data/config into code/client/bin, nowhere else. After clarification the human
explicitly selected removal of all twelve existing client aliases, including
seven external legacy links, accepting that older experiment paths stop working.
This revises only those specific compatibility requirements, not other model
aliases or permission to delete unrelated experiment copies. C source/tests and
development docs stay directly under code/client.

move-client-bin.sh preflight recorded 27 runtime files and twelve exact link
identities/text/resolved targets. Same-filesystem moves passed all full hashes
and device/inode/size/mtime checks; existing whole-vocabulary tests and CLI passed
from bin. Then only the twelve approved symlinks were unlinked. Final check:
no old runtime paths or internal bin symlinks, 17 completed journal rows, C source
retained. All eight local runtime files moved with unchanged hashes/size/mtime
and local bin-only tests passed. Large numeric payloads remain AX102-only.

CLIENT-BIN-MOVE-PLAN.tsv, CLIENT-BIN-ALIASES.tsv and CLIENT-BIN-MOVE-JOURNAL.tsv
are separate records, downloaded locally. Prior snapshots are historical and
unchanged; their removed-alias assumptions no longer hold. Build/README/ignore
paths now use bin. Documentation edits occur after preservation checks and are
not changes to numeric payloads. No inference, C behavior, tests, original model
data, other aliases, git commit or push changed. Finish with bin-relative CLI
and sanitizer checks, source/asset hashes, current links and documentation sync.

Final verification passed strict source compilation, existing AX102 sanitized
whole-vocabulary tests with leak detection and both bin-only CLI directions.
There are no internal runtime symlinks or runtime siblings outside bin in client.
Local validation passed 42 current links, eight ignore checks, all relocation
record counts and five original source/asset hashes. Editor checks are clean;
reserved clover-one/context.md remains empty. Only current docs/ignore rules are
synced afterward, not local numeric notes or changed data. Old folder-tree and
relocation snapshots retain their observation-time layouts, not current paths.

## Layer 0 Server Implemented

The human requested code/server/server.c, individual functions, no libraries,
and the trunk-0-qkv operator folder in server/dataset. After a scope question,
they pointed to dataset/trunk-0 and asked to exclude common all-layer work that
does not execute at layer 0. Real code and both manifests show no MLA/MoE/router/
shared experts or pre-attention AR at layer 0, but KDA beta/decay/output and dense
MLP execute. All their constants already exist in prepared trunk-0. No missing
data creation is needed. Standard C/math only, Linux -lm, no third-party runtime.

server.c loads QKV plus thirteen active prepared records, with independent
model/sequence state. Functions split normalization/projection/convolution/history,
activation/L2/decay/KDA state/output, two-source aggregation and dense/residual.
Raw 7168-float inputs produce layer-0 residual and original S0; sequence state
stays in RAM. CLI is stdin/stdout, no network protocol or client embedding hookup.
Prepared trunk-0 stays in its existing location, passed explicitly.

First numerical comparison caught wrong postnorm binding (ARP slot 1 instead of
POST_LN slot 5); identical vector shape had passed structural checks. Corrected
actual semantic role, unchanged reference assertions then passed both existing
five-token France/Japan layer-0 residuals and snapshots, independent/reset states
and invalid inputs. Test reference code is hash-pinned and stops before tail.
Standalone ASan/UBSan/leak and real CLI framing/error checks passed. No later-layer
model run, new data capture or performance comparison. Reference test initially
reported core handling; no work-directory core file found, test core limit now0.

QKV folder moved to code/server/dataset/trunk-0-qkv with both file SHA256 and
inode/size/mtime unchanged. Old operator alias chains preserved; twelve removed
client aliases not restored. QKV move plan/journal and manifest copied locally
as metadata only, no large payload downloads. Original datasets/code/tests and
historical inventories unchanged. See server/CONTEXT for detailed outcome/limits.

Final layer-0 server tests also passed with portable (non-native) compiler flags
and the relocated data paths. Windows strict source compilation and all50current
documentation links passed. Crash correction: the first test abort had created
an apport report containing process memory in /var/crash, missed by the initial
work-directory-only check. Verified that exact artifact's executable identity and
removed only it, with absence confirmed and no copy retained. All later tests had
core limit0; launch docs now state that boundary explicitly. No host crash setting
changed. Server context records the mistake and correction instead of claiming
that no temporary memory artifact was ever created.

## Complete Trunk 0 Moved Into Server Dataset

The human subsequently requested trunk-0 inside code/server/dataset. Moved the
complete six-file directory there on AX102 by same-filesystem rename, retaining
old main-dataset and clover-data paths as aliases. Full SHA256/device/inode/size/
mtime plus directory identity verified unchanged. The server's source and binary
hashes remained unchanged and --inspect loaded both QKV and prepared trunk data
from its own dataset directory. No new numerical run or dataset preparation.

Server TRUNK0-MOVE-PLAN/JOURNAL.tsv and the unchanged manifest were downloaded
as metadata only. Updated current README paths; historical inventories remain
unchanged. This supersedes the earlier external trunk-0 dependency, not the
reference test's need for original raw model data. No client aliases restored,
other data moved, commit or push.

## Server Bin Layout Matched To Client

The human corrected the server packaging: executables were in bin but datasets
were siblings. The complete nine-file server/dataset now lives under server/bin,
with all full hashes and file/directory identities preserved before note updates.
The two existing model aliases point directly to the new directories; no new
server/dataset alias remains. Client alias retirements are unchanged.

The unmodified server loaded both datasets from inside bin via --inspect. Source
and binary hashes matched; no rebuild or new numerical run. Local three-file
metadata directory moved similarly; large data remains AX102-only. Current build/
run/docs/ignore paths updated; BIN-DATASET-PLAN/JOURNAL.tsv record this correction.
Older snapshots are historical. No other dataset moves, commit or push.

## Transformer 1 Implemented And Packaged

The human requested code/tansformers/transformer-1 with source and a bin runtime
holding trunk-1, root-1 and operators/qkv-all/layer-1. They explicitly approved
writing the compressed-placement decoder in C after clarifying that it does not
replace direct use of precomputed values, then required individual functions for
each action rather than one main. Implemented that exact path/spelling and scope.

Standalone transformer-1.c uses nineteen active prepared trunk records and the
layer1 KDA operator. root.h loads the original66BF16constants/maps/templates and
small index; decode.h implements bounded zlib/DEFLATE in C. Each block validates
framing/Adler and original decoded-value CRC, then live expert projections use
stored values directly. No original scale arrays/full matrices are reconstructed,
no observation replay in candidate, no third-party runtime/OpenMP. Standard math
and platform64bitfileIO only. Model fixed data and per-sequence state are owned;
calls sharing the model's file cursor are serial. Main is CLI coordination only.

Both existing five-token sequences passed all layer1 residual/S0 values and16routes
against original layer arithmetic with authenticated exact-input-bound historical
expert observations as TEST-ONLY oracle (original expert tensors were retired).
Reset/invalid controls, standalone ASan/UBSan/leak tests, realCLI framing/errors,
27distributed realblockCRC tests and full expert0zero projections passed. Decoder
24valid/118negative oracle cases passed Windows and Linux sanitizers. Batched the
test transport to avoid repeatedSSH overhead and require exact rejection statuses.
Core dumps disabled, no persisted new runtime input/activation/output datasets.

All40originalfiles moved unchanged to transformer-1/bin/dataset:7trunk,31root,
2operator; fullSHA/device/inode/size/mtime and directoryidentities passed pergroup.
Old main-dataset/clover-data aliases retained, clientaliases not restored. Package
loads with bin-relative default dataset. LocalWindows executable and metadata-only
matching directories created; largepayloadsAX102only. Main guides/contexts updated;
historical inventories/tree snapshots unchanged. No other layers or git actions.