import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

const here = new URL('./', import.meta.url);
const text = name => readFileSync(new URL(name, here), 'utf8');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const rows = name => text(name).trim().split(/\r?\n/).map(line => line.split('\t'));
const validation = text('validation.log');
const verified = [...validation.matchAll(/^VERIFIED\t(\d+)\t(france|japan)\t5\t(\d+)\r?$/gm)];
assert.equal(verified.length, 182);
assert.equal([...text('preflight.log').matchAll(/^PREFLIGHT\t\d+\tPASS\r?$/gm)].length, 91);
assert.equal([...text('move.log').matchAll(/^PACKAGED\t\d+\tPASS\r?$/gm)].length, 91);
assert(text('move.log').includes('273 verified directory moves'));
for (const layer of [2, 3, 12, 92]) assert(text('validation-run.log').includes(`CONTROLS\t${layer}\tPASS`));
const cli = JSON.parse(text('cli-results.json'));
assert.equal(cli.gate, 'PASS');
assert.deepEqual(cli.results.map(result => result.layer), [2, 3, 12, 92]);
const remotePrefix = '/opt/clover-k3/clover-intelegence/code/tansformers/';
const locks = new Map(text('SOURCE-LOCK.sha256').trim().split(/\r?\n/).map(line => {
  const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
  assert(match && match[2].startsWith(remotePrefix));
  return [match[2].slice(remotePrefix.length), match[1]];
}));
for (const name of ['reference-all.inc', 'test-reference.c', 'test-reference.sh', 'test-controls.c'])
  assert.equal(digest(readFileSync(new URL(name, here))), locks.get(`remaining/${name}`));

const outcomes = [];
let movedFiles = 0, movedBytes = 0n, snapshotValues = 0;
for (let layer = 2; layer <= 92; layer++) {
  const prefix = `../transformer-${layer}/`;
  const generation = JSON.parse(text(prefix + 'generation.json'));
  assert.equal(generation.layer, layer);
  assert.equal(generation.input_snapshots, Math.ceil(layer / 12));
  assert.equal(generation.output_snapshots, Math.floor(layer / 12) + 1);
  for (const [name, hash] of Object.entries(generation.files)) {
    assert.equal(digest(readFileSync(new URL(prefix + name, here))), hash);
    assert.equal(locks.get(`transformer-${layer}/${name}`), hash);
  }
  for (const prompt of ['france', 'japan']) {
    const matches = verified.filter(match => Number(match[1]) === layer && match[2] === prompt);
    assert.equal(matches.length, 1);
    assert.equal(Number(matches[0][3]), generation.output_snapshots);
  }
  const groups = [`trunk-${layer}`, `root-${layer}`, `operators/qkv-all/layer-${layer}`];
  assert.deepEqual(rows(prefix + 'MOVE-JOURNAL.tsv'), groups.map(group => [group, 'MOVED_AND_VERIFIED']));
  const plan = rows(prefix + 'MOVE-PLAN.tsv');
  assert.equal(new Set(plan.map(row => row[2])).size, plan.length);
  let bytes = 0n;
  for (const [group, source, target, identity, hash] of plan) {
    assert(groups.includes(group));
    const original = `/opt/clover-k3/clover-intelegence/dataset/${group}/`;
    assert(source.startsWith(original));
    const relative = source.slice('/opt/clover-k3/clover-intelegence/dataset/'.length);
    assert.equal(target, remotePrefix + `transformer-${layer}/bin/dataset/` + relative);
    assert(/^[a-f0-9]{64}$/.test(hash));
    const fields = identity.split(':');
    assert(fields.length >= 4);
    bytes += BigInt(fields[2]);
    const local = new URL(prefix + 'bin/dataset/' + relative, here);
    if (existsSync(local)) assert.equal(digest(readFileSync(local)), hash);
  }
  movedFiles += plan.length;
  movedBytes += bytes;
  snapshotValues += generation.output_snapshots * 7168 * 10;
  outcomes.push({ layer, gate: 'PASS', attention: generation.attention,
    input_snapshots: generation.input_snapshots, output_snapshots: generation.output_snapshots,
    cases: ['france', 'japan'], positions_per_case: 5, residual_values_checked: 71680,
    route_ids_checked: 160, moved_directories: 3, moved_files: plan.length, moved_bytes: bytes.toString(),
    source_sha256: generation.files, move_plan_sha256: digest(readFileSync(new URL(prefix + 'MOVE-PLAN.tsv', here))),
    move_journal_sha256: digest(readFileSync(new URL(prefix + 'MOVE-JOURNAL.tsv', here))) });
}
assert.equal(movedFiles, 3325);
assert.equal(outcomes.filter(result => result.attention === 'MLA').length, 24);
const report = { gate: 'PASS', layers: 91, layer_cases: 182, positions: 910,
  residual_values_checked: 91 * 71680, snapshot_values_checked: snapshotValues,
  route_ids_checked: 91 * 160, moved_directories: 273, moved_files: movedFiles,
  moved_bytes: movedBytes.toString(), sanitizer_layers: [2, 3, 12, 92], cli_layers: [2, 3, 12, 92],
  evidence_archive_sha256: digest(readFileSync(new URL('completed-evidence.tar', here))),
  validation_log_sha256: digest(readFileSync(new URL('validation.log', here))),
  source_lock_sha256: digest(readFileSync(new URL('SOURCE-LOCK.sha256', here))),
  scope: 'Independent per-layer comparisons against original layer arithmetic and exact-input-bound existing expert observations; not a chained standalone pipeline, arbitrary-input proof, or performance benchmark',
  outcomes };
function save(name, content) {
  const file = new URL(name, here);
  if (existsSync(file)) assert.equal(readFileSync(file, 'utf8'), content);
  else writeFileSync(file, content, { flag: 'wx' });
}
for (const outcome of outcomes) {
  const prefix = `../transformer-${outcome.layer}/`;
  save(prefix + 'verification.json', JSON.stringify(outcome, null, 2) + '\n');
  const file = new URL(prefix + 'CONTEXT.md', here);
  const note = `\n## Completed Outcome\n\nLayer ${outcome.layer} passed France and Japan at all five positions: every residual, sixteen route IDs and all ${outcome.output_snapshots} output snapshots matched the independent reference. The candidate computes live from stored parameters; historical observations are reference-only and require exact input matches. All ${outcome.moved_files} files across the three requested directories moved unchanged into bin/dataset, with full hashes/file and directory identities checked and old model aliases preserved. Package-local --inspect passed.\n\n[verification.json](verification.json) records this layer's source and move identities. [Campaign results](../remaining/results.json) contain the full evidence and representative sanitizer/CLI controls. Initial pending notes above describe the generation stage, not current completion. Large payloads remain AX102-only; local data folders contain metadata. No client/server/transformer1 edits or git actions. Broader inputs, long contexts and a chained standalone pipeline were not tested.\n`;
  const current = readFileSync(file, 'utf8');
  if (!current.includes('## Completed Outcome')) writeFileSync(file, current + note);
  else assert(current.endsWith(note));
}
save('results.json', JSON.stringify(report, null, 2) + '\n');
const index = ['# Transformer Packages', '',
  'Layers 2-92 passed both existing five-token cases independently. Transformer 1 retains its earlier separate validation.', '',
  '| Layer | Attention | Input Snapshots | Output Snapshots | Evidence |',
  '|---|---|---:|---:|---|',
  ...outcomes.map(result => `| [${result.layer}](../transformer-${result.layer}/README.md) | ${result.attention} | ${result.input_snapshots} | ${result.output_snapshots} | [PASS](../transformer-${result.layer}/verification.json) |`), ''].join('\n');
save('LAYERS.md', index);
console.log(`PASS: 91 source sets, 182 numerical cases, ${movedFiles} files/${movedBytes} bytes, 273 moves; per-layer results and context finalized`);