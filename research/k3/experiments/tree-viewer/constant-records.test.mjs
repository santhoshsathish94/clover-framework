import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { findRecord, traceWeight, validateSnapshot, valueLabel } from './records-model.mjs';

const data = JSON.parse(readFileSync(new URL('constant-records-data.json', import.meta.url), 'utf8'));
const prior = JSON.parse(readFileSync(new URL('records-data.json', import.meta.url), 'utf8'));
const report = JSON.parse(readFileSync(new URL('../expert-constant-results.json', import.meta.url), 'utf8'));

test('all 66 compiled constants match bytes, C source and native verification identity', () => {
  const source = readFileSync(new URL('../../expert-constant-palette.h', import.meta.url), 'utf8');
  const literals = [...source.matchAll(/0x([0-9a-f]{4})/g)].map(match => parseInt(match[1], 16));
  const raw = Buffer.from(data.tables[0].hex.replaceAll(' ', ''), 'hex');
  assert.equal(raw.length, 132);
  assert.equal(createHash('sha256').update(raw).digest('hex'), report.compiled_palette_sha256);
  assert.equal(data.scalar_value_bytes_in_dataset, 0);
  assert.equal(data.experts_complete, 896);
  assert.equal(data.matrix_records, 2688);
  assert.equal(report.values_verified, 29595009024);
  assert.equal(report.gate, 'PASS');
  for (let index = 0; index < 66; index++) assert.equal(raw.readUInt16LE(index * 2), literals[index]);
});

test('every actual matrix index and group points only to the compiled constants', () => {
  validateSnapshot(data);
  for (const record of data.records) {
    assert.deepEqual(record.locals, []);
    const raw = Buffer.from(record.record_hex.replaceAll(' ', ''), 'hex');
    assert.equal(raw.length, 24);
    assert.equal(Number(raw.readBigUInt64LE()), record.mapping_offset - data.payload_start);
    assert.equal(raw.readUInt32LE(8), record.map_count);
    assert.equal(raw.readUInt32LE(12), record.first_block_index);
    assert.equal(raw.readUInt32LE(16), record.blocks.length);
    assert.equal(raw.readUInt32LE(20), 0);
    const codes = Buffer.from(record.group.codes_hex.replaceAll(' ', ''), 'hex');
    const mapping = Buffer.from(record.group.mapping_hex.replaceAll(' ', ''), 'hex');
    for (let coordinate = 0; coordinate < 32; coordinate++) {
      const trace = traceWeight(record, coordinate);
      assert.equal(trace.code, (codes[Math.floor(coordinate / 2)] >> (coordinate % 2 * 4)) & 15);
      assert.equal(mapping[trace.code], trace.reference);
      assert.equal(trace.binding.table, 'constants');
      assert.equal(trace.offset, trace.reference * 2);
      assert.equal(trace.binding.bits, data.tables[0].values[trace.reference].bits);
      assert.equal(valueLabel(trace.binding), `k3_l1_value_bits[${trace.reference}]`);
    }
  }
});

test('new representation agrees with every prior actual displayed weight', () => {
  for (const earlier of prior.records) {
    const current = findRecord(data, earlier.expert, earlier.matrix);
    assert.deepEqual(current.first_values.map(value => value.bits), earlier.first_values.map(value => value.bits));
    assert.equal(current.first_payload_hex, earlier.first_payload_hex);
  }
  assert.equal(traceWeight(findRecord(data, 0, 'w1'), 9).value, '-0');
  assert.equal(findRecord(data, 895, 'w2').blocks.length, 56);
});