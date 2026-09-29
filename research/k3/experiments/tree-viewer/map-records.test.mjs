import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { validateSnapshot, traceWeight } from './records-model.mjs';

const read = name => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const data = read('map-records-data.json');
const previous = read('constant-records-data.json');
const result = read('../expert-map-results.json');
const manifest = read('../expert-map-manifest.json');

test('compiled map arrays reproduce all stored map-list assignments', () => {
  validateSnapshot(data);
  assert.equal(data.records.length, 2688);
  const source = readFileSync(new URL('../expert-map-constants.h', import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'), result.generated_header_sha256);
  const components = [Buffer.from(data.tables[0].hex.replaceAll(' ', ''), 'hex'), Buffer.from(data.compiled_maps.flat())];
  for (const values of [data.template_offsets, data.template_map_ids]) {
    const bytes = Buffer.alloc(values.length * 2);
    values.forEach((value, index) => bytes.writeUInt16LE(value, index * 2));
    components.push(bytes);
  }
  const constants = Buffer.concat(components);
  assert.equal(constants.length, 15880);
  assert.equal(createHash('sha256').update(constants).digest('hex'), result.compiled_constant_sha256);
  data.records.forEach((record, index) => {
    const raw = Buffer.from(record.record_hex.replaceAll(' ', ''), 'hex');
    assert.equal(raw.length, 2);
    assert.equal(raw.readUInt16LE(), manifest.matrix_template_ids[index]);
    assert.equal(record.template_id, raw.readUInt16LE());
    assert.equal(record.record_offset, 32 + 2 * index);
    const start = data.template_offsets[record.template_id];
    const end = data.template_offsets[record.template_id + 1];
    assert.deepEqual(record.map_ids, data.template_map_ids.slice(start, end));
  });
});

test('exact equation and derived offsets match every actual displayed group', () => {
  let offset = data.payload_start;
  for (const [index, record] of data.records.entries()) {
    assert.deepEqual(record.first_values, previous.records[index].first_values);
    assert.equal(record.first_payload_hex, previous.records[index].first_payload_hex);
    assert.equal(record.mapping_bytes, 0);
    for (let coordinate = 0; coordinate < 32; coordinate++) {
      const trace = traceWeight(record, coordinate);
      const map = record.map_ids[trace.selector];
      const reference = data.compiled_maps[map][trace.code];
      assert.equal(reference, trace.reference);
      assert.equal(data.tables[0].values[reference].bits, record.first_values[coordinate].bits);
    }
    for (const block of record.blocks) {
      assert.equal(block.offset, offset);
      offset += block.bytes;
    }
  }
  assert.equal(offset, result.dataset_bytes);
  assert.equal(result.values_verified, 29595009024);
  assert.equal(result.native_float32_rows_checked, 60);
  assert.equal(result.gate, 'PASS');
});