import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { findRecord, traceWeight, validateSnapshot } from './records-model.mjs';

const data = JSON.parse(readFileSync(new URL('records-data.json', import.meta.url), 'utf8'));

test('actual root file bytes match all displayed cells and foreign keys', () => {
  for (const table of data.tables) {
    const raw = Buffer.from(table.hex.replaceAll(' ', ''), 'hex');
    assert.equal(raw.length, table.bytes);
    assert.equal(raw.readUInt16LE(0), 1);
    for (const [index, cell] of table.values.entries()) {
      assert.equal(raw.readUInt16LE(2 + index * 2), parseInt(cell.bits, 16));
      const value = Buffer.alloc(4);
      value.writeUInt32LE(parseInt(cell.bits, 16) * 65536);
      assert.ok(Object.is(value.readFloatLE(), Number(cell.value)));
    }
  }
  assert.deepEqual(data.tables.map(table => table.bytes), [46, 2, 2]);
});

test('every exported group traces actual nibbles to stored root or local cells', () => {
  validateSnapshot(data);
  for (const record of data.records) {
    if (!record.group) continue;
    const packed = Buffer.from(record.group.codes_hex.replaceAll(' ', ''), 'hex');
    for (let coordinate = 0; coordinate < 32; coordinate++) {
      const trace = traceWeight(record, coordinate);
      const byte = packed[Math.floor(coordinate / 2)];
      assert.equal(trace.code, coordinate % 2 ? byte >> 4 : byte & 15);
      const table = trace.binding.table === 'local' ? record.locals : data.tables.find(item => item.name === trace.binding.table).values;
      assert.equal(table[trace.binding.column - 1].bits, trace.binding.bits);
    }
    assert.equal(record.blocks.length, record.shape[0] / 64);
    for (let index = 1; index < record.blocks.length; index++) {
      assert.equal(record.blocks[index].offset, record.blocks[index - 1].offset + record.blocks[index - 1].bytes);
    }
  }
});

test('known actual expert sample, negative zero, local refs and invalid selections', () => {
  const record = findRecord(data, 0, 'w1');
  assert.equal(traceWeight(record, 0).value, '0.0625');
  assert.equal(traceWeight(record, 9).value, '-0');
  assert.ok(data.records.some(item => item.group?.refs.some(ref => item.bindings[ref].table === 'local')));
  assert.throws(() => findRecord(data, -1, 'w1'));
  assert.throws(() => findRecord(data, 0, 'bad'));
  assert.throws(() => traceWeight(record, 32));
  if (data.experts_complete < 896) assert.throws(() => findRecord(data, 895, 'w2'));
});