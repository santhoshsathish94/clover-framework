export function findRecord(data, expert, matrix) {
  if (!Number.isInteger(expert) || expert < 0 || expert > 895) throw new Error('Expert ID must be 0 - 895.');
  if (!['w1', 'w3', 'w2'].includes(matrix)) throw new Error('Unknown matrix.');
  const record = data.records.find(item => item.expert === expert && item.matrix === matrix);
  if (!record) throw new Error(`Expert ${expert} / ${matrix}: not present in this snapshot.`);
  return record;
}

export function traceWeight(record, coordinate) {
  if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= 32) throw new Error('Sample coordinate must be 0 - 31.');
  if (!record.group) return null;
  const code = record.group.codes[coordinate];
  const reference = record.group.mapping[code];
  const binding = record.bindings[reference];
  if (!binding || binding.bits !== record.first_values[coordinate].bits) throw new Error('Stored reference does not match the exported weight.');
  const offset = binding.table === 'local' ? record.locals[binding.column - 1].offset : (binding.table === 'constants' ? 0 : 2) + (binding.column - 1) * 2;
  return { coordinate, code, reference, binding, offset, selector: record.group.selector,
    mapOffset: record.group.mapping_offset + code, value: record.first_values[coordinate].value };
}

export function validateSnapshot(data) {
  if (data.layer !== 1 || !Array.isArray(data.records) || data.records.length !== data.matrix_records) throw new Error('Invalid record snapshot.');
  const keys = new Set();
  for (const record of data.records) {
    const key = `${record.expert}/${record.matrix}`;
    if (keys.has(key) || record.root_id !== 1) throw new Error('Duplicate record or invalid root reference.');
    keys.add(key);
    if (record.group) for (let coordinate = 0; coordinate < 32; coordinate++) traceWeight(record, coordinate);
  }
  return data;
}

export function valueLabel(binding) {
  return binding.table === 'constants' ? `k3_l1_value_bits[${binding.id}]` : `${binding.table}.value_${binding.column}`;
}