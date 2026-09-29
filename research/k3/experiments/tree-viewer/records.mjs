import { findRecord, traceWeight, validateSnapshot, valueLabel } from './records-model.mjs';

const byId = id => document.getElementById(id);
const number = value => value.toLocaleString('en-US');
const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[character]);
const table = (headings, rows) => `<thead><tr>${headings.map(heading => `<th scope="col">${escape(heading)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(value => `<td>${value}</td>`).join('')}</tr>`).join('')}</tbody>`;
let data, record, matrix = 'w1', expert = 0;
const format = new URLSearchParams(location.search).get('format') || 'maps';
const mapEquation = format === 'maps';
const constants = format !== 'original';

function cellButton(cell, name, index) {
  return `<button type="button" class="value-cell" data-table="${name}" data-cell="${index}" id="cell-${name}-${index}" aria-label="${name} value ${index + 1}: ${escape(cell.value)}">${escape(cell.value)}<small>${cell.bytes}</small></button>`;
}

function showCell(name, index, scroll = false) {
  const values = name === 'local' ? record.locals : data.tables.find(item => item.name === name).values;
  const value = values[index];
  document.querySelectorAll('.value-cell.selected').forEach(element => element.classList.remove('selected'));
  const element = byId(`cell-${name}-${index}`);
  element.classList.add('selected');
  const filename = name === 'local' ? data.expert_file : `${name}.bin`;
  byId('cell-detail').innerHTML = `<strong>${name}.value_${index + 1} = ${escape(value.value)}</strong><span>BF16 <code>0x${value.bits}</code></span><span>2 stored bytes <code>${value.bytes}</code></span><span>${filename} @ ${number(value.offset)}</span>`;
  if (name === 'constants') byId('cell-detail').innerHTML = `<strong>k3_l1_value_bits[${index}] = ${escape(value.value)}</strong><span>BF16 <code>0x${value.bits}</code></span><span>2 compiled bytes <code>${value.bytes}</code></span><span>C array byte offset ${number(value.offset)} / 0 scalar bytes in dataset</span>`;
  if (scroll) element.scrollIntoView({ block:'center', inline:'center', behavior:'smooth' });
}

function showTables() {
  byId('shared-heading').textContent = constants ? 'Compiled C constants' : 'Shared value tables';
  if (constants) {
    const item = data.tables[0];
    byId('root-tables').innerHTML = `<div class="shared-table"><header><h3>static const uint16_t k3_l1_value_bits[66]</h3><span>66 values / 132 compiled bytes</span></header><div class="table-scroll" tabindex="0" role="region" aria-label="Compiled C palette"><table class="data-table value-table">${table(item.values.map((value, index) => `[${index}]`), [item.values.map((value, index) => cellButton(value, 'constants', index))])}</table></div></div>`;
    byId('shared-total').textContent = mapEquation ? '15,880 compiled bytes / scalar + map + list arrays' : '0 scalar-value bytes in dataset';
    return;
  }
  const parts = data.tables.map(item => `<div class="shared-table"><header><h3>${item.name}</h3><span>${item.values.length} values / ${item.bytes} bytes</span></header><div class="table-scroll" tabindex="0" role="region" aria-label="${item.name} stored row"><table class="data-table value-table">${table([item.name === 'root' ? 'id' : 'root_id', ...item.values.map((value, index) => `value_${index + 1}`)], [[`<span class="key-cell">${item.id}</span>`, ...item.values.map((value, index) => cellButton(value, item.name, index))]])}</table></div>${item.values.length ? '' : '<p class="empty-cell">No additional common values</p>'}</div>`);
  byId('root-tables').innerHTML = parts[0] + `<div class="child-tables">${parts.slice(1).join('')}</div>`;
  byId('shared-total').textContent = `${data.tables.reduce((sum, item) => sum + item.bytes, 0)} bytes total / values + ID only`;
}

function showWeight(coordinate) {
  const trace = traceWeight(record, coordinate);
  byId('trace-heading').textContent = `Weight ${coordinate} / row 0`;
  document.querySelectorAll('#weight-records tbody tr').forEach((row, index) => row.classList.toggle('selected', index === coordinate));
  if (!trace) {
    byId('weight-trace').textContent = `Direct reference / ${record.first_values[coordinate].value}`;
    return;
  }
  byId('weight-trace').innerHTML = `<div class="trace-step"><span>4-bit code</span><strong>${trace.code}</strong><small>${trace.code.toString(2).padStart(4, '0')} / ${coordinate % 2 ? 'high' : 'low'} nibble</small></div><div class="trace-step"><span>Group palette ${trace.selector}</span><strong>Reference ${trace.reference}</strong><small>1 byte @ ${number(trace.mapOffset)}</small></div><div class="trace-step"><span>Value cell</span><strong>${trace.binding.table}.value_${trace.binding.column}</strong><small>2 bytes @ ${number(trace.offset)}</small></div><div class="trace-value"><strong>${escape(trace.value)}</strong><span>BF16 0x${trace.binding.bits}</span><a href="#cell-${trace.binding.table}-${trace.binding.column - 1}" id="trace-cell">${trace.binding.bytes} / Show cell</a></div>`;
  if (constants) {
    const step = byId('weight-trace').querySelectorAll('.trace-step')[2];
    step.querySelector('span').textContent = 'Compiled C value';
    step.querySelector('strong').textContent = valueLabel(trace.binding);
    step.querySelector('small').textContent = `2 compiled bytes / array offset ${trace.offset}`;
  }
  if (mapEquation) {
    const step = byId('weight-trace').querySelectorAll('.trace-step')[1];
    step.querySelector('span').textContent = `Selector ${trace.selector} / C map ${record.group.map_id}`;
    step.querySelector('small').textContent = `Compiled map byte offset ${number(trace.mapOffset)}`;
    byId('equation-actual').textContent = `T = ${record.template_id}, selector = ${trace.selector}, code = ${trace.code}: L[${record.template_id}][${trace.selector}] = ${record.group.map_id}; G[${record.group.map_id}][${trace.code}] = ${trace.reference}; V[${trace.reference}] = ${trace.value}`;
  }
  byId('trace-cell').addEventListener('click', event => { event.preventDefault(); showCell(trace.binding.table, trace.binding.column - 1, true); });
  showCell(trace.binding.table, trace.binding.column - 1);
}

function showRecord(nextExpert = expert, nextMatrix = matrix) {
  try {
    const next = findRecord(data, nextExpert, nextMatrix);
    record = next; expert = nextExpert; matrix = nextMatrix;
    byId('record-error').hidden = true;
  } catch (error) { byId('record-error').textContent = error.message; byId('record-error').hidden = false; return; }
  byId('record-expert').value = expert;
  byId('record-identity').textContent = `root_id ${record.root_id} / expert ${expert} / ${matrix}`;
  document.querySelectorAll('button[data-matrix]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.matrix === matrix)));
  byId('record-prev').disabled = expert === 0;
  byId('record-next').disabled = !data.records.some(item => item.expert === expert + 1 && item.matrix === matrix);
  byId('expert-fields').innerHTML = mapEquation
    ? table(['map_list_id (uint16LE)'], [[record.template_id]])
    : constants
    ? table(['map_offset (payload-relative)', 'map_count', 'first_block', 'block_count', 'reserved'], [[number(record.mapping_offset - data.payload_start), record.map_count, record.first_block_index, record.blocks.length, 0]])
    : table(['root_id', 'expert_id', 'matrix', 'shape', 'local_count', 'local_offset (bytes)', 'group_palette_offset (bytes)', 'group_palette_bytes', 'group_scales', 'blocks'], [[1, expert, matrix, record.shape.join(' x '), record.locals.length, number(record.local_offset), number(record.mapping_offset), number(record.mapping_bytes), record.group_scales.join(', '), record.blocks.length]]);
  if (constants) byId('record-identity').textContent = `Layer 1 / expert ${expert} / ${matrix} / ${record.shape.join(' x ')} (implicit)`;
  byId('binary-record').hidden = !constants;
  if (constants) {
    byId('binary-record-hex').textContent = record.record_hex;
    byId('binary-record-location').textContent = `${mapEquation ? 2 : 24} bytes / file offset ${number(record.record_offset)}`;
  }
  byId('equation-record').hidden = !mapEquation;
  if (mapEquation) {
    byId('equation-template').textContent = `T = ${record.template_id} / ${record.map_ids.length} maps`;
    byId('equation-map-list').textContent = `L[${record.template_id}] = [${record.map_ids.join(', ')}]`;
  }
  byId('local-count').textContent = `${record.locals.length} values / ${record.locals.length * 2} bytes`;
  byId('local-values').innerHTML = record.locals.length ? table(record.locals.map((value, index) => `value_${index + 1}`), [record.locals.map((value, index) => cellButton(value, 'local', index))]) : '<tbody><tr><td class="empty-cell">No local values</td></tr></tbody>';
  if (constants) byId('local-values').innerHTML = '<tbody><tr><td class="empty-cell">Not stored / all scalar values supplied by the C palette</td></tr></tbody>';
  byId('block-summary').textContent = `${record.blocks.length} stored blocks / ${number(record.blocks.reduce((sum, block) => sum + block.bytes, 0))} payload bytes`;
  byId('block-index').innerHTML = table(['Block', 'First row', 'Rows', 'File offset (bytes)', 'Stored bytes', 'Codec', constants ? 'Decoded CRC32' : 'Decoded SHA256'], record.blocks.map((block, index) => [index, index * 64, 64, number(block.offset), number(block.bytes), block.codec, block.crc32 ?? block.sha256 ?? 'Not in partial index']));
  byId('weight-records').innerHTML = table(['Weight', 'Code', 'Ref ID', 'Value cell', 'Exact value'], record.first_values.map((value, index) => {
    const trace = traceWeight(record, index);
    return [`<button class="weight-select" data-coordinate="${index}" aria-label="Inspect weight ${index}">${index}</button>`, trace?.code ?? '-', trace?.reference ?? '-', trace ? valueLabel(trace.binding) : 'Direct', escape(value.value)];
  }));
  byId('stored-hex').textContent = record.first_payload_hex;
  byId('stored-range').textContent = `${data.expert_file} @ ${number(record.blocks[0].offset)} / ${record.blocks[0].codec} / ${number(record.blocks[0].bytes)} bytes in block`;
  byId('group-hex').textContent = record.group?.codes_hex ?? 'Direct-reference block';
  byId('palette-address').textContent = record.group ? `Selector ${record.group.selector} / ${constants ? 'global constant references' : 'original scale ' + record.group.scale} / ${data.expert_file} @ ${number(record.group.mapping_offset)}` : 'No group selector in direct block';
  if (mapEquation) {
    byId('palette-address').textContent = `k3_group_maps[${record.group.map_id}] / compiled byte offset ${record.group.mapping_offset} / 0 map-row bytes in dataset`;
  }
  document.querySelector('.palette-details summary').textContent = `${mapEquation ? 'Compiled' : 'Stored'} group palette / 16 references`;
  byId('palette-hex').textContent = record.group?.mapping_hex ?? '';
  byId('group-palette').innerHTML = record.group ? table(['Code', 'Ref byte', 'Value cell', 'Exact value'], record.group.mapping.map((ref, code) => {
    const binding = record.bindings[ref];
    return [code, ref, binding ? valueLabel(binding) : 'Absent pair', binding ? escape(binding.value) : '-'];
  })) : '';
  showWeight(0);
  history.replaceState(null, '', `#expert=${expert}&matrix=${matrix}`);
  document.body.dataset.expert = expert;
  document.body.dataset.matrix = matrix;
}

async function load() {
  byId('records-error').hidden = true;
  try {
    const filename = mapEquation ? 'map-records-data.json' : constants ? 'constant-records-data.json' : 'records-data.json';
    const response = await fetch(filename, { cache:'no-store' });
    if (!response.ok) throw new Error(`Record snapshot unavailable (HTTP ${response.status}).`);
    data = validateSnapshot(await response.json());
    byId('snapshot-status').textContent = data.status;
    byId('snapshot-count').textContent = `${number(data.matrix_records)} matrix records / ${data.experts_complete} complete experts / ${data.full_verification ? 'full verification passed' : 'full verification pending'}`;
    document.querySelector('.snapshot-bar').classList.toggle('verified', data.full_verification);
    showTables();
    byId('record-origin').textContent = data.container_final ? 'Record source: stored container metadata and payload bytes.' : 'Record source: stored partial payload. Boundaries from the completed build audit; final metadata not yet written.';
    if (constants) byId('record-origin').textContent = 'Record source: verified binary index and payload. Scalar values compiled in C; expert, matrix and shape derived from record index.';
    if (mapEquation) byId('record-origin').textContent = 'Stored matrix data: one 2-byte map-list ID. Map lists and scalar values compiled in C; block offsets derived from stored lengths.';
    byId('record-format').value = format;
    document.querySelector('[aria-label="Open actual record snapshot JSON"]').href = filename;
    byId('record-provenance').innerHTML = [['Source directory', data.source_directory], ['Expert file', data.expert_file], ['Record origin', data.container_final ? 'Stored container metadata' : 'Stored payload + completed build audit'], ['Sampled bytes SHA256', data.sampled_bytes_sha256], ['Exporter SHA256', data.exporter_sha256]].map(([key, value]) => `<dt>${key}</dt><dd>${escape(value)}</dd>`).join('');
    byId('sample-scope').textContent = data.sample_scope;
    byId('records-main').hidden = false;
    const route = new URLSearchParams(location.hash.slice(1));
    showRecord(0, 'w1');
    if (route.has('expert')) showRecord(Number(route.get('expert')), route.get('matrix') || 'w1');
    document.body.dataset.ready = 'true';
  } catch (error) {
    byId('records-error').querySelector('span').textContent = error.message;
    byId('records-error').hidden = false;
    byId('snapshot-status').textContent = 'Snapshot unavailable';
  }
}

byId('record-locator').addEventListener('submit', event => { event.preventDefault(); showRecord(byId('record-expert').valueAsNumber); });
byId('record-prev').addEventListener('click', () => showRecord(expert - 1));
byId('record-next').addEventListener('click', () => showRecord(expert + 1));
document.querySelectorAll('button[data-matrix]').forEach(button => button.addEventListener('click', () => showRecord(expert, button.dataset.matrix)));
document.addEventListener('click', event => {
  const cell = event.target.closest('[data-cell]');
  if (cell) showCell(cell.dataset.table, Number(cell.dataset.cell));
  const row = event.target.closest('#weight-records tbody tr');
  if (row) showWeight(Number(row.querySelector('[data-coordinate]').dataset.coordinate));
});
byId('records-retry').addEventListener('click', load);
byId('record-format').addEventListener('change', event => { location.search = `?format=${event.target.value}`; });
globalThis.lucide?.createIcons();
load();