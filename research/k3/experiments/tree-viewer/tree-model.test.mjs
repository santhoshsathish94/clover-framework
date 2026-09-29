import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createTreeIndex } from './tree-model.mjs';

const read = name => JSON.parse(readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'));
const tree = read('layer1-binary-tree.json');
const checks = read('layer1-binary-tree-checks.json');
const index = createTreeIndex(tree, checks);

test('complete stored hierarchy and exact root partitions', () => {
  assert.equal(index.storedCount, 5375);
  assert.equal(index.matrices.size, 2688);
  assert.equal(index.experts.size, 896);
  assert.deepEqual(index.root.children.map(node => node.range), [[0, 448], [448, 896]]);
  assert.equal(index.countWeights(index.root), 29595009024);
  for (const node of index.nodes.values()) {
    if (node.kind === 'branch') assert.equal(node.children.length, 2);
  }
});

test('all measured leaves match verified Python paths and preserve exceptions', () => {
  for (const sample of checks.actual_leaves_checked) {
    const leaf = index.locate(sample.expert, sample.matrix, sample.row, sample.coordinate);
    const data = index.leafData(leaf);
    assert.equal(data.code_byte_offset, sample.code_byte_offset);
    assert.equal(data.scale_byte_offset, sample.scale_byte_offset);
    assert.equal(data.nibble_shift, sample.nibble_shift);
    assert.equal(data.sample.float32_bits_hex, sample.float32_bits_hex);
    assert.equal(index.factOrigin(leaf, 'most_frequent_scale'), index.root);
    assert.equal(index.countWeights(leaf), 1);
    const actualPath = index.path(leaf).slice(1).map(node => node.parent.children.indexOf(node) === 0 ? 'left' : 'right');
    assert.deepEqual(actualPath, sample.path.map(step => step.direction));
  }
  assert.equal(index.leafData(index.locate(0, 'w1', 0, 224)).sample.scale, 120);
  assert.equal(index.leafData(index.locate(0, 'w1', 0, 9)).sample.float32_bits_hex, '80000000');
});

test('last row and coordinate resolve in every matrix of every expert', () => {
  for (const matrix of index.matrices.values()) {
    const row = matrix.shape[0] - 1;
    const coordinate = matrix.shape[1] - 1;
    const leaf = index.locate(matrix.expert, matrix.matrix, row, coordinate);
    const data = index.leafData(leaf);
    assert.equal(data.code_byte_offset, matrix.raw.tensors.weight_packed.offset + matrix.shape[0] * matrix.shape[1] / 2 - 1);
    assert.equal(data.scale_byte_offset, matrix.raw.tensors.weight_scale.offset + matrix.shape[0] * matrix.shape[1] / 32 - 1);
  }
});

test('no invented values for unmeasured leaves and invalid inputs rejected', () => {
  assert.equal(index.leafData(index.locate(1, 'w1', 0, 0)).sample, null);
  for (const input of [[896, 'w1', 0, 0], [-1, 'w1', 0, 0], [0, 'w1', 3072, 0], [0, 'w2', 0, 3072], [0, 'other', 0, 0]]) {
    assert.throws(() => index.locate(...input));
  }
  assert.ok(index.outline(index.root, 3).children.length === 2);
});