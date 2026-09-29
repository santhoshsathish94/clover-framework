export const matrixOrder = ['w1', 'w3', 'w2'];

export function createTreeIndex(tree, checks) {
  if (tree.format !== 'layer1-binary-source-tree-v1' || tree.expert_count !== 896) {
    throw new Error('Unexpected dataset: layer 1 expert tree required.');
  }
  const nodes = new Map();
  const matrices = new Map();
  const experts = new Map();
  const samples = new Map((checks.actual_leaves_checked || []).map(sample => [
    `${sample.expert}/${sample.matrix}/${sample.row}/${sample.coordinate}`, sample,
  ]));

  function add(raw, parent = null, id = 'root', inherited = {}) {
    const node = { id, parent, raw, stored: true, kind: raw.kind, axis: raw.axis,
      range: raw.range, expert: raw.expert, matrix: raw.matrix,
      own: raw.shared, facts: { ...inherited, ...raw.shared }, children: [] };
    node.label = id === 'root' ? 'Layer 1' : raw.kind === 'matrix'
      ? `${raw.matrix} · ${tree.matrix_schemas[raw.matrix].role}`
      : raw.axis === 'expert' ? `Experts ${raw.range[0]}–${raw.range[1] - 1}`
        : raw.range[0] === 0 && raw.range[1] === 3 ? `Expert ${raw.expert}` : 'Gate + up';
    nodes.set(id, node);
    if (raw.kind === 'branch') {
      if (!raw.left || !raw.right) throw new Error('Non-binary stored branch.');
      node.children = [add(raw.left, node, `${id}L`, node.facts), add(raw.right, node, `${id}R`, node.facts)];
      if (raw.axis === 'matrix' && raw.range[0] === 0 && raw.range[1] === 3) experts.set(raw.expert, node);
    } else {
      matrices.set(`${raw.expert}/${raw.matrix}`, node);
      node.shape = tree.shape_templates[tree.matrix_schemas[raw.matrix].shape_id];
      node.matrixNode = node;
    }
    return node;
  }
  const root = add(tree.root);
  const storedCount = nodes.size;
  if (storedCount !== 5375 || matrices.size !== 2688 || experts.size !== 896) {
    throw new Error('Tree coverage does not match the verified dataset.');
  }

  function virtual(parent, axis, low, high, side) {
    const matrixNode = parent.matrixNode;
    const node = { id: `${parent.id}/${side}`, parent, stored: false, kind: 'range', axis,
      range: [low, high], matrixNode, expert: matrixNode.expert, matrix: matrixNode.matrix,
      row: parent.row, group: parent.group, facts: parent.facts, own: {}, children: [] };
    const singular = high - low === 1;
    if (axis === 'row' && singular) node.row = low;
    if (axis === 'group' && singular) node.group = low;
    if (axis === 'coordinate' && singular) {
      node.kind = 'weight';
      node.coordinate = node.group * 32 + low;
    }
    const noun = axis === 'coordinate' ? 'Weight' : axis === 'group' ? 'Group' : 'Row';
    const displayLow = axis === 'coordinate' ? node.group * 32 + low : low;
    const displayHigh = axis === 'coordinate' ? node.group * 32 + high - 1 : high - 1;
    node.label = singular ? `${noun} ${displayLow}` : `${noun}s ${displayLow}–${displayHigh}`;
    nodes.set(node.id, node);
    return node;
  }

  function children(node) {
    if (node.kind === 'weight' || node.children.length) return node.children;
    if (node.stored && node.kind !== 'matrix') return [];
    let axis, low, high;
    if (node.kind === 'matrix') {
      axis = 'row'; low = 0; high = node.shape[0];
    } else if (node.range[1] - node.range[0] > 1) {
      axis = node.axis; [low, high] = node.range;
    } else if (node.axis === 'row') {
      axis = 'group'; low = 0; high = node.matrixNode.shape[1] / 32;
    } else {
      axis = 'coordinate'; low = 0; high = 32;
    }
    const middle = Math.floor((low + high) / 2);
    node.children = [virtual(node, axis, low, middle, 'L'), virtual(node, axis, middle, high, 'R')];
    return node.children;
  }

  function locate(expert, matrix, row, coordinate) {
    if (!Number.isInteger(expert) || expert < 0 || expert >= 896 || !matrixOrder.includes(matrix)) {
      throw new Error('Expert must be 0–895; matrix must be w1, w3 or w2.');
    }
    let node = matrices.get(`${expert}/${matrix}`);
    if (!Number.isInteger(row) || row < 0 || row >= node.shape[0]
      || !Number.isInteger(coordinate) || coordinate < 0 || coordinate >= node.shape[1]) {
      throw new Error(`Row must be 0–${node.shape[0] - 1}; weight must be 0–${node.shape[1] - 1}.`);
    }
    while (node.kind !== 'weight') {
      const next = children(node);
      const axis = next[0].axis;
      const target = axis === 'row' ? row : axis === 'group' ? Math.floor(coordinate / 32) : coordinate % 32;
      node = next.find(child => child.range[0] <= target && target < child.range[1]);
      if (!node) throw new Error('Binary path did not resolve.');
    }
    return node;
  }

  function path(node) {
    const result = [];
    for (let current = node; current; current = current.parent) result.unshift(current);
    return result;
  }

  function leafData(node) {
    if (node.kind !== 'weight') return null;
    const width = node.matrixNode.shape[1];
    const tensors = node.matrixNode.raw.tensors;
    return {
      expert: node.expert, matrix: node.matrix, row: node.row, coordinate: node.coordinate, group: node.group,
      code_byte_offset: tensors.weight_packed.offset + node.row * width / 2 + Math.floor(node.coordinate / 2),
      nibble_shift: (node.coordinate % 2) * 4,
      scale_byte_offset: tensors.weight_scale.offset + node.row * width / 32 + node.group,
      sample: samples.get(`${node.expert}/${node.matrix}/${node.row}/${node.coordinate}`) || null,
    };
  }

  function factOrigin(node, key) {
    for (let current = node; current; current = current.parent) {
      if (Object.hasOwn(current.own, key)) return current;
    }
    return null;
  }

  function outline(node, depth = 3) {
    return { node, children: depth > 0 ? children(node).map(child => outline(child, depth - 1)) : [] };
  }

  function countWeights(node) {
    if (node.kind === 'weight') return 1;
    if (node.axis === 'row') return (node.range[1] - node.range[0]) * node.matrixNode.shape[1];
    if (node.axis === 'group') return (node.range[1] - node.range[0]) * 32;
    if (node.axis === 'coordinate') return node.range[1] - node.range[0];
    if (node.kind === 'matrix') return node.shape[0] * node.shape[1];
    if (node.axis === 'expert') return (node.range[1] - node.range[0]) * 33030144;
    return (node.range[1] - node.range[0]) * 11010048;
  }

  return { root, tree, nodes, matrices, experts, samples, storedCount, children, locate, path,
    leafData, factOrigin, outline, countWeights };
}