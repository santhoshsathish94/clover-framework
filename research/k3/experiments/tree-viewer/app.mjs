import { createTreeIndex } from './tree-model.mjs';

const $ = id => document.getElementById(id);
const escapeText = value => String(value).replace(/[&<>"']/g, symbol => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[symbol]);
const format = value => typeof value === 'number' ? value.toLocaleString('en-US') : Array.isArray(value) ? value.join(', ') : String(value);
const icons = () => globalThis.lucide?.createIcons();
const canvas = $('graph');
const context = canvas.getContext('2d');
let index, selected, focus, mode = 'branch', transform, points = [], spatial, history = [], zoom;
let viewportWidth = 1, viewportHeight = 1, ready = false, pointerStart;
const colors = { expert:'#32644a', w1:'#c06c5b', w3:'#bd922e', w2:'#4c8392', generated:'#388a9c', selected:'#153f2b' };

function color(node) { return node.stored ? node.matrix ? colors[node.matrix] : colors.expert : colors.generated; }
function nodeType(node) {
  if (node.id === 'root') return 'Layer root';
  if (node.kind === 'weight') return 'Generated weight leaf';
  if (!node.stored) return `Generated ${node.axis} branch`;
  if (node.kind === 'matrix') return 'Stored matrix record';
  return node.axis === 'expert' ? 'Expert range' : 'Expert matrix branch';
}
function detailRows(rows) {
  return `<dl>${rows.map(([label,value,mono]) => `<dt>${escapeText(label)}</dt><dd${mono ? ' class="mono"' : ''}>${escapeText(format(value))}</dd>`).join('')}</dl>`;
}

function setForm(node) {
  if (node.expert !== undefined) $('expert').value = node.expert;
  if (node.matrix) $('matrix').value = node.matrix;
  if (node.row !== undefined) $('row').value = node.row;
  if (node.coordinate !== undefined) $('coordinate').value = node.coordinate;
  updateBounds();
}

function updateBounds() {
  const down = $('matrix').value === 'w2';
  $('row').max = down ? 3583 : 3071;
  $('coordinate').max = down ? 3071 : 3583;
}

function select(node, options = {}) {
  if (!ready) return;
  if (selected && selected !== node && options.record !== false) history.push({ selected, focus, mode });
  selected = node;
  if (options.focus) {
    focus = node;
    mode = 'branch';
  }
  setForm(node);
  $('back').disabled = history.length === 0;
  $('up').disabled = !selected.parent;
  $('focus').disabled = false;
  renderInspector();
  if (options.focus) layout(); else draw();
  if (node.kind === 'weight') {
    const params = new URLSearchParams({ expert:node.expert, matrix:node.matrix, row:node.row, coordinate:node.coordinate });
    window.history.replaceState(null, '', `?${params}`);
  } else window.history.replaceState(null, '', window.location.pathname);
}

function showLeaf(node) {
  select(node);
  focus = node.parent?.parent?.parent || node;
  mode = 'branch';
  layout();
}

function renderInspector() {
  $('selection-heading').innerHTML = `<span class="node-tag">${escapeText(nodeType(selected))}</span><h2>${escapeText(selected.label)}</h2>`;
  const info = [['Weight descendants', index.countWeights(selected)], ['Representation', selected.stored ? 'Stored node' : 'Generated address']];
  if (selected.expert !== undefined) info.unshift(['Expert', selected.expert]);
  if (selected.matrix) info.push(['Matrix', selected.matrix]);
  if (selected.row !== undefined) info.push(['Row', selected.row]);
  if (selected.group !== undefined) info.push(['Group', selected.group]);
  if (selected.kind === 'matrix') info.push(['Weight shape', selected.shape.join(' x ')]);
  let html = `<section class="node-summary">${detailRows(info)}</section>`;
  const next = index.children(selected);
  if (next.length) {
    html += `<section><h3>Branches <span>Left / right</span></h3><div class="branch-links">${next.map((child, position) =>
      `<button data-child="${position}"><small><i data-lucide="${position ? 'corner-down-right' : 'corner-down-left'}"></i>${position ? 'RIGHT' : 'LEFT'}</small><span>${escapeText(child.label)}</span></button>`).join('')}</div></section>`;
  }
  const leaf = index.leafData(selected);
  if (leaf) {
    const sample = leaf.sample;
    html += `<section><h3>Weight value</h3><div class="value-status ${sample ? '' : 'unknown'}"><i data-lucide="${sample ? 'circle-check' : 'circle-dashed'}"></i>${sample ? 'Verified checkpoint sample' : 'Value not loaded'}</div>`;
    if (sample) {
      const value = sample.float32_bits_hex === '80000000' ? '-0.0' : String(sample.weight);
      html += `<div class="weight-value">${escapeText(value)}</div>${detailRows([['Code',sample.code],['Actual scale byte',sample.scale],['Matches common scale',sample.scale_matches_root_default ? 'Yes' : 'No'],['Float32 bits',sample.float32_bits_hex,true]])}`;
    }
    html += `</section><section><h3>Source addresses <span>Byte offsets</span></h3>${detailRows([['Packed code',leaf.code_byte_offset,true],['Nibble shift',leaf.nibble_shift],['Scale',leaf.scale_byte_offset,true]])}</section>`;
  }
  if (selected.id === 'root') {
    const defaults = index.tree.defaults;
    html += `<section><h3>Most frequent values <span>Not universal</span></h3><div class="default-list">${[
      ['Scale byte',defaults.scale_byte,defaults.scale_occurrences,defaults.scale_total],
      ['Weight code',defaults.code,defaults.code_occurrences,defaults.code_total],
    ].map(([label,value,count,total]) => `<div><div class="default-item"><span>${label}</span><strong>${value}</strong><span>${(count/total*100).toFixed(2)}%</span></div><div class="frequency"><i style="width:${count/total*100}%"></i></div></div>`).join('')}</div></section>`;
  }
  const entries = Object.entries(selected.facts);
  const own = entries.filter(([key]) => Object.hasOwn(selected.own, key));
  const inherited = entries.filter(([key]) => !Object.hasOwn(selected.own, key));
  function facts(rows) {
    return rows.map(([key,value]) => {
      const origin = index.factOrigin(selected, key);
      return `<div class="fact"><div class="fact-main"><span class="fact-name">${escapeText(key.replaceAll('_',' '))}</span><span class="fact-value mono">${escapeText(format(value))}</span></div><span class="fact-origin">${origin === selected ? 'Stored here' : `From ${escapeText(origin?.label || 'root')}`}</span></div>`;
    }).join('');
  }
  if (own.length) html += `<section><h3>Stored here <span>${own.length} shared fields</span></h3>${facts(own)}</section>`;
  else html += '<section><h3>Stored here <span>0 shared fields</span></h3></section>';
  if (inherited.length) html += `<section><details open><summary>Inherited facts (${inherited.length})</summary>${facts(inherited)}</details></section>`;
  const matrix = selected.matrixNode;
  if (matrix && selected.kind !== 'weight') {
    const counts = matrix.raw.code_counts;
    const maximum = Math.max(...Object.values(counts));
    html += `<section><h3>Matrix code frequencies</h3><div class="bars">${Array.from({length:16},(_,code) => `<div title="Code ${code}: ${format(counts[code] || 0)}"><i style="height:${(counts[code] || 0)/maximum*49}px"></i><small>${code}</small></div>`).join('')}</div></section>`;
  }
  if (matrix) html += `<section><details><summary>Source tensor hashes</summary>${detailRows([['Packed codes',matrix.raw.tensors.weight_packed.sha256,true],['Scales',matrix.raw.tensors.weight_scale.sha256,true]])}</details></section>`;
  $('node-details').innerHTML = html;
  $('node-details').querySelectorAll('[data-child]').forEach(button => button.addEventListener('click', () => select(next[Number(button.dataset.child)], { focus:true })));
  $('sample-select').value = '';
  icons();
}

function renderBreadcrumbs() {
  const route = index.path(focus);
  const visible = route.length > 5 ? [route[0], ...route.slice(-4)] : route;
  $('breadcrumbs').innerHTML = visible.map((node, position) => `${position ? '<i data-lucide="chevron-right"></i>' : ''}<button data-path="${node.id}">${escapeText(position === 1 && route.length > 5 ? '... / ' + node.label : node.label)}</button>`).join('');
  $('breadcrumbs').querySelectorAll('[data-path]').forEach(button => button.addEventListener('click', () => select(index.nodes.get(button.dataset.path), { focus:true })));
  icons();
}

function layout() {
  if (!ready) return;
  const full = mode === 'full';
  $('branch-view').classList.toggle('active', !full);
  $('full-view').classList.toggle('active', full);
  $('branch-view').setAttribute('aria-selected', String(!full));
  $('full-view').setAttribute('aria-selected', String(full));
  $('depth-control').hidden = full;
  let hierarchy;
  if (full) {
    hierarchy = d3.hierarchy(index.root, node => node.kind === 'branch' && node.stored ? node.children : null);
    d3.tree().size([Math.PI * 2, 900]).separation(() => 1)(hierarchy);
    points = hierarchy.descendants().map(item => ({ node:item.data, x:Math.sin(item.x)*item.y, y:-Math.cos(item.x)*item.y, angle:item.x, radius:item.y, parent:null, item }));
  } else {
    hierarchy = d3.hierarchy(index.outline(focus, Number($('depth').value)), item => item.children);
    d3.tree().nodeSize([38,220])(hierarchy);
    points = hierarchy.descendants().map(item => ({ node:item.data.node, x:item.y, y:item.x, parent:null, item }));
  }
  const lookup = new Map(points.map(point => [point.item,point]));
  for (const point of points) point.parent = lookup.get(point.item.parent) || null;
  spatial = d3.quadtree(points, point => point.x, point => point.y);
  $('graph-heading').textContent = full ? 'All 896 experts' : focus.label;
  $('graph-type').textContent = full ? 'COMPLETE STORED HIERARCHY' : nodeType(focus).toUpperCase();
  $('graph-count').textContent = full ? '5,375 nodes / 2,688 matrix endpoints' : `${format(index.countWeights(focus))} weight descendants`;
  const generated = points.filter(point => !point.node.stored).length;
  $('render-status').textContent = `${format(points.length)} nodes in view${generated ? ` / ${generated} generated` : ''}`;
  canvas.dataset.nodes = String(points.length);
  canvas.dataset.mode = mode;
  canvas.setAttribute('aria-label', `${full ? 'Full layer 1 tree' : focus.label}: ${points.length} displayed nodes`);
  renderBreadcrumbs();
  fit();
}

function fit() {
  if (!ready || !points.length) return;
  const full = mode === 'full';
  const minX = d3.min(points, point => point.x) - (full ? 25 : 30);
  const maxX = d3.max(points, point => point.x) + (full ? 25 : 180);
  const minY = d3.min(points, point => point.y) - 35;
  const maxY = d3.max(points, point => point.y) + 35;
  const scale = Math.min(1.4, Math.max(.045, Math.min((viewportWidth - 70)/(maxX-minX), (viewportHeight - 170)/(maxY-minY))));
  const centerY = viewportHeight/2 + 20;
  const next = d3.zoomIdentity.translate(viewportWidth/2-(minX+maxX)*scale/2,centerY-(minY+maxY)*scale/2).scale(scale);
  d3.select(canvas).call(zoom.transform,next);
}

function draw() {
  if (!context || !ready || !transform) return;
  const ratio = window.devicePixelRatio || 1;
  context.setTransform(ratio,0,0,ratio,0,0);
  context.clearRect(0,0,viewportWidth,viewportHeight);
  context.save();
  context.translate(transform.x,transform.y);
  context.scale(transform.k,transform.k);
  const ancestors = new Set(index.path(selected).map(node => node.id));
  const full = mode === 'full';
  for (const point of points) {
    if (!point.parent) continue;
    const parent = point.parent;
    const active = ancestors.has(point.node.id);
    context.beginPath();
    context.moveTo(parent.x,parent.y);
    if (full) {
      const middle = (point.radius + parent.radius)/2;
      context.bezierCurveTo(Math.sin(parent.angle)*middle,-Math.cos(parent.angle)*middle,
        Math.sin(point.angle)*middle,-Math.cos(point.angle)*middle,point.x,point.y);
    } else {
      const middle = (point.x + parent.x)/2;
      context.bezierCurveTo(middle,parent.y,middle,point.y,point.x,point.y);
    }
    context.strokeStyle = active ? '#bc8b27' : full ? '#b5c9bbaa' : '#c2d1c3';
    context.lineWidth = (active ? 1.6 : full ? .65 : 1.05)/transform.k;
    context.stroke();
  }
  const labelBounds = [];
  const labelPriority = point => point.node === selected ? 0 : !point.item.children ? 1 : 2;
  const orderedPoints = [...points].sort((first,second) => labelPriority(first)-labelPriority(second));
  for (const point of orderedPoints) {
    const screenX = point.x * transform.k + transform.x;
    const screenY = point.y * transform.k + transform.y;
    if (screenX < -180 || screenX > viewportWidth+30 || screenY < -30 || screenY > viewportHeight+30) continue;
    const active = point.node === selected;
    const radius = (active ? 5.5 : full ? point.node.kind === 'matrix' ? 1.6 : 2.1 : 4)/transform.k;
    context.beginPath();
    context.arc(point.x,point.y,radius,0,Math.PI*2);
    context.fillStyle = color(point.node);
    context.fill();
    if (active) {
      context.beginPath(); context.arc(point.x,point.y,radius+4/transform.k,0,Math.PI*2);
      context.strokeStyle='#bf932f'; context.lineWidth=1.3/transform.k; context.stroke();
    }
    const showLabel = (!full && (transform.k >= .6 || !point.item.children)) || active || (full && transform.k > 2.5 && point.node.kind === 'matrix');
    if (showLabel) {
      const size = 11;
      context.font = `${active ? 600 : 400} ${size/transform.k}px Bahnschrift, Aptos, sans-serif`;
      const label = full && point.node.matrix ? `E${point.node.expert} / ${point.node.matrix}` : point.node.label;
      const width = context.measureText(label).width;
      const screenTextWidth = width * transform.k;
      let textX = point.x + 12/transform.k;
      if (screenX+12+screenTextWidth > viewportWidth-8) textX = point.x-width-12/transform.k;
      const bounds = { left:textX*transform.k+transform.x-2, right:textX*transform.k+transform.x+screenTextWidth+3,
        top:screenY-8, bottom:screenY+7 };
      const collision = labelBounds.some(other => bounds.left < other.right+3 && bounds.right > other.left-3
        && bounds.top < other.bottom && bounds.bottom > other.top);
      if (!collision && bounds.left >= 4 && bounds.right <= viewportWidth-4) {
        labelBounds.push(bounds);
        context.fillStyle='#f6f8f5ee';
        context.fillRect(textX-2/transform.k,point.y-8/transform.k,width+5/transform.k,15/transform.k);
        context.fillStyle=active ? '#153f2b' : '#43594a';
        context.fillText(label,textX,point.y+4/transform.k);
      }
    }
    if (!full && point.node.kind !== 'weight' && !point.item.children) {
      context.fillStyle=color(point.node);
      context.fillRect(point.x-1.2/transform.k,point.y+9/transform.k,2.4/transform.k,2.4/transform.k);
    }
  }
  context.restore();
  canvas.dataset.labels = JSON.stringify(labelBounds);
  $('zoom-level').textContent = `${Math.round(transform.k*100)}%`;
}

function hit(event) {
  if (!ready || !spatial) return null;
  const rect = canvas.getBoundingClientRect();
  const [worldX,worldY] = transform.invert([event.clientX-rect.left,event.clientY-rect.top]);
  return spatial.find(worldX,worldY,12/transform.k);
}

function setupGraph() {
  transform = d3.zoomIdentity;
  zoom = d3.zoom().scaleExtent([.025,16]).on('zoom', event => {
    transform = event.transform;
    $('node-tooltip').hidden = true;
    draw();
  });
  d3.select(canvas).call(zoom).on('dblclick.zoom',null);
  canvas.addEventListener('pointerdown', event => { pointerStart = [event.clientX,event.clientY]; });
  canvas.addEventListener('click', event => {
    if (pointerStart && Math.hypot(event.clientX-pointerStart[0],event.clientY-pointerStart[1]) > 6) return;
    const point = hit(event);
    if (point) select(point.node);
  });
  canvas.addEventListener('dblclick', event => { const point = hit(event); if (point) select(point.node,{focus:true}); });
  canvas.addEventListener('pointermove', event => {
    if (event.buttons) return;
    const point = hit(event);
    const tooltip = $('node-tooltip');
    tooltip.hidden = !point;
    canvas.style.cursor = point ? 'pointer' : 'grab';
    if (point) {
      const rect = canvas.getBoundingClientRect();
      tooltip.textContent = `${point.node.expert !== undefined ? `E${point.node.expert} / ` : ''}${point.node.label} · ${nodeType(point.node)}`;
      tooltip.style.left = `${Math.max(8,Math.min(event.clientX-rect.left+14,viewportWidth-245))}px`;
      tooltip.style.top = `${Math.max(8,Math.min(event.clientY-rect.top+14,viewportHeight-50))}px`;
    }
  });
  canvas.addEventListener('pointerleave', () => { $('node-tooltip').hidden = true; });
  canvas.addEventListener('keydown', event => {
    if (!ready) return;
    let destination;
    if (event.key === 'ArrowLeft') destination = selected.parent;
    if (event.key === 'ArrowRight') destination = index.children(selected)[0];
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      const siblings = selected.parent ? index.children(selected.parent) : [];
      destination = siblings[siblings.indexOf(selected)+(event.key === 'ArrowDown' ? 1 : -1)];
    }
    if (event.key === 'Enter') destination = selected;
    if (destination) { event.preventDefault(); select(destination,{focus:true}); }
  });
  new ResizeObserver(() => {
    const bounds = $('viewport').getBoundingClientRect();
    viewportWidth = bounds.width; viewportHeight = bounds.height;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(viewportWidth*ratio); canvas.height = Math.round(viewportHeight*ratio);
    if (ready) fit();
  }).observe($('viewport'));
}

$('locator').addEventListener('submit', event => {
  event.preventDefault();
  if (!ready) return;
  $('form-error').hidden = true;
  try { showLeaf(index.locate(Number($('expert').value),$('matrix').value,Number($('row').value),Number($('coordinate').value))); }
  catch (error) { $('form-error').textContent = error.message; $('form-error').hidden = false; }
});
$('matrix').addEventListener('change', updateBounds);
$('branch-view').addEventListener('click', () => { if (ready) { mode='branch'; layout(); } });
$('full-view').addEventListener('click', () => { if (ready) { mode='full'; layout(); } });
$('depth').addEventListener('input', () => { $('depth-value').value=$('depth').value; layout(); });
$('home').addEventListener('click', () => select(index.root,{focus:true}));
$('up').addEventListener('click', () => { if (selected.parent) select(selected.parent,{focus:true}); });
$('focus').addEventListener('click', () => select(selected,{focus:true,record:false}));
$('back').addEventListener('click', () => {
  const previous = history.pop();
  if (!previous) return;
  focus = previous.focus; mode = previous.mode;
  select(previous.selected,{record:false}); layout();
});
$('zoom-in').addEventListener('click', () => { if (ready) d3.select(canvas).call(zoom.scaleBy,1.5); });
$('zoom-out').addEventListener('click', () => { if (ready) d3.select(canvas).call(zoom.scaleBy,1/1.5); });
$('fit').addEventListener('click', fit);
$('sample-select').addEventListener('change', event => {
  if (event.target.value === '') return;
  const sample = [...index.samples.values()][Number(event.target.value)];
  showLeaf(index.locate(sample.expert,sample.matrix,sample.row,sample.coordinate));
});
$('export').addEventListener('click', () => {
  const data = { dataset:'layer1-routed-experts', node:selected.label, kind:nodeType(selected),
    path:index.path(selected).map(node => node.label), own_shared_facts:selected.own, inherited_and_own_facts:selected.facts,
    source_weight:index.leafData(selected), weight_descendants:index.countWeights(selected) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const anchor = document.createElement('a');
  anchor.href=url; anchor.download='layer1-selected-node.json'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});

async function load() {
  ready = false; $('load-state').hidden=false; $('retry').hidden=true;
  $('load-message').textContent='Loading verified tree';
  try {
    if (!globalThis.d3 || !globalThis.lucide) throw new Error('Local graph libraries unavailable.');
    const responses = await Promise.all([fetch('../layer1-binary-tree.json'),fetch('../layer1-binary-tree-checks.json')]);
    if (responses.some(response => !response.ok)) throw new Error('Source tree could not be loaded.');
    const [tree,checks] = await Promise.all(responses.map(response => response.json()));
    index = createTreeIndex(tree,checks);
    selected=index.root; focus=index.root; history=[]; ready=true;
    $('load-state').hidden=true;
    for (const name of ['export','home','focus','locate-button','sample-select']) $(name).disabled=false;
    $('sample-select').innerHTML='<option value="">Select recorded sample</option>'+[...index.samples.values()].map((sample,position) => `<option value="${position}">E${sample.expert} / ${sample.matrix} / row ${sample.row} / weight ${sample.coordinate}</option>`).join('');
    renderInspector(); layout();
    const query = new URLSearchParams(window.location.search);
    if (query.has('expert')) {
      try { showLeaf(index.locate(Number(query.get('expert')),query.get('matrix') || 'w1',Number(query.get('row') || 0),Number(query.get('coordinate') || 0))); }
      catch (error) { $('form-error').textContent=error.message; $('form-error').hidden=false; }
    }
    document.body.dataset.ready='true';
  } catch (error) {
    $('load-message').textContent=error.message;
    $('retry').hidden=false;
  }
}
$('retry').addEventListener('click',load);
icons();
setupGraph();
load();