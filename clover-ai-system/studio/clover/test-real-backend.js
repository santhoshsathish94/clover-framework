/* Real LFM3-Omni backend regression. Model/tool directories are read through junctions;
   conversations, media, configuration and logs stay in this workspace's temporary base. */
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const root = __dirname;
const installed = path.join(process.env.USERPROFILE || '', 'CloverAI');
const python = path.join(installed, 'venv', 'Scripts', 'python.exe');
const base = path.join(root, `test-real-base-${process.pid}`);
const artifacts = path.join(root, 'test-artifacts');
const port = 7101;
const origin = `http://127.0.0.1:${port}`;
const results = { checks: [], realModel: null, audioChatExercised: false };
let server;
let output = '';
let sessionId = '';

function pass(name) { results.checks.push(name); console.log(`PASS ${name}`); }
function saveResults() {
  fs.mkdirSync(artifacts, { recursive: true });
  fs.writeFileSync(path.join(artifacts, 'real-backend-results.json'), JSON.stringify(results, null, 2));
  fs.writeFileSync(path.join(artifacts, 'real-backend-server.log'), output);
}
function removeTemporaryBase(dir) {
  // Clover's temporary Git repository creates read-only object files on Windows.
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const target = path.join(dir, entry.name);
    if (entry.isSymbolicLink()) { fs.unlinkSync(target); continue; }
    try { fs.chmodSync(target, entry.isDirectory() ? 0o777 : 0o666); } catch (_) {}
    if (entry.isDirectory()) removeTemporaryBase(target);
  }
  try { fs.chmodSync(dir, 0o777); } catch (_) {}
  fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3, retryDelay: 250 });
}
async function request(route, body, timeout = 180000) {
  const response = await fetch(origin + route, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeout)
  });
  const type = response.headers.get('content-type') || '';
  const value = type.includes('json') ? await response.json() : Buffer.from(await response.arrayBuffer());
  if (!response.ok) throw new Error(`${route}: HTTP ${response.status}: ${JSON.stringify(value)}`);
  return value;
}
async function waitReady() {
  const deadline = Date.now() + 240000;
  let last = '';
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`Clover exited with code ${server.exitCode}`);
    try {
      const status = await request('/api/status', undefined, 5000);
      last = `${status.status || ''} ${status.step || ''}`;
      if (status.status === 'ready') return status;
      if (status.status === 'error' || status.status === 'setup') throw new Error(JSON.stringify(status));
    } catch (error) {
      if (/setup|engine is loading|failed \(.*model|status.*error/i.test(error.message)) throw error;
      last = error.message;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error(`Real model was not ready within 240 seconds (${last})`);
}
function validateWav(wav) {
  assert.ok(Buffer.isBuffer(wav) && wav.length > 2048, 'TTS returned too little audio');
  assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
  assert.equal(wav.toString('ascii', 8, 12), 'WAVE');
  assert.ok(wav.some((byte, index) => index >= 44 && byte !== 0), 'TTS samples are all zero');
}
async function audioChat(wav) {
  sessionId = `voice_real_${process.pid}`;
  const response = await fetch(origin + '/api/chat/stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: '', session_id: sessionId, model: 'omni', history: [],
      media: [{ kind: 'audio', name: 'real-tts-question.wav', data: wav.toString('base64') }],
      sampling: 'general', thinking: false, max_new_tokens: 128
    }),
    signal: AbortSignal.timeout(240000)
  });
  if (response.status !== 200) throw new Error(`Real audio chat HTTP ${response.status}: ${await response.text()}`);
  const sse = await response.text();
  const events = sse.split(/\r?\n\r?\n/).filter(line => line.startsWith('data: ')).map(line => JSON.parse(line.slice(6)));
  const error = events.find(event => event.type === 'error');
  if (error) throw new Error(`Real audio chat: ${error.error}`);
  const done = events.find(event => event.type === 'done');
  assert.ok(done && String(done.reply || '').trim(), `No reply in real audio chat: ${sse.slice(-2000)}`);
  results.audioChatExercised = true;
  results.audioReply = String(done.reply).slice(0, 1000);
  pass('Real LFM3-Omni accepted the generated speech as audio input and returned a reply');
}

(async () => {
  try {
    assert.ok(fs.existsSync(python), `Installed Clover Python is missing: ${python}`);
    fs.mkdirSync(base, { recursive: true });
    fs.symlinkSync(path.join(installed, 'data'), path.join(base, 'data'), 'junction');
    fs.symlinkSync(path.join(installed, 'tools'), path.join(base, 'tools'), 'junction');
    server = spawn(python, [path.join(root, 'server.py'), '--base', base, '--port', String(port), '--log', path.join(artifacts, 'real-backend-process.log')], {
      cwd: root, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true
    });
    server.stdout.on('data', data => { output += data; });
    server.stderr.on('data', data => { output += data; });
    const status = await waitReady();
    results.realModel = { model: status.model, engine: status.engine, inputs: status.inputs };
    assert.ok((status.inputs || []).includes('audio'));
    pass(`Real local model is ready with audio input (${status.engine || 'engine not named'})`);

    const health = await request('/api/health');
    assert.equal(health.status, 'ready');
    pass('Real /api/health reports ready');

    const parts = await request('/api/tts/parts', { text: 'Hello Clover. Please answer this spoken greeting with one short sentence.' });
    assert.ok(Array.isArray(parts.parts) && parts.parts.length > 0);
    pass('Real speech segmentation returns speakable parts');

    const wav = await request('/api/tts', { text: 'Hello Clover. Please answer this spoken greeting with one short sentence.', voice: 'female' });
    validateWav(wav);
    fs.writeFileSync(path.join(artifacts, 'real-tts-question.wav'), wav);
    results.ttsBytes = wav.length;
    pass(`Real LFM3-Omni synthesized a nonempty WAV (${wav.length.toLocaleString()} bytes)`);

    await audioChat(wav);
    saveResults();
    console.log('ALL REAL BACKEND CHECKS PASSED');
  } catch (error) {
    results.error = error.stack || String(error);
    saveResults();
    console.error(error);
    process.exitCode = 1;
  } finally {
    if (sessionId && server && server.exitCode === null) {
      try { await request('/api/session/forget', { session_id: sessionId, delete: true }, 10000); } catch (_) {}
    }
    if (server && server.exitCode === null) {
      try { await request('/api/shutdown', {}, 5000); } catch (_) {}
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (server.exitCode === null) {
        if (process.platform === 'win32') spawnSync('taskkill.exe', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
        else server.kill('SIGTERM');
      }
    }
    for (const name of ['data', 'tools']) {
      try { fs.unlinkSync(path.join(base, name)); } catch (_) {}
    }
    removeTemporaryBase(base);
    saveResults();
  }
})();
