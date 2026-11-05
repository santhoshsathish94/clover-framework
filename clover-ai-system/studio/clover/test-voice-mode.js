const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const root = __dirname;
const artifacts = path.join(root, 'test-artifacts');
const base = path.join(root, `test-runtime-base-${process.pid}`);
const port = 7117;
const origin = `http://127.0.0.1:${port}`;
fs.rmSync(artifacts, { recursive: true, force: true });
fs.mkdirSync(artifacts, { recursive: true });
// Presence markers bypass Setup in the isolated fixture; API mocks below supply all runtime behavior.
const gguf = path.join(base, 'data', 'omni', 'gguf');
fs.mkdirSync(gguf, { recursive: true });
fs.writeFileSync(path.join(gguf, 'LFM3-Omni-2B-Q4_0.gguf'), 'UI fixture only');
fs.writeFileSync(path.join(gguf, 'mmproj-LFM3-Omni-2B-Q8_0.gguf'), 'UI fixture only');

function wav(seconds, frequency = 0) {
  const rate = 16000, count = Math.floor(seconds * rate), data = Buffer.alloc(44 + count * 2);
  data.write('RIFF', 0); data.writeUInt32LE(36 + count * 2, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(rate, 24); data.writeUInt32LE(rate * 2, 28); data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(count * 2, 40);
  for (let i = 0; i < count; i++) data.writeInt16LE(frequency ? Math.round(Math.sin(i / rate * frequency * Math.PI * 2) * 10000) : 0, 44 + i * 2);
  return data;
}
const captureWav = path.join(artifacts, 'fake-microphone.wav');
const captureAudio = wav(5.4, 440);
captureAudio.fill(0, 44 + Math.floor(1.4 * 16000) * 2); // tone, then four seconds of silence
fs.writeFileSync(captureWav, captureAudio);
const spokenWav = wav(2.4, 220);

let server;
let browser;
const results = { checks: [], consoleErrors: [], pageErrors: [], failedRequests: [], chatBodies: [] };
function pass(name) { results.checks.push(name); console.log(`PASS ${name}`); }
async function waitServer() {
  for (let i = 0; i < 80; i++) {
    try { const r = await fetch(origin); if (r.ok) return; } catch (_) {}
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error('Clover test server did not start');
}
async function mockApi(page) {
  await page.route('**/api/**', async route => {
    const req = route.request(), url = new URL(req.url()), p = url.pathname;
    const json = value => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(value) });
    if (p === '/api/status') return json({ status: 'ready', model: 'LFM3-Omni-2B', model_key: 'omni', mode: 'lfm3', engine: 'gpu', inputs: ['text', 'vision', 'audio', 'video'], model_catalog: [{ id: 'omni', name: 'LFM3-Omni', badge: 'This laptop', desc: 'Local Omni', host: 'local', local: true }] });
    if (p === '/api/health') return json({ status: 'ready', mode: 'lfm3', model: 'LFM3-Omni-2B', engine: { engine: 'gpu', inputs: ['text', 'vision', 'audio', 'video'] } });
    if (p === '/api/sessions') return json({ sessions: [] });
    if (p === '/api/permissions') return json({ permissions: {} });
    if (p === '/api/chat/stream') {
      const body = JSON.parse(req.postData() || '{}'); results.chatBodies.push(body);
      const events = [
        { type: 'start', turn_id: 'voice-test-turn' },
        { type: 'token', kind: 'answer', text: 'I heard your voice. Clover is listening and talking back.' },
        { type: 'done', reply: 'I heard your voice. Clover is listening and talking back.', generation: [] }
      ];
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('') });
    }
    if (p === '/api/tts/parts') return json({ parts: ['I heard your voice.'] });
    if (p === '/api/tts') {
      await new Promise(resolve => setTimeout(resolve, 650));
      return route.fulfill({ status: 200, contentType: 'audio/wav', body: spokenWav });
    }
    return json({});
  });
}
function observe(page) {
  page.on('pageerror', e => results.pageErrors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') results.consoleErrors.push(m.text()); });
  page.on('requestfailed', r => results.failedRequests.push(`${r.url()} ${r.failure() && r.failure().errorText}`));
  page.on('response', r => { if (r.status() >= 400) results.failedRequests.push(`${r.status()} ${r.url()}`); });
}
async function instrumentTracks(page) {
  await page.evaluate(() => {
    window.__voiceTracks = [];
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      const stream = await original(constraints);
      stream.getTracks().forEach(t => window.__voiceTracks.push(t));
      return stream;
    };
  });
}

(async () => {
  try {
    server = spawn('python', [path.join(root, 'server.py'), '--base', base, '--port', String(port), '--log', path.join(artifacts, 'server.log')], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
    server.stdout.on('data', () => {}); server.stderr.on('data', d => process.stderr.write(d));
    await waitServer(); pass('Modified Clover server and static assets start on port 7117');

    browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${captureWav}%noloop`, '--enable-unsafe-swiftshader'] });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, permissions: ['microphone'] });
    const page = await context.newPage(); observe(page); await mockApi(page);
    await page.goto(origin, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    assert.equal(await page.locator('#voice-toggle').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.locator('#composer #voice-toggle').count(), 1);
    assert.equal((await page.locator('#voice-toggle').textContent()).trim(), '');
    assert.equal(await page.locator('#voice-toggle use').getAttribute('href'), '#i-mic'); pass('Icon-only microphone control is inside the composer beside Send');
    assert.equal(await page.locator('#voice-companion').isVisible(), false); pass('Voice mode and microphone default to off');
    await instrumentTracks(page);
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => CloverVoiceMode.getState().capturing && CloverVoiceMode.getState().phase === 'listening', null, { timeout: 10000 });
    await page.waitForFunction(() => document.querySelector('#robot-poster').complete && document.querySelector('#robot-poster').naturalWidth > 0);
    assert.equal(await page.locator('#robot-canvas').count(), 0);
    assert.equal(await page.evaluate(() => typeof window.CloverRobot), 'undefined');
    results.listeningVisual = await page.evaluate(() => {
      const poster = document.querySelector('#robot-poster');
      const status = document.querySelector('.voice-companion__status');
      const canvas = document.createElement('canvas');
      canvas.width = poster.naturalWidth; canvas.height = poster.naturalHeight;
      const ctx = canvas.getContext('2d'); ctx.drawImage(poster, 0, 0);
      const corners = [[0, 0], [canvas.width - 1, 0], [0, canvas.height - 1], [canvas.width - 1, canvas.height - 1]];
      return {
        robotOpacity: getComputedStyle(poster).opacity,
        robotAnimation: getComputedStyle(poster).animationName,
        robotSource: poster.getAttribute('src'),
        transparentCorners: corners.map(([x, y]) => ctx.getImageData(x, y, 1, 1).data[3]),
        panelBackground: getComputedStyle(document.querySelector('#voice-companion')).backgroundColor,
        statusBackground: getComputedStyle(status).backgroundColor,
        statusBackdrop: getComputedStyle(status).backdropFilter,
        orbitAnimation: getComputedStyle(document.querySelector('.voice-orbit__runner')).animationName,
        pauseMs: CloverVoiceMode.getState().pauseMs
      };
    });
    assert.equal(results.listeningVisual.robotOpacity, '0.58');
    assert.equal(results.listeningVisual.robotAnimation, 'none');
    assert.equal(results.listeningVisual.robotSource, 'robo-clover-transparent.webp');
    assert.ok(results.listeningVisual.transparentCorners.every(alpha => alpha === 0));
    assert.equal(results.listeningVisual.panelBackground, 'rgba(0, 0, 0, 0)');
    assert.equal(results.listeningVisual.statusBackground, 'rgba(0, 0, 0, 0)');
    assert.equal(results.listeningVisual.statusBackdrop, 'none');
    assert.equal(results.listeningVisual.pauseMs, 3000);
    assert.equal(results.listeningVisual.orbitAnimation, 'voice-orbit-clockwise'); pass('Transparent static robot and status retain only the clockwise voice ring');
    await page.screenshot({ path: path.join(artifacts, 'voice-desktop.png'), fullPage: true });
    await page.waitForFunction(() => CloverVoiceMode.getState().elapsed >= 3200, null, { timeout: 10000 });
    assert.equal(await page.evaluate(() => CloverVoiceMode.getState().capturing), true);
    assert.equal(results.chatBodies.length, 0); pass('Voice keeps listening after 3.2 seconds instead of cutting off a natural pause');
    const audioResponse = page.waitForResponse('**/api/chat/stream', { timeout: 20000 });
    await page.waitForFunction(() => CloverVoiceMode.getState().phase === 'thinking', null, { timeout: 20000 });
    await audioResponse;
    assert.equal(results.chatBodies.length, 1);
    assert.equal(results.chatBodies[0].media.length, 1);
    assert.equal(results.chatBodies[0].media[0].kind, 'audio');
    assert.equal(results.chatBodies[0].media[0].name.startsWith('Voice conversation'), true);
    assert.ok(results.chatBodies[0].media[0].data.length > 1000); pass('The completed utterance still becomes a real WAV audio request');
    await page.waitForFunction(() => CloverVoiceMode.getState().speaking, null, { timeout: 10000 });
    assert.equal(await page.locator('#log .u').count(), 0);
    assert.equal(await page.locator('#log .b').count(), 0);
    assert.equal(await page.locator('#hero').isVisible(), true);
    assert.equal(await page.locator('#chat-title').textContent(), 'Voice conversation');
    assert.equal((await page.locator('#log').textContent()).includes('reading your message'), false); pass('Continuous Voice hides transcript and read-aloud turns while retaining the session');
    await page.waitForFunction(() => CloverVoiceMode.getState().speaking && CloverVoiceMode.getState().phase === 'thinking', null, { timeout: 10000 });
    results.preparingVisual = await page.evaluate(() => getComputedStyle(document.querySelector('.voice-orbit__runner')).animationName);
    assert.equal(results.preparingVisual, 'none'); pass('Ring and robot stay still while speech audio is being prepared');
    await page.waitForFunction(() => CloverVoiceMode.getState().phase === 'speaking' && getComputedStyle(document.querySelector('.voice-orbit__runner')).animationName === 'voice-orbit-clockwise', null, { timeout: 10000 });
    assert.equal(await page.locator('#voice-title').textContent(), 'Speaking…');
    results.speakingVisual = await page.evaluate(() => {
      const animation = document.querySelector('.voice-orbit__runner').getAnimations()[0];
      return animation ? animation.effect.getKeyframes().map(frame => frame.transform) : [];
    });
    assert.ok(results.speakingVisual.some(value => /rotate\(360deg\)/.test(value))); pass('Only the white ring moves, clockwise through 360 degrees during playback');
    await page.screenshot({ path: path.join(artifacts, 'voice-speaking.png'), fullPage: true });
    await page.locator('#voice-companion').screenshot({ path: path.join(artifacts, 'robot-panel.png') });
    await page.waitForFunction(() => CloverVoiceMode.getState().capturing && CloverVoiceMode.getState().phase === 'listening', null, { timeout: 15000 });
    results.finishedVisual = await page.evaluate(() => getComputedStyle(document.querySelector('.voice-orbit__runner')).animationName);
    assert.equal(results.finishedVisual, 'voice-orbit-clockwise'); pass('Listening and the clockwise voice ring resume after spoken playback');
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => !CloverVoiceMode.getState().active);
    const trackStates = await page.evaluate(() => window.__voiceTracks.map(t => t.readyState));
    assert.ok(trackStates.length > 0 && trackStates.every(s => s === 'ended'));
    assert.equal(await page.locator('#voice-companion').isVisible(), false); pass('Disabling Voice stops every microphone track and hides the robot');
    await context.close();

    const textContext = await browser.newContext({ viewport: { width: 1200, height: 850 }, permissions: ['microphone'] });
    const tp = await textContext.newPage(); observe(tp); await mockApi(tp);
    await tp.goto(origin, { waitUntil: 'networkidle' });
    await tp.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    await tp.locator('#prompt').fill('Hello in text');
    await tp.locator('#send').click();
    await tp.waitForSelector('#log .b', { timeout: 10000 });
    assert.equal(await tp.locator('#log .u').count(), 1);
    assert.match(await tp.locator('#log .u').textContent(), /Hello in text/);
    assert.equal(await tp.locator('#hero').isVisible(), false);
    assert.equal(await tp.locator('#chat-title').textContent(), 'Hello in text'); pass('Ordinary text chat still renders its user and assistant turns normally');
    const sendBox = await tp.locator('#send').boundingBox();
    await tp.mouse.move(sendBox.x + sendBox.width / 2, sendBox.y + sendBox.height / 2);
    await tp.mouse.down(); await tp.waitForTimeout(1800); await tp.mouse.up();
    await tp.waitForSelector('#log .u .media', { timeout: 10000 });
    await tp.waitForFunction(() => document.querySelectorAll('#log .b').length === 2, null, { timeout: 10000 });
    assert.equal(await tp.locator('#log .u').count(), 2);
    assert.equal(await tp.locator('#log .b').count(), 2); pass('Hold-to-speak remains a visible voice message, separate from continuous Voice');
    await textContext.close();

    const mobile = await browser.newContext({ viewport: { width: 375, height: 812 }, permissions: ['microphone'], isMobile: true, hasTouch: true });
    const mp = await mobile.newPage(); observe(mp); await mockApi(mp);
    await mp.goto(origin, { waitUntil: 'networkidle' });
    await mp.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    await mp.locator('#voice-toggle').click();
    await mp.waitForFunction(() => CloverVoiceMode.getState().capturing, null, { timeout: 10000 });
    const overflow = await mp.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    assert.equal(overflow, false);
    await mp.screenshot({ path: path.join(artifacts, 'voice-mobile.png'), fullPage: true }); pass('375 px layout has no horizontal overflow and keeps Voice accessible');
    await mp.locator('#voice-toggle').click(); await mobile.close();

    const denied = await browser.newContext({ viewport: { width: 1100, height: 800 } });
    const dp = await denied.newPage(); observe(dp); await mockApi(dp);
    await dp.addInitScript(() => {
      const reject = () => Promise.reject(new DOMException('Permission denied by test', 'NotAllowedError'));
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: reject } });
    });
    await dp.goto(origin, { waitUntil: 'networkidle' });
    await dp.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    await dp.locator('#voice-toggle').click();
    await dp.waitForFunction(() => CloverVoiceMode.getState().failed);
    assert.match(await dp.locator('#voice-detail').textContent(), /permission was not granted/i);
    await dp.locator('#voice-toggle').click();
    assert.equal(await dp.locator('#voice-companion').isVisible(), false); pass('Microphone denial is explained and can be dismissed without breaking chat');
    await denied.close();

    assert.deepEqual(results.pageErrors, []);
    assert.deepEqual(results.consoleErrors, []);
    assert.deepEqual(results.failedRequests, []); pass('No browser exceptions, console errors, failed requests, or 404s');
    fs.writeFileSync(path.join(artifacts, 'results.json'), JSON.stringify(results, null, 2));
    console.log('ALL VOICE MODE CHECKS PASSED');
  } catch (error) {
    fs.writeFileSync(path.join(artifacts, 'results.json'), JSON.stringify(results, null, 2));
    console.error(error);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    if (server && server.exitCode === null) server.kill('SIGTERM');
  }
})();
