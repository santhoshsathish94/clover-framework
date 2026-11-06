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
async function mockApi(page, options = {}) {
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
    if (p === '/api/tts/warm') { results.ttsWarmRequests = (results.ttsWarmRequests || 0) + 1; return json({}); }
    if (p === '/api/tts') {
      results.ttsBodies = results.ttsBodies || [];
      results.ttsBodies.push(JSON.parse(req.postData() || '{}'));
      await new Promise(resolve => setTimeout(resolve, 650));
      return route.fulfill({ status: 200, contentType: 'audio/wav', body: options.invalidSpeech ? 'invalid audio fixture' : spokenWav });
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
    await page.addInitScript(() => localStorage.setItem('clover_studio_settings_v1', JSON.stringify({ autospeak: false })));
    await page.goto(origin, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    assert.equal(await page.locator('#voice-toggle').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.locator('#composer #voice-toggle').count(), 0);
    assert.equal(await page.locator('.bar__side.end #voice-toggle').count(), 1);
    assert.equal(await page.locator('#voice-toggle').evaluate(button => button.nextElementSibling.id), 'set-btn');
    assert.equal(await page.locator('#voice-toggle img').getAttribute('src'), 'favicon.svg'); pass('Clover flower opens conversation mode immediately beside Settings');
    assert.equal(await page.locator('#app').isVisible(), true);
    assert.equal(await page.locator('#voice-companion').isVisible(), false); pass('Chat is the default and conversation capture defaults to off');
    await instrumentTracks(page);
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => CloverVoiceMode.getState().capturing && CloverVoiceMode.getState().phase === 'listening', null, { timeout: 10000 });
    await page.waitForFunction(() => {
      const frame = document.querySelector('#robot-frame');
      return frame.contentWindow && frame.contentWindow.__SCENE_READY__ === true;
    }, null, { timeout: 15000 });
    results.listeningVisual = await page.evaluate(() => {
      const frame = document.querySelector('#robot-frame');
      const robot = frame.contentWindow;
      const panel = document.querySelector('#voice-companion');
      const rect = panel.getBoundingClientRect();
      return {
        ...robot.__getSceneState(),
        robotOpacity: getComputedStyle(frame).opacity,
        robotSource: frame.getAttribute('src'),
        robotResources: robot.performance.getEntriesByType('resource').map(entry => entry.name),
        panelBackground: getComputedStyle(panel).backgroundColor,
        panelWidth: rect.width,
        panelHeight: rect.height,
        appVisibility: getComputedStyle(document.querySelector('#app')).visibility,
        appInert: document.querySelector('#app').inert,
        statusPointer: getComputedStyle(document.querySelector('.voice-companion__status')).pointerEvents,
        resultsPointer: getComputedStyle(document.querySelector('#voice-results')).pointerEvents,
        resultsBackground: getComputedStyle(document.querySelector('#voice-results')).backgroundColor,
        toggleParent: document.querySelector('#voice-toggle').parentElement.id,
        toggleVisible: !!document.querySelector('#voice-toggle').offsetParent,
        toggleSymbol: document.querySelector('#voice-toggle img').getAttribute('src'),
        orbitCount: document.querySelectorAll('.voice-orbit').length,
        pauseMs: CloverVoiceMode.getState().pauseMs
      };
    });
    assert.equal(results.listeningVisual.ready, true);
    assert.equal(results.listeningVisual.robotSource, 'robot-viewer.html');
    assert.ok(results.listeningVisual.robotResources.some(url => url === `${origin}/clover-robo.js`));
    assert.ok(results.listeningVisual.robotResources.every(url => new URL(url).origin === origin));
    assert.equal(results.listeningVisual.canvasWidth, 1440);
    assert.equal(results.listeningVisual.canvasHeight, 960);
    assert.equal(results.listeningVisual.panelWidth, 1440);
    assert.equal(results.listeningVisual.panelHeight, 960);
    assert.ok(results.listeningVisual.calls > 0 && results.listeningVisual.triangles > 0);
    assert.equal(results.listeningVisual.canvasAlpha, true);
    assert.equal(results.listeningVisual.backgroundTransparent, true);
    assert.equal(results.listeningVisual.groundVisible, false);
    assert.equal(results.listeningVisual.distantVisible, false);
    assert.equal(results.listeningVisual.contactShadowVisible, false);
    assert.equal(results.listeningVisual.robotOpacity, '1');
    assert.equal(results.listeningVisual.panelBackground, 'rgba(0, 0, 0, 0)');
    assert.equal(results.listeningVisual.appVisibility, 'hidden');
    assert.equal(results.listeningVisual.appInert, true);
    assert.equal(results.listeningVisual.statusPointer, 'none');
    assert.equal(results.listeningVisual.resultsPointer, 'none');
    assert.equal(results.listeningVisual.resultsBackground, 'rgba(0, 0, 0, 0)');
    assert.equal(results.listeningVisual.toggleParent, 'voice-companion');
    assert.equal(results.listeningVisual.toggleVisible, true);
    assert.equal(results.listeningVisual.toggleSymbol, 'favicon.svg');
    assert.equal(results.listeningVisual.orbitCount, 0);
    assert.equal(results.listeningVisual.pauseMs, 3000); pass('Full-screen robot keeps a visible Clover toggle and transparent, non-blocking status/results overlays');
    await page.screenshot({ path: path.join(artifacts, 'voice-desktop.png'), fullPage: true });
    await page.waitForFunction(() => CloverVoiceMode.getState().elapsed >= 3200, null, { timeout: 10000 });
    assert.equal(await page.evaluate(() => CloverVoiceMode.getState().capturing), true);
    assert.equal(results.chatBodies.length, 0); pass('Voice keeps listening after 3.2 seconds instead of cutting off a natural pause');
    const audioResponse = page.waitForResponse('**/api/chat/stream', { timeout: 20000 });
    await page.waitForFunction(() => CloverVoiceMode.getState().phase === 'thinking', null, { timeout: 20000 });
    await audioResponse;
    results.thinkingReaction = await page.evaluate(() => document.querySelector('#robot-frame').contentWindow.__getSceneState().reaction);
    assert.equal(results.thinkingReaction, 'thinking'); pass('Thinking state selects the supplied robot’s thinking reaction');
    assert.equal(results.chatBodies.length, 1);
    assert.equal(results.chatBodies[0].media.length, 1);
    assert.equal(results.chatBodies[0].media[0].kind, 'audio');
    assert.equal(results.chatBodies[0].media[0].name.startsWith('Voice conversation'), true);
    assert.ok(results.chatBodies[0].media[0].data.length > 1000); pass('The completed utterance still becomes a real WAV audio request');
    assert.equal(results.chatBodies[0].conversation, true);
    assert.equal(results.chatBodies[0].max_new_tokens, 256);
    assert.equal(results.ttsWarmRequests || 0, 0); pass('Conversation requests concise replies without competing speech-engine warm-up');
    await page.waitForFunction(() => CloverVoiceMode.getState().speaking, null, { timeout: 10000 });
    await page.waitForFunction(() => document.querySelector('#voice-results').textContent.includes('I heard your voice. Clover is listening and talking back.'));
    assert.equal(await page.locator('#voice-results .voice-companion__turn--user').count(), 1);
    assert.equal(await page.locator('#voice-results .voice-companion__turn--assistant').count(), 1);
    assert.match(await page.locator('#voice-results .voice-companion__turn--user').textContent(), /Voice message/);
    assert.equal(await page.locator('#log .u').count(), 1);
    assert.equal(await page.locator('#log .b').count(), 1);
    assert.equal(await page.locator('#hero').getAttribute('hidden'), '');
    assert.equal(await page.locator('#chat-title').textContent(), 'Voice conversation');
    assert.equal((await page.locator('#log').textContent()).includes('reading your message'), false); pass('Voice turn and streaming reply appear over the robot and remain in the session transcript');
    await page.waitForFunction(() => CloverVoiceMode.getState().speaking && CloverVoiceMode.getState().phase === 'thinking', null, { timeout: 10000 });
    results.preparingVisual = await page.evaluate(() => document.querySelector('#robot-frame').contentWindow.__getSceneState().reaction);
    assert.equal(results.preparingVisual, 'thinking'); pass('Robot keeps its thinking reaction while speech audio is prepared');
    await page.waitForFunction(() => CloverVoiceMode.getState().phase === 'speaking', null, { timeout: 10000 });
    assert.equal(await page.locator('#voice-title').textContent(), 'Speaking…');
    assert.equal(results.ttsBodies[0].conversation, true); pass('Conversation speech asks the backend to reuse the resident model');
    assert.equal(await page.locator('#s-autospeak').isChecked(), false);
    results.speakingVisual = await page.evaluate(() => document.querySelector('#robot-frame').contentWindow.__getSceneState().reaction);
    assert.equal(results.speakingVisual, 'answering'); pass('Conversation speaks even when ordinary voice-message autoplay is off and selects the answering reaction');
    await page.screenshot({ path: path.join(artifacts, 'voice-speaking.png'), fullPage: true });
    await page.locator('#voice-companion').screenshot({ path: path.join(artifacts, 'robot-panel.png') });
    await page.waitForFunction(() => CloverVoiceMode.getState().capturing && CloverVoiceMode.getState().phase === 'listening', null, { timeout: 15000 });
    pass('Listening resumes after spoken playback');
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => !CloverVoiceMode.getState().active);
    assert.equal(await page.locator('#app').isVisible(), true);
    assert.equal(await page.locator('#app').evaluate(app => app.inert), false);
    assert.equal(await page.locator('#log .u').count(), 1);
    assert.equal(await page.locator('#log .b').count(), 1);
    assert.equal(await page.locator('#voice-toggle').evaluate(button => button.nextElementSibling.id), 'set-btn');
    assert.equal(await page.locator('#robot-frame').getAttribute('src'), 'about:blank');
    assert.equal(await page.locator('#voice-companion').isVisible(), false); pass('The same Clover symbol returns to chat, restores the transcript, and releases the robot');
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => document.querySelector('#robot-frame').contentWindow.__SCENE_READY__ === true, null, { timeout: 15000 });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !CloverVoiceMode.getState().active);
    assert.equal(await page.locator('#app').isVisible(), true); pass('Escape still returns to chat after toggling conversation back on');
    await page.locator('#voice-toggle').click();
    await page.waitForFunction(() => document.querySelector('#robot-frame').contentWindow.__SCENE_READY__ === true, null, { timeout: 15000 });
    await page.frameLocator('#robot-frame').locator('canvas').dispatchEvent('dblclick');
    await page.waitForFunction(() => !CloverVoiceMode.getState().active);
    const trackStates = await page.evaluate(() => window.__voiceTracks.map(t => t.readyState));
    assert.ok(trackStates.length > 0 && trackStates.every(s => s === 'ended'));
    assert.equal(await page.locator('#app').isVisible(), true); pass('Double-clicking the interactive robot also returns to chat and stops every microphone track');
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
    assert.equal(results.chatBodies[1].conversation, false);
    assert.equal(results.chatBodies[1].max_new_tokens, 2048);
    assert.equal(results.ttsWarmRequests || 0, 0); pass('Ordinary text chat keeps its full response budget');
    const sendBox = await tp.locator('#send').boundingBox();
    await tp.mouse.move(sendBox.x + sendBox.width / 2, sendBox.y + sendBox.height / 2);
    await tp.mouse.down(); await tp.waitForTimeout(1800); await tp.mouse.up();
    await tp.waitForSelector('#log .u .media', { timeout: 10000 });
    await tp.waitForFunction(() => document.querySelectorAll('#log .b').length === 2, null, { timeout: 10000 });
    assert.equal(await tp.locator('#log .u').count(), 2);
    assert.equal(await tp.locator('#log .b').count(), 2);
    assert.equal(results.chatBodies[2].conversation, false);
    assert.equal(results.chatBodies[2].max_new_tokens, 2048);
    assert.equal(results.ttsWarmRequests, 1); pass('Hold-to-speak remains a visible voice message with its existing warm-up, separate from continuous Voice');
    await textContext.close();

    const mobile = await browser.newContext({ viewport: { width: 375, height: 812 }, permissions: ['microphone'], isMobile: true, hasTouch: true });
    const mp = await mobile.newPage(); observe(mp); await mockApi(mp);
    await mp.goto(origin, { waitUntil: 'networkidle' });
    await mp.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    await mp.locator('#voice-toggle').click();
    await mp.waitForFunction(() => CloverVoiceMode.getState().capturing, null, { timeout: 10000 });
    await mp.waitForFunction(() => document.querySelector('#robot-frame').contentWindow.__SCENE_READY__ === true, null, { timeout: 15000 });
    const mobileLayout = await mp.evaluate(() => ({
      overflow: document.documentElement.scrollWidth > window.innerWidth,
      width: document.querySelector('#robot-frame').contentWindow.__getSceneState().canvasWidth,
      height: document.querySelector('#robot-frame').contentWindow.__getSceneState().canvasHeight,
      appVisibility: getComputedStyle(document.querySelector('#app')).visibility
    }));
    assert.equal(mobileLayout.overflow, false);
    assert.equal(mobileLayout.width, 375);
    assert.equal(mobileLayout.height, 812);
    assert.equal(mobileLayout.appVisibility, 'hidden');
    await mp.screenshot({ path: path.join(artifacts, 'voice-mobile.png'), fullPage: true }); pass('375 px conversation keeps the robot, Clover toggle, and transparent status without overflow');
    await mp.frameLocator('#robot-frame').locator('canvas').dispatchEvent('dblclick');
    await mp.waitForFunction(() => !CloverVoiceMode.getState().active); await mobile.close();

    const speechFailure = await browser.newContext({ viewport: { width: 1100, height: 800 }, permissions: ['microphone'] });
    const sp = await speechFailure.newPage(); observe(sp); await mockApi(sp, { invalidSpeech: true });
    await sp.goto(origin, { waitUntil: 'networkidle' });
    await sp.waitForFunction(() => window.CloverVoiceBridge && CloverVoiceBridge.getState().ready);
    await sp.locator('#voice-toggle').click();
    await sp.waitForFunction(() => document.querySelector('#voice-results').textContent.includes('Speech unavailable:'), null, { timeout: 25000 });
    await sp.waitForFunction(() => CloverVoiceMode.getState().capturing && CloverVoiceMode.getState().phase === 'listening', null, { timeout: 10000 });
    assert.equal(await sp.evaluate(() => CloverVoiceMode.getState().failed), false);
    assert.equal(await sp.locator('#voice-toggle').isVisible(), true);
    assert.match(await sp.locator('#voice-results').textContent(), /Speech unavailable:.*Listening again/);
    await sp.locator('#voice-toggle').click();
    await sp.waitForFunction(() => !CloverVoiceMode.getState().active); pass('Speech playback failure is visible and automatically returns to listening without trapping conversation mode');
    await speechFailure.close();

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
    await dp.waitForFunction(() => document.querySelector('#robot-frame').contentWindow.__SCENE_READY__ === true, null, { timeout: 15000 });
    await dp.keyboard.press('Escape');
    await dp.waitForFunction(() => !CloverVoiceMode.getState().active);
    assert.equal(await dp.locator('#app').isVisible(), true);
    assert.equal(await dp.locator('#voice-companion').isVisible(), false); pass('Microphone denial is announced accessibly and Escape restores chat');
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
