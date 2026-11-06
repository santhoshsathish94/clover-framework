/* Continuous, opt-in voice mode for Clover AI Studio.
   Audio is sent to the selected local LFM3-Omni engine through Clover's existing chat pipeline.
   No speech-recognition cloud service is used. */
(function () {
  'use strict';
  var active = false, speaking = false, failed = false, processing = false, phase = 'off';
  var capture = null, serial = 0, monitor = null;
  var button, panel, title, detail, robotFrame, results, answerLine, robotReaction = '';

  function bridge() { return window.CloverVoiceBridge || null; }
  function syncRobot(force) {
    if (!active || !robotFrame) return;
    var reactions = { starting: 'wave', waiting: 'sleepy', thinking: 'thinking', speaking: 'answering' };
    var reaction = reactions[phase];
    if (!reaction) { robotReaction = ''; return; }
    if (!force && reaction === robotReaction) return;
    try {
      var robot = robotFrame.contentWindow;
      if (robot && robot.__SCENE_READY__ === true && typeof robot.__triggerReaction === 'function') {
        robot.__triggerReaction(reaction);
        robotReaction = reaction;
      }
    } catch (e) { robotReaction = ''; }
  }
  function loadRobot() {
    if (!robotFrame || robotFrame.getAttribute('src') !== 'about:blank') return;
    robotReaction = '';
    robotFrame.src = robotFrame.dataset.src;
  }
  function addTurn(role, text) {
    if (!results) return null;
    var line = document.createElement('p');
    line.className = 'voice-companion__turn voice-companion__turn--' + role;
    var label = document.createElement('span');
    label.className = 'voice-companion__turn-label';
    label.textContent = role === 'user' ? 'You' : 'Clover';
    var content = document.createElement('span');
    content.className = 'voice-companion__turn-text';
    content.textContent = text || '';
    line.appendChild(label); line.appendChild(content); results.appendChild(line);
    while (results.children.length > 8) results.removeChild(results.firstChild);
    return line;
  }
  function showUser(text) { answerLine = null; addTurn('user', text); }
  function showAnswer(text) {
    if (!answerLine || !answerLine.isConnected) answerLine = addTurn('assistant', '');
    if (answerLine) answerLine.querySelector('.voice-companion__turn-text').textContent = text || '';
  }
  function setPhase(next, text) {
    phase = next;
    if (!button || !panel) return;
    var labels = {
      off: 'Voice mode off', waiting: 'Waiting for Clover…', starting: 'Opening microphone…',
      listening: 'Listening…', thinking: 'Thinking…', speaking: 'Speaking…', error: 'Microphone error'
    };
    var details = {
      off: '', waiting: 'Voice will begin when the local model is ready.', starting: 'Allow microphone access when your browser asks.',
      listening: 'Speak naturally. Pause for three seconds when you finish.',
      thinking: 'Microphone paused while Clover answers.', speaking: 'Microphone paused until Clover finishes speaking.',
      error: text || 'Microphone access failed. Toggle Voice off and try again.'
    };
    button.dataset.phase = next;
    button.setAttribute('aria-label', active ? 'Leave Clover conversation' : 'Start Clover conversation');
    button.title = active ? 'Return to chat' : 'Talk with Clover';
    panel.dataset.phase = next;
    title.textContent = labels[next] || 'Voice mode';
    detail.textContent = text || details[next] || '';
    syncRobot(false);
  }

  function stopTracks(stream) {
    if (stream) stream.getTracks().forEach(function (track) { track.stop(); });
  }

  function cancelCapture() {
    serial++;
    var current = capture;
    capture = null;
    if (!current) return;
    current.discard = true;
    current.stopped = true;
    clearInterval(current.meter);
    stopTracks(current.stream);
    if (current.audioContext) current.audioContext.close().catch(function () {});
    if (current.recorder && current.recorder.state !== 'inactive') {
      try { current.recorder.stop(); } catch (e) {}
    }
  }

  function finishCapture(sendAudio) {
    var current = capture;
    if (!current || current.stopped) return;
    current.stopped = true;
    current.discard = !sendAudio;
    capture = null;
    clearInterval(current.meter);
    stopTracks(current.stream);
    if (current.audioContext) current.audioContext.close().catch(function () {});
    if (current.recorder.state !== 'inactive') current.recorder.stop();
  }

  function startCapture() {
    var b = bridge();
    if (!active || failed || capture || speaking || !b) return;
    var st = b.getState();
    if (!st.ready || st.busy || !st.omni || !st.audio) {
      setPhase(st.busy ? 'thinking' : 'waiting');
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      failed = true; setPhase('error', 'This window does not support audio recording. Hold-to-speak and text chat remain available.'); return;
    }
    var token = ++serial;
    setPhase('starting');
    navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }, video: false
    }).then(function (stream) {
      if (!active || token !== serial) { stopTracks(stream); return; }
      var recorder;
      try { recorder = new MediaRecorder(stream); } catch (e) {
        stopTracks(stream); failed = true; setPhase('error', e.message); return;
      }
      var chunks = [], startedAt = Date.now(), heardAt = 0, lastSound = 0;
      var AC = window.AudioContext || window.webkitAudioContext;
      var audioContext = null, analyser = null, samples = null;
      if (AC) {
        try {
          audioContext = new AC(); analyser = audioContext.createAnalyser(); analyser.fftSize = 1024;
          audioContext.createMediaStreamSource(stream).connect(analyser);
          samples = new Float32Array(analyser.fftSize);
          audioContext.resume().catch(function () {});
        } catch (e) { audioContext = null; analyser = null; }
      }
      var current = { stream: stream, recorder: recorder, audioContext: audioContext, meter: null, stopped: false, discard: false };
      capture = current;
      recorder.ondataavailable = function (ev) { if (ev.data && ev.data.size) chunks.push(ev.data); };
      recorder.onerror = function () { failed = true; cancelCapture(); setPhase('error', 'The microphone reported a recording error.'); };
      recorder.onstop = function () {
        if (current.discard || !active || token !== serial || !chunks.length) return;
        var seconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
        processing = true;
        setPhase('thinking', 'Voice captured. Sending it to Clover.');
        new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }).arrayBuffer()
          .then(b.toWav16k)
          .then(function (wav) {
            if (!active || token !== serial) return;
            b.sendAudio(wav, seconds);
            processing = false;
          })
          .catch(function (e) {
            processing = false;
            if (!active || token !== serial) return;
            failed = true; setPhase('error', 'Could not read the recording: ' + e.message);
          });
      };
      recorder.start(250);
      setPhase('listening');
      // Leave room for natural mid-sentence pauses before completing an utterance.
      var PAUSE_MS = 3000, MAX_UTTERANCE_MS = 45000, NO_SPEECH_MS = 20000;
      current.pauseMs = PAUSE_MS;
      current.meter = setInterval(function () {
        if (!capture || capture !== current || current.stopped) return;
        var now = Date.now(), elapsed = now - startedAt, rms = 0;
        if (analyser && samples) {
          analyser.getFloatTimeDomainData(samples);
          var sum = 0;
          for (var i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
          rms = Math.sqrt(sum / samples.length);
        }
        current.rms = rms; current.heard = !!heardAt; current.elapsed = elapsed;
        if (elapsed > 300 && rms > 0.018) {
          if (!heardAt) heardAt = now;
          lastSound = now;
          detail.textContent = 'Hearing you… take your time.';
        }
        if (heardAt && lastSound && now - heardAt >= 400 && now - lastSound >= PAUSE_MS) {
          finishCapture(true); return;
        }
        if (elapsed >= MAX_UTTERANCE_MS) { finishCapture(!!heardAt); return; }
        if (!heardAt && elapsed >= NO_SPEECH_MS) { finishCapture(false); }
      }, 100);
    }).catch(function (e) {
      if (!active || token !== serial) return;
      failed = true;
      var message = e && e.name === 'NotAllowedError'
        ? 'Microphone permission was not granted. Allow it in the browser’s site settings, then try again.'
        : 'Microphone not available: ' + (e.message || e.name || 'unknown error');
      setPhase('error', message);
    });
  }

  function monitorState() {
    if (!active) return;
    var b = bridge();
    if (!b) return;
    var st = b.getState();
    if (speaking) { if (capture) cancelCapture(); setPhase(phase === 'speaking' ? 'speaking' : 'thinking'); return; }
    if (processing) { setPhase('thinking'); return; }
    if (st.busy) { if (capture) cancelCapture(); setPhase('thinking'); return; }
    if (failed) return;
    if (!st.omni || !st.audio) {
      failed = true; setPhase('error', 'Voice mode needs LFM3-Omni with audio input. Select it from the model menu, then try again.'); return;
    }
    if (!st.ready) { if (capture) cancelCapture(); setPhase('waiting'); return; }
    if (!capture) startCapture();
  }

  function enable() {
    var b = bridge();
    if (!b) return;
    var st = b.getState();
    if (!st.omni || !st.audio) {
      alert('Conversation mode needs LFM3-Omni with audio input. Select LFM3-Omni from the model menu and try again.'); return;
    }
    active = true; failed = false; speaking = false; processing = false;
    var app = document.getElementById('app');
    panel.hidden = false;
    panel.appendChild(button);  // Keep the same Clover control outside the inert chat application.
    if (results) results.replaceChildren();
    answerLine = null;
    document.body.classList.add('clover-conversation');
    if (app) { app.inert = true; app.setAttribute('aria-hidden', 'true'); }
    button.setAttribute('aria-pressed', 'true');
    loadRobot();
    setPhase(st.busy ? 'thinking' : st.ready ? 'starting' : 'waiting');
    monitorState();
  }

  function disable() {
    var wasActive = active;
    active = false; failed = false; speaking = false; processing = false;
    cancelCapture();
    var b = bridge(); if (b) b.stopSpeaking();
    if (window.speechSynthesis) speechSynthesis.cancel();
    var settings = document.getElementById('set-btn');
    if (settings) settings.parentNode.insertBefore(button, settings);
    panel.hidden = true;
    button.setAttribute('aria-pressed', 'false');
    document.body.classList.remove('clover-conversation');
    var app = document.getElementById('app');
    if (app) { app.inert = false; app.removeAttribute('aria-hidden'); }
    if (robotFrame) { robotFrame.src = 'about:blank'; robotReaction = ''; }
    answerLine = null;
    setPhase('off');
    if (wasActive) button.focus();
  }

  function init() {
    button = document.getElementById('voice-toggle');
    panel = document.getElementById('voice-companion');
    title = document.getElementById('voice-title');
    detail = document.getElementById('voice-detail');
    robotFrame = document.getElementById('robot-frame');
    results = document.getElementById('voice-results');
    if (!button || !panel || !title || !detail || !robotFrame || !results) return;
    robotFrame.addEventListener('load', function () {
      robotReaction = ''; syncRobot(true);
      if (active) robotFrame.focus();
    });
    button.addEventListener('click', function () { if (active) disable(); else enable(); });
    document.addEventListener('clover:conversation-exit', function () { if (active) disable(); });
    document.addEventListener('clover:voice-user', function (event) {
      if (active) showUser(event.detail && event.detail.text);
    });
    document.addEventListener('clover:voice-answer', function (event) {
      if (!active) return;
      showAnswer(event.detail && event.detail.text);
      if (event.detail && event.detail.done) answerLine = null;
    });
    document.addEventListener('keydown', function (event) {
      if (active && event.key === 'Escape') { event.preventDefault(); disable(); }
    });
    document.addEventListener('clover:speech-start', function () {
      if (!active) return;
      speaking = true; processing = false; cancelCapture(); setPhase('thinking');
    });
    document.addEventListener('clover:speech-play', function () {
      if (!active) return;
      speaking = true; setPhase('speaking');
    });
    document.addEventListener('clover:speech-end', function (event) {
      if (!active) return;
      speaking = false;
      var error = event.detail && event.detail.error;
      if (error) { answerLine = null; addTurn('assistant', 'Speech unavailable: ' + error + '. Listening again.'); }
      setPhase('listening');
      monitorState();
    });
    document.addEventListener('visibilitychange', function () {
      if (!active) return;
      if (document.hidden) cancelCapture(); else monitorState();
    });
    window.addEventListener('beforeunload', disable);
    monitor = setInterval(monitorState, 400);
    setPhase('off');
  }

  window.CloverVoiceMode = {
    isEnabled: function () { return active; },
    getState: function () { return { active: active, phase: phase, capturing: !!capture, speaking: speaking, failed: failed,
      pauseMs: capture ? capture.pauseMs || 3000 : 3000,
      rms: capture ? capture.rms || 0 : 0, heard: capture ? !!capture.heard : false, elapsed: capture ? capture.elapsed || 0 : 0 }; }
  };
  init();
})();
