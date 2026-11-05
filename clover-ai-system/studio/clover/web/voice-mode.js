/* Continuous, opt-in voice mode for Clover AI Studio.
   Audio is sent to the selected local LFM3-Omni engine through Clover's existing chat pipeline.
   No speech-recognition cloud service is used. */
(function () {
  'use strict';
  var active = false, speaking = false, failed = false, processing = false, phase = 'off';
  var capture = null, serial = 0, monitor = null;
  var button, panel, title, detail;

  function bridge() { return window.CloverVoiceBridge || null; }
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
    button.setAttribute('aria-label', (active ? 'Disable ' : 'Enable ') + (labels[next] || 'voice mode'));
    button.title = active ? (labels[next] || 'Voice mode') + ' · click to turn off' : 'Enable voice mode';
    panel.dataset.phase = next;
    title.textContent = labels[next] || 'Voice mode';
    detail.textContent = details[next] || text || '';
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
      alert('Voice mode needs LFM3-Omni with audio input. Select LFM3-Omni from the model menu and try again.'); return;
    }
    active = true; failed = false; speaking = false; processing = false;
    panel.hidden = false;
    button.setAttribute('aria-pressed', 'true');
    setPhase(st.busy ? 'thinking' : st.ready ? 'starting' : 'waiting');
    monitorState();
  }

  function disable() {
    active = false; failed = false; speaking = false; processing = false;
    cancelCapture();
    var b = bridge(); if (b) b.stopSpeaking();
    if (window.speechSynthesis) speechSynthesis.cancel();
    panel.hidden = true;
    button.setAttribute('aria-pressed', 'false');
    setPhase('off');
  }

  function init() {
    button = document.getElementById('voice-toggle');
    panel = document.getElementById('voice-companion');
    title = document.getElementById('voice-title');
    detail = document.getElementById('voice-detail');
    if (!button || !panel || !title || !detail) return;
    button.addEventListener('click', function () { if (active) disable(); else enable(); });
    document.addEventListener('clover:speech-start', function () {
      if (!active) return;
      speaking = true; processing = false; cancelCapture(); setPhase('thinking');
    });
    document.addEventListener('clover:speech-play', function () {
      if (!active) return;
      speaking = true; setPhase('speaking');
    });
    document.addEventListener('clover:speech-end', function () {
      if (!active) return;
      speaking = false;
      setPhase('listening'); monitorState();
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
