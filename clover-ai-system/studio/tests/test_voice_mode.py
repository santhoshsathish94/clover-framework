"""Offline checks for voice mode: the robot script is served and loaded only through the page's nonce, the toggle and
stage exist with chat mode as the default, turn events and speech drive the robot, hands-free listening keeps the
microphone closed while Clover thinks or speaks, and the built robot.js exposes the API the page uses.
Usage: python tests/test_voice_mode.py"""
import re
import subprocess
import sys
from pathlib import Path

STUDIO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDIO / "clover"))
import server as SV  # noqa: E402

ok, n = True, 0


def check(name, cond):
    global ok, n
    n += 1
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


app = (STUDIO / "clover" / "web" / "app.html").read_text(encoding="utf-8")
robot_js = STUDIO / "clover" / "web" / "robot.js"
built = robot_js.read_text(encoding="utf-8") if robot_js.is_file() else ""

# ---------------- served and loaded ----------------
check("robot.js is in the static allow-list as JavaScript", SV.STATIC.get("/robot.js") == ("robot.js", "text/javascript"))
check("the built robot.js exists and is a single classic script", built.startswith("var CloverRobot") and "import " not in built[:200])
check("the page's CSP still allows only nonce scripts (no CDN, no unsafe-inline)",
      "script-src 'nonce-{nonce}'" in Path(SV.__file__).read_text(encoding="utf-8"))
check("the page loads robot.js with its own nonce, read when the inline script first runs",
      "s.src = 'robot.js'; s.nonce = SCRIPT_NONCE;" in app and "document.currentScript && document.currentScript.nonce" in app)
check("robot.js is loaded only when voice mode is first turned on", app.count("s.src = 'robot.js'") == 1 and "function loadRobot()" in app)
check("no external host appears in robot.js", not re.search(r"https?://(?!www\.w3\.org)[a-z0-9.-]+\.[a-z]{2,}/", built))

# ---------------- toggle and stage ----------------
check("a clover button with the small multipetal app logo sits beside Settings",
      re.search(r'id="voice-btn"[^>]*><img class="voice-logo" src="icon\.png"[^>]*></button>\s*\n\s*<button[^>]*id="set-btn"', app) is not None)
check("icon.png is the multipetal app logo (not the 5-leaf version)",
      (STUDIO / "clover" / "web" / "icon.png").read_bytes()
      == (STUDIO.parent.parent / "assets" / "clover-yellow-flower-icon-pack" / "web" / "favicons" / "clover-256.png").read_bytes()
      and '<symbol id="i-clover"' not in app)
check("chat mode is the default (the button starts unpressed, the stage hidden)",
      'id="voice-btn" aria-pressed="false"' in app and 'id="voice-stage" hidden' in app and "var voice = { on: false" in app)
check("the stage takes no clicks in chat mode; in voice mode a tap on the robot talks, sends or stops its speech",
      ".voice-stage { position: absolute; inset: 0; z-index: 1; pointer-events: none; }" in app
      and "html.voice .voice-stage { pointer-events: auto;" in app
      and "if (speech) { stopSpeech(); voiceSpoken(); return; }" in app
      and "if (listen.phase === 'speaking') finishUtterance();" in app and "startUtterance(true);" in app)
check("in voice mode the chat behind the robot is transparent", "html.voice .stream .col { opacity: .22;" in app)
check("rendering stops when voice mode is off or the window is hidden",
      "if (voice.robot) voice.robot.stop();" in app and "document.hidden) voice.robot.stop()" in app)

# ---------------- turn events and speech ----------------
check("every turn event reaches the robot", "voiceTurnEvent(ev);" in app)
check("tools map to reading or working", "READ_TOOLS.test(ev.name || '') ? 'reading' : 'working'" in app)
check("a permission request makes the robot look up and wave", "else if (t === 'permission') { voiceState('idle'); voice.robot.react('wave'); }" in app)
check("each spoken part drives lip sync with its real duration", "a.onplaying = function () { voiceLips(parts[i], a.duration); };" in app)
check("voice mode speaks every reply with this device's own voice first, Clover's Omni voice when there is none",
      "(voice.on || (state.speakNext && state.settings.autospeak))" in app
      and "if (speakIt && voice.on && speakDevice(m.text)) return;" in app and "if (speakIt && state.omni) speakOmni(m.text);" in app)
check("the device voice is on-device only (local browser voices or the Android app's voice)",
      "v.localService && /^en/i.test(v.lang)" in app and "typeof window.CloverAndroidBridge.ttsSpeak === 'function'" in app)
check("the device voice speaks sentence by sentence; word events keep the mouth in step",
      "function speechParts(text)" in app and "u.onboundary = function (e) { if (e.name === 'word') on('word', e.charIndex); };" in app)
check("the Omni voice is warmed only when it will be used",
      "if (state.omni && (voice.on ? !deviceVoice() : state.speakNext && state.settings.autospeak)) api('POST', '/api/tts/warm'" in app)
check("the robot's own status never wipes the listening caption",
      "robot.onStatus(function (status) { if (!speech && state.busy) voiceCaption(status); });" in app)

# ---------------- hands-free listening ----------------
check("listening needs voice mode, a ready engine that hears audio, no turn and no speech",
      "voice.on && !listen.blocked && state.ready && !state.busy && !speech && state.omni && state.inputs.indexOf('audio') >= 0" in app)
check("the microphone is released after each utterance and when voice mode turns off",
      "listen.stream.getTracks().forEach(function (t) { t.stop(); })" in app and "voiceStopListening(); stopSpeech();" in app)
check("a turn started by typing closes the microphone", re.search(r"setStop\(true\);\s*voiceStopListening\(\);", app) is not None)
check("listening resumes only after the reply has been spoken", "setTimeout(voiceListen, 400)" in app)
check("voice detection follows the room: 1 s to hear it first, start 0.2 s above 1.5x its peak, end after 0.8 s quiet, 20 s cap",
      "var VAD = { calibrateMs: 1000, preMs: 500, startMs: 200, endMs: 800, minMs: 400, maxMs: 20000, lagMs: 500, roomMs: 3000," in app
      and "listen.thr = Math.max(VAD.minLevel, quantile(room, 0.2) * 4, quantile(room, 0.95) * 1.5);" in app)
check("the room is what was heard 0.5-3.5 s ago, not the speaker's own first words",
      "room = listen.hist.length > lag ? listen.hist.slice(0, -lag) : []" in app)
vad_m = re.search(r"var VAD = \{(.*?)\};", app, re.S)
vad = {k: float(v) for k, v in re.findall(r"(\w+): ([\d.]+)", vad_m.group(1))} if vad_m else {}
check("the first decision comes only after some room has been heard (calibrateMs > lagMs; on the phone 500 = 500 "
      "let a video start an utterance at once)", vad.get("calibrateMs", 0) > vad.get("lagMs", 1e9))
check("a loud room pauses hands-free and asks for a tap; after a tap only a tap (or the cap) sends in a loud room",
      "var noisy = listen.thr > VAD.noisyLevel;" in app and "It\\u2019s noisy here. Tap me to talk." in app
      and "listen.loudMs = !noisy && rms > listen.thr ? listen.loudMs + ms : 0;" in app
      and "if (listen.tapped && (listen.thr > VAD.noisyLevel || listen.voicedMs < VAD.minMs)) return;" in app)
check("while it hears you the caption says a tap sends", "'Hearing you\\u2026 tap to send'" in app)
check("no cloud speech recognition is used", "SpeechRecognition" not in app and "webkitSpeechRecognition" not in app)
check("hold-to-talk stays out of the way while hands-free listening runs", "if (listen.on || rec || nativeRecT0" in app)

# ---------------- the robot API the page uses ----------------
api = re.search(r"const api = \{(.*?)\};", (STUDIO / "robot" / "src" / "robot.js").read_text(encoding="utf-8"), re.S)
names = set(re.findall(r"\b([a-zA-Z]+)\b", api.group(1))) if api else set()
for fn in ("mount", "start", "stop", "setState", "react", "setLevel", "startLipSync", "stopLipSync", "onStatus", "debugState"):
    check(f"robot API has {fn}", fn in names)
used = set(re.findall(r"(?:voice\.robot|robot)\.(\w+)\(", app))
check("the page calls only functions the robot provides", used <= names | {"setState"} and used, )
src = (STUDIO / "robot" / "src" / "robot.js").read_text(encoding="utf-8")
check("the robot has no page coupling left", not re.search(r"getElementById|querySelector|innerWidth|window\.__", src))
check("lasting states are the Studio steps", "const LASTING = new Set(['listening', 'thinking', 'working', 'reading', 'answering']);" in src)
check("the mouth is white light only: no dark cavity, the light itself opens with each sound",
      "mouthCavity" not in src and "mouth.scale.set(0.62 + speechPose.width * 6.5, 0.22 + speechPose.open * 9.5, 1);" in src)
check("while the user is speaking into the microphone live, the robot reacts in real time (lean-in, audio-pod rings, chest clover, listening smile, and backchannel nod)",
      "const liveListen = r.listening * levelSmooth;" in src
      and "const nodDip = r.nod * Math.sin(r.u * TAU * 1.5)" in src
      and "torso.rotation.x = liveListen * 0.065 + r.nod * 0.025;" in src
      and "podRings.forEach((ring) => ring.scale.setScalar(1 + r.listening * levelSmooth * 0.16));" in src
      and "const listenSmile = Math.max(r.nod * 0.65, liveListen > 0.12 ? (liveListen - 0.12) * 0.55 : 0);" in src
      and "voice.robot.react('nod');" in app)

# ---------------- the built file matches its source ----------------
node_modules = STUDIO / "robot" / "node_modules"
if node_modules.is_dir():
    r = subprocess.run("npx vite build --mode studio --logLevel error", cwd=STUDIO / "robot", shell=True, capture_output=True, text=True)
    fresh = (STUDIO / "robot" / "dist-studio" / "robot.js")
    check("a fresh build of src/robot.js equals the committed clover/web/robot.js",
          r.returncode == 0 and fresh.is_file() and fresh.read_bytes().replace(b"\r\n", b"\n") == robot_js.read_bytes().replace(b"\r\n", b"\n"))
else:
    print("SKIP rebuild comparison (run npm ci in studio/robot to enable it)")

print(f"{n} checks" + (" passed" if ok else ", SOME FAILED"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
