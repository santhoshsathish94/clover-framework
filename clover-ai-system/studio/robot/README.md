# Clover Robo — refined interactive character

This project is a procedural Three.js character based on the Clover robot reference.

## Character identity refinements
- Warm ivory/cream armour with dark mechanical joints.
- Deep glossy black face visor with luminous lime crescent eyes.
- Olive-green side audio pods with luminous rings.
- Five-leaf Clover marks on the chest, cape and antenna.
- Curved antenna sprout above the head.
- Green draped cape and neck wrap retained.

## Animation refinements
- Stand is now the initial resting pose; walking is an explicit action.
- Idle motion is slower and softer, with irregular blinks, breathing, small head movement and secondary cape/antenna motion.
- Action transitions blend more slowly.
- Reactions have an entrance, hold and exit instead of remaining stuck indefinitely.
- Happy uses a single soft hop and cushioned landing.
- Walking uses a gentler cadence, reduced bob, alternating knee flex and ankle roll.
- Sit no longer squashes the chair vertically during the transition.
- Changing body action cancels the active reaction cleanly.

## Available actions
- Stand
- Sit
- Walk (in place)
- Dance

## Available reactions
- Wave
- Happy
- Thinking
- Working
- Reading
- Answering
- Sleepy

## Run locally
Requires Node.js 20.19+ or 22.12+.

```bash
npm install
npm run dev
```

For a production build:

```bash
npm run build
npm run preview
```

## Notes
The model is generated procedurally in `src/scene.js`; no external 3D model or image assets are required.
The current animation system remains a hand-authored procedural rig. For a later production pass, consider exporting a skinned GLB with named bones and authored animation clips, while retaining the same Clover design language.

## Warm cream finish and speech lip-sync (revision 2)
- Armour palette is now a deeper warm champagne/cream (`#D8BD9E`) with lighter panels in warm ivory (`#E4CDAE`), avoiding a cold white appearance.
- The mouth now has a dark inner aperture and animated viseme shapes for closed lips (M/B/P), teeth/lip contact (F/V), wide vowels (E/I), open vowels (A), rounded vowels (O), and small rounded vowels (U/W).
- Text is converted into a timed viseme sequence. This is a lightweight grapheme-to-viseme approximation, not a phoneme recognizer; for best accuracy pass the exact spoken text and audio duration from the TTS layer.

### Lip-sync API

```js
// Call when your own TTS audio starts; pass the exact spoken text and audio duration in seconds.
window.__startLipSync("Hello! I am Clover.", 1.85);

// Call when the audio ends or is interrupted.
window.__stopLipSync();

// Optional built-in browser speech demo (uses SpeechSynthesis).
window.__speakText("Hello! I am Clover.", { lang: "en-IN", rate: 1, pitch: 1.05 });
```

For an external speech provider, call `__startLipSync(text, audio.duration)` immediately when playback begins and `__stopLipSync()` on the audio `ended`, `pause`, or cancellation event. The animation timeline uses the same simulation clock as the character.
