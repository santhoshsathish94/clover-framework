# Clover Universal Icon Pack

Brand direction: five-leaf clover pinwheel, subtle green tonal shading, warm cream surfaces.

## Contents
- `brand/`: master SVG and PNG sizes (16–1024px)
- `app/android/`: Android launcher icon, dark icon, adaptive foreground/background
- `app/windows/`: multi-resolution ICO and 1024px PNG
- `app/web/`: favicon and PWA icons
- `robo-clover/`: current Robo Clover mascot artwork
- `loading/`: rotating and pulsing SVG/CSS indicators
- `icons/`: standalone SVG interface icons, grouped by use

## Integration notes
- SVG assets are editable vector source.
- Android adaptive icon foreground/background are supplied separately; configure safe-zone and manifest metadata in the Android project.
- Windows ICO includes common resolutions through 256px.
- Web PWA icons are PNG. Configure manifest paths in your application.
- The Robo Clover image is a rendered mascot reference, not a layered source file or a set of independently rendered animation frames.
- This pack supplies visual assets, not a complete Android resource XML, Windows installer, or animated robot rig.

Palette: cream `#F7F3E9`, clover green `#176B2C`, deep green `#0D4D20`, leaf highlight `#83B83D`.
