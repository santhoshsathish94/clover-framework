# Asset credits

## Clover robot reference artwork

`web/robo-clover-master.webp` is an optimized copy of the user-supplied
`assets/clover-yellow-flower-icon-pack/robo-clover/robo-clover-master.png`.
`web/robo-clover-transparent.webp` uses those same pixels with the backdrop, halo, and floor
shadow removed by a temporary local `rembg`/`u2netp` foreground-segmentation pass. These images
remain available for the existing Studio presentation.

## Interactive 3D conversation robot

The conversation robot began with the readable Three.js source supplied in `Kimi_Agent_Clover Ball
with Face.zip`, corresponding to the user-provided page at `https://6o5kwklcyon4q.kimi.page/`.
It now integrates the user-supplied `Clover_Robo_Refined_Source.zip`: refined ivory armour, glossy
visor, five-leaf Clover marks, curved antenna, woven cape, articulated posing, and softer actions.
The maintained source is `robot/scene.js`; `npm run build:robot` bundles it locally to
`web/clover-robo.js`. The Studio adaptation makes the renderer and scene transparent and hides
the hosted ground, distant scenery, contact shadow, controls, and status text. Conversation mode
shows only the interactive robot.

The robot and its runtime are served from Studio itself; no hosted page or remote model runtime
is loaded during Voice. Three.js 0.180.0 is distributed under the MIT license reproduced in
`THREE-LICENSE.txt`.

No new license or ownership claim over the user-supplied robot source or artwork is made here.
The segmentation library and model were temporary build tools and are not shipped with Studio.
