# Asset credits

## Clover robot reference

`web/robo-clover-master.webp` is an optimized copy of the user-supplied
`assets/clover-yellow-flower-icon-pack/robo-clover/robo-clover-master.png`.
`web/robo-clover-transparent.webp` uses those same pixels with the backdrop, halo, and floor
shadow removed by a temporary local `rembg`/`u2netp` foreground-segmentation pass. No
replacement robot or generated artwork was substituted.

The companion displays the isolated artwork as a static, translucent image. A separate SVG
circle provides the clockwise voice indicator; no Three.js, WebGL mesh, mouth layer, or
procedural robot geometry remains in the current implementation.

No new license or ownership claim over the supplied artwork is made here. The segmentation
library and model were temporary build tools and are not shipped with the Studio runtime.
