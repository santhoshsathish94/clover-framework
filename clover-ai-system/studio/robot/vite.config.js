import { defineConfig } from 'vite';

// `vite build`: the standalone demo (base './' keeps dist/ portable for sub-path hosting and file:// inspection).
// `vite build --mode studio`: one script for Clover AI Studio (window.CloverRobot, three bundled, no network).
export default defineConfig(({ mode }) => (mode === 'studio' ? {
  build: {
    target: 'es2020',
    outDir: 'dist-studio',
    emptyOutDir: true,
    lib: { entry: 'src/robot.js', name: 'CloverRobot', formats: ['iife'], fileName: () => 'robot.js' },
  },
} : {
  base: './',
  build: { target: 'es2022', outDir: 'dist' },
}));
