// Kimi Work preview entry point. All host/port arguments are forwarded to Clover's Python server.
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const args = process.argv.slice(2);
let port = '7100';
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' && args[i + 1]) port = args[++i];
  else if (args[i].startsWith('--port=')) port = args[i].split('=')[1];
}
if (!/^\d+$/.test(port)) throw new Error(`Invalid port: ${port}`);

const root = __dirname;
const installedBase = process.env.USERPROFILE ? path.join(process.env.USERPROFILE, 'CloverAI') : '';
const base = process.env.CLOVER_BASE || (installedBase && fs.existsSync(installedBase) ? installedBase : path.join(root, 'dev-data'));
const installedPython = base ? path.join(base, 'venv', 'Scripts', 'python.exe') : '';
const python = process.env.CLOVER_PYTHON || (process.platform === 'win32' && fs.existsSync(installedPython) ? installedPython : 'python');
const log = path.join(root, 'logs', 'voice-preview.log');
fs.mkdirSync(path.dirname(log), { recursive: true });

const child = spawn(python, [path.join(root, 'server.py'), '--base', base, '--port', port, '--log', log], {
  cwd: root,
  stdio: 'inherit',
  windowsHide: false
});
let stopping = false;
function stop() {
  if (stopping || child.exitCode !== null) return;
  stopping = true;
  child.kill('SIGTERM');
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
child.on('exit', code => { process.exitCode = code || 0; });
child.on('error', error => { console.error(error); process.exitCode = 1; });
