import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const types = { '.html':'text/html; charset=utf-8', '.css':'text/css; charset=utf-8',
  '.mjs':'text/javascript; charset=utf-8', '.js':'text/javascript; charset=utf-8',
  '.json':'application/json; charset=utf-8', '.txt':'text/plain; charset=utf-8' };
const server = createServer(async (request,response) => {
  try {
    if (!['GET','HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
    const url = new URL(request.url,'http://127.0.0.1');
    const pathname = decodeURIComponent(url.pathname === '/' ? '/tree-viewer/index.html' : url.pathname);
    const path = resolve(root,`.${pathname}`);
    if (!path.startsWith(root+sep)) { response.writeHead(403).end(); return; }
    const info = await stat(path);
    if (!info.isFile() || !types[extname(path)]) { response.writeHead(404).end(); return; }
    response.writeHead(200,{'Content-Type':types[extname(path)],'Content-Length':info.size,
      'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    if (request.method === 'HEAD') response.end();
    else createReadStream(path).on('error',() => response.destroy()).pipe(response);
  } catch { response.writeHead(404).end('Not found'); }
});
let port = Number(process.env.PORT || 8770);
server.on('error',error => {
  if (error.code === 'EADDRINUSE' && port < 8780) { port++; server.listen(port,'127.0.0.1'); }
  else { console.error(error.message); process.exitCode=1; }
});
server.listen(port,'127.0.0.1',() => console.log(`Expert tree viewer: http://127.0.0.1:${port}/tree-viewer/index.html`));