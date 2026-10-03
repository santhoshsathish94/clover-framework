import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
const bytes = readFileSync(new URL('../../../../clover-k3/clover-k3.c', import.meta.url));
assert.equal(createHash('sha256').update(bytes).digest('hex'), '5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65');
let source = bytes.toString('utf8').replace(/\r\n/g, '\n');
function replace(before, after) {
  assert.equal(source.split(before).length, 2, `ambiguous reference anchor: ${before.slice(0, 70)}`);
  source = source.replace(before, after);
}
replace(`    PRange *g = find_rg(fid, off);
    return g ? g->mem : file_ptr(fid) + off;`,
`    (void)fid; (void)off;
    return NULL;`);
const begin = source.indexOf('static void Xm(float *const *Y,'), end = source.indexOf('/* Bf: BF16 projection', begin);
assert(begin >= 0 && end > begin);
source = source.slice(0, begin) + `static void Xm(float *const *Y, const float *const *Xs, int T,
               const unsigned char *pk, const unsigned char *sc, int inn, int rows,
               const unsigned char *pk2, const unsigned char *sc2)
{
    (void)pk; (void)sc; (void)pk2; (void)sc2;
    test_observed_expert(cur_L, cur_e, cur_part, Y, Xs, T, inn, rows);
}

` + source.slice(end);
replace('        cur_L = L;', '        cur_L = L;\n        test_capture_input(L);');
replace('                if (route_tab) route_note(L, t, idsel_all[t]);',
  '                test_capture_route(L, t, idsel_all[t]);\n                if (route_tab) route_note(L, t, idsel_all[t]);');
const boundary = '    /* ---------------------------------------------------------------- tail */';
assert.equal(source.split(boundary).length, 2);
source = source.slice(0, source.indexOf(boundary)) + '    free(fa); free(fm);\n    return test_compare_layer();\n}\n';
source = `static void test_observed_expert(int, int, const char *, float *const *, const float *const *, int, int, int);
static void test_capture_input(int);
static void test_capture_route(int, int, const int *);
static int test_compare_layer(void);
` + source;
const destination = new URL('reference-one.inc', import.meta.url);
if (existsSync(destination)) assert.equal(readFileSync(destination, 'utf8'), source);
else writeFileSync(destination, source, { flag: 'wx' });
console.log('PASS: original layer 0/1 arithmetic retained; reference-only expert outputs require exact recorded inputs');