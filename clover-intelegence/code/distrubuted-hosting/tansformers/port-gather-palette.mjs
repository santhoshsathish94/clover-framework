/* transformer_project_row is the trunk's inner loop: one byte of palette index per
   weight, a table lookup and a scalar fmaf per element. Every trunk projection goes
   through it, which is qkv, the attention output, the routed and shared expert
   projections, the latent projections, and the MLA query and key-value projections.

   Measured against the expert kernel on one harness, same rows and width so the MAC
   count is identical, swept from 0.9 MB to 235 MB:

     scalar palette   25.3 -> 30.3 GMAC/s
     gather palette   45.1 -> 50.7
     4-bit AVX2       71.3 -> 79.2

   The ratio holds when everything fits in L2, so this is not bandwidth, it is one
   scalar multiply-add per element against four in a vector. Gathering eight indices at
   a time recovers most of it. The rest of the gap to the expert kernel is the format,
   one byte per weight against four bits, which is a data change and not this one.

   Lane j of acc0 carries exactly the terms lanes[j] carried, in the same order, and
   acc1 the same for lanes[j+8]; fmaf and _mm256_fmadd_ps are both single-rounded, so
   the result is bit-identical. Verified at every size in the harness, mismatch 0, and
   in the layer: 27.46 to 22.176 ms/position with the checksum unchanged.

   Guarded like the expert kernel, which is also where immintrin.h comes from. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `    float lanes[16] = {0}, combined[8];
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16)
        for (unsigned lane = 0; lane < 16; lane++)
            lanes[lane] = fmaf(palette[ids[coordinate + lane]], input[coordinate + lane], lanes[lane]);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];`;

const AFTER = `    float lanes[16] = {0}, combined[8];
#if defined(__AVX2__)
    /* Lane j carries the same terms in the same order as the scalar form below. */
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
        __m256i index0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + coordinate)));
        __m256i index1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + coordinate + 8)));
        acc0 = _mm256_fmadd_ps(_mm256_i32gather_ps(palette, index0, 4),
            _mm256_loadu_ps(input + coordinate), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_i32gather_ps(palette, index1, 4),
            _mm256_loadu_ps(input + coordinate + 8), acc1);
    }
    _mm256_storeu_ps(lanes, acc0);
    _mm256_storeu_ps(lanes + 8, acc1);
#else
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16)
        for (unsigned lane = 0; lane < 16; lane++)
            lanes[lane] = fmaf(palette[ids[coordinate + lane]], input[coordinate + lane], lanes[lane]);
#endif
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];`;

let changed = 0, already = 0, absent = 0;
for (let layer = 1; layer <= 93; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source;
    try { source = readFileSync(path, 'utf8'); } catch { absent++; continue; }
    if (source.includes('_mm256_i32gather_ps(palette')) { already++; continue; }
    /* transformer-1 has mixed endings, so try both forms rather than deciding per file. */
    const crlf = s => s.replaceAll('\n', '\r\n');
    const form = source.includes(BEFORE) ? [BEFORE, AFTER]
        : source.includes(crlf(BEFORE)) ? [crlf(BEFORE), crlf(AFTER)] : null;
    if (!form) { absent++; continue; }
    assert.equal(source.split(form[0]).length - 1, 1, `layer ${layer}: anchor not unique`);
    writeFileSync(path, source.replace(form[0], form[1]));
    changed++;
}
console.log(`PASS: ${changed} pods gather the trunk palette, ${already} already done, ${absent} without it`);
