#!/usr/bin/env python3
"""Copy clover-one.c to a new path with Qm's AVX2 section token-blocked.

Tokens are taken 8 at a time with the accumulators held in registers, each token keeping
the original accumulation order and final reduction, so results are bit-identical to the
original. Any remainder tokens run through the original code unchanged. The source file is
never modified.   usage: patch-qm.py SRC DST
"""
import sys

src = open(sys.argv[1], newline="").read()
nl = "\r\n" if "\r\n" in src[:src.index("static void Qm(")] else "\n"
text = src.replace("\r\n", "\n")

head = "#if defined(__AVX2__)\n        __m256 v0[NPOS_SLOTS], v1[NPOS_SLOTS];\n"
tail = "#else\n        float B[NPOS_SLOTS][16];\n"
assert text.count(head) == 1 and text.count(tail) == 1, "Qm anchors not found exactly once"
start = text.index(head)
end = text.index(tail, start)
region = text[start + len("#if defined(__AVX2__)\n"):end]
assert region.count("for (int t = 0; t < T; t++)") == 4, region.count("for (int t = 0; t < T; t++)")
remainder = region.replace("for (int t = 0; t < T; t++)", "for (int t = tb; t < T; t++)")

blocked = r'''#if defined(__AVX2__)
        /* Tokens in blocks of eight, accumulators in registers. Same per-token order. */
        int tb = 0;
        for (; tb + 8 <= T; tb += 8) {
            const float *xs[8];
            __m256 q0[8], q1[8];
#pragma GCC unroll 8
            for (int k = 0; k < 8; k++) { xs[k] = Xs[tb + k]; q0[k] = _mm256_setzero_ps(); q1[k] = _mm256_setzero_ps(); }
            if (direct_codes) {
                for (int j = 0; j + 15 < in; j += 16) {
                    const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + j))));
                    const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + j + 8))));
#pragma GCC unroll 8
                    for (int k = 0; k < 8; k++) {
                        q0[k] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xs[k] + j), q0[k]);
                        q1[k] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xs[k] + j + 8), q1[k]);
                    }
                }
            } else {
                for (int j = 0; j + 15 < in; j += 16) {
                    const __m256 w0 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(w + j))), 4);
                    const __m256 w1 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(w + j + 8))), 4);
#pragma GCC unroll 8
                    for (int k = 0; k < 8; k++) {
                        q0[k] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xs[k] + j), q0[k]);
                        q1[k] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xs[k] + j + 8), q1[k]);
                    }
                }
            }
#pragma GCC unroll 8
            for (int k = 0; k < 8; k++) {
                __m256 vs = _mm256_add_ps(q0[k], q1[k]);
                __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
                lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
                lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                acc[tb + k] = _mm_cvtss_f32(lo);
            }
        }
'''
new = blocked + "        if (tb < T) {\n" + remainder + "        } else {\n            i = in & ~15;\n        }\n"
out = text[:start] + new + text[end:]
if nl == "\r\n":
    out = out.replace("\n", "\r\n")
open(sys.argv[2], "w", newline="").write(out)
print("patched Qm: %d bytes -> %d bytes" % (len(src), len(out)))
