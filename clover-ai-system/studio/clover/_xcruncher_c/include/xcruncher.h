/*
 * x-cruncher — High-Performance Arbitrary-Precision Multiplication Engine
 * Copyright (c) 2026 Santhosh Narayanan. Released under the MIT License.
 *
 * Combines:
 *   1. Register-scale raw 64-bit / Karatsuba limb multiplication (<= ~7,400 digits)
 *   2. Vedic Signed Power-of-2 (NAF) sparse polynomial fast-path (near-2^k / sparse runs)
 *   3. 51-Bit Table-Free Lazy Harvey [0, 2P) 2-Prime AVX-512 IFMA52 + VBMI Four-Step
 *      Radix-4 Exact Integer NTT (~400K .. 101M digits per operand) with 4-cursor fused
 *      Four-Step twiddles, Carry-Free 40-bit Vedic Vinculum limbs, and signed 102-bit Garner CRT
 *   4. Crandall-Fagin Right-Angle (Y^M + i = 0) Adaptive 10..16-bit Carry-Free Vedic Vinculum
 *      Four-Step Wave Engine with provable worst-case IEEE-754 bounds, Precomputed WaveTrunks,
 *      64-bit Block-Hash Wave Collapse, and 50%-RAM Raw-Bit Karatsuba Top-Split
 *      (AVX-512 / AVX2+FMA / ARM64 NEON / Portable Scalar).
 */

#ifndef XCRUNCHER_H
#define XCRUNCHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef XCRUNCHER_BUILD_DLL
    #define XC_API __declspec(dllexport)
  #elif defined(XCRUNCHER_USE_DLL)
    #define XC_API __declspec(dllimport)
  #else
    #define XC_API
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define XC_API __attribute__((visibility("default")))
#else
  #define XC_API
#endif

#define XCRUNCHER_VERSION "1.6.1"

/* Error / status codes */
#define XC_OK          0
#define XC_ERR_ARG    -1
#define XC_ERR_NOMEM  -2
#define XC_ERR_RANGE  -3

/* Execution path used by the most recent call (for diagnostics / benchmarking) */
typedef enum {
    XC_PATH_ZERO        = 0,
    XC_PATH_SCHOOLBOOK  = 1,
    XC_PATH_KARATSUBA   = 2,
    XC_PATH_VEDIC_SPARSE= 3,
    XC_PATH_WAVE_COLD   = 4,
    XC_PATH_WAVE_TRUNK  = 5,
    XC_PATH_WAVE_HASH   = 6,
    XC_PATH_LOWMEM_KARA = 7
} xc_path_t;

typedef struct xc_plan_s  xc_plan_t;
typedef struct xc_trunk_s xc_trunk_t;

/*
 * Multiply two unsigned big integers represented as little-endian uint64_t limb arrays:
 *   A = sum_{i=0}^{na-1} a[i] * 2^(64*i)
 *   B = sum_{j=0}^{nb-1} b[j] * 2^(64*j)
 *   out[0 .. na + nb - 1] = A * B
 *
 * Automatically selects the fastest exact mathematical path:
 *   - Raw 64x64->128 schoolbook / Karatsuba for register/L1 sizes
 *   - Vedic Signed Power-of-2 (NAF) sparse polynomial solver for near-2^k / long 0-or-1 runs
 *   - 2-Prime 51-bit Lazy Harvey AVX-512 IFMA52 + VBMI NTT (~400K .. 101M digits/operand)
 *   - Adaptive 10..16-bit Crandall-Fagin Vedic Wave Engine & Low-RAM Karatsuba Top-Split
 *
 * Thread safety: `xc_mul_u64` and `xc_mul_u64_lowmem` are safe to call concurrently from
 * multiple application threads (assuming distinct `out` buffers).
 */
XC_API int xc_mul_u64(uint64_t *out, const uint64_t *a, size_t na,
                      const uint64_t *b, size_t nb);

/*
 * Low-RAM Raw-Bit Karatsuba Top-Split Multiply:
 * Splits raw uint64_t arrays in half BEFORE wave/NTT expansion, cutting peak workspace RAM by 50%.
 */
XC_API int xc_mul_u64_lowmem(uint64_t *out, const uint64_t *a, size_t na,
                             const uint64_t *b, size_t nb);

/*
 * Reusable Plan & Precomputed Static Trunk API:
 *   - `xc_plan_create(max_na, max_nb)` allocates reusable NTT or Wave workspaces once.
 *   - `xc_trunk_create(plan, b, nb)` pre-transforms constant multiplier `B` into `Trunk(B)`.
 *   - `xc_mul_trunk(plan, out, a, na, trunk)` multiplies fresh `A` by `Trunk(B)`.
 *
 * Thread safety: A single `xc_plan_t` holds mutable workspace buffers and MUST NOT be used
 * concurrently by multiple threads at the same time (create one `xc_plan_t` per caller thread).
 * An immutable `const xc_trunk_t *` may be shared read-only across threads that each use their
 * own compatible `xc_plan_t`.
 */
XC_API xc_plan_t  *xc_plan_create(size_t max_na, size_t max_nb);
XC_API void        xc_plan_destroy(xc_plan_t *plan);

XC_API int         xc_plan_mul(xc_plan_t *plan, uint64_t *out,
                               const uint64_t *a, size_t na,
                               const uint64_t *b, size_t nb);

XC_API xc_trunk_t *xc_trunk_create(const xc_plan_t *plan, const uint64_t *b, size_t nb);
XC_API void        xc_trunk_destroy(xc_trunk_t *trunk);

XC_API int         xc_mul_trunk(xc_plan_t *plan, uint64_t *out,
                                const uint64_t *a, size_t na,
                                const xc_trunk_t *b_trunk);

/*
 * Decimal String Conversion & Multiplication Helpers (Base 10^9 limbs -> uint64_t limbs)
 */
XC_API int   xc_from_decimal(const char *dec_str, uint64_t **out_limbs, size_t *out_nlimbs);
XC_API char *xc_to_decimal(const uint64_t *limbs, size_t nlimbs);
XC_API char *xc_mul_decimal(const char *dec_a, const char *dec_b);
XC_API void  xc_free(void *ptr);

/*
 * Zero-Storage N-th Hexadecimal Digit Extractor for Pi (Fused 8-Series Huvent BBP):
 *   Computes `num_digits` (1 .. 20) exact hexadecimal digits of Pi starting after 16^(-hex_offset)
 *   for `0 <= hex_offset <= 2^44` (17,592,186,044,416, ~1.76e13) using < 1 KB RAM and 0 disk
 *   (`hex_offset = 0` -> "243f6a8885a308d31319"). Thread-safe for concurrent callers.
 */
XC_API int xc_pi_hex_digits(uint64_t hex_offset, char *out_hex, size_t num_digits);

/*
 * Runtime Hardware & Per-Thread Diagnostics:
 *   `xc_last_path_used`, `xc_last_unique_slices`, and `xc_last_limb_bits` report thread-local
 *   diagnostics for the most recent multiplication executed on the calling thread.
 */
XC_API const char *xc_version(void);
XC_API const char *xc_active_isa_name(void);
XC_API xc_path_t   xc_last_path_used(void);
XC_API int         xc_last_unique_slices(void);
XC_API int         xc_last_limb_bits(void);

#ifdef __cplusplus
}
#endif

#endif /* XCRUNCHER_H */
