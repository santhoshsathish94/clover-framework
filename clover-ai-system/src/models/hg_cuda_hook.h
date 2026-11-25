/*
 * Optional Runtime `dlopen` Bridge (`hg_cuda_hook.h`) for Clover C11 `HG4/HG2` Engines.
 *
 * Keeps all four C11 engines (`lfm3`, `clover35b`, `nemotron`, `kimi_k31`) 100% buildable with plain
 * `gcc`/`clang` without CUDA headers installed, while automatically dispatching 16-row `HG` matmuls
 * and MoE routed pairs to `libhg_cuda_accel.so` whenever NVIDIA GPUs are available (`CLOVER_CUDA_GPUS`).
 */
#ifndef CLOVER_HG_CUDA_HOOK_H
#define CLOVER_HG_CUDA_HOOK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif

typedef int (*hg_cuda_try_matmul_fn)(
    const uint8_t *h_tiles, const uint16_t *h_scales,
    uint32_t R, uint32_t K, uint32_t bits,
    const int8_t *h_xq, const float *h_xsc,
    float *h_Y, int T, int act_mode
);

typedef int (*hg_cuda_try_moe_pairs_fn)(
    const uint8_t *h_exp_base, uint32_t n_exp_total, size_t exp_stride,
    size_t up_tiles_off, size_t dn_blob_off,
    uint32_t R_up, uint32_t K_up, uint32_t R_dn, uint32_t K_dn,
    uint32_t bits, int act_mode, int bf16_dn_round,
    const int8_t *h_xq, const float *h_xsc, int T,
    const int *h_pair_tok, const int *h_pair_slot, const float *h_pair_w, int n_pairs,
    float *h_out_add
);

typedef int (*hg_cuda_prepin_blob_fn)(const uint8_t *h_base, size_t total_b);

static int g_hg_cuda_hook_state = 0; /* 0 = uninitialized, 1 = active, -1 = disabled/absent */
static hg_cuda_try_matmul_fn    g_fn_hg_cuda_matmul = NULL;
static hg_cuda_try_moe_pairs_fn g_fn_hg_cuda_moe    = NULL;
static hg_cuda_prepin_blob_fn   g_fn_hg_cuda_prepin = NULL;

static void hg_cuda_hook_init_once(void) {
    if (g_hg_cuda_hook_state != 0) return;
    const char *env = getenv("CLOVER_CUDA_GPUS");
    if (env && (strcmp(env, "none") == 0 || strcmp(env, "off") == 0 || strcmp(env, "cpu") == 0)) {
        g_hg_cuda_hook_state = -1;
        return;
    }
#ifdef _WIN32
    HMODULE h = LoadLibraryA("hg_cuda_accel.dll");
    if (h) {
        g_fn_hg_cuda_matmul = (hg_cuda_try_matmul_fn)GetProcAddress(h, "hg_cuda_try_matmul");
        g_fn_hg_cuda_moe    = (hg_cuda_try_moe_pairs_fn)GetProcAddress(h, "hg_cuda_try_moe_pairs");
        g_fn_hg_cuda_prepin = (hg_cuda_prepin_blob_fn)GetProcAddress(h, "hg_cuda_prepin_blob");
    }
#else
    void *h = NULL;
    const char *lib_dir = getenv("CLOVER_NATIVE_LIB_DIR");
    if (lib_dir && lib_dir[0]) {
        char path_buf[1024];
        snprintf(path_buf, sizeof(path_buf), "%s/libhg_cuda_accel.so", lib_dir);
        h = dlopen(path_buf, RTLD_LAZY | RTLD_LOCAL);
    }
    if (!h) {
        Dl_info info;
        if (dladdr((const void *)&hg_cuda_hook_init_once, &info) && info.dli_fname) {
            const char *slash = strrchr(info.dli_fname, '/');
            if (slash) {
                char path_buf[1024];
                size_t dlen = (size_t)(slash - info.dli_fname);
                if (dlen < sizeof(path_buf) - 32) {
                    memcpy(path_buf, info.dli_fname, dlen);
                    snprintf(path_buf + dlen, sizeof(path_buf) - dlen, "/libhg_cuda_accel.so");
                    h = dlopen(path_buf, RTLD_LAZY | RTLD_LOCAL);
                }
            }
        }
    }
    if (!h) h = dlopen("libhg_cuda_accel.so", RTLD_LAZY | RTLD_LOCAL);
    if (!h) h = dlopen("/usr/local/lib/libhg_cuda_accel.so", RTLD_LAZY | RTLD_LOCAL);
    if (h) {
        g_fn_hg_cuda_matmul = (hg_cuda_try_matmul_fn)dlsym(h, "hg_cuda_try_matmul");
        g_fn_hg_cuda_moe    = (hg_cuda_try_moe_pairs_fn)dlsym(h, "hg_cuda_try_moe_pairs");
        g_fn_hg_cuda_prepin = (hg_cuda_prepin_blob_fn)dlsym(h, "hg_cuda_prepin_blob");
    }
#endif
    g_hg_cuda_hook_state = (g_fn_hg_cuda_matmul && g_fn_hg_cuda_moe) ? 1 : -1;
}

static inline int hg_cuda_dispatch_prepin(const uint8_t *h_base, size_t total_b) {
    if (g_hg_cuda_hook_state == 0) hg_cuda_hook_init_once();
    if (g_hg_cuda_hook_state != 1 || !g_fn_hg_cuda_prepin || !h_base || total_b == 0) return 0;
    return g_fn_hg_cuda_prepin(h_base, total_b);
}

static inline int hg_cuda_dispatch_matmul(
    const uint8_t *h_tiles, const uint16_t *h_scales,
    uint32_t R, uint32_t K, uint32_t bits,
    const int8_t *h_xq, const float *h_xsc,
    float *h_Y, int T, int act_mode
) {
    if (g_hg_cuda_hook_state == 0) hg_cuda_hook_init_once();
    if (g_hg_cuda_hook_state != 1) return 0;
    return g_fn_hg_cuda_matmul(h_tiles, h_scales, R, K, bits, h_xq, h_xsc, h_Y, T, act_mode);
}

static inline int hg_cuda_dispatch_moe_pairs(
    const uint8_t *h_exp_base, uint32_t n_exp_total, size_t exp_stride,
    size_t up_tiles_off, size_t dn_blob_off,
    uint32_t R_up, uint32_t K_up, uint32_t R_dn, uint32_t K_dn,
    uint32_t bits, int act_mode, int bf16_dn_round,
    const int8_t *h_xq, const float *h_xsc, int T,
    const int *h_pair_tok, const int *h_pair_slot, const float *h_pair_w, int n_pairs,
    float *h_out_add
) {
    if (g_hg_cuda_hook_state == 0) hg_cuda_hook_init_once();
    if (g_hg_cuda_hook_state != 1) return 0;
    return g_fn_hg_cuda_moe(
        h_exp_base, n_exp_total, exp_stride, up_tiles_off, dn_blob_off,
        R_up, K_up, R_dn, K_dn, bits, act_mode, bf16_dn_round,
        h_xq, h_xsc, T, h_pair_tok, h_pair_slot, h_pair_w, n_pairs, h_out_add
    );
}

#endif /* CLOVER_HG_CUDA_HOOK_H */
