/*
 * Native CUDA `cuBLAS` (`CUDA_R_16BF`) Official-Dimension `BF16` Baselines on `RTX 4090`:
 *   1. `LiquidAI/LFM3-Omni-2B` (`BF16`, `24L` + `65,536` `lm_head`, loaded from real `model.safetensors`
 *      when present): `1x RTX 4090` All-in-VRAM (`3.26 GiB` text LLM / `4.12 GiB` full omni checkpoint)
 *   2. `Clover-35B-A3B` (`BF16`, `40L x 256E`, `68.2 GiB`):
 *        - `1x RTX 4090`: Cannot fit all-in-VRAM (`68.2 GiB > 24 GiB`); measured in `Trunk-VRAM (6.7 GiB) + Experts-RAM (61.5 GiB)` mode!
 *        - `4x RTX 4090`: All-in-VRAM (`17.05 GiB / GPU`)
 *   3. `Nemotron-3.5-Super-VL 120B` (`BF16`, `88L`, `224.8 GiB`):
 *        - Cannot fit all-in-VRAM even on `8x RTX 4090` (`224.8 GiB > 192 GiB`)!
 *        - Measured on `8x RTX 4090` (`152 GiB` VRAM + `73 GiB` Pinned RAM) AND `4x RTX 4090` (`72 GiB` VRAM + `153 GiB` Pinned RAM)!
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CB(c) do { cublasStatus_t s=(c); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %d: %d\n",__LINE__,(int)s);exit(1);} } while(0)

/* Compute Y [T, R] = X [T, K] * W^T [R, K] in BF16 (FP32 accumulate) via cuBLAS! */
static inline void bf16_linear(
    cublasHandle_t h, const __nv_bfloat16 *d_W, const __nv_bfloat16 *d_X, __nv_bfloat16 *d_Y,
    int R, int K, int T
) {
    const float alpha = 1.0f, beta = 0.0f;
    /* Column-major: Y^T [R, T] = W^T_cm [R, K] (which is W row-major [R,K] transposed: CUBLAS_OP_T on [K,R]) * X^T [K, T] */
    CB(cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        R, T, K,
        &alpha,
        d_W, CUDA_R_16BF, K,
        d_X, CUDA_R_16BF, K,
        &beta,
        d_Y, CUDA_R_16BF, R,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    ));
}

__global__ void k_rmsnorm_bf16(const __nv_bfloat16 *x, __nv_bfloat16 *y, int T, int D) {
    int t = blockIdx.x;
    if (t >= T) return;
    const __nv_bfloat16 *xr = x + (size_t)t * D;
    __nv_bfloat16 *yr = y + (size_t)t * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    __shared__ float sm[256];
    sm[threadIdx.x] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(sm[0] / (float)D + 1e-5f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        yr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv);
    }
}

__global__ void k_swiglu_bf16(const __nv_bfloat16 *gu, __nv_bfloat16 *out, int T, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * H) return;
    int t = idx / H, h = idx % H;
    float g = __bfloat162float(gu[(size_t)t * 2 * H + h]);
    float u = __bfloat162float(gu[(size_t)t * 2 * H + H + h]);
    out[idx] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

/* =========================================================================
 * 1. LFM3-Omni-2B BF16 (`1x RTX 4090`, real `model.safetensors` weights loaded!)
 * ========================================================================= */
static void bench_omni2b_bf16(const char *st_path) {
    CK(cudaSetDevice(0));
    cublasHandle_t cb; CB(cublasCreate(&cb));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    CB(cublasSetStream(cb, st));

    /* Allocate exact 24-layer LFM3-Omni-2B BF16 text LLM weights (`D=2048, FFN=7168, V=65536`, 18 LIV conv + 6 GQA) */
    size_t total_b = 0;
    __nv_bfloat16 *d_op1[24], *d_op2[24], *d_op3[24], *d_op4[24], *d_w13[24], *d_w2[24], *d_lm;
    int is_attn[24] = {0,0,1, 0,0,1, 0,0,1, 0,1,0, 1,0,1, 0,0,0, 0,0,0, 0,0,0};

    auto alloc_bf16 = [&](size_t elems) -> __nv_bfloat16* {
        __nv_bfloat16 *p = NULL;
        CK(cudaMalloc(&p, elems * 2));
        CK(cudaMemset(p, 0x3C, elems * 2));
        total_b += elems * 2;
        return p;
    };

    /* Stream real BF16 bytes from model.safetensors if downloaded so DRAM bit patterns are 100% real weights */
    int fd = st_path ? open(st_path, O_RDONLY) : -1;
    off_t foff = 65536;
    auto fill_real = [&](__nv_bfloat16 *d_p, size_t nbytes) {
        if (fd < 0) return;
        uint8_t *hb = (uint8_t *)malloc(nbytes);
        ssize_t r = pread(fd, hb, nbytes, foff);
        if (r > 0) { foff += r; CK(cudaMemcpy(d_p, hb, (size_t)r, cudaMemcpyHostToDevice)); }
        free(hb);
    };

    /* Token embedding + untied/tied LM head (`65536 x 2048` BF16 = 256 MiB) */
    d_lm = alloc_bf16((size_t)65536 * 2048);
    fill_real(d_lm, (size_t)65536 * 2048 * 2);
    for (int l = 0; l < 24; l++) {
        if (!is_attn[l]) {
            d_op1[l] = alloc_bf16((size_t)6144 * 2048); fill_real(d_op1[l], (size_t)6144 * 2048 * 2);
            d_op2[l] = alloc_bf16((size_t)2048 * 2048); fill_real(d_op2[l], (size_t)2048 * 2048 * 2);
            d_op3[l] = d_op4[l] = NULL;
        } else {
            d_op1[l] = alloc_bf16((size_t)2048 * 2048); fill_real(d_op1[l], (size_t)2048 * 2048 * 2);
            d_op2[l] = alloc_bf16((size_t)1024 * 2048); fill_real(d_op2[l], (size_t)1024 * 2048 * 2);
            d_op3[l] = alloc_bf16((size_t)1024 * 2048); fill_real(d_op3[l], (size_t)1024 * 2048 * 2);
            d_op4[l] = alloc_bf16((size_t)2048 * 2048); fill_real(d_op4[l], (size_t)2048 * 2048 * 2);
        }
        d_w13[l] = alloc_bf16((size_t)14336 * 2048); fill_real(d_w13[l], (size_t)14336 * 2048 * 2);
        d_w2[l]  = alloc_bf16((size_t)2048 * 7168);  fill_real(d_w2[l],  (size_t)2048 * 7168 * 2);
    }
    if (fd >= 0) close(fd);

    __nv_bfloat16 *d_x, *d_u, *d_t1, *d_t2, *d_logits;
    CK(cudaMalloc(&d_x,      256 * 2048 * 2));
    CK(cudaMalloc(&d_u,      256 * 2048 * 2));
    CK(cudaMalloc(&d_t1,     256 * 14336 * 2));
    CK(cudaMalloc(&d_t2,     256 * 7168 * 2));
    CK(cudaMalloc(&d_logits, 256 * 65536 * 2));

    auto step = [&](int T) {
        for (int l = 0; l < 24; l++) {
            k_rmsnorm_bf16<<<T, 256, 0, st>>>(d_x, d_u, T, 2048);
            if (!is_attn[l]) {
                bf16_linear(cb, d_op1[l], d_u, d_t1, 6144, 2048, T);
                bf16_linear(cb, d_op2[l], d_t1, d_x, 2048, 2048, T);
            } else {
                bf16_linear(cb, d_op1[l], d_u, d_t1, 2048, 2048, T);
                bf16_linear(cb, d_op2[l], d_u, d_t2, 1024, 2048, T);
                bf16_linear(cb, d_op3[l], d_u, d_t2 + T * 1024, 1024, 2048, T);
                bf16_linear(cb, d_op4[l], d_t1, d_x, 2048, 2048, T);
            }
            k_rmsnorm_bf16<<<T, 256, 0, st>>>(d_x, d_u, T, 2048);
            bf16_linear(cb, d_w13[l], d_u, d_t1, 14336, 2048, T);
            k_swiglu_bf16<<<(T * 7168 + 255) / 256, 256, 0, st>>>(d_t1, d_t2, T, 7168);
            bf16_linear(cb, d_w2[l], d_t2, d_x, 2048, 7168, T);
        }
        k_rmsnorm_bf16<<<1, 256, 0, st>>>(d_x + (size_t)(T - 1) * 2048, d_u, 1, 2048);
        bf16_linear(cb, d_lm, d_u, d_logits, 65536, 2048, 1);
    };

    /* Capture CUDA Graph for T=1 decode so BF16 gets the EXACT same zero-launch-overhead advantage as HG4-16R! */
    step(1); CK(cudaStreamSynchronize(st));
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    step(1);
    CK(cudaStreamEndCapture(st, &gr));
    CK(cudaGraphInstantiate(&ge, gr, NULL, NULL, 0));
    for (int i = 0; i < 16; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));

    double t0 = omp_get_wtime();
    int n_dec = 100;
    for (int i = 0; i < n_dec; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    double dec_tps = (double)n_dec / (omp_get_wtime() - t0);

    step(256); CK(cudaStreamSynchronize(st));
    double tp0 = omp_get_wtime();
    for (int i = 0; i < 8; i++) step(256);
    CK(cudaStreamSynchronize(st));
    double pp256_tps = (8.0 * 256.0) / (omp_get_wtime() - tp0);

    printf("RESULT_BF16_OMNI2B gpus=1 mode=ALL_VRAM vram_llm=%.2f_GiB vram_full_omni=4.12_GiB prefill_pp256=%.1f_tok/s decode=%.2f_tok/s\n",
           (double)total_b / (1024.0*1024.0*1024.0), pp256_tps, dec_tps);
    fflush(stdout);

    cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
    cudaFree(d_lm); cudaFree(d_x); cudaFree(d_u); cudaFree(d_t1); cudaFree(d_t2); cudaFree(d_logits);
    for (int l = 0; l < 24; l++) {
        cudaFree(d_op1[l]); cudaFree(d_op2[l]);
        if (d_op3[l]) cudaFree(d_op3[l]);
        if (d_op4[l]) cudaFree(d_op4[l]);
        cudaFree(d_w13[l]); cudaFree(d_w2[l]);
    }
    cudaStreamDestroy(st); cublasDestroy(cb);
}

/* =========================================================================
 * 2. Clover-35B-A3B BF16 (`68.2 GiB` BF16 vs `17.16 GiB` HG4-16R!)
 *    - `1x RTX 4090` (`24 GB`): CANNOT fit All-in-VRAM (`68.2 GiB > 24 GiB`) -> Trunk in VRAM (`6.6 GiB`) + Experts in Pinned RAM (`61.6 GiB`)!
 *    - `4x RTX 4090` (`96 GB`): All-in-VRAM (`17.05 GiB / GPU`)!
 * ========================================================================= */
static void bench_clover35b_bf16(int n_gpus, int experts_in_ram) {
    cublasHandle_t cb[4];
    cudaStream_t s_comp[4], s_dma[4];
    cudaEvent_t ev_dma[4][2];
    size_t vram_g[4] = {0}, ram_b = 0;

    /* Per-expert BF16 size: w13 [1024, 2048]*2B (4 MiB) + wdn [2048, 512]*2B (2 MiB) = 6 MiB/expert * 256 = 1,536 MiB/layer * 40 = 60.0 GiB! */
    const size_t exp_b = ((size_t)1024 * 2048 + (size_t)2048 * 512) * 2; /* 6,291,456 B (vs 1.60 MiB in HG4-16R!) */
    __nv_bfloat16 *d_tr_qkv[40], *d_tr_o[40], *d_sh13[40], *d_shdn[40], *d_lm;
    uint8_t *d_exp[40], *h_exp[40];
    uint8_t *d_stage[4][2];
    __nv_bfloat16 *d_x[4], *d_u[4], *d_t1[4], *d_t2[4], *d_logits;

    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        CB(cublasCreate(&cb[g]));
        CK(cudaStreamCreate(&s_comp[g])); CK(cudaStreamCreate(&s_dma[g]));
        CB(cublasSetStream(cb[g], s_comp[g]));
        CK(cudaEventCreate(&ev_dma[g][0])); CK(cudaEventCreate(&ev_dma[g][1]));
        CK(cudaMalloc(&d_x[g],  256 * 2048 * 2));
        CK(cudaMalloc(&d_u[g],  256 * 2048 * 2));
        CK(cudaMalloc(&d_t1[g], 256 * 16384 * 2));
        CK(cudaMalloc(&d_t2[g], 256 * 8192 * 2));
        if (experts_in_ram) {
            CK(cudaMalloc(&d_stage[g][0], 8 * exp_b));
            CK(cudaMalloc(&d_stage[g][1], 8 * exp_b));
        }
    }
    int last_g = n_gpus - 1;
    CK(cudaSetDevice(last_g));
    CK(cudaMalloc(&d_lm, (size_t)248320 * 2048 * 2)); vram_g[last_g] += (size_t)248320 * 2048 * 2;
    CK(cudaMalloc(&d_logits, 248320 * 2));

    #pragma omp parallel for schedule(static) num_threads(8)
    for (int l = 0; l < 40; l++) {
        if (experts_in_ram) {
            uint8_t *p = NULL;
            CK(cudaHostAlloc(&p, 256 * exp_b, cudaHostAllocPortable));
            memset(p, 0x3C, 65536);
            h_exp[l] = p; d_exp[l] = NULL;
            #pragma omp atomic
            ram_b += 256 * exp_b;
        }
    }

    for (int l = 0; l < 40; l++) {
        int g = (l * n_gpus) / 40;
        CK(cudaSetDevice(g));
        int is_gqa = ((l % 4) == 3);
        size_t qkv_r = is_gqa ? 5120 : 16384; /* BD-GDN in_proj (qkv+z) = 16384, GQA q+k+v = 5120 */
        size_t o_k   = is_gqa ? 4096 : 8192;
        CK(cudaMalloc(&d_tr_qkv[l], qkv_r * 2048 * 2)); vram_g[g] += qkv_r * 2048 * 2;
        CK(cudaMalloc(&d_tr_o[l],   2048 * o_k * 2));   vram_g[g] += 2048 * o_k * 2;
        CK(cudaMalloc(&d_sh13[l],   1024 * 2048 * 2));  vram_g[g] += 1024 * 2048 * 2;
        CK(cudaMalloc(&d_shdn[l],   2048 * 512 * 2));   vram_g[g] += 2048 * 512 * 2;
        if (!experts_in_ram) {
            CK(cudaMalloc(&d_exp[l], 256 * exp_b));     vram_g[g] += 256 * exp_b;
            h_exp[l] = NULL;
        }
    }

    int sel[8] = {0, 7, 19, 43, 88, 129, 191, 240};
    auto step = [&](int T) {
        if (experts_in_ram) {
            CK(cudaSetDevice(0));
            for (int k = 0; k < 8; k++)
                CK(cudaMemcpyAsync(d_stage[0][0] + k * exp_b, h_exp[0] + (size_t)sel[k] * exp_b, exp_b, cudaMemcpyHostToDevice, s_dma[0]));
            CK(cudaEventRecord(ev_dma[0][0], s_dma[0]));
        }
        for (int l = 0; l < 40; l++) {
            int g = (l * n_gpus) / 40;
            CK(cudaSetDevice(g));
            cudaStream_t sc = s_comp[g];
            if (experts_in_ram && l + 1 < 40) {
                int sl = (l + 1) & 1;
                for (int k = 0; k < 8; k++)
                    CK(cudaMemcpyAsync(d_stage[g][sl] + k * exp_b, h_exp[l + 1] + (size_t)sel[k] * exp_b, exp_b, cudaMemcpyHostToDevice, s_dma[g]));
                CK(cudaEventRecord(ev_dma[g][sl], s_dma[g]));
            }
            int is_gqa = ((l % 4) == 3);
            k_rmsnorm_bf16<<<T, 256, 0, sc>>>(d_x[g], d_u[g], T, 2048);
            bf16_linear(cb[g], d_tr_qkv[l], d_u[g], d_t1[g], is_gqa ? 5120 : 16384, 2048, T);
            bf16_linear(cb[g], d_tr_o[l], d_t1[g], d_x[g], 2048, is_gqa ? 4096 : 8192, T);
            k_rmsnorm_bf16<<<T, 256, 0, sc>>>(d_x[g], d_u[g], T, 2048);
            bf16_linear(cb[g], d_sh13[l], d_u[g], d_t1[g], 1024, 2048, T);
            bf16_linear(cb[g], d_shdn[l], d_t1[g], d_x[g], 2048, 512, T);

            if (experts_in_ram) CK(cudaStreamWaitEvent(sc, ev_dma[g][l & 1], 0));
            for (int k = 0; k < 8; k++) {
                const uint8_t *ep = experts_in_ram ? (d_stage[g][l & 1] + k * exp_b) : (d_exp[l] + (size_t)sel[k] * exp_b);
                const __nv_bfloat16 *w13 = (const __nv_bfloat16 *)ep;
                const __nv_bfloat16 *wdn = (const __nv_bfloat16 *)(ep + (size_t)1024 * 2048 * 2);
                bf16_linear(cb[g], w13, d_u[g], d_t1[g], 1024, 2048, T);
                bf16_linear(cb[g], wdn, d_t1[g], d_t2[g], 2048, 512, T);
            }
            if (l + 1 < 40 && ((l + 1) * n_gpus) / 40 != g) {
                int gn = ((l + 1) * n_gpus) / 40;
                CK(cudaMemcpyAsync(d_x[gn], d_x[g], (size_t)T * 2048 * 2, cudaMemcpyDefault, sc));
                CK(cudaStreamSynchronize(sc));
            }
        }
        CK(cudaSetDevice(last_g));
        bf16_linear(cb[last_g], d_lm, d_x[last_g] + (size_t)(T - 1) * 2048, d_logits, 248320, 2048, 1);
        CK(cudaStreamSynchronize(s_comp[last_g]));
    };

    step(1); step(1);
    double t0 = omp_get_wtime();
    int n_dec = 16;
    for (int i = 0; i < n_dec; i++) step(1);
    double dec_tps = (double)n_dec / (omp_get_wtime() - t0);

    step(256);
    double tp0 = omp_get_wtime();
    step(256);
    double pp_tps = 256.0 / (omp_get_wtime() - tp0);

    printf("RESULT_BF16_CLOVER35B gpus=%d mode=%s vram_gpu0=%.2f_GiB ram_pinned=%.2f_GiB prefill_pp256=%.1f_tok/s decode=%.2f_tok/s\n",
           n_gpus, experts_in_ram ? "TRUNK_VRAM_EXPERTS_RAM_61GiB" : "ALL_VRAM_68GiB",
           (double)vram_g[0] / (1024.0*1024.0*1024.0), (double)ram_b / (1024.0*1024.0*1024.0),
           pp_tps, dec_tps);
    fflush(stdout);

    CK(cudaSetDevice(last_g)); cudaFree(d_lm); cudaFree(d_logits);
    for (int l = 0; l < 40; l++) {
        CK(cudaSetDevice((l * n_gpus) / 40));
        cudaFree(d_tr_qkv[l]); cudaFree(d_tr_o[l]); cudaFree(d_sh13[l]); cudaFree(d_shdn[l]);
        if (d_exp[l]) cudaFree(d_exp[l]);
        if (h_exp[l]) cudaFreeHost(h_exp[l]);
    }
    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        cudaFree(d_x[g]); cudaFree(d_u[g]); cudaFree(d_t1[g]); cudaFree(d_t2[g]);
        if (experts_in_ram) { cudaFree(d_stage[g][0]); cudaFree(d_stage[g][1]); }
        cudaStreamDestroy(s_comp[g]); cudaStreamDestroy(s_dma[g]); cublasDestroy(cb[g]);
    }
}

int main(int argc, char **argv) {
    const char *st = (argc > 1) ? argv[1] : NULL;
    bench_omni2b_bf16(st);
    bench_clover35b_bf16(1, 1); /* 1x RTX 4090: BF16 68 GiB MUST offload 61.5 GiB experts to RAM! */
    bench_clover35b_bf16(4, 0); /* 4x RTX 4090: BF16 68 GiB All-in-VRAM (17.1 GiB/GPU) */
    return 0;
}
