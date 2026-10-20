// The mixture-of-experts kernels shared by the GPU layer programs: router (all-double, or the fast version with -DFAST_ROUTER), top 16, the expert products
// (expert_proj, or the register-tiled expert_proj_v2 with -DEXPERT_V2), the rank-ordered sum and the final residual sum. See layer1.cu for what they follow.
#pragma once
#include "kernels.cuh"
enum { LATENT = 3584, EXPERT = 3072, SHARED = 6144, EXPERT_RAW = 17547264, EXP_BLOCK01 = 121856, EXP_BLOCK2 = 104448, EXPERTS = 896, TOP = 16 };
#ifdef EXPERT_V2
#define EXPERT_LAUNCH(rbs, ...) expert_proj_v2<<<dim3(rbs, EXPERTS), 128>>>(__VA_ARGS__)
#else
#define EXPERT_LAUNCH(rbs, ...) expert_proj<<<dim3(rbs, EXPERTS), 256>>>(__VA_ARGS__)
#endif

// ---------------------------------------------------------------- the router
__global__ void transpose_rows(float *out, const float *in, int N, int width)   // out[c * N + n] = in[n * width + c]
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * width) return;
    const int n = (int)(i / width), c = (int)(i % width);
    out[(size_t)c * N + n] = in[i];
}

// score of expert e for token n: sigmoid((float) sum over c of (double)w[e][c] * (double)x[n][c]), summed in index order like the CPU
__global__ void router_scores(float *scores, float *choices, const unsigned char *router, const unsigned char *bias, const float *post_t, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x, e = blockIdx.y;
    if (n >= N) return;
    const float *w = (const float *)router + (size_t)e * W;
    double s = 0.0;
    for (int c = 0; c < W; c++) s += (double)w[c] * (double)post_t[(size_t)c * N + n];
    const float sc = sigmoidf_((float)s);
    scores[(size_t)n * EXPERTS + e] = sc;
    choices[(size_t)n * EXPERTS + e] = sc + f32at(bias + (size_t)e * 4);
}

// the 16 best choices in index order (a tie goes to the lower expert), and their weights; with `list`, token i of the launch is token list[i]
__global__ void top16(const float *scores, const float *choices, int *selected, float *weights, int *bad, int N, const int *list = nullptr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int n = list ? list[i] : i;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;
    float best[TOP]; int sel[TOP]; int count = 0;
    for (int e = 0; e < EXPERTS; e++) {
        const float choice = ch[e];
        if (!isfinite(choice)) { atomicAdd(bad, 1); return; }
        if (count == TOP && !(choice > best[TOP - 1])) continue;
        int pos = count < TOP ? count : TOP - 1;
        while (pos && choice > best[pos - 1]) { best[pos] = best[pos - 1]; sel[pos] = sel[pos - 1]; pos--; }
        best[pos] = choice; sel[pos] = e;
        if (count < TOP) count++;
    }
    double total = 0.0;
    for (int r = 0; r < TOP; r++) total += (double)sc[sel[r]];
    const float inverse = (float)(1.0 / (total + 1e-20));
    for (int r = 0; r < TOP; r++) { selected[n * TOP + r] = sel[r]; weights[n * TOP + r] = sc[sel[r]] * inverse; }
}

// ---------------------------------------------------------------- the router, fast version (-DFAST_ROUTER)
// The float product (cuBLAS) gives every expert's score to about 1e-5. The 16 chosen experts are then the ones whose approximate choice is within
// MARGIN of the 16th: for those only, the exact double score is summed in index order as above, and the selection and the weights are made from the
// exact values. If the true top 16 are all among the candidates the result is the same as the all-double router's; main() checks that on every token.
#ifdef FAST_ROUTER
#ifndef FAST_CAND
#define FAST_CAND 48
#endif
enum { CAND = FAST_CAND };
#define MARGIN 1e-3f
#ifdef ROUTER_CHECK
__device__ int g_ntrue[8192];
#endif
__global__ void router_candidates(const float *approx, const unsigned char *bias, int *cand, int *ncand, int *overflow, int *over_list, int N)   // approx[e * N + n]
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *a = approx + n;
    float best[TOP]; int count = 0;
    for (int e = 0; e < EXPERTS; e++) {
        const float choice = sigmoidf_(a[(size_t)e * N]) + f32at(bias + (size_t)e * 4);
        if (count == TOP && !(choice > best[TOP - 1])) continue;
        int pos = count < TOP ? count : TOP - 1;
        while (pos && choice > best[pos - 1]) { best[pos] = best[pos - 1]; pos--; }
        best[pos] = choice;
        if (count < TOP) count++;
    }
    const float threshold = best[TOP - 1] - MARGIN;
    int c = 0;
    for (int e = 0; e < EXPERTS; e++) {
        if (sigmoidf_(a[(size_t)e * N]) + f32at(bias + (size_t)e * 4) >= threshold) { if (c < CAND) cand[n * CAND + c] = e; c++; }
    }
#ifdef ROUTER_CHECK
    if (n < 8192) g_ntrue[n] = c;
#endif
    if (c > CAND) { over_list[atomicAdd(overflow, 1)] = n; c = 0; }   // this token is routed by router_scores_tokens + top16 instead
    ncand[n] = c;
}

// exact scores and choices of every expert for the tokens in list[0..count), summed in double in index order exactly like router_scores
__global__ void router_scores_tokens(float *scores, float *choices, const unsigned char *router, const unsigned char *bias, const float *post, const int *list, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * EXPERTS) return;
    const int n = list[i / EXPERTS], e = i % EXPERTS;
    const float *w = (const float *)router + (size_t)e * W, *x = post + (size_t)n * W;
    double s = 0.0;
    for (int c = 0; c < W; c++) s += (double)w[c] * (double)x[c];
    const float sc = sigmoidf_((float)s);
    scores[(size_t)n * EXPERTS + e] = sc;
    choices[(size_t)n * EXPERTS + e] = sc + f32at(bias + (size_t)e * 4);
}

// One block per token, one thread per candidate. The candidates' weight rows are staged through shared memory in chunks of KC columns so the global loads are coalesced;
// each thread still sums its own row in index order. (double)w * (double)x is exact (24 + 24 bits), so the fused multiply-add rounds exactly like the multiply and add of router_scores.
__global__ void __launch_bounds__(64) router_exact(float *cand_score, const int *cand, const int *ncand, const unsigned char *router, const float *post, int N)
{
    static_assert(CAND <= 64, "one thread per candidate");
    enum { KC = 128 };
    __shared__ float sw[CAND * (KC + 1)];
    __shared__ double sx[KC];
    const int n = blockIdx.x, nc = ncand[n], t = threadIdx.x;
    if (nc == 0) return;
    const float *x = post + (size_t)n * W;
    double s = 0.0;
    for (int kc = 0; kc < W; kc += KC) {
        for (int i = t; i < nc * KC; i += 64) { const int c = i / KC, k = i % KC; sw[c * (KC + 1) + k] = ((const float *)router)[(size_t)cand[n * CAND + c] * W + kc + k]; }
        for (int k = t; k < KC; k += 64) sx[k] = (double)x[kc + k];
        __syncthreads();
        if (t < nc) { const float *r = sw + t * (KC + 1); for (int k = 0; k < KC; k++) s = __fma_rn((double)r[k], sx[k], s); }
        __syncthreads();
    }
    if (t < nc) cand_score[n * CAND + t] = sigmoidf_((float)s);
}

// A candidate whose exact score is further than MARGIN/4 from its approximate one means the approximation is not behaving as the margin assumes: that token goes to the all-double path.
__global__ void top16_candidates(const float *cand_score, const int *cand, const int *ncand, const unsigned char *bias, const float *approx, int *selected, float *weights, int *overflow, int *over_list, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    if (ncand[n] == 0) return;   // routed by the all-double path
    for (int c = 0; c < ncand[n]; c++)
        if (fabsf(sigmoidf_(approx[(size_t)cand[n * CAND + c] * N + n]) - cand_score[n * CAND + c]) > MARGIN / 4) { over_list[atomicAdd(overflow, 1)] = n; return; }
    float best[TOP]; int sel[TOP], slot[TOP]; int count = 0;
    for (int c = 0; c < ncand[n]; c++) {   // candidates are in index order, so a tie goes to the lower expert
        const int e = cand[n * CAND + c];
        const float choice = cand_score[n * CAND + c] + f32at(bias + (size_t)e * 4);
        if (count == TOP && !(choice > best[TOP - 1])) continue;
        int pos = count < TOP ? count : TOP - 1;
        while (pos && choice > best[pos - 1]) { best[pos] = best[pos - 1]; sel[pos] = sel[pos - 1]; slot[pos] = slot[pos - 1]; pos--; }
        best[pos] = choice; sel[pos] = e; slot[pos] = c;
        if (count < TOP) count++;
    }
    double total = 0.0;
    for (int r = 0; r < TOP; r++) total += (double)cand_score[n * CAND + slot[r]];
    const float inverse = (float)(1.0 / (total + 1e-20));
    for (int r = 0; r < TOP; r++) { selected[n * TOP + r] = sel[r]; weights[n * TOP + r] = cand_score[n * CAND + slot[r]] * inverse; }
}

#ifdef ROUTER_CHECK
// stats: [0] max |approximate choice - exact choice| (float bits), [1] max true candidate count, [2] tokens whose selection or weights differ from the all-double router's,
// [3] tokens with more than CAND candidates, [8 + k] tokens with k true candidates (k capped at 119)
__global__ void router_check(const float *approx, const float *choices, const unsigned char *bias, const int *sel_fast, const float *w_fast, const int *sel_ref, const float *w_ref, unsigned *stats, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float dmax = 0.f;
    for (int e = 0; e < EXPERTS; e++) {
        const float d = fabsf((sigmoidf_(approx[(size_t)e * N + n]) + f32at(bias + (size_t)e * 4)) - choices[(size_t)n * EXPERTS + e]);
        if (d > dmax) dmax = d;
    }
    atomicMax(&stats[0], __float_as_uint(dmax));
    const int nt = n < 8192 ? g_ntrue[n] : 0;
    atomicMax(&stats[1], (unsigned)nt);
    if (nt > CAND) atomicAdd(&stats[3], 1u);
    atomicAdd(&stats[8 + (nt < 119 ? nt : 119)], 1u);
    bool same = true;
    for (int r = 0; r < TOP; r++) same = same && sel_fast[n * TOP + r] == sel_ref[n * TOP + r] && __float_as_uint(w_fast[n * TOP + r]) == __float_as_uint(w_ref[n * TOP + r]);
    if (!same) atomicAdd(&stats[2], 1u);
}
#endif
#endif

// ---------------------------------------------------------------- the experts
// One block = 64 rows of one matrix of one expert, for every token routed to that expert, 32 tokens at a time.
// matrix 0 (gate) and 1 (up): input = the token's latent (3584), output 3072 rows. matrix 2 (down): input = that slot's activation (3072), output 3584 rows.
// A row holds 4-bit codes (low nibble = even column) and one selector byte per 32 columns; the selector picks a 16-value map through the template of the matrix.
__global__ void __launch_bounds__(256) expert_proj(const unsigned char *direct, const float *mapvals, const uint16_t *refs, const uint32_t *ref_base, int matrix,
    const float *X, int x_stride, const int *slot_token, const int *expert_start, const int *expert_count, float *O, int o_stride)
{
    const int e = blockIdx.y, rb = blockIdx.x;
    const int count = expert_count[e];
    if (!count) return;
    const int start = expert_start[e];
    const int K = matrix == 2 ? EXPERT : LATENT;
    const unsigned char *block = direct + (size_t)e * EXPERT_RAW + (matrix < 2 ? (size_t)(matrix * 48 + rb) * EXP_BLOCK01 : (size_t)96 * EXP_BLOCK01 + (size_t)rb * EXP_BLOCK2);
    const unsigned char *sel_base = block + 64 * K / 2;
    const uint32_t rbase = ref_base[e * 3 + matrix];
    __shared__ float Ws[64][65], Xs[64][33];
    const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
    for (int t0 = 0; t0 < count; t0 += 32) {
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += 64) {
            {   // 64 rows x 64 columns of weights: 4 threads per row, 16 columns each
                const int row = tid >> 2, part = tid & 3, col = k0 + part * 16;
                const uint2 bytes = *(const uint2 *)(block + (size_t)row * (K / 2) + col / 2);
                const unsigned sel = sel_base[(size_t)row * (K / 32) + col / 32];
                const float *vals = mapvals + (size_t)refs[rbase + sel] * 16;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const unsigned b = ((j < 4 ? bytes.x : bytes.y) >> ((j & 3) * 8)) & 0xff;
                    Ws[part * 16 + 2 * j][row] = vals[b & 15];
                    Ws[part * 16 + 2 * j + 1][row] = vals[b >> 4];
                }
            }
            {   // 32 tokens x 64 inputs
                const int tk = tid >> 3, kp = (tid & 7) * 8;
                float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                if (t0 + tk < count) {
                    const int slot = start + t0 + tk;
                    const float *src = X + (size_t)(matrix == 2 ? slot : slot_token[slot]) * x_stride + k0 + kp;
                    const float4 a = *(const float4 *)src, b = *(const float4 *)(src + 4);
                    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) Xs[kp + j][tk] = v[j];
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
                const float x = Xs[kk][tx];
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[i] = fmaf(Ws[kk][ty * 8 + i], x, acc[i]);
            }
            __syncthreads();
        }
        if (t0 + tx < count) {
            float *o = O + (size_t)(start + t0 + tx) * o_stride + rb * 64 + ty * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) o[i] = acc[i];
        }
    }
}

// Same products with the same order of additions per output (so the same bits as expert_proj), with each thread holding 4 rows x 4 tokens so that a
// 16-byte shared read feeds 8 multiply-adds instead of 1: a block = 64 rows, 32 tokens per step, 128 threads (-DEXPERT_V2).
__global__ void __launch_bounds__(128) expert_proj_v2(const unsigned char *direct, const float *mapvals, const uint16_t *refs, const uint32_t *ref_base, int matrix,
    const float *X, int x_stride, const int *slot_token, const int *expert_start, const int *expert_count, float *O, int o_stride)
{
    const int e = blockIdx.y, rb = blockIdx.x;
    const int count = expert_count[e];
    if (!count) return;
    const int start = expert_start[e];
    const int K = matrix == 2 ? EXPERT : LATENT;
    const unsigned char *block = direct + (size_t)e * EXPERT_RAW + (matrix < 2 ? (size_t)(matrix * 48 + rb) * EXP_BLOCK01 : (size_t)96 * EXP_BLOCK01 + (size_t)rb * EXP_BLOCK2);
    const unsigned char *sel_base = block + 64 * K / 2;
    const uint32_t rbase = ref_base[e * 3 + matrix];
    __shared__ __align__(16) float Ws[64][68];
    __shared__ __align__(16) float Xs[64][36];
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    for (int t0 = 0; t0 < count; t0 += 32) {
        float acc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += 64) {
            {   // 64 rows x 64 columns: 2 threads per row, 32 columns (16 bytes, one selector) each
                const int row = tid >> 1, part = tid & 1, col = k0 + part * 32;
                const uint4 bytes = *(const uint4 *)(block + (size_t)row * (K / 2) + col / 2);
                const unsigned sel = sel_base[(size_t)row * (K / 32) + col / 32];
                const float *vals = mapvals + (size_t)refs[rbase + sel] * 16;
                const unsigned words[4] = { bytes.x, bytes.y, bytes.z, bytes.w };
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        const unsigned v = (words[q] >> (b * 8)) & 0xff;
                        const int k = part * 32 + (q * 4 + b) * 2;
                        Ws[k][row] = vals[v & 15];
                        Ws[k + 1][row] = vals[v >> 4];
                    }
            }
            {   // 32 tokens x 64 inputs: 4 threads per token, 16 inputs each
                const int tk = tid >> 2, kp = (tid & 3) * 16;
                float4 v[4];
                #pragma unroll
                for (int q = 0; q < 4; q++) v[q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (t0 + tk < count) {
                    const int slot = start + t0 + tk;
                    const float *src = X + (size_t)(matrix == 2 ? slot : slot_token[slot]) * x_stride + k0 + kp;
                    #pragma unroll
                    for (int q = 0; q < 4; q++) v[q] = *(const float4 *)(src + q * 4);
                }
                #pragma unroll
                for (int q = 0; q < 4; q++) { Xs[kp + q * 4 + 0][tk] = v[q].x; Xs[kp + q * 4 + 1][tk] = v[q].y; Xs[kp + q * 4 + 2][tk] = v[q].z; Xs[kp + q * 4 + 3][tk] = v[q].w; }
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
                const float4 w4 = *(const float4 *)&Ws[kk][ty * 4], x4 = *(const float4 *)&Xs[kk][tx * 4];
                const float w[4] = { w4.x, w4.y, w4.z, w4.w }, x[4] = { x4.x, x4.y, x4.z, x4.w };
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = fmaf(w[i], x[j], acc[i][j]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = t0 + tx * 4 + j;
            if (t < count) *(float4 *)(O + (size_t)(start + t) * o_stride + rb * 64 + ty * 4) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
        }
    }
}

// ---------------------------------------------------------------- expert_proj_v3
// The same products with the same order of additions per output as expert_proj (k ascending, one fmaf per step, from 0), organised per warp:
// a work item is one expert and up to 32 of its tokens; a block is ONE warp that computes a 2048-output tile (rows x tokens) with 8 rows x 8 tokens per
// lane, so one step of k costs 3 shared-memory reads for 64 multiply-adds. The tile is 64 rows x 32 tokens, or 128 rows x 16 tokens for an item of at most
// 16 tokens (half the padding). The weights are decoded into shared memory as bfloat16 pairs (the map values are bfloat16 exactly, so the product is
// unchanged), 32 columns at a time; the next 64 columns are fetched into registers while the current ones are multiplied.
template <int T>
__device__ __forceinline__ void expert_tile_v3(unsigned *Ws, float *Xs, const unsigned char *eblk, const float *mapvals, const uint16_t *refs, uint32_t rbase,
    int matrix, const float *X, int x_stride, const int *slot_token, int slot0, int n, float *O, int o_stride)
{
    constexpr int R = 2048 / T, U = R / 32, LX = T / 8, Q = T / 4;
    const int K = matrix == 2 ? EXPERT : LATENT, rows = matrix == 2 ? LATENT : EXPERT;
    const int row0 = blockIdx.x * R;
    if (row0 >= rows) return;
    const int lane = threadIdx.x, lx = lane % LX, ly = lane / LX;
    const unsigned char *cp[U], *sp[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
        const int gr = row0 + lane + 32 * u, rb = gr >> 6, lr = gr & 63;
        const unsigned char *blk = eblk + (matrix < 2 ? (size_t)(matrix * 48 + rb) * EXP_BLOCK01 : (size_t)96 * EXP_BLOCK01 + (size_t)rb * EXP_BLOCK2);
        cp[u] = blk + (size_t)lr * (K / 2);
        sp[u] = blk + (size_t)64 * K / 2 + (size_t)lr * (K / 32);
    }
    const float *xp[Q];
    #pragma unroll
    for (int q = 0; q < Q; q++) {
        const int t = q * 4 + (lane >> 3);
        xp[q] = t < n ? X + (size_t)(matrix == 2 ? slot0 + t : slot_token[slot0 + t]) * x_stride + (lane & 7) * 4 : nullptr;
    }
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    uint4 wr[U][2]; unsigned short sv[U]; float4 xr[2][Q];
    auto fetch = [&](int tile) {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            wr[u][0] = *(const uint4 *)(cp[u] + tile * 32); wr[u][1] = *(const uint4 *)(cp[u] + tile * 32 + 16);
            sv[u] = *(const unsigned short *)(sp[u] + tile * 2);
        }
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int q = 0; q < Q; q++) xr[h][q] = xp[q] ? *(const float4 *)(xp[q] + tile * 64 + h * 32) : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    fetch(0);
    for (int tile = 0; tile < K / 64; tile++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            __syncwarp();
            #pragma unroll
            for (int u = 0; u < U; u++) {
                const unsigned sel = ((h ? sv[u] >> 8 : sv[u]) & 0xff);
                const float *vals = mapvals + (size_t)refs[rbase + sel] * 16;
                const uint4 c = wr[u][h];
                const unsigned words[4] = { c.x, c.y, c.z, c.w };
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        const unsigned v = (words[q] >> (b * 8)) & 0xff;
                        Ws[(q * 4 + b) * R + lane + 32 * u] = __byte_perm(__float_as_uint(__ldg(vals + (v & 15))), __float_as_uint(__ldg(vals + (v >> 4))), 0x7632);
                    }
            }
            #pragma unroll
            for (int q = 0; q < Q; q++) {
                const int t = q * 4 + (lane >> 3), k4 = (lane & 7) * 4;
                const float4 f = xr[h][q];
                Xs[(k4 + 0) * T + t] = f.x; Xs[(k4 + 1) * T + t] = f.y; Xs[(k4 + 2) * T + t] = f.z; Xs[(k4 + 3) * T + t] = f.w;
            }
            if (h == 1 && tile + 1 < K / 64) fetch(tile + 1);
            __syncwarp();
            #pragma unroll 4
            for (int kp = 0; kp < 16; kp++) {
                const uint4 wa = *(const uint4 *)&Ws[kp * R + ly * 8], wb = *(const uint4 *)&Ws[kp * R + ly * 8 + 4];
                const unsigned wd[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
                #pragma unroll
                for (int par = 0; par < 2; par++) {
                    const float4 xa = *(const float4 *)&Xs[(2 * kp + par) * T + lx * 8], xb = *(const float4 *)&Xs[(2 * kp + par) * T + lx * 8 + 4];
                    const float xv[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float w = __uint_as_float(par ? (wd[i] & 0xffff0000u) : (wd[i] << 16));
                        #pragma unroll
                        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(w, xv[j], acc[i][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int t = lx * 8 + j;
        if (t < n) {
            float *o = O + (size_t)(slot0 + t) * o_stride + row0 + ly * 8;
            *(float4 *)o = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            *(float4 *)(o + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
        }
    }
}

// items[i] = (expert, first slot, tokens, tile width 16 or 32); grid = (rows / 64, items)
__global__ void __launch_bounds__(32) expert_proj_v3(const unsigned char *direct, const float *mapvals, const uint16_t *refs, const uint32_t *ref_base, int matrix,
    const float *X, int x_stride, const int *slot_token, const int4 *items, float *O, int o_stride)
{
    __shared__ __align__(16) unsigned smem[2048 + 1024];
    const int4 it = items[blockIdx.y];
    const unsigned char *eblk = direct + (size_t)it.x * EXPERT_RAW;
    const uint32_t rbase = ref_base[it.x * 3 + matrix];
    if (it.w == 16) expert_tile_v3<16>(smem, (float *)(smem + 2048), eblk, mapvals, refs, rbase, matrix, X, x_stride, slot_token, it.y, it.z, O, o_stride);
    else expert_tile_v3<32>(smem, (float *)(smem + 2048), eblk, mapvals, refs, rbase, matrix, X, x_stride, slot_token, it.y, it.z, O, o_stride);
}

// mixture[n][c] = sum over ranks in order of weight * down[slot of (n, rank)][c]
__global__ void mix_ranks(float *mixture, const float *down, const int *slot_of, const float *weights, int N)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * LATENT) return;
    const int n = (int)(i / LATENT), c = (int)(i % LATENT);
    float m = 0.0f;
    for (int r = 0; r < TOP; r++) m = m + weights[n * TOP + r] * down[(size_t)slot_of[n * TOP + r] * LATENT + c];
    mixture[i] = m;
}

__global__ void final_sum(float *res, const float *routed, const float *shared, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) res[i] = res[i] + (routed[i] + shared[i]);
}

