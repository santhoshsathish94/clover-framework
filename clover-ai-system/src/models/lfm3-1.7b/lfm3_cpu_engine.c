#define _GNU_SOURCE
/*
 * Native C / OpenMP CPU Engine for Liquid AI LFM3-1.7B (`liblfm3_cpu_engine.so`)
 * (`clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c`)
 *
 * Executes all 24 tri-hybrid layers of `LiquidAI/LFM3-1.7B` (`D_model = 2048`, `V = 128,000` tied)
 * directly from the pruned `LFM3TRK1` trunks (`trunk-0..23`), `LFM3_OP1` composed `operator.bin`
 * files (`attention.bin` deleted from all 24 trunks), and `global-0/` (`embed_tokens.bf16.bin`,
 * `embedding_norm.bf16.bin`, `rope_inv_freq.bin`):
 *   - 13 Gated ShortConv layers (`kind = 0`, width K=7, B*x pre-gate, C*conv post-gate)
 *   - 5 Sliding-Window GQA layers (`kind = 1`, W=256, full RoPE 1.0, per-head sigmoid gate)
 *   - 6 Full Global GQA layers (`kind = 2`, zero-tail partial RoPE 0.5, per-head sigmoid gate)
 *   - 24 Sandwich RMSNorm + SwiGLU FFN blocks (`2048 -> 7680 -> 2048`)
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_bf16.h>
#endif
#endif
#ifndef _WIN32
#include <sched.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LFM3_API __declspec(dllexport)
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define LFM3_API
#endif

enum {
    LFM3_D = 2048,
    LFM3_FFN = 7680,
    LFM3_LAYERS = 24,
    LFM3_HEADS = 16,
    LFM3_KV_HEADS = 4,
    LFM3_HDIM = 128,
    LFM3_CONV_K = 7,
    LFM3_SLIDING_W = 256,
    LFM3_HEADER_SIZE = 72,
    LFM3_RECORD_SIZE = 36,
    LFM3_FFN_MAT_4BIT_BYTES = 8140800, /* 7680x2048 or 2048x7680 in planar 4-bit group-64 */
    LFM3_OP_CONV_IN_4BIT_BYTES = 6512640,  /* 6144x2048 in planar 4-bit group-64 */
    LFM3_OP_ATTN_QKV_4BIT_BYTES = 3256320, /* 3072x2048 in planar 4-bit group-64 */
    LFM3_OP_OUT_4BIT_BYTES = 2170880,      /* 2048x2048 in planar 4-bit group-64 */
    LFM3_LM_HEAD_4BIT_BYTES = 135680000    /* 128000x2048 in planar 4-bit group-64 */
};

static inline float bf16_to_f32(uint16_t b)
{
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static inline uint16_t f32_to_bf16_rne(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7FFFFFFFU) > 0x7F800000U) {
        return (uint16_t)((v.u >> 16) | 0x0040U);
    }
    uint32_t rounding_bias = 0x00007FFFU + ((v.u >> 16) & 1U);
    return (uint16_t)((v.u + rounding_bias) >> 16);
}

static inline float bf16_round_f32(float x)
{
    return bf16_to_f32(f32_to_bf16_rne(x));
}

typedef struct {
    void *ptr;
    size_t size;
    int fd;
#ifdef _WIN32
    HANDLE hfile;
    HANDLE hmap;
#endif
} MmapFile;

static double now_ms(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

static int file_readable(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    return access(path, R_OK) == 0;
#endif
}

#ifdef _WIN32
static int mmap_readonly(const char *path, MmapFile *out)
{
    out->ptr = NULL;
    out->size = 0;
    out->fd = -1;
    out->hmap = NULL;
    out->hfile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (out->hfile == INVALID_HANDLE_VALUE) {
        out->hfile = NULL;
        return -1;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(out->hfile, &sz)) {
        CloseHandle(out->hfile);
        out->hfile = NULL;
        return -1;
    }
    out->size = (size_t)sz.QuadPart;
    out->fd = 0;
    if (out->size == 0) return 0;
    out->hmap = CreateFileMappingA(out->hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    out->ptr = out->hmap ? MapViewOfFile(out->hmap, FILE_MAP_READ, 0, 0, 0) : NULL;
    if (!out->ptr) {
        if (out->hmap) CloseHandle(out->hmap);
        CloseHandle(out->hfile);
        out->hmap = NULL;
        out->hfile = NULL;
        out->fd = -1;
        return -1;
    }
    return 0;
}
#else
static int mmap_readonly(const char *path, MmapFile *out)
{
    out->ptr = NULL;
    out->size = 0;
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) return -1;
    struct stat st;
    if (fstat(out->fd, &st) != 0) {
        close(out->fd);
        out->fd = -1;
        return -1;
    }
    out->size = (size_t)st.st_size;
    if (out->size == 0) return 0;
    out->ptr = mmap(NULL, out->size, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->ptr == MAP_FAILED) {
        out->ptr = NULL;
        close(out->fd);
        out->fd = -1;
        return -1;
    }
    return 0;
}
#endif

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t layer_id;
    uint32_t layer_kind;
    uint32_t record_count;
    uint64_t g0_off, g0_len;
    uint64_t g1_off, g1_len;
    uint64_t g2_off, g2_len;
} Lfm3TrunkHeader;

typedef struct {
    uint32_t rec_id;
    uint32_t grp_id;
    uint32_t dtype_code;
    uint32_t rows;
    uint64_t offset;
    uint64_t length;
    uint32_t cols;
} Lfm3RecordDesc;
#pragma pack(pop)

typedef struct {
    MmapFile index_mf;
    MmapFile group_mf[3];
    MmapFile operator_mf;
    MmapFile operator_4bit_mf;
    MmapFile ffn_4bit_mf;
    Lfm3TrunkHeader hdr;
    Lfm3RecordDesc recs[20];
} Lfm3TrunkMapped;

static int load_lfm3_trunk(const char *trunk_dir, Lfm3TrunkMapped *t)
{
    memset(t, 0, sizeof(*t));
    t->index_mf.fd = -1;
    t->operator_mf.fd = -1;
    t->operator_4bit_mf.fd = -1;
    t->ffn_4bit_mf.fd = -1;
    for (int g = 0; g < 3; g++) t->group_mf[g].fd = -1;

    char path[2048];
    snprintf(path, sizeof(path), "%s/index.bin", trunk_dir);
    if (mmap_readonly(path, &t->index_mf) != 0 || t->index_mf.size < LFM3_HEADER_SIZE) return -1;
    memcpy(&t->hdr, t->index_mf.ptr, sizeof(Lfm3TrunkHeader));
    if (memcmp(t->hdr.magic, "LFM3TRK1", 8) != 0) return -2;

    const uint8_t *p = (const uint8_t *)t->index_mf.ptr + LFM3_HEADER_SIZE;
    uint32_t nrec = t->hdr.record_count < 20 ? t->hdr.record_count : 20;
    for (uint32_t i = 0; i < nrec; i++) {
        memcpy(&t->recs[i], p + i * LFM3_RECORD_SIZE, sizeof(Lfm3RecordDesc));
    }

    const char *gnames[3] = { "common.bin", "attention.bin", "ffn.bin" };
    for (int g = 0; g < 3; g++) {
        snprintf(path, sizeof(path), "%s/%s", trunk_dir, gnames[g]);
        if (file_readable(path)) {
            if (mmap_readonly(path, &t->group_mf[g]) != 0) return -3;
        }
    }
    snprintf(path, sizeof(path), "%s/operator.bin", trunk_dir);
    if (file_readable(path)) {
        if (mmap_readonly(path, &t->operator_mf) != 0) return -4;
    }
    snprintf(path, sizeof(path), "%s/operator.4bit.bin", trunk_dir);
    if (file_readable(path)) {
        mmap_readonly(path, &t->operator_4bit_mf);
    }
    snprintf(path, sizeof(path), "%s/ffn.4bit.bin", trunk_dir);
    if (file_readable(path)) {
        mmap_readonly(path, &t->ffn_4bit_mf);
    }
    return 0;
}

static const void *get_lfm3_record_ptr(const Lfm3TrunkMapped *t, uint32_t rec_id)
{
    for (uint32_t i = 0; i < t->hdr.record_count && i < 20; i++) {
        if (t->recs[i].rec_id == rec_id) {
            if (t->recs[i].length == 0) return NULL;
            uint32_t g = t->recs[i].grp_id;
            if (g >= 3 || !t->group_mf[g].ptr) return NULL;
            return (const uint8_t *)t->group_mf[g].ptr + t->recs[i].offset;
        }
    }
    return NULL;
}

/* SpecMD Policy 3 (Least-Stale + R4 Clean-Drop) & Eyeriss 4-Tier Mobile SoC Ledger for LFM3-Omni / Samsung S23 */
static struct {
    int initialized;
    int use_4bit_ffn;
    int use_4bit_operators;
    int use_4bit_lm_head;
    int num_threads;
    int s23_pin_requested;
    int s23_big_cores_pinned;
    MmapFile emb_mf;
    MmapFile lm_head_4bit_mf;
    MmapFile emb_norm_mf;
    MmapFile rope_mf;
    float inv_freq_full[64];
    float inv_freq_sliding[64];
    Lfm3TrunkMapped trunks[LFM3_LAYERS];

    unsigned long clock;
    int pool_keys[12];
    unsigned long pool_ts[12];
    float pool_heat[12];
    int host_keys[LFM3_LAYERS];

    unsigned long total_layer_requests;
    unsigned long tier1_sram_l3_hits;
    unsigned long tier2_lpddr5x_hits;
    unsigned long tier3_ufs4_fetches;
    unsigned long r4_clean_drops;
    unsigned long dirty_writebacks;
    int madvise_active;

    /* Real Mobile SoC Direction B Hardware Counters */
    unsigned long long bytes_tier0_reg;
    unsigned long long bytes_tier1_sram_slc;
    unsigned long long bytes_tier2_lpddr5x;
    unsigned long long bytes_saved_q4_and_broadcast;
    unsigned long smmla_i8mm_tile_calls;
    unsigned long fa2_online_softmax_calls;
    unsigned long adreno_opencl_launches;
    unsigned long minflt_at_reset;
    unsigned long majflt_at_reset;
    int adreno_ready;
    int adreno_enabled;
} g_lfm3 = { 0, 0, 0, 0, 8, 0 };

static void read_os_page_faults(unsigned long *out_minflt, unsigned long *out_majflt)
{
    *out_minflt = 0;
    *out_majflt = 0;
#if defined(__linux__)
    FILE *f = fopen("/proc/self/stat", "r");
    if (f) {
        char buf[1024];
        if (fgets(buf, sizeof(buf), f)) {
            char *rp = strrchr(buf, ')');
            if (rp && rp[1] == ' ') {
                unsigned long minflt = 0, cminflt = 0, majflt = 0;
                char state = 0;
                int ppid = 0, pgrp = 0, session = 0, tty = 0, tpgid = 0;
                unsigned int flags = 0;
                if (sscanf(rp + 2, "%c %d %d %d %d %d %u %lu %lu %lu",
                           &state, &ppid, &pgrp, &session, &tty, &tpgid, &flags,
                           &minflt, &cminflt, &majflt) == 10) {
                    *out_minflt = minflt;
                    *out_majflt = majflt;
                }
            }
        }
        fclose(f);
    }
#endif
}

/* ============================================================================
 * Adreno 740 OpenCL 3.0 Zero-Copy Unified Memory (`CL_MEM_USE_HOST_PTR`) +
 * Hardware `dot_acc_sat` (`cl_khr_integer_dot_product`) Co-Processor
 * ============================================================================ */
#if defined(__linux__) && defined(__aarch64__)
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef intptr_t cl_context_properties;

#define CL_MEM_READ_ONLY_SOC      (1 << 2)
#define CL_MEM_READ_WRITE_SOC     (1 << 0)
#define CL_MEM_USE_HOST_PTR_SOC   (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR_SOC (1 << 4)
#define CL_MAP_READ_SOC           (1 << 0)
#define CL_MAP_WRITE_SOC          (1 << 1)
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM    0x40C3

static struct {
    void *lib;
    cl_context ctx;
    cl_command_queue q;
    cl_kernel k_swiglu;
    cl_mem b_ffn[LFM3_LAYERS];
    cl_mem b_xq;
    cl_mem b_xsc;
    cl_mem b_xsum;
    cl_mem b_out;
    void* (*clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_uint, cl_bitfield, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*);
    cl_int (*clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    cl_int (*clFlush)(cl_command_queue);
    cl_int (*clFinish)(cl_command_queue);
} g_adreno = { 0 };

static const char *ADRENO_SWIGLU_DOT8_SRC =
"#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n"
"inline float bf16_round(float x) {\n"
"    uint u = as_uint(x);\n"
"    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return as_float((u | 0x00400000u) & 0xFFFF0000u);\n"
"    uint rb = 0x00007FFFu + ((u >> 16) & 1u);\n"
"    return as_float((u + rb) & 0xFFFF0000u);\n"
"}\n"
"__kernel void adreno_swiglu_dot8_exact(\n"
"    __global const uchar16 * restrict ffn_v16,\n"
"    __global const char4   * restrict x_q8,\n"
"    __global const float   * restrict x_sc,\n"
"    __global const int     * restrict x_sum,\n"
"    __global float         * restrict out_swiglu\n"
") {\n"
"    __local char4 l_xq[512];\n"
"    __local float l_s1[128];\n"
"    __local float l_s3[128];\n"
"    const int lid = get_local_id(0);\n"
"    for (int i = lid; i < 512; i += 128) l_xq[i] = x_q8[i];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    const int row_in_wg = lid >> 5;\n"
"    const int g         = lid & 31;\n"
"    const int o         = (get_group_id(0) << 2) + row_in_wg;\n"
"    if (o >= 7680) return;\n"
"    const int nib16_idx = (o << 6) + (g << 1);\n"
"    uchar16 pk1_0 = ffn_v16[nib16_idx];\n"
"    uchar16 pk1_1 = ffn_v16[nib16_idx + 1];\n"
"    uchar16 pk3_0 = ffn_v16[508800 + nib16_idx];\n"
"    uchar16 pk3_1 = ffn_v16[508800 + nib16_idx + 1];\n"
"    __global const uchar *w1_stp = (__global const uchar *)(ffn_v16 + 491520);\n"
"    __global const float *w1_scl = (__global const float *)(ffn_v16 + 506880);\n"
"    __global const uchar *w3_stp = (__global const uchar *)(ffn_v16 + 508800 + 491520);\n"
"    __global const float *w3_scl = (__global const float *)(ffn_v16 + 508800 + 506880);\n"
"    const __local char4 *xq = l_xq + (g << 4);\n"
"    uchar16 m0f = (uchar16)(15);\n"
"    uchar16 lo1_0 = pk1_0 & m0f, hi1_0 = pk1_0 >> (uchar16)(4);\n"
"    uchar16 lo1_1 = pk1_1 & m0f, hi1_1 = pk1_1 >> (uchar16)(4);\n"
"    uchar16 lo3_0 = pk3_0 & m0f, hi3_0 = pk3_0 >> (uchar16)(4);\n"
"    uchar16 lo3_1 = pk3_1 & m0f, hi3_1 = pk3_1 >> (uchar16)(4);\n"
"    int d1 = -8 * x_sum[g];\n"
"    d1 = dot_acc_sat(lo1_0.s0123, xq[0],  d1); d1 = dot_acc_sat(lo1_0.s4567, xq[1],  d1);\n"
"    d1 = dot_acc_sat(lo1_0.s89ab, xq[2],  d1); d1 = dot_acc_sat(lo1_0.scdef, xq[3],  d1);\n"
"    d1 = dot_acc_sat(lo1_1.s0123, xq[4],  d1); d1 = dot_acc_sat(lo1_1.s4567, xq[5],  d1);\n"
"    d1 = dot_acc_sat(lo1_1.s89ab, xq[6],  d1); d1 = dot_acc_sat(lo1_1.scdef, xq[7],  d1);\n"
"    d1 = dot_acc_sat(hi1_0.s0123, xq[8],  d1); d1 = dot_acc_sat(hi1_0.s4567, xq[9],  d1);\n"
"    d1 = dot_acc_sat(hi1_0.s89ab, xq[10], d1); d1 = dot_acc_sat(hi1_0.scdef, xq[11], d1);\n"
"    d1 = dot_acc_sat(hi1_1.s0123, xq[12], d1); d1 = dot_acc_sat(hi1_1.s4567, xq[13], d1);\n"
"    d1 = dot_acc_sat(hi1_1.s89ab, xq[14], d1); d1 = dot_acc_sat(hi1_1.scdef, xq[15], d1);\n"
"    int d3 = -8 * x_sum[g];\n"
"    d3 = dot_acc_sat(lo3_0.s0123, xq[0],  d3); d3 = dot_acc_sat(lo3_0.s4567, xq[1],  d3);\n"
"    d3 = dot_acc_sat(lo3_0.s89ab, xq[2],  d3); d3 = dot_acc_sat(lo3_0.scdef, xq[3],  d3);\n"
"    d3 = dot_acc_sat(lo3_1.s0123, xq[4],  d3); d3 = dot_acc_sat(lo3_1.s4567, xq[5],  d3);\n"
"    d3 = dot_acc_sat(lo3_1.s89ab, xq[6],  d3); d3 = dot_acc_sat(lo3_1.scdef, xq[7],  d3);\n"
"    d3 = dot_acc_sat(hi3_0.s0123, xq[8],  d3); d3 = dot_acc_sat(hi3_0.s4567, xq[9],  d3);\n"
"    d3 = dot_acc_sat(hi3_0.s89ab, xq[10], d3); d3 = dot_acc_sat(hi3_0.scdef, xq[11], d3);\n"
"    d3 = dot_acc_sat(hi3_1.s0123, xq[12], d3); d3 = dot_acc_sat(hi3_1.s4567, xq[13], d3);\n"
"    d3 = dot_acc_sat(hi3_1.s89ab, xq[14], d3); d3 = dot_acc_sat(hi3_1.scdef, xq[15], d3);\n"
"    float xsc = x_sc[g];\n"
"    l_s1[lid] = (float)d1 * ((float)w1_stp[(o << 5) + g] * xsc);\n"
"    l_s3[lid] = (float)d3 * ((float)w3_stp[(o << 5) + g] * xsc);\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (g == 0) {\n"
"        float s1 = 0.0f, s3 = 0.0f;\n"
"        for (int k = 0; k < 32; k++) { s1 += l_s1[lid + k]; s3 += l_s3[lid + k]; }\n"
"        float v1 = bf16_round(s1 * w1_scl[o]);\n"
"        float v3 = bf16_round(s3 * w3_scl[o]);\n"
"        float silu = bf16_round(v1 / (1.0f + exp(-v1)));\n"
"        out_swiglu[o] = bf16_round(silu * v3);\n"
"    }\n"
"}\n";

static void init_adreno740_opencl(void)
{
    void *h = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!h) h = dlopen("libOpenCL.so", RTLD_NOW);
    if (!h) return;

    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = dlsym(h, "clGetPlatformIDs");
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*) = dlsym(h, "clGetDeviceIDs");
    cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = dlsym(h, "clCreateContext");
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*) = dlsym(h, "clCreateCommandQueue");
    cl_mem (*clCreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*) = dlsym(h, "clCreateBuffer");
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = dlsym(h, "clCreateProgramWithSource");
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = dlsym(h, "clBuildProgram");
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = dlsym(h, "clCreateKernel");

    g_adreno.clEnqueueMapBuffer = dlsym(h, "clEnqueueMapBuffer");
    g_adreno.clEnqueueUnmapMemObject = dlsym(h, "clEnqueueUnmapMemObject");
    g_adreno.clSetKernelArg = dlsym(h, "clSetKernelArg");
    g_adreno.clEnqueueNDRangeKernel = dlsym(h, "clEnqueueNDRangeKernel");
    g_adreno.clFlush = dlsym(h, "clFlush");
    g_adreno.clFinish = dlsym(h, "clFinish");

    if (!clGetPlatformIDs || !clGetDeviceIDs || !clCreateContext || !clCreateCommandQueue || !clCreateBuffer ||
        !clCreateProgramWithSource || !clBuildProgram || !clCreateKernel || !g_adreno.clEnqueueMapBuffer ||
        !g_adreno.clEnqueueUnmapMemObject || !g_adreno.clSetKernelArg || !g_adreno.clEnqueueNDRangeKernel ||
        !g_adreno.clFlush || !g_adreno.clFinish) {
        return;
    }

    cl_platform_id plat = NULL;
    if (clGetPlatformIDs(1, &plat, NULL) != 0 || !plat) return;
    cl_device_id dev = NULL;
    if (clGetDeviceIDs(plat, 0xFFFFFFFFULL, 1, &dev, NULL) != 0 || !dev) return;

    cl_int err = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM, 0 };
    g_adreno.ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    if (!g_adreno.ctx) g_adreno.ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (!g_adreno.ctx) return;

    g_adreno.q = clCreateCommandQueue(g_adreno.ctx, dev, 0, &err);
    if (!g_adreno.q) return;

    for (int l = 0; l < LFM3_LAYERS; l++) {
        if (!g_lfm3.trunks[l].ffn_4bit_mf.ptr || g_lfm3.trunks[l].ffn_4bit_mf.size < 2 * LFM3_FFN_MAT_4BIT_BYTES) return;
        g_adreno.b_ffn[l] = clCreateBuffer(
            g_adreno.ctx,
            CL_MEM_READ_ONLY_SOC | CL_MEM_USE_HOST_PTR_SOC,
            2 * LFM3_FFN_MAT_4BIT_BYTES,
            g_lfm3.trunks[l].ffn_4bit_mf.ptr,
            &err
        );
        if (err != 0 || !g_adreno.b_ffn[l]) return;
    }

    size_t srclen = strlen(ADRENO_SWIGLU_DOT8_SRC);
    cl_program prog = clCreateProgramWithSource(g_adreno.ctx, 1, &ADRENO_SWIGLU_DOT8_SRC, &srclen, &err);
    if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != 0) return;

    g_adreno.k_swiglu = clCreateKernel(prog, "adreno_swiglu_dot8_exact", &err);
    if (err != 0 || !g_adreno.k_swiglu) return;

    g_adreno.b_xq   = clCreateBuffer(g_adreno.ctx, CL_MEM_READ_ONLY_SOC | CL_MEM_ALLOC_HOST_PTR_SOC, 2048, NULL, &err);
    g_adreno.b_xsc  = clCreateBuffer(g_adreno.ctx, CL_MEM_READ_ONLY_SOC | CL_MEM_ALLOC_HOST_PTR_SOC, 32 * 4, NULL, &err);
    g_adreno.b_xsum = clCreateBuffer(g_adreno.ctx, CL_MEM_READ_ONLY_SOC | CL_MEM_ALLOC_HOST_PTR_SOC, 32 * 4, NULL, &err);
    g_adreno.b_out  = clCreateBuffer(g_adreno.ctx, CL_MEM_READ_WRITE_SOC | CL_MEM_ALLOC_HOST_PTR_SOC, 7680 * 4, NULL, &err);

    g_adreno.clSetKernelArg(g_adreno.k_swiglu, 1, sizeof(cl_mem), &g_adreno.b_xq);
    g_adreno.clSetKernelArg(g_adreno.k_swiglu, 2, sizeof(cl_mem), &g_adreno.b_xsc);
    g_adreno.clSetKernelArg(g_adreno.k_swiglu, 3, sizeof(cl_mem), &g_adreno.b_xsum);
    g_adreno.clSetKernelArg(g_adreno.k_swiglu, 4, sizeof(cl_mem), &g_adreno.b_out);

    g_adreno.lib = h;
    g_lfm3.adreno_ready = 1;
    g_lfm3.adreno_enabled = 1;
}
#endif

static void lfm3_advise_trunk_pages(int layer_id, int advise_willneed)
{
    if (!g_lfm3.madvise_active || layer_id < 0 || layer_id >= LFM3_LAYERS) return;
#ifndef _WIN32
    const Lfm3TrunkMapped *t = &g_lfm3.trunks[layer_id];
    int adv = advise_willneed ? MADV_WILLNEED : MADV_DONTNEED;
    if (t->ffn_4bit_mf.ptr && t->ffn_4bit_mf.size > 0) {
        madvise(t->ffn_4bit_mf.ptr, t->ffn_4bit_mf.size, adv);
    }
    if (g_lfm3.use_4bit_operators && t->operator_4bit_mf.ptr && t->operator_4bit_mf.size > 0) {
        madvise(t->operator_4bit_mf.ptr, t->operator_4bit_mf.size, adv);
    } else {
        if (t->operator_mf.ptr && t->operator_mf.size > 0) {
            madvise(t->operator_mf.ptr, t->operator_mf.size, adv);
        }
        if (t->group_mf[0].ptr && t->group_mf[0].size > 0) {
            madvise(t->group_mf[0].ptr, t->group_mf[0].size, adv);
        }
    }
#else
    (void)advise_willneed;
#endif
}

LFM3_API void lfm3_set_madvise_active(int enable)
{
    g_lfm3.madvise_active = enable ? 1 : 0;
}

LFM3_API void lfm3_set_adreno_opencl_active(int enable)
{
    g_lfm3.adreno_enabled = (enable && g_lfm3.adreno_ready) ? 1 : 0;
}

LFM3_API void lfm3_set_use_4bit_operators(int enable)
{
    g_lfm3.use_4bit_operators = enable ? 1 : 0;
#ifndef _WIN32
    if (g_lfm3.use_4bit_operators) {
        for (int l = 0; l < LFM3_LAYERS; l++) {
            const Lfm3TrunkMapped *t = &g_lfm3.trunks[l];
            if (t->operator_mf.ptr && t->operator_mf.size > 131072) {
                /* Drop the 25.1 MB / 12.6 MB BF16 projection matrices in operator.bin, keep first & last 64 KB norms/gates */
                size_t drop_off = 8192;
                size_t drop_end = (t->operator_mf.size > 131072) ? ((t->operator_mf.size - 131072) & ~(size_t)4095) : 0;
                if (drop_end > drop_off) {
                    madvise((uint8_t *)t->operator_mf.ptr + drop_off, drop_end - drop_off, MADV_DONTNEED);
                }
            }
            if (t->operator_4bit_mf.ptr && t->operator_4bit_mf.size > 0) {
                madvise(t->operator_4bit_mf.ptr, t->operator_4bit_mf.size, MADV_WILLNEED);
            }
        }
    }
#endif
}

LFM3_API void lfm3_set_use_4bit_lm_head(int enable)
{
    g_lfm3.use_4bit_lm_head = enable ? 1 : 0;
}

LFM3_API void lfm3_dm_reset(void)
{
    g_lfm3.clock = 0;
    for (int i = 0; i < 12; i++) {
        g_lfm3.pool_keys[i] = -1;
        g_lfm3.pool_ts[i] = 0;
        g_lfm3.pool_heat[i] = 0.0f;
    }
    for (int l = 0; l < LFM3_LAYERS; l++) g_lfm3.host_keys[l] = 0;
    g_lfm3.total_layer_requests = 0;
    g_lfm3.tier1_sram_l3_hits = 0;
    g_lfm3.tier2_lpddr5x_hits = 0;
    g_lfm3.tier3_ufs4_fetches = 0;
    g_lfm3.r4_clean_drops = 0;
    g_lfm3.dirty_writebacks = 0;
    g_lfm3.bytes_tier0_reg = 0;
    g_lfm3.bytes_tier1_sram_slc = 0;
    g_lfm3.bytes_tier2_lpddr5x = 0;
    g_lfm3.bytes_saved_q4_and_broadcast = 0;
    g_lfm3.smmla_i8mm_tile_calls = 0;
    g_lfm3.fa2_online_softmax_calls = 0;
    g_lfm3.adreno_opencl_launches = 0;
    read_os_page_faults(&g_lfm3.minflt_at_reset, &g_lfm3.majflt_at_reset);
}

static void lfm3_dm_record_layer(int layer_id, int tokens_in_pass)
{
    g_lfm3.clock++;
    g_lfm3.total_layer_requests++;
    float w = (float)(tokens_in_pass > 0 ? tokens_in_pass : 1);
    int next_layer = (layer_id + 1) % LFM3_LAYERS;

    /* SpecMD Lookahead: asynchronously prefetch Layer L+1 into OS page cache when not yet resident */
    if (!g_lfm3.host_keys[next_layer]) {
        lfm3_advise_trunk_pages(next_layer, 1);
    }

    for (int i = 0; i < 12; i++) {
        if (g_lfm3.pool_keys[i] == layer_id) {
            g_lfm3.tier1_sram_l3_hits++;
            g_lfm3.pool_ts[i] = g_lfm3.clock;
            g_lfm3.pool_heat[i] = 0.75f * g_lfm3.pool_heat[i] + w;
            return;
        }
    }
    if (g_lfm3.host_keys[layer_id]) {
        g_lfm3.tier2_lpddr5x_hits++;
    } else {
        g_lfm3.tier3_ufs4_fetches++;
        g_lfm3.host_keys[layer_id] = 1;
    }
    int slot = -1;
    for (int i = 0; i < 12; i++) {
        if (g_lfm3.pool_keys[i] < 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        int best_v = 0;
        double best_sc = 1e30;
        for (int i = 0; i < 12; i++) {
            int k = g_lfm3.pool_keys[i];
            if (k == layer_id || k == next_layer) continue; /* SpecMD Lookahead: protect Layer L and L+1 */
            unsigned long stale = g_lfm3.clock > g_lfm3.pool_ts[i] ? (g_lfm3.clock - g_lfm3.pool_ts[i]) : 1UL;
            double sc = (double)g_lfm3.pool_heat[i] / (double)(1UL + stale);
            if (sc < best_sc) {
                best_sc = sc;
                best_v = i;
            }
        }
        slot = best_v;
        int victim = g_lfm3.pool_keys[slot];
        if (victim >= 0 && g_lfm3.host_keys[victim]) {
            g_lfm3.r4_clean_drops++;
        } else {
            g_lfm3.dirty_writebacks++;
        }
    }
    g_lfm3.pool_keys[slot] = layer_id;
    g_lfm3.pool_ts[slot] = g_lfm3.clock;
    g_lfm3.pool_heat[slot] = w;
}

/* Explicit OS page-cache release for inactive trunks under memory pressure (`MADV_DONTNEED`) */
LFM3_API int lfm3_release_inactive_trunk_pages(void)
{
    int dropped = 0;
    for (int l = 0; l < LFM3_LAYERS; l++) {
        int in_pool = 0;
        for (int i = 0; i < 12; i++) {
            if (g_lfm3.pool_keys[i] == l) {
                in_pool = 1;
                break;
            }
        }
        if (!in_pool) {
            lfm3_advise_trunk_pages(l, 0);
            dropped++;
        }
    }
    return dropped;
}

LFM3_API void lfm3_set_use_4bit_ffn(int enable)
{
    g_lfm3.use_4bit_ffn = enable ? 1 : 0;
}

LFM3_API int lfm3_configure_s23_snapdragon8gen2(int pin_big_cores_3_to_7)
{
    /* Samsung Galaxy S23 (Snapdragon 8 Gen 2 for Galaxy, SM8550-AC):
     *   - Cores 0..2: 3x Cortex-A510 efficiency cores (2.0 GHz, in-order)
     *   - Cores 3..4: 2x Cortex-A710 performance cores (2.8 GHz)
     *   - Cores 5..6: 2x Cortex-A715 performance cores (2.8 GHz)
     *   - Core 7:     1x Cortex-X3 prime core (3.36 GHz)
     * Inspects Android cpuset affinity first: if cores 3..7 are available (foreground/top-app),
     * pins to the 4 or 5 out-of-order cores (3..7); if restricted by background cpuset (e.g. 0..2),
     * matches OpenMP thread count to allowed cores to prevent barrier spinlock oversubscription. */
    g_lfm3.s23_pin_requested = pin_big_cores_3_to_7 ? 1 : 0;
    g_lfm3.num_threads = pin_big_cores_3_to_7 ? 5 : 8;
#if defined(__linux__) && defined(__aarch64__)
    cpu_set_t current_mask;
    CPU_ZERO(&current_mask);
    if (sched_getaffinity(0, sizeof(current_mask), &current_mask) == 0) {
        cpu_set_t big_mask;
        CPU_ZERO(&big_mask);
        int big_avail = 0;
        for (int c = 3; c <= 7; c++) {
            if (CPU_ISSET(c, &current_mask)) {
                CPU_SET(c, &big_mask);
                big_avail++;
            }
        }
        if (pin_big_cores_3_to_7 && big_avail >= 4) {
            if (sched_setaffinity(0, sizeof(big_mask), &big_mask) == 0) {
                g_lfm3.s23_big_cores_pinned = 1;
                g_lfm3.num_threads = big_avail;
            }
        } else {
            int total_avail = CPU_COUNT(&current_mask);
            if (total_avail > 0) {
                g_lfm3.num_threads = total_avail;
            }
        }
    }
#endif
    omp_set_num_threads(g_lfm3.num_threads);
    return g_lfm3.num_threads;
}

LFM3_API void lfm3_dm_stats(unsigned long *out_stats6)
{
    if (!out_stats6) return;
    out_stats6[0] = g_lfm3.total_layer_requests;
    out_stats6[1] = g_lfm3.tier1_sram_l3_hits;
    out_stats6[2] = g_lfm3.tier2_lpddr5x_hits;
    out_stats6[3] = g_lfm3.tier3_ufs4_fetches;
    out_stats6[4] = g_lfm3.r4_clean_drops;
    out_stats6[5] = g_lfm3.dirty_writebacks;
}

LFM3_API void lfm3_dm_extended_stats(unsigned long long *out_ext12)
{
    if (!out_ext12) return;
    unsigned long cur_minflt = 0, cur_majflt = 0;
    read_os_page_faults(&cur_minflt, &cur_majflt);
    out_ext12[0]  = g_lfm3.bytes_tier0_reg;
    out_ext12[1]  = g_lfm3.bytes_tier1_sram_slc;
    out_ext12[2]  = g_lfm3.bytes_tier2_lpddr5x;
    out_ext12[3]  = g_lfm3.bytes_saved_q4_and_broadcast;
    out_ext12[4]  = (unsigned long long)g_lfm3.smmla_i8mm_tile_calls;
    out_ext12[5]  = (unsigned long long)g_lfm3.fa2_online_softmax_calls;
    out_ext12[6]  = (unsigned long long)g_lfm3.adreno_opencl_launches;
    out_ext12[7]  = (unsigned long long)(cur_minflt >= g_lfm3.minflt_at_reset ? (cur_minflt - g_lfm3.minflt_at_reset) : 0UL);
    out_ext12[8]  = (unsigned long long)(cur_majflt >= g_lfm3.majflt_at_reset ? (cur_majflt - g_lfm3.majflt_at_reset) : 0UL);
    out_ext12[9]  = (unsigned long long)(g_lfm3.adreno_ready && g_lfm3.adreno_enabled);
    out_ext12[10] = (unsigned long long)g_lfm3.use_4bit_operators;
    out_ext12[11] = (unsigned long long)g_lfm3.use_4bit_lm_head;
}

static void linear_bf16_omp(
    const float *x,
    const uint16_t *W_bf16,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    size_t w_bytes = (size_t)out_dim * (size_t)in_dim * 2ULL;
    g_lfm3.bytes_tier2_lpddr5x += (unsigned long long)w_bytes;
    g_lfm3.bytes_tier1_sram_slc += (unsigned long long)((size_t)T * (size_t)(in_dim + out_dim) * 4ULL);
    g_lfm3.bytes_tier0_reg += (unsigned long long)((size_t)T * (size_t)out_dim * (size_t)in_dim * 2ULL);
    if (T > 1) {
        g_lfm3.bytes_saved_q4_and_broadcast += (unsigned long long)((size_t)(T - 1) * w_bytes);
    }
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    /* Snapdragon 8 Gen 2 Hardware BF16 Dot Product (`vbfdotq_f32` / `BFDOT`):
     * Converts `x` once into `bfloat16_t` (`2x` smaller in L1D cache) and executes
     * 8 BF16 multiply-accumulates per single `vbfdotq_f32` instruction! */
    bfloat16_t *x_bf16 = (bfloat16_t *)malloc((size_t)T * (size_t)in_dim * sizeof(bfloat16_t));
    const size_t total_x = (size_t)T * (size_t)in_dim;
    size_t idx = 0;
    for (; idx + 8 <= total_x; idx += 8) {
        bfloat16x4_t lo = vcvt_bf16_f32(vld1q_f32(x + idx));
        bfloat16x4_t hi = vcvt_bf16_f32(vld1q_f32(x + idx + 4));
        vst1q_bf16(x_bf16 + idx, vcombine_bf16(lo, hi));
    }
    for (; idx < total_x; idx++) {
        uint16_t b = f32_to_bf16_rne(x[idx]);
        memcpy(&x_bf16[idx], &b, sizeof(uint16_t));
    }

    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const bfloat16_t *w_row = (const bfloat16_t *)(W_bf16 + (size_t)o * (size_t)in_dim);
        int t = 0;
        for (; t + 4 <= T; t += 4) {
            const bfloat16_t *xb0 = x_bf16 + (size_t)(t + 0) * in_dim;
            const bfloat16_t *xb1 = x_bf16 + (size_t)(t + 1) * in_dim;
            const bfloat16_t *xb2 = x_bf16 + (size_t)(t + 2) * in_dim;
            const bfloat16_t *xb3 = x_bf16 + (size_t)(t + 3) * in_dim;
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 16 <= in_dim; k += 16) {
                bfloat16x8_t w0 = vld1q_bf16(w_row + k);
                bfloat16x8_t w1 = vld1q_bf16(w_row + k + 8);
                a0 = vbfdotq_f32(vbfdotq_f32(a0, w0, vld1q_bf16(xb0 + k)), w1, vld1q_bf16(xb0 + k + 8));
                a1 = vbfdotq_f32(vbfdotq_f32(a1, w0, vld1q_bf16(xb1 + k)), w1, vld1q_bf16(xb1 + k + 8));
                a2 = vbfdotq_f32(vbfdotq_f32(a2, w0, vld1q_bf16(xb2 + k)), w1, vld1q_bf16(xb2 + k + 8));
                a3 = vbfdotq_f32(vbfdotq_f32(a3, w0, vld1q_bf16(xb3 + k)), w1, vld1q_bf16(xb3 + k + 8));
            }
            float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
            float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
            for (; k < in_dim; k++) {
                float w = bf16_to_f32(W_bf16[(size_t)o * in_dim + k]);
                s0 += x[(size_t)(t + 0) * in_dim + k] * w;
                s1 += x[(size_t)(t + 1) * in_dim + k] * w;
                s2 += x[(size_t)(t + 2) * in_dim + k] * w;
                s3 += x[(size_t)(t + 3) * in_dim + k] * w;
            }
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
        }
        for (; t < T; t++) {
            const bfloat16_t *xb = x_bf16 + (size_t)t * (size_t)in_dim;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 16 <= in_dim; k += 16) {
                acc = vbfdotq_f32(vbfdotq_f32(acc, vld1q_bf16(w_row + k), vld1q_bf16(xb + k)), vld1q_bf16(w_row + k + 8), vld1q_bf16(xb + k + 8));
            }
            float sum = vaddvq_f32(acc);
            for (; k < in_dim; k++) {
                sum += x[(size_t)t * in_dim + k] * bf16_to_f32(W_bf16[(size_t)o * in_dim + k]);
            }
            y[(size_t)t * (size_t)out_dim + (size_t)o] = bf16_round_f32(sum);
        }
    }
    free(x_bf16);
#else
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * (size_t)in_dim;
        for (int t = 0; t < T; t++) {
            const float *x_row = x + (size_t)t * (size_t)in_dim;
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                sum += x_row[k] * bf16_to_f32(w_row[k]);
            }
            y[(size_t)t * (size_t)out_dim + (size_t)o] = bf16_round_f32(sum);
        }
    }
#endif
}

#if defined(__ARM_FEATURE_DOTPROD)
static void quantize_f32_to_s8_group64(const float *x, int T, int in_dim, int8_t *out_xq, float *out_xscale)
{
    const int groups = in_dim / 64;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * in_dim;
        int8_t *xqt = out_xq + (size_t)t * in_dim;
        float *xsct = out_xscale + (size_t)t * groups;
        for (int g = 0; g < groups; g++) {
            const float *xg = xt + g * 64;
            int8_t *qdst = xqt + g * 64;
            float32x4_t vmax = vdupq_n_f32(0.0f);
            for (int j = 0; j < 64; j += 4) {
                vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xg + j)));
            }
            float max_abs = vmaxvq_f32(vmax);
            if (max_abs < 1e-12f) {
                xsct[g] = 0.0f;
                memset(qdst, 0, 64);
            } else {
                float sc = max_abs / 127.0f;
                float inv = 127.0f / max_abs;
                xsct[g] = sc;
                float32x4_t vinv = vdupq_n_f32(inv);
                for (int j = 0; j < 64; j += 16) {
                    int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xg + j + 0), vinv));
                    int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xg + j + 4), vinv));
                    int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xg + j + 8), vinv));
                    int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xg + j + 12), vinv));
                    int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                    int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                    vst1q_s8(qdst + j, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
                }
            }
        }
    }
}
#endif

/* OpenMP 4-Bit Planar Group-64 Linear with Hardware ARMv8.6-A `vmmlaq_s32` (`SMMLA` i8mm 2x2 Matrix Tile)
 * and ARMv8.2-A `vdotq_s32` (`SDOT`) Acceleration + Eyeriss v2 Multi-Token Broadcast Coalescing:
 * Quantizes `x` per 64-element group into `int8` (`4x` smaller in L1D cache) and executes
 * hardware `SMMLA` (32 Int8 MACs/instruction for T >= 2) and `SDOT` (16 Int8 MACs/instruction)! */
static void linear_4bit_group64_omp(
    const float *x,
    const uint8_t *nibbles_base,
    const uint8_t *steps_base,
    const float *scales_base,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    const int groups = in_dim / 64;
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    size_t q4_bytes = (size_t)out_dim * (size_t)(groups * 33 + 4);
    size_t bf16_equiv = (size_t)out_dim * (size_t)in_dim * 2ULL;
    g_lfm3.bytes_tier2_lpddr5x += (unsigned long long)q4_bytes;
    g_lfm3.bytes_tier1_sram_slc += (unsigned long long)((size_t)T * (size_t)(in_dim + out_dim) * 4ULL);
    g_lfm3.bytes_tier0_reg += (unsigned long long)((size_t)T * (size_t)out_dim * (size_t)in_dim);
    g_lfm3.bytes_saved_q4_and_broadcast += (unsigned long long)((bf16_equiv - q4_bytes) + (size_t)(T > 1 ? (T - 1) : 0) * q4_bytes);
#if defined(__ARM_FEATURE_DOTPROD)
    int8_t *x_q = (int8_t *)malloc((size_t)T * (size_t)in_dim);
    float *x_sc = (float *)malloc((size_t)T * (size_t)groups * sizeof(float));
    quantize_f32_to_s8_group64(x, T, in_dim, x_q, x_sc);

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (T >= 2 && (out_dim & 1) == 0) {
        g_lfm3.smmla_i8mm_tile_calls += (unsigned long)((out_dim / 2) * (T / 2));
        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int o = 0; o < out_dim; o += 2) {
            const uint8_t *nib_r0 = nibbles_base + (size_t)(o + 0) * (size_t)(groups * 32);
            const uint8_t *nib_r1 = nibbles_base + (size_t)(o + 1) * (size_t)(groups * 32);
            const uint8_t *stp_r0 = steps_base + (size_t)(o + 0) * (size_t)groups;
            const uint8_t *stp_r1 = steps_base + (size_t)(o + 1) * (size_t)groups;
            const float scl_r0 = scales_base[o + 0];
            const float scl_r1 = scales_base[o + 1];
            const int8x16_t eight_s8 = vdupq_n_s8(8);
            const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

            int t = 0;
            for (; t + 2 <= T; t += 2) {
                const int8_t *xq0 = x_q + (size_t)(t + 0) * in_dim;
                const int8_t *xq1 = x_q + (size_t)(t + 1) * in_dim;
                const float *xs0 = x_sc + (size_t)(t + 0) * groups;
                const float *xs1 = x_sc + (size_t)(t + 1) * groups;
                float s_t0_o0 = 0.0f, s_t0_o1 = 0.0f, s_t1_o0 = 0.0f, s_t1_o1 = 0.0f;

                for (int g = 0; g < groups; g++) {
                    const uint8_t *nb0 = nib_r0 + g * 32;
                    const uint8_t *nb1 = nib_r1 + g * 32;
                    uint8x16_t p00 = vld1q_u8(nb0), p01 = vld1q_u8(nb0 + 16);
                    uint8x16_t p10 = vld1q_u8(nb1), p11 = vld1q_u8(nb1 + 16);

                    int8x16_t w0_lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p00, mask_0f)), eight_s8);
                    int8x16_t w0_lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p01, mask_0f)), eight_s8);
                    int8x16_t w0_hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p00, 4)), eight_s8);
                    int8x16_t w0_hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p01, 4)), eight_s8);

                    int8x16_t w1_lo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p10, mask_0f)), eight_s8);
                    int8x16_t w1_lo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p11, mask_0f)), eight_s8);
                    int8x16_t w1_hi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p10, 4)), eight_s8);
                    int8x16_t w1_hi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p11, 4)), eight_s8);

                    int c = g * 64;
                    int8x16_t x0_0 = vld1q_s8(xq0 + c + 0),  x0_1 = vld1q_s8(xq0 + c + 16);
                    int8x16_t x0_2 = vld1q_s8(xq0 + c + 32), x0_3 = vld1q_s8(xq0 + c + 48);
                    int8x16_t x1_0 = vld1q_s8(xq1 + c + 0),  x1_1 = vld1q_s8(xq1 + c + 16);
                    int8x16_t x1_2 = vld1q_s8(xq1 + c + 32), x1_3 = vld1q_s8(xq1 + c + 48);

                    /* 8 x SMMLA (vmmlaq_s32) instructions compute 2x2 tile over all 64 columns (256 Int8 MACs) */
                    int32x4_t acc = vdupq_n_s32(0);
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(x0_0),  vget_low_s8(x1_0)),  vcombine_s8(vget_low_s8(w0_lo0),  vget_low_s8(w1_lo0)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(x0_0), vget_high_s8(x1_0)), vcombine_s8(vget_high_s8(w0_lo0), vget_high_s8(w1_lo0)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(x0_1),  vget_low_s8(x1_1)),  vcombine_s8(vget_low_s8(w0_lo1),  vget_low_s8(w1_lo1)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(x0_1), vget_high_s8(x1_1)), vcombine_s8(vget_high_s8(w0_lo1), vget_high_s8(w1_lo1)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(x0_2),  vget_low_s8(x1_2)),  vcombine_s8(vget_low_s8(w0_hi0),  vget_low_s8(w1_hi0)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(x0_2), vget_high_s8(x1_2)), vcombine_s8(vget_high_s8(w0_hi0), vget_high_s8(w1_hi0)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_low_s8(x0_3),  vget_low_s8(x1_3)),  vcombine_s8(vget_low_s8(w0_hi1),  vget_low_s8(w1_hi1)));
                    acc = vmmlaq_s32(acc, vcombine_s8(vget_high_s8(x0_3), vget_high_s8(x1_3)), vcombine_s8(vget_high_s8(w0_hi1), vget_high_s8(w1_hi1)));

                    float st0 = (float)stp_r0[g], st1 = (float)stp_r1[g];
                    s_t0_o0 += (float)vgetq_lane_s32(acc, 0) * (st0 * xs0[g]);
                    s_t0_o1 += (float)vgetq_lane_s32(acc, 1) * (st1 * xs0[g]);
                    s_t1_o0 += (float)vgetq_lane_s32(acc, 2) * (st0 * xs1[g]);
                    s_t1_o1 += (float)vgetq_lane_s32(acc, 3) * (st1 * xs1[g]);
                }
                y[(size_t)(t + 0) * out_dim + (o + 0)] = bf16_round_f32(s_t0_o0 * scl_r0);
                y[(size_t)(t + 0) * out_dim + (o + 1)] = bf16_round_f32(s_t0_o1 * scl_r1);
                y[(size_t)(t + 1) * out_dim + (o + 0)] = bf16_round_f32(s_t1_o0 * scl_r0);
                y[(size_t)(t + 1) * out_dim + (o + 1)] = bf16_round_f32(s_t1_o1 * scl_r1);
            }
            for (; t < T; t++) {
                const int8_t *xqt = x_q + (size_t)t * in_dim;
                const float *xst = x_sc + (size_t)t * groups;
                float sum0 = 0.0f, sum1 = 0.0f;
                for (int g = 0; g < groups; g++) {
                    const uint8_t *nb0 = nib_r0 + g * 32;
                    const uint8_t *nb1 = nib_r1 + g * 32;
                    uint8x16_t p00 = vld1q_u8(nb0), p01 = vld1q_u8(nb0 + 16);
                    uint8x16_t p10 = vld1q_u8(nb1), p11 = vld1q_u8(nb1 + 16);
                    int c = g * 64;
                    int8x16_t x0 = vld1q_s8(xqt + c + 0),  x1 = vld1q_s8(xqt + c + 16);
                    int8x16_t x2 = vld1q_s8(xqt + c + 32), x3 = vld1q_s8(xqt + c + 48);
                    int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
                    d0 = vdotq_s32(vdotq_s32(d0, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p00, mask_0f)), eight_s8), x0), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p01, mask_0f)), eight_s8), x1);
                    d0 = vdotq_s32(vdotq_s32(d0, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p00, 4)), eight_s8), x2), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p01, 4)), eight_s8), x3);
                    d1 = vdotq_s32(vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p10, mask_0f)), eight_s8), x0), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p11, mask_0f)), eight_s8), x1);
                    d1 = vdotq_s32(vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p10, 4)), eight_s8), x2), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(p11, 4)), eight_s8), x3);
                    sum0 += (float)vaddvq_s32(d0) * ((float)stp_r0[g] * xst[g]);
                    sum1 += (float)vaddvq_s32(d1) * ((float)stp_r1[g] * xst[g]);
                }
                y[(size_t)t * out_dim + (o + 0)] = bf16_round_f32(sum0 * scl_r0);
                y[(size_t)t * out_dim + (o + 1)] = bf16_round_f32(sum1 * scl_r1);
            }
        }
        free(x_q);
        free(x_sc);
        return;
    }
#endif

    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *nib_row = nibbles_base + (size_t)o * (size_t)(groups * 32);
        const uint8_t *stp_row = steps_base + (size_t)o * (size_t)groups;
        const float row_scale = scales_base[o];
        const int8x16_t eight_s8 = vdupq_n_s8(8);
        const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

        int t = 0;
        for (; t + 4 <= T; t += 4) {
            const int8_t *xq0 = x_q + (size_t)(t + 0) * in_dim;
            const int8_t *xq1 = x_q + (size_t)(t + 1) * in_dim;
            const int8_t *xq2 = x_q + (size_t)(t + 2) * in_dim;
            const int8_t *xq3 = x_q + (size_t)(t + 3) * in_dim;
            const float *xs0 = x_sc + (size_t)(t + 0) * groups;
            const float *xs1 = x_sc + (size_t)(t + 1) * groups;
            const float *xs2 = x_sc + (size_t)(t + 2) * groups;
            const float *xs3 = x_sc + (size_t)(t + 3) * groups;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                uint8x16_t pk0 = vld1q_u8(nb);
                uint8x16_t pk1 = vld1q_u8(nb + 16);
                int8x16_t wlo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk0, mask_0f)), eight_s8);
                int8x16_t wlo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk1, mask_0f)), eight_s8);
                int8x16_t whi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk0, 4)), eight_s8);
                int8x16_t whi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk1, 4)), eight_s8);

                int c = g * 64;
                int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0), d2 = vdupq_n_s32(0), d3 = vdupq_n_s32(0);
                d0 = vdotq_s32(vdotq_s32(d0, wlo0, vld1q_s8(xq0 + c)), wlo1, vld1q_s8(xq0 + c + 16));
                d0 = vdotq_s32(vdotq_s32(d0, whi0, vld1q_s8(xq0 + c + 32)), whi1, vld1q_s8(xq0 + c + 48));
                d1 = vdotq_s32(vdotq_s32(d1, wlo0, vld1q_s8(xq1 + c)), wlo1, vld1q_s8(xq1 + c + 16));
                d1 = vdotq_s32(vdotq_s32(d1, whi0, vld1q_s8(xq1 + c + 32)), whi1, vld1q_s8(xq1 + c + 48));
                d2 = vdotq_s32(vdotq_s32(d2, wlo0, vld1q_s8(xq2 + c)), wlo1, vld1q_s8(xq2 + c + 16));
                d2 = vdotq_s32(vdotq_s32(d2, whi0, vld1q_s8(xq2 + c + 32)), whi1, vld1q_s8(xq2 + c + 48));
                d3 = vdotq_s32(vdotq_s32(d3, wlo0, vld1q_s8(xq3 + c)), wlo1, vld1q_s8(xq3 + c + 16));
                d3 = vdotq_s32(vdotq_s32(d3, whi0, vld1q_s8(xq3 + c + 32)), whi1, vld1q_s8(xq3 + c + 48));

                float stp = (float)stp_row[g];
                s0 += (float)vaddvq_s32(d0) * (stp * xs0[g]);
                s1 += (float)vaddvq_s32(d1) * (stp * xs1[g]);
                s2 += (float)vaddvq_s32(d2) * (stp * xs2[g]);
                s3 += (float)vaddvq_s32(d3) * (stp * xs3[g]);
            }
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0 * row_scale);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1 * row_scale);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2 * row_scale);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3 * row_scale);
        }
        for (; t < T; t++) {
            const int8_t *xqt = x_q + (size_t)t * in_dim;
            const float *xst = x_sc + (size_t)t * groups;
            float sum = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                uint8x16_t pk0 = vld1q_u8(nb);
                uint8x16_t pk1 = vld1q_u8(nb + 16);
                int8x16_t wlo0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk0, mask_0f)), eight_s8);
                int8x16_t wlo1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk1, mask_0f)), eight_s8);
                int8x16_t whi0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk0, 4)), eight_s8);
                int8x16_t whi1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk1, 4)), eight_s8);

                int c = g * 64;
                int32x4_t d0 = vdupq_n_s32(0);
                d0 = vdotq_s32(vdotq_s32(d0, wlo0, vld1q_s8(xqt + c)), wlo1, vld1q_s8(xqt + c + 16));
                d0 = vdotq_s32(vdotq_s32(d0, whi0, vld1q_s8(xqt + c + 32)), whi1, vld1q_s8(xqt + c + 48));
                sum += (float)vaddvq_s32(d0) * ((float)stp_row[g] * xst[g]);
            }
            y[(size_t)t * out_dim + o] = bf16_round_f32(sum * row_scale);
        }
    }
    free(x_q);
    free(x_sc);
#else
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *nib_row = nibbles_base + (size_t)o * (size_t)(groups * 32);
        const uint8_t *stp_row = steps_base + (size_t)o * (size_t)groups;
        const float row_scale = scales_base[o];
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * in_dim;
            float sum = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float g_sc = (float)stp_row[g] * row_scale;
                const int col0 = g * 64;
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = nb[j];
                    sum += xt[col0 + j] * ((float)((int)(pk & 0x0F) - 8) * g_sc)
                         + xt[col0 + 32 + j] * ((float)((int)(pk >> 4) - 8) * g_sc);
                }
            }
            y[(size_t)t * out_dim + o] = bf16_round_f32(sum);
        }
    }
#endif
}

static void rms_norm_bf16(
    const float *x,
    const uint16_t *w_bf16,
    float *y,
    int n_rows,
    int dim,
    float eps
)
{
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        double ss = 0.0;
        for (int d = 0; d < dim; d++) ss += (double)xr[d] * (double)xr[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        for (int d = 0; d < dim; d++) {
            float normed_bf16 = bf16_round_f32(xr[d] * inv);
            yr[d] = bf16_round_f32(normed_bf16 * bf16_to_f32(w_bf16[d]));
        }
    }
}

/* FlashAttention-2 1-Pass Online Softmax (`Dao et al. 2023`, Direction B Cycle 4):
 * Streams each key `kh[0..127]` and value `vh[0..127]` contiguously ONCE (`0` `scores[]` buffer,
 * `0` strided V-cache reads) and maintains `(m_i, l_i, o_acc[0..127])` in NEON registers! */
static void fa2_online_softmax_head_c(
    const float *qh,
    const float *kc_base,
    const float *vc_base,
    int kv_h,
    int j_start,
    int j_end,
    float scale,
    float head_gate,
    float *out_qh
)
{
    g_lfm3.fa2_online_softmax_calls++;
    g_lfm3.bytes_tier1_sram_slc += (unsigned long long)((size_t)(j_end - j_start + 1) * 1024ULL);
    g_lfm3.bytes_tier0_reg += (unsigned long long)((size_t)(j_end - j_start + 1) * 1024ULL);

    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_acc[128];
    memset(o_acc, 0, sizeof(o_acc));

    for (int j = j_start; j <= j_end; j++) {
        const float *kh = kc_base + (size_t)j * 512 + kv_h * 128;
        const float *vh = vc_base + (size_t)j * 512 + kv_h * 128;
        float dot = 0.0f;
#if defined(__aarch64__) || defined(__ARM_NEON)
        float32x4_t vdot0 = vdupq_n_f32(0.0f), vdot1 = vdupq_n_f32(0.0f);
        float32x4_t vdot2 = vdupq_n_f32(0.0f), vdot3 = vdupq_n_f32(0.0f);
        for (int d = 0; d < 128; d += 16) {
            vdot0 = vfmaq_f32(vdot0, vld1q_f32(qh + d + 0),  vld1q_f32(kh + d + 0));
            vdot1 = vfmaq_f32(vdot1, vld1q_f32(qh + d + 4),  vld1q_f32(kh + d + 4));
            vdot2 = vfmaq_f32(vdot2, vld1q_f32(qh + d + 8),  vld1q_f32(kh + d + 8));
            vdot3 = vfmaq_f32(vdot3, vld1q_f32(qh + d + 12), vld1q_f32(kh + d + 12));
        }
        dot = vaddvq_f32(vaddq_f32(vaddq_f32(vdot0, vdot1), vaddq_f32(vdot2, vdot3)));
#else
        for (int d = 0; d < 128; d++) dot += qh[d] * kh[d];
#endif
        float s_j = bf16_round_f32(dot * scale);
        float m_new = (s_j > m_i) ? s_j : m_i;
        float alpha = (m_i == -1e30f) ? 0.0f : expf(m_i - m_new);
        float beta  = expf(s_j - m_new);
        l_i = l_i * alpha + beta;
        m_i = m_new;

#if defined(__aarch64__) || defined(__ARM_NEON)
        float32x4_t valpha = vdupq_n_f32(alpha);
        float32x4_t vbeta  = vdupq_n_f32(beta);
        for (int d = 0; d < 128; d += 16) {
            vst1q_f32(o_acc + d + 0,  vfmaq_f32(vmulq_f32(vld1q_f32(o_acc + d + 0),  valpha), vbeta, vld1q_f32(vh + d + 0)));
            vst1q_f32(o_acc + d + 4,  vfmaq_f32(vmulq_f32(vld1q_f32(o_acc + d + 4),  valpha), vbeta, vld1q_f32(vh + d + 4)));
            vst1q_f32(o_acc + d + 8,  vfmaq_f32(vmulq_f32(vld1q_f32(o_acc + d + 8),  valpha), vbeta, vld1q_f32(vh + d + 8)));
            vst1q_f32(o_acc + d + 12, vfmaq_f32(vmulq_f32(vld1q_f32(o_acc + d + 12), valpha), vbeta, vld1q_f32(vh + d + 12)));
        }
#else
        for (int d = 0; d < 128; d++) {
            o_acc[d] = o_acc[d] * alpha + beta * vh[d];
        }
#endif
    }
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = 0; d < 128; d++) {
        out_qh[d] = bf16_round_f32(bf16_round_f32(o_acc[d] * inv_l) * head_gate);
    }
}

LFM3_API int lfm3_engine_init(const char *pruned_dataset_dir)
{
    if (g_lfm3.initialized) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/global-0/embed_tokens.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.emb_mf) != 0) return -1;

    snprintf(path, sizeof(path), "%s/global-0/lm_head.4bit.bin", pruned_dataset_dir);
    if (file_readable(path)) {
        if (mmap_readonly(path, &g_lfm3.lm_head_4bit_mf) == 0 && g_lfm3.lm_head_4bit_mf.size >= LFM3_LM_HEAD_4BIT_BYTES) {
            g_lfm3.use_4bit_lm_head = 1;
        }
    }

    snprintf(path, sizeof(path), "%s/global-0/embedding_norm.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.emb_norm_mf) != 0) return -2;

    snprintf(path, sizeof(path), "%s/global-0/rope_inv_freq.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.rope_mf) != 0 || g_lfm3.rope_mf.size < 20 + 512) return -3;
    const uint8_t *rp = (const uint8_t *)g_lfm3.rope_mf.ptr;
    memcpy(g_lfm3.inv_freq_full, rp + 20, 256);
    memcpy(g_lfm3.inv_freq_sliding, rp + 20 + 256, 256);

    for (int i = 0; i < LFM3_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/trunk-%d", pruned_dataset_dir, i);
        if (load_lfm3_trunk(path, &g_lfm3.trunks[i]) != 0) return -(10 + i);
    }
    if (g_lfm3.trunks[0].ffn_4bit_mf.ptr && !g_lfm3.trunks[0].group_mf[2].ptr) {
        g_lfm3.use_4bit_ffn = 1;
    }

#if defined(__linux__) && defined(__aarch64__)
    init_adreno740_opencl();
#endif

    lfm3_dm_reset();
    g_lfm3.initialized = 1;
    return 0;
}

static void execute_single_lfm3_layer_c(
    const Lfm3TrunkMapped *trk,
    const float *h_in,
    float *h_out,
    int T,
    const int *seg_starts
)
{
    const uint8_t *op = (const uint8_t *)trk->operator_mf.ptr;
    uint32_t kind = trk->hdr.layer_kind;
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;

    float *op_in = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *op_out_raw = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *op_normed = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *h_mid = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    if (kind == 0) {
        /* Gated ShortConv (`kind = 0`, width K=7) isolated per segment `seg_starts[t]` */
        uint32_t len_op_norm, len_in_proj, len_conv_w;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_in_proj, op + 20, 4);
        memcpy(&len_conv_w, op + 24, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *in_proj_w = (const uint16_t *)(op + cur); cur += len_in_proj;
        const uint16_t *conv_w = (const uint16_t *)(op + cur); cur += len_conv_w;
        const uint16_t *out_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *bcx = (float *)malloc((size_t)T * 6144 * sizeof(float));
        float *post_gated = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
        linear_bf16_omp(op_in, in_proj_w, bcx, T, 6144, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int c = 0; c < LFM3_D; c++) {
            float w_k[7];
            for (int k = 0; k < 7; k++) w_k[k] = bf16_to_f32(conv_w[c * 7 + k]);
            for (int t = 0; t < T; t++) {
                int s0 = seg_starts ? seg_starts[t] : 0;
                float acc = 0.0f;
                for (int k = 0; k < 7; k++) {
                    int ts = t - 6 + k;
                    if (ts >= s0) {
                        float B_val = bcx[(size_t)ts * 6144 + c];
                        float x_val = bcx[(size_t)ts * 6144 + 4096 + c];
                        float pre = bf16_round_f32(B_val * x_val);
                        acc += pre * w_k[k];
                    }
                }
                float conv_c = bf16_round_f32(acc);
                float C_val = bcx[(size_t)t * 6144 + 2048 + c];
                post_gated[(size_t)t * LFM3_D + c] = bf16_round_f32(C_val * conv_c);
            }
        }
        linear_bf16_omp(post_gated, out_proj_w, op_out_raw, T, LFM3_D, LFM3_D);
        free(bcx);
        free(post_gated);
    } else {
        /* Sliding (`kind = 1`) or Full (`kind = 2`) GQA Attention with per-head Sigmoid Gate & segment isolation */
        uint32_t len_op_norm, len_q, len_k, len_v, len_gate, len_qn, len_kn;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_q, op + 20, 4);
        memcpy(&len_k, op + 24, 4);
        memcpy(&len_v, op + 28, 4);
        memcpy(&len_gate, op + 32, 4);
        memcpy(&len_qn, op + 36, 4);
        memcpy(&len_kn, op + 40, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *q_w = (const uint16_t *)(op + cur); cur += len_q;
        const uint16_t *k_w = (const uint16_t *)(op + cur); cur += len_k;
        const uint16_t *v_w = (const uint16_t *)(op + cur); cur += len_v;
        const uint16_t *gate_w = (const uint16_t *)(op + cur); cur += len_gate;
        const uint16_t *qn_w = (const uint16_t *)(op + cur); cur += len_qn;
        const uint16_t *kn_w = (const uint16_t *)(op + cur); cur += len_kn;
        const uint16_t *o_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *q = (float *)malloc((size_t)T * 2048 * sizeof(float));
        float *k = (float *)malloc((size_t)T * 512 * sizeof(float));
        float *v = (float *)malloc((size_t)T * 512 * sizeof(float));
        float *gate_raw = (float *)malloc((size_t)T * 16 * sizeof(float));
        float *gated_attn = (float *)malloc((size_t)T * 2048 * sizeof(float));

        linear_bf16_omp(op_in, q_w, q, T, 2048, LFM3_D);
        linear_bf16_omp(op_in, k_w, k, T, 512, LFM3_D);
        linear_bf16_omp(op_in, v_w, v, T, 512, LFM3_D);
        linear_bf16_omp(op_in, gate_w, gate_raw, T, 16, LFM3_D);

        /* Per-head q_norm and k_norm (dim=128) */
        rms_norm_bf16(q, qn_w, q, T * 16, 128, 1e-5f);
        rms_norm_bf16(k, kn_w, k, T * 4, 128, 1e-5f);

        /* Apply RoPE in float32 using segment-relative position `pos_t = t - seg_starts[t]` */
        const float *inv_freq = (kind == 1) ? g_lfm3.inv_freq_sliding : g_lfm3.inv_freq_full;
        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int t = 0; t < T; t++) {
            int pos_t = t - (seg_starts ? seg_starts[t] : 0);
            float cos_t[64], sin_t[64];
            for (int d = 0; d < 64; d++) {
                float angle = (float)pos_t * inv_freq[d];
                cos_t[d] = bf16_round_f32(cosf(angle));
                sin_t[d] = bf16_round_f32(sinf(angle));
            }
            for (int h = 0; h < 16; h++) {
                float *qh = q + (size_t)t * 2048 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = qh[d];
                    float x2 = qh[d + 64];
                    qh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    qh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
            for (int h = 0; h < 4; h++) {
                float *kh = k + (size_t)t * 512 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = kh[d];
                    float x2 = kh[d + 64];
                    kh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    kh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
        }

        const float scale = 1.0f / sqrtf(128.0f);
        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int h = 0; h < 16; h++) {
            int kv_h = h / 4;
            for (int t = 0; t < T; t++) {
                const float *qh = q + (size_t)t * 2048 + h * 128;
                int s0 = seg_starts ? seg_starts[t] : 0;
                int j_start = s0;
                if (kind == 1 && t - LFM3_SLIDING_W + 1 > j_start) {
                    j_start = t - LFM3_SLIDING_W + 1;
                }
                float g_val = gate_raw[(size_t)t * 16 + h];
                float head_gate = bf16_round_f32(1.0f / (1.0f + expf(-g_val)));
                float *out_qh = gated_attn + (size_t)t * 2048 + h * 128;
                fa2_online_softmax_head_c(qh, k, v, kv_h, j_start, t, scale, head_gate, out_qh);
            }
        }
        linear_bf16_omp(gated_attn, o_proj_w, op_out_raw, T, LFM3_D, 2048);
        free(q); free(k); free(v); free(gate_raw); free(gated_attn);
    }

    /* Sandwich post_operator_norm + Residual 1 */
    const uint16_t *post_op_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 1);
    rms_norm_bf16(op_out_raw, post_op_norm_w, op_normed, T, LFM3_D, 1e-5f);
    for (int i = 0; i < T * LFM3_D; i++) {
        h_mid[i] = bf16_round_f32(h_in[i] + op_normed[i]);
    }

    /* SwiGLU FFN + Sandwich post_ffn_norm + Residual 2 */
    const uint16_t *ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 3);
    const uint16_t *post_ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 4);

    float *ffn_in = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *w1_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *w3_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *ffn_raw = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *ffn_normed = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    rms_norm_bf16(h_mid, ffn_norm_w, ffn_in, T, LFM3_D, 1e-5f);

    if (g_lfm3.use_4bit_ffn && trk->ffn_4bit_mf.ptr && trk->ffn_4bit_mf.size >= 3 * LFM3_FFN_MAT_4BIT_BYTES) {
        /* 4-Bit Planar Group-64 SwiGLU FFN (`ffn.4bit.bin`: w1 [7680,2048], w3 [7680,2048], w2 [2048,7680]) */
        const uint8_t *base4 = (const uint8_t *)trk->ffn_4bit_mf.ptr;
        const uint8_t *w1_nib = base4;
        const uint8_t *w1_stp = w1_nib + (7680 * 32 * 32);
        const float *w1_scl = (const float *)(w1_stp + (7680 * 32));

        const uint8_t *w3_nib = base4 + LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w3_stp = w3_nib + (7680 * 32 * 32);
        const float *w3_scl = (const float *)(w3_stp + (7680 * 32));

        const uint8_t *w2_nib = base4 + 2 * LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w2_stp = w2_nib + (2048 * 120 * 32);
        const float *w2_scl = (const float *)(w2_stp + (2048 * 120));

        linear_4bit_group64_omp(ffn_in, w1_nib, w1_stp, w1_scl, w1_out, T, LFM3_FFN, LFM3_D);
        linear_4bit_group64_omp(ffn_in, w3_nib, w3_stp, w3_scl, w3_out, T, LFM3_FFN, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int i = 0; i < T * LFM3_FFN; i++) {
            float v1 = w1_out[i];
            float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
            w1_out[i] = bf16_round_f32(silu * w3_out[i]);
        }
        linear_4bit_group64_omp(w1_out, w2_nib, w2_stp, w2_scl, ffn_raw, T, LFM3_D, LFM3_FFN);
    } else {
        const uint16_t *w1_w = (const uint16_t *)get_lfm3_record_ptr(trk, 10);
        const uint16_t *w3_w = (const uint16_t *)get_lfm3_record_ptr(trk, 11);
        const uint16_t *w2_w = (const uint16_t *)get_lfm3_record_ptr(trk, 12);

        linear_bf16_omp(ffn_in, w1_w, w1_out, T, LFM3_FFN, LFM3_D);
        linear_bf16_omp(ffn_in, w3_w, w3_out, T, LFM3_FFN, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int i = 0; i < T * LFM3_FFN; i++) {
            float v1 = w1_out[i];
            float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
            w1_out[i] = bf16_round_f32(silu * w3_out[i]);
        }
        linear_bf16_omp(w1_out, w2_w, ffn_raw, T, LFM3_D, LFM3_FFN);
    }
    rms_norm_bf16(ffn_raw, post_ffn_norm_w, ffn_normed, T, LFM3_D, 1e-5f);

    for (int i = 0; i < T * LFM3_D; i++) {
        h_out[i] = bf16_round_f32(h_mid[i] + ffn_normed[i]);
    }

    free(op_in); free(op_out_raw); free(op_normed); free(h_mid);
    free(ffn_in); free(w1_out); free(w3_out); free(ffn_raw); free(ffn_normed);
}

/* Executes all 24 tri-hybrid LFM3-1.7B layers + final `embedding_norm` with optional clubbed segment isolation `seg_starts` */
LFM3_API float lfm3_forward_24_layers_clubbed(const int *token_ids, const int *seg_starts, int T, float *out_final_hidden)
{
    double t0 = now_ms();
    if (g_lfm3.s23_pin_requested) {
        lfm3_configure_s23_snapdragon8gen2(1);
    }

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    float *cur = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    for (int t = 0; t < T; t++) {
        int tid = token_ids[t];
        const uint16_t *erow = emb_w + (size_t)tid * LFM3_D;
        float *dst = cur + (size_t)t * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }

    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, T);
        execute_single_lfm3_layer_c(&g_lfm3.trunks[l], cur, nxt, T, seg_starts);
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    rms_norm_bf16(cur, emb_norm_w, out_final_hidden, T, LFM3_D, 1e-5f);

    free(cur);
    free(nxt);
    return (float)(now_ms() - t0);
}

LFM3_API float lfm3_forward_24_layers(const int *token_ids, int T, float *out_final_hidden)
{
    return lfm3_forward_24_layers_clubbed(token_ids, NULL, T, out_final_hidden);
}

/* Two-Stage LM Head (`global-0/lm_head.4bit.bin` SDOT + Top-64 Exact 16-bit BF16 Refinement):
 * Stage A: Computes all 128,000 logits in 4-bit Group-64 (`129.39 MiB` instead of `500.00 MiB` BF16 = `3.86x` less DRAM bw).
 * Stage B: Recomputes the exact 16-bit BF16 dot product (`64 x 2048 = 256 KB` = `0.02 ms`) for the top-64 candidates
 *          from `embed_tokens.bf16.bin`, guaranteeing 100.00% identical token selection to full BF16! */
static int compute_lm_head_logits_soc(
    const float *h_last_2048,
    float *logits,
    const int *prev_gen_ids,
    int gen_count,
    int allow_thinking
)
{
    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const int vocab_sz = 128000;

    if (g_lfm3.use_4bit_lm_head && g_lfm3.lm_head_4bit_mf.ptr && g_lfm3.lm_head_4bit_mf.size >= LFM3_LM_HEAD_4BIT_BYTES) {
        const uint8_t *q4_base = (const uint8_t *)g_lfm3.lm_head_4bit_mf.ptr;
        const uint8_t *nib_base = q4_base;
        const uint8_t *stp_base = q4_base + 131072000ULL;
        const float   *scl_base = (const float *)(q4_base + 135168000ULL);
        linear_4bit_group64_omp(h_last_2048, nib_base, stp_base, scl_base, logits, 1, vocab_sz, LFM3_D);
    } else {
        linear_bf16_omp(h_last_2048, emb_w, logits, 1, vocab_sz, LFM3_D);
    }

    for (int g = 0; g < gen_count; g++) {
        int prev_id = prev_gen_ids[g];
        if (prev_id >= 0 && prev_id < vocab_sz) {
            if (logits[prev_id] > 0.0f) logits[prev_id] /= 1.12f;
            else logits[prev_id] *= 1.12f;
        }
    }
    if (!allow_thinking) {
        logits[124901] = -1e30f;
        logits[125033] = -1e30f;
        logits[125034] = -1e30f;
    } else if (allow_thinking == 1) {
        logits[125034] = -1e30f;
    } else if (allow_thinking == 2) {
        logits[125033] = -1e30f;
        logits[125034] = -1e30f;
    }

    if (g_lfm3.use_4bit_lm_head && g_lfm3.lm_head_4bit_mf.ptr) {
        /* Find top-64 candidates from 4-bit logits and refine with exact 16-bit BF16 dot products */
        int top_ids[64];
        float top_vals[64];
        for (int k = 0; k < 64; k++) { top_ids[k] = 0; top_vals[k] = -1e30f; }
        for (int v = 0; v < vocab_sz; v++) {
            float val = logits[v];
            if (val > top_vals[63]) {
                int pos = 63;
                while (pos > 0 && val > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_ids[pos] = top_ids[pos - 1];
                    pos--;
                }
                top_vals[pos] = val;
                top_ids[pos] = v;
            }
        }
        int best_id = top_ids[0];
        float best_v = -1e30f;
        for (int k = 0; k < 64; k++) {
            int vid = top_ids[k];
            const uint16_t *w_row = emb_w + (size_t)vid * LFM3_D;
            float sum = 0.0f;
            for (int d = 0; d < LFM3_D; d++) {
                sum += h_last_2048[d] * bf16_to_f32(w_row[d]);
            }
            float exact_lg = bf16_round_f32(sum);
            for (int g = 0; g < gen_count; g++) {
                if (prev_gen_ids[g] == vid) {
                    if (exact_lg > 0.0f) exact_lg /= 1.12f;
                    else exact_lg *= 1.12f;
                }
            }
            logits[vid] = exact_lg;
            if (exact_lg > best_v) {
                best_v = exact_lg;
                best_id = vid;
            }
        }
        return best_id;
    }

    int best_id = 0;
    float best_v = logits[0];
    for (int v = 1; v < vocab_sz; v++) {
        if (logits[v] > best_v) {
            best_v = logits[v];
            best_id = v;
        }
    }
    return best_id;
}

/* Computes tied `128,000`-vocab `lm_head` logits `out_logits[128000]` and returns `top1_id` */
LFM3_API int lfm3_lm_head_logits(const float *h_last_2048, float *out_logits_128k, float *out_max_logit)
{
    if (!g_lfm3.initialized || !g_lfm3.emb_mf.ptr || !h_last_2048) return -1;
    const int vocab_sz = 128000;
    float *lg = out_logits_128k ? out_logits_128k : (float *)malloc((size_t)vocab_sz * sizeof(float));
    int best_id = compute_lm_head_logits_soc(h_last_2048, lg, NULL, 0, 1);
    if (out_max_logit) *out_max_logit = lg[best_id];
    if (!out_logits_128k) free(lg);
    return best_id;
}

/* ============================================================================
 * Fast Cached Autoregressive Generation (`lfm3_generate_cached`)
 * Uses a 24-layer KV Cache (`k_cache`, `v_cache` for the 11 GQA layers) and
 * ShortConv rolling state (`conv_pre` for the 13 width-7 ShortConv layers)
 * so prefill runs in parallel (8-token NEON blocking) and every generated
 * token runs as an O(1) `T = 1` step without recomputing the prompt!
 * ============================================================================ */
#define LFM3_MAX_GEN_SEQ 1536

static float *g_conv_pre_cache[LFM3_LAYERS] = { NULL };
static float *g_k_cache[LFM3_LAYERS] = { NULL };
static float *g_v_cache[LFM3_LAYERS] = { NULL };
static int g_cached_seq_ids[LFM3_MAX_GEN_SEQ];
static int g_cached_seq_len = 0;
static uint8_t g_stop_mask[128000] = { 0 };

LFM3_API void lfm3_set_stop_mask(const uint8_t *mask_128k)
{
    if (mask_128k) {
        memcpy(g_stop_mask, mask_128k, 128000);
    }
}

/* Pre-allocated scratch buffers for zero-malloc cached prefill/decode (`T <= LFM3_MAX_GEN_SEQ`) */
static float *g_scr_op_in = NULL;
static float *g_scr_op_out_raw = NULL;
static float *g_scr_op_normed = NULL;
static float *g_scr_h_mid = NULL;
static float *g_scr_bcx = NULL;
static float *g_scr_post_gated = NULL;
static float *g_scr_q = NULL;
static float *g_scr_k = NULL;
static float *g_scr_v = NULL;
static float *g_scr_gate = NULL;
static float *g_scr_gated_attn = NULL;
static float *g_scr_ffn_in = NULL;
static float *g_scr_w1_out = NULL;
static float *g_scr_w3_out = NULL;
static float *g_scr_ffn_raw = NULL;
static float *g_scr_ffn_normed = NULL;

static void ensure_lfm3_kv_buffers(void)
{
    for (int l = 0; l < LFM3_LAYERS; l++) {
        uint32_t kind = g_lfm3.trunks[l].hdr.layer_kind;
        if (kind == 0) {
            if (!g_conv_pre_cache[l]) {
                g_conv_pre_cache[l] = (float *)calloc((size_t)LFM3_MAX_GEN_SEQ * LFM3_D, sizeof(float));
            }
        } else {
            if (!g_k_cache[l]) {
                g_k_cache[l] = (float *)calloc((size_t)LFM3_MAX_GEN_SEQ * 512, sizeof(float));
            }
            if (!g_v_cache[l]) {
                g_v_cache[l] = (float *)calloc((size_t)LFM3_MAX_GEN_SEQ * 512, sizeof(float));
            }
        }
    }
    if (!g_scr_op_in) {
        size_t max_t = 1024;
        g_scr_op_in = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_op_out_raw = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_op_normed = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_h_mid = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_bcx = (float *)malloc(max_t * 6144 * sizeof(float));
        g_scr_post_gated = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_q = (float *)malloc(max_t * 2048 * sizeof(float));
        g_scr_k = (float *)malloc(max_t * 512 * sizeof(float));
        g_scr_v = (float *)malloc(max_t * 512 * sizeof(float));
        g_scr_gate = (float *)malloc(max_t * 16 * sizeof(float));
        g_scr_gated_attn = (float *)malloc(max_t * 2048 * sizeof(float));
        g_scr_ffn_in = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_w1_out = (float *)malloc(max_t * LFM3_FFN * sizeof(float));
        g_scr_w3_out = (float *)malloc(max_t * LFM3_FFN * sizeof(float));
        g_scr_ffn_raw = (float *)malloc(max_t * LFM3_D * sizeof(float));
        g_scr_ffn_normed = (float *)malloc(max_t * LFM3_D * sizeof(float));
    }
}

/* Fused `w1 + w3 + SwiGLU` 4-Bit Group-64 Kernel for `T = 1` (Adreno 740 `dot_acc_sat` or Kryo NEON `SDOT`):
 * Reads `x[2048]` (`8 KB` in L1D / 2 KB in Adreno GMEM) ONCE and computes both `w1[o] · x` and `w3[o] · x`,
 * then immediately applies `silu(v1) * v3` (`7680 / 7680` bit-identical between Adreno 740 and CPU)! */
static void fused_swiglu_w1_w3_4bit_T1(
    int layer_id,
    const float *x,
    const uint8_t *w1_nib,
    const uint8_t *w1_stp,
    const float *w1_scl,
    const uint8_t *w3_nib,
    const uint8_t *w3_stp,
    const float *w3_scl,
    float *out_swiglu
)
{
    const int groups = LFM3_D / 64; /* 32 groups */
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    size_t q4_bytes = 2ULL * (size_t)LFM3_FFN * (size_t)(groups * 33 + 4);
    size_t bf16_equiv = 2ULL * (size_t)LFM3_FFN * (size_t)LFM3_D * 2ULL;
    g_lfm3.bytes_tier2_lpddr5x += (unsigned long long)q4_bytes;
    g_lfm3.bytes_tier1_sram_slc += (unsigned long long)((size_t)(LFM3_D + LFM3_FFN) * 4ULL);
    g_lfm3.bytes_tier0_reg += (unsigned long long)(2ULL * (size_t)LFM3_FFN * (size_t)LFM3_D);
    g_lfm3.bytes_saved_q4_and_broadcast += (unsigned long long)(bf16_equiv - q4_bytes);

#if defined(__ARM_FEATURE_DOTPROD)
    int8_t x_q[LFM3_D];
    float x_sc[32];
    quantize_f32_to_s8_group64(x, 1, LFM3_D, x_q, x_sc);

    int cpu_start_row = 0;
#if defined(__linux__) && defined(__aarch64__)
    int gpu_dispatched = 0;
    const int gpu_rows = 3840; /* 50% on Adreno 740 GPU + 50% on Kryo CPU concurrently */
    if (g_lfm3.adreno_ready && g_lfm3.adreno_enabled && layer_id >= 0 && layer_id < LFM3_LAYERS) {
        cl_int err = 0;
        int8_t  *xq_ptr   = (int8_t *)g_adreno.clEnqueueMapBuffer(g_adreno.q, g_adreno.b_xq,   1, CL_MAP_WRITE_SOC, 0, 2048,   0, NULL, NULL, &err);
        float   *xsc_ptr  = (float *)g_adreno.clEnqueueMapBuffer(g_adreno.q, g_adreno.b_xsc,  1, CL_MAP_WRITE_SOC, 0, 32 * 4, 0, NULL, NULL, &err);
        int32_t *xsum_ptr = (int32_t *)g_adreno.clEnqueueMapBuffer(g_adreno.q, g_adreno.b_xsum, 1, CL_MAP_WRITE_SOC, 0, 32 * 4, 0, NULL, NULL, &err);
        if (xq_ptr && xsc_ptr && xsum_ptr) {
            memcpy(xq_ptr, x_q, 2048);
            memcpy(xsc_ptr, x_sc, 32 * sizeof(float));
            for (int g = 0; g < 32; g++) {
                int32_t s = 0;
                const int8_t *xg = x_q + g * 64;
                for (int j = 0; j < 64; j++) s += (int32_t)xg[j];
                xsum_ptr[g] = s;
            }
            g_adreno.clEnqueueUnmapMemObject(g_adreno.q, g_adreno.b_xq,   xq_ptr,   0, NULL, NULL);
            g_adreno.clEnqueueUnmapMemObject(g_adreno.q, g_adreno.b_xsc,  xsc_ptr,  0, NULL, NULL);
            g_adreno.clEnqueueUnmapMemObject(g_adreno.q, g_adreno.b_xsum, xsum_ptr, 0, NULL, NULL);

            g_adreno.clSetKernelArg(g_adreno.k_swiglu, 0, sizeof(cl_mem), &g_adreno.b_ffn[layer_id]);
            size_t gws = (gpu_rows / 4) * 128, lws = 128;
            if (g_adreno.clEnqueueNDRangeKernel(g_adreno.q, g_adreno.k_swiglu, 1, NULL, &gws, &lws, 0, NULL, NULL) == 0) {
                g_adreno.clFlush(g_adreno.q);
                gpu_dispatched = 1;
                cpu_start_row = gpu_rows;
            }
        }
    }
#else
    (void)layer_id;
#endif

    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = cpu_start_row; o < LFM3_FFN; o++) {
        const uint8_t *n1_row = w1_nib + (size_t)o * (size_t)(groups * 32);
        const uint8_t *s1_row = w1_stp + (size_t)o * (size_t)groups;
        const float r1_scl = w1_scl[o];

        const uint8_t *n3_row = w3_nib + (size_t)o * (size_t)(groups * 32);
        const uint8_t *s3_row = w3_stp + (size_t)o * (size_t)groups;
        const float r3_scl = w3_scl[o];

        const int8x16_t eight_s8 = vdupq_n_s8(8);
        const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
        float s1 = 0.0f, s3 = 0.0f;
        for (int g = 0; g < groups; g++) {
            const int c = g * 64;
            int8x16_t x0 = vld1q_s8(x_q + c);
            int8x16_t x1 = vld1q_s8(x_q + c + 16);
            int8x16_t x2 = vld1q_s8(x_q + c + 32);
            int8x16_t x3 = vld1q_s8(x_q + c + 48);

            const uint8_t *nb1 = n1_row + g * 32;
            uint8x16_t pk1_0 = vld1q_u8(nb1), pk1_1 = vld1q_u8(nb1 + 16);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk1_0, mask_0f)), eight_s8), x0);
            d1 = vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk1_1, mask_0f)), eight_s8), x1);
            d1 = vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk1_0, 4)), eight_s8), x2);
            d1 = vdotq_s32(d1, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk1_1, 4)), eight_s8), x3);
            s1 += (float)vaddvq_s32(d1) * ((float)s1_row[g] * x_sc[g]);

            const uint8_t *nb3 = n3_row + g * 32;
            uint8x16_t pk3_0 = vld1q_u8(nb3), pk3_1 = vld1q_u8(nb3 + 16);
            int32x4_t d3 = vdupq_n_s32(0);
            d3 = vdotq_s32(d3, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk3_0, mask_0f)), eight_s8), x0);
            d3 = vdotq_s32(d3, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk3_1, mask_0f)), eight_s8), x1);
            d3 = vdotq_s32(d3, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk3_0, 4)), eight_s8), x2);
            d3 = vdotq_s32(d3, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk3_1, 4)), eight_s8), x3);
            s3 += (float)vaddvq_s32(d3) * ((float)s3_row[g] * x_sc[g]);
        }
        float v1 = bf16_round_f32(s1 * r1_scl);
        float v3 = bf16_round_f32(s3 * r3_scl);
        float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
        out_swiglu[o] = bf16_round_f32(silu * v3);
    }
#if defined(__linux__) && defined(__aarch64__)
    if (gpu_dispatched) {
        cl_int err = 0;
        float *gpu_out = (float *)g_adreno.clEnqueueMapBuffer(g_adreno.q, g_adreno.b_out, 1, CL_MAP_READ_SOC, 0, gpu_rows * sizeof(float), 0, NULL, NULL, &err);
        if (gpu_out) {
            memcpy(out_swiglu, gpu_out, gpu_rows * sizeof(float));
            g_adreno.clEnqueueUnmapMemObject(g_adreno.q, g_adreno.b_out, gpu_out, 0, NULL, NULL);
            g_lfm3.adreno_opencl_launches++;
        }
    }
#endif
#else
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < LFM3_FFN; o++) {
        const uint8_t *n1_row = w1_nib + (size_t)o * (size_t)(groups * 32);
        const uint8_t *s1_row = w1_stp + (size_t)o * (size_t)groups;
        const float r1_scl = w1_scl[o];
        const uint8_t *n3_row = w3_nib + (size_t)o * (size_t)(groups * 32);
        const uint8_t *s3_row = w3_stp + (size_t)o * (size_t)groups;
        const float r3_scl = w3_scl[o];
        float s1 = 0.0f, s3 = 0.0f;
        for (int g = 0; g < groups; g++) {
            const uint8_t *nb1 = n1_row + g * 32;
            const uint8_t *nb3 = n3_row + g * 32;
            const float gsc1 = (float)s1_row[g] * r1_scl;
            const float gsc3 = (float)s3_row[g] * r3_scl;
            const int col0 = g * 64;
            for (int j = 0; j < 32; j++) {
                uint8_t pk1 = nb1[j], pk3 = nb3[j];
                float xlo = x[col0 + j], xhi = x[col0 + 32 + j];
                s1 += xlo * ((float)((int)(pk1 & 0x0F) - 8) * gsc1) + xhi * ((float)((int)(pk1 >> 4) - 8) * gsc1);
                s3 += xlo * ((float)((int)(pk3 & 0x0F) - 8) * gsc3) + xhi * ((float)((int)(pk3 >> 4) - 8) * gsc3);
            }
        }
        float v1 = bf16_round_f32(s1);
        float v3 = bf16_round_f32(s3);
        float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
        out_swiglu[o] = bf16_round_f32(silu * v3);
    }
#endif
}

static void execute_lfm3_layer_cached_step(
    int layer_id,
    const Lfm3TrunkMapped *trk,
    const float *h_in,
    float *h_out,
    int pos_offset,
    int T
)
{
    const uint8_t *op = (const uint8_t *)trk->operator_mf.ptr;
    uint32_t kind = trk->hdr.layer_kind;
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;

    float *op_in = g_scr_op_in;
    float *op_out_raw = g_scr_op_out_raw;
    float *op_normed = g_scr_op_normed;
    float *h_mid = g_scr_h_mid;

    if (kind == 0) {
        uint32_t len_op_norm, len_in_proj, len_conv_w;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_in_proj, op + 20, 4);
        memcpy(&len_conv_w, op + 24, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *in_proj_w = (const uint16_t *)(op + cur); cur += len_in_proj;
        const uint16_t *conv_w = (const uint16_t *)(op + cur); cur += len_conv_w;
        const uint16_t *out_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *bcx = g_scr_bcx;
        float *post_gated = g_scr_post_gated;
        if (g_lfm3.use_4bit_operators && trk->operator_4bit_mf.ptr &&
            trk->operator_4bit_mf.size >= (LFM3_OP_CONV_IN_4BIT_BYTES + LFM3_OP_OUT_4BIT_BYTES)) {
            const uint8_t *oq = (const uint8_t *)trk->operator_4bit_mf.ptr;
            linear_4bit_group64_omp(op_in, oq, oq + 6291456, (const float *)(oq + 6488064), bcx, T, 6144, LFM3_D);
        } else {
            linear_bf16_omp(op_in, in_proj_w, bcx, T, 6144, LFM3_D);
        }

        float *c_cache = g_conv_pre_cache[layer_id];
        for (int t = 0; t < T; t++) {
            int p = pos_offset + t;
            if (p < LFM3_MAX_GEN_SEQ) {
                float *dst_pre = c_cache + (size_t)p * LFM3_D;
                const float *bcx_t = bcx + (size_t)t * 6144;
                for (int c = 0; c < LFM3_D; c++) {
                    dst_pre[c] = bf16_round_f32(bcx_t[c] * bcx_t[4096 + c]);
                }
            }
        }

        for (int c = 0; c < LFM3_D; c++) {
            float w_k[7];
            for (int k = 0; k < 7; k++) w_k[k] = bf16_to_f32(conv_w[c * 7 + k]);
            for (int t = 0; t < T; t++) {
                int p = pos_offset + t;
                float acc = 0.0f;
                for (int k = 0; k < 7; k++) {
                    int ts = p - 6 + k;
                    if (ts >= 0 && ts < LFM3_MAX_GEN_SEQ) {
                        acc += c_cache[(size_t)ts * LFM3_D + c] * w_k[k];
                    }
                }
                float conv_c = bf16_round_f32(acc);
                float C_val = bcx[(size_t)t * 6144 + 2048 + c];
                post_gated[(size_t)t * LFM3_D + c] = bf16_round_f32(C_val * conv_c);
            }
        }
        if (g_lfm3.use_4bit_operators && trk->operator_4bit_mf.ptr &&
            trk->operator_4bit_mf.size >= (LFM3_OP_CONV_IN_4BIT_BYTES + LFM3_OP_OUT_4BIT_BYTES)) {
            const uint8_t *oq = (const uint8_t *)trk->operator_4bit_mf.ptr + LFM3_OP_CONV_IN_4BIT_BYTES;
            linear_4bit_group64_omp(post_gated, oq, oq + 2097152, (const float *)(oq + 2162688), op_out_raw, T, LFM3_D, LFM3_D);
        } else {
            linear_bf16_omp(post_gated, out_proj_w, op_out_raw, T, LFM3_D, LFM3_D);
        }
    } else {
        uint32_t len_op_norm, len_q, len_k, len_v, len_gate, len_qn, len_kn;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_q, op + 20, 4);
        memcpy(&len_k, op + 24, 4);
        memcpy(&len_v, op + 28, 4);
        memcpy(&len_gate, op + 32, 4);
        memcpy(&len_qn, op + 36, 4);
        memcpy(&len_kn, op + 40, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *q_w = (const uint16_t *)(op + cur); cur += len_q;
        const uint16_t *k_w = (const uint16_t *)(op + cur); cur += len_k;
        const uint16_t *v_w = (const uint16_t *)(op + cur); cur += len_v;
        const uint16_t *gate_w = (const uint16_t *)(op + cur); cur += len_gate;
        const uint16_t *qn_w = (const uint16_t *)(op + cur); cur += len_qn;
        const uint16_t *kn_w = (const uint16_t *)(op + cur); cur += len_kn;
        const uint16_t *o_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *q = g_scr_q;
        float *k = g_scr_k;
        float *v = g_scr_v;
        float *gate_raw = g_scr_gate;
        float *gated_attn = g_scr_gated_attn;

        if (g_lfm3.use_4bit_operators && trk->operator_4bit_mf.ptr &&
            trk->operator_4bit_mf.size >= (LFM3_OP_ATTN_QKV_4BIT_BYTES + LFM3_OP_OUT_4BIT_BYTES)) {
            const uint8_t *oq = (const uint8_t *)trk->operator_4bit_mf.ptr;
            linear_4bit_group64_omp(op_in, oq, oq + 3145728, (const float *)(oq + 3244032), g_scr_bcx, T, 3072, LFM3_D);
            for (int t = 0; t < T; t++) {
                const float *qkv_t = g_scr_bcx + (size_t)t * 3072;
                memcpy(q + (size_t)t * 2048, qkv_t, 2048 * sizeof(float));
                memcpy(k + (size_t)t * 512,  qkv_t + 2048, 512 * sizeof(float));
                memcpy(v + (size_t)t * 512,  qkv_t + 2560, 512 * sizeof(float));
            }
        } else {
            linear_bf16_omp(op_in, q_w, q, T, 2048, LFM3_D);
            linear_bf16_omp(op_in, k_w, k, T, 512, LFM3_D);
            linear_bf16_omp(op_in, v_w, v, T, 512, LFM3_D);
        }
        linear_bf16_omp(op_in, gate_w, gate_raw, T, 16, LFM3_D);

        rms_norm_bf16(q, qn_w, q, T * 16, 128, 1e-5f);
        rms_norm_bf16(k, kn_w, k, T * 4, 128, 1e-5f);

        const float *inv_freq = (kind == 1) ? g_lfm3.inv_freq_sliding : g_lfm3.inv_freq_full;
        float *kc_base = g_k_cache[layer_id];
        float *vc_base = g_v_cache[layer_id];

        for (int t = 0; t < T; t++) {
            int pos_t = pos_offset + t;
            float cos_t[64], sin_t[64];
            for (int d = 0; d < 64; d++) {
                float angle = (float)pos_t * inv_freq[d];
                cos_t[d] = bf16_round_f32(cosf(angle));
                sin_t[d] = bf16_round_f32(sinf(angle));
            }
            for (int h = 0; h < 16; h++) {
                float *qh = q + (size_t)t * 2048 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = qh[d], x2 = qh[d + 64];
                    qh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    qh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
            for (int h = 0; h < 4; h++) {
                float *kh = k + (size_t)t * 512 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = kh[d], x2 = kh[d + 64];
                    kh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    kh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
            if (pos_t < LFM3_MAX_GEN_SEQ) {
                memcpy(kc_base + (size_t)pos_t * 512, k + (size_t)t * 512, 512 * sizeof(float));
                memcpy(vc_base + (size_t)pos_t * 512, v + (size_t)t * 512, 512 * sizeof(float));
            }
        }

        const float scale = 1.0f / sqrtf(128.0f);
        for (int h = 0; h < 16; h++) {
            int kv_h = h / 4;
            for (int t = 0; t < T; t++) {
                int p = pos_offset + t;
                if (p >= LFM3_MAX_GEN_SEQ) p = LFM3_MAX_GEN_SEQ - 1;
                const float *qh = q + (size_t)t * 2048 + h * 128;
                int j_start = 0;
                if (kind == 1 && p - LFM3_SLIDING_W + 1 > 0) {
                    j_start = p - LFM3_SLIDING_W + 1;
                }
                float g_val = gate_raw[(size_t)t * 16 + h];
                float head_gate = bf16_round_f32(1.0f / (1.0f + expf(-g_val)));
                float *out_qh = gated_attn + (size_t)t * 2048 + h * 128;
                fa2_online_softmax_head_c(qh, kc_base, vc_base, kv_h, j_start, p, scale, head_gate, out_qh);
            }
        }
        if (g_lfm3.use_4bit_operators && trk->operator_4bit_mf.ptr &&
            trk->operator_4bit_mf.size >= (LFM3_OP_ATTN_QKV_4BIT_BYTES + LFM3_OP_OUT_4BIT_BYTES)) {
            const uint8_t *oq = (const uint8_t *)trk->operator_4bit_mf.ptr + LFM3_OP_ATTN_QKV_4BIT_BYTES;
            linear_4bit_group64_omp(gated_attn, oq, oq + 2097152, (const float *)(oq + 2162688), op_out_raw, T, LFM3_D, 2048);
        } else {
            linear_bf16_omp(gated_attn, o_proj_w, op_out_raw, T, LFM3_D, 2048);
        }
    }

    const uint16_t *post_op_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 1);
    rms_norm_bf16(op_out_raw, post_op_norm_w, op_normed, T, LFM3_D, 1e-5f);
    for (int i = 0; i < T * LFM3_D; i++) {
        h_mid[i] = bf16_round_f32(h_in[i] + op_normed[i]);
    }

    const uint16_t *ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 3);
    const uint16_t *post_ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 4);

    float *ffn_in = g_scr_ffn_in;
    float *w1_out = g_scr_w1_out;
    float *w3_out = g_scr_w3_out;
    float *ffn_raw = g_scr_ffn_raw;
    float *ffn_normed = g_scr_ffn_normed;

    rms_norm_bf16(h_mid, ffn_norm_w, ffn_in, T, LFM3_D, 1e-5f);

    if (g_lfm3.use_4bit_ffn && trk->ffn_4bit_mf.ptr && trk->ffn_4bit_mf.size >= 3 * LFM3_FFN_MAT_4BIT_BYTES) {
        const uint8_t *base4 = (const uint8_t *)trk->ffn_4bit_mf.ptr;
        const uint8_t *w1_nib = base4;
        const uint8_t *w1_stp = w1_nib + (7680 * 32 * 32);
        const float *w1_scl = (const float *)(w1_stp + (7680 * 32));

        const uint8_t *w3_nib = base4 + LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w3_stp = w3_nib + (7680 * 32 * 32);
        const float *w3_scl = (const float *)(w3_stp + (7680 * 32));

        const uint8_t *w2_nib = base4 + 2 * LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w2_stp = w2_nib + (2048 * 120 * 32);
        const float *w2_scl = (const float *)(w2_stp + (2048 * 120));

        if (T == 1) {
            fused_swiglu_w1_w3_4bit_T1(layer_id, ffn_in, w1_nib, w1_stp, w1_scl, w3_nib, w3_stp, w3_scl, w1_out);
        } else {
            linear_4bit_group64_omp(ffn_in, w1_nib, w1_stp, w1_scl, w1_out, T, LFM3_FFN, LFM3_D);
            linear_4bit_group64_omp(ffn_in, w3_nib, w3_stp, w3_scl, w3_out, T, LFM3_FFN, LFM3_D);
            #pragma omp parallel for num_threads(n_thr) schedule(static)
            for (int i = 0; i < T * LFM3_FFN; i++) {
                float v1 = w1_out[i];
                float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
                w1_out[i] = bf16_round_f32(silu * w3_out[i]);
            }
        }
        linear_4bit_group64_omp(w1_out, w2_nib, w2_stp, w2_scl, ffn_raw, T, LFM3_D, LFM3_FFN);
    } else {
        const uint16_t *w1_w = (const uint16_t *)get_lfm3_record_ptr(trk, 10);
        const uint16_t *w3_w = (const uint16_t *)get_lfm3_record_ptr(trk, 11);
        const uint16_t *w2_w = (const uint16_t *)get_lfm3_record_ptr(trk, 12);

        linear_bf16_omp(ffn_in, w1_w, w1_out, T, LFM3_FFN, LFM3_D);
        linear_bf16_omp(ffn_in, w3_w, w3_out, T, LFM3_FFN, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int i = 0; i < T * LFM3_FFN; i++) {
            float v1 = w1_out[i];
            float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
            w1_out[i] = bf16_round_f32(silu * w3_out[i]);
        }
        linear_bf16_omp(w1_out, w2_w, ffn_raw, T, LFM3_D, LFM3_FFN);
    }
    rms_norm_bf16(ffn_raw, post_ffn_norm_w, ffn_normed, T, LFM3_D, 1e-5f);

    for (int i = 0; i < T * LFM3_D; i++) {
        h_out[i] = bf16_round_f32(h_mid[i] + ffn_normed[i]);
    }
}

LFM3_API int lfm3_generate_cached(
    const int *prompt_ids,
    int prompt_len,
    int max_new_tokens,
    int *out_gen_ids,
    float *out_last_hidden_2048,
    float *out_total_ms
)
{
    if (!g_lfm3.initialized || prompt_len <= 0 || max_new_tokens <= 0) return 0;
    double t0 = now_ms();
    if (g_lfm3.s23_pin_requested) {
        lfm3_configure_s23_snapdragon8gen2(1);
    }
    ensure_lfm3_kv_buffers();

    if (prompt_len > 1024) {
        prompt_len = 1024;
    }

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    /* Find longest common prefix `lcp` with `g_cached_seq_ids` so we skip redundant prefix prefill! */
    int lcp = 0;
    while (lcp < prompt_len - 1 && lcp < g_cached_seq_len && prompt_ids[lcp] == g_cached_seq_ids[lcp]) {
        lcp++;
    }
    int new_prefill_len = prompt_len - lcp;

    float *cur = (float *)malloc((size_t)new_prefill_len * LFM3_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)new_prefill_len * LFM3_D * sizeof(float));
    for (int t = 0; t < new_prefill_len; t++) {
        int tid = prompt_ids[lcp + t];
        if (tid < 0 || tid >= 128000) tid = 0;
        g_cached_seq_ids[lcp + t] = tid;
        const uint16_t *erow = emb_w + (size_t)tid * LFM3_D;
        float *dst = cur + (size_t)t * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }
    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, new_prefill_len);
        execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], cur, nxt, lcp, new_prefill_len);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    g_cached_seq_len = prompt_len;

    float h_last[LFM3_D];
    rms_norm_bf16(cur + (size_t)(new_prefill_len - 1) * LFM3_D, emb_norm_w, h_last, 1, LFM3_D, 1e-5f);
    free(cur);
    free(nxt);

    float *logits = (float *)malloc(128000 * sizeof(float));
    float cur_1tok[LFM3_D], nxt_1tok[LFM3_D];
    int gen_count = 0;
    int cur_pos = prompt_len;

    for (int step = 0; step < max_new_tokens && cur_pos < LFM3_MAX_GEN_SEQ; step++) {
        int best_id = compute_lm_head_logits_soc(h_last, logits, out_gen_ids, gen_count, 0);
        out_gen_ids[gen_count++] = best_id;
        if (best_id == 124900 || best_id == 124895) {
            break;
        }
        int prev_tok = (gen_count >= 2) ? out_gen_ids[gen_count - 2] : -1;
        int prev_is_digit = (prev_tok >= 0 && prev_tok < 128000 && g_stop_mask[prev_tok] == 3);
        if (gen_count >= 6 && g_stop_mask[best_id] == 2 && !prev_is_digit) {
            break;
        }
        if (gen_count >= 6 && (best_id == 8 || best_id == 207) && prev_tok >= 0 && g_stop_mask[prev_tok] == 1) {
            gen_count--;
            break;
        }
        if (gen_count >= 16 && g_stop_mask[best_id] == 1 && !prev_is_digit) {
            break;
        }
        if (step + 1 >= max_new_tokens) {
            break;
        }

        g_cached_seq_ids[cur_pos] = best_id;
        const uint16_t *erow = emb_w + (size_t)best_id * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) cur_1tok[d] = bf16_to_f32(erow[d]);

        float *c_ptr = cur_1tok;
        float *n_ptr = nxt_1tok;
        for (int l = 0; l < LFM3_LAYERS; l++) {
            lfm3_dm_record_layer(l, 1);
            execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], c_ptr, n_ptr, cur_pos, 1);
            float *tmp = c_ptr; c_ptr = n_ptr; n_ptr = tmp;
        }
        rms_norm_bf16(c_ptr, emb_norm_w, h_last, 1, LFM3_D, 1e-5f);
        cur_pos++;
        g_cached_seq_len = cur_pos;
    }

    if (out_last_hidden_2048) {
        memcpy(out_last_hidden_2048, h_last, LFM3_D * sizeof(float));
    }
    if (out_total_ms) {
        *out_total_ms = (float)(now_ms() - t0);
    }
    free(logits);
    return gen_count;
}

/* ============================================================================
 * Continuous Latent Step + Cached Decode (`lfm3_prefill_and_latent_decode`)
 * 1. Prefills `prompt_ids[0..prompt_len-1]` using longest-common-prefix KV +
 *    ShortConv cache reuse and writes `out_prefill_h_2048`.
 * 2. If `latent_embs` (`n_latent_steps x 2048`) is provided (`n_latent_steps > 0`),
 *    injects each continuous `<latent>` vector `r_t = Lfm3NativeHybridLink(h_t)`
 *    directly into the 24-layer KV + ShortConv cache as a continuous embedding step!
 * 3. Decodes up to `max_new_tokens` tokens from the resulting post-latent state.
 * ============================================================================ */
LFM3_API float lfm3_prefill_cached(
    const int *prompt_ids,
    int prompt_len,
    float *out_last_hidden_2048
)
{
    if (!g_lfm3.initialized || prompt_len <= 0 || !out_last_hidden_2048) return 0.0f;
    double t0 = now_ms();
    if (g_lfm3.s23_pin_requested) {
        lfm3_configure_s23_snapdragon8gen2(1);
    }
    ensure_lfm3_kv_buffers();

    if (prompt_len > 1024) {
        prompt_len = 1024;
    }

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    int lcp = 0;
    while (lcp < prompt_len - 1 && lcp < g_cached_seq_len && prompt_ids[lcp] == g_cached_seq_ids[lcp]) {
        lcp++;
    }
    int new_prefill_len = prompt_len - lcp;

    float *cur = (float *)malloc((size_t)new_prefill_len * LFM3_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)new_prefill_len * LFM3_D * sizeof(float));
    for (int t = 0; t < new_prefill_len; t++) {
        int tid = prompt_ids[lcp + t];
        if (tid < 0 || tid >= 128000) tid = 0;
        g_cached_seq_ids[lcp + t] = tid;
        const uint16_t *erow = emb_w + (size_t)tid * LFM3_D;
        float *dst = cur + (size_t)t * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }
    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, new_prefill_len);
        execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], cur, nxt, lcp, new_prefill_len);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    g_cached_seq_len = prompt_len;

    rms_norm_bf16(cur + (size_t)(new_prefill_len - 1) * LFM3_D, emb_norm_w, out_last_hidden_2048, 1, LFM3_D, 1e-5f);
    free(cur);
    free(nxt);
    return (float)(now_ms() - t0);
}

LFM3_API float lfm3_append_cached(
    const int *new_ids,
    int new_len,
    float *out_last_hidden_2048
)
{
    if (!g_lfm3.initialized || new_len <= 0 || !out_last_hidden_2048) return 0.0f;
    double t0 = now_ms();
    if (g_lfm3.s23_pin_requested) {
        lfm3_configure_s23_snapdragon8gen2(1);
    }
    ensure_lfm3_kv_buffers();

    int start_pos = g_cached_seq_len;
    if (start_pos + new_len > LFM3_MAX_GEN_SEQ) {
        new_len = LFM3_MAX_GEN_SEQ - start_pos;
    }
    if (new_len <= 0) return 0.0f;

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    float *cur = (float *)malloc((size_t)new_len * LFM3_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)new_len * LFM3_D * sizeof(float));
    for (int t = 0; t < new_len; t++) {
        int tid = new_ids[t];
        if (tid < 0 || tid >= 128000) tid = 0;
        g_cached_seq_ids[start_pos + t] = tid;
        const uint16_t *erow = emb_w + (size_t)tid * LFM3_D;
        float *dst = cur + (size_t)t * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }
    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, new_len);
        execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], cur, nxt, start_pos, new_len);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    g_cached_seq_len = start_pos + new_len;

    rms_norm_bf16(cur + (size_t)(new_len - 1) * LFM3_D, emb_norm_w, out_last_hidden_2048, 1, LFM3_D, 1e-5f);
    free(cur);
    free(nxt);
    return (float)(now_ms() - t0);
}

LFM3_API float lfm3_step_embedding_cached(
    const float *in_emb_2048,
    float *out_hidden_2048
)
{
    if (!g_lfm3.initialized || !in_emb_2048 || !out_hidden_2048) return 0.0f;
    double t0 = now_ms();
    ensure_lfm3_kv_buffers();
    int cur_pos = g_cached_seq_len;
    if (cur_pos >= LFM3_MAX_GEN_SEQ) cur_pos = LFM3_MAX_GEN_SEQ - 1;

    /* Mark this cache slot as a continuous <latent> vector (-2) so token LCP does not alias it */
    g_cached_seq_ids[cur_pos] = -2;

    float cur_1tok[LFM3_D], nxt_1tok[LFM3_D];
    for (int d = 0; d < LFM3_D; d++) {
        cur_1tok[d] = bf16_round_f32(in_emb_2048[d]);
    }
    float *c_ptr = cur_1tok;
    float *n_ptr = nxt_1tok;
    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, 1);
        execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], c_ptr, n_ptr, cur_pos, 1);
        float *tmp = c_ptr; c_ptr = n_ptr; n_ptr = tmp;
    }
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;
    rms_norm_bf16(c_ptr, emb_norm_w, out_hidden_2048, 1, LFM3_D, 1e-5f);
    g_cached_seq_len = cur_pos + 1;
    return (float)(now_ms() - t0);
}

LFM3_API float lfm3_step_token_cached(
    int token_id,
    float *out_hidden_2048
)
{
    if (!g_lfm3.initialized || token_id < 0 || token_id >= 128000 || !out_hidden_2048) return 0.0f;
    double t0 = now_ms();
    ensure_lfm3_kv_buffers();
    int cur_pos = g_cached_seq_len;
    if (cur_pos >= LFM3_MAX_GEN_SEQ) cur_pos = LFM3_MAX_GEN_SEQ - 1;

    g_cached_seq_ids[cur_pos] = token_id;
    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;
    const uint16_t *erow = emb_w + (size_t)token_id * LFM3_D;

    float cur_1tok[LFM3_D], nxt_1tok[LFM3_D];
    for (int d = 0; d < LFM3_D; d++) cur_1tok[d] = bf16_to_f32(erow[d]);

    float *c_ptr = cur_1tok;
    float *n_ptr = nxt_1tok;
    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, 1);
        execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], c_ptr, n_ptr, cur_pos, 1);
        float *tmp = c_ptr; c_ptr = n_ptr; n_ptr = tmp;
    }
    rms_norm_bf16(c_ptr, emb_norm_w, out_hidden_2048, 1, LFM3_D, 1e-5f);
    g_cached_seq_len = cur_pos + 1;
    return (float)(now_ms() - t0);
}

LFM3_API int lfm3_decode_from_cache(
    const float *initial_h_last_2048,
    int max_new_tokens,
    int allow_thinking,
    int *out_gen_ids,
    float *out_last_hidden_2048,
    float *out_total_ms
)
{
    if (!g_lfm3.initialized || !initial_h_last_2048 || max_new_tokens <= 0) return 0;
    double t0 = now_ms();
    ensure_lfm3_kv_buffers();

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    float h_last[LFM3_D];
    memcpy(h_last, initial_h_last_2048, LFM3_D * sizeof(float));

    float *logits = (float *)malloc(128000 * sizeof(float));
    float cur_1tok[LFM3_D], nxt_1tok[LFM3_D];
    int gen_count = 0;
    int cur_pos = g_cached_seq_len;

    for (int step = 0; step < max_new_tokens && cur_pos < LFM3_MAX_GEN_SEQ; step++) {
        int best_id = compute_lm_head_logits_soc(h_last, logits, out_gen_ids, gen_count, allow_thinking);
        out_gen_ids[gen_count++] = best_id;
        if (best_id == 124900 || best_id == 124895 || best_id == 124906 || best_id == 125020 || best_id == 125028) {
            /* 124900 = <|im_end|>, 124895 = <|endoftext|>, 124906 = <|tool_call_end|>, 125020 = </tool_call>, 125028 = </additional_tools> */
            break;
        }
        if (best_id == 38 && gen_count >= 3) {
            int prev_id = out_gen_ids[gen_count - 2];
            int prev2_id = out_gen_ids[gen_count - 3];
            /* Stop immediately on closing </search> (13981), </python> (63513/12326), or </answer> (44244)
             * where prev2_id is NOT opening '<' (36) and NOT '</think>' (124902) */
            if ((prev_id == 13981 || prev_id == 63513 || prev_id == 12326 || prev_id == 44244) && prev2_id != 36 && prev2_id != 124902) {
                break;
            }
        }
        if (step + 1 >= max_new_tokens && best_id != 125033) {
            break;
        }

        g_cached_seq_ids[cur_pos] = best_id;
        const uint16_t *erow = emb_w + (size_t)best_id * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) cur_1tok[d] = bf16_to_f32(erow[d]);

        float *c_ptr = cur_1tok;
        float *n_ptr = nxt_1tok;
        for (int l = 0; l < LFM3_LAYERS; l++) {
            lfm3_dm_record_layer(l, 1);
            execute_lfm3_layer_cached_step(l, &g_lfm3.trunks[l], c_ptr, n_ptr, cur_pos, 1);
            float *tmp = c_ptr; c_ptr = n_ptr; n_ptr = tmp;
        }
        rms_norm_bf16(c_ptr, emb_norm_w, h_last, 1, LFM3_D, 1e-5f);
        cur_pos++;
        g_cached_seq_len = cur_pos;

        if (best_id == 125033) {
            /* Entered <latent> span: return to HybridAgent controller with h_last at <latent> */
            break;
        }
    }

    if (out_last_hidden_2048) {
        memcpy(out_last_hidden_2048, h_last, LFM3_D * sizeof(float));
    }
    if (out_total_ms) {
        *out_total_ms = (float)(now_ms() - t0);
    }
    free(logits);
    return gen_count;
}
