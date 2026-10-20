// Shared by the matrix-product benches: loads trunk-0 as layer0.cu does and describes the three weight shapes of layer 0 that matter.
#pragma once
#include "../kernels.cuh"

struct Shape { const char *name; const unsigned char *codes; size_t code_stride; const unsigned char *scales; size_t scale_stride; int R, K; };

static void load_trunk0(const char *dir, Trunk &T)
{
    char path[4096];
    memset(&T, 0, sizeof T);
    snprintf(path, sizeof path, "%s/index.bin", dir);
    std::vector<unsigned char> index = slurp(path);
    if (index.size() < 1168 || memcmp(index.data(), "K3TRK001", 8)) { fprintf(stderr, "index.bin is not a trunk index\n"); exit(1); }
    const unsigned count = u32(index.data() + 16);
    for (unsigned p = 0; p < count; p++) {
        const unsigned char *e = index.data() + 1168 + (size_t)p * 72;
        Record r = { u32(e), u32(e + 4), u32(e + 8), u32(e + 12), u32(e + 16), u64(e + 24), u64(e + 32) };
        if (r.id >= 64 || r.group > 2) { fprintf(stderr, "bad record\n"); exit(1); }
        T.rec[r.id] = r;
    }
    const char *names[3] = { "common.bin", "kda.bin", "dense.bin" };
    for (int g = 0; g < 3; g++) {
        snprintf(path, sizeof path, "%s/%s", dir, names[g]);
        std::vector<unsigned char> bytes = slurp(path);
        CHECK(cudaMalloc(&T.group[g], bytes.size()));
        CHECK(cudaMemcpy(T.group[g], bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
    snprintf(path, sizeof path, "%s/operator.bin", dir);
    std::vector<unsigned char> op = slurp(path);
    CHECK(cudaMalloc(&T.op, op.size()));
    CHECK(cudaMemcpy(T.op, op.data(), op.size(), cudaMemcpyHostToDevice));
}

static Shape record_shape(const Trunk &T, unsigned id, const char *name)
{
    const Record &r = T.rec[id];
    const unsigned char *d = rec_data(T, id);
    return Shape{ name, d + (size_t)r.rows * 4, r.cols, d, 4, (int)r.rows, (int)r.cols };
}
static Shape qkv_shape(const Trunk &T, int comp, const char *name)
{
    return Shape{ name, T.op + QKV_PREFIX + (size_t)comp * QKV_COMP + 20, (size_t)3 * QKV_COMP, T.op + QKV_PREFIX + (size_t)comp * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, W };
}

__global__ void fill_random(float *p, size_t n, unsigned seed)   // roughly bell-shaped values, a few large ones, like activations
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    unsigned y = x * 747796405u + 2891336453u; y ^= y >> 15;
    const float u = ((x & 0xffff) + (y & 0xffff) - 65535) / 65536.0f;
    p[i] = u * ((x >> 28) == 0 ? 8.0f : 1.0f);
}

static float time_kernel(cudaEvent_t a, cudaEvent_t b, int reps, void (*launch)(void *), void *arg)
{
    launch(arg); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) launch(arg);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

static void compare_floats(const std::vector<float> &x, const std::vector<float> &ref, size_t *same, double *maxabs, double *rel_rms)
{
    size_t s = 0; double mx = 0, d2 = 0, r2 = 0;
    for (size_t i = 0; i < x.size(); i++) {
        s += memcmp(&x[i], &ref[i], 4) == 0;
        const double d = (double)x[i] - (double)ref[i];
        if (fabs(d) > mx) mx = fabs(d);
        d2 += d * d; r2 += (double)ref[i] * ref[i];
    }
    *same = s; *maxabs = mx; *rel_rms = sqrt(d2 / (r2 + 1e-300));
}
