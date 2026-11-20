/*
 * Comprehensive Exactness, Concurrency & Throughput Suite for Clover's
 * x-cruncher-Accelerated LFM3 Native C Engine (`lfm3_cpu_engine.c`).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
  #include <windows.h>
  static double bench_now_ms(void) {
      LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f);
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
  }
#else
  #include <pthread.h>
  static double bench_now_ms(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
  }
#endif

extern const char *lfm3_active_isa_name(void);
extern int lfm3_last_kernel_tier(void);
extern int lfm3_set_num_threads(int n_threads);
extern const int8_t *lfm3_get_preset_lut16(int preset_id);
extern void lfm3_linear_bf16_direct(const float *x, const uint16_t *W_bf16, float *y, int T, int out_dim, int in_dim);
extern void lfm3_linear_4bit_group64_direct(const float *x, const uint8_t *nib, const uint8_t *stp, const float *scl, float *y, int T, int out_dim, int in_dim);
extern int lfm3_linear_4bit_lut16_omp(const float *x, const uint8_t *nib, const int8_t *lut16_s8, const float *grp_scl, float *y, int T, int out_dim, int in_dim);
extern int lfm3_compute_log_mel_f32(const float *pcm, int n_samples, int sample_rate, int n_fft, int win_length, int hop_length, int n_mels, float fmin_hz, float fmax_hz, float *out_mel, int max_frames);

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else      { g_fail++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (*s = x);
}

static uint16_t f32_to_bf16_test(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    uint32_t rb = 0x00007FFFU + ((v.u >> 16) & 1U);
    return (uint16_t)((v.u + rb) >> 16);
}

static float bf16_to_f32_test(uint16_t b) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na * nb) + 1e-30);
}

typedef struct {
    int tid;
    const float *x;
    const uint8_t *nib;
    const int8_t *lut;
    const float *gscl;
    const float *ref;
    int out_dim;
    int in_dim;
    int ok;
} mt_lut_arg_t;

#ifdef _WIN32
static DWORD WINAPI mt_lut_worker(LPVOID p)
#else
static void *mt_lut_worker(void *p)
#endif
{
    mt_lut_arg_t *a = (mt_lut_arg_t *)p;
    float *y = (float *)malloc((size_t)a->out_dim * sizeof(float));
    int ok = (y != NULL);
    for (int it = 0; it < 6 && ok; it++) {
        if (lfm3_linear_4bit_lut16_omp(a->x, a->nib, a->lut, a->gscl, y, 1, a->out_dim, a->in_dim) != 0) ok = 0;
        if (cosine_sim(y, a->ref, (size_t)a->out_dim) < 0.9995) ok = 0;
        if (!lfm3_active_isa_name() || lfm3_last_kernel_tier() < 0) ok = 0;
    }
    free(y);
    a->ok = ok;
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

int main(void) {
    int nt = lfm3_set_num_threads(0);
    printf("========================================================================\n");
    printf(" Clover x-cruncher LFM3 Engine Suite | Active ISA: %s | Threads: %d\n",
           lfm3_active_isa_name(), nt);
    printf("========================================================================\n");

    uint64_t rng = 0xC10FE2161ULL;

    /* 1. BF16 Linear Exactness (T=5, out=256, in=2048) */
    {
        const int T = 5, out_d = 256, in_d = 2048;
        float *x = (float *)malloc((size_t)T * in_d * sizeof(float));
        uint16_t *W = (uint16_t *)malloc((size_t)out_d * in_d * sizeof(uint16_t));
        float *y = (float *)malloc((size_t)T * out_d * sizeof(float));
        float *ref = (float *)malloc((size_t)T * out_d * sizeof(float));
        for (int i = 0; i < T * in_d; i++) x[i] = ((float)(int32_t)(xorshift64(&rng) & 0xFFFF) - 32768.0f) / 16384.0f;
        for (int i = 0; i < out_d * in_d; i++) W[i] = f32_to_bf16_test(((float)(int32_t)(xorshift64(&rng) & 0xFFFF) - 32768.0f) / 32768.0f);

        for (int o = 0; o < out_d; o++) {
            for (int t = 0; t < T; t++) {
                float s = 0.0f;
                for (int k = 0; k < in_d; k++) s += x[t * in_d + k] * bf16_to_f32_test(W[(size_t)o * in_d + k]);
                ref[t * out_d + o] = bf16_to_f32_test(f32_to_bf16_test(s));
            }
        }
        lfm3_linear_bf16_direct(x, W, y, T, out_d, in_d);
        double cs = cosine_sim(y, ref, (size_t)T * out_d);
        free(x); free(W); free(y); free(ref);
        CHECK(cs > 0.999999, "Runtime-dispatched BF16 Linear (T=5, 256x2048) matches scalar reference (cos > 0.999999)");
    }

    /* 2. 4-Bit Planar Group-64 Linear + Vectorized Int8 Quantizer + Cached Scratch */
    {
        const int T = 5, out_d = 512, in_d = 2048, groups = in_d / 64;
        float *x = (float *)malloc((size_t)T * in_d * sizeof(float));
        uint8_t *nib = (uint8_t *)malloc((size_t)out_d * groups * 32);
        uint8_t *stp = (uint8_t *)malloc((size_t)out_d * groups);
        float *scl = (float *)malloc((size_t)out_d * sizeof(float));
        float *y = (float *)malloc((size_t)T * out_d * sizeof(float));
        float *ref = (float *)malloc((size_t)T * out_d * sizeof(float));

        for (int i = 0; i < T * in_d; i++) x[i] = ((float)(int32_t)(xorshift64(&rng) & 0xFFFF) - 32768.0f) / 8192.0f;
        for (size_t i = 0; i < (size_t)out_d * groups * 32; i++) nib[i] = (uint8_t)(xorshift64(&rng) & 0xFF);
        for (int i = 0; i < out_d * groups; i++) stp[i] = (uint8_t)(1 + (xorshift64(&rng) % 15));
        for (int o = 0; o < out_d; o++) scl[o] = 0.002f + 0.0001f * (float)(o & 31);

        for (int o = 0; o < out_d; o++) {
            for (int t = 0; t < T; t++) {
                float sum = 0.0f;
                for (int g = 0; g < groups; g++) {
                    const uint8_t *nb = nib + ((size_t)o * groups + g) * 32;
                    float gsc = (float)stp[o * groups + g] * scl[o];
                    for (int j = 0; j < 32; j++) {
                        uint8_t pk = nb[j];
                        sum += x[t * in_d + g * 64 + j] * ((float)((int)(pk & 0x0F) - 8) * gsc)
                             + x[t * in_d + g * 64 + 32 + j] * ((float)((int)(pk >> 4) - 8) * gsc);
                    }
                }
                ref[t * out_d + o] = bf16_to_f32_test(f32_to_bf16_test(sum));
            }
        }
        lfm3_linear_4bit_group64_direct(x, nib, stp, scl, y, T, out_d, in_d);
        double cs = cosine_sim(y, ref, (size_t)T * out_d);
        free(x); free(nib); free(stp); free(scl); free(y); free(ref);
        CHECK(cs > 0.9999, "Runtime-dispatched 4-Bit Group-64 Linear + VecQuant (T=5, 512x2048, cos > 0.9999)");
    }

    /* 3. 1-Cycle vpermb / vpshufb / vqtbl1q LUT16 Codebook (MXFP4 E2M1, NF4, Vinculum) + Multi-Thread Concurrency */
    {
        const int out_d = 384, in_d = 2048, groups = in_d / 64;
        float *x = (float *)malloc((size_t)in_d * sizeof(float));
        uint8_t *nib = (uint8_t *)malloc((size_t)out_d * groups * 32);
        float *gscl = (float *)malloc((size_t)out_d * groups * sizeof(float));
        float *y = (float *)malloc((size_t)out_d * sizeof(float));
        float *ref = (float *)malloc((size_t)out_d * sizeof(float));

        for (int i = 0; i < in_d; i++) x[i] = ((float)(int32_t)(xorshift64(&rng) & 0xFFFF) - 32768.0f) / 16384.0f;
        for (size_t i = 0; i < (size_t)out_d * groups * 32; i++) nib[i] = (uint8_t)(xorshift64(&rng) & 0xFF);
        for (int i = 0; i < out_d * groups; i++) gscl[i] = 0.01f + 0.001f * (float)(i & 15);

        int all_presets_ok = 1;
        for (int pid = 0; pid < 3; pid++) {
            const int8_t *lut = lfm3_get_preset_lut16(pid);
            for (int o = 0; o < out_d; o++) {
                float sum = 0.0f;
                for (int g = 0; g < groups; g++) {
                    const uint8_t *nb = nib + ((size_t)o * groups + g) * 32;
                    float sc = gscl[o * groups + g];
                    for (int j = 0; j < 32; j++) {
                        uint8_t pk = nb[j];
                        sum += x[g * 64 + j] * ((float)lut[pk & 0x0F] * sc)
                             + x[g * 64 + 32 + j] * ((float)lut[pk >> 4] * sc);
                    }
                }
                ref[o] = sum;
            }
            int rc = lfm3_linear_4bit_lut16_omp(x, nib, lut, gscl, y, 1, out_d, in_d);
            double cs = cosine_sim(y, ref, (size_t)out_d);
            if (rc != 0 || cs < 0.9999) all_presets_ok = 0;
        }
        CHECK(all_presets_ok, "1-Cycle Register LUT16 Codebook MatVec (MXFP4 E2M1, NF4, Vinculum, cos > 0.9999)");

        /* Concurrent 4-thread caller race check (uses NF4 preset = pid 1, matching `ref`? Recompute ref for pid=1!) */
        const int8_t *nf4 = lfm3_get_preset_lut16(1);
        for (int o = 0; o < out_d; o++) {
            float sum = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib + ((size_t)o * groups + g) * 32;
                float sc = gscl[o * groups + g];
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = nb[j];
                    sum += x[g * 64 + j] * ((float)nf4[pk & 0x0F] * sc)
                         + x[g * 64 + 32 + j] * ((float)nf4[pk >> 4] * sc);
                }
            }
            ref[o] = sum;
        }
        mt_lut_arg_t mta[4];
#ifdef _WIN32
        HANDLE th[4];
        for (int t = 0; t < 4; t++) {
            mta[t] = (mt_lut_arg_t){t, x, nib, nf4, gscl, ref, out_d, in_d, 0};
            th[t] = CreateThread(NULL, 0, mt_lut_worker, &mta[t], 0, NULL);
        }
        WaitForMultipleObjects(4, th, TRUE, INFINITE);
        for (int t = 0; t < 4; t++) CloseHandle(th[t]);
#else
        pthread_t th[4];
        for (int t = 0; t < 4; t++) {
            mta[t] = (mt_lut_arg_t){t, x, nib, nf4, gscl, ref, out_d, in_d, 0};
            pthread_create(&th[t], NULL, mt_lut_worker, &mta[t]);
        }
        for (int t = 0; t < 4; t++) pthread_join(th[t], NULL);
#endif
        CHECK(mta[0].ok && mta[1].ok && mta[2].ok && mta[3].ok,
              "Concurrent 4-thread LUT16 + cached qscratch + thread-local tier race-free");

        free(x); free(nib); free(gscl); free(y); free(ref);
    }

    /* 4. Zero-Dependency C11 Radix-4 / Radix-2 STFT Log-Mel Spectrogram vs Textbook O(N^2) DFT */
    {
        const int sr = 16000, n_samples = 16000; /* 1.0s synthetic multi-tone speech-like signal */
        float *pcm = (float *)malloc((size_t)n_samples * sizeof(float));
        for (int i = 0; i < n_samples; i++) {
            double t = (double)i / (double)sr;
            pcm[i] = (float)(0.6 * sin(2.0 * 3.141592653589793 * 440.0 * t)
                           + 0.3 * cos(2.0 * 3.141592653589793 * 1760.0 * t)
                           + 0.1 * sin(2.0 * 3.141592653589793 * 3520.0 * t));
        }

        /* Test both even log2 (n_fft=256, pure Radix-4) and odd log2 (n_fft=512, Radix-2 + Radix-4)! */
        int fft_sizes[2] = {256, 512};
        int mel_ok = 1;
        for (int s = 0; s < 2; s++) {
            int n_fft = fft_sizes[s], win_len = (n_fft == 512) ? 400 : 256, hop = 160, n_mels = 80;
            float mel_fast[128 * 80], mel_ref[4 * 80];
            int nf = lfm3_compute_log_mel_f32(pcm, n_samples, sr, n_fft, win_len, hop, n_mels, 0.0f, 8000.0f, mel_fast, 4);
            if (nf != 4) { mel_ok = 0; break; }

            /* Independent O(N^2) textbook DFT reference for first 4 frames */
            int n_bins = n_fft / 2 + 1;
            double m_min = 2595.0 * log10(1.0 + 0.0 / 700.0);
            double m_max = 2595.0 * log10(1.0 + 8000.0 / 700.0);
            double hz_pts[82];
            for (int m = 0; m < n_mels + 2; m++) {
                double mel = m_min + (m_max - m_min) * (double)m / (double)(n_mels + 1);
                hz_pts[m] = 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
            }
            for (int fr = 0; fr < 4; fr++) {
                double pwr[257];
                for (int b = 0; b < n_bins; b++) {
                    double re = 0.0, im = 0.0;
                    for (int n = 0; n < win_len; n++) {
                        double w = 0.5 - 0.5 * cos(6.283185307179586476925286766559 * (double)n / (double)win_len);
                        double x = (double)pcm[fr * hop + n] * w;
                        double ang = -6.283185307179586476925286766559 * (double)b * (double)n / (double)n_fft;
                        re += x * cos(ang);
                        im += x * sin(ang);
                    }
                    pwr[b] = re * re + im * im;
                }
                for (int m = 0; m < n_mels; m++) {
                    double fl = hz_pts[m], fc = hz_pts[m + 1], fr_hz = hz_pts[m + 2];
                    double norm = 2.0 / (fr_hz - fl + 1e-12);
                    double e = 0.0;
                    for (int b = 0; b < n_bins; b++) {
                        double f = (double)b * (double)sr / (double)n_fft;
                        double wt = 0.0;
                        if (f >= fl && f <= fc && fc > fl) wt = (f - fl) / (fc - fl);
                        else if (f > fc && f <= fr_hz && fr_hz > fc) wt = (fr_hz - f) / (fr_hz - fc);
                        e += (wt * norm) * pwr[b];
                    }
                    if (e < 1e-10) e = 1e-10;
                    mel_ref[fr * n_mels + m] = (float)log(e);
                    float diff = fabsf(mel_fast[fr * n_mels + m] - mel_ref[fr * n_mels + m]);
                    if (diff > 5e-5f) mel_ok = 0;
                }
            }
        }
        /* Benchmark 10 seconds of 16 kHz audio (160,000 samples -> 998 frames x 80 mels) */
        float *pcm10 = (float *)malloc(160000 * sizeof(float));
        float *mel10 = (float *)malloc(1000 * 80 * sizeof(float));
        for (int i = 0; i < 160000; i++) pcm10[i] = pcm[i % n_samples];
        double t0 = bench_now_ms();
        int nf10 = lfm3_compute_log_mel_f32(pcm10, 160000, 16000, 512, 400, 160, 80, 0.0f, 8000.0f, mel10, 1000);
        double dt_mel = bench_now_ms() - t0;
        free(pcm10); free(mel10); free(pcm);
        CHECK(mel_ok && nf10 == 998, "Zero-dependency C11 Radix-4/2 STFT Log-Mel exact vs O(N^2) DFT (< 5e-5 err)");
        printf("  [BENCH] 10.0s 16 kHz Audio -> %d x 80 Log-Mel Frames: %.3f ms\n", nf10, dt_mel);
    }

    /* 5. Throughput Benchmark on Real LFM3 FFN Shape (7680 x 2048): Group-64 Q4 & LUT16 MXFP4 */
    {
        const int out_d = 7680, in_d = 2048, groups = in_d / 64;
        float *x = (float *)malloc((size_t)in_d * sizeof(float));
        uint8_t *nib = (uint8_t *)malloc((size_t)out_d * groups * 32);
        uint8_t *stp = (uint8_t *)malloc((size_t)out_d * groups);
        float *scl = (float *)malloc((size_t)out_d * sizeof(float));
        float *gscl = (float *)malloc((size_t)out_d * groups * sizeof(float));
        float *y = (float *)malloc((size_t)out_d * sizeof(float));
        for (int i = 0; i < in_d; i++) x[i] = 0.01f * (float)(i & 63);
        memset(nib, 0x95, (size_t)out_d * groups * 32);
        memset(stp, 3, (size_t)out_d * groups);
        for (int o = 0; o < out_d; o++) scl[o] = 0.005f;
        for (int i = 0; i < out_d * groups; i++) gscl[i] = 0.005f;

        lfm3_linear_4bit_group64_direct(x, nib, stp, scl, y, 1, out_d, in_d); /* warmup */
        double best_q4 = 1e9, best_lut = 1e9;
        for (int r = 0; r < 15; r++) {
            double t0 = bench_now_ms();
            lfm3_linear_4bit_group64_direct(x, nib, stp, scl, y, 1, out_d, in_d);
            double dt = bench_now_ms() - t0;
            if (dt < best_q4) best_q4 = dt;
        }
        const int8_t *mxfp4 = lfm3_get_preset_lut16(0);
        lfm3_linear_4bit_lut16_omp(x, nib, mxfp4, gscl, y, 1, out_d, in_d);
        for (int r = 0; r < 15; r++) {
            double t0 = bench_now_ms();
            lfm3_linear_4bit_lut16_omp(x, nib, mxfp4, gscl, y, 1, out_d, in_d);
            double dt = bench_now_ms() - t0;
            if (dt < best_lut) best_lut = dt;
        }
        printf("  [BENCH] LFM3 FFN 7680x2048 Q4 Group-64 : %.3f ms | LUT16 MXFP4 (vpermb/vpshufb): %.3f ms\n",
               best_q4, best_lut);
        free(x); free(nib); free(stp); free(scl); free(gscl); free(y);
    }

    printf("========================================================================\n");
    printf(" RESULT: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
