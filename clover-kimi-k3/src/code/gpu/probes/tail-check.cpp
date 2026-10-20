// End-to-end check of a GPU run: the next token the CPU tail (final norm and language-model head) predicts from the GPU run's last-layer rows, against the same from the pod's
// recorded rows, and against the token that really follows in the prompt, for a list of positions.
//
//   tail-check BUILD_DIR OPEN_WEIGHTS_DIR REFS_DIR INPUTS_DIR TOKENS_FILE FIRST STEP COUNT
//
// BUILD_DIR      holds pipeline-stage.so            OPEN_WEIGHTS_DIR  the open weights (lm head, tokenizer)
// REFS_DIR       embed.f32, final.snaps.f32 (8 snapshots), final.resid.f32 of the pod (N x 7168 floats each)
// INPUTS_DIR     a MODEL_GPU_DUMP_DIR of a full run: snap-1..7.f32 and in-93.f32 (the output of layer 92)
// TOKENS_FILE    the prompt's token ids (any separators)
// Positions FIRST, FIRST+STEP, ... (COUNT of them). Reads only. build: g++ -O2 -std=c++14 -o tail-check tail-check.cpp -ldl
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

typedef struct { const unsigned char *bytes; size_t length; } Text;
static const int W = 7168;

static std::vector<float> slurp_f32(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t)n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read of %s\n", path.c_str()); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char **argv)
{
    if (argc < 9) { fprintf(stderr, "usage: tail-check BUILD_DIR OPEN_WEIGHTS_DIR REFS_DIR INPUTS_DIR TOKENS_FILE FIRST STEP COUNT\n"); return 2; }
    const std::string build = argv[1], ow = argv[2], refs = argv[3], inputs = argv[4];
    const int first = atoi(argv[6]), step = atoi(argv[7]), count = atoi(argv[8]);
    std::vector<float> embed = slurp_f32(refs + "/embed.f32"), pod_resid = slurp_f32(refs + "/final.resid.f32"), pod_snaps = slurp_f32(refs + "/final.snaps.f32");
    std::vector<float> gpu_out = slurp_f32(inputs + "/in-93.f32");
    std::vector<std::vector<float>> gpu_snap(8);
    for (int k = 1; k < 8; k++) gpu_snap[k] = slurp_f32(inputs + "/snap-" + std::to_string(k) + ".f32");
    const size_t N = embed.size() / W;
    if (gpu_out.size() != N * W || pod_resid.size() < N * W || pod_snaps.size() < N * W * 8) { fprintf(stderr, "row files do not agree on the window size\n"); return 1; }
    std::vector<unsigned long> ids;
    { FILE *f = fopen(argv[5], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[5]); return 1; }
      std::string s; char buf[4096]; size_t k; while ((k = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k); fclose(f);
      const char *p = s.c_str(); while (*p) { if (*p >= '0' && *p <= '9') { char *e; ids.push_back(strtoul(p, &e, 10)); p = e; } else p++; } }

    void *tail = dlopen((build + "/pipeline-stage.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!tail) { fprintf(stderr, "tail dlopen: %s\n", dlerror()); return 1; }
    int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) = (int (*)(const char *, const char *, const char *, const char *, const char *, void **))dlsym(tail, "transformer_93_open");
    void *(*tail_make)(const void *) = (void *(*)(const void *))dlsym(tail, "transformer_93_sequence_create");
    int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *) = (int (*)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *))dlsym(tail, "transformer_93_process");
    if (!tail_open || !tail_make || !tail_run) { fprintf(stderr, "tail symbols missing\n"); return 1; }
    const std::string leaves = ow + "/model-00094-of-000096.safetensors", head = leaves + "#language_model.lm_head.weight", tik = ow + "/tiktoken.model", config = ow + "/tokenizer_config.json";
    void *model = NULL;
    if (!tail_open(leaves.c_str(), head.c_str(), tik.c_str(), NULL, config.c_str(), &model)) { fprintf(stderr, "tail open failed\n"); return 1; }

    int n = 0, pod_gpu_same = 0, pod_true = 0, gpu_true = 0, with_truth = 0;
    printf("window %zu rows, %zu prompt ids\n position  pod  gpu  prompt's next id\n", N, ids.size());
    for (int i = 0; i < count; i++) {
        const int pos = first + i * step;
        if (pos < 0 || (size_t)pos >= N) break;
        std::vector<float> ps((size_t)8 * W), gs((size_t)8 * W);
        for (int k = 0; k < 8; k++) {
            memcpy(&ps[(size_t)k * W], &pod_snaps[((size_t)k * N + pos) * W], W * 4);
            const float *g = k == 0 ? &embed[(size_t)pos * W] : &gpu_snap[k][(size_t)pos * W];
            memcpy(&gs[(size_t)k * W], g, W * 4);
        }
        uint32_t tp = 0, tg = 0; Text w = { NULL, 0 };
        void *seq1 = tail_make(model), *seq2 = tail_make(model);
        if (!seq1 || !seq2 || !tail_run(model, seq1, &pod_resid[(size_t)pos * W], ps.data(), 8, 1, "", &tp, &w)) { fprintf(stderr, "tail failed (pod, position %d)\n", pos); return 1; }
        if (!tail_run(model, seq2, &gpu_out[(size_t)pos * W], gs.data(), 8, 1, "", &tg, &w)) { fprintf(stderr, "tail failed (gpu, position %d)\n", pos); return 1; }
        const bool truth = (size_t)pos + 1 < ids.size();
        printf(" %5d  %6u  %6u  %6lu%s\n", pos, tp, tg, truth ? ids[pos + 1] : 0ul, tp == tg ? "" : "   <- differ");
        n++; pod_gpu_same += tp == tg;
        if (truth) { with_truth++; pod_true += tp == ids[pos + 1]; gpu_true += tg == ids[pos + 1]; }
    }
    printf("%d positions: GPU and pod predict the same next id at %d (%.1f%%); against the prompt's real next id: pod %d of %d, GPU %d of %d\n", n, pod_gpu_same, n ? 100.0 * pod_gpu_same / n : 0.0, pod_true, with_truth, gpu_true, with_truth);
    return 0;
}
