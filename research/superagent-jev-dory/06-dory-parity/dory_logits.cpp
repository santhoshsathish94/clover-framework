// Final-position logits from the llama.cpp Dory implementation (PR #30306) for given token ids.
// usage: dory_logits <model.gguf> <num_loops> <per_loop|last_loop> <threads> <ids.txt> <out.f32>
// ids.txt: one prompt per line, space-separated token ids. out.f32: n_prompts x n_vocab float32, row-major.
// Each prompt is decoded as ONE batch from an empty cache, so prefill chunking cannot differ between prompts.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s model.gguf num_loops per_loop|last_loop threads ids.txt out.f32\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const long loops = strtol(argv[2], nullptr, 10);
    const std::string mode = argv[3];
    const int threads = atoi(argv[4]);

    std::vector<std::vector<llama_token>> prompts;
    size_t longest = 0;
    {
        std::ifstream in(argv[5]);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::vector<llama_token> ids;
            long v;
            while (ss >> v) ids.push_back((llama_token) v);
            if (!ids.empty()) {
                longest = ids.size() > longest ? ids.size() : longest;
                prompts.push_back(ids);
            }
        }
    }
    if (prompts.empty()) { fprintf(stderr, "no prompts\n"); return 2; }

    llama_model_kv_override ov[3];
    memset(ov, 0, sizeof(ov));
    ov[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(ov[0].key, sizeof(ov[0].key), "dory.num_loops");
    ov[0].val_i64 = loops;
    ov[1].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    snprintf(ov[1].key, sizeof(ov[1].key), "dory.recurrent_kv_cache_mode");
    snprintf(ov[1].val_str, sizeof(ov[1].val_str), "%s", mode.c_str());
    ov[2].key[0] = 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.kv_overrides = ov;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    const uint32_t n = (uint32_t) (longest + 16);
    cp.n_ctx = n < 4096 ? 4096 : n;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * out = fopen(argv[6], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[6]); return 1; }
    for (size_t i = 0; i < prompts.size(); ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(prompts[i].data(), (int32_t) prompts[i].size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed on prompt %zu\n", i); return 1; }
        const float * logits = llama_get_logits_ith(ctx, -1);
        fwrite(logits, sizeof(float), (size_t) n_vocab, out);
        fprintf(stderr, "prompt %zu tokens %zu done\n", i, prompts[i].size());
    }
    fclose(out);
    printf("n_prompts=%zu n_vocab=%d loops=%ld mode=%s\n", prompts.size(), n_vocab, loops, mode.c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
