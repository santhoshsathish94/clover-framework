// Re-runs the study's single-case Dory claims on the llama.cpp PR implementation, with the study's chunking:
// the prompt is decoded as one batch, then each further token one at a time against the cache.
//   dory_claims score    <model.gguf> <loops> <per_loop|last_loop> <threads> <prompt.txt> <options.txt>
//       prints per option: the mean log-probability of its tokens (the study's score) and each token's log-probability
//   dory_claims generate <model.gguf> <loops> <per_loop|last_loop> <threads> <prompt.txt> <max_new> <eos,eos>
//       prints the greedy continuation's token ids
// prompt.txt: space-separated token ids; options.txt: one option per line, space-separated token ids.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::vector<llama_token>> read_lines(const char * path) {
    std::vector<std::vector<llama_token>> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::vector<llama_token> ids;
        long v;
        while (ss >> v) ids.push_back((llama_token) v);
        if (!ids.empty()) out.push_back(ids);
    }
    return out;
}

static double log_softmax_at(const float * logits, int n_vocab, llama_token t) {
    double m = logits[0];
    for (int i = 1; i < n_vocab; ++i) m = logits[i] > m ? logits[i] : m;
    double s = 0.0;
    for (int i = 0; i < n_vocab; ++i) s += std::exp((double) logits[i] - m);
    return (double) logits[t] - m - std::log(s);
}

static int decode_one(llama_context * ctx, llama_token tok, int pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1;
    b.token[0] = tok;
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see header\n"); return 2; }
    const std::string mode = argv[1];
    const long loops = strtol(argv[3], nullptr, 10);
    const std::string kv_mode = argv[4];
    const int threads = atoi(argv[5]);
    std::vector<llama_token> prompt = read_lines(argv[6]).at(0);

    llama_model_kv_override ov[3];
    memset(ov, 0, sizeof(ov));
    ov[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(ov[0].key, sizeof(ov[0].key), "dory.num_loops");
    ov[0].val_i64 = loops;
    ov[1].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    snprintf(ov[1].key, sizeof(ov[1].key), "dory.recurrent_kv_cache_mode");
    snprintf(ov[1].val_str, sizeof(ov[1].val_str), "%s", kv_mode.c_str());
    ov[2].key[0] = 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.kv_overrides = ov;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = 4096;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto prefill = [&]() -> const float * {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); exit(1); }
        return llama_get_logits_ith(ctx, -1);
    };

    if (mode == "score") {
        for (const auto & opt : read_lines(argv[7])) {
            const float * logits = prefill();
            double total = log_softmax_at(logits, n_vocab, opt[0]);
            std::string per = std::to_string(total);
            for (size_t j = 0; j + 1 < opt.size(); ++j) {
                if (decode_one(ctx, opt[j], (int) (prompt.size() + j)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                double lp = log_softmax_at(llama_get_logits_ith(ctx, -1), n_vocab, opt[j + 1]);
                total += lp;
                per += " " + std::to_string(lp);
            }
            printf("OPTION mean=%.6f tokens=%zu per=%s\n", total / (double) opt.size(), opt.size(), per.c_str());
        }
    } else if (mode == "generate") {
        const int max_new = atoi(argv[7]);
        std::set<llama_token> eos;
        if (argc > 8) {
            std::stringstream ss(argv[8]);
            std::string item;
            while (std::getline(ss, item, ',')) eos.insert((llama_token) atoi(item.c_str()));
        }
        const float * logits = prefill();
        std::string out;
        int pos = (int) prompt.size();
        for (int i = 0; i < max_new; ++i) {
            llama_token best = 0;
            for (int t = 1; t < n_vocab; ++t) best = logits[t] > logits[best] ? t : best;
            out += (out.empty() ? "" : " ") + std::to_string(best);
            if (eos.count(best)) break;
            if (decode_one(ctx, best, pos++) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            logits = llama_get_logits_ith(ctx, -1);
        }
        printf("GENERATED %s\n", out.c_str());
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
