struct CallSet { int layer; std::vector<int> experts; std::vector<int> counts; };
static std::vector<CallSet> g_rc;
static size_t g_rc_pos = 0, g_rc_planned = 0, g_pend_cap = 0;
static long g_rc_cur = -1;   // number of the call being processed
static std::unordered_map<int, std::vector<int>> g_uses;   // expert key -> numbers of the calls that need it
static std::unordered_map<int, size_t> g_use_ptr;
// next call after the current one that needs the expert (policy 2), or a huge number
static int rc_nu(int key)
{
    auto it = g_uses.find(key);
    if (it == g_uses.end()) return 1000000000;
    const std::vector<int> &l = it->second; size_t &p = g_use_ptr[key];
    while (p < l.size() && l[p] <= g_rc_cur) p++;
    return p < l.size() ? l[p] : 1000000000;
}
static void rc_load(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read route cache %s\n", path); exit(1); }
    char *line = nullptr; size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        char *p = line; char *end;
        CallSet cs; cs.layer = (int)strtol(p, &end, 10); p = end; strtol(p, &end, 10); p = end;
        for (;;) {
            while (*p == ' ') p++;
            if (*p < '0' || *p > '9') break;
            cs.experts.push_back((int)strtol(p, &end, 10)); p = end;
            if (*p == ':') { cs.counts.push_back((int)strtol(p + 1, &end, 10)); p = end; }
        }
        g_rc.push_back(std::move(cs));
    }
    free(line); fclose(f);
    for (size_t i = 0; i < g_rc.size(); i++) for (int e : g_rc[i].experts) g_uses[ekey(g_rc[i].layer, e)].push_back((int)i);
}
