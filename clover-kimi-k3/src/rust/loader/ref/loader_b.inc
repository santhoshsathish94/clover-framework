static Handle *open_layer(const char *dataset, unsigned L, unsigned k, unsigned positions, Handle *into)
{
    configure();
    Handle *h = into ? into : new Handle;
    h->L = (int)L; h->card = card_of((int)L); h->k_in = (int)k;
    if (L <= (unsigned)N_LAYERS) g_handles[L] = h;
    const std::string ds(dataset);
    loader_submit(h->card, [h, ds, L, positions]() {
        CHECK(cudaSetDevice(h->card));
        ensure_work(h->card);
        layer_load_trunk(ds.c_str(), (int)L, h->y);
        if (L >= 1) layer_load_expert_tables(ds.c_str(), (int)L, h->y);
        loader_state_alloc(h, positions);   // ADAPTED (6): replaces the five lines h->mla = h->y.mla ... kda_state_alloc(h->kda); see NOTES.md
        if (L >= 1) {
            char path[4096]; snprintf(path, sizeof path, "%s/expert-%u/experts.direct", ds.c_str(), L);
            h->fd = open(path, O_RDONLY | O_DIRECT);
            if (h->fd < 0) h->fd = open(path, O_RDONLY);
            if (h->fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
            if (g_prefill) {   // 4 real records of this layer stand in for every expert of it
                CHECK(cudaHostAlloc((void **)&g_slab[L], (size_t)4 * EXPERT_RAW, cudaHostAllocDefault));
                for (int r = 0; r < 4; r++) {
                    size_t got = 0;
                    while (got < (size_t)EXPERT_RAW) {
                        const ssize_t rd = pread(h->fd, g_slab[L] + (size_t)r * EXPERT_RAW + got, EXPERT_RAW - got, (off_t)r * EXPERT_RAW + (off_t)got);
                        if (rd <= 0) { fprintf(stderr, "prefill: cannot read a record of layer %u\n", L); exit(1); }
                        got += (size_t)rd;
                    }
                }
            }
        }
    });
    return h;
}
