struct HostCache {
    SlotCacheSel c; size_t cap_slots = 0; bool ready = false;
    enum { CHUNK_SLOTS = 32 };
    void init(size_t bytes)
    {
        cap_slots = bytes / EXPERT_RAW; ready = true;
        const size_t chunk_bytes = (size_t)CHUNK_SLOTS * EXPERT_RAW, n_chunks = cap_slots / CHUNK_SLOTS;
        std::vector<unsigned char *> base(n_chunks, nullptr);
        std::atomic<size_t> next(0);
        const double t0 = wall_now_();
        auto touch = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= n_chunks) return;
                void *p = mmap(nullptr, chunk_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) { fprintf(stderr, "host expert cache: cannot map %zu bytes\n", chunk_bytes); exit(1); }
                madvise(p, chunk_bytes, MADV_HUGEPAGE);
                memset(p, 0, chunk_bytes);
                base[i] = (unsigned char *)p;
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < 16; t++) th.emplace_back(touch);
        touch();
        for (auto &t : th) t.join();
        CHECK(cudaSetDevice(0));
        for (size_t i = 0; i < n_chunks; i++) CHECK(cudaHostRegister(base[i], chunk_bytes, cudaHostRegisterPortable));
        for (size_t i = n_chunks; i-- > 0;)
            for (int s = CHUNK_SLOTS; s-- > 0;) c.add_spare(base[i] + (size_t)s * EXPERT_RAW);
        cap_slots = n_chunks * CHUNK_SLOTS;
        fprintf(stderr, "K3GPU: host expert cache of %zu slots (%.1f GB) pinned in %.1f s\n", cap_slots, cap_slots * (double)EXPERT_RAW / 1e9, wall_now_() - t0);
    }
};

struct ExpertStats { unsigned long needed = 0, hit_card = 0, hit_host = 0, from_drive = 0, written_back = 0; double t_read = 0, t_copy = 0, t_ensure = 0; };

enum { MAX_READERS = 32 };
struct CardPool {
    int dev = 0, slots = 0, readers = 0; bool ready = false;
    SlotCacheSel c;
    unsigned char *base = nullptr;
    cudaStream_t copy = nullptr;                       // host RAM <-> card traffic of the main thread
    cudaStream_t rstream[MAX_READERS];                 // one per reading thread
    unsigned char *bounce[MAX_READERS];                // one pinned slot per reading thread: the drive's bytes land here and go on to the card
    cudaEvent_t written_back = nullptr;
    const unsigned char **d_eptr = nullptr;
    std::vector<const unsigned char *> h_eptr;
    ExpertStats st;
    void init(int device, size_t reserve_bytes, int nreaders)
    {
        dev = device; readers = std::min(nreaders, (int)MAX_READERS);
        CHECK(cudaSetDevice(dev));
        size_t f, t; CHECK(cudaMemGetInfo(&f, &t));
        const size_t usable = f > reserve_bytes ? f - reserve_bytes : 0;
        slots = (int)(usable / EXPERT_RAW);
        if (slots < 64) { fprintf(stderr, "card %d: only %d expert slots fit after the reserve\n", dev, slots); exit(1); }
        CHECK(cudaMalloc(&base, (size_t)slots * EXPERT_RAW));
        for (int i = slots; i-- > 0;) c.add_spare(base + (size_t)i * EXPERT_RAW);
        CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (int i = 0; i < readers; i++) {
            CHECK(cudaStreamCreateWithFlags(&rstream[i], cudaStreamNonBlocking));
            CHECK(cudaHostAlloc((void **)&bounce[i], EXPERT_RAW, cudaHostAllocDefault));
        }
        CHECK(cudaEventCreateWithFlags(&written_back, cudaEventDisableTiming));
        CHECK(cudaMalloc(&d_eptr, EXPERTS * sizeof(void *)));
        h_eptr.assign(EXPERTS, nullptr);
        ready = true;
    }
};
