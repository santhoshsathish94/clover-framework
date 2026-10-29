// runs the jobs on up to nthreads threads; a job is told the number of the thread that runs it
static void run_parallel(std::vector<std::function<void(int)>> &jobs, int nthreads)
{
    if (jobs.empty()) return;
    std::atomic<size_t> next(0);
    auto work = [&](int tid) { for (;;) { const size_t i = next.fetch_add(1); if (i >= jobs.size()) return; jobs[i](tid); } };
    const int n = std::min<int>(nthreads, (int)jobs.size());
    std::vector<std::thread> th;
    for (int i = 1; i < n; i++) th.emplace_back(work, i);
    work(0);
    for (auto &t : th) t.join();
}
static double wall_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static unsigned long g_wb_skipped = 0;   // policy 2: experts a card gave up that were not worth a RAM slot (needed later than everything held there)
// makes the given experts of `layer` resident in the card's pool; afterwards pool.h_eptr[e] is the address of expert e
// Where an expert comes from: the card's pool (hit); else the RAM cache, which is taken out of it (the expert moves up to the card); else the drive, read
// straight into a small pinned slot of the reading thread and on to the card (it is NOT kept in RAM). The RAM cache holds what the cards have let go: an expert
// that a card replaces is copied down into it first, so the card pool and the RAM cache never hold the same expert and their sizes add up.
// `counts[e]` is the number of tokens of this call that chose expert e (the heat of the expert).
static void pool_ensure(CardPool &pool, HostCache &hc, int layer, int fd, const std::vector<int> &experts, const int *counts, int readers)
{
    struct Timed { double &acc; double t0; Timed(double &a) : acc(a), t0(wall_now()) {} ~Timed() { acc += wall_now() - t0; } } timed(pool.st.t_ensure);
    if (g_vq) CHECK(cudaStreamSynchronize(0)); else CHECK(cudaDeviceSynchronize());   // kernels of an earlier group or layer may still be reading slots that are about to be replaced (with the queue on, copies of other streams are not waited for)
    std::unordered_set<int> prot, noprot;
    for (int e : experts) prot.insert(ekey(layer, e));
    struct Need { int e; unsigned char *dst, *host; };
    struct Victim { int key; float heat; unsigned char *src; };
    std::vector<Need> need; std::vector<Victim> victims;
    pool.st.needed += experts.size();
    for (int e : experts) {
        const int key = ekey(layer, e);
        unsigned char *p = pool.c.find(key);
        if (p) { if (g_vq) arr_wait(key); pool.c.touch(key, (float)counts[e]); pool.h_eptr[e] = p; pool.st.hit_card++; continue; }
        int vk; float vh;
        p = pool.c.acquire(key, prot, (float)counts[e], &vk, &vh);
        if (vk >= 0) { if (g_vq) arr_wait(vk); victims.push_back(Victim{ vk, vh, p }); }
        pool.h_eptr[e] = p; need.push_back(Need{ e, p, nullptr });
    }
    if (need.empty()) return;
    // 1. experts found in RAM leave the RAM cache; their slots are held until the copies are done, so nothing is written over them
    std::vector<unsigned char *> held;
    for (auto &n : need) {
        const int key = ekey(layer, n.e);
        if (hc.c.find(key)) { pend_take(key); n.host = hc.c.remove(key); held.push_back(n.host); pool.st.hit_host++; }
    }
    CHECK(cudaSetDevice(pool.dev));
    // 2. the experts the card gives up are copied down into the RAM cache (if the cache is full its least useful entry is dropped)
    for (auto &v : victims) {
        if (!hc.cap_slots || g_prefill) break;
        if (!hc.c.worth(v.key)) { g_wb_skipped++; continue; }
        int dk; float dh;
        unsigned char *h = hc.c.acquire(v.key, g_pend_keys, v.heat, &dk, &dh);
        CHECK(cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy));
        pool.st.written_back++;
    }
    CHECK(cudaEventRecord(pool.written_back, pool.copy));   // from here on a slot of the pool can be overwritten
    // 3. RAM -> card, queued behind the copies down on the same stream
    for (auto &n : need) if (n.host) CHECK(cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy));
    // 4. drive -> pinned slot -> card, one expert at a time per reading thread; the reads start at once, the copy to the card waits for the copies down
    std::vector<std::function<void(int)>> jobs;
    const int dev = pool.dev;
    int n_drive = 0;
    for (auto &n : need) if (!n.host) n_drive++;
    // a call with few experts cannot fill the drive with whole-expert reads (each takes ~10 ms alone), so each expert is read in chunks by several threads
    const int pool_readers = std::min(readers, pool.readers);
    int nchunk = g_chunks > 0 ? g_chunks : (n_drive >= pool_readers || n_drive == 0) ? 1 : std::min(8, (pool_readers + n_drive - 1) / n_drive);
    const size_t chunk = ((((size_t)EXPERT_RAW + nchunk - 1) / nchunk) + 4095) / 4096 * 4096;
    for (auto &n : need) {
        if (n.host) continue;
        pool.st.from_drive++;
        for (int k = 0; k < nchunk; k++) {
            const size_t co = (size_t)k * chunk; if (co >= (size_t)EXPERT_RAW) break;
            const size_t len = std::min(chunk, (size_t)EXPERT_RAW - co);
            unsigned char *dev_dst = n.dst + co; const off_t off = (off_t)n.e * EXPERT_RAW + (off_t)co; const int e = n.e;
            jobs.push_back([=, &pool](int tid) {
                unsigned char *b = pool.bounce[tid];
                if (g_prefill) b = g_slab[layer] + (size_t)(e & 3) * EXPERT_RAW + co;
                else {
                size_t got = 0;
                while (got < len) {
                    const ssize_t r = pread(fd, b + got, len - got, off + (off_t)got);
                    if (r <= 0) { fprintf(stderr, "reading expert %d of layer %d failed: %s\n", e, layer, r < 0 ? strerror(errno) : "end of file"); exit(1); }
                    got += (size_t)r;
                }
                }
                CHECK(cudaSetDevice(dev));
                CHECK(cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0));
                CHECK(cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid]));
                CHECK(cudaStreamSynchronize(pool.rstream[tid]));   // the pinned slot is free for this thread's next read
            });
        }
    }
    const double t0 = wall_now();
    run_parallel(jobs, pool_readers);
    const double t1 = wall_now();
    CHECK(cudaStreamSynchronize(pool.copy));
    for (unsigned char *h : held) hc.c.release(h);
    pool.st.t_read += t1 - t0; pool.st.t_copy += wall_now() - t1;
}
