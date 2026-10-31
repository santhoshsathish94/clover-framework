// Several files read into RAM from up to three sources at the same time, the chunks (64 MiB by default) handed out one at a time to whichever worker is free, so each
// source takes as much as it can carry: the local drive (O_DIRECT), Hugging Face (range requests through curl), and the other server (a forced-command ssh stream).
//
//   multi-read FILE_REL[,FILE_REL...] DISK_THREADS HF_THREADS K3_THREADS [CHUNK_MIB=64] [DATASET_DIR]
//   FILE_REL    e.g. expert-50/experts.direct,expert-51/experts.direct (the same relative path on every source); the files are one queue of chunks, in order
// Network workers stop taking chunks when only a few are left, so the end of the queue goes to the drive. Reported: the whole time, and, because a short run is held up by
// the network chunks still in flight when the drive is done, also what every source had delivered by the moment the drive finished (the sustained rate).
// The chunks that came over the network are compared with the local files afterwards. Reads only.
//
// build: g++ -O2 -std=c++14 -pthread -o multi-read multi-read.cpp
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool read_pipe(const std::string &cmd, unsigned char *dst, size_t bytes)
{
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) return false;
    size_t got = 0;
    while (got < bytes) { size_t k = fread(dst + got, 1, bytes - got, p); if (k == 0) break; got += k; }
    const int rc = pclose(p);
    return got == bytes && rc == 0;
}

struct File { std::string rel, local; size_t size, base; int fd; };
struct Chunk { int file; size_t off, n, base; };

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: multi-read FILE_REL[,FILE_REL...] DISK_THREADS HF_THREADS K3_THREADS [CHUNK_MIB] [DATASET_DIR]\n"); return 2; }
    const int nd = atoi(argv[2]), nh = atoi(argv[3]), nk = atoi(argv[4]);
    const size_t chunk = (size_t)(argc > 5 ? atoi(argv[5]) : 64) << 20;
    const std::string dir = argc > 6 ? argv[6] : "/opt/clover-k3/datasets/computed";
    std::vector<File> files; size_t total = 0;
    { std::stringstream ss(argv[1]); std::string r; while (std::getline(ss, r, ',')) {
        File f; f.rel = r; f.local = dir + "/" + r; struct stat st; if (stat(f.local.c_str(), &st)) { perror(f.local.c_str()); return 1; }
        f.size = (size_t)st.st_size; f.base = total; f.fd = open(f.local.c_str(), O_RDONLY | O_DIRECT); total += f.size; files.push_back(f); } }
    std::vector<Chunk> list;
    for (size_t fi = 0; fi < files.size(); fi++) for (size_t off = 0; off < files[fi].size; off += chunk) list.push_back({ (int)fi, off, files[fi].size - off < chunk ? files[fi].size - off : chunk, files[fi].base + off });
    unsigned char *buf = (unsigned char *)mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }
    const std::string base_url = "https://huggingface.co/datasets/Santhoshsathish94/clover-ai-studio/resolve/main/kimi/experts/";
    const std::string home = getenv("HOME") ? getenv("HOME") : ".";
    const std::string ssh = "ssh -i " + home + "/.ssh/k3_read -o BatchMode=yes -o IdentitiesOnly=yes -c aes128-gcm@openssh.com root@k3";
    const size_t chunks = list.size();
    std::atomic<size_t> next(0), left(chunks);
    std::vector<char> source(chunks, 'x'); std::vector<double> done_at(chunks, 0.0);
    std::atomic<size_t> bytes_by[3] = { {0}, {0}, {0} }; std::atomic<int> count_by[3] = { {0}, {0}, {0} };
    std::atomic<int> failures(0);
    std::atomic<int> disk_workers_left(nd); std::atomic<double> t_disk_done(0.0);
    const double t0 = now();
    auto worker = [&](int kind) {   // 0 disk, 1 hugging face, 2 k3
        for (;;) {
            if (kind != 0 && left.load() < 8) break;    // the end of the queue is the drive's
            const size_t i = next++;
            if (i >= chunks) break;
            const Chunk &c = list[i];
            bool ok;
            if (kind == 0) { size_t got = 0; ok = true; while (got < c.n) { ssize_t k = pread(files[c.file].fd, buf + c.base + got, c.n - got, (off_t)(c.off + got)); if (k <= 0) { ok = false; break; } got += (size_t)k; } }
            else if (kind == 1) ok = read_pipe("curl -sL --fail -r " + std::to_string(c.off) + "-" + std::to_string(c.off + c.n - 1) + " '" + base_url + files[c.file].rel + "'", buf + c.base, c.n);
            else ok = read_pipe(ssh + " 'read " + files[c.file].rel + " " + std::to_string(c.off) + " " + std::to_string(c.n) + "'", buf + c.base, c.n);
            if (!ok) { failures++; fprintf(stderr, "chunk %zu from source %d failed\n", i, kind); break; }
            source[i] = "dhk"[kind]; done_at[i] = now() - t0; bytes_by[kind] += c.n; count_by[kind]++; left--;
        }
        if (kind == 0 && --disk_workers_left == 0) t_disk_done = now() - t0;
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nd; i++) th.emplace_back(worker, 0);
    for (int i = 0; i < nh; i++) th.emplace_back(worker, 1);
    for (int i = 0; i < nk; i++) th.emplace_back(worker, 2);
    for (auto &t : th) t.join();
    const double t = now() - t0, td = t_disk_done.load();
    double net_by_disk_done[3] = { 0, 0, 0 }, disk_by_done = 0;
    for (size_t i = 0; i < chunks; i++) if (source[i] != 'x' && done_at[i] <= td) { if (source[i] == 'd') disk_by_done += list[i].n; else net_by_disk_done[source[i] == 'h' ? 1 : 2] += list[i].n; }
    printf("%zu file(s), %.2f GB in %.2f s = %.2f GB/s whole; workers %d disk, %d Hugging Face, %d k3; from the drive %.2f GB (%d chunks), Hugging Face %.2f GB (%d), k3 %.2f GB (%d); failures %d\n",
           files.size(), total / 1e9, t, total / 1e9 / t, nd, nh, nk, bytes_by[0] / 1e9, count_by[0].load(), bytes_by[1] / 1e9, count_by[1].load(), bytes_by[2] / 1e9, count_by[2].load(), failures.load());
    if (nh + nk) printf("   by the moment the drive finished (%.2f s): drive %.2f GB, Hugging Face %.2f GB, k3 %.2f GB: %.2f GB/s together, the drive %.2f GB/s, Hugging Face %.2f GB/s, k3 %.2f GB/s\n",
           td, disk_by_done / 1e9, net_by_disk_done[1] / 1e9, net_by_disk_done[2] / 1e9, (disk_by_done + net_by_disk_done[1] + net_by_disk_done[2]) / 1e9 / td, disk_by_done / 1e9 / td, net_by_disk_done[1] / 1e9 / td, net_by_disk_done[2] / 1e9 / td);
    size_t checked = 0, bad = 0; std::vector<unsigned char> ref(chunk);
    for (size_t i = 0; i < chunks && !failures; i++) if (source[i] == 'h' || source[i] == 'k') {
        const Chunk &c = list[i]; int lfd = open(files[c.file].local.c_str(), O_RDONLY);
        if (pread(lfd, ref.data(), c.n, (off_t)c.off) != (ssize_t)c.n || memcmp(ref.data(), buf + c.base, c.n)) bad++;
        close(lfd); checked += c.n;
    }
    if (checked) printf("   network chunks compared with the local files: %.2f GB, %zu differ\n", checked / 1e9, bad);
    return failures ? 1 : 0;
}