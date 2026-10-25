#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
enum { MAX_CARDS = 8, N_LAYERS = 92 };
struct Loader {
    std::mutex mu; std::condition_variable cv; std::deque<std::function<void()>> q; std::thread th; bool started = false;
};
static Loader *g_loader[MAX_CARDS];   // never destroyed: a loader thread waits for work until the process ends
static std::mutex &pending_mu() { static std::mutex *m = new std::mutex; return *m; }
static std::condition_variable &pending_cv() { static std::condition_variable *c = new std::condition_variable; return *c; }
static int g_pending = 0;
static void loader_submit(int card, std::function<void()> job)
{
    if (!g_loader[card]) g_loader[card] = new Loader;
    Loader &l = *g_loader[card];
    { std::lock_guard<std::mutex> g(pending_mu()); g_pending++; }
    std::lock_guard<std::mutex> g(l.mu);
    if (!l.started) {
        l.started = true;
        l.th = std::thread([card, &l]() {
            CHECK(cudaSetDevice(card));
            for (;;) {
                std::function<void()> j;
                { std::unique_lock<std::mutex> u(l.mu); l.cv.wait(u, [&]() { return !l.q.empty(); }); j = std::move(l.q.front()); l.q.pop_front(); }
                j();
                { std::lock_guard<std::mutex> g2(pending_mu()); g_pending--; }
                pending_cv().notify_all();
            }
        });
        l.th.detach();
    }
    l.q.push_back(std::move(job));
    l.cv.notify_one();
}
static void loader_wait()
{
    std::unique_lock<std::mutex> u(pending_mu());
    pending_cv().wait(u, []() { return g_pending == 0; });
}
