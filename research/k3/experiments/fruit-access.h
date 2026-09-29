static int fruit_guard_applied, fruit_guard_probe_passed;
static size_t fruit_guard_bytes, fruit_boundary_bytes;

static int fruit_enabled(void)
{
    return getenv("K3_FRUIT_HEAD") != NULL;
}

static void fruit_guard_mapping(int file_id)
{
    if (!fruit_enabled() || file_id != mrec[4].file_id || fruit_guard_applied) return;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || mrec[4].off < 0 || mrec[4].nbytes <= 0 ||
        (uint64_t)mrec[4].off + (uint64_t)mrec[4].nbytes > fsize[file_id]) die("fruit guard range invalid");
    const size_t start = ((size_t)mrec[4].off + (size_t)page - 1) / (size_t)page * (size_t)page;
    const size_t end = ((size_t)mrec[4].off + (size_t)mrec[4].nbytes) / (size_t)page * (size_t)page;
    if (end <= start || mprotect(fmap[file_id] + start, end - start, PROT_NONE)) die("fruit original head guard failed");
    fruit_guard_bytes = end - start;
    fruit_boundary_bytes = (size_t)mrec[4].nbytes - fruit_guard_bytes;
    fruit_guard_applied = 1;
    if (getenv("K3_FRUIT_GUARD_PROBE")) {
        const pid_t child = fork();
        if (child < 0) die("fruit guard fork failed");
        if (!child) {
            const struct rlimit limit = {0, 0};
            if (setrlimit(RLIMIT_CORE, &limit)) _exit(3);
            const volatile unsigned char *protected_bytes = fmap[file_id] + start;
            const unsigned char observed = *protected_bytes;
            _exit(observed ? 4 : 5);
        }
        int status;
        if (waitpid(child, &status, 0) != child || !WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV)
            die("fruit guard did not reject original head read");
        fruit_guard_probe_passed = 1;
    }
}