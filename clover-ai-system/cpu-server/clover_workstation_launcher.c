#define _GNU_SOURCE
/*
 * Clover AI Studio — Native Linux Workstation Launcher (`clover-ai-studio`)
 * (`clover-ai-system/cpu-server/clover_workstation_launcher.c`)
 *
 * The Linux Workstation counterpart to `CloverAIStudio.exe` (Windows Laptop) and
 * `CloverAIStudio.apk` (Android Mobile).
 *
 * Commands:
 *   clover-ai-studio                 Starts the 7-layer Clover Workstation engine (if not running)
 *                                    with NVIDIA Nemotron 3.5 Super VL (120B-A12B, 1M context) as default,
 *                                    and opens Clover AI Studio in a dedicated app window (when a display
 *                                    is attached) or prints the live workstation status.
 *   clover-ai-studio --engine-only   Starts the background engine for VS Code / Web Studio and exits.
 *   clover-ai-studio --status        Queries http://127.0.0.1:8765/api/health and prints live status.
 *   clover-ai-studio --verify        Runs the 7-layer Nemotron workstation verification suite.
 *   clover-ai-studio --stop          Stops the background workstation engine.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -s clover_workstation_launcher.c -o clover-ai-studio
 */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CLOVER_PORT 8765
#define CLOVER_VERSION "1.0.0"
#define PID_FILE "/tmp/clover-ai-workstation.pid"
#define LOG_FILE "/var/log/clover-ai-workstation.log"

static int file_exists(const char *path)
{
    return path && access(path, F_OK) == 0;
}

static int file_executable(const char *path)
{
    return path && access(path, X_OK) == 0;
}

/* Minimal HTTP/1.0 client for 127.0.0.1:8765. Returns HTTP status code (e.g. 200) or -1 on error. */
static int http_request(
    const char *method,
    const char *path,
    const char *body,
    char *out_buf,
    size_t out_cap,
    int timeout_ms
)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(CLOVER_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(sock);
        return -1;
    }

    char hdr[256] = "";
    if (body && body[0]) {
        snprintf(
            hdr,
            sizeof(hdr),
            "Content-Type: application/json\r\nContent-Length: %zu\r\n",
            strlen(body)
        );
    }

    char req[2048];
    int n = snprintf(
        req,
        sizeof(req),
        "%s %s HTTP/1.0\r\nHost: 127.0.0.1:%d\r\nUser-Agent: CloverAIStudio-Workstation/%s\r\n%s\r\n%s",
        method,
        path,
        CLOVER_PORT,
        CLOVER_VERSION,
        hdr,
        body ? body : ""
    );
    if (n <= 0 || (size_t)n >= sizeof(req) || send(sock, req, (size_t)n, 0) != n) {
        close(sock);
        return -1;
    }

    size_t total = 0;
    ssize_t r;
    while (total + 1 < out_cap && (r = recv(sock, out_buf + total, out_cap - 1 - total, 0)) > 0) {
        total += (size_t)r;
    }
    out_buf[total] = '\0';
    close(sock);

    int status_code = -1;
    if (sscanf(out_buf, "HTTP/%*s %d", &status_code) == 1) {
        return status_code;
    }
    return -1;
}

static int is_engine_ready(char *health_json, size_t cap)
{
    char raw[8192];
    int code = http_request("GET", "/api/health", NULL, raw, sizeof(raw), 2500);
    if (code != 200) return 0;
    if (health_json && cap > 0) {
        const char *body = strstr(raw, "\r\n\r\n");
        if (body) body += 4;
        else body = raw;
        snprintf(health_json, cap, "%s", body);
    }
    return 1;
}

static const char *resolve_python_bin(void)
{
    static const char *candidates[] = {
        "/root/ncm-cpu/venv/bin/python",
        "/opt/clover-ai-workstation/venv/bin/python",
        "/opt/clover-k3/venv/bin/python",
        "/root/hybridagent-venv/bin/python",
        "/usr/bin/python3",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (file_executable(candidates[i])) return candidates[i];
    }
    return "python3";
}

static const char *resolve_server_script(void)
{
    static const char *candidates[] = {
        "/opt/clover-ai-workstation/src/tools/hybrid-ncm-k3/clover-chat-server.py",
        "/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py",
        "/opt/clover-k3/clover-framework/clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_chat_server.py",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (file_exists(candidates[i])) return candidates[i];
    }
    return NULL;
}

static const char *resolve_verifier_script(void)
{
    static const char *candidates[] = {
        "/opt/clover-ai-workstation/cpu-server/nemotron_workstation_clover.py",
        "/opt/clover-k3/clover-framework/clover-ai-system/cpu-server/nemotron_workstation_clover.py",
        "/opt/clover-k3/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_workstation_clover.py",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        if (file_exists(candidates[i])) return candidates[i];
    }
    return NULL;
}

static void print_workstation_banner(void)
{
    struct utsname un;
    struct sysinfo si;
    memset(&un, 0, sizeof(un));
    uname(&un);
    double ram_gib = 0.0;
    if (sysinfo(&si) == 0) {
        ram_gib = (double)si.totalram * (double)si.mem_unit / (1024.0 * 1024.0 * 1024.0);
    }
    long n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    printf("========================================================================\n");
    printf("  Clover AI Studio — Linux Workstation Edition v%s\n", CLOVER_VERSION);
    printf("  Host: %s (%s %s) | CPUs: %ld threads | RAM: %.1f GiB\n",
           un.nodename, un.sysname, un.machine, n_cpus > 0 ? n_cpus : 1L, ram_gib);
    printf("  Default Model: NVIDIA Nemotron 3.5 Super VL (120B-A12B, 1M Context)\n");
    printf("  7-Layer Stack: C/OpenMP + 8-GPU HybridAgent + 11-Level NCM + Memory\n");
    printf("                 + Ontology GraphRAG + Generative Compiler + SpecMD R4\n");
    printf("========================================================================\n");
}

static int start_background_engine(void)
{
    char health[4096];
    if (is_engine_ready(health, sizeof(health))) {
        printf("[Clover-Workstation] Engine is already running and healthy on http://127.0.0.1:%d\n", CLOVER_PORT);
        return 0;
    }

    const char *py = resolve_python_bin();
    const char *srv = resolve_server_script();
    if (!srv) {
        fprintf(stderr, "[Clover-Workstation] Error: Could not locate clover-chat-server.py\n");
        return 1;
    }

    printf("[Clover-Workstation] Starting 7-Layer Clover Workstation Server using %s ...\n", py);
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        setsid();
        setenv("CLOVER_NEMO_USE_4BIT_EXPERTS", "1", 1);
        setenv("PYTHONIOENCODING", "utf-8", 1);
        int fd = open(LOG_FILE, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd < 0) fd = open("/tmp/clover-ai-workstation.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            if (fd > 2) close(fd);
        }
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDIN_FILENO);
            if (null_fd > 2) close(null_fd);
        }
        execlp(py, py, "-u", srv, "--host", "127.0.0.1", "--port", "8765", (char *)NULL);
        _exit(127);
    }

    FILE *pf = fopen(PID_FILE, "w");
    if (pf) {
        fprintf(pf, "%d\n", (int)pid);
        fclose(pf);
    }

    printf("[Clover-Workstation] Waiting for engine initialization (PID %d)...\n", (int)pid);
    for (int i = 0; i < 90; i++) {
        usleep(1000 * 1000);
        if (is_engine_ready(health, sizeof(health))) {
            printf("[Clover-Workstation] READY on http://127.0.0.1:%d\n", CLOVER_PORT);
            return 0;
        }
    }
    fprintf(stderr, "[Clover-Workstation] Engine still loading; check logs at /tmp/clover-ai-workstation.log\n");
    return 0;
}

static int stop_background_engine(void)
{
    FILE *pf = fopen(PID_FILE, "r");
    if (pf) {
        int pid = 0;
        if (fscanf(pf, "%d", &pid) == 1 && pid > 1) {
            printf("[Clover-Workstation] Stopping engine PID %d...\n", pid);
            kill((pid_t)pid, SIGTERM);
            usleep(500 * 1000);
        }
        fclose(pf);
        unlink(PID_FILE);
    }
    int rc = system("systemctl is-active --quiet clover-ai-workstation 2>/dev/null && systemctl stop clover-ai-workstation 2>/dev/null");
    (void)rc;
    printf("[Clover-Workstation] Stopped.\n");
    return 0;
}

static void launch_desktop_browser_if_display_present(void)
{
    const char *disp = getenv("DISPLAY");
    const char *way = getenv("WAYLAND_DISPLAY");
    if ((!disp || !disp[0]) && (!way || !way[0])) {
        printf("[Clover-Workstation] Headless workstation detected (no DISPLAY/WAYLAND_DISPLAY).\n");
        printf("[Clover-Workstation] Studio Web UI & OpenAI-compatible API active at: http://127.0.0.1:%d\n", CLOVER_PORT);
        return;
    }

    static const char *browsers[] = {
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/brave-browser",
        "/usr/bin/microsoft-edge",
        NULL
    };
    for (int i = 0; browsers[i]; i++) {
        if (file_executable(browsers[i])) {
            pid_t bpid = fork();
            if (bpid == 0) {
                execl(
                    browsers[i],
                    browsers[i],
                    "--app=http://127.0.0.1:8765",
                    "--user-data-dir=/tmp/clover-studio-profile",
                    (char *)NULL
                );
                _exit(0);
            }
            printf("[Clover-Workstation] Opened Clover AI Studio window via %s\n", browsers[i]);
            return;
        }
    }
    if (file_executable("/usr/bin/xdg-open")) {
        pid_t xpid = fork();
        if (xpid == 0) {
            execl("/usr/bin/xdg-open", "xdg-open", "http://127.0.0.1:8765", (char *)NULL);
            _exit(0);
        }
        printf("[Clover-Workstation] Opened http://127.0.0.1:%d via xdg-open\n", CLOVER_PORT);
    }
}

int main(int argc, char **argv)
{
    const char *arg = (argc > 1) ? argv[1] : "";

    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
        print_workstation_banner();
        printf("Usage:\n");
        printf("  clover-ai-studio                 Start 7-layer workstation engine & open Studio\n");
        printf("  clover-ai-studio --engine-only   Start background engine only (for VS Code / Web)\n");
        printf("  clover-ai-studio --status        Show live health, active model, and SpecMD R4 ledger\n");
        printf("  clover-ai-studio --verify        Run full 7-layer Nemotron workstation verification\n");
        printf("  clover-ai-studio --stop          Stop background engine\n");
        return 0;
    }

    if (strcmp(arg, "--version") == 0 || strcmp(arg, "-v") == 0) {
        printf("clover-ai-studio workstation v%s\n", CLOVER_VERSION);
        return 0;
    }

    if (strcmp(arg, "--stop") == 0) {
        return stop_background_engine();
    }

    if (strcmp(arg, "--status") == 0) {
        print_workstation_banner();
        char health[8192];
        if (is_engine_ready(health, sizeof(health))) {
            printf("[Status] ONLINE at http://127.0.0.1:%d\n", CLOVER_PORT);
            printf("[Health] %.600s\n", health);
            char caps[8192];
            if (http_request("GET", "/api/models/capabilities", NULL, caps, sizeof(caps), 2500) == 200) {
                const char *b = strstr(caps, "\r\n\r\n");
                printf("[Models] %.420s...\n", b ? (b + 4) : caps);
            }
            return 0;
        } else {
            printf("[Status] OFFLINE (run `clover-ai-studio --start` to launch)\n");
            return 1;
        }
    }

    if (strcmp(arg, "--verify") == 0) {
        print_workstation_banner();
        const char *py = resolve_python_bin();
        const char *ver = resolve_verifier_script();
        if (!ver) {
            fprintf(stderr, "Error: nemotron_workstation_clover.py not found.\n");
            return 1;
        }
        printf("[Clover-Workstation] Running 7-Layer Nemotron Workstation Verification (%s)...\n", ver);
        execlp(py, py, ver, (char *)NULL);
        perror("execlp");
        return 1;
    }

    print_workstation_banner();
    int rc = start_background_engine();
    if (rc != 0) return rc;

    if (strcmp(arg, "--engine-only") == 0 || strcmp(arg, "--start") == 0) {
        return 0;
    }

    launch_desktop_browser_if_display_present();
    return 0;
}
