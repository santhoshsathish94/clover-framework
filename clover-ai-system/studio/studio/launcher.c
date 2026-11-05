/* CloverAIStudio.exe - starts the local Clover engine (if it is not running) and opens Clover AI Studio.
 *
 *   CloverAIStudio.exe                 engine + Studio window. Closing the window stops the engine it started,
 *                                      unless VS Code used Clover in the last 15 minutes.
 *   CloverAIStudio.exe --engine-only   start the engine (for VS Code) and exit; it stops after 30 idle minutes.
 *   CloverAIStudio.exe --stop          stop the engine.
 *
 * The engine is CloverAI\venv\Scripts\pythonw.exe CloverAI\clover\server.py, listening on 127.0.0.1:8765 only.
 * The window is Microsoft Edge in app mode with its own profile folder (CloverAI\studio-profile), so the user's
 * normal browser profile is not touched. If Edge is missing, the default browser opens the same page.
 *
 * Build (mingw-w64): windres studio.rc -O coff -o studio_res.o
 *                    gcc -O2 -municode -mwindows launcher.c studio_res.o -lws2_32 -ladvapi32 -lshell32 -o CloverAIStudio.exe
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#define PORT 8765
#define IDLE_MIN 30
#define VSCODE_GRACE_S 900
#define TITLE L"Clover AI Studio"

static wchar_t g_base[MAX_PATH];

static void fail(const wchar_t *msg)
{
    MessageBoxW(NULL, msg, TITLE, MB_ICONERROR | MB_OK);
}

static int exists(const wchar_t *p)
{
    return GetFileAttributesW(p) != INVALID_FILE_ATTRIBUTES;
}

/* Minimal HTTP/1.0 client for 127.0.0.1:PORT. Returns the status code, or -1 if the engine is not reachable. */
static int http(const char *method, const char *path, const char *body, char *out, int cap, int timeout_ms)
{
    static int inited = 0;
    if (!inited) {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return -1;
        inited = 1;
    }
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return -1;
    DWORD to = (DWORD)timeout_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof(to));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&to, sizeof(to));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons(PORT);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, (struct sockaddr *)&a, sizeof(a)) != 0) {
        closesocket(s);
        return -1;
    }
    char hdr[128] = "";
    if (body) snprintf(hdr, sizeof(hdr), "Content-Type: application/json\r\nContent-Length: %d\r\n", (int)strlen(body));
    char req[1024];
    int n = snprintf(req, sizeof(req), "%s %s HTTP/1.0\r\nHost: 127.0.0.1:%d\r\nUser-Agent: CloverAIStudio\r\n%s\r\n%s",
                     method, path, PORT, hdr, body ? body : "");
    if (n <= 0 || n >= (int)sizeof(req) || send(s, req, n, 0) != n) {
        closesocket(s);
        return -1;
    }
    int got = 0, r;
    while (got < cap - 1 && (r = recv(s, out + got, cap - 1 - got, 0)) > 0) got += r;
    out[got > 0 ? got : 0] = 0;
    closesocket(s);
    int code = 0;
    return sscanf(out, "HTTP/%*s %d", &code) == 1 ? code : -1;
}

static int engine_up(void)
{
    char buf[2048];
    return http("GET", "/api/status?passive=1", NULL, buf, sizeof(buf), 2000) == 200;
}

static HANDLE start_engine(void)
{
    wchar_t py[MAX_PATH], script[MAX_PATH], cmd[4 * MAX_PATH];
    swprintf(py, MAX_PATH, L"%ls\\venv\\Scripts\\pythonw.exe", g_base);
    swprintf(script, MAX_PATH, L"%ls\\clover\\server.py", g_base);
    if (!exists(py) || !exists(script)) {
        wchar_t msg[1200];
        swprintf(msg, 1200, L"The Clover runtime was not found in\n%ls\n\nExpected venv\\Scripts\\pythonw.exe and clover\\server.py.", g_base);
        fail(msg);
        return NULL;
    }
    swprintf(cmd, 4 * MAX_PATH, L"\"%ls\" \"%ls\" --base \"%ls\" --idle-exit-min %d", py, script, g_base, IDLE_MIN);
    SetEnvironmentVariableW(L"HF_HUB_OFFLINE", L"1");
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    DWORD flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP;
    if (!CreateProcessW(py, cmd, NULL, NULL, FALSE, flags | CREATE_BREAKAWAY_FROM_JOB, NULL, g_base, &si, &pi) &&
        !CreateProcessW(py, cmd, NULL, NULL, FALSE, flags, NULL, g_base, &si, &pi)) {
        wchar_t msg[600];
        swprintf(msg, 600, L"Could not start the Clover engine (error %lu).", GetLastError());
        fail(msg);
        return NULL;
    }
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

/* Waits until the engine answers HTTP (the model keeps loading in the background; the Studio shows progress). */
static int wait_engine(HANDLE proc)
{
    for (int i = 0; i < 240; i++) {
        if (engine_up()) return 1;
        if (proc && WaitForSingleObject(proc, 0) == WAIT_OBJECT_0) {
            DWORD code = 0;
            GetExitCodeProcess(proc, &code);
            wchar_t msg[700];
            if (code == 3)
                swprintf(msg, 700, L"Port %d on 127.0.0.1 is used by another program, so the Clover engine cannot start.", PORT);
            else
                swprintf(msg, 700, L"The Clover engine stopped while starting (exit code %lu). See logs\\studio-server.log.",
                         (unsigned long)code);
            fail(msg);
            return 0;
        }
        Sleep(250);
    }
    fail(L"The Clover engine did not start within 60 seconds. See logs\\studio-server.log.");
    return 0;
}

static int find_edge(wchar_t *out)
{
    const wchar_t *vars[] = { L"ProgramFiles(x86)", L"ProgramFiles", L"LOCALAPPDATA" };
    wchar_t dir[MAX_PATH];
    for (int i = 0; i < 3; i++) {
        if (GetEnvironmentVariableW(vars[i], dir, MAX_PATH)) {
            swprintf(out, MAX_PATH, L"%ls\\Microsoft\\Edge\\Application\\msedge.exe", dir);
            if (exists(out)) return 1;
        }
    }
    DWORD sz = MAX_PATH * sizeof(wchar_t);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\msedge.exe", NULL,
                     RRF_RT_REG_SZ, NULL, out, &sz) == ERROR_SUCCESS && exists(out))
        return 1;
    return 0;
}

int WINAPI wWinMain(HINSTANCE hi, HINSTANCE hp, PWSTR args, int show)
{
    (void)hi; (void)hp; (void)show;
    SetProcessDPIAware();
    GetModuleFileNameW(NULL, g_base, MAX_PATH);
    wchar_t *slash = wcsrchr(g_base, L'\\');
    if (slash) *slash = 0;

    int engine_only = args && wcsstr(args, L"--engine-only") != NULL;
    int stop = args && wcsstr(args, L"--stop") != NULL;
    char buf[4096];

    if (stop) {
        int code = http("POST", "/api/shutdown", "{}", buf, sizeof(buf), 3000);
        MessageBoxW(NULL, code == 200 ? L"The Clover engine was stopped." : L"The Clover engine was not running.", TITLE,
                    MB_ICONINFORMATION | MB_OK);
        return 0;
    }

    int started_here = 0;
    if (!engine_up()) {
        HANDLE proc = start_engine();
        if (!proc) return 1;
        int ok = wait_engine(proc);
        CloseHandle(proc);
        if (!ok) return 1;
        started_here = 1;
    }
    if (engine_only) return 0;

    wchar_t url[64];
    swprintf(url, 64, L"http://127.0.0.1:%d/", PORT);
    wchar_t edge[MAX_PATH];
    if (!find_edge(edge)) {
        ShellExecuteW(NULL, L"open", url, NULL, NULL, SW_SHOWNORMAL);
        return 0; /* no window to wait for: the engine stops after IDLE_MIN idle minutes */
    }
    wchar_t cmd[4 * MAX_PATH];
    swprintf(cmd, 4 * MAX_PATH,
             L"\"%ls\" --app=%ls --user-data-dir=\"%ls\\studio-profile\" --no-first-run --no-default-browser-check "
             L"--disable-sync --disable-background-networking --disable-component-update --window-size=1240,880",
             edge, url, g_base);
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    if (!CreateProcessW(edge, cmd, NULL, NULL, FALSE, 0, NULL, g_base, &si, &pi)) {
        ShellExecuteW(NULL, L"open", url, NULL, NULL, SW_SHOWNORMAL);
        return 0;
    }
    CloseHandle(pi.hThread);
    DWORD t0 = GetTickCount();
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    /* A quick exit means Edge handed the window to an already open Studio: that window owns the session. */
    if (started_here && GetTickCount() - t0 > 5000) {
        char body[96];
        snprintf(body, sizeof(body), "{\"unless_vscode_within_s\": %d}", VSCODE_GRACE_S);
        http("POST", "/api/shutdown", body, buf, sizeof(buf), 3000);
    }
    return 0;
}
