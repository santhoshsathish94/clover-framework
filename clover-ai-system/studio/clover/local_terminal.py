"""Commands for Clover's tools: PowerShell on this laptop (and ssh.exe for the SSH tool), run as the user in a hidden
process. The whole process tree stops when the time limit passes or the user stops the turn; output is read as UTF-8
and capped, so a command that prints without end cannot fill memory."""
import os
import re
import subprocess
import threading
import time

MAX_TIMEOUT_S = 600
DEFAULT_TIMEOUT_S = 60
OUT_CAP_BYTES = 1_000_000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# PowerShell 5.1 writes in the console's OEM code page unless told otherwise; the last line turns a failure into an exit code
PS_HEAD = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8; " \
          "$ProgressPreference = 'SilentlyContinue'\n"
PS_TAIL = "\nif (-not $?) { exit $(if ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }) }"
# no window can answer an editor, pager or password prompt: `git rebase -i` opened Vim and hung (2026-11-01)
QUIET_GIT = {"GIT_EDITOR": "true", "GIT_SEQUENCE_EDITOR": "true", "GIT_PAGER": "cat", "PAGER": "cat",
             "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
# commands asked about even when the terminal is allowed: they delete, rewrite git history, publish, or stop the system
RISKY = (
    (re.compile(r"\bgit\s+(?:rebase|reset|clean|filter-branch|filter-repo|rm)\b|\bgit\s+push\b|"
                r"\bgit\s+(?:checkout|switch)\s+(?:-f\b|--force\b|--\s|\.\s*$|--discard-changes)|\bgit\s+restore\b|"
                r"\bgit\s+branch\s+(?:-D|--delete\s+--force)\b|\bgit\s+stash\s+(?:drop|clear)\b|"
                r"\bgit\s+commit\b[^\n;|]*--amend\b|\bgit\s+reflog\s+(?:expire|delete)\b", re.I),
     "it rewrites git history, discards changes or publishes to a remote"),
    (re.compile(r"(?:^|[;|&({\s])(?:Remove-Item|ri|rm|rmdir|rd|del|erase|Clear-Content|Clear-RecycleBin)(?=\s|$)", re.I),
     "it deletes files or folders"),
    (re.compile(r"\b(?:Format-Volume|Clear-Disk|Initialize-Disk|diskpart|format\s+[a-z]:|bcdedit|vssadmin|cipher\s+/w|"
                r"Stop-Computer|Restart-Computer|shutdown(?:\.exe)?\s+/[srp]|Set-ExecutionPolicy|reg(?:\.exe)?\s+delete|"
                r"Remove-ItemProperty|takeown|icacls\b[^\n;|]*/(?:grant|reset|remove))\b", re.I),
     "it changes the system itself"),
    (re.compile(r"\b(?:Stop-Process|taskkill|kill|Uninstall-Package|Uninstall-Module|Remove-AppxPackage)\b|"
                r"\bwinget\s+uninstall\b|\bchoco\s+uninstall\b", re.I),
     "it stops programs or uninstalls software"),
)


def risky(command):
    """Why a command cannot easily be undone (asked about even when commands are allowed), or None."""
    for pat, why in RISKY:
        if pat.search(str(command or "")):
            return why
    return None


def strip_ansi(text):
    return ANSI.sub("", str(text or ""))


def clamp_timeout(t):
    try:
        t = int(t)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    return max(1, min(MAX_TIMEOUT_S, t))


def run(args, cwd=None, timeout_s=DEFAULT_TIMEOUT_S, cancel=None, env=None):
    """-> (output text, exit code or None, how it ended: "exit" | "timeout" | "stopped")"""
    p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, creationflags=NO_WINDOW)
    chunks, size = [], [0]

    def reader():
        for b in iter(lambda: p.stdout.read(8192), b""):
            if size[0] < OUT_CAP_BYTES:
                chunks.append(b)
            size[0] += len(b)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t0, how = time.time(), "exit"
    while p.poll() is None:
        if cancel is not None and cancel.is_set():
            how = "stopped"
            break
        if time.time() - t0 > timeout_s:
            how = "timeout"
            break
        time.sleep(0.1)
    if how != "exit":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=NO_WINDOW)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
    t.join(3)
    out = b"".join(chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    if size[0] > OUT_CAP_BYTES:
        out += f"\n[output cut: {size[0]:,} bytes printed, the first {OUT_CAP_BYTES:,} kept]"
    return out, (p.returncode if how == "exit" else None), how


def ps51_chain(command):
    """`a && b` / `a || b` (PowerShell 7 and cmd) in the form Windows PowerShell 5.1 runs: `git add x && git commit`
    failed as a syntax error and the commit was never made (2026-11-01). Quoted text is left alone."""
    if "&&" not in command and "||" not in command:
        return command
    parts, cur, quote, i = [], "", None, 0
    while i < len(command):
        c = command[i]
        if quote:
            cur += c
            if c == quote:
                quote = None
        elif c in "'\"":
            quote, cur = c, cur + c
        elif command[i:i + 2] in ("&&", "||"):
            parts.append(cur.strip())
            parts.append(command[i:i + 2])
            cur, i = "", i + 2
            continue
        else:
            cur += c
        i += 1
    parts.append(cur.strip())
    out = parts[0]
    for op, nxt in zip(parts[1::2], parts[2::2]):
        out = f"{out}; if ({'$?' if op == '&&' else '-not $?'}) {{ {nxt} }}"
    return out


def powershell(command, cwd=None, timeout_s=DEFAULT_TIMEOUT_S, cancel=None):
    """A PowerShell command -> the text the model sees: output, then the exit code or why it stopped."""
    t = clamp_timeout(timeout_s)
    out, code, how = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", PS_HEAD + ps51_chain(command) + PS_TAIL],
                         cwd=cwd, timeout_s=t, cancel=cancel, env=dict(os.environ, **QUIET_GIT))
    return result_text(out, code, how, t)


def result_text(out, code, how, timeout_s):
    out = strip_ansi(out).strip()
    end = {"exit": f"[exit code {code}]", "timeout": f"[stopped after {timeout_s} seconds: the time limit]",
           "stopped": "[stopped: the user stopped the turn]"}[how]
    return (out + "\n" if out else "(no output)\n") + end


def resolve_cwd(raw, workdir=None):
    """The working folder the model asked for: ~ and %VARS% expanded, relative to the workspace; default the workspace
    or the home folder. -> (path, error)"""
    from pathlib import Path
    s = str(raw or "").strip()
    p = Path(os.path.expandvars(os.path.expanduser(s))) if s else Path(workdir or Path.home())
    if not p.is_absolute():
        p = Path(workdir or Path.home()) / p
    if not p.is_dir():
        return None, f"Error: the folder {p} does not exist."
    return p, None
