# Resumes the ladder benchmark processes: calls NtResumeProcess on a fresh handle until no thread is suspended.
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class Rs {
  [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@
function Get-LadderPids { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*bench_layers.py*' } | ForEach-Object { [int]$_.ProcessId }) }
function Get-Suspended([int]$procId) { @((Get-Process -Id $procId -ErrorAction SilentlyContinue).Threads | Where-Object { $_.WaitReason -eq 'Suspended' }).Count }
function Resume-Ladder {
    foreach ($procId in Get-LadderPids) {
        $h = [Rs]::OpenProcess(0x0800, $false, $procId)
        if ($h -eq [IntPtr]::Zero) { "cannot open $procId"; continue }
        try {
            for ($i = 1; $i -le 64 -and (Get-Suspended $procId) -gt 0; $i++) { [void][Rs]::NtResumeProcess($h) }
            "pid $procId suspended threads now: $(Get-Suspended $procId)"
        } finally { [void][Rs]::CloseHandle($h) }
    }
}
if ($MyInvocation.InvocationName -ne '.') { Resume-Ladder }
