# LFM3 speed on the Iris Xe GPU (llama.cpp Vulkan) vs the CPU (llama.cpp CPU build, Clover's C engine), pp512 / tg128.
# The ladder benchmark is suspended while this runs (resumed in `finally`) so the numbers are not shared with it, and
# every measured process has Windows power throttling (EcoQoS) turned off, as the Clover processes do.
param([int]$Reps = 2)
$ErrorActionPreference = 'Stop'
$b = Join-Path $env:USERPROFILE 'CloverAI'
$out = Join-Path $b 'logs\gpu-vs-cpu.log'
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class ProcCtl {
  [StructLayout(LayoutKind.Sequential)] public struct PPT { public uint Version; public uint ControlMask; public uint StateMask; }
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessInformation(IntPtr h, int cls, ref PPT info, int size);
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  public static bool NoThrottle(IntPtr h) { var p = new PPT { Version = 1, ControlMask = 0x5, StateMask = 0 }; return SetProcessInformation(h, 4, ref p, Marshal.SizeOf(p)); }
}
"@
function Log([string]$m) { Add-Content -Path $out -Value $m; Write-Output $m }
function Run([string]$exe, [string]$arguments, [string]$label) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo $exe, $arguments
    $psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.CreateNoWindow = $true
    $psi.WorkingDirectory = Split-Path $exe
    $p = [System.Diagnostics.Process]::Start($psi)
    [void][ProcCtl]::NoThrottle($p.Handle)
    $errTask = $p.StandardError.ReadToEndAsync()
    $stdout = $p.StandardOutput.ReadToEnd()
    $p.WaitForExit()
    $stderr = $errTask.Result
    Log "== $label (exit $($p.ExitCode))"
    ($stderr -split "`n" | Where-Object { $_ -match 'Vulkan|Iris|ggml_vulkan|device|error|fail' } | Select-Object -First 8) | ForEach-Object { Log ("   " + $_.Trim()) }
    ($stdout -split "`n" | Where-Object { $_ -match '\||CLOVER_C_ENGINE' }) | ForEach-Object { Log $_.TrimEnd() }
}

Set-Content -Path $out -Value ("gpu-vs-cpu " + (Get-Date -Format s) + " llama.cpp " + (Get-Content (Join-Path $b 'tools\llama.cpp\RELEASE.txt') -ErrorAction SilentlyContinue))
$getLog = Join-Path $b 'logs\llamacpp-get.log'
for ($i = 0; $i -lt 90 -and -not (Select-String -Path $getLog -Pattern 'LLAMACPP_GET_(DONE|FAILED)' -Quiet); $i++) { Start-Sleep -Seconds 10 }
$verified = @(Select-String -Path $getLog -Pattern '^\S+ ok (\S+\.gguf) ' | ForEach-Object { $_.Matches[0].Groups[1].Value })
$bat = Get-CimInstance Win32_Battery
Log ("power: battery " + $bat.EstimatedChargeRemaining + "% status " + $bat.BatteryStatus + " (2 = on AC)")
. (Join-Path $PSScriptRoot 'resume_ladder.ps1')
$handles = @()
try {
    foreach ($procId in Get-LadderPids) { $h = [Rs]::OpenProcess(0x0800, $false, $procId); if ($h -ne [IntPtr]::Zero) { [void][Rs]::NtSuspendProcess($h); $handles += $h } }
    Log ("suspended ladder processes: " + ((Get-LadderPids) -join ', '))
    Start-Sleep -Seconds 3
    $models = Get-ChildItem (Join-Path $b 'data\lfm3\gguf') -Filter '*.gguf' | Where-Object { $verified -contains $_.FullName } | Sort-Object Length
    foreach ($m in $models) {
        Run (Join-Path $b 'tools\llama.cpp\vulkan\llama-bench.exe') "-m `"$($m.FullName)`" -ngl 99 -p 512 -n 128 -r $Reps" ("Iris Xe (Vulkan) " + $m.Name)
        foreach ($t in 4, 10) {
            Run (Join-Path $b 'tools\llama.cpp\cpu\llama-bench.exe') "-m `"$($m.FullName)`" -t $t -p 512 -n 128 -r $Reps" ("CPU llama.cpp t=$t " + $m.Name)
        }
    }
    Run (Join-Path $b 'venv\Scripts\python.exe') "`"$(Join-Path $b 'tests\engine_speed.py')`"" 'Clover C engine bf16 (current Studio engine)'
} finally {
    foreach ($h in $handles) { [void][Rs]::CloseHandle($h) }
    Resume-Ladder | ForEach-Object { Log $_ }
}
