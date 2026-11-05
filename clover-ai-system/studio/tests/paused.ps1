# Runs one Python test from CloverAI with the ladder benchmark suspended, for clean timings. The resume is checked:
# every ladder thread must be running again before this script ends (a stale handle once left the ladder frozen).
param([Parameter(Mandatory = $true)][string]$Script)
$ErrorActionPreference = 'Stop'
$b = Join-Path $env:USERPROFILE 'CloverAI'
. (Join-Path $PSScriptRoot 'resume_ladder.ps1')
$held = @()
try {
    foreach ($procId in Get-LadderPids) {
        $h = [Rs]::OpenProcess(0x0800, $false, $procId)
        if ($h -ne [IntPtr]::Zero) { [void][Rs]::NtSuspendProcess($h); $held += $h }
    }
    "paused ladder: " + ((Get-LadderPids) -join ', ')
    $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
    $ErrorActionPreference = 'Continue'
    & (Join-Path $b 'venv\Scripts\python.exe') (Join-Path $b $Script) 2>&1 | Where-Object { $_ -notmatch 'transformers\]|^\s*$' } | Out-String -Width 260
} finally {
    foreach ($h in $held) { [void][Rs]::CloseHandle($h) }
    Resume-Ladder
}
