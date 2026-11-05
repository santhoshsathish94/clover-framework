# Restart the Studio server on the laptop: ask it to stop, wait until port 8765 is free (stop leftover Studio server
# processes after 20 s), start one server, wait until it is ready. Prints the pid that answers. -Base: another install.
param([string]$Base = (Join-Path $env:USERPROFILE 'CloverAI'))
$ErrorActionPreference = 'Continue'
$b = $Base
$url = 'http://127.0.0.1:8765'
function Listening { [bool](Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) }
try { Invoke-RestMethod -Method Post -TimeoutSec 5 -ContentType 'application/json' -Headers @{ Origin = $url } -Body '{}' "$url/api/shutdown" | Out-Null } catch { }
$t0 = Get-Date
while ((Listening) -and ((Get-Date) - $t0).TotalSeconds -lt 20) { Start-Sleep 1 }
if (Listening) {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*clover\server.py*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped leftover Studio server pid $($_.ProcessId)" }
    Start-Sleep 2
}
$line = '"' + (Join-Path $b 'venv\Scripts\pythonw.exe') + '" "' + (Join-Path $b 'clover\server.py') + '" --base "' + $b + '" --idle-exit-min 30'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $b }
$st = $null
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep 1
    try { $st = Invoke-RestMethod -TimeoutSec 3 "$url/api/status?passive=1"; if ($st.status -eq 'ready') { break } } catch { }
}
$owner = (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
"studio $($st.status) model=$($st.model) engine=$($st.engine) listening pid=$owner (launcher pid $($r.ProcessId))"
