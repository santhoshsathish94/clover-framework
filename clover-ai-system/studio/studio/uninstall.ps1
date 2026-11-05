param([switch]$Quiet)
# Removes Clover AI Studio (shortcuts, VS Code extension, uninstall entry, the Studio exe and its browser profile).
# The Clover runtime in the same folder (Python venv, models, engine, chats in logs) is kept.
$ErrorActionPreference = 'Continue'
$base = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
try { Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/api/shutdown' -ContentType 'application/json' -Body '{}' -TimeoutSec 5 | Out-Null; Start-Sleep 2 } catch { }
$programs = [Environment]::GetFolderPath('Programs')
$desktop = [Environment]::GetFolderPath('Desktop')
foreach ($l in (Join-Path $programs 'Clover AI Studio.lnk'), (Join-Path $programs 'Clover Engine for VS Code.lnk'), (Join-Path $desktop 'Clover AI Studio.lnk')) {
    if (Test-Path $l) { Remove-Item $l -Force }
}
$code = @("$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd", "$env:ProgramFiles\Microsoft VS Code\bin\code.cmd") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if ($code) { & $code --uninstall-extension clover-framework.clover-local 2>&1 | Out-Null }
Remove-Item 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio' -Recurse -Force -ErrorAction SilentlyContinue
foreach ($p in (Join-Path $base 'CloverAIStudio.exe'), (Join-Path $base 'studio-profile')) {
    if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue }
}
$msg = "Clover AI Studio was removed.`n`nThe folder $base was kept: it holds Python, the models and tools, and your own data (chats, memory, and what Clover learned about you). Delete that folder to remove everything."
if ($Quiet) { Write-Output $msg } else { Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show($msg, 'Clover AI Studio') }
