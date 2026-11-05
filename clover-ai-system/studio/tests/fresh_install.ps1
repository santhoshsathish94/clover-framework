# Fresh-install test: runs the real CloverAIStudio-Setup.exe (IExpress) quietly into a separate folder, so nothing of
# the main install is used: every file comes from its publisher. Detached (outlives the ssh session).
param([string]$Target = (Join-Path $env:USERPROFILE 'CloverAI-fresh'), [string]$Components = 'core,csi,pictures',
      [string]$Reuse = '')
$ErrorActionPreference = 'Stop'
$setup = Join-Path $env:USERPROFILE 'CloverAI-build\setup\CloverAIStudio-Setup.exe'
$x = Join-Path $env:TEMP 'clover-setup-x'
if (Test-Path $x) { Remove-Item $x -Recurse -Force }
Remove-Item (Join-Path $env:TEMP 'CloverAIStudio-setup.log') -ErrorAction SilentlyContinue
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File install.ps1 -Quiet -Base $Target -NoShortcuts -Components $Components"
if ($Reuse) { $cmd += " -Reuse $Reuse" }
$line = "`"$setup`" /Q /T:$x /C:`"$cmd`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $env:TEMP }
"started pid $($r.ProcessId) rc $($r.ReturnValue): $line"
