# Restart the Studio and run the natural-conversation regression detached (it outlives the ssh session).
# Output: CloverAI-build\verify\natural\<tag>-<stamp>.jsonl and .log. Extra harness arguments go in -HarnessArgs.
param([string]$HarnessArgs = '', [string]$Tag = 'run', [switch]$NoRestart, [string]$Base = (Join-Path $env:USERPROFILE 'CloverAI'))
$ErrorActionPreference = 'Stop'
$b = $Base
if (-not $NoRestart) { & (Join-Path $env:USERPROFILE 'CloverAI-build\tests\restart_studio.ps1') -Base $b }
$dir = Join-Path $env:USERPROFILE 'CloverAI-build\verify\natural'  # records stay out of the install
New-Item -ItemType Directory -Force $dir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $dir "$Tag-$stamp.jsonl"
$log = Join-Path $dir "$Tag-$stamp.log"
$py = Join-Path $b 'venv\Scripts\python.exe'
$harness = Join-Path $env:USERPROFILE 'CloverAI-build\tests\natural_regression.py'  # tests live in the build folder
$line = "cmd.exe /c `"`"$py`" -u `"$harness`" --out `"$out`" $HarnessArgs > `"$log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $b }
"started pid $($r.ProcessId) log $log"
