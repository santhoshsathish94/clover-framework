# Measure picture presets on this laptop as it is (Studio running): seconds, sd-cli peak memory, lowest free RAM.
# Runs are preset:seed:prompt-index. Pictures in logs\image-bench\<preset>; summary.txt there ends in BENCH_DONE.
param([string]$Runs = "sd15-lcm:42:0,sd15-lcm:7:1,sd15:42:0")
$ErrorActionPreference = 'Continue'
$b = Join-Path $env:USERPROFILE 'CloverAI'
$dir = Join-Path $b 'logs\image-bench'
New-Item -ItemType Directory -Force $dir | Out-Null
$sum = Join-Path $dir 'summary.txt'
$py = Join-Path $b 'venv\Scripts\python.exe'
function Free-MB { [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024) }
$total = [int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)
$studio = [bool](Get-Process llama-server -ErrorAction SilentlyContinue)
"start $(Get-Date -Format s) free $(Free-MB) of $total MB, Studio model loaded: $studio" | Add-Content $sum
$i = 0
foreach ($run in $Runs.Split(',')) {
    $i++
    $name, $seed, $pi = $run.Split(':')
    $res = Join-Path $dir "run-$(Get-Date -Format HHmmss)-$i.json"
    $free0 = Free-MB; $minFree = $free0; $peak = 0; $private = 0
    $p = Start-Process -FilePath $py -ArgumentList "tests\bench_image.py $name `"$res`" $seed 512 $pi" -WorkingDirectory $b -WindowStyle Hidden -PassThru
    $null = $p.Handle  # keeps the exit code readable after the process ends
    while (-not $p.HasExited) {
        Start-Sleep -Milliseconds 500
        $f = Free-MB; if ($f -lt $minFree) { $minFree = $f }
        $s = Get-Process sd-cli -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($s) {
            if ($s.PeakWorkingSet64 -gt $peak) { $peak = $s.PeakWorkingSet64 }
            if ($s.PrivateMemorySize64 -gt $private) { $private = $s.PrivateMemorySize64 }
        }
    }
    $r = if (Test-Path $res) { Get-Content $res -Raw | ConvertFrom-Json } else { $null }
    $what = if ($r -and $r.ok) { "ok $($r.seconds) s $($r.file)" } elseif ($r) { "FAILED $($r.error)" } else { "FAILED no result (exit $($p.ExitCode))" }
    "$name seed $seed prompt $pi : $what | sd-cli peak working set $([int]($peak / 1MB)) MB, private $([int]($private / 1MB)) MB | free RAM $free0 -> lowest $minFree MB" | Add-Content $sum
}
"BENCH_DONE $(Get-Date -Format s)" | Add-Content $sum
