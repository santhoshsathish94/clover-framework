# Lays out the phone's fused C1 -> C2 -> C3 runtime (clover-framework s23_on_device_clover.py) on this laptop for a
# measurement, from the precomputed parts setup downloaded (data\omni\precomputed = the clover-ai-studio dataset's
# liquid-lfm3/): junctions and hard links only, nothing is copied or changed. Builds the C engine with this laptop's
# gcc (mingw, -O3 -march=native) under the file name the runtime loads. Usage: fused_probe_setup.ps1
$ErrorActionPreference = 'Stop'
$b = Join-Path $env:USERPROFILE 'CloverAI'          # the install: precomputed parts
$w = Join-Path $env:USERPROFILE 'CloverAI-build'    # the probe, gcc, the framework's code (push.ps1)
$p = Join-Path $b 'data\omni\precomputed'
# the probe holds junctions into $p: remove it with `cmd /c rmdir /s /q` (Windows PowerShell 5.1's Remove-Item follows them)
$f = Join-Path $w 'fused-probe'
$zip = Join-Path $w 'clover\fused\framework.zip'
if (-not (Test-Path (Join-Path $p 'trunks\global-0\embed_tokens.bf16.bin'))) { throw "precomputed parts missing in $p" }
if (Test-Path (Join-Path $f 'code')) { Remove-Item (Join-Path $f 'code') -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath (Join-Path $f 'code') -Force
$m = Join-Path $f 'code\clover-ai-system\mobile'
$env:PATH = (Join-Path $w 'tools\mingw64\bin') + ';' + $env:PATH
$t0 = Get-Date
# -static: the OpenMP and gcc runtimes go into the DLL (Python does not search mingw's bin folder for them)
& gcc -O3 -march=native -fopenmp -shared -static (Join-Path $m 'lfm3_cpu_engine.c') -lm -o (Join-Path $m 'liblfm3_cpu_engine.so') 2>&1 | Select-Object -Last 15
if ($LASTEXITCODE) { throw "gcc failed ($LASTEXITCODE)" }
"engine built in {0:N0} s: {1:N0} bytes" -f ((Get-Date) - $t0).TotalSeconds, (Get-Item (Join-Path $m 'liblfm3_cpu_engine.so')).Length

$base = Join-Path $f 'base'
if (Test-Path $base) { cmd /c rmdir /s /q "$base" | Out-Null }
$ncm = Join-Path $base 'ncm'
$pre = Join-Path $ncm 'precomputed'
New-Item -ItemType Directory -Force $pre | Out-Null
function New-Junction([string]$link, [string]$target) { cmd /c mklink /J "$link" "$target" | Out-Null; if (-not (Test-Path $link)) { throw "junction $link" } }
function New-HardLink([string]$link, [string]$target) { New-Item -ItemType HardLink -Path $link -Target $target | Out-Null }
Copy-Item (Join-Path $p 'trunks\tokenizer.json') (Join-Path $base 'tokenizer.json')
New-Junction (Join-Path $base 'pruned') (Join-Path $p 'trunks')
foreach ($n in 'lfm3_ncm_core_bf16.pt', 'vqvae_wte.bf16.bin', 'vqvae_dec.bf16.bin', 'config.yaml', 'ncm_runtime.py') { New-HardLink (Join-Path $ncm $n) (Join-Path $p "ncm\$n") }
New-Junction (Join-Path $ncm 'models') (Join-Path $p 'ncm\models')
New-Junction (Join-Path $ncm 'utils') (Join-Path $p 'ncm\utils')
Copy-Item (Join-Path $p 'ncm\manifest.json') (Join-Path $pre 'manifest.json')
New-Junction (Join-Path $pre 'plan-trunk-0') (Join-Path $p 'hybridagent\plan-trunk-0')
New-Junction (Join-Path $pre 'bridge-trunk-0') (Join-Path $p 'bridges\bridge-trunk-0')
Get-ChildItem (Join-Path $p 'ncm') -Directory | Where-Object { $_.Name -match '^(ncm|ptr|vqvae)-' } | ForEach-Object { New-Junction (Join-Path $pre $_.Name) $_.FullName }
"base laid out: " + (Get-ChildItem $pre -Directory).Count + " precomputed directories"
